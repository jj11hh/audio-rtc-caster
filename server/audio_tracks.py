import asyncio
import logging
import numpy as np
import pyaudiowpatch as pyaudio # Assuming pyaudiowpatch is the intended library
from aiortc.contrib.media import MediaStreamTrack
from aiortc.mediastreams import AudioFrame

logger = logging.getLogger("audio_tracks") # Or pass logger instance if preferred

class BaseAudioTrack(MediaStreamTrack):
    kind = "audio"

    def __init__(self, sample_rate=48000, channels=1):
        super().__init__()
        self.sample_rate = int(sample_rate)
        self.channels = int(channels)
        self._stop_event = asyncio.Event()
        self._pts_counter = 0
        self._frame_duration_seconds = 0.02 # Default, can be overridden or calculated

    def _create_audio_frame(self, audio_data_reshaped, num_samples_per_channel):
        """
        Creates an AudioFrame from processed audio data.
        audio_data_reshaped: numpy array shaped (num_channels, num_samples_per_channel)
        num_samples_per_channel: int, number of samples per channel in this frame
        """
        current_layout = "mono" if self.channels == 1 else "stereo"

        frame = AudioFrame.from_ndarray(
            audio_data_reshaped,
            format='s16',
            layout=current_layout
        )
        frame.pts = self._pts_counter
        self._pts_counter += num_samples_per_channel # Increment by actual samples in this frame
        frame.sample_rate = self.sample_rate
        
        # Reduced verbosity for general frame creation logging
        # logger.info(
        #     f"{self.__class__.__name__}.recv: Created AudioFrame - pts: {frame.pts}, "
        #     f"samples: {frame.samples}, sample_rate: {frame.sample_rate}"
        # )
        return frame

    async def _generate_or_capture_frame_data(self):
        """
        Abstract method to be implemented by subclasses.
        Should return a tuple: (audio_data_reshaped, num_samples_per_channel)
        audio_data_reshaped: numpy array of int16 audio samples, shaped (channels, samples_per_channel)
        num_samples_per_channel: number of samples per channel in this frame.
        Return (None, None) for transient issues where recv should retry.
        Raise StopAsyncIteration for fatal errors or normal track termination.
        """
        raise NotImplementedError("Subclasses must implement _generate_or_capture_frame_data")

    def _should_stop_iteration(self):
        """Checks if the iteration should stop."""
        return self._stop_event.is_set()

    def _handle_stop_iteration(self):
        """Raises StopAsyncIteration when stopping."""
        logger.debug(f"{self.__class__.__name__} stopping iteration.")
        raise StopAsyncIteration

    async def _handle_no_data(self):
        """
        Handles cases where _generate_or_capture_frame_data returns no data,
        allowing for a retry or graceful stop.
        """
        if self._should_stop_iteration():
            self._handle_stop_iteration()
        # Short pause before retrying to avoid busy-looping on transient errors
        await asyncio.sleep(0.01)
        return await self.recv()


    async def _post_process_frame(self, frame):
        """
        Hook for subclasses to perform actions after a frame is generated
        but before it's returned, e.g., simulate real-time delay.
        By default, does nothing.
        """
        pass

    async def recv(self):
        if self._should_stop_iteration():
            return self._handle_stop_iteration()

        try:
            audio_data_reshaped, num_samples_this_frame = await self._generate_or_capture_frame_data()
        except StopAsyncIteration: # Allow _generate_or_capture_frame_data to signal stop
            raise
        except Exception as e:
            logger.error(f"{self.__class__.__name__} error in _generate_or_capture_frame_data: {e}")
            self._stop_event.set() # Ensure stop on unexpected error
            return self._handle_stop_iteration()


        if audio_data_reshaped is None or num_samples_this_frame == 0:
            # logger.debug(f"{self.__class__.__name__} received no data, handling...")
            return await self._handle_no_data()

        if audio_data_reshaped.shape[1] == 0 : # Double check if data is empty after processing
            logger.warning(f"{self.__class__.__name__}.recv: audio_data_reshaped has 0 samples, attempting to get next frame.")
            return await self._handle_no_data()

        frame = self._create_audio_frame(audio_data_reshaped, num_samples_this_frame)
        
        await self._post_process_frame(frame) # Hook for subclasses

        return frame

    async def stop(self):
        logger.info(f"Stopping {self.__class__.__name__}...")
        self._stop_event.set()
        # Subclasses should override to join tasks or release resources if needed,
        # then call super().stop() or manage _stop_event directly.


class AudioInputTrack(BaseAudioTrack):
    def __init__(self, p_instance, device_index, sample_rate=48000, channels=2, frames_per_buffer_ms=20):
        super().__init__(sample_rate=sample_rate, channels=channels)
        self.p = p_instance
        self.device_index = device_index
        self.format = pyaudio.paInt16 # paInt16
        self.frames_per_buffer = int(self.sample_rate * frames_per_buffer_ms / 1000)
        self._frame_duration_seconds = frames_per_buffer_ms / 1000.0

        self._queue = asyncio.Queue()
        self._capture_task = None
        self.stream = None
        self._is_pyaudio_owner = False

        if self.p is None:
            logger.info("PyAudio instance not provided, creating a new one for AudioInputTrack.")
            self.p = pyaudio.PyAudio()
            self._is_pyaudio_owner = True

        if self.device_index is None or self.device_index < 0:
            logger.error("AudioInputTrack: No valid device_index provided.")
            self._stop_event.set()
            return

        try:
            device_info = self.p.get_device_info_by_index(self.device_index)
            logger.info(
                f"AudioInputTrack: Initializing with device '{device_info['name']}' (Index {self.device_index}) "
                f"Rate: {self.sample_rate}, Channels: {self.channels}, Format: paInt16, "
                f"Frames per buffer: {self.frames_per_buffer}"
            )
        except Exception as e:
            logger.error(f"AudioInputTrack: Failed to get device info for index {self.device_index}: {e}")
            self._stop_event.set()
            return
        
        logger.debug(f"AudioInputTrack __init__: p_instance={'provided' if p_instance else 'None'}, device_index={device_index}, sample_rate={sample_rate}, channels={channels}")
        # Start capture thread immediately if not stopping
        if not self._stop_event.is_set():
            logger.debug("AudioInputTrack __init__: _stop_event not set, ensuring capture task is running.")
            self._ensure_capture_task_running()
        else:
            logger.warning("AudioInputTrack __init__: _stop_event IS SET. Capture task will not start.")


    def _start_capture_thread_blocking(self):
        """This runs in a separate thread to read from PyAudio stream."""
        logger.debug(f"AudioInputTrack: _start_capture_thread_blocking called for device {self.device_index}.")
        if self.device_index == -1:
            logger.error("Cannot start capture: No valid audio device.")
            self._stop_event.set()
            return

        try:
            self.stream = self.p.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=self.device_index,
                frames_per_buffer=self.frames_per_buffer,
                stream_callback=None
            )
            logger.info(f"PyAudio stream opened successfully for device {self.device_index} with sr={self.sample_rate}, ch={self.channels}, frames_per_buffer={self.frames_per_buffer}.")
        except Exception as e:
            logger.error(f"Failed to open PyAudio stream for device {self.device_index}: {e}")
            self.stream = None
            self._stop_event.set()
            return

        loop = asyncio.get_event_loop() # Get loop from the main thread context where run_in_executor was called
        logger.debug("AudioInputTrack: Starting PyAudio read loop.")
        while not self._stop_event.is_set() and self.stream and self.stream.is_active():
            try:
                # logger.debug(f"AudioInputTrack: Attempting to read {self.frames_per_buffer} frames from PyAudio stream.")
                raw_audio_bytes = self.stream.read(self.frames_per_buffer, exception_on_overflow=False)
                logger.debug(f"AudioInputTrack: Read {len(raw_audio_bytes)} bytes from PyAudio stream. Putting into queue.")
                asyncio.run_coroutine_threadsafe(self._queue.put(raw_audio_bytes), loop)
            except IOError as e:
                if hasattr(pyaudio, 'paInputOverflowed') and e.errno == pyaudio.paInputOverflowed: # type: ignore
                    logger.warning("PyAudio input overflowed. Some audio data may have been lost.")
                else:
                    logger.error(f"PyAudio read error: {e}")
                    self._stop_event.set()
                    break
            except Exception as e:
                logger.error(f"Exception in PyAudio capture loop: {e}")
                self._stop_event.set()
                break
        
        if self.stream:
            try:
                if self.stream.is_active():
                    self.stream.stop_stream()
                self.stream.close()
                logger.info("PyAudio stream stopped and closed.")
            except Exception as e:
                logger.error(f"Error closing PyAudio stream: {e}")
        self.stream = None
        logger.debug("AudioInputTrack: Exited PyAudio read loop.")
        # Signal queue that no more items will be added if we stopped
        if self._stop_event.is_set():
            logger.debug("AudioInputTrack: _stop_event is set, putting None sentinel into queue.")
            asyncio.run_coroutine_threadsafe(self._queue.put(None), loop)


    def _ensure_capture_task_running(self):
        logger.debug(f"AudioInputTrack: _ensure_capture_task_running called. Task is {self._capture_task}, _stop_event is {self._stop_event.is_set()}")
        if self._capture_task is None and not self._stop_event.is_set():
            if self.device_index == -1: # Should be caught by __init__
                logger.error("AudioInputTrack: Cannot start capture task, no valid audio device (_ensure_capture_task_running).")
                self._stop_event.set()
                return False # Indicate failure to start

            loop = asyncio.get_event_loop()
            self._capture_task = loop.run_in_executor(None, self._start_capture_thread_blocking)
            logger.info("AudioInputTrack capture task has been started.")
            return True # Indicate task started
        elif self._capture_task is not None and not self._stop_event.is_set():
            logger.debug("AudioInputTrack: Capture task already running.")
            return True # Already running
        elif self._stop_event.is_set():
            logger.debug("AudioInputTrack: _ensure_capture_task_running: _stop_event is set, not starting task.")
            return False
        logger.debug("AudioInputTrack: _ensure_capture_task_running: Condition not met to start or confirm task.")
        return False # Not started because stop_event is set or other issue


    async def _generate_or_capture_frame_data(self):
        logger.debug("AudioInputTrack: _generate_or_capture_frame_data called.")
        if not self._ensure_capture_task_running() and self._queue.empty():
            logger.debug("AudioInputTrack: Capture task not running and queue empty, stopping iteration.")
            # If task couldn't start (e.g. stop_event set) and queue is empty
            raise StopAsyncIteration

        try:
            logger.debug(f"AudioInputTrack: Waiting for item from queue (qsize: {self._queue.qsize()}). Timeout: {self._frame_duration_seconds * 2}s")
            raw_audio_bytes = await asyncio.wait_for(self._queue.get(), timeout=self._frame_duration_seconds * 2)
            
            if raw_audio_bytes is None: # Sentinel for end of stream from capture thread
                logger.info("AudioInputTrack received None sentinel from queue, signaling stop.")
                raise StopAsyncIteration
            logger.debug(f"AudioInputTrack: Got {len(raw_audio_bytes)} bytes from queue.")
            
        except asyncio.TimeoutError:
            logger.debug(f"AudioInputTrack: Timeout waiting for item from queue. _stop_event: {self._stop_event.is_set()}, qsize: {self._queue.qsize()}")
            if self._should_stop_iteration() and self._queue.empty(): # Check if we should stop
                logger.debug("AudioInputTrack: Stopping iteration due to timeout and stop condition.")
                raise StopAsyncIteration
            logger.debug("AudioInputTrack: Timeout, but not stopping. Returning (None,0) to retry.")
            return None, 0 # Indicate no data this attempt, recv will retry or stop

        samples_int16 = np.frombuffer(raw_audio_bytes, dtype=np.int16)
        logger.debug(f"AudioInputTrack: Converted to {samples_int16.size} int16 samples from {len(raw_audio_bytes)} bytes.")
        # logger.debug(f"AudioInputTrack.recv: Converted to {samples_int16.size} int16 samples.")
        # if samples_int16.size > 0:
        #     logger.debug(f"AudioInputTrack.recv: Sample values min: {np.min(samples_int16)}, max: {np.max(samples_int16)}, mean: {np.mean(samples_int16)}")

        if samples_int16.size == 0:
            logger.warning("AudioInputTrack: samples_int16.size is 0 after frombuffer. Raw bytes length was %s.", len(raw_audio_bytes))
            return None, 0 # No data

        num_samples_per_channel_total = samples_int16.size // self.channels
        logger.debug(f"AudioInputTrack: num_samples_per_channel_total = {num_samples_per_channel_total}")
        
        # Reshape based on channels
        if self.channels == 1:
            audio_data_reshaped = samples_int16.reshape(1, -1)
        elif self.channels == 2:
            if samples_int16.ndim == 1:
                left_channel = samples_int16[0::2]
                right_channel = samples_int16[1::2]
                audio_data_reshaped = np.array([left_channel, right_channel], dtype=np.int16)
            elif samples_int16.shape[1] == self.channels: # Already (samples, channels)
                 audio_data_reshaped = samples_int16.T # Transpose to (channels, samples)
            else:
                logger.error(f"AudioInputTrack: Unexpected samples_int16 shape {samples_int16.shape} for {self.channels} channels.")
                return None, 0 # Error in data shape
        else:
            logger.error(f"AudioInputTrack: Unsupported number of channels: {self.channels}")
            return None, 0 # Unsupported channels

        logger.debug(f"AudioInputTrack: Reshaped audio data to {audio_data_reshaped.shape}. Returning data.")
        return audio_data_reshaped, num_samples_per_channel_total


    def _should_stop_iteration(self):
        # Override to check queue status as well if stopping
        if self._stop_event.is_set() and self._queue.empty():
            return True
        return self._stop_event.is_set()


    async def stop(self):
        logger.info(f"AudioInputTrack.stop() called. Current _stop_event: {self._stop_event.is_set()}")
        await super().stop() # Sets _stop_event
        logger.info(f"AudioInputTrack.stop(): super().stop() called, _stop_event is now: {self._stop_event.is_set()}")

        if self._capture_task is not None:
            logger.info("AudioInputTrack: Attempting to stop and join audio capture task...")
            # The capture thread checks _stop_event, and should put None in queue before exiting.
            try:
                # It's possible the task is already finishing if _stop_event was set and queue processed None
                if not self._capture_task.done():
                     # Ensure the queue is signaled if not already by the thread, especially if stop is forceful
                    if self._queue and self._queue.empty(): # Check if queue is empty before potentially adding another None
                        logger.debug("AudioInputTrack.stop(): Capture task not done, queue empty, putting None sentinel.")
                        self._queue.put_nowait(None) # Help unblock queue.get() in capture task if it's stuck there
                
                logger.debug(f"AudioInputTrack.stop(): Waiting for capture task {self._capture_task} to complete.")
                await asyncio.wait_for(self._capture_task, timeout=3.0) # Increased timeout slightly
                logger.info("AudioInputTrack: Audio capture task finished.")
            except asyncio.TimeoutError:
                logger.warning("AudioInputTrack: Audio capture task did not finish in time during stop.")
            except asyncio.CancelledError:
                logger.warning("AudioInputTrack: Audio capture task was cancelled during stop.")
            except Exception as e:
                logger.error(f"AudioInputTrack: Exception while waiting for audio capture task to stop: {e}")
            self._capture_task = None
        else:
            logger.info("AudioInputTrack.stop(): No capture task to stop.")
        
        # Ensure queue is drained or cleared to release any waiting recv
        logger.debug(f"AudioInputTrack.stop(): Draining queue (current size: {self._queue.qsize() if self._queue else 'N/A'}).")
        if self._queue:
            while not self._queue.empty():
                try:
                    item = self._queue.get_nowait()
                    logger.debug(f"AudioInputTrack.stop(): Drained item from queue (type: {type(item)}).")
                except asyncio.QueueEmpty:
                    break
            logger.debug("AudioInputTrack.stop(): Queue drained.")
        
        if self.p and self._is_pyaudio_owner:
            logger.info("AudioInputTrack: Terminating owned PyAudio instance.")
            self.p.terminate()
            self.p = None
            logger.info("AudioInputTrack: Owned PyAudio instance terminated.")


class SineWaveTrack(BaseAudioTrack):
    def __init__(self, frequency=440, sample_rate=48000, channels=1, samples_per_frame=None):
        super().__init__(sample_rate=sample_rate, channels=channels)
        self.frequency = float(frequency)
        # If samples_per_frame is not given, calculate for 20ms, common for WebRTC
        self.samples_per_frame = int(samples_per_frame if samples_per_frame is not None else self.sample_rate * 0.02)
        self._frame_duration_seconds = self.samples_per_frame / self.sample_rate

        self._time_offset = 0.0  # Current time offset for sine wave generation
        self.amplitude = 0.3 * 32767  # Amplitude for 16-bit audio

        self._time_increment_per_sample = 1.0 / self.sample_rate
        logger.info(
            f"SineWaveTrack: Initializing with Freq: {self.frequency}Hz, Rate: {self.sample_rate}Hz, "
            f"Channels: {self.channels}, Samples/Frame: {self.samples_per_frame}"
        )

    async def _generate_or_capture_frame_data(self):
        num_samples = self.samples_per_frame
        
        # Generate time array for the current frame
        t = np.arange(
            self._time_offset, 
            self._time_offset + num_samples * self._time_increment_per_sample, 
            self._time_increment_per_sample
        )[:num_samples]
        
        # Generate sine wave
        wave_mono = self.amplitude * np.sin(2 * np.pi * self.frequency * t)
        samples_int16_mono = wave_mono.astype(np.int16)
        # logger.debug(f"SineWaveTrack.recv: Generated {samples_int16_mono.size} mono int16 samples.")

        # Prepare frame_data based on channels
        if self.channels == 1:
            frame_data = samples_int16_mono.reshape(1, -1)
        elif self.channels == 2:
            # Duplicate mono wave for stereo
            frame_data = np.array([samples_int16_mono, samples_int16_mono], dtype=np.int16)
        else:
            logger.error(f"SineWaveTrack.recv: Unsupported number of channels: {self.channels}")
            # Fallback to silent mono to prevent crash, and signal stop
            self._stop_event.set()
            raise StopAsyncIteration("Unsupported channel count for SineWaveTrack")
            
        # logger.debug(f"SineWaveTrack.recv: Reshaped audio data to {frame_data.shape}")
            
        self._time_offset += num_samples * self._time_increment_per_sample
        return frame_data, num_samples

    async def _post_process_frame(self, frame):
        """
        Hook for subclasses to perform actions after a frame is generated
        but before it's returned. For SineWaveTrack, this was previously
        used for an asyncio.sleep, which is now removed to potentially
        improve smoothness as the consumer should handle pacing.
        """
        pass # Removed: await asyncio.sleep(self._frame_duration_seconds)

    async def stop(self):
        await super().stop() # Sets _stop_event
        logger.info("SineWaveTrack stopped.")