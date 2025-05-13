import asyncio
import logging
import numpy as np
from aiortc.contrib.media import MediaStreamTrack
from aiortc.mediastreams import AudioFrame
from audio_capture_manager import NumpyRingBuffer # Import RingBuffer

logger = logging.getLogger("audio_tracks") # Or pass logger instance if preferred
# logger_ringbuffer = logging.getLogger("audio_tracks.ringbuffer") # RingBuffer logs from its own module


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

    # _handle_no_data method is removed as its logic is incorporated into recv

    async def _post_process_frame(self, frame):
        """
        Hook for subclasses to perform actions after a frame is generated
        but before it's returned, e.g., simulate real-time delay.
        By default, does nothing.
        """
        pass

    async def recv(self):
        # Loop until a frame is available or stop is signaled
        while True: # Loop will be broken by returning a frame or raising StopAsyncIteration
            if self._should_stop_iteration():
                # If the track is meant to stop (e.g. event set and conditions met by subclass override)
                logger.debug(f"{self.__class__.__name__}.recv: Stop iteration condition met.")
                self._handle_stop_iteration() # Raises StopAsyncIteration

            try:
                audio_data_reshaped, num_samples_this_frame = await self._generate_or_capture_frame_data()
            except StopAsyncIteration:
                logger.debug(f"{self.__class__.__name__}.recv: StopAsyncIteration caught from _generate_or_capture_frame_data.")
                raise # Propagate stop signal
            except Exception as e:
                logger.error(f"{self.__class__.__name__} error in _generate_or_capture_frame_data: {e}")
                self._stop_event.set() # Ensure stop on unexpected error
                self._handle_stop_iteration() # Raises StopAsyncIteration

            if audio_data_reshaped is not None and num_samples_this_frame > 0:
                # Data is available
                if audio_data_reshaped.shape[1] == 0: # Should ideally not happen if num_samples_this_frame > 0
                    logger.warning(f"{self.__class__.__name__}.recv: audio_data_reshaped has 0 samples despite num_samples_this_frame={num_samples_this_frame}. Retrying.")
                    # Treat as no data / continue loop after delay
                else:
                    frame = self._create_audio_frame(audio_data_reshaped, num_samples_this_frame)
                    await self._post_process_frame(frame)
                    return frame # Frame is ready, exit loop and method

            # No data available from _generate_or_capture_frame_data, or inconsistent data, but not stopping yet.
            # Loop again after a short delay. This makes recv() block until a frame is ready.
            if self._should_stop_iteration(): # Check again if stop was signaled during the no-data phase
                logger.debug(f"{self.__class__.__name__}.recv: Stop iteration condition met after attempting to get data.")
                self._handle_stop_iteration()
            
            await asyncio.sleep(0.01) # Wait a bit before retrying _generate_or_capture_frame_data

    def stop(self):
        logger.info(f"Stopping {self.__class__.__name__}...")
        self._stop_event.set() # This is a synchronous operation
        # Subclasses should override to join tasks or release resources if needed,
        # then call super().stop() or manage _stop_event directly.


class AudioInputTrack(BaseAudioTrack):
    def __init__(self, ring_buffer: NumpyRingBuffer, sample_rate: int, channels: int, device_actual_channels: int, frames_per_buffer_ms=20):
        super().__init__(sample_rate=sample_rate, channels=channels) # Output channels
        self._ring_buffer = ring_buffer
        self.device_actual_channels = device_actual_channels # Channels of the actual audio source
        self._frame_duration_seconds = frames_per_buffer_ms / 1000.0
        
        if not isinstance(self._ring_buffer, NumpyRingBuffer):
            logger.error("AudioInputTrack: Invalid ring_buffer provided.")
            self._stop_event.set() # Signal error state
            return

        logger.info(
            f"AudioInputTrack: Initializing with provided RingBuffer. "
            f"Output Rate: {self.sample_rate}, Output Channels: {self.channels}, "
            f"Device Actual Channels: {self.device_actual_channels}, Frame Duration: {self._frame_duration_seconds}s"
        )

    # Removed _start_capture_thread_blocking and _ensure_capture_task_running
    # Capture is now managed externally by AudioCaptureManager

    async def _generate_or_capture_frame_data(self):
        logger.debug("AudioInputTrack: _generate_or_capture_frame_data called.")
        
        if self._ring_buffer.is_closed() and self._ring_buffer.is_empty():
            logger.info("AudioInputTrack: RingBuffer is closed and empty. Stopping iteration.")
            raise StopAsyncIteration

        try:
            # Timeout slightly longer than frame duration to allow for processing delays
            raw_audio_bytes = await asyncio.wait_for(self._ring_buffer.read(), timeout=self._frame_duration_seconds * 2.5)
            if raw_audio_bytes is None: # RingBuffer is closed and empty
                logger.info("AudioInputTrack received None from ring_buffer.read(), signaling stop.")
                raise StopAsyncIteration
            
        except asyncio.TimeoutError:
            logger.debug(f"AudioInputTrack: Timeout waiting for data from ring_buffer. Buffer empty: {self._ring_buffer.is_empty()}, closed: {self._ring_buffer.is_closed()}")
            # If stopping and buffer is truly empty (and potentially closed), then stop.
            if self._should_stop_iteration(): # Checks stop_event and buffer emptiness
                logger.info("AudioInputTrack: Timeout and should_stop_iteration is true. Stopping.")
                raise StopAsyncIteration
            return None, 0 # Indicate no data this attempt, recv will retry or stop
        except Exception as e:
            logger.error(f"AudioInputTrack: Error reading from ring_buffer: {e}", exc_info=True)
            self._stop_event.set()
            raise StopAsyncIteration


        samples_int16 = np.frombuffer(raw_audio_bytes, dtype=np.int16)

        if samples_int16.size == 0:
            logger.warning("AudioInputTrack: samples_int16.size is 0 after frombuffer. Raw bytes length was %s.", len(raw_audio_bytes))
            return None, 0 # No data

        # Samples per channel based on ACTUAL device channels used for capture
        num_samples_per_channel_total = samples_int16.size // self.device_actual_channels
        audio_data_reshaped = samples_int16.reshape(1, -1)  # Reshape to (1, num_samples), which is required by aiortc, whatever how many channels we have

        return np.ascontiguousarray(audio_data_reshaped), num_samples_per_channel_total


    def _should_stop_iteration(self):
        # Stop iteration only if the stop event is set AND the ring_buffer is empty and closed.
        # This allows processing of remaining items in the ring_buffer after stop() is called externally.
        if self._stop_event.is_set():
            if self._ring_buffer.is_empty() and self._ring_buffer.is_closed():
                logger.debug("AudioInputTrack._should_stop_iteration: True (stop event set AND ring_buffer empty & closed).")
                return True
            elif self._ring_buffer.is_empty() and not self._ring_buffer.is_closed():
                 logger.debug(f"AudioInputTrack._should_stop_iteration: False (stop event set, ring_buffer empty but NOT YET closed. Waiting for buffer to close or data). Size: {self._ring_buffer.qsize()}")
                 return False # Still wait for buffer to be formally closed by writer
            elif not self._ring_buffer.is_empty():
                logger.debug(f"AudioInputTrack._should_stop_iteration: False (stop event set, but ring_buffer not empty, size: {self._ring_buffer.qsize()}).")
                return False # Continue processing ring_buffer
        # logger.debug("AudioInputTrack._should_stop_iteration: False (stop event not set).")
        return False # Stop event not set, continue normally


    def stop(self):
        logger.info(f"AudioInputTrack.stop() called. Current _stop_event: {self._stop_event.is_set()}")
        super().stop() # Sets self._stop_event = True
        # The ring_buffer is managed (closed) by AudioCaptureManager.
        # This track instance no longer owns PyAudio or the capture thread.
        logger.info(f"AudioInputTrack.stop(): super().stop() completed. _stop_event is now: {self._stop_event.is_set()}. RingBuffer will be processed if items remain.")
        # Ensure any waiting read() in _generate_or_capture_frame_data can see the stop
        # by potentially waking it if it's waiting on _data_available.
        # However, ring_buffer.close() from the manager should handle this.


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

    def stop(self):
        super().stop() # Sets _stop_event
        logger.info("SineWaveTrack stopped.")