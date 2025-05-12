import asyncio
import logging
import numpy as np
import pyaudiowpatch as pyaudio # Assuming pyaudiowpatch is the intended library
import threading # Added for RingBuffer
from aiortc.contrib.media import MediaStreamTrack
from aiortc.mediastreams import AudioFrame

logger = logging.getLogger("audio_tracks") # Or pass logger instance if preferred
logger_ringbuffer = logging.getLogger("audio_tracks.ringbuffer")


class NumpyRingBuffer:
    def __init__(self, capacity_items: int, item_size_bytes: int):
        self.capacity_items = capacity_items
        self.item_size_bytes = item_size_bytes
        
        self.buffer = np.zeros(capacity_items * item_size_bytes, dtype=np.uint8)
        self.write_idx = 0  # Index for next item to write
        self.read_idx = 0   # Index for next item to read
        self.count = 0      # Number of items currently in buffer
        
        self._lock = threading.Lock() # For write access and shared state (count, write_idx, read_idx, _closed)
        self._read_lock = asyncio.Lock() # Ensures only one async reader coroutine critical section at a time
        self._data_available = asyncio.Event() # Signaled when new data is written, for async readers
        self._closed = False

    def write(self, data_bytes: bytes) -> bool:
        """Writes a single item (data_bytes) to the buffer.
           data_bytes must be of length self.item_size_bytes.
           Returns True if write was successful, False if closed or data size mismatch.
        """
        if len(data_bytes) != self.item_size_bytes:
            logger_ringbuffer.error(f"RingBuffer.write: Data size {len(data_bytes)} != expected item_size_bytes {self.item_size_bytes}")
            return False

        with self._lock:
            if self._closed:
                logger_ringbuffer.debug("RingBuffer.write: Attempted write to closed buffer.")
                return False

            # Overwrite oldest data if buffer is full
            if self.count == self.capacity_items:
                # logger_ringbuffer.warning("RingBuffer.write: Buffer full, overwriting oldest data.")
                # Advance read_idx because we are overwriting the oldest item
                self.read_idx = (self.read_idx + 1) % self.capacity_items
                self.count -= 1 # Effectively removing the oldest item being overwritten

            buffer_start_offset = self.write_idx * self.item_size_bytes
            buffer_end_offset = buffer_start_offset + self.item_size_bytes
            self.buffer[buffer_start_offset:buffer_end_offset] = np.frombuffer(data_bytes, dtype=np.uint8)
            
            self.write_idx = (self.write_idx + 1) % self.capacity_items
            self.count += 1
            
            self._data_available.set() # Signal any waiting readers
        return True

    async def read(self) -> bytes | None:
        """Reads a single item from the buffer.
           Waits if buffer is empty until data is available or buffer is closed.
           Returns item_size_bytes of data, or None if closed and empty.
        """
        async with self._read_lock:
            while True:
                with self._lock: # Short critical section to check state and potentially retrieve data
                    if self.count > 0:
                        buffer_start_offset = self.read_idx * self.item_size_bytes
                        buffer_end_offset = buffer_start_offset + self.item_size_bytes
                        data_item_np = self.buffer[buffer_start_offset:buffer_end_offset]
                        
                        self.read_idx = (self.read_idx + 1) % self.capacity_items
                        self.count -= 1
                        
                        if self.count == 0: # If buffer became empty after read
                            self._data_available.clear() # No more data for now, subsequent reads will wait
                        
                        return data_item_np.tobytes()

                    if self._closed: # Buffer is empty and closed
                        self._data_available.set() # Ensure any other waiters also wake up and see it's closed
                        return None
                
                # Buffer is empty but not closed, wait for data
                await self._data_available.wait()

    def close(self):
        """Signals that no more data will be written to the buffer."""
        with self._lock:
            if not self._closed:
                logger_ringbuffer.info("RingBuffer.close: Closing buffer.")
                self._closed = True
                self._data_available.set() # Wake up any waiting readers so they can see it's closed

    def is_empty(self) -> bool:
        with self._lock:
            return self.count == 0

    def qsize(self) -> int:
        """Returns the number of items currently in the buffer."""
        with self._lock:
            return self.count
            
    def is_closed(self) -> bool:
        with self._lock:
            return self._closed


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

    async def stop(self):
        logger.info(f"Stopping {self.__class__.__name__}...")
        self._stop_event.set()
        # Subclasses should override to join tasks or release resources if needed,
        # then call super().stop() or manage _stop_event directly.


class AudioInputTrack(BaseAudioTrack):
    def __init__(self, p_instance, device_index, sample_rate=48000, channels=2, frames_per_buffer_ms=20):
        self.device_actual_channels = channels # Store actual device channels
        super().__init__(sample_rate=sample_rate, channels=channels)
        self.p = p_instance
        self.device_index = device_index
        self.format = pyaudio.paInt16 # paInt16
        self.frames_per_buffer = int(self.sample_rate * frames_per_buffer_ms / 1000)
        self._frame_duration_seconds = frames_per_buffer_ms / 1000.0

        # RingBuffer setup
        self.bytes_per_pyaudio_buffer = self.frames_per_buffer * self.device_actual_channels * 2 # 2 bytes for paInt16
        BUFFER_CAPACITY_IN_CHUNKS = 100  # Store 100 PyAudio chunks (e.g., 100 * 20ms = 2 seconds)
        self._ring_buffer = NumpyRingBuffer(
            capacity_items=BUFFER_CAPACITY_IN_CHUNKS,
            item_size_bytes=self.bytes_per_pyaudio_buffer
        )
        
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
                f"Rate: {self.sample_rate}, Device Actual Channels: {self.device_actual_channels}, Output Track Channels: {self.channels}, Format: paInt16, "
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


    def _start_capture_thread_blocking(self, main_loop):
        """This runs in a separate thread to read from PyAudio stream."""
        logger.debug(f"AudioInputTrack: _start_capture_thread_blocking called for device {self.device_index} with main_loop: {main_loop}.")
        if self.device_index == -1:
            logger.error("Cannot start capture: No valid audio device.")
            self._stop_event.set()
            return

        try:
            self.stream = self.p.open(
                format=self.format,
                channels=self.device_actual_channels, # Use actual device channels for capture
                rate=self.sample_rate,
                input=True,
                input_device_index=self.device_index,
                frames_per_buffer=self.frames_per_buffer,
                stream_callback=None
            )
            logger.info(f"PyAudio stream opened successfully for device {self.device_index} with sr={self.sample_rate}, ch={self.device_actual_channels}, frames_per_buffer={self.frames_per_buffer}.")
        except Exception as e:
            logger.error(f"Failed to open PyAudio stream for device {self.device_index}: {e}")
            self.stream = None
            self._stop_event.set()
            return

        logger.debug(f"AudioInputTrack (capture_thread): Using provided main_loop: {main_loop}. Starting PyAudio read loop.")
        while not self._stop_event.is_set() and self.stream and self.stream.is_active():
            try:
                logger.debug(f"AudioInputTrack (capture_thread): Loop start. Stop event: {self._stop_event.is_set()}, Stream active: {self.stream.is_active() if self.stream else 'N/A'}")
                logger.debug(f"AudioInputTrack (capture_thread): Attempting to read {self.frames_per_buffer} frames from PyAudio stream.")
                raw_audio_bytes = self.stream.read(self.frames_per_buffer, exception_on_overflow=False)
                bytes_read = len(raw_audio_bytes)
                logger.debug(f"AudioInputTrack (capture_thread): Read {bytes_read} bytes from PyAudio stream.")

                if bytes_read == 0:
                    logger.warning("AudioInputTrack (capture_thread): Read 0 bytes from stream. Stream might be closing or no data available.")
                    # Optionally, add a small sleep here if this happens often without error
                    # await asyncio.sleep(0.001) # Requires loop.call_soon_threadsafe or similar if using asyncio.sleep
                    # For a blocking thread, a simple time.sleep might be okay if issues persist
                    # import time
                    # time.sleep(0.001)
                    continue # Try reading again

                logger.debug(f"AudioInputTrack (capture_thread): Writing {bytes_read} bytes into ring_buffer.")
                if not self._ring_buffer.write(raw_audio_bytes):
                    logger.error("AudioInputTrack (capture_thread): Failed to write to ring_buffer (closed or size mismatch). Stopping.")
                    self._stop_event.set() # Signal stop if write fails critically
                    break
                logger.debug(f"AudioInputTrack (capture_thread): Successfully wrote data to ring_buffer. Buffer size: {self._ring_buffer.qsize()}")

            except IOError as e:
                # Check for specific PyAudio error codes if available and useful
                # For example, paInputOverflowed is already handled.
                # Add more specific checks if other IOError types are common.
                if hasattr(pyaudio, 'paInputOverflowed') and e.errno == pyaudio.paInputOverflowed: # type: ignore
                    logger.warning(f"AudioInputTrack (capture_thread): PyAudio input overflowed. {e}")
                else:
                    logger.error(f"AudioInputTrack (capture_thread): PyAudio IOError in read loop: {e} (errno: {e.errno if hasattr(e, 'errno') else 'N/A'})")
                    self._stop_event.set()
                    logger.info("AudioInputTrack (capture_thread): Set stop_event due to PyAudio IOError.")
                    break # Exit loop
            except Exception as e:
                logger.error(f"AudioInputTrack (capture_thread): Unexpected exception in PyAudio capture loop: {e}", exc_info=True)
                self._stop_event.set()
                logger.info("AudioInputTrack (capture_thread): Set stop_event due to unexpected exception.")
                break # Exit loop
        
        logger.debug(f"AudioInputTrack (capture_thread): Exited read loop. Stop event: {self._stop_event.is_set()}, Stream: {self.stream}")
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
        # Signal ring_buffer that no more items will be added
        logger.debug("AudioInputTrack (capture_thread): Closing ring_buffer.")
        self._ring_buffer.close()
        logger.debug("AudioInputTrack (capture_thread): ring_buffer closed.")


    def _ensure_capture_task_running(self):
        logger.debug(f"AudioInputTrack: _ensure_capture_task_running called. Task is {self._capture_task}, _stop_event is {self._stop_event.is_set()}")
        if self._capture_task is None and not self._stop_event.is_set():
            if self.device_index == -1: # Should be caught by __init__
                logger.error("AudioInputTrack: Cannot start capture task, no valid audio device (_ensure_capture_task_running).")
                self._stop_event.set()
                return False # Indicate failure to start

            current_loop = asyncio.get_event_loop() # This will be the main event loop
            logger.debug(f"AudioInputTrack: Scheduling _start_capture_thread_blocking with current_loop: {current_loop}")
            self._capture_task = current_loop.run_in_executor(None, self._start_capture_thread_blocking, current_loop) # Pass current_loop
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
        # If task isn't running (or couldn't start) and buffer is empty & closed, stop.
        if not self._ensure_capture_task_running() and self._ring_buffer.is_empty() and self._ring_buffer.is_closed():
            logger.debug("AudioInputTrack: Capture task not running, ring_buffer empty and closed. Stopping iteration.")
            raise StopAsyncIteration
        elif not self._ensure_capture_task_running() and self._ring_buffer.is_empty():
             logger.warning("AudioInputTrack: Capture task not running and ring_buffer empty, but buffer not marked closed. May lead to StopAsyncIteration on timeout if no data arrives.")


        try:
            raw_audio_bytes = await asyncio.wait_for(self._ring_buffer.read(), timeout=self._frame_duration_seconds * 2)
            if raw_audio_bytes is None: # RingBuffer is closed and empty
                logger.info("AudioInputTrack received None from ring_buffer.read(), signaling stop.")
                raise StopAsyncIteration
            
        except asyncio.TimeoutError:
            # If stopping and buffer is truly empty (and potentially closed), then stop.
            if self._should_stop_iteration() and self._ring_buffer.is_empty():
                raise StopAsyncIteration
            return None, 0 # Indicate no data this attempt, recv will retry or stop

        samples_int16 = np.frombuffer(raw_audio_bytes, dtype=np.int16)

        if samples_int16.size == 0:
            logger.warning("AudioInputTrack: samples_int16.size is 0 after frombuffer. Raw bytes length was %s.", len(raw_audio_bytes))
            return None, 0 # No data

        # Calculate samples per channel based on ACTUAL device channels used for capture
        num_samples_per_channel_total = samples_int16.size // self.device_actual_channels
        audio_data_reshaped = samples_int16.reshape(1, -1)

        return np.ascontiguousarray(audio_data_reshaped), num_samples_per_channel_total


    def _should_stop_iteration(self):
        # Override to check queue status as well if stopping.
        # Stop iteration only if the stop event is set AND the queue is empty.
        # This allows processing of remaining items in the ring_buffer after stop() is called.
        if self._stop_event.is_set():
            if self._ring_buffer.is_empty(): # And implicitly, if it's also closed, read() will return None.
                logger.debug("AudioInputTrack._should_stop_iteration: True (stop event set AND ring_buffer empty).")
                return True
            else:
                logger.debug(f"AudioInputTrack._should_stop_iteration: False (stop event set, but ring_buffer not empty, size: {self._ring_buffer.qsize()}).")
                return False # Continue processing ring_buffer
        logger.debug("AudioInputTrack._should_stop_iteration: False (stop event not set).")
        return False # Stop event not set, continue normally


    async def stop(self):
        logger.info(f"AudioInputTrack.stop() called. Current _stop_event: {self._stop_event.is_set()}")
        await super().stop() # Sets _stop_event
        logger.info(f"AudioInputTrack.stop(): super().stop() called, _stop_event is now: {self._stop_event.is_set()}")

        if self._capture_task is not None:
            logger.info("AudioInputTrack: Attempting to stop and join audio capture task...")
            # The capture thread checks _stop_event, and should close the ring_buffer before exiting.
            # _stop_event is set by super().stop() already.
            try:
                # If the task is not done, it means the capture thread might still be running.
                # It should see _stop_event and call _ring_buffer.close() upon exit.
                # No explicit ring_buffer.close() here, as capture thread owns writing and closing.
                logger.debug(f"AudioInputTrack.stop(): Waiting for capture task {self._capture_task} to complete.")
                await asyncio.wait_for(self._capture_task, timeout=3.0)
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
        
        # Ring buffer doesn't need explicit draining here like the queue did.
        # The capture thread is responsible for closing it.
        # Consumers (recv via _generate_or_capture_frame_data) will get None from read()
        # once the buffer is closed and empty.
        logger.debug(f"AudioInputTrack.stop(): Ring buffer state: size={self._ring_buffer.qsize()}, closed={self._ring_buffer.is_closed()}")
        
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