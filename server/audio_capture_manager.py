import asyncio
import logging
import numpy as np
import pyaudiowpatch as pyaudio
import threading
import platform # Added for device selection logic

logger = logging.getLogger("audio_capture_manager")
logger_ringbuffer = logging.getLogger("audio_capture_manager.ringbuffer")

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

class AudioCaptureManager:
    def __init__(self, cli_args, frames_per_buffer_ms=20, buffer_capacity_chunks=100):
        self.cli_args = cli_args
        self.p_instance = None
        self.selected_device_info = None
        self.ring_buffer = None
        self.capture_thread = None
        self._stop_event = threading.Event()
        self.frames_per_buffer_ms = frames_per_buffer_ms
        self.buffer_capacity_chunks = buffer_capacity_chunks
        self.stream = None
        self.main_event_loop = None # To be set before starting thread

    def _initialize_pyaudio_and_select_device(self):
        logger.info("AudioCaptureManager: Initializing PyAudio and selecting device...")
        self.p_instance = pyaudio.PyAudio()
        self._list_audio_devices_pyaudio() # For logging

        if self.cli_args.input_device is not None:
            try:
                dev_info = self.p_instance.get_device_info_by_index(self.cli_args.input_device)
                if dev_info.get('maxInputChannels', 0) > 0:
                    self.selected_device_info = {
                        "index": self.cli_args.input_device,
                        "name": dev_info['name'],
                        "rate": int(dev_info['defaultSampleRate']),
                        "channels": dev_info['maxInputChannels'],
                        "format": pyaudio.paInt16 # Assuming paInt16
                    }
                    logger.info(f"AudioCaptureManager: Using input device from command line: {self.selected_device_info['name']} (Index: {self.selected_device_info['index']})")
                else:
                    logger.error(f"AudioCaptureManager: Device index {self.cli_args.input_device} specified via CLI is not an input device. Will attempt auto/TUI selection.")
            except Exception as e:
                logger.error(f"AudioCaptureManager: Invalid device index {self.cli_args.input_device} from CLI: {e}. Will attempt auto/TUI selection.")
        
        if self.selected_device_info is None and platform.system() == "Windows":
            logger.info("AudioCaptureManager: Attempting to find PyAudioWPatch loopback device for Windows...")
            loopback_info = self._find_pyaudiowpatch_loopback_device()
            if loopback_info:
                self.selected_device_info = loopback_info
                logger.info(f"AudioCaptureManager: Automatically selected PyAudioWPatch loopback device: {self.selected_device_info['name']} (Index: {self.selected_device_info['index']})")
                logger.info(f"  Using device parameters: Rate: {self.selected_device_info['rate']}, Channels: {self.selected_device_info['channels']}")

        if self.selected_device_info is None:
            logger.info("AudioCaptureManager: Loopback device not auto-detected or no CLI device. Prompting user for TUI selection...")
            chosen_index = self._prompt_user_for_device()
            if chosen_index is not None:
                try:
                    dev_info = self.p_instance.get_device_info_by_index(chosen_index)
                    self.selected_device_info = {
                        "index": chosen_index,
                        "name": dev_info['name'],
                        "rate": int(dev_info['defaultSampleRate']),
                        "channels": dev_info['maxInputChannels'],
                        "format": pyaudio.paInt16
                    }
                    logger.info(f"AudioCaptureManager: Using device from TUI selection: {self.selected_device_info['name']} (Index: {self.selected_device_info['index']})")
                    logger.info(f"  Using device parameters: Rate: {self.selected_device_info['rate']}, Channels: {self.selected_device_info['channels']}")
                except Exception as e:
                    logger.error(f"AudioCaptureManager: Error getting device info for TUI selected index {chosen_index}: {e}")

        if self.selected_device_info is None:
            logger.warning("AudioCaptureManager: No device selected via CLI, auto-detection, or TUI. Attempting to use default input device.")
            try:
                default_input_info_raw = self.p_instance.get_default_input_device_info()
                self.selected_device_info = {
                    "index": default_input_info_raw['index'],
                    "name": default_input_info_raw['name'],
                    "rate": int(default_input_info_raw['defaultSampleRate']),
                    "channels": default_input_info_raw['maxInputChannels'],
                    "format": pyaudio.paInt16
                }
                logger.info(f"AudioCaptureManager: Using PyAudio's default input device: {self.selected_device_info['name']} (Index: {self.selected_device_info['index']})")
                logger.info(f"  Using device parameters: Rate: {self.selected_device_info['rate']}, Channels: {self.selected_device_info['channels']}")
            except IOError:
                logger.error("AudioCaptureManager: CRITICAL: No default input device available and no device selected. Audio streaming will not work.")
                self._stop_event.set() # Prevent starting capture
            except Exception as e:
                logger.error(f"AudioCaptureManager: CRITICAL: Error getting default input device: {e}. Audio streaming will not work.")
                self._stop_event.set()

        if self.selected_device_info:
            frames_per_buffer = int(self.selected_device_info["rate"] * self.frames_per_buffer_ms / 1000)
            bytes_per_pyaudio_buffer = frames_per_buffer * self.selected_device_info["channels"] * pyaudio.get_sample_size(pyaudio.paInt16)
            self.ring_buffer = NumpyRingBuffer(
                capacity_items=self.buffer_capacity_chunks,
                item_size_bytes=bytes_per_pyaudio_buffer
            )
            self.selected_device_info["frames_per_buffer"] = frames_per_buffer # Store for stream opening
            logger.info(f"AudioCaptureManager: RingBuffer initialized. Capacity: {self.buffer_capacity_chunks} chunks, Item size: {bytes_per_pyaudio_buffer} bytes.")
        else:
            logger.error("AudioCaptureManager: No audio device selected. Cannot initialize RingBuffer or start capture.")
            self._stop_event.set()

    def _find_pyaudiowpatch_loopback_device(self):
        if platform.system() != "Windows":
            logger.info("PyAudioWPatch loopback detection is specific to Windows WASAPI.")
            return None
        try:
            wasapi_info = self.p_instance.get_host_api_info_by_type(pyaudio.paWASAPI)
        except OSError:
            logger.warning("WASAPI is not available on the system according to PyAudioWPatch.")
            return None
        except Exception as e:
            logger.error(f"Error getting WASAPI host API info: {e}")
            return None
        try:
            default_speakers = self.p_instance.get_device_info_by_index(wasapi_info["defaultOutputDevice"])
        except Exception as e:
            logger.error(f"Could not get default WASAPI output device info: {e}")
            return None
        if not default_speakers.get("isLoopbackDevice", False): 
            found_loopback = None
            try:
                for loopback in self.p_instance.get_loopback_device_info_generator():
                    if default_speakers["name"] in loopback["name"]:
                        found_loopback = loopback
                        break
                if found_loopback:
                    default_speakers = found_loopback
                    logger.info(f"Found matching loopback device for '{default_speakers['name']}': ({default_speakers['index']}) {default_speakers['name']}")
                else:
                    logger.warning(f"Could not find a matching loopback device for default speakers '{default_speakers['name']}'. Manual selection may be needed.")
                    return None
            except Exception as e:
                logger.error(f"Error iterating loopback devices: {e}")
                return None 
        else:
            logger.info(f"Default output device IS already a loopback device: ({default_speakers['index']}){default_speakers['name']}")
        if default_speakers.get("maxInputChannels", 0) == 0:
            logger.warning(f"Selected loopback device '{default_speakers['name']}' has no input channels.")
            return None
        return {
            "index": default_speakers["index"],
            "name": default_speakers["name"],
            "rate": int(default_speakers["defaultSampleRate"]),
            "channels": default_speakers["maxInputChannels"],
            "format": pyaudio.paInt16
        }

    def _prompt_user_for_device(self):
        all_devices = []
        input_device_indices = set()
        try:
            for i in range(self.p_instance.get_device_count()):
                dev_info = self.p_instance.get_device_info_by_index(i)
                all_devices.append(dev_info)
                if dev_info.get('maxInputChannels', 0) > 0:
                    input_device_indices.add(dev_info['index'])
        except Exception as e:
            logger.error(f"Error enumerating audio devices for TUI: {e}")
            return None
        if not all_devices:
            logger.warning("No audio devices found by PyAudio.")
            return None
        print("\nAvailable Audio Devices (Input and Output):")
        print("------------------------------------------------------------------------------------")
        print(f"{'ID':<5} {'Name':<60} {'Type':<10} {'InCh':<5} {'OutCh':<5} {'Def SR':<8}")
        print("------------------------------------------------------------------------------------")
        for dev_info in all_devices:
            dev_type = "Input" if dev_info['index'] in input_device_indices else "Output"
            if dev_info['index'] in input_device_indices and dev_info.get('maxOutputChannels', 0) > 0:
                dev_type = "In/Out" 
            name = dev_info['name']
            if len(name) > 58: name = name[:55] + "..."
            print(f"{dev_info['index']:<5} {name:<60} {dev_type:<10} {dev_info.get('maxInputChannels',0):<5} {dev_info.get('maxOutputChannels',0):<5} {int(dev_info.get('defaultSampleRate',0)):<8}")
        print("------------------------------------------------------------------------------------")
        if not input_device_indices:
            logger.warning("No INPUT audio devices found for selection in TUI.")
            return None
        while True:
            try:
                choice_str = input("Enter the ID of the INPUT device you want to use for recording (or 'c' to cancel): ")
                if choice_str.lower() == 'c':
                    logger.info("Device selection cancelled by user.")
                    return None
                choice = int(choice_str)
                if choice in input_device_indices:
                    chosen_device_info = self.p_instance.get_device_info_by_index(choice)
                    logger.info(f"User selected input device: {chosen_device_info['name']} (ID: {choice})")
                    return choice
                else:
                    is_valid_device_id = any(dev['index'] == choice for dev in all_devices)
                    if is_valid_device_id: print(f"Device ID {choice} is an OUTPUT device. Please select an INPUT device ID.")
                    else: print("Invalid ID. Please choose an INPUT device ID from the list above.")
            except ValueError: print("Invalid input. Please enter a number (ID) or 'c'.")
            except Exception as e:
                logger.error(f"Error during TUI device selection: {e}")
                return None

    def _list_audio_devices_pyaudio(self):
        logger.info("AudioCaptureManager: Available PyAudio audio devices:")
        logger.info("---------------------------------")
        for i in range(self.p_instance.get_device_count()):
            try:
                dev_info = self.p_instance.get_device_info_by_index(i)
                host_api_info = self.p_instance.get_host_api_info_by_index(dev_info['hostApi'])
                logger.info(
                    f"  Index {dev_info['index']}: {dev_info['name']}\n"
                    f"    Input Channels: {dev_info['maxInputChannels']}, Output Channels: {dev_info['maxOutputChannels']}\n"
                    f"    Default Sample Rate: {dev_info['defaultSampleRate']}\n"
                    f"    Host API: {host_api_info['name']} (Type: {host_api_info['type']})"
                )
                if platform.system() == "Windows" and host_api_info['type'] == pyaudio.paWASAPI:
                    if "loopback" in dev_info["name"].lower() and dev_info["maxInputChannels"] > 0:
                        logger.info(f"    ^ Potential WASAPI loopback device for capture.")
            except Exception as e:
                logger.warning(f"Could not get full info for device index {i}: {e}")
        logger.info("---------------------------------")
        try:
            default_input_info = self.p_instance.get_default_input_device_info()
            logger.info(f"Default Input Device (according to PyAudio): {default_input_info['name']} (Index: {default_input_info['index']})")
        except IOError: logger.warning("PyAudio reports: No Default Input Device Available.")
        except Exception as e: logger.error(f"Error getting default input device info: {e}")
        if platform.system() == "Windows":
            try:
                default_host_api = self.p_instance.get_default_host_api_info()
                logger.info(f"Default Host API (Windows): {default_host_api['name']} (Type: {default_host_api['type']})")
                if default_host_api['type'] == pyaudio.paWASAPI:
                    logger.info("WASAPI is available. Loopback capture should be possible if a loopback device exists.")
            except Exception as e: logger.error(f"Error getting default host API info: {e}")

    def _capture_loop(self):
        logger.info(f"AudioCaptureManager: Capture thread started for device {self.selected_device_info['name']}.")
        if not self.selected_device_info or self._stop_event.is_set():
            logger.error("AudioCaptureManager: Cannot start capture, no device selected or stop event set.")
            return

        try:
            self.stream = self.p_instance.open(
                format=self.selected_device_info["format"],
                channels=self.selected_device_info["channels"],
                rate=self.selected_device_info["rate"],
                input=True,
                input_device_index=self.selected_device_info["index"],
                frames_per_buffer=self.selected_device_info["frames_per_buffer"],
                stream_callback=None 
            )
            logger.info(f"AudioCaptureManager: PyAudio stream opened successfully for device {self.selected_device_info['index']}.")
        except Exception as e:
            logger.error(f"AudioCaptureManager: Failed to open PyAudio stream: {e}")
            self.stream = None
            self._stop_event.set() # Signal issues
            self.ring_buffer.close() # Close buffer as no data will come
            return

        while not self._stop_event.is_set() and self.stream and self.stream.is_active():
            try:
                raw_audio_bytes = self.stream.read(self.selected_device_info["frames_per_buffer"], exception_on_overflow=False)
                bytes_read = len(raw_audio_bytes)
                if bytes_read == 0:
                    logger.warning("AudioCaptureManager: Read 0 bytes from stream.")
                    continue
                if not self.ring_buffer.write(raw_audio_bytes):
                    logger.error("AudioCaptureManager: Failed to write to ring_buffer (closed or size mismatch). Stopping capture.")
                    self._stop_event.set()
                    break
            except IOError as e:
                if hasattr(pyaudio, 'paInputOverflowed') and e.errno == pyaudio.paInputOverflowed:
                    logger.warning(f"AudioCaptureManager: PyAudio input overflowed. {e}")
                else:
                    logger.error(f"AudioCaptureManager: PyAudio IOError in read loop: {e}")
                    self._stop_event.set()
                    break
            except Exception as e:
                logger.error(f"AudioCaptureManager: Unexpected exception in PyAudio capture loop: {e}", exc_info=True)
                self._stop_event.set()
                break
        
        logger.info("AudioCaptureManager: Capture loop exited.")
        if self.stream:
            try:
                if self.stream.is_active(): self.stream.stop_stream()
                self.stream.close()
                logger.info("AudioCaptureManager: PyAudio stream stopped and closed.")
            except Exception as e:
                logger.error(f"AudioCaptureManager: Error closing PyAudio stream: {e}")
        self.stream = None
        self.ring_buffer.close() # Ensure buffer is closed on exit
        logger.info("AudioCaptureManager: RingBuffer closed.")


    def start_capture(self, loop: asyncio.AbstractEventLoop):
        self.main_event_loop = loop
        self._initialize_pyaudio_and_select_device()
        if self._stop_event.is_set() or not self.selected_device_info or not self.ring_buffer:
            logger.error("AudioCaptureManager: Pre-conditions not met. Cannot start capture thread.")
            if self.p_instance: # Terminate if initialized but not proceeding
                self.p_instance.terminate()
                self.p_instance = None
            return False

        logger.info("AudioCaptureManager: Starting audio capture thread...")
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        logger.info("AudioCaptureManager: Audio capture thread started.")
        return True

    def stop_capture(self):
        logger.info("AudioCaptureManager: Stopping audio capture...")
        self._stop_event.set()
        if self.capture_thread and self.capture_thread.is_alive():
            logger.info("AudioCaptureManager: Waiting for capture thread to join...")
            self.capture_thread.join(timeout=5.0) # Wait for thread to finish
            if self.capture_thread.is_alive():
                logger.warning("AudioCaptureManager: Capture thread did not join in time.")
        else:
            logger.info("AudioCaptureManager: Capture thread was not running or already finished.")
        
        # Ring buffer should be closed by the capture_loop itself upon stopping
        # but as a safeguard, ensure it's closed if the thread didn't do it.
        if self.ring_buffer and not self.ring_buffer.is_closed():
            logger.warning("AudioCaptureManager: RingBuffer was not closed by capture thread, closing now.")
            self.ring_buffer.close()

        if self.p_instance:
            logger.info("AudioCaptureManager: Terminating PyAudio instance.")
            self.p_instance.terminate()
            self.p_instance = None
        logger.info("AudioCaptureManager: Audio capture stopped and resources released.")

    def get_ring_buffer(self):
        return self.ring_buffer

    def get_selected_device_params(self):
        if self.selected_device_info:
            return {
                "sample_rate": self.selected_device_info["rate"],
                "channels": self.selected_device_info["channels"] # This is device_actual_channels
            }
        return None