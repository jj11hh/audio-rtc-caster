import asyncio
import logging
import os
import json
import platform
import argparse # Added for CLI arguments
from aiohttp import web
from aiortc import RTCIceCandidate, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaRelay, MediaStreamTrack
from aiortc.mediastreams import AudioFrame
# import pyaudio
import pyaudiowpatch as pyaudio
import numpy as np
import wave # For debugging audio capture if needed
import math # For sine wave generation

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pc")

ROOT = os.path.dirname(__file__)
CLIENT_DIR = os.path.join(os.path.dirname(ROOT), "client") # ../client

pcs = set()
relay = MediaRelay()

class AudioInputTrack(MediaStreamTrack):
    """
    一个从声卡输入捕获音频的 MediaStreamTrack。
    """
    kind = "audio"

    def __init__(self, p_instance, device_index, sample_rate=48000, channels=2, frames_per_buffer_ms=20):
        super().__init__()
        self.p = p_instance # Use passed PyAudio instance
        self.device_index = device_index
        self.sample_rate = int(sample_rate)
        self.channels = int(channels)
        self.format = pyaudio.paInt16
        self.frames_per_buffer = int(self.sample_rate * frames_per_buffer_ms / 1000)

        self._queue = asyncio.Queue()
        self._stop_event = asyncio.Event()
        self._capture_task = None
        self.stream = None
        self._is_pyaudio_owner = False # Flag to indicate if this instance should terminate PyAudio

        if self.p is None:
            logger.info("PyAudio instance not provided, creating a new one for AudioInputTrack.")
            self.p = pyaudio.PyAudio()
            self._is_pyaudio_owner = True


        if self.device_index is None or self.device_index < 0:
            logger.error("AudioInputTrack: No valid device_index provided.")
            # This state should ideally be prevented by the calling code
            self._stop_event.set() # Prevent recv from starting capture
            return

        try:
            device_info = self.p.get_device_info_by_index(self.device_index)
            logger.info(
                f"AudioInputTrack: Initializing with device '{device_info['name']}' (Index {self.device_index}) "
                f"Rate: {self.sample_rate}, Channels: {self.channels}, Format: paInt16, "
                f"Frames per buffer: {self.frames_per_buffer}"
            )
            # Potentially adjust channels and sample_rate based on device capabilities here
            # For simplicity, we assume the provided ones are supported or PyAudio handles it.
            # device_max_channels = device_info.get('maxInputChannels')
            # if self.channels > device_max_channels:
            #     logger.warning(f"Requested {self.channels} channels, but device '{device_info['name']}' supports max {device_max_channels}. Using {device_max_channels}.")
            #     self.channels = device_max_channels

        except Exception as e:
            logger.error(f"AudioInputTrack: Failed to get device info for index {self.device_index}: {e}")
            self._stop_event.set() # Prevent recv
            return

    def _start_capture_thread(self):
        """This runs in a separate thread to read from PyAudio stream."""
        if self.device_index == -1: # Should be caught by __init__ check
            logger.error("Cannot start capture: No valid audio device.")
            self._stop_event.set() # Signal recv to stop
            return

        try:
            self.stream = self.p.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=self.device_index,
                frames_per_buffer=self.frames_per_buffer,
                stream_callback=None # We'll read manually in a loop
            )
            logger.info("PyAudio stream opened successfully.")
        except Exception as e:
            logger.error(f"Failed to open PyAudio stream: {e}")
            self.stream = None
            self._stop_event.set() # Signal recv to stop
            return

        loop = asyncio.get_event_loop()
        while not self._stop_event.is_set() and self.stream and self.stream.is_active():
            try:
                # Read data from stream
                raw_audio_bytes = self.stream.read(self.frames_per_buffer, exception_on_overflow=False)
                # Put data into asyncio queue to be consumed by recv()
                asyncio.run_coroutine_threadsafe(self._queue.put(raw_audio_bytes), loop)
            except IOError as e: # Such as input overflow
                if e.errno == pyaudio.paInputOverflowed:
                    logger.warning("PyAudio input overflowed. Some audio data may have been lost.")
                else:
                    logger.error(f"PyAudio read error: {e}")
                    self._stop_event.set() # Stop on other IOErrors
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


    async def recv(self):
        if self._stop_event.is_set() and self._queue.empty():
            # If stopping and queue is empty, this iteration is done.
            raise StopAsyncIteration

        if self._capture_task is None and not self._stop_event.is_set():
            if self.device_index == -1: # Should have been caught in __init__
                logger.error("Cannot start capture task, no valid audio device (recv).")
                self._stop_event.set()
                raise StopAsyncIteration

            # Run PyAudio blocking calls in a separate thread
            loop = asyncio.get_event_loop()
            self._capture_task = loop.run_in_executor(None, self._start_capture_thread)
            await asyncio.sleep(0.2) # Give stream time to open, adjust if needed

        try:
            # Wait for data from the queue, with a timeout to check _stop_event
            raw_audio_bytes = await asyncio.wait_for(self._queue.get(), timeout=0.1)
            logger.info(f"AudioInputTrack.recv: Read {len(raw_audio_bytes)} bytes from queue.")
        except asyncio.TimeoutError:
            # If timeout, check if we should stop or just try again
            if self._stop_event.is_set() and self._queue.empty():
                raise StopAsyncIteration
            return await self.recv() # Recurse to try again or raise if stopped

        # Convert bytes to numpy array of int16
        samples_int16 = np.frombuffer(raw_audio_bytes, dtype=np.int16)
        logger.info(f"AudioInputTrack.recv: Converted to {samples_int16.size} int16 samples.")
        if samples_int16.size > 0:
            logger.info(f"AudioInputTrack.recv: Sample values min: {np.min(samples_int16)}, max: {np.max(samples_int16)}, mean: {np.mean(samples_int16)}")


        if samples_int16.size == 0: # Should not happen if stream is good and data is flowing
             logger.warning("AudioInputTrack.recv: samples_int16.size is 0, attempting to get next frame.")
             return await self.recv() # Try to get next frame

        # Reshape based on channels for pyav, expecting (num_channels, num_samples)
        current_layout = "mono" if self.channels == 1 else "stereo"
        if self.channels == 1:
            # For mono, pyav with layout="mono" expects (1, n_samples)
            audio_data_reshaped = samples_int16.reshape(1, -1)
        elif self.channels == 2:
            # For stereo, pyav with layout="stereo" expects (2, n_samples)
            # samples_int16 from PyAudio is interleaved (L R L R...). De-interleave it.
            if samples_int16.ndim == 1: # Ensure it's 1D before de-interleaving
                left_channel = samples_int16[0::2]
                right_channel = samples_int16[1::2]
                audio_data_reshaped = np.array([left_channel, right_channel], dtype=np.int16)
            elif samples_int16.shape[1] == self.channels: # Already (samples, channels)
                 audio_data_reshaped = samples_int16.T # Transpose to (channels, samples)
            else: # Fallback or error
                logger.error(f"AudioInputTrack.recv: Unexpected samples_int16 shape {samples_int16.shape} for {self.channels} channels.")
                return await self.recv() # Or raise error
        else:
            logger.error(f"AudioInputTrack.recv: Unsupported number of channels: {self.channels}")
            return await self.recv() # Or raise error
        
        logger.info(f"AudioInputTrack.recv: Reshaped audio data to {audio_data_reshaped.shape} for layout '{current_layout}'")

        frame = AudioFrame.from_ndarray(
            audio_data_reshaped,
            format='s16', # 's16' is for packed, 's16p' for planar. pyav handles this based on array shape and layout.
            layout=current_layout
        )
        # Explicitly set pts. This assumes each frame is sequential.
        # A more robust PTS might involve a running sample count or time.
        # For now, let's use a simple incrementing PTS based on frames_per_buffer.
        # This might not be perfect but could resolve the NoneType error.
        # Number of samples in this frame: audio_data_reshaped.shape[1]
        # Time duration of this frame: num_samples / sample_rate
        # Let's use number of samples as PTS for simplicity, assuming time_base will be 1/sample_rate
        if not hasattr(self, '_pts_counter'):
            self._pts_counter = 0
        frame.pts = self._pts_counter
        self._pts_counter += audio_data_reshaped.shape[1] # Increment by number of samples in this frame

        frame.sample_rate = self.sample_rate
        logger.info(f"AudioInputTrack.recv: Created AudioFrame - pts: {frame.pts}, time: {frame.time}, samples: {frame.samples}, sample_rate: {frame.sample_rate}, channels: {len(frame.layout.channels) if frame.layout else 'Unknown'}")
        return frame

    async def stop(self):
        self._stop_event.set()
        if self._capture_task is not None:
            logger.info("Stopping audio capture task...")
            try:
                await asyncio.wait_for(self._capture_task, timeout=2.0) 
                logger.info("Audio capture task finished.")
            except asyncio.TimeoutError:
                logger.warning("Audio capture task did not finish in time.")
            except Exception as e:
                logger.error(f"Exception while waiting for audio capture task to stop: {e}")
            self._capture_task = None
        
        # Clean up PyAudio instance only if this class created it
        if self.p and self._is_pyaudio_owner:
            self.p.terminate()
            self.p = None
            logger.info("Owned PyAudio instance terminated by AudioInputTrack.")

# Global audio track
audio_track = None # Will be initialized in main

class SineWaveTrack(MediaStreamTrack):
    """
    A MediaStreamTrack that generates a 440 Hz sine wave.
    """
    kind = "audio"

    def __init__(self, frequency=440, sample_rate=48000, channels=1, samples_per_frame=960):
        super().__init__()
        self.frequency = float(frequency)
        self.sample_rate = int(sample_rate)
        self.channels = int(channels)
        self.samples_per_frame = int(samples_per_frame)  # e.g., 48000 * 20 / 1000 = 960 for 20ms
        self._time = 0.0
        self._stop_event = asyncio.Event()
        self.amplitude = 0.3 * 32767  # Amplitude for 16-bit audio, scaled down

        self._time_increment = 1.0 / self.sample_rate
        logger.info(
            f"SineWaveTrack: Initializing with Freq: {self.frequency}Hz, Rate: {self.sample_rate}Hz, "
            f"Channels: {self.channels}, Samples/Frame: {self.samples_per_frame}"
        )

    async def recv(self):
        if self._stop_event.is_set():
            raise StopAsyncIteration

        num_samples = self.samples_per_frame
        
        # Generate time array for the current frame
        t = np.arange(self._time, self._time + num_samples * self._time_increment, self._time_increment)[:num_samples]
        
        # Generate sine wave
        wave = self.amplitude * np.sin(2 * np.pi * self.frequency * t)
        samples_int16 = wave.astype(np.int16)
        logger.info(f"SineWaveTrack.recv: Generated {samples_int16.size} int16 samples.")
        if samples_int16.size > 0:
            logger.info(f"SineWaveTrack.recv: Sample values min: {np.min(samples_int16)}, max: {np.max(samples_int16)}, mean: {np.mean(samples_int16)}")


        # Prepare frame_data based on channels, expecting (num_channels, num_samples)
        current_layout = "mono" if self.channels == 1 else "stereo"
        if self.channels == 1:
            # For mono, pyav with layout="mono" expects (1, n_samples)
            frame_data = samples_int16.reshape(1, -1)
        elif self.channels == 2:
            # For stereo, pyav with layout="stereo" expects (2, n_samples)
            # SineWaveTrack generates a mono wave; duplicate it for L and R channels.
            left_channel = samples_int16
            right_channel = samples_int16 # Duplicate mono wave for stereo
            frame_data = np.array([left_channel, right_channel], dtype=np.int16)
        else:
            logger.error(f"SineWaveTrack.recv: Unsupported number of channels: {self.channels}")
            # Create a silent mono frame as fallback to prevent crash
            frame_data = np.zeros((1, num_samples), dtype=np.int16)
            current_layout = "mono"

        logger.info(f"SineWaveTrack.recv: Reshaped audio data to {frame_data.shape} for layout '{current_layout}'")
            
        self._time += num_samples * self._time_increment

        frame = AudioFrame.from_ndarray(
            frame_data,
            format='s16', # 's16' is for packed, 's16p' for planar. pyav handles this.
            layout=current_layout
        )
        # Explicitly set pts
        if not hasattr(self, '_pts_counter'):
            self._pts_counter = 0
        frame.pts = self._pts_counter
        self._pts_counter += self.samples_per_frame # samples_per_frame is num_samples for SineWaveTrack

        frame.sample_rate = self.sample_rate
        # logger.info(f"SineWaveTrack.recv: Created AudioFrame - pts: {frame.pts}, time: {frame.time}, samples: {frame.samples}, sample_rate: {frame.sample_rate}, channels: {len(frame.layout.channels) if frame.layout else 'Unknown'}")
        
        # Simulate real-time audio by waiting for the frame duration
        await asyncio.sleep(float(num_samples) / self.sample_rate)
        return frame

    async def stop(self): # Made async
        self._stop_event.set()
        logger.info("SineWaveTrack stop called, event set.")

# --- Helper functions for device selection ---

def find_pyaudiowpatch_loopback_device(p_instance):
    """
    Tries to find the correct WASAPI loopback device using PyAudioWPatch's specific methods.
    Returns a dictionary with 'index', 'name', 'rate', 'channels' if found, else None.
    """
    if platform.system() != "Windows":
        logger.info("PyAudioWPatch loopback detection is specific to Windows WASAPI.")
        return None

    try:
        wasapi_info = p_instance.get_host_api_info_by_type(pyaudio.paWASAPI)
    except OSError:
        logger.warning("WASAPI is not available on the system according to PyAudioWPatch.")
        return None
    except Exception as e:
        logger.error(f"Error getting WASAPI host API info: {e}")
        return None

    try:
        default_speakers = p_instance.get_device_info_by_index(wasapi_info["defaultOutputDevice"])
    except Exception as e:
        logger.error(f"Could not get default WASAPI output device info: {e}")
        return None

    if not default_speakers.get("isLoopbackDevice", False): 
        found_loopback = None
        try:
            for loopback in p_instance.get_loopback_device_info_generator():
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
        logger.warning(f"Selected loopback device '{default_speakers['name']}' has no input channels. This is unexpected for a loopback capture device.")
        return None
        
    return {
        "index": default_speakers["index"],
        "name": default_speakers["name"],
        "rate": int(default_speakers["defaultSampleRate"]),
        "channels": default_speakers["maxInputChannels"] 
    }


def prompt_user_for_device(p_instance):
    """Lists all audio devices (input and output) and prompts user for selection of an INPUT device via TUI."""
    all_devices = []
    input_device_indices = set()
    try:
        for i in range(p_instance.get_device_count()):
            dev_info = p_instance.get_device_info_by_index(i)
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
        if len(name) > 58:
            name = name[:55] + "..."

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
                chosen_device_info = p_instance.get_device_info_by_index(choice)
                logger.info(f"User selected input device: {chosen_device_info['name']} (ID: {choice})")
                return choice
            else:
                is_valid_device_id = any(dev['index'] == choice for dev in all_devices)
                if is_valid_device_id:
                    print(f"Device ID {choice} is an OUTPUT device. Please select an INPUT device ID for recording.")
                else:
                    print("Invalid ID. Please choose an INPUT device ID from the list above.")
        except ValueError:
            print("Invalid input. Please enter a number (ID) or 'c'.")
        except Exception as e:
            logger.error(f"Error during TUI device selection: {e}")
            return None

async def offer(request):
    params = await request.json()
    offer_sdp = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("icecandidate")
    async def on_icecandidate(candidate):
        if candidate:
            pass 

    @pc.on("track")
    async def on_track(track):
        logger.info(f"Track {track.kind} received")

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.info(f"Connection state is {pc.connectionState}")
        if pc.connectionState == "failed" or pc.connectionState == "closed":
            await pc.close()
            pcs.discard(pc)
            # if audio_track and hasattr(audio_track, 'stop_if_no_consumers'): 
            #     await audio_track.stop_if_no_consumers(pcs)


    if audio_track:
        # sender = pc.addTrack(relay.subscribe(audio_track)) # Bypassing MediaRelay for testing
        sender = pc.addTrack(audio_track) # Add track directly
        logger.info(f"Audio track added directly to PC (bypassing MediaRelay). Sender: {sender}, Sender Track: {sender.track}")
        if sender.track:
            logger.info(f"  Directly added track kind: {sender.track.kind}, id: {sender.track.id}")
        # logger.info("Audio track added to PC via MediaRelay.")
    else:
        logger.warning("No audio track available to add to PC.")


    await pc.setRemoteDescription(offer_sdp)
    answer_sdp = await pc.createAnswer()
    await pc.setLocalDescription(answer_sdp)

    logger.info(f"Generated Answer SDP for client:\n{pc.localDescription.sdp}") # Log the generated SDP

    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
        ),
    )

async def serve_client_file(request):
    path = request.match_info.get('filename', 'index.html')
    file_path = os.path.join(CLIENT_DIR, path)
    if not os.path.exists(file_path) or not os.path.isfile(file_path):
        file_path_js = os.path.join(CLIENT_DIR, "js", path)
        if os.path.exists(file_path_js) and os.path.isfile(file_path_js):
            file_path = file_path_js
        else:
            return web.Response(status=404, text="File not found")
    
    if file_path.endswith(".html"):
        content_type = "text/html"
    elif file_path.endswith(".js"):
        content_type = "application/javascript"
    elif file_path.endswith(".css"):
        content_type = "text/css"
    else:
        content_type = "application/octet-stream" 

    try:
        with open(file_path, 'rb') as f:
            return web.Response(body=f.read(), content_type=content_type)
    except Exception as e:
        return web.Response(status=500, text=f"Error reading file: {e}")


async def on_startup(app):
    global audio_track
    if audio_track:
        if isinstance(audio_track, AudioInputTrack):
            try:
                if hasattr(audio_track, 'p') and audio_track.p is not None and \
                   hasattr(audio_track, 'device_index') and \
                   audio_track.device_index is not None and audio_track.device_index >= 0:
                    device_name = audio_track.p.get_device_info_by_index(audio_track.device_index)['name']
                    logger.info(f"Global audio track (AudioInputTrack: '{device_name}') initialized and ready.")
                else:
                    logger.info("Global audio track (AudioInputTrack: device info unavailable due to pre-init or init error) initialized and ready.")
            except Exception as e:
                 logger.error(f"Error getting device name for AudioInputTrack on startup: {e}")
                 logger.info("Global audio track (AudioInputTrack: unknown device due to error during name retrieval) initialized and ready.")
        elif isinstance(audio_track, SineWaveTrack):
            logger.info(f"Global audio track (SineWaveTrack @ {audio_track.frequency}Hz) initialized and ready.")
        else:
            logger.info(f"Global audio track (type: {type(audio_track).__name__}) initialized and ready.")
    else:
        logger.warning("Global audio track is not available or failed to initialize. Audio streaming will not work.")

async def on_shutdown(app):
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()
    if audio_track and hasattr(audio_track, 'stop'):
        logger.info(f"Stopping global audio track ({type(audio_track).__name__}) on shutdown...")
        await audio_track.stop() # Ensure stop is awaited
        logger.info(f"Global audio track ({type(audio_track).__name__}) stopped on shutdown.")


def list_audio_devices_pyaudio(p_instance):
    logger.info("Available PyAudio audio devices:")
    logger.info("---------------------------------")
    for i in range(p_instance.get_device_count()):
        try:
            dev_info = p_instance.get_device_info_by_index(i)
            host_api_info = p_instance.get_host_api_info_by_index(dev_info['hostApi'])
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
        default_input_info = p_instance.get_default_input_device_info()
        logger.info(f"Default Input Device (according to PyAudio): {default_input_info['name']} (Index: {default_input_info['index']})")
    except IOError:
        logger.warning("PyAudio reports: No Default Input Device Available.")
    except Exception as e:
        logger.error(f"Error getting default input device info: {e}")

    if platform.system() == "Windows":
        try:
            default_host_api = p_instance.get_default_host_api_info()
            logger.info(f"Default Host API (Windows): {default_host_api['name']} (Type: {default_host_api['type']})")
            if default_host_api['type'] == pyaudio.paWASAPI:
                logger.info("WASAPI is available. Loopback capture should be possible if a loopback device exists.")
        except Exception as e:
            logger.error(f"Error getting default host API info: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WebRTC audio streaming server.")
    parser.add_argument(
        "--input-device", type=int, default=None, help="Index of the PyAudio input device to use."
    )
    parser.add_argument(
        "--sine-wave", action="store_true", help="Stream a 440Hz sine wave instead of system audio."
    )
    args = parser.parse_args()

    p = None # Initialize PyAudio instance to None; will be created if not using sine wave
    # audio_track is global, declared at module level and assigned here

    if args.sine_wave:
        logger.info("Sine wave mode enabled. Skipping PyAudio initialization and device selection.")
        # For sine wave, use 1 channel, 48kHz, 20ms frames (960 samples)
        audio_track = SineWaveTrack(frequency=440, sample_rate=48000, channels=1, samples_per_frame=int(48000 * 20 / 1000))
        if hasattr(audio_track, '_stop_event') and audio_track._stop_event.is_set(): # Should not happen for SineWaveTrack
            logger.error("SineWaveTrack initialization failed unexpectedly.")
            audio_track = None
    else:
        p = pyaudio.PyAudio() # Create PyAudio instance for actual audio input
        list_audio_devices_pyaudio(p) 

        selected_device_info = None 

        if args.input_device is not None:
            try:
                dev_info = p.get_device_info_by_index(args.input_device)
                if dev_info.get('maxInputChannels', 0) > 0:
                    selected_device_info = {
                        "index": args.input_device,
                        "name": dev_info['name'],
                        "rate": int(dev_info['defaultSampleRate']),
                        "channels": dev_info['maxInputChannels']
                    }
                    logger.info(f"Using input device from command line: {selected_device_info['name']} (Index: {selected_device_info['index']})")
                else:
                    logger.error(f"Device index {args.input_device} specified via CLI is not an input device. Will attempt auto/TUI selection.")
            except Exception as e:
                logger.error(f"Invalid device index {args.input_device} from CLI: {e}. Will attempt auto/TUI selection.")
        
        if selected_device_info is None and platform.system() == "Windows":
            logger.info("Attempting to find PyAudioWPatch loopback device for Windows...")
            loopback_info = find_pyaudiowpatch_loopback_device(p)
            if loopback_info:
                selected_device_info = loopback_info
                logger.info(f"Automatically selected PyAudioWPatch loopback device: {selected_device_info['name']} (Index: {selected_device_info['index']})")
                logger.info(f"  Using device parameters: Rate: {selected_device_info['rate']}, Channels: {selected_device_info['channels']}")

        if selected_device_info is None:
            logger.info("Loopback device not auto-detected or no CLI device. Prompting user for TUI selection...")
            chosen_index = prompt_user_for_device(p)
            if chosen_index is not None:
                try:
                    dev_info = p.get_device_info_by_index(chosen_index)
                    selected_device_info = {
                        "index": chosen_index,
                        "name": dev_info['name'],
                        "rate": int(dev_info['defaultSampleRate']),
                        "channels": dev_info['maxInputChannels']
                    }
                    logger.info(f"Using device from TUI selection: {selected_device_info['name']} (Index: {selected_device_info['index']})")
                    logger.info(f"  Using device parameters: Rate: {selected_device_info['rate']}, Channels: {selected_device_info['channels']}")
                except Exception as e:
                    logger.error(f"Error getting device info for TUI selected index {chosen_index}: {e}")

        if selected_device_info is None:
            logger.warning("No device selected via CLI, auto-detection, or TUI. Attempting to use default input device.")
            try:
                default_input_info_raw = p.get_default_input_device_info()
                selected_device_info = {
                    "index": default_input_info_raw['index'],
                    "name": default_input_info_raw['name'],
                    "rate": int(default_input_info_raw['defaultSampleRate']),
                    "channels": default_input_info_raw['maxInputChannels']
                }
                logger.info(f"Using PyAudio's default input device: {selected_device_info['name']} (Index: {selected_device_info['index']})")
                logger.info(f"  Using device parameters: Rate: {selected_device_info['rate']}, Channels: {selected_device_info['channels']}")
            except IOError:
                logger.error("CRITICAL: No default input device available and no device selected. Audio streaming will not work.")
            except Exception as e:
                logger.error(f"CRITICAL: Error getting default input device: {e}. Audio streaming will not work.")

        if selected_device_info:
            audio_track = AudioInputTrack(
                p_instance=p,
                device_index=selected_device_info["index"],
                sample_rate=selected_device_info["rate"],
                channels=selected_device_info["channels"]
            )
            if audio_track._stop_event.is_set(): # Check if AudioInputTrack itself failed init
                 logger.error("AudioInputTrack initialization failed. Audio streaming disabled.")
                 audio_track = None 
        else:
            logger.error("No valid audio input device configured for AudioInputTrack. Audio streaming will be disabled.")
            audio_track = None
            
    app = web.Application()
    app.on_startup.append(on_startup)
    app.on_shutdown.append(on_shutdown)
    app.router.add_post("/offer", offer)
    app.router.add_get("/", serve_client_file) 
    app.router.add_get("/{filename:.*}", serve_client_file) 

    hostname = platform.node() or "localhost"
    port = 8088

    logger.info(f"Starting server on http://0.0.0.0:{port}")
    logger.info(f"Access client from http://{hostname}:{port} or http://localhost:{port}")
    
    try:
        web.run_app(app, host="0.0.0.0", port=port)
    finally:
        if p: # p will be None if --sine-wave was used
            logger.info("Terminating main PyAudio instance.")
            p.terminate()