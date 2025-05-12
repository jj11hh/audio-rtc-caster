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
        if self.device_index == -1:
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
                # Ensure this is thread-safe
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
            raise StopAsyncIteration

        if self._capture_task is None and not self._stop_event.is_set():
            if self.device_index == -1: # No device, can't start
                logger.error("Cannot start capture task, no valid audio device.")
                self._stop_event.set()
                raise StopAsyncIteration

            # Run PyAudio blocking calls in a separate thread
            loop = asyncio.get_event_loop()
            self._capture_task = loop.run_in_executor(None, self._start_capture_thread)
            await asyncio.sleep(0.2) # Give stream time to open

        try:
            # Wait for data from the queue, with a timeout to check _stop_event
            raw_audio_bytes = await asyncio.wait_for(self._queue.get(), timeout=0.1)
        except asyncio.TimeoutError:
            if self._stop_event.is_set() and self._queue.empty():
                raise StopAsyncIteration
            return await self.recv() # Recurse to try again or raise if stopped

        # Convert bytes to numpy array of int16
        samples_int16 = np.frombuffer(raw_audio_bytes, dtype=np.int16)

        # Create AudioFrame
        # aiortc's AudioFrame.from_ndarray expects a 2D array (samples, channels)
        # or a 1D array if format and layout are specified correctly.
        # Our samples_int16 is 1D, interleaved.
        if samples_int16.size == 0: # Should not happen if stream is good
             return await self.recv()


        frame = AudioFrame.from_ndarray(
            samples_int16.reshape(-1, self.channels), format='s16', layout='interleaved'
        )
        frame.sample_rate = self.sample_rate
        # frame.channels and frame.samples are derived by from_ndarray
        return frame

    async def stop(self):
        self._stop_event.set()
        if self._capture_task is not None:
            logger.info("Stopping audio capture task...")
            # The capture thread will see _stop_event and exit its loop, closing the stream.
            try:
                await asyncio.wait_for(self._capture_task, timeout=2.0) # Wait for thread to finish
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
audio_track = None # Will be initialized in main after device selection

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

    if not default_speakers.get("isLoopbackDevice", False): # Check if 'isLoopbackDevice' key exists
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
            return None # Could not find it
    else:
        logger.info(f"Default output device IS already a loopback device: ({default_speakers['index']}){default_speakers['name']}")


    # Ensure the device has input channels, as we need to record from it.
    if default_speakers.get("maxInputChannels", 0) == 0:
        logger.warning(f"Selected loopback device '{default_speakers['name']}' has no input channels. This is unexpected for a loopback capture device.")
        return None
        
    return {
        "index": default_speakers["index"],
        "name": default_speakers["name"],
        "rate": int(default_speakers["defaultSampleRate"]),
        "channels": default_speakers["maxInputChannels"] # Use maxInputChannels for loopback
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
            dev_type = "In/Out" # Could be a full-duplex device
        
        # Sanitize name to prevent very long lines, though usually not an issue with PyAudio names
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
                # Check if it's a valid device ID at all, but not an input device
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
            # 发送 ICE candidate 给客户端
            # 实际应用中，这应该通过信令服务器发送
            # 这里我们简化，假设客户端能处理直接的 candidate 信息
            # 或者，更常见的是，在 answer 中收集所有 candidates
            pass # Candidates will be included in the answer

    @pc.on("track")
    async def on_track(track):
        logger.info(f"Track {track.kind} received")
        # 我们是发送方，不应该接收轨道
        # 如果需要双向，可以在这里处理

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.info(f"Connection state is {pc.connectionState}")
        if pc.connectionState == "failed" or pc.connectionState == "closed":
            await pc.close()
            pcs.discard(pc)
            if audio_track and hasattr(audio_track, 'stop_if_no_consumers'): # 假设我们添加了这个方法
                await audio_track.stop_if_no_consumers(pcs)


    if audio_track:
        pc.addTrack(relay.subscribe(audio_track))
        logger.info("Audio track added to PC.")
    else:
        logger.warning("No audio track available to add to PC.")


    await pc.setRemoteDescription(offer_sdp)
    answer_sdp = await pc.createAnswer()
    await pc.setLocalDescription(answer_sdp)

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
        # 尝试 js/filename
        file_path_js = os.path.join(CLIENT_DIR, "js", path)
        if os.path.exists(file_path_js) and os.path.isfile(file_path_js):
            file_path = file_path_js
        else:
            return web.Response(status=404, text="File not found")
    
    # 确定MIME类型
    if file_path.endswith(".html"):
        content_type = "text/html"
    elif file_path.endswith(".js"):
        content_type = "application/javascript"
    elif file_path.endswith(".css"):
        content_type = "text/css"
    else:
        content_type = "application/octet-stream" # 默认

    try:
        with open(file_path, 'rb') as f:
            return web.Response(body=f.read(), content_type=content_type)
    except Exception as e:
        return web.Response(status=500, text=f"Error reading file: {e}")


async def on_startup(app):
    global audio_track
    if audio_track:
        logger.info(f"Global audio track '{audio_track.p.get_device_info_by_index(audio_track.device_index)['name']}' initialized and ready.")
    else:
        logger.warning("Global audio track is not available or failed to initialize. Audio streaming will not work.")

async def on_shutdown(app):
    # 关闭所有 WebRTC 连接
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()
    # 停止音频捕获
    if audio_track and hasattr(audio_track, 'stop'):
        await audio_track.stop()
        logger.info("Global audio track capture stopped on shutdown.")


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
    args = parser.parse_args()

    p = pyaudio.PyAudio() # Create one PyAudio instance for the application

    list_audio_devices_pyaudio(p) # List all devices for user info

    selected_device_info = None # Will store dict: {'index', 'name', 'rate', 'channels'}

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
        logger.info("No CLI device specified or CLI device was invalid, attempting to find PyAudioWPatch loopback device for Windows...")
        loopback_info = find_pyaudiowpatch_loopback_device(p)
        if loopback_info:
            selected_device_info = loopback_info # Contains index, name, rate, channels
            logger.info(f"Automatically selected PyAudioWPatch loopback device: {selected_device_info['name']} (Index: {selected_device_info['index']})")
            logger.info(f"  Using device parameters: Rate: {selected_device_info['rate']}, Channels: {selected_device_info['channels']}")

    if selected_device_info is None:
        logger.info("Loopback device not found automatically or no CLI device specified (or not on Windows for auto PyAudioWPatch). Prompting user for TUI selection...")
        chosen_index = prompt_user_for_device(p)
        if chosen_index is not None:
            try:
                dev_info = p.get_device_info_by_index(chosen_index)
                selected_device_info = {
                    "index": chosen_index,
                    "name": dev_info['name'],
                    "rate": int(dev_info['defaultSampleRate']), # Use device's default rate
                    "channels": dev_info['maxInputChannels']   # Use device's max input channels
                }
                logger.info(f"Using device from TUI selection: {selected_device_info['name']} (Index: {selected_device_info['index']})")
                logger.info(f"  Using device parameters: Rate: {selected_device_info['rate']}, Channels: {selected_device_info['channels']}")
            except Exception as e:
                logger.error(f"Error getting device info for TUI selected index {chosen_index}: {e}")

    if selected_device_info is None:
        logger.warning("No device selected via CLI, auto-detection, or TUI. Attempting to use default input device as a last resort.")
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
        if audio_track._stop_event.is_set():
             logger.error("AudioInputTrack initialization failed. Audio streaming disabled.")
             audio_track = None
    else:
        logger.error("No valid audio input device configured. Audio streaming will be disabled.")
        audio_track = None
    
    app = web.Application()
    app.on_startup.append(on_startup)
    app.on_shutdown.append(on_shutdown)
    app.router.add_post("/offer", offer)
    app.router.add_get("/", serve_client_file) # Serve index.html at root
    app.router.add_get("/{filename:.*}", serve_client_file) # Serve other files like main.js

    # 获取主机名，用于提示
    hostname = platform.node() or "localhost"
    port = 8088

    logger.info(f"Starting server on http://0.0.0.0:{port}")
    logger.info(f"Access client from http://{hostname}:{port} or http://localhost:{port}")
    
    try:
        web.run_app(app, host="0.0.0.0", port=port)
    finally:
        if p: # Ensure PyAudio is terminated if app crashes or exits
            logger.info("Terminating main PyAudio instance.")
            p.terminate()