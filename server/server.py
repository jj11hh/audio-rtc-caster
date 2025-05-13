import asyncio
import logging
import os
import json
import platform
import argparse # Added for CLI arguments
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaRelay
# import pyaudio
import pyaudiowpatch as pyaudio
from audio_tracks import AudioInputTrack, SineWaveTrack # Import new classes

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("pc")

ROOT = os.path.dirname(__file__)
CLIENT_DIR = os.path.join(os.path.dirname(ROOT), "client") # ../client

pcs = set()
relay = MediaRelay()

# Global variable to store command line arguments
cli_args = None

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

    if audio_track:
        sender = pc.addTrack(audio_track)
        logger.info(f"Audio track added directly to PC. Sender: {sender}, Sender Track: {sender.track}")
        if sender.track:
            logger.info(f"  Directly added track kind: {sender.track.kind}, id: {sender.track.id}")
    else:
        logger.warning("No audio track available to add to PC.")

    await pc.setRemoteDescription(offer_sdp)
    answer_sdp = await pc.createAnswer()

    logger.info("Original Answer SDP:\n%s", answer_sdp.sdp)
    sdp_lines = answer_sdp.sdp.strip().split('\r\n')

    # --- BEGIN SDP Modification based on CLI arguments ---
    codec_payload_type = None
    rtpmap_line_index = -1
    fmtp_line_index = -1
    media_line_index = -1 # For 'm=audio ...'

    # Find m=audio line, preferred codec's payload type, and its rtpmap/fmtp line indices
    for i, line in enumerate(sdp_lines):
        if line.startswith("m=audio"):
            media_line_index = i
        
        if line.startswith("a=rtpmap:"):
            try:
                parts = line.split(" ", 1)
                pt_part = parts[0].split(":")[1]
                codec_part = parts[1]
                codec_name_in_rtpmap = codec_part.split("/")[0].lower()

                if codec_name_in_rtpmap == cli_args.preferred_codec.lower():
                    if codec_payload_type is None: # Take the first match
                        codec_payload_type = pt_part
                        rtpmap_line_index = i
                        logger.info(f"Found rtpmap for preferred codec '{cli_args.preferred_codec}': PT={codec_payload_type}, line='{line}'")
                        # Search for corresponding fmtp line
                        for j, fmtp_candidate_line in enumerate(sdp_lines):
                            if fmtp_candidate_line.startswith(f"a=fmtp:{codec_payload_type}"):
                                fmtp_line_index = j
                                logger.info(f"Found existing fmtp line for PT {codec_payload_type} at index {j}: {fmtp_candidate_line}")
                                break
                        break # Found preferred codec rtpmap, stop main loop
            except Exception as e:
                logger.warning(f"Could not parse rtpmap line: '{line}'. Error: {e}")

    new_fmtp_params_list = []
    if cli_args.preferred_codec.lower() == "opus":
        new_fmtp_params_list.append("stereo=1") # Default stereo for Opus
        new_fmtp_params_list.append("sprop-stereo=1")
        if cli_args.opus_maxaveragebitrate:
            new_fmtp_params_list.append(f"maxaveragebitrate={cli_args.opus_maxaveragebitrate}")
        if cli_args.opus_maxplaybackrate:
            new_fmtp_params_list.append(f"maxplaybackrate={cli_args.opus_maxplaybackrate}")
        if cli_args.opus_cbr:
            new_fmtp_params_list.append("cbr=1")
        if cli_args.opus_useinbandfec:
            new_fmtp_params_list.append("useinbandfec=1")
        if cli_args.opus_usedtx:
            new_fmtp_params_list.append("usedtx=1")

    if codec_payload_type:
        if fmtp_line_index != -1: # Existing fmtp line found
            base_fmtp_line_parts = sdp_lines[fmtp_line_index].split(' ', 1)
            existing_params_str = base_fmtp_line_parts[1] if len(base_fmtp_line_parts) > 1 else ""
            
            current_params_dict = {}
            if existing_params_str:
                for p_item in existing_params_str.split(';'):
                    key_val = p_item.strip().split('=', 1)
                    if len(key_val) == 2:
                        current_params_dict[key_val[0]] = key_val[1]
                    elif len(key_val) == 1 and key_val[0]:
                        current_params_dict[key_val[0]] = True
            
            for p_item_str in new_fmtp_params_list:
                key_val = p_item_str.split('=', 1)
                if len(key_val) == 2:
                    current_params_dict[key_val[0]] = key_val[1]
                elif len(key_val) == 1 and key_val[0]:
                     current_params_dict[key_val[0]] = True

            final_fmtp_parts = []
            for k, v in current_params_dict.items():
                if isinstance(v, bool) and v is True: final_fmtp_parts.append(k)
                else: final_fmtp_parts.append(f"{k}={v}")
            
            if final_fmtp_parts:
                sdp_lines[fmtp_line_index] = f"a=fmtp:{codec_payload_type} {';'.join(final_fmtp_parts)}"
                logger.info(f"Updated fmtp line {fmtp_line_index} for PT {codec_payload_type}: {sdp_lines[fmtp_line_index]}")
            else:
                logger.info(f"No parameters to set for existing fmtp line {fmtp_line_index} for PT {codec_payload_type}. Line remains: {sdp_lines[fmtp_line_index]}")

        elif rtpmap_line_index != -1 and new_fmtp_params_list: # No existing fmtp, but rtpmap found and params to add
            new_fmtp_line = f"a=fmtp:{codec_payload_type} {';'.join(new_fmtp_params_list)}"
            insert_after_idx = rtpmap_line_index
            for k_idx in range(rtpmap_line_index + 1, len(sdp_lines)):
                line_k = sdp_lines[k_idx]
                if line_k.startswith(f"a=rtcp-fb:{codec_payload_type}") or line_k.startswith(f"a=fmtp:{codec_payload_type}"):
                    insert_after_idx = k_idx
                elif line_k.startswith("a=rtpmap:") or line_k.startswith("m="): break
            sdp_lines.insert(insert_after_idx + 1, new_fmtp_line)
            logger.info(f"Inserted new fmtp line for PT {codec_payload_type} after line {insert_after_idx}: {new_fmtp_line}")
        elif not new_fmtp_params_list:
             logger.info(f"No new fmtp parameters specified via CLI for codec '{cli_args.preferred_codec}'.")
    else:
        logger.warning(f"Preferred codec '{cli_args.preferred_codec}' (rtpmap) not found. Cannot apply fmtp modifications.")

    if cli_args.audio_bitrate and media_line_index != -1:
        bitrate_kbps = cli_args.audio_bitrate // 1000
        bandwidth_line = f"b=AS:{bitrate_kbps}"
        b_line_exists_index = -1
        next_m_line_index = len(sdp_lines)
        for i in range(media_line_index + 1, len(sdp_lines)):
            if sdp_lines[i].startswith("m="): next_m_line_index = i; break
            if sdp_lines[i].startswith("b=AS:"): b_line_exists_index = i; break
        
        if b_line_exists_index != -1:
            sdp_lines[b_line_exists_index] = bandwidth_line
            logger.info(f"Updated bandwidth line at index {b_line_exists_index}: {bandwidth_line}")
        else:
            insert_b_at_index = media_line_index + 1
            if insert_b_at_index < next_m_line_index and sdp_lines[insert_b_at_index].startswith("i="): insert_b_at_index +=1
            if insert_b_at_index < next_m_line_index and sdp_lines[insert_b_at_index].startswith("c="): insert_b_at_index += 1
            if insert_b_at_index >= next_m_line_index: insert_b_at_index = media_line_index + 1
            sdp_lines.insert(insert_b_at_index, bandwidth_line)
            logger.info(f"Inserted bandwidth line at index {insert_b_at_index}: {bandwidth_line}")

    modified_sdp = "\r\n".join(sdp_lines) + "\r\n"
    answer_sdp = RTCSessionDescription(sdp=modified_sdp, type=answer_sdp.type)
    logger.info(f"Final Modified Answer SDP with CLI params:\n%s", modified_sdp)
    # --- END SDP Modification ---
    
    await pc.setLocalDescription(answer_sdp)

    # Log the final SDP being sent
    logger.info(f"Final Answer SDP sent to client:\n{pc.localDescription.sdp}")

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
        audio_track.stop()
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
    parser.add_argument(
        "--preferred-codec", type=str, default="opus", choices=["opus", "pcmu", "pcma", "g722"],
        help="Preferred audio codec. Server will try to use this if offered by client."
    )
    parser.add_argument(
        "--audio-bitrate", type=int, default=96000,
        help="Target audio bitrate in bps (e.g., 32000). Sets 'b=AS:' line in SDP (converted to kbps)."
    )
    parser.add_argument(
        "--opus-maxaveragebitrate", type=int, default=None,
        help="Opus specific: Sets 'maxaveragebitrate' in bps (e.g., 20000)."
    )
    parser.add_argument(
        "--opus-maxplaybackrate", type=int, default=48000, choices=[8000, 12000, 16000, 24000, 48000],
        help="Opus specific: Sets 'maxplaybackrate' (e.g., 48000 for fullband)."
    )
    parser.add_argument(
        "--opus-cbr", action="store_true",
        help="Opus specific: Enable constant bitrate ('cbr=1'). May affect other bitrate settings."
    )
    parser.add_argument(
        "--opus-useinbandfec", action="store_true", default=True,
        help="Opus specific: Enable inband Forward Error Correction ('useinbandfec=1')."
    )
    parser.add_argument(
        "--opus-usedtx", action="store_true", default=True,
        help="Opus specific: Enable Discontinuous Transmission ('usedtx=1')."
    )
    
    cli_args = parser.parse_args() # Assign to the global cli_args
    args = cli_args # Keep 'args' for local use in main if preferred, or replace 'args' with 'cli_args' below

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
            # Log the intended channel count for the created track
            logger.info(f"AudioInputTrack created for device '{selected_device_info['name']}' (Index: {selected_device_info['index']}) with {selected_device_info['channels']} channel(s) at {selected_device_info['rate']} Hz.")
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