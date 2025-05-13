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
from audio_tracks import AudioInputTrack, SineWaveTrack
from audio_capture_manager import AudioCaptureManager # Import new manager

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
audio_capture_manager = None # Global instance for the capture manager
audio_track = None # Global audio track (either SineWaveTrack or AudioInputTrack)

# --- Device selection and PyAudio management are now handled by AudioCaptureManager ---
# Functions find_pyaudiowpatch_loopback_device, prompt_user_for_device,
# and list_audio_devices_pyaudio will be removed or are internal to AudioCaptureManager.

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
    global audio_track, audio_capture_manager # Ensure audio_capture_manager is accessible
    
    # Start AudioCaptureManager here if not using sine wave
    # This ensures it's started within the asyncio loop that web.run_app will use/manage
    if not cli_args.sine_wave and audio_capture_manager:
        logger.info("on_startup: Starting AudioCaptureManager...")
        # The loop argument for start_capture might be useful if it needs to schedule
        # tasks back on the main loop, but current AudioCaptureManager uses a separate thread.
        if audio_capture_manager.start_capture(asyncio.get_event_loop()):
            logger.info("on_startup: AudioCaptureManager started successfully.")
            device_params = audio_capture_manager.get_selected_device_params()
            ring_buffer = audio_capture_manager.get_ring_buffer()

            if device_params and ring_buffer:
                # Determine output channels for the track (e.g., stereo)
                # This could be made configurable later if needed.
                output_track_channels = 2 # Default to stereo output for the track
                if device_params['channels'] == 1: # If source is mono, track can also be mono
                    output_track_channels = 1
                
                audio_track = AudioInputTrack(
                    ring_buffer=ring_buffer,
                    sample_rate=device_params['sample_rate'],
                    channels=output_track_channels, # Output channels for WebRTC
                    device_actual_channels=device_params['channels'] # Actual channels from device
                )
                logger.info(f"AudioInputTrack created in on_startup for device '{device_params.get('name', 'Unknown')}' "
                            f"Rate: {device_params['sample_rate']}, DeviceCh: {device_params['channels']}, TrackCh: {output_track_channels}")
                if audio_track._stop_event.is_set():
                    logger.error("AudioInputTrack initialization failed in on_startup.")
                    audio_track = None
            else:
                logger.error("on_startup: Failed to get device parameters or ring buffer from AudioCaptureManager. AudioInputTrack not created.")
                audio_track = None
        else:
            logger.error("on_startup: AudioCaptureManager failed to start. No audio input will be available.")
            audio_track = None

    # Logging for the global audio_track status
    if audio_track:
        if isinstance(audio_track, AudioInputTrack):
            # AudioInputTrack no longer directly holds device name or PyAudio instance details.
            # We can log its configured parameters.
            logger.info(f"Global audio track (AudioInputTrack) initialized: "
                        f"Sample Rate: {audio_track.sample_rate}, "
                        f"Track Channels: {audio_track.channels}, "
                        f"Device Actual Channels: {audio_track.device_actual_channels}. Ready.")
        elif isinstance(audio_track, SineWaveTrack):
            logger.info(f"Global audio track (SineWaveTrack @ {audio_track.frequency}Hz) initialized and ready.")
        else:
            logger.info(f"Global audio track (type: {type(audio_track).__name__}) initialized and ready.")
    else:
        logger.warning("Global audio track is not available or failed to initialize after on_startup. Audio streaming might not work.")


async def on_shutdown(app):
    global audio_track, audio_capture_manager
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()

    if audio_track and hasattr(audio_track, 'stop'):
        logger.info(f"Stopping global audio track ({type(audio_track).__name__}) on shutdown...")
        audio_track.stop() # This now mainly sets an event for the track's recv loop
        logger.info(f"Global audio track ({type(audio_track).__name__}) stop signaled.")

    if audio_capture_manager:
        logger.info("Shutting down AudioCaptureManager...")
        audio_capture_manager.stop_capture()
        logger.info("AudioCaptureManager shut down.")


# list_audio_devices_pyaudio is removed as its functionality is in AudioCaptureManager

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WebRTC audio streaming server.")
    parser.add_argument(
        "--input-device", type=int, default=None, help="Index of the PyAudio input device to use (via AudioCaptureManager)."
    )
    parser.add_argument(
        "--sine-wave", action="store_true", help="Stream a 440Hz sine wave instead of system audio."
    )
    # ... (other argparse arguments remain the same) ...
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
    
    cli_args = parser.parse_args()
    # args = cli_args # No longer need 'args' alias if cli_args is used consistently

    # audio_track is global, declared at module level and assigned here or in on_startup
    # p (PyAudio instance) is no longer managed here globally.

    if cli_args.sine_wave:
        logger.info("Sine wave mode enabled. AudioCaptureManager will not be used.")
        audio_track = SineWaveTrack(frequency=440, sample_rate=48000, channels=1, samples_per_frame=int(48000 * 0.02))
        if hasattr(audio_track, '_stop_event') and audio_track._stop_event.is_set():
            logger.error("SineWaveTrack initialization failed unexpectedly.")
            audio_track = None
    else:
        logger.info("Real audio input mode. AudioCaptureManager will be initialized and started in on_startup.")
        # Instantiate AudioCaptureManager here, but start it in on_startup
        audio_capture_manager = AudioCaptureManager(cli_args=cli_args)
        # AudioInputTrack will be created in on_startup after manager has selected a device and started capture.
        audio_track = None # Placeholder, will be set in on_startup

            
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
        # PyAudio instance 'p' is no longer managed here.
        # AudioCaptureManager handles its own PyAudio instance termination in its stop_capture method,
        # which is called in on_shutdown.
        logger.info("Server shutdown sequence initiated in finally block (if applicable).")