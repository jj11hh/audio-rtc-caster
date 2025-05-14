# WebRTC Audio Streaming Project

This project demonstrates a WebRTC audio streaming setup where a Python server captures audio (either from a system input device via an `AudioCaptureManager` or a generated sine wave) and streams it to a web client, which then plays the audio and displays loudness meters.

## Project Structure

```
.
├── requirements.txt        # Python dependencies
├── client/
│   ├── index.html          # Main HTML page for the client
│   └── js/
│       └── main.js         # Client-side JavaScript for WebRTC connection, audio playback, and meters
└── server/
    ├── server.py           # Python server using aiohttp and aiortc for WebRTC signaling and streaming
    ├── audio_tracks.py     # Python classes for audio tracks (input from buffer, sine wave)
    └── audio_capture_manager.py # Handles audio capture
```

## Features

-   **Python WebRTC Server (`server/server.py`)**:
    -   Uses `aiohttp` for the web server (signaling and serving client files).
    -   Uses `aiortc` for WebRTC peer connection and media handling.
    -   **Audio Input Management (via `AudioCaptureManager`)**:
        -   Captures audio from a selected system input device. Likely uses `pyaudio` and potentially `pyaudiowpatch` (for Windows WASAPI loopback).
        -   Device selection is handled by `AudioCaptureManager`, which might include TUI or automatic selection logic.
    -   **Alternative Sine Wave Source**: Option to stream a generated 440Hz sine wave for testing (`--sine-wave` argument).
    -   **SDP Customization**:
        -   Modifies SDP answers based on command-line arguments to prefer specific codecs (Opus, PCMU, PCMA, G722).
        -   Allows fine-tuning of Opus parameters (e.g., `maxaveragebitrate`, `maxplaybackrate`, `cbr`, `useinbandfec`, `usedtx`).
        -   Sets audio bitrate (`b=AS:` line) in the SDP.
    -   Serves the HTML/JS client.
    -   Logs to `app.log` and console.

-   **Web Client (`client/js/main.js`, `client/index.html`)**:
    -   Connects to the Python server via WebRTC.
    -   Receives the audio stream and plays it using the HTML `<audio>` element.
    -   **Audio Loudness Meters**: Displays real-time left and right channel audio loudness using the Web Audio API and SVG.
    -   **SDP Offer Modification**: Modifies its SDP offer to explicitly request stereo for the Opus codec.
    -   Displays connection status and attempts automatic reconnection on disconnections or failures.

## Setup and Installation

1.  **Clone the repository (if applicable).**
2.  **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Ensure you have an audio input device configured** if you are not using the sine wave option.

## Running the Application

1.  **Start the Python Server:**
    Navigate to the project root or `server/` directory:
    ```bash
    python server/server.py [OPTIONS]
    ```
    Or, if you are in the `server` directory:
    ```bash
    python server.py [OPTIONS]
    ```

    **Command-line arguments for `server.py`:**
    *   `--input-device <INDEX>`: Index of the PyAudio input device to use (via `AudioCaptureManager`).
    *   `--sine-wave`: Stream a 440Hz sine wave instead of system audio.
    *   `--preferred-codec <CODEC>`: Preferred audio codec. Choices: `opus`, `pcmu`, `pcma`, `g722`. Default: `opus`.
    *   `--audio-bitrate <BPS>`: Target audio bitrate in bps (e.g., 32000, 96000). Default: `96000`.
    *   `--opus-maxaveragebitrate <BPS>`: Opus specific: Sets 'maxaveragebitrate' in bps.
    *   `--opus-maxplaybackrate <HZ>`: Opus specific: Sets 'maxplaybackrate' (e.g., 8000, 12000, 16000, 24000, 48000). Default: `48000`.
    *   `--opus-cbr`: Opus specific: Enable constant bitrate ('cbr=1').
    *   `--opus-useinbandfec`: Opus specific: Enable inband Forward Error Correction ('useinbandfec=1'). Default: enabled.
    *   `--opus-usedtx`: Opus specific: Enable Discontinuous Transmission ('usedtx=1'). Default: enabled.

    If input device/mode is not specified, `AudioCaptureManager` will likely handle device selection (e.g., by listing devices or attempting auto-detection).

    The server will log its status and the URL to access the client (usually `http://localhost:8088` or `http://<your-hostname>:8088`).

2.  **Open the Client in a Web Browser:**
    Open the URL provided by the server in a WebRTC-compatible browser (e.g., Chrome, Firefox). This is typically `http://localhost:8088`. The client attempts to connect automatically on load.

3.  **Audio Playback:**
    Once connected, audio from the server should start playing, and the loudness meters will activate. The "Start" button changes state to reflect the connection and playback status.

## How It Works

1.  The Python server ([`server/server.py`](server/server.py:1)) starts an `aiohttp` web server.
2.  It serves the client files ([`client/index.html`](client/index.html:1) and [`client/js/main.js`](client/js/main.js:1)).
3.  The client ([`client/js/main.js`](client/js/main.js:1)) automatically initiates a WebRTC connection on page load:
    *   It creates an `RTCPeerConnection`.
    *   It adds an audio transceiver set to `recvonly`.
    *   It creates an SDP offer, potentially modifying it (e.g., to request stereo for Opus), and sends it to the `/offer` endpoint on the server.
4.  The server receives the offer:
    *   It creates its own `RTCPeerConnection`.
    *   It sets the remote description from the client's offer.
    *   It adds an audio track to the peer connection. This track is either:
        *   An `AudioInputTrack` ([`server/audio_tracks.py`](server/audio_tracks.py:120)): This track reads audio data from a `NumpyRingBuffer`. The ring buffer is populated by `AudioCaptureManager`, which handles the actual audio capture from the selected device in a separate thread/process.
        *   A `SineWaveTrack` ([`server/audio_tracks.py`](server/audio_tracks.py:209)): This track generates a sine wave.
    *   It creates an SDP answer, modifying it based on CLI arguments (codec preferences, Opus parameters, bitrate), and sends it back to the client.
5.  The client receives the server's answer and sets it as the remote description.
6.  The WebRTC connection is established.
7.  When the server's audio track starts sending data, the client's `ontrack` event fires. The received audio stream is then attached to an HTML `<audio>` element for playback.
8.  The client also sets up Web Audio API components (AnalyserNodes) to process the received stream and update the SVG loudness meters.

## Logging

-   The server logs information to `app.log` (in the `server/` directory) and to the console.
-   The client logs information to the browser's developer console.

## Key Files

-   **[`server/server.py`](server/server.py:1)**: Main server logic, WebRTC signaling, CLI argument parsing, SDP answer modification.
-   **[`server/audio_tracks.py`](server/audio_tracks.py:1)**: Defines `AudioInputTrack` (consumes from ring buffer) and `SineWaveTrack`.
-   **`server/audio_capture_manager.py`** (Content not provided but inferred): Responsible for audio device enumeration, selection (possibly TUI), and capturing audio into a `NumpyRingBuffer` for `AudioInputTrack`.
-   **[`client/index.html`](client/index.html:1)**: HTML structure for the client, including the audio player and SVG meters.
-   **[`client/js/main.js`](client/js/main.js:1)**: Client-side WebRTC logic, SDP offer modification, audio playback, and loudness meter updates.
-   **[`requirements.txt`](requirements.txt:1)**: Python package dependencies.

## Potential Improvements / Future Work

-   Add video streaming capabilities.
-   Implement STUN/TURN server configuration for connections over NAT/firewalls.
-   Allow client to select audio source or control server-side settings dynamically.
-   Secure the signaling channel (e.g., HTTPS/WSS).
-   More detailed error reporting on the client for specific WebRTC/Audio issues.
-   Add unit and integration tests for both server and client components.
-   Explore different audio processing options on the client or server.