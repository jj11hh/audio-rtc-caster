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
├── server-go/              # Server implemented in Go
│   └── main.go           # Main Go server application
└── server/                 # Server implemented in Python
    └── server.py         # Main Python server application

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

-   **Go WebRTC Server (`server-go/main.go`)**:
    -   Uses Go's standard `net/http` package for the web server (signaling and serving client files).
    -   Uses `pion/webrtc/v4` for WebRTC peer connection and media handling.
    -   **Audio Input Management**:
        -   Captures audio from the system's default input device using WASAPI on Windows (via `go-ole` and custom WASAPI bindings - not shown in `main.go` but inferred from `CoInitialize` and typical usage).
    -   **Alternative Sine Wave Source**: Option to stream a generated 440Hz sine wave for testing (`--sine-wave` argument).
    -   **SDP Customization / Codec Handling**:
        -   Determines codec capabilities based on command-line arguments to prefer specific codecs (Opus, PCMU, PCMA, G722).
        -   Allows fine-tuning of Opus parameters (e.g., `opus-maxaveragebitrate`, `opus-maxplaybackrate`, `opus-cbr`, `opus-useinbandfec`, `opus-usedtx`).
    -   Serves the HTML/JS client (attempts to serve from `./client` then `../client`).
    -   Logs to the console.

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

2.  **Start the Go Server (Alternative to Python Server):**
    Navigate to the `server-go/` directory:
    ```bash
    cd server-go
    go run main.go [OPTIONS]
    ```
    Or build it first:
    ```bash
    go build -o webrtc_audio_server main.go
    ./webrtc_audio_server [OPTIONS]
    ```

    **Command-line arguments for `main.go`:**
    *   `--sine-wave`: Stream a 440Hz sine wave instead of system audio. Default: `false`.
    *   `--preferred-codec <CODEC>`: Preferred audio codec. Choices: `opus`, `pcmu`, `pcma`, `g722`. Default: `opus`.
    *   `--audio-bitrate <BPS>`: Target audio bitrate in bps for Opus encoder if `opus-maxaveragebitrate` is not set. Default: `96000`.
    *   `--opus-maxaveragebitrate <BPS>`: Opus specific: Sets 'maxaveragebitrate' in bps. Overrides `--audio-bitrate` for Opus. `0` means not set. Default: `0`.
    *   `--opus-maxplaybackrate <HZ>`: Opus specific: Sets 'maxplaybackrate' (e.g., 8000, 12000, 16000, 24000, 48000). Default: `48000`.
    *   `--opus-cbr`: Opus specific: Enable constant bitrate. Default: `false`.
    *   `--opus-useinbandfec`: Opus specific: Enable inband Forward Error Correction. Default: `true`.
    *   `--opus-usedtx`: Opus specific: Enable Discontinuous Transmission. Default: `true`.
    *   `--port <PORT>`: Port for the HTTP server. Default: `8088`.

    The server will log its status and the URL to access the client (usually `http://localhost:8088` or `http://<your-hostname>:8088`).

3.  **Open the Client in a Web Browser (after starting one of the servers):**
    Open the URL provided by the server in a WebRTC-compatible browser (e.g., Chrome, Firefox). This is typically `http://localhost:8088`. The client attempts to connect automatically on load.

4.  **Audio Playback:**
    Once connected, audio from the server should start playing, and the loudness meters will activate. The "Start" button changes state to reflect the connection and playback status.

## How It Works

## How It Works (General, focusing on Python server example)

*The Go server ([`server-go/main.go`](server-go/main.go:1)) follows a similar WebRTC flow but uses Go's `net/http` for serving and `pion/webrtc` for media handling.*

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

-   **[`server/server.py`](server/server.py:1)**: Main Python server logic, WebRTC signaling, CLI argument parsing, SDP answer modification.
-   **[`server-go/main.go`](server-go/main.go:1)**: Main Go server logic, WebRTC signaling, CLI argument parsing, audio capture (WASAPI or sine wave), and media handling with Pion WebRTC.
-   **[`server/audio_tracks.py`](server/audio_tracks.py:1)**: (Python server) Defines `AudioInputTrack` (consumes from ring buffer) and `SineWaveTrack`.
-   **`server/audio_capture_manager.py`** (Content not provided but inferred for Python server): Responsible for audio device enumeration, selection (possibly TUI), and capturing audio into a `NumpyRingBuffer` for `AudioInputTrack`.
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

## Creating a Release Package

A Python script ([`create_release_package.py`](create_release_package.py:1)) is provided to simplify the process of packaging the Go server and the client application for distribution. This is particularly useful because the Go server can be compiled into a standalone executable, removing the need for a Go environment on the target machine, unlike the Python server which requires a Python environment and dependencies.

### Prerequisites

-   Python 3.x installed and in your PATH.
-   Go programming language installed and in your PATH (for compiling the Go server).
-   Git installed and in your PATH (for versioning the package name).
-   The project should be a Git repository to allow the script to determine the branch/tag and commit hash.

### How to Use

1.  Navigate to the root directory of the project in your terminal.
2.  Run the script:
    ```bash
    python create_release_package.py
    ```

### What the Script Does

1.  **Determines Platform:** Identifies the operating system (e.g., `windows`, `linux`, `darwin`).
2.  **Fetches Git Information:**
    -   Tries to get the current Git tag. If no tag is on the current commit, it uses the current branch name.
    -   Gets the short commit hash of the current HEAD.
3.  **Builds Go Server:**
    -   The build command is executed from the project's root directory.
    -   It compiles the Go server package located at `./server-go` (which includes [`main.go`](server-go/main.go:1)).
    -   The output executable will be named `server-go.exe` on Windows, and `server-go` on other platforms (e.g., Linux, macOS). The executable is placed inside the `server-go/` directory (e.g., `server-go/server-go.exe`).
4.  **Creates Zip Package:**
    -   A zip file is created in the project root directory.
    -   The naming convention for the zip file is: `rtc-caster-[platform]-[tag_or_branch]-[commit_hash].zip`
        -   Example: `rtc-caster-windows-main-a1b2c3d.zip` or `rtc-caster-linux-v1.0.0-e4f5g6h.zip`
    -   **Contents of the zip file:**
        -   The compiled Go server executable (e.g., `server-go.exe` or `server-go`) at the root of the archive.
        -   The entire `client/` folder and its contents, maintaining its structure (e.g., [`client/index.html`](client/index.html:1), [`client/js/main.js`](client/js/main.js:1)).

After running the script, you will find the zip package in the project's root directory, ready for distribution.