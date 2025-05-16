# WebRTC Audio Streaming Project

This project demonstrates a WebRTC audio streaming setup where a server captures audio (either from a system input (or loopback) device or a generated sine wave) and streams it to a web client, which then plays the audio and displays loudness meters.

## Project Structure

```
.
├── requirements.txt        # Python dependencies
├── client/
│   ├── index.html          # Main HTML page for the client
│   └── js/
│       └── main.js         # Client-side JavaScript for WebRTC connection, audio playback, and meters
├── server-go/              # Server implemented in Go
│   ├── main.go             # Main Go server application
│   └── ...
└── server/                 # Server implemented in Python
    ├── server.py         # Main Python server application
│   └── ...

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

If you are using Windows, just download the zip file from release page and run the `.exe` file.

To use the the Python server, you will need to install the following dependencies:

1.  **Clone the repository (if applicable).**
2.  **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Ensure you have an audio input device configured** if you are not using the sine wave option.

## Running the Application

1.  **Start the server:**
    Navigate to the project root or `server/` directory:
    ```bash
    python server/server.py [OPTIONS]
    ```

    Alternatively, to start the go server (Currently only Windows is supported):
    ```bash
    go build ./server-go
    ./server-go/server-go [OPTIONS]
    ```   

    **Command-line arguments:**
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
