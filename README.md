# WebRTC Audio Streaming Project

This project demonstrates a WebRTC audio streaming setup where a Python server captures audio (either from a system input device or a generated sine wave) and streams it to a web client, which then plays the audio.

## Project Structure

```
.
├── requirements.txt    # Python dependencies
├── serverlog.txt       # Older server log (if any)
├── client/
│   ├── index.html      # Main HTML page for the client
│   └── js/
│       └── main.js     # Client-side JavaScript for WebRTC connection and audio playback
└── server/
    ├── audio_tracks.py # Python classes for handling audio input (device/sine wave) as MediaStreamTrack
    └── server.py       # Python server using aiohttp and aiortc to handle WebRTC signaling and streaming
```

## Features

- **Python WebRTC Server**:
    - Uses `aiohttp` for the web server (signaling and serving client files).
    - Uses `aiortc` for WebRTC peer connection and media handling.
    - Audio Input:
        - Captures audio from a selected system input device using `pyaudiowpatch` (for Windows WASAPI loopback) or `pyaudio`.
        - Can automatically detect loopback devices on Windows.
        - Provides a TUI (Text User Interface) to select an input device if not specified or auto-detected.
        - Option to stream a generated 440Hz sine wave for testing (`--sine-wave` argument).
    - Serves a simple HTML/JS client.
- **Web Client**:
    - Connects to the Python server via WebRTC.
    - Receives the audio stream and plays it using the HTML `<audio>` element.
    - Displays connection status.

## Setup and Installation

1.  **Clone the repository (if applicable).**
2.  **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    The `requirements.txt` file should include:
    ```
    aiohttp
    aiortc
    pyaudiowpatch # For Windows loopback audio capture
    pyaudio       # For general audio input
    numpy
    ```
    *Note: `pyaudiowpatch` is Windows-specific. For other OS, `pyaudio` will be the primary way to access microphones. Loopback audio capture might require different OS-specific solutions.*

3.  **Ensure you have an audio input device configured** if you are not using the sine wave option.

## Running the Application

1.  **Start the Python Server:**
    Navigate to the `server/` directory or run from the project root:
    ```bash
    python server/server.py
    ```
    Or, if you are in the `server` directory:
    ```bash
    python server.py
    ```

    **Command-line arguments for the server:**
    *   `--input-device <INDEX>`: Specify the PyAudio input device index to use.
    *   `--sine-wave`: Stream a 440Hz sine wave instead of system audio.

    If no arguments are provided, the server will:
    *   On Windows, attempt to find a WASAPI loopback device.
    *   If not found or on other OS, or if loopback detection fails, it will list available input devices and prompt for selection via a TUI.
    *   If a default input device is found and no other selection is made, it might use that.

    The server will log its status and the URL to access the client (usually `http://localhost:8088` or `http://<your-hostname>:8088`).

2.  **Open the Client in a Web Browser:**
    Open the URL provided by the server in a WebRTC-compatible browser (e.g., Chrome, Firefox). This is typically `http://localhost:8088`.

3.  **Start Streaming:**
    On the web page, click the "Start" button. The client will attempt to connect to the server. Once connected, audio from the server should start playing. The status will be updated on the page.

## How It Works

1.  The Python server ([`server/server.py`](server/server.py:1)) starts an `aiohttp` web server.
2.  It serves the client files ([`client/index.html`](client/index.html:1) and [`client/js/main.js`](client/js/main.js:1)).
3.  When the client's "Start" button is clicked, [`client/js/main.js`](client/js/main.js:1) initiates a WebRTC connection:
    *   It creates an `RTCPeerConnection`.
    *   It adds an audio transceiver set to `recvonly` (as it only receives audio).
    *   It creates an SDP offer and sends it to the `/offer` endpoint on the server.
4.  The server receives the offer:
    *   It creates its own `RTCPeerConnection`.
    *   It sets the remote description from the client's offer.
    *   It adds an audio track to the peer connection. This track is either:
        *   An `AudioInputTrack` ([`server/audio_tracks.py`](server/audio_tracks.py:217)) which captures audio from the selected PyAudio device. This class uses a ring buffer to pass audio data from a separate capture thread to the aiortc event loop.
        *   A `SineWaveTrack` ([`server/audio_tracks.py`](server/audio_tracks.py:466)) which generates a sine wave.
    *   It creates an SDP answer and sends it back to the client.
5.  The client receives the server's answer and sets it as the remote description.
6.  The WebRTC connection is established.
7.  When the server's audio track starts sending data, the client's `ontrack` event fires. The received audio stream is then attached to an HTML `<audio>` element for playback.

## Logging

-   The server logs information to `app.log` and to the console.
-   The client logs information to the browser's developer console.

## Key Files

-   **[`server/server.py`](server/server.py:1)**: Main server logic, WebRTC signaling, device selection.
-   **[`server/audio_tracks.py`](server/audio_tracks.py:1)**: Defines `AudioInputTrack` for capturing real audio and `SineWaveTrack` for generating test audio. Includes `NumpyRingBuffer` for efficient audio data transfer between threads.
-   **[`client/index.html`](client/index.html:1)**: Basic HTML structure for the client.
-   **[`client/js/main.js`](client/js/main.js:1)**: Client-side WebRTC logic.
-   **[`requirements.txt`](requirements.txt:1)**: Python package dependencies.

## Potential Improvements / Future Work

-   Add video streaming capabilities.
-   Implement STUN/TURN server configuration for connections over NAT/firewalls.
-   More robust error handling and user feedback on the client.
-   Allow client to select audio source or control server-side settings.
-   Secure the signaling channel (e.g., HTTPS/WSS).
-   Refactor device selection for better cross-platform compatibility and ease of use.
-   Add unit and integration tests.