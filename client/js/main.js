'use strict';

const startButton = document.getElementById('startButton');
const audioPlayer = document.getElementById('audio');
const statusDiv = document.getElementById('status');
const leftMeterBar = document.getElementById('leftMeterBar');
const rightMeterBar = document.getElementById('rightMeterBar');
const lowLatencyToggle = document.getElementById('lowLatencyToggle');
const latencyTestButton = document.getElementById('latencyTestButton');
const latencyResultDiv = document.getElementById('latencyResult');

let pc;
let localStream; // Might be used if track needs adding to a new stream
let reconnectInterval = 5000; // 5 seconds
let reconnectTimerId = null;

// Web Audio API variables
let audioContext = null;
let analyserL = null;
let analyserR = null;
let splitter = null;
let source = null;
let dataArrayL = null;
let dataArrayR = null;
let rafId = null; // requestAnimationFrame ID

// Latency Test Variables
let pulseStartTime = 0;
let isLatencyTestRunning = false;
const PULSE_FREQUENCY = 1000; // Hz
const PULSE_DURATION_SEC = 0.05; // 50ms
const PULSE_AMPLITUDE = 0.5; // Peak amplitude of the pulse
const PULSE_DETECTION_THRESHOLD = 0.05; // RMS threshold to detect the returning pulse. (Kept for now, but new logic uses sample threshold)
const PULSE_SAMPLE_DETECTION_THRESHOLD = 0.02; // Threshold for individual audio samples to detect pulse arrival.
let lastTestInitiationTime = 0;
const MIN_TIME_BETWEEN_TESTS_MS = 2000; // Cooldown period

async function connect() {
    if (reconnectTimerId) {
        clearTimeout(reconnectTimerId);
        reconnectTimerId = null;
    }

    if (pc && pc.connectionState !== "closed" && pc.connectionState !== "failed" && pc.connectionState !== "new") {
        console.log('Already connected or connecting. Current state:', pc.connectionState);
        statusDiv.textContent = '状态：已经连接或正在连接中。';
        return;
    }
    
    statusDiv.textContent = '状态：正在连接...';
    startButton.disabled = true;
    startButton.textContent = '正在连接...';

    // Clean up previous connection if any
    if (pc) {
        pc.close();
        pc = null;
    }

    const configuration = {}; // No STUN/TURN servers for local demo

    pc = new RTCPeerConnection(configuration);

    pc.onicecandidate = event => {
        if (event.candidate) {
            console.log('Local ICE candidate:', event.candidate);
        }
    };

    pc.oniceconnectionstatechange = event => {
        console.log('ICE connection state change:', pc.iceConnectionState);
        statusDiv.textContent = `状态：ICE 连接状态: ${pc.iceConnectionState}`;
        switch (pc.iceConnectionState) {
            case 'connected':
            case 'completed':
                statusDiv.textContent = '状态：已连接。';
                startButton.textContent = '已连接';
                startButton.disabled = true; // Keep disabled while connected
                if (reconnectTimerId) {
                    clearTimeout(reconnectTimerId);
                    reconnectTimerId = null;
                }
                break;
            case 'disconnected':
                statusDiv.textContent = '状态：连接已断开。尝试重新连接...';
                // Browsers might automatically try to reconnect. If not, we can trigger.
                // For now, rely on 'failed' or 'closed' for explicit reconnect logic.
                // Attempt to reconnect if disconnected
                if (!reconnectTimerId) {
                    console.log(`ICE disconnected. Attempting to reconnect in ${reconnectInterval / 1000}s...`);
                    reconnectTimerId = setTimeout(connect, reconnectInterval);
                }
                break;
            case 'failed':
                statusDiv.textContent = '状态：连接失败。请检查服务器和网络。尝试重新连接...';
                if (pc) pc.close();
                startButton.disabled = false;
                startButton.textContent = '重新连接';
                if (!reconnectTimerId) {
                    console.log(`ICE failed. Attempting to reconnect in ${reconnectInterval / 1000}s...`);
                    reconnectTimerId = setTimeout(connect, reconnectInterval);
                }
                break;
            case 'closed':
                statusDiv.textContent = '状态：连接已关闭。';
                startButton.disabled = false;
                startButton.textContent = '开始播放';
                // Do not automatically reconnect if closed by user/application
                if (reconnectTimerId) {
                    clearTimeout(reconnectTimerId);
                    reconnectTimerId = null;
                }
                break;
        }
    };

    pc.ontrack = event => {
        console.log('Remote track received:', event.track);
        const track = event.track;
        console.log(`Remote track details - ID: ${track.id}, Kind: ${track.kind}, Label: ${track.label}, Enabled: ${track.enabled}, Muted: ${track.muted}, ReadyState: ${track.readyState}`);

        if (track.kind === 'audio') {
            console.log(`Initial remote track state: Enabled: ${track.enabled}, Muted: ${track.muted}`);
            // Log channel count from track settings
            try {
                const settings = track.getSettings();
                console.log(`Remote track settings: channelCount=${settings.channelCount}`, settings);
            } catch (e) {
                console.error('Could not get track settings:', e);
            }
            // Apply low latency settings if applicable
            if (event.receiver) {
                applyLowLatencySettings(event.receiver);
            } else {
                console.warn('Low latency mode: event.receiver not available on track event. Cannot apply playoutDelayHint.');
            }
        }

        if (event.streams && event.streams[0]) {
            console.log('Attaching stream to audio player:', event.streams[0]);
            audioPlayer.srcObject = event.streams[0];
            statusDiv.textContent = '状态：正在接收音频...';
            console.log('Audio stream attached to player. audioPlayer.srcObject set.');
        } else {
            console.log('Using fallback to attach track to player.');
            if (!localStream) {
                localStream = new MediaStream();
                console.log('Created new local MediaStream for fallback.');
            }
            localStream.addTrack(track);
            audioPlayer.srcObject = localStream;
            statusDiv.textContent = '状态：正在接收音频 (fallback)...';
            console.log('Audio track added to player via fallback. audioPlayer.srcObject set.');
        }

        // Web Audio API setup will be handled inside playAudio after user gesture confirmation

        // Log audio element state BEFORE trying to play
        console.log(`Audio player state before play attempt: muted=${audioPlayer.muted}, paused=${audioPlayer.paused}, volume=${audioPlayer.volume}, readyState=${audioPlayer.readyState}, networkState=${audioPlayer.networkState}`);

        // Attempt to unmute the audio player element itself
        if (audioPlayer.muted) {
            console.log('Audio player HTML element was muted, attempting to set audioPlayer.muted = false.');
            audioPlayer.muted = false;
            console.log(`Audio player HTML element muted status after attempt: ${audioPlayer.muted}`);
        }

        // Try to play
        console.log('Attempting audioPlayer.play()...');

        const playAudio = async () => {
            try {
                // For iOS Safari, ensure AudioContext is resumed by user gesture
                // The 'connect' button click is our primary user gesture.
                // If audioContext was created and is suspended, try to resume it.
                if (audioContext && audioContext.state === 'suspended') {
                    console.log('AudioContext is suspended, attempting to resume...');
                    await audioContext.resume();
                    console.log('AudioContext resumed, state:', audioContext.state);
                }

                await audioPlayer.play();
                console.log('audioPlayer.play() promise resolved successfully.');

                // Ensure AudioContext is running and setup analysis
                if (await ensureAudioContextRunning()) {
                    const streamForAnalysis = (event.streams && event.streams[0]) || (localStream && localStream.getAudioTracks().length > 0 ? localStream : null);
                    if (streamForAnalysis) {
                        setupAudioAnalysis(streamForAnalysis);
                    } else {
                        console.warn("No valid stream found for audio analysis setup.");
                    }
                } else {
                    console.warn("AudioContext could not be started/resumed. Audio analysis/meters will be disabled.");
                    latencyTestButton.disabled = true;
                }

                console.log(`Audio player state after play success: muted=${audioPlayer.muted}, paused=${audioPlayer.paused}, volume=${audioPlayer.volume}`);
                statusDiv.textContent = '状态：音频正在播放。';
                startButton.textContent = '正在播放'; // Update button text
                if (audioContext && audioContext.state === 'running') { // Only enable if context is good
                    latencyTestButton.disabled = false;
                } else {
                    latencyTestButton.disabled = true;
                }
                startButton.disabled = true;
            } catch (e) {
                console.error("Error calling audioPlayer.play() or resuming AudioContext:", e);
                statusDiv.textContent = `状态：播放错误: ${e.message}`;
                latencyTestButton.disabled = true; // Disable on error
                console.log(`Audio player state after play error: muted=${audioPlayer.muted}, paused=${audioPlayer.paused}, volume=${audioPlayer.volume}`);
                // Consider if a reconnect attempt is needed here
                // On iOS, if play() fails due to no user interaction, this message might appear.
                if (e.name === 'NotAllowedError') {
                    statusDiv.textContent += ' (请确保浏览器允许自动播放或在设置中启用声音)';
                }
            }
        };

        playAudio();
    };
    
    pc.addTransceiver('audio', { direction: 'recvonly' });

    try {
        // Define offer options with audio constraints
        const offerOptions = {
            offerToReceiveAudio: true, // Explicitly state we want to receive audio
            offerToReceiveVideo: false, // We don't want video
            // Request disabling of common audio processing
            echoCancellation: false,
            noiseSuppression: false,
            autoGainControl: false
        };
        console.log('Creating offer with options:', offerOptions);
        const offer = await pc.createOffer(offerOptions);

        // Modify SDP offer to explicitly request stereo for Opus
        if (offer && offer.sdp) {
            console.log('Original SDP offer line for Opus fmtp:', offer.sdp.match(/^a=fmtp:111.*$/m)?.[0]);
            let sdpLines = offer.sdp.split('\r\n');
            let opusPayloadType = null;
            let opusRtpmapIndex = -1; // Store index of a=rtpmap line for Opus

            // Find Opus payload type and its rtpmap line index
            for (let i = 0; i < sdpLines.length; i++) {
                // Match 'opus' with any sample rate, and 2 channels
                const rtpmapMatch = sdpLines[i].match(/^a=rtpmap:(\d+)\s+opus\/(\d+)\/2/i);
                if (rtpmapMatch) {
                    opusPayloadType = rtpmapMatch[1];
                    const sampleRate = rtpmapMatch[2]; // Captured sample rate
                    opusRtpmapIndex = i; // Save the index of the rtpmap line
                    console.log(`Found Opus payload type: ${opusPayloadType} (sample rate: ${sampleRate}, channels: 2) at line ${i}`);
                    break;
                }
            }

            if (opusPayloadType) {
                let fmtpLineIndex = -1;
                // Find existing fmtp line for Opus
                for (let i = 0; i < sdpLines.length; i++) {
                    if (sdpLines[i].startsWith(`a=fmtp:${opusPayloadType}`)) {
                        fmtpLineIndex = i;
                        console.log(`Found existing fmtp line for Opus (payload ${opusPayloadType}) at line ${i}: ${sdpLines[i]}`);
                        break;
                    }
                }

                if (fmtpLineIndex !== -1) {
                    // Modify existing fmtp line
                    let fmtpLine = sdpLines[fmtpLineIndex];
                    if (!fmtpLine.includes('stereo=1')) {
                        fmtpLine += ';stereo=1';
                    }
                    if (!fmtpLine.includes('sprop-stereo=1')) {
                        fmtpLine += ';sprop-stereo=1';
                    }
                    sdpLines[fmtpLineIndex] = fmtpLine;
                    console.log(`Modified SDP offer line for Opus fmtp (payload ${opusPayloadType}): ${fmtpLine}`);
                } else if (opusRtpmapIndex !== -1) {
                    // Add new fmtp line if rtpmap exists but fmtp doesn't
                    // Use opusRtpmapIndex to insert the new fmtp line correctly
                    const newFmtpLine = `a=fmtp:${opusPayloadType} minptime=10;useinbandfec=1;stereo=1;sprop-stereo=1`;
                    let insertIndex = opusRtpmapIndex + 1;
                    // Find a good place to insert: after other 'a=' lines for this media description, before next 'm=' or end.
                    while(insertIndex < sdpLines.length &&
                          sdpLines[insertIndex].startsWith('a=') &&
                          !sdpLines[insertIndex].startsWith('m=')) {
                        insertIndex++;
                    }
                    sdpLines.splice(insertIndex, 0, newFmtpLine);
                    console.log(`Added SDP offer line for Opus fmtp (payload ${opusPayloadType}): ${newFmtpLine}`);
                }
                offer.sdp = sdpLines.join('\r\n');
            } else {
                console.warn('Opus codec (opus/.../2) not found in SDP offer. Cannot modify for stereo.');
            }
        }

        await pc.setLocalDescription(offer);
        console.log('Offer created and set as local description (with stereo hint).');
        console.log('--- BEGIN SDP OFFER ---');
        console.log(offer.sdp);
        console.log('--- END SDP OFFER ---');

        const response = await fetch('/offer', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ sdp: offer.sdp, type: offer.type }),
        });

        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`Server responded with ${response.status}: ${errorText}`);
        }

        const answer = await response.json();
        console.log('Answer received from server.');
        console.log('--- BEGIN SDP ANSWER ---');
        console.log(answer.sdp);
        console.log('--- END SDP ANSWER ---');
        await pc.setRemoteDescription(new RTCSessionDescription(answer));
        statusDiv.textContent = '状态：已连接，等待音频流...';
        // startButton.disabled = true; // Already handled in oniceconnectionstatechange

    } catch (e) {
        console.error('Error during WebRTC negotiation:', e);
        statusDiv.textContent = `状态：连接错误: ${e.message}. 尝试重新连接...`;
        cleanupAudioAnalysis(); // Clean up audio resources on negotiation error
        if (pc) {
            pc.close();
            pc = null; // Ensure pc is nullified after close
        }
        startButton.disabled = false;
        latencyTestButton.disabled = true;
        startButton.textContent = '重新连接';
        if (!reconnectTimerId) {
            console.log(`Negotiation error. Attempting to reconnect in ${reconnectInterval / 1000}s...`);
            reconnectTimerId = setTimeout(connect, reconnectInterval);
        }
    }

}

async function ensureAudioContextRunning() {
    if (!audioContext) {
        console.log("AudioContext is null, creating new one (triggered by user gesture or allowed autoplay).");
        try {
            audioContext = new (window.AudioContext || window.webkitAudioContext)();
            console.log("AudioContext created, state:", audioContext.state);
        } catch (e) {
            console.error("Failed to create AudioContext:", e);
            if (statusDiv) statusDiv.textContent += ' (Error: Audio system init failed.)';
            if (latencyTestButton) latencyTestButton.disabled = true;
            return false;
        }
    }

    if (audioContext.state === 'suspended') {
        console.log("AudioContext is suspended, attempting to resume...");
        try {
            await audioContext.resume();
            console.log("AudioContext resumed, state:", audioContext.state);
        } catch (e) {
            console.error("Failed to resume AudioContext:", e);
            if (statusDiv) statusDiv.textContent += ' (Error: Could not resume audio.)';
            if (latencyTestButton) latencyTestButton.disabled = true;
            return false;
        }
    }

    if (audioContext.state === 'running') {
        return true;
    } else {
        console.warn("AudioContext not running after creation/resume attempt. State:", audioContext.state);
        if (latencyTestButton) latencyTestButton.disabled = true;
        return false;
    }
}

function setupAudioAnalysis(stream) {
    if (!audioContext || audioContext.state !== 'running') {
        console.log('setupAudioAnalysis: AudioContext not available or not running. Aborting setup.');
        return;
    }

    if (source && source.mediaStream === stream) {
        console.log("setupAudioAnalysis: Already initialized with the same stream.");
        return;
    }

    if (source) {
        console.log("setupAudioAnalysis: Stream changed or re-initializing nodes.");
        cleanupAudioAnalysisInternal();
    } else {
        console.log("setupAudioAnalysis: Initializing nodes for stream:", stream);
    }
    
    try {
        if (!stream || stream.getAudioTracks().length === 0) {
            console.error("Stream has no audio tracks for analysis.");
            // Don't close/nullify audioContext here as it's managed by ensureAudioContextRunning
            return;
        }
        
        source = audioContext.createMediaStreamSource(stream);
        splitter = audioContext.createChannelSplitter(2);
        analyserL = audioContext.createAnalyser();
        analyserL.fftSize = 256;
        analyserR = audioContext.createAnalyser();
        analyserR.fftSize = 256;

        source.connect(splitter);
        splitter.connect(analyserL, 0);
        splitter.connect(analyserR, 1);

        const bufferLengthL = analyserL.fftSize;
        dataArrayL = new Float32Array(bufferLengthL);
        const bufferLengthR = analyserR.fftSize;
        dataArrayR = new Float32Array(bufferLengthR);

        console.log('Audio analysis nodes created and connected.');
        startMeterUpdates();

    } catch (e) {
        console.error('Error setting up Web Audio API nodes:', e);
        if (statusDiv) statusDiv.textContent += ' (无法初始化音频分析节点)';
        cleanupAudioAnalysisInternal(); // Clean up any partially created nodes
        if (latencyTestButton) latencyTestButton.disabled = true;
    }
}

function cleanupAudioAnalysisInternal() {
    console.log("Cleaning up audio analysis nodes (internal).");
    stopMeterUpdates();
    if (source) {
        try { source.disconnect(); } catch(e) { console.warn("Error disconnecting source:", e); }
        source = null;
    }
    if (splitter) {
        try { splitter.disconnect(); } catch(e) { console.warn("Error disconnecting splitter:", e); }
        splitter = null;
    }
    analyserL = null;
    analyserR = null;
    // dataArrayL and dataArrayR are nulled in the main cleanup or re-created in setup
}

function stopMeterUpdates() {
    if (rafId) {
        cancelAnimationFrame(rafId);
        rafId = null;
    }
    // Reset meter display
    if (leftMeterBar) leftMeterBar.setAttribute('width', '0%');
    if (rightMeterBar) rightMeterBar.setAttribute('width', '0%');
}

function startMeterUpdates() {
    if (!analyserL || !analyserR || !leftMeterBar || !rightMeterBar) {
        console.log("Analysers or meter elements not ready. Cannot start updates.");
        return;
    }
    console.log("Starting meter updates loop.");
    stopMeterUpdates(); // Ensure no previous loop is running

    const updateMeters = () => {
        // Get time domain data
        analyserL.getFloatTimeDomainData(dataArrayL);
        analyserR.getFloatTimeDomainData(dataArrayR);

        // Calculate RMS (Root Mean Square) as a measure of loudness
        let sumSquaresL = 0.0;
        for (const amplitude of dataArrayL) { sumSquaresL += amplitude * amplitude; }
        const rmsL = Math.sqrt(sumSquaresL / dataArrayL.length);

        let sumSquaresR = 0.0;
        for (const amplitude of dataArrayR) { sumSquaresR += amplitude * amplitude; }
        const rmsR = Math.sqrt(sumSquaresR / dataArrayR.length);

        // Convert RMS to percentage (adjust multiplier for sensitivity)
        // RMS values are typically between 0 and 1. Multiply to scale to 0-100.
        // A multiplier of ~150-200 often gives reasonable visual results.
        const meterScale = 200;
        const percentL = Math.min(100, rmsL * meterScale);
        const percentR = Math.min(100, rmsR * meterScale);

        // Update SVG bar width
        leftMeterBar.setAttribute('width', percentL + '%');
        rightMeterBar.setAttribute('width', percentR + '%');

        // Latency Test Pulse Detection (New Logic)
        if (isLatencyTestRunning && audioContext && audioContext.state === 'running' && dataArrayL) {
            const frameCurrentTime = audioContext.currentTime; // Capture AudioContext time for this frame of analysis

            for (let i = 0; i < dataArrayL.length; i++) {
                if (Math.abs(dataArrayL[i]) > PULSE_SAMPLE_DETECTION_THRESHOLD) {
                    // Estimate the actual time the detected sample occurred
                    // dataArrayL contains 'dataArrayL.length' samples.
                    // The last sample (index dataArrayL.length - 1) corresponds most closely to 'frameCurrentTime'.
                    // Sample 'i' occurred (dataArrayL.length - 1 - i) samples *before* the last sample in the buffer.
                    const samplesBeforeEndOfBuffer = dataArrayL.length - 1 - i;
                    const timeBeforeEndOfBuffer = samplesBeforeEndOfBuffer / audioContext.sampleRate;
                    
                    const pulseEventTime = frameCurrentTime - timeBeforeEndOfBuffer;
                    const latencyMs = (pulseEventTime - pulseStartTime) * 1000;

                    console.log(`Pulse sample detected at index ${i}. Value: ${dataArrayL[i].toFixed(4)}. Est. Event Time: ${pulseEventTime.toFixed(3)} (Frame time: ${frameCurrentTime.toFixed(3)}), Start: ${pulseStartTime.toFixed(3)}, Latency: ${latencyMs.toFixed(2)} ms`);
                    
                    if (latencyResultDiv) {
                        latencyResultDiv.textContent = `Latency: ${latencyMs.toFixed(2)} ms`;
                    }
                    isLatencyTestRunning = false; // Stop test
                    latencyTestButton.disabled = false; // Re-enable button
                    // No need to update lastTestInitiationTime here, that's for starting
                    break; // Exit loop once pulse is detected
                }
            }
        }

        // Schedule next update
        rafId = requestAnimationFrame(updateMeters);
    };

    rafId = requestAnimationFrame(updateMeters);
}

function cleanupAudioAnalysis() {
    console.log("Cleaning up audio analysis resources (full).");
    cleanupAudioAnalysisInternal(); // Disconnects nodes
    if (audioContext && audioContext.state !== 'closed') {
        audioContext.close().then(() => {
            console.log("AudioContext closed.");
            audioContext = null; // Nullify after successful close
        }).catch(e => {
            console.error("Error closing AudioContext:", e);
            audioContext = null; // Still nullify on error
        });
    } else {
        audioContext = null; // Ensure it's null if already closed or never existed
    }
    dataArrayL = null;
    dataArrayR = null;
}

function applyLowLatencySettings(receiver) {
    if (!lowLatencyToggle) { // Check if the toggle element exists
        console.warn('Low latency toggle element not found.');
        return;
    }

    if (!receiver || typeof receiver.playoutDelayHint === 'undefined') {
        console.log('Low latency mode: playoutDelayHint API not supported by this browser or on this receiver.');
        if (lowLatencyToggle.checked) {
            console.warn('Low latency mode is enabled, but cannot be applied due to API unavailability.');
        }
        return;
    }

    if (lowLatencyToggle.checked) {
        try {
            receiver.playoutDelayHint = 0.02; // Use a small buffer (e.g., 20ms) for better stability
            console.log('Low latency mode: Applied playoutDelayHint = 0.02 to receiver.');
        } catch (e) {
            console.error('Low latency mode: Error setting playoutDelayHint to 0.02:', e);
        }
    } else {
        try {
            receiver.playoutDelayHint = null; // Revert to browser default buffering
            console.log('Low latency mode: Cleared playoutDelayHint (reverted to browser default).');
        } catch (e) {
            console.error('Low latency mode: Error clearing playoutDelayHint:', e);
        }
    }
}

// Note: The pc.oniceconnectionstatechange handler was already correctly defined
// within the global scope, as it's an event handler assigned to pc.
// It does not use `await` directly.

startButton.onclick = connect;

if (lowLatencyToggle) {
    lowLatencyToggle.addEventListener('change', () => {
        console.log(`Low latency mode toggled: ${lowLatencyToggle.checked}`);
        if (pc && (pc.connectionState === 'connected' || pc.connectionState === 'completed')) {
            const receivers = pc.getReceivers();
            receivers.forEach(receiver => {
                if (receiver.track && receiver.track.kind === 'audio') {
                    // applyLowLatencySettings already checks for playoutDelayHint support
                    applyLowLatencySettings(receiver);
                }
            });
        } else { // Corrected brace placement and paired else
            console.log('Low latency mode: Toggle changed, but no active WebRTC connection. Setting will be used on next connection via ontrack.');
        }
    });
} // Moved functions and button assignment outside this listener

function playClientPulse() {
    if (!audioContext || audioContext.state !== 'running') {
        console.warn('Cannot play pulse: AudioContext not available or not running.');
        if (latencyResultDiv) latencyResultDiv.textContent = 'Latency: Error (Audio Offline)';
        latencyTestButton.disabled = false;
        isLatencyTestRunning = false;
        return;
    }

    console.log('Playing client pulse...');
    const oscillator = audioContext.createOscillator();
    const gainNode = audioContext.createGain();

    oscillator.type = 'sine';
    oscillator.frequency.setValueAtTime(PULSE_FREQUENCY, audioContext.currentTime);

    gainNode.gain.setValueAtTime(0, audioContext.currentTime);
    gainNode.gain.linearRampToValueAtTime(PULSE_AMPLITUDE, audioContext.currentTime + PULSE_DURATION_SEC / 5); // Quick ramp up
    gainNode.gain.setValueAtTime(PULSE_AMPLITUDE, audioContext.currentTime + PULSE_DURATION_SEC * 4 / 5);
    gainNode.gain.linearRampToValueAtTime(0, audioContext.currentTime + PULSE_DURATION_SEC); // Quick ramp down

    oscillator.connect(gainNode);
    gainNode.connect(audioContext.destination); // Play directly to output

    oscillator.start(audioContext.currentTime);
    oscillator.stop(audioContext.currentTime + PULSE_DURATION_SEC + 0.01); // Stop slightly after pulse ends

    // For latency measurement, pulseStartTime is recorded just before calling this.
    console.log(`Pulse initiated at audioContext.currentTime: ${pulseStartTime.toFixed(3)}`);
}

async function handleLatencyTest() { // Made async
    console.log('handleLatencyTest function entered.');
    const now = performance.now();
    if (now - lastTestInitiationTime < MIN_TIME_BETWEEN_TESTS_MS) {
        console.log(`Latency test skipped: Cooldown period active. Wait ${((MIN_TIME_BETWEEN_TESTS_MS - (now - lastTestInitiationTime))/1000).toFixed(1)}s.`);
        if (latencyResultDiv) latencyResultDiv.textContent = `Latency: Cooldown...`;
        return;
    }

    if (!await ensureAudioContextRunning()) {
        console.error('Cannot start latency test: AudioContext could not be started/resumed.');
        if (latencyResultDiv) latencyResultDiv.textContent = 'Latency: Error (Audio Offline)';
        isLatencyTestRunning = false;
        latencyTestButton.disabled = false;
        return;
    }
    
    // Ensure analysers are set up for pulse detection
    if (!analyserL || !dataArrayL) { // Check if analysis components are ready
        console.error('Cannot start latency test: Audio analysis (analyserL or dataArrayL) not set up.');
        if (latencyResultDiv) latencyResultDiv.textContent = 'Latency: Error (Analyser N/A)';
        isLatencyTestRunning = false;
        latencyTestButton.disabled = false;
        return;
    }

    if (isLatencyTestRunning) {
        console.log('Latency test already in progress.');
        return;
    }

    console.log('Starting latency test...');
    isLatencyTestRunning = true;
    lastTestInitiationTime = now;
    latencyTestButton.disabled = true;
    if (latencyResultDiv) latencyResultDiv.textContent = 'Latency: Testing...';

    // Record start time just before playing the pulse
    // audioContext.currentTime provides a high-resolution timestamp relative to the AudioContext's own clock
    pulseStartTime = audioContext.currentTime;
    playClientPulse();

    // Set a timeout to stop the test if no pulse is detected, to prevent button remaining disabled
    setTimeout(() => {
        if (isLatencyTestRunning) {
            console.warn('Latency test timed out. No returning pulse detected.');
            if (latencyResultDiv) latencyResultDiv.textContent = 'Latency: Timeout';
            isLatencyTestRunning = false;
            latencyTestButton.disabled = false;
        }
    }, 5000); // 5-second timeout for pulse detection
}

latencyTestButton.disabled = true; // Initially disabled until connection is up
latencyTestButton.onclick = handleLatencyTest;

// Auto-connect on page load has been removed. User must click "Start".

// Clean up on page unload
window.addEventListener('beforeunload', () => {
    if (reconnectTimerId) {
        clearTimeout(reconnectTimerId);
    }
    cleanupAudioAnalysis(); // Clean up audio analysis
    if (pc) {
        // Remove handlers before closing to avoid errors during shutdown
        try { pc.onicecandidate = null; } catch(e) {}
        try { pc.oniceconnectionstatechange = null; } catch(e) {}
        try { pc.ontrack = null; } catch(e) {}
        pc.close();
        pc = null;
    }
});