'use strict';

const startButton = document.getElementById('startButton');
const audioPlayer = document.getElementById('audio');
const statusDiv = document.getElementById('status');
const leftMeterBar = document.getElementById('leftMeterBar');
const rightMeterBar = document.getElementById('rightMeterBar');

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

        // Setup Web Audio API for meters if not already done
        if (!audioContext && event.streams && event.streams[0]) {
             setupAudioAnalysis(event.streams[0]);
        } else if (!audioContext && localStream && localStream.getAudioTracks().length > 0) {
            // Fallback if stream was constructed manually
            setupAudioAnalysis(localStream);
        }

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
                console.log(`Audio player state after play success: muted=${audioPlayer.muted}, paused=${audioPlayer.paused}, volume=${audioPlayer.volume}`);
                statusDiv.textContent = '状态：音频正在播放。';
                startButton.textContent = '正在播放'; // Update button text
                startButton.disabled = true;
            } catch (e) {
                console.error("Error calling audioPlayer.play() or resuming AudioContext:", e);
                statusDiv.textContent = `状态：播放错误: ${e.message}`;
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
        startButton.textContent = '重新连接';
        if (!reconnectTimerId) {
            console.log(`Negotiation error. Attempting to reconnect in ${reconnectInterval / 1000}s...`);
            reconnectTimerId = setTimeout(connect, reconnectInterval);
        }
    }

}

function setupAudioAnalysis(stream) {
    console.log('Setting up Web Audio analysis for meters...');
    try {
        if (audioContext) {
            console.log("AudioContext already exists. Closing previous one.");
            audioContext.close(); // Close existing context before creating new
        }
        audioContext = new (window.AudioContext || window.webkitAudioContext)();

        // Check if the stream has audio tracks
        if (!stream || stream.getAudioTracks().length === 0) {
            console.error("Stream has no audio tracks for analysis.");
            if (audioContext) audioContext.close();
            audioContext = null;
            return;
        }
        
        source = audioContext.createMediaStreamSource(stream);
        splitter = audioContext.createChannelSplitter(2); // Assume stereo
        analyserL = audioContext.createAnalyser();
        analyserL.fftSize = 256; // Smaller FFT size for faster processing
        analyserR = audioContext.createAnalyser();
        analyserR.fftSize = 256;

        // Connect nodes: source -> splitter -> analysers
        source.connect(splitter);
        splitter.connect(analyserL, 0); // Connect channel 0 (Left) to analyserL
        splitter.connect(analyserR, 1); // Connect channel 1 (Right) to analyserR
        // Note: We don't connect analysers to destination, we just read data

        // Prepare data arrays
        const bufferLengthL = analyserL.frequencyBinCount; // or analyserL.fftSize for time domain
        dataArrayL = new Float32Array(bufferLengthL); // Use Float32Array for time domain data
        const bufferLengthR = analyserR.frequencyBinCount;
        dataArrayR = new Float32Array(bufferLengthR);

        console.log('Audio analysis nodes created and connected.');

        // Start updating meters
        startMeterUpdates();

    } catch (e) {
        console.error('Error setting up Web Audio API:', e);
        statusDiv.textContent += ' (无法初始化音频分析)';
        // Clean up partially created context
        if (audioContext) audioContext.close();
        audioContext = null;
    }
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

        // Schedule next update
        rafId = requestAnimationFrame(updateMeters);
    };

    rafId = requestAnimationFrame(updateMeters);
}

function cleanupAudioAnalysis() {
    console.log("Cleaning up audio analysis resources.");
    stopMeterUpdates();
    if (source) {
        source.disconnect();
        source = null;
    }
    if (splitter) {
        splitter.disconnect();
        splitter = null;
    }
    // Analysers don't need explicit disconnect if source is disconnected
    analyserL = null;
    analyserR = null;
    if (audioContext && audioContext.state !== 'closed') {
        audioContext.close().then(() => console.log("AudioContext closed."));
    }
    audioContext = null;
    dataArrayL = null;
    dataArrayR = null;
}

// Note: The pc.oniceconnectionstatechange handler was already correctly defined
// within the global scope, as it's an event handler assigned to pc.
// It does not use `await` directly.

startButton.onclick = connect;

// Auto-connect on page load
window.addEventListener('load', () => {
    console.log('Page loaded. Attempting to connect...');
    connect();
});

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