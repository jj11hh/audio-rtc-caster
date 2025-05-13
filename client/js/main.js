'use strict';

const startButton = document.getElementById('startButton');
const audioPlayer = document.getElementById('audio');
const statusDiv = document.getElementById('status');

let pc;
let localStream; // Not used for sending, but good practice to declare
let reconnectInterval = 5000; // 5 seconds
let reconnectTimerId = null;

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
        audioPlayer.play()
            .then(() => {
                console.log('audioPlayer.play() promise resolved successfully.');
                console.log(`Audio player state after play success: muted=${audioPlayer.muted}, paused=${audioPlayer.paused}, volume=${audioPlayer.volume}`);
                statusDiv.textContent = '状态：音频正在播放。';
                startButton.textContent = '正在播放'; // Update button text
                startButton.disabled = true;
            })
            .catch(e => {
                console.error("Error calling audioPlayer.play():", e);
                statusDiv.textContent = `状态：播放错误: ${e.message}`;
                console.log(`Audio player state after play error: muted=${audioPlayer.muted}, paused=${audioPlayer.paused}, volume=${audioPlayer.volume}`);
                // Consider if a reconnect attempt is needed here
            });
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
        await pc.setLocalDescription(offer);
        console.log('Offer created and set as local description.');
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
        if (pc) {
            pc.close();
        }
        startButton.disabled = false;
        startButton.textContent = '重新连接';
        if (!reconnectTimerId) {
            console.log(`Negotiation error. Attempting to reconnect in ${reconnectInterval / 1000}s...`);
            reconnectTimerId = setTimeout(connect, reconnectInterval);
        }
    }
}

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
    if (pc) {
        pc.onicecandidate = null;
        pc.oniceconnectionstatechange = null;
        pc.ontrack = null;
        pc.close();
        pc = null;
    }
});