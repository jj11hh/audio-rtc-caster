'use strict';

const startButton = document.getElementById('startButton');
const audioPlayer = document.getElementById('audio');
const statusDiv = document.getElementById('status');

let pc;
let localStream; // Not used for sending, but good practice to declare

startButton.onclick = async () => {
    if (pc && pc.connectionState !== "closed" && pc.connectionState !== "failed") {
        statusDiv.textContent = '状态：已经连接或正在连接中。';
        return;
    }
    
    statusDiv.textContent = '状态：正在连接...';
    startButton.disabled = true;

    const configuration = {}; // No STUN/TURN servers for local demo

    pc = new RTCPeerConnection(configuration);

    pc.onicecandidate = event => {
        // In this simple setup, ICE candidates are exchanged as part of the offer/answer
        // No separate signaling for candidates is implemented here.
        // The server (aiortc) will gather candidates and include them in its answer.
        if (event.candidate) {
            console.log('Local ICE candidate:', event.candidate);
        }
    };

    pc.oniceconnectionstatechange = event => {
        console.log('ICE connection state change:', pc.iceConnectionState);
        statusDiv.textContent = `状态：ICE 连接状态: ${pc.iceConnectionState}`;
        if (pc.iceConnectionState === 'connected' || pc.iceConnectionState === 'completed') {
            // Connection established
        } else if (pc.iceConnectionState === 'failed') {
            statusDiv.textContent = '状态：连接失败。请检查服务器和网络。';
            pc.close();
            startButton.disabled = false;
        } else if (pc.iceConnectionState === 'closed') {
            statusDiv.textContent = '状态：连接已关闭。';
            startButton.disabled = false;
        }
    };

    pc.ontrack = event => {
        console.log('Remote track received:', event.track);
        const track = event.track;
        console.log(`Remote track details - ID: ${track.id}, Kind: ${track.kind}, Label: ${track.label}, Enabled: ${track.enabled}, Muted: ${track.muted}, ReadyState: ${track.readyState}`);

        if (track.kind === 'audio') {
            console.log(`Initial remote track state: Enabled: ${track.enabled}, Muted: ${track.muted}`);
            if (track.muted || !track.enabled) { // If muted OR not enabled, try to force unmute/enable
                console.warn(`Track initially: Enabled: ${track.enabled}, Muted: ${track.muted}. Attempting to force unmute/enable.`);
                try {
                    console.log('Attempting to set track.enabled = false (to toggle)...');
                    track.enabled = false;
                    console.log(`Track state after setting enabled=false: Enabled: ${track.enabled}, Muted: ${track.muted}`);

                    // Use a microtask (or short setTimeout) to allow the state change to propagate before setting back to true
                    Promise.resolve().then(() => {
                        console.log('Attempting to set track.enabled = true (to complete toggle)...');
                        track.enabled = true;
                        console.log(`Track state after setting enabled=true (toggled): Enabled: ${track.enabled}, Muted: ${track.muted}`);
                        if (track.enabled && track.muted) {
                             console.warn('Track is STILL enabled but muted after toggle. This is very persistent.');
                        } else if (track.enabled && !track.muted) {
                            console.info('Track is now enabled and unmuted after toggle. Audio should play.');
                        }
                    });

                } catch (e) {
                    console.error('Error toggling track.enabled:', e);
                }
            } else {
                 console.info('Track received as enabled and not muted.');
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
            })
            .catch(e => {
                console.error("Error calling audioPlayer.play():", e);
                statusDiv.textContent = `状态：播放错误: ${e.message}`;
                console.log(`Audio player state after play error: muted=${audioPlayer.muted}, paused=${audioPlayer.paused}, volume=${audioPlayer.volume}`);
            });
    };
    
    // We are only receiving audio, so we add a transceiver for receiving audio.
    // This tells the other peer that we are ready to receive an audio track.
    pc.addTransceiver('audio', { direction: 'recvonly' });

    try {
        const offer = await pc.createOffer();
        await pc.setLocalDescription(offer);
        console.log('Offer created and set as local description:', offer.sdp);

        // Send offer to server
        const response = await fetch('/offer', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ sdp: offer.sdp, type: offer.type }),
        });

        if (!response.ok) {
            throw new Error(`Server responded with ${response.status}: ${await response.text()}`);
        }

        const answer = await response.json();
        console.log('Answer received from server:', answer.sdp);
        await pc.setRemoteDescription(new RTCSessionDescription(answer));
        statusDiv.textContent = '状态：已连接，等待音频流...';

    } catch (e) {
        console.error('Error during WebRTC negotiation:', e);
        statusDiv.textContent = `状态：连接错误: ${e.message}`;
        if (pc) {
            pc.close();
        }
        startButton.disabled = false;
    }
};

// Clean up on page unload
window.addEventListener('beforeunload', () => {
    if (pc) {
        pc.close();
        pc = null;
    }
});