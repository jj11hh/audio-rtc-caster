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
        if (event.streams && event.streams[0]) {
            audioPlayer.srcObject = event.streams[0];
            statusDiv.textContent = '状态：正在接收音频...';
            console.log('Audio stream attached to player.');
        } else {
            // Fallback for older browsers or specific scenarios
            if (!localStream) {
                localStream = new MediaStream();
            }
            localStream.addTrack(event.track);
            audioPlayer.srcObject = localStream;
            statusDiv.textContent = '状态：正在接收音频 (fallback)...';
            console.log('Audio track added to player via fallback.');
        }
        audioPlayer.play().catch(e => console.error("Error playing audio:", e));
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