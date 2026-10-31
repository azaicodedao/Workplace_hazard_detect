(function () {
    "use strict";

    const params = new URLSearchParams(location.search);
    const sessionId = params.get("session");
    const token = params.get("token");
    const video = document.getElementById("phone-video");
    const placeholder = document.getElementById("phone-camera-placeholder");
    const status = document.getElementById("phone-status");
    const message = document.getElementById("phone-message");
    const facingSelect = document.getElementById("phone-facing-mode");
    const startButton = document.getElementById("phone-start-btn");
    const stopButton = document.getElementById("phone-stop-btn");

    let stream = null;
    let socket = null;
    let peer = null;
    let iceServers = [];
    let pendingIce = [];

    function setStatus(type, label) {
        status.className = "status-badge badge-" + type;
        status.textContent = label;
    }

    function setMessage(text, error) {
        message.textContent = text || "";
        message.style.color = error ? "var(--hazard)" : "var(--text-muted)";
    }

    function websocketUrl() {
        const protocol = location.protocol === "https:" ? "wss://" : "ws://";
        return protocol + location.host + "/api/camera-sessions/ws/" + encodeURIComponent(sessionId) +
            "?role=phone&token=" + encodeURIComponent(token);
    }

    function sendSignal(payload) {
        if (socket && socket.readyState === WebSocket.OPEN) socket.send(JSON.stringify(payload));
    }

    async function createPeer() {
        if (peer) peer.close();
        peer = new RTCPeerConnection({ iceServers });
        stream.getTracks().forEach((track) => peer.addTrack(track, stream));
        peer.onicecandidate = (event) => {
            if (event.candidate) sendSignal({ type: "ice_candidate", candidate: event.candidate });
        };
        peer.onconnectionstatechange = () => {
            if (peer.connectionState === "connected") {
                setStatus("streaming", "Streaming");
                setMessage("The computer is receiving this camera. Keep this page open.", false);
            } else if (["failed", "disconnected"].includes(peer.connectionState)) {
                setStatus("idle", "Connection lost");
                setMessage("The connection to the computer was lost.", true);
            }
        };
    }

    async function flushIce() {
        if (!peer || !peer.remoteDescription) return;
        const candidates = pendingIce.splice(0);
        for (const candidate of candidates) await peer.addIceCandidate(candidate);
    }

    function connectSignaling() {
        socket = new WebSocket(websocketUrl());
        socket.onopen = () => {
            setStatus("streaming", "Waiting for computer");
            sendSignal({ type: "ready" });
        };
        socket.onmessage = async (event) => {
            const signal = JSON.parse(event.data);
            if (signal.type === "connected") {
                iceServers = signal.ice_servers || [];
            } else if (signal.type === "offer") {
                await createPeer();
                await peer.setRemoteDescription(signal.sdp);
                await flushIce();
                const answer = await peer.createAnswer();
                await peer.setLocalDescription(answer);
                sendSignal({ type: "answer", sdp: peer.localDescription });
            } else if (signal.type === "ice_candidate" && signal.candidate) {
                if (peer && peer.remoteDescription) await peer.addIceCandidate(signal.candidate);
                else pendingIce.push(signal.candidate);
            } else if (signal.type === "peer_left") {
                setStatus("idle", "Computer disconnected");
                setMessage("The computer closed the camera session.", true);
            }
        };
        socket.onclose = () => {
            if (stream) {
                setStatus("idle", "Session closed");
                setMessage("The connection session is no longer available. Scan a new QR code.", true);
            }
        };
        socket.onerror = () => setMessage("Could not connect to the computer.", true);
    }

    async function start() {
        if (!sessionId || !token) {
            setMessage("This phone link is incomplete. Scan the QR code again.", true);
            return;
        }
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            setMessage("Camera access requires HTTPS and a supported browser.", true);
            return;
        }
        startButton.disabled = true;
        setStatus("streaming", "Opening camera");
        try {
            stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    facingMode: { ideal: facingSelect.value },
                    width: { ideal: 1280 },
                    height: { ideal: 720 },
                    frameRate: { ideal: 30, max: 30 }
                },
                audio: false
            });
            video.srcObject = stream;
            video.style.display = "block";
            placeholder.hidden = true;
            facingSelect.disabled = true;
            stopButton.disabled = false;
            await video.play();
            connectSignaling();
        } catch (error) {
            startButton.disabled = false;
            setStatus("idle", "Camera unavailable");
            setMessage("Could not open the phone camera: " + (error.message || error.name), true);
        }
    }

    function stop() {
        if (peer) peer.close();
        if (socket) socket.close();
        if (stream) stream.getTracks().forEach((track) => track.stop());
        peer = null;
        socket = null;
        stream = null;
        video.srcObject = null;
        video.style.display = "none";
        placeholder.hidden = false;
        facingSelect.disabled = false;
        startButton.disabled = false;
        stopButton.disabled = true;
        setStatus("idle", "Disconnected");
        setMessage("", false);
    }

    startButton.addEventListener("click", () => void start());
    stopButton.addEventListener("click", stop);
    window.addEventListener("beforeunload", stop);

    if (!sessionId || !token) {
        startButton.disabled = true;
        setMessage("Invalid phone camera link. Scan the QR code from the computer.", true);
    } else if (!window.isSecureContext && !["localhost", "127.0.0.1"].includes(location.hostname)) {
        setMessage("Open this page over HTTPS to allow phone camera access.", true);
    }
})();
