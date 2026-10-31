(function () {
    "use strict";

    const PREDICT_URL = "/predict/";
    const LIVE_PREDICT_URL = "/predict/live/";
    const CAMERA_SESSIONS_URL = "/api/camera-sessions";
    const ALLOWED_EXTENSIONS = [".mp4", ".avi", ".mov"];
    const LIVE_FRAME_COUNT = 16;
    const LIVE_WINDOW_MS = 5000;
    const CAMERA_WARMUP_MS = 2500;
    const REQUIRED_CONSECUTIVE_HAZARDS = 2;

    const byId = (id) => document.getElementById(id);
    const sourceButtons = {
        upload: byId("tab-btn-video"),
        computer_camera: byId("tab-btn-computer"),
        phone_camera: byId("tab-btn-phone")
    };

    const uploadView = byId("tab-content-video");
    const liveView = byId("tab-content-live");
    const dropzone = byId("dropzone");
    const fileInput = byId("file-input");
    const browseBtn = byId("browse-btn");
    const analyzeBtn = byId("analyze-btn");
    const clearBtn = byId("clear-btn");
    const analyzeLabel = byId("analyze-label");
    const fileMeta = byId("file-meta");
    const fileNameEl = byId("file-name");
    const fileSizeEl = byId("file-size");
    const previewWrap = byId("preview-wrap");
    const videoPreview = byId("video-preview");
    const previewFallback = byId("preview-fallback");
    const formHint = byId("form-hint");
    const resultBody = byId("result-body");

    const liveVideo = byId("live-video");
    const liveCanvas = byId("live-canvas");
    const livePlaceholder = byId("live-placeholder");
    const livePlaceholderTitle = byId("live-placeholder-title");
    const livePlaceholderSub = byId("live-placeholder-sub");
    const liveHeading = byId("live-heading");
    const liveDescription = byId("live-description");
    const liveStartBtn = byId("live-start-btn");
    const liveStartLabel = byId("live-start-label");
    const liveStopBtn = byId("live-stop-btn");
    const liveStatusBadge = byId("live-status-badge");
    const liveBufferCounter = byId("live-buffer-counter");
    const liveStreamState = byId("live-stream-state");
    const liveHint = byId("live-hint");
    const liveResultBody = byId("live-result-body");
    const progressBar = byId("capture-progress-bar");
    const computerControls = byId("computer-controls");
    const phoneControls = byId("phone-controls");
    const cameraDeviceSelect = byId("camera-device-select");
    const refreshCamerasBtn = byId("refresh-cameras-btn");
    const phoneSessionEmpty = byId("phone-session-empty");
    const phoneSessionDetails = byId("phone-session-details");
    const phoneQr = byId("phone-qr");
    const phoneJoinLink = byId("phone-join-link");
    const phoneAddressMode = byId("phone-address-mode");
    const phoneFallbackWrap = byId("phone-fallback-wrap");
    const phoneFallbackLink = byId("phone-fallback-link");
    const copyPhoneLinkBtn = byId("copy-phone-link-btn");
    const phoneSessionExpiry = byId("phone-session-expiry");

    let selectedFile = null;
    let objectUrl = null;
    let isProcessing = false;

    const live = {
        source: "upload",
        revision: 0,
        stream: null,
        active: false,
        waiters: new Set(),
        analyzing: false,
        pendingWindow: null,
        windowNumber: 0,
        consecutiveHazards: 0,
        phoneSession: null,
        signalingSocket: null,
        peerConnection: null,
        pendingIce: []
    };

    function getExtension(filename) {
        const index = filename.lastIndexOf(".");
        return index === -1 ? "" : filename.slice(index).toLowerCase();
    }

    function formatBytes(bytes) {
        if (bytes < 1024) return bytes + " B";
        if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
        return (bytes / (1024 * 1024)).toFixed(1) + " MB";
    }

    function formatPercent(value) {
        const number = Number(value);
        return Number.isFinite(number) ? (number * 100).toFixed(1) + "%" : "-";
    }

    function escapeHtml(value) {
        return String(value)
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;");
    }

    async function extractErrorMessage(response) {
        try {
            const payload = await response.json();
            if (typeof payload.detail === "string") return payload.detail;
            if (payload.detail) return JSON.stringify(payload.detail);
        } catch (_) {
            // Fall back to the HTTP status text.
        }
        return response.statusText || "Unexpected server error.";
    }

    function buildResultCardHtml(data, subtitle) {
        const status = String(data.status || "Unknown");
        const hazard = status.toLowerCase() === "hazard";
        const actions = Array.isArray(data.detected_actions)
            ? data.detected_actions.slice().sort((a, b) => Number(b.confidence) - Number(a.confidence))
            : [];
        const actionHtml = actions.length
            ? '<ul class="action-list">' + actions.map((item) => (
                '<li class="action-item"><div class="action-item-head"><strong>' +
                escapeHtml(item.action) + "</strong><span>" + formatPercent(item.confidence) +
                "</span></div></li>"
            )).join("") + "</ul>"
            : '<p class="no-actions">No workplace actions were returned.</p>';

        return [
            '<article class="result-card">',
            '  <div class="result-banner ' + (hazard ? "hazard" : "safe") + '">',
            '    <p class="status-kicker">' + escapeHtml(subtitle || "Analysis result") + "</p>",
            '    <h3 class="status-title">' + escapeHtml(status.toUpperCase()) + "</h3>",
            '    <p class="status-copy">' + (hazard ? "A possible workplace hazard was detected." : "No hazard was confirmed in this window.") + "</p>",
            "  </div>",
            '  <div class="confidence-block">',
            '    <div class="confidence-row"><span>Hazard confidence</span><strong>' + formatPercent(data.hazard_confidence) + "</strong></div>",
            '    <div class="meter"><div class="meter-fill ' + (hazard ? "hazard" : "safe") + '" style="width:' +
                Math.max(0, Math.min(100, Number(data.hazard_confidence || 0) * 100)) + '%"></div></div>',
            "  </div>",
            '  <div class="actions-block"><h3>Detected actions</h3>' + actionHtml + "</div>",
            "</article>"
        ].join("");
    }

    function renderError(container, title, message) {
        container.innerHTML = '<div class="error-card" role="alert"><h3>' +
            escapeHtml(title) + "</h3><p>" + escapeHtml(message) + "</p></div>";
    }

    function renderLoading(container, title) {
        container.innerHTML = '<div class="loading-state"><div class="spinner" aria-hidden="true"></div><div><strong>' +
            escapeHtml(title) + '</strong><p class="no-actions">The model is analyzing the captured frames.</p></div></div>';
    }

    // ------------------------------------------------------------------
    // Upload mode: the file and /predict/ contract remain unchanged.
    // ------------------------------------------------------------------
    function revokePreview() {
        if (objectUrl) URL.revokeObjectURL(objectUrl);
        objectUrl = null;
        videoPreview.removeAttribute("src");
        videoPreview.load();
    }

    function resetUpload() {
        selectedFile = null;
        fileInput.value = "";
        revokePreview();
        fileMeta.hidden = true;
        previewWrap.hidden = true;
        previewFallback.hidden = true;
        analyzeBtn.disabled = true;
        clearBtn.hidden = true;
        formHint.textContent = "";
    }

    function selectUpload(file) {
        formHint.textContent = "";
        if (!file) return;
        if (!ALLOWED_EXTENSIONS.includes(getExtension(file.name))) {
            resetUpload();
            formHint.textContent = "Supported formats: MP4, AVI, MOV.";
            return;
        }
        selectedFile = file;
        fileNameEl.textContent = file.name;
        fileSizeEl.textContent = formatBytes(file.size);
        fileMeta.hidden = false;
        clearBtn.hidden = false;
        analyzeBtn.disabled = false;
        revokePreview();
        objectUrl = URL.createObjectURL(file);
        videoPreview.src = objectUrl;
        previewWrap.hidden = false;
        videoPreview.onerror = () => { previewFallback.hidden = false; };
        videoPreview.onloadedmetadata = () => { previewFallback.hidden = true; };
    }

    async function analyzeUpload() {
        if (!selectedFile || isProcessing) return;
        isProcessing = true;
        analyzeBtn.disabled = true;
        browseBtn.disabled = true;
        clearBtn.disabled = true;
        analyzeLabel.textContent = "Analyzing...";
        renderLoading(resultBody, "Running video analysis");
        const body = new FormData();
        body.append("file", selectedFile);
        try {
            const response = await fetch(PREDICT_URL, { method: "POST", body });
            if (!response.ok) throw new Error(await extractErrorMessage(response));
            resultBody.innerHTML = buildResultCardHtml(await response.json(), "Video analysis result");
        } catch (error) {
            renderError(resultBody, "Video analysis failed", error.message);
        } finally {
            isProcessing = false;
            analyzeBtn.disabled = !selectedFile;
            browseBtn.disabled = false;
            clearBtn.disabled = false;
            analyzeLabel.textContent = "Analyze video";
        }
    }

    browseBtn.addEventListener("click", (event) => { event.stopPropagation(); fileInput.click(); });
    dropzone.addEventListener("click", () => { if (!isProcessing) fileInput.click(); });
    dropzone.addEventListener("keydown", (event) => {
        if ((event.key === "Enter" || event.key === " ") && !isProcessing) fileInput.click();
    });
    ["dragenter", "dragover"].forEach((type) => dropzone.addEventListener(type, (event) => {
        event.preventDefault();
        dropzone.classList.add("is-dragover");
    }));
    ["dragleave", "drop"].forEach((type) => dropzone.addEventListener(type, (event) => {
        event.preventDefault();
        dropzone.classList.remove("is-dragover");
    }));
    dropzone.addEventListener("drop", (event) => selectUpload(event.dataTransfer.files && event.dataTransfer.files[0]));
    fileInput.addEventListener("change", () => selectUpload(fileInput.files && fileInput.files[0]));
    analyzeBtn.addEventListener("click", analyzeUpload);
    clearBtn.addEventListener("click", resetUpload);

    // ------------------------------------------------------------------
    // Shared five-second live capture.
    // ------------------------------------------------------------------
    function setLiveBadge(type, label) {
        liveStatusBadge.className = "status-badge badge-" + type;
        liveStatusBadge.textContent = label;
    }

    function setLiveHint(message, error) {
        liveHint.textContent = message || "";
        liveHint.style.color = error ? "var(--hazard)" : "var(--text-muted)";
    }

    function resetLiveResult() {
        liveResultBody.innerHTML = '<div class="empty-state"><div class="empty-icon" aria-hidden="true"></div>' +
            '<h3>Waiting for a five-second window</h3><p>Start a camera source to collect 16 frames and run analysis.</p></div>';
    }

    function cancelWaiters() {
        live.waiters.forEach((waiter) => {
            clearTimeout(waiter.timer);
            waiter.resolve(false);
        });
        live.waiters.clear();
    }

    function waitUntil(targetTime, revision) {
        const delay = Math.max(0, targetTime - performance.now());
        return new Promise((resolve) => {
            const waiter = { timer: null, resolve };
            waiter.timer = setTimeout(() => {
                live.waiters.delete(waiter);
                resolve(live.active && live.revision === revision);
            }, delay);
            live.waiters.add(waiter);
        });
    }

    function captureFrameBlob() {
        return new Promise((resolve) => {
            if (!liveVideo.videoWidth || !liveVideo.videoHeight) return resolve(null);
            const targetHeight = 240;
            const targetWidth = Math.max(1, Math.round((liveVideo.videoWidth / liveVideo.videoHeight) * targetHeight));
            liveCanvas.width = targetWidth;
            liveCanvas.height = targetHeight;
            const context = liveCanvas.getContext("2d");
            if (!context) return resolve(null);
            context.drawImage(liveVideo, 0, 0, targetWidth, targetHeight);
            liveCanvas.toBlob(resolve, "image/jpeg", 0.85);
        });
    }

    async function captureFiveSecondWindow(revision) {
        const frames = [];
        const startedAt = performance.now();
        const interval = LIVE_WINDOW_MS / (LIVE_FRAME_COUNT - 1);
        for (let index = 0; index < LIVE_FRAME_COUNT; index += 1) {
            if (index > 0 && !await waitUntil(startedAt + interval * index, revision)) return null;
            if (!live.active || live.revision !== revision) return null;
            const frame = await captureFrameBlob();
            if (!frame) return null;
            frames.push(frame);
            const elapsed = Math.min(LIVE_WINDOW_MS, performance.now() - startedAt);
            liveBufferCounter.textContent = frames.length + " / " + LIVE_FRAME_COUNT + " frames · " +
                (elapsed / 1000).toFixed(1) + " / 5.0s";
            progressBar.style.width = Math.min(100, elapsed / LIVE_WINDOW_MS * 100) + "%";
        }
        return {
            frames,
            durationMs: Math.round(performance.now() - startedAt),
            number: ++live.windowNumber
        };
    }

    async function analyzeLiveWindow(windowData, revision) {
        if (!live.active || revision !== live.revision) return;
        live.analyzing = true;
        const body = new FormData();
        windowData.frames.forEach((frame, index) => body.append("frames", frame, "frame_" + index + ".jpg"));
        body.append("source_type", live.source);
        body.append("captured_duration_ms", String(windowData.durationMs));
        body.append("window_id", String(windowData.number));
        liveStreamState.textContent = "Analyzing window #" + windowData.number;

        try {
            const response = await fetch(LIVE_PREDICT_URL, { method: "POST", body });
            if (!live.active || revision !== live.revision) return;
            if (response.status === 429) {
                setLiveHint("Model is busy. The newest completed window will be used next.", false);
                return;
            }
            if (!response.ok) throw new Error(await extractErrorMessage(response));
            const data = await response.json();
            const rawHazard = String(data.status).toLowerCase() === "hazard";
            live.consecutiveHazards = rawHazard ? live.consecutiveHazards + 1 : 0;
            const confirmedHazard = live.consecutiveHazards >= REQUIRED_CONSECUTIVE_HAZARDS;
            const displayData = Object.assign({}, data, { status: confirmedHazard ? "Hazard" : "Safe" });
            let subtitle = "Five-second window #" + windowData.number;
            if (rawHazard && !confirmedHazard) subtitle += " · potential hazard, verifying next window";
            if (confirmedHazard) subtitle += " · confirmed across consecutive windows";
            liveResultBody.innerHTML = buildResultCardHtml(displayData, subtitle);
            setLiveHint(rawHazard && !confirmedHazard
                ? "A potential hazard was detected once; one more consecutive window is required for an alert."
                : "", false);
        } catch (error) {
            if (live.active && revision === live.revision) renderError(liveResultBody, "Live analysis failed", error.message);
        } finally {
            live.analyzing = false;
            if (live.active && revision === live.revision && live.pendingWindow) {
                const newest = live.pendingWindow;
                live.pendingWindow = null;
                void analyzeLiveWindow(newest, revision);
            }
        }
    }

    function queueLiveWindow(windowData, revision) {
        if (live.analyzing) {
            live.pendingWindow = windowData;
            liveStreamState.textContent = "Recording continues · newest window queued";
            return;
        }
        void analyzeLiveWindow(windowData, revision);
    }

    async function captureLoop(revision) {
        while (live.active && revision === live.revision) {
            setLiveBadge("streaming", "Recording");
            liveStreamState.textContent = "Recording a complete five-second window";
            progressBar.style.width = "0%";
            const windowData = await captureFiveSecondWindow(revision);
            if (!windowData || !live.active || revision !== live.revision) break;
            progressBar.style.width = "100%";
            queueLiveWindow(windowData, revision);
        }
    }

    async function beginLiveStream(stream, revision) {
        if (revision !== live.revision) {
            stream.getTracks().forEach((track) => track.stop());
            return;
        }
        live.stream = stream;
        liveVideo.srcObject = stream;
        liveVideo.style.display = "block";
        livePlaceholder.style.display = "none";
        await liveVideo.play();
        live.active = true;
        liveStartBtn.disabled = true;
        liveStopBtn.disabled = false;
        setLiveBadge("warming", "Warming up");
        liveStreamState.textContent = "Stabilizing exposure and focus";
        liveBufferCounter.textContent = "Warm-up 2.5s";
        const ready = await waitUntil(performance.now() + CAMERA_WARMUP_MS, revision);
        if (!ready) return;
        void captureLoop(revision);
    }

    async function refreshCameraDevices() {
        if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) return;
        const devices = await navigator.mediaDevices.enumerateDevices();
        const current = cameraDeviceSelect.value;
        cameraDeviceSelect.innerHTML = '<option value="">Default camera</option>';
        devices.filter((device) => device.kind === "videoinput").forEach((device, index) => {
            const option = document.createElement("option");
            option.value = device.deviceId;
            option.textContent = device.label || "Camera " + (index + 1);
            cameraDeviceSelect.appendChild(option);
        });
        if ([...cameraDeviceSelect.options].some((option) => option.value === current)) cameraDeviceSelect.value = current;
    }

    async function startComputerCamera() {
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            setLiveHint("This browser does not support camera access.", true);
            return;
        }
        const revision = ++live.revision;
        setLiveBadge("streaming", "Connecting");
        try {
            const deviceId = cameraDeviceSelect.value;
            const stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    deviceId: deviceId ? { exact: deviceId } : undefined,
                    width: { ideal: 1280 },
                    height: { ideal: 720 },
                    frameRate: { ideal: 30, max: 30 }
                },
                audio: false
            });
            await refreshCameraDevices();
            await beginLiveStream(stream, revision);
        } catch (error) {
            setLiveBadge("idle", "Camera off");
            setLiveHint("Could not open the computer camera: " + (error.message || error.name), true);
        }
    }

    // ------------------------------------------------------------------
    // Phone pairing and WebRTC receiving side.
    // ------------------------------------------------------------------
    function websocketUrl(path) {
        return (location.protocol === "https:" ? "wss://" : "ws://") + location.host + path;
    }

    function sendSignal(payload) {
        if (live.signalingSocket && live.signalingSocket.readyState === WebSocket.OPEN) {
            live.signalingSocket.send(JSON.stringify(payload));
        }
    }

    async function flushPendingIce() {
        if (!live.peerConnection || !live.peerConnection.remoteDescription) return;
        const candidates = live.pendingIce.splice(0);
        for (const candidate of candidates) await live.peerConnection.addIceCandidate(candidate);
    }

    async function createHostPeerConnection() {
        if (!live.phoneSession || !live.signalingSocket) return;
        if (live.peerConnection) live.peerConnection.close();
        const revision = live.revision;
        const peer = new RTCPeerConnection({ iceServers: live.phoneSession.ice_servers || [] });
        live.peerConnection = peer;
        live.pendingIce = [];
        peer.addTransceiver("video", { direction: "recvonly" });
        peer.onicecandidate = (event) => {
            if (event.candidate) sendSignal({ type: "ice_candidate", candidate: event.candidate });
        };
        peer.ontrack = (event) => {
            const stream = event.streams[0] || new MediaStream([event.track]);
            void beginLiveStream(stream, revision);
        };
        peer.onconnectionstatechange = () => {
            if (["failed", "disconnected", "closed"].includes(peer.connectionState) && live.source === "phone_camera") {
                setLiveBadge("idle", "Phone disconnected");
                liveStreamState.textContent = "Waiting for the phone to reconnect";
            }
        };
        const offer = await peer.createOffer();
        await peer.setLocalDescription(offer);
        sendSignal({ type: "offer", sdp: peer.localDescription });
    }

    function connectHostSignaling() {
        const session = live.phoneSession;
        const path = CAMERA_SESSIONS_URL + "/ws/" + encodeURIComponent(session.session_id) +
            "?role=host&token=" + encodeURIComponent(session.host_token);
        const socket = new WebSocket(websocketUrl(path));
        live.signalingSocket = socket;
        socket.onopen = () => {
            setLiveBadge("streaming", "Waiting for phone");
            liveStreamState.textContent = "Scan the QR code on the phone";
        };
        socket.onmessage = async (event) => {
            const message = JSON.parse(event.data);
            if (message.type === "peer_joined" && message.role === "phone") {
                setLiveBadge("streaming", "Phone connected");
                await createHostPeerConnection();
            } else if (message.type === "answer" && live.peerConnection) {
                await live.peerConnection.setRemoteDescription(message.sdp);
                await flushPendingIce();
            } else if (message.type === "ice_candidate" && message.candidate) {
                if (live.peerConnection && live.peerConnection.remoteDescription) {
                    await live.peerConnection.addIceCandidate(message.candidate);
                } else {
                    live.pendingIce.push(message.candidate);
                }
            } else if (message.type === "peer_left") {
                ++live.revision;
                stopMediaOnly();
                setLiveBadge("idle", "Phone disconnected");
                liveStreamState.textContent = "Waiting for the phone to reconnect";
            }
        };
        socket.onclose = () => {
            if (live.source === "phone_camera" && live.phoneSession) setLiveHint("Phone signaling connection closed.", true);
        };
    }

    async function startPhoneCamera() {
        if (!window.RTCPeerConnection) {
            setLiveHint("This browser does not support WebRTC.", true);
            return;
        }
        if (location.protocol !== "https:" && !["localhost", "127.0.0.1"].includes(location.hostname)) {
            setLiveHint("Phone camera access requires HTTPS. Start the server with SSL_CERTFILE and SSL_KEYFILE.", true);
        }
        ++live.revision;
        liveStartBtn.disabled = true;
        setLiveBadge("streaming", "Creating session");
        try {
            const response = await fetch(CAMERA_SESSIONS_URL, { method: "POST" });
            if (!response.ok) throw new Error(await extractErrorMessage(response));
            live.phoneSession = await response.json();
            phoneQr.src = live.phoneSession.qr_url;
            phoneJoinLink.href = live.phoneSession.join_url;
            phoneJoinLink.textContent = live.phoneSession.join_url;
            phoneAddressMode.textContent = live.phoneSession.address_mode === "mdns"
                ? "Stable local hostname · no IP changes required"
                : "LAN address detected automatically";
            const hasFallback = live.phoneSession.fallback_join_url &&
                live.phoneSession.fallback_join_url !== live.phoneSession.join_url;
            phoneFallbackWrap.hidden = !hasFallback;
            if (hasFallback) phoneFallbackLink.href = live.phoneSession.fallback_join_url;
            phoneSessionEmpty.hidden = true;
            phoneSessionDetails.hidden = false;
            phoneSessionExpiry.textContent = "Link expires in 5 minutes";
            setLiveHint(
                live.phoneSession.address_mode === "mdns"
                    ? "QR uses industrial-safety.local. Both devices must be on the same network."
                    : "QR uses the laptop's current LAN IP, detected automatically.",
                false
            );
            liveStopBtn.disabled = false;
            connectHostSignaling();
        } catch (error) {
            liveStartBtn.disabled = false;
            setLiveBadge("idle", "Not connected");
            setLiveHint("Could not create phone connection: " + error.message, true);
        }
    }

    function stopMediaOnly() {
        live.active = false;
        cancelWaiters();
        if (live.stream) live.stream.getTracks().forEach((track) => track.stop());
        live.stream = null;
        liveVideo.srcObject = null;
        liveVideo.style.display = "none";
        livePlaceholder.style.display = "flex";
        live.pendingWindow = null;
        live.analyzing = false;
        progressBar.style.width = "0%";
        liveBufferCounter.textContent = "0 / 16 frames";
    }

    async function stopLiveSource() {
        const previousSession = live.phoneSession;
        ++live.revision;
        stopMediaOnly();
        if (live.peerConnection) live.peerConnection.close();
        if (live.signalingSocket) live.signalingSocket.close();
        live.peerConnection = null;
        live.signalingSocket = null;
        live.phoneSession = null;
        live.pendingIce = [];
        live.windowNumber = 0;
        live.consecutiveHazards = 0;
        phoneSessionDetails.hidden = true;
        phoneSessionEmpty.hidden = false;
        phoneFallbackWrap.hidden = true;
        phoneQr.removeAttribute("src");
        if (previousSession) {
            const url = CAMERA_SESSIONS_URL + "/" + encodeURIComponent(previousSession.session_id) +
                "?host_token=" + encodeURIComponent(previousSession.host_token);
            fetch(url, { method: "DELETE", keepalive: true }).catch(() => {});
        }
        liveStartBtn.disabled = false;
        liveStopBtn.disabled = true;
        setLiveBadge("idle", "Camera off");
        liveStreamState.textContent = "Idle";
        setLiveHint("", false);
    }

    async function switchSource(source) {
        if (source === live.source) return;
        await stopLiveSource();
        live.source = source;
        Object.entries(sourceButtons).forEach(([key, button]) => {
            const active = key === source;
            button.classList.toggle("active", active);
            button.setAttribute("aria-selected", String(active));
        });
        const upload = source === "upload";
        uploadView.hidden = !upload;
        uploadView.classList.toggle("active", upload);
        liveView.hidden = upload;
        liveView.classList.toggle("active", !upload);
        if (upload) return;

        const phone = source === "phone_camera";
        computerControls.hidden = phone;
        phoneControls.hidden = !phone;
        liveHeading.textContent = phone ? "Phone camera" : "Computer camera";
        liveDescription.textContent = phone
            ? "Scan the QR code to stream a phone camera. Analysis begins only after each complete five-second window."
            : "Preview remains live while 16 frames are recorded across exactly five seconds for each analysis.";
        liveStartLabel.textContent = phone ? "Create phone connection" : "Start camera";
        livePlaceholderTitle.textContent = phone ? "No phone connected" : "Camera is currently inactive";
        livePlaceholderSub.textContent = phone
            ? "Create a connection and scan the QR code on the phone."
            : "Click Start camera to begin real-time safety monitoring.";
        resetLiveResult();
        if (!phone) refreshCameraDevices().catch(() => {});
    }

    sourceButtons.upload.addEventListener("click", () => void switchSource("upload"));
    sourceButtons.computer_camera.addEventListener("click", () => void switchSource("computer_camera"));
    sourceButtons.phone_camera.addEventListener("click", () => void switchSource("phone_camera"));
    refreshCamerasBtn.addEventListener("click", () => refreshCameraDevices().catch((error) => setLiveHint(error.message, true)));
    liveStartBtn.addEventListener("click", () => {
        if (live.source === "computer_camera") void startComputerCamera();
        if (live.source === "phone_camera") void startPhoneCamera();
    });
    liveStopBtn.addEventListener("click", () => void stopLiveSource());
    copyPhoneLinkBtn.addEventListener("click", async () => {
        if (!live.phoneSession) return;
        await navigator.clipboard.writeText(live.phoneSession.join_url);
        copyPhoneLinkBtn.textContent = "Copied";
        setTimeout(() => { copyPhoneLinkBtn.textContent = "Copy link"; }, 1200);
    });

    window.addEventListener("beforeunload", () => {
        ++live.revision;
        stopMediaOnly();
        if (live.peerConnection) live.peerConnection.close();
        if (live.signalingSocket) live.signalingSocket.close();
    });

    resetUpload();
    resetLiveResult();
})();
