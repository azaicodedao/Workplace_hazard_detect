(function () {
    "use strict";

    // ======================================================================
    // API Endpoints & Cấu hình
    // ======================================================================
    const PREDICT_URL = "/predict/";
    const LIVE_PREDICT_URL = "/predict/live/";
    const ALLOWED_EXTENSIONS = [".mp4", ".avi", ".mov"];

    // Cấu hình Live Camera (khớp với routes.py)
    const LIVE_NUM_FRAMES = 16;
    const LIVE_SAMPLE_INTERVAL_MS = 333;   // ~3 FPS: 16 frames đại diện ~5.3 giây cửa sổ quan sát
    const WARMUP_DURATION_MS = 2500;       // 2.5s warm-up để camera ổn định auto-exposure/focus
    const REQUIRED_CONSECUTIVE_HAZARDS = 2; // Cần ít nhất 2 cửa sổ liên tiếp dự đoán Hazard để báo động

    // ======================================================================
    // DOM Elements: Tabs
    // ======================================================================
    const tabBtnVideo = document.getElementById("tab-btn-video");
    const tabBtnLive = document.getElementById("tab-btn-live");
    const tabContentVideo = document.getElementById("tab-content-video");
    const tabContentLive = document.getElementById("tab-content-live");

    // ======================================================================
    // DOM Elements: Tab 1 (Video Analysis) - Bảo toàn 100% ID cũ
    // ======================================================================
    const dropzone = document.getElementById("dropzone");
    const fileInput = document.getElementById("file-input");
    const browseBtn = document.getElementById("browse-btn");
    const analyzeBtn = document.getElementById("analyze-btn");
    const clearBtn = document.getElementById("clear-btn");
    const analyzeLabel = document.getElementById("analyze-label");
    const fileMeta = document.getElementById("file-meta");
    const fileNameEl = document.getElementById("file-name");
    const fileSizeEl = document.getElementById("file-size");
    const previewWrap = document.getElementById("preview-wrap");
    const videoPreview = document.getElementById("video-preview");
    const previewFallback = document.getElementById("preview-fallback");
    const formHint = document.getElementById("form-hint");
    const resultBody = document.getElementById("result-body");

    // ======================================================================
    // DOM Elements: Tab 2 (Live Camera)
    // ======================================================================
    const liveVideo = document.getElementById("live-video");
    const liveCanvas = document.getElementById("live-canvas");
    const livePlaceholder = document.getElementById("live-placeholder");
    const liveStartBtn = document.getElementById("live-start-btn");
    const liveStopBtn = document.getElementById("live-stop-btn");
    const liveStatusBadge = document.getElementById("live-status-badge");
    const liveBufferCounter = document.getElementById("live-buffer-counter");
    const liveStreamState = document.getElementById("live-stream-state");
    const liveHint = document.getElementById("live-hint");
    const liveResultBody = document.getElementById("live-result-body");

    // ======================================================================
    // Shared State: Video Analysis
    // ======================================================================
    let selectedFile = null;
    let objectUrl = null;
    let isProcessing = false;

    // ======================================================================
    // Shared State: Live Camera & Session Tracking
    // ======================================================================
    let cameraStream = null;
    let cameraActive = false;
    let currentSessionId = 0;          // Tăng mỗi lần Start Camera hoặc Stop để vô hiệu hóa request cũ
    let liveSampleTimer = null;
    let warmupTimer = null;            // Timer chờ camera warm-up ổn định
    let isWarmingUp = false;           // Cờ trạng thái đang trong giai đoạn warm-up
    let frameBuffer = [];              // Chứa các Blob JPEG
    let isLiveAnalyzing = false;       // Khóa gửi request để không gửi đè request mới
    let consecutiveHazardCount = 0;    // Bộ đếm số cửa sổ Hazard liên tiếp cho Temporal Smoothing
    let lastSmoothedStatus = "Safe";   // Trạng thái đã làm mượt gần nhất ("Safe" | "Hazard")

    // ======================================================================
    // Helper Functions dùng chung
    // ======================================================================
    function getExtension(filename) {
        const idx = filename.lastIndexOf(".");
        return idx === -1 ? "" : filename.slice(idx).toLowerCase();
    }

    function isAllowedVideo(file) {
        return ALLOWED_EXTENSIONS.includes(getExtension(file.name));
    }

    function formatBytes(bytes) {
        if (bytes < 1024) return bytes + " B";
        if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
        return (bytes / (1024 * 1024)).toFixed(1) + " MB";
    }

    function formatPercent(value) {
        const n = Number(value);
        if (!Number.isFinite(n)) return "—";
        return (n * 100).toFixed(1) + "%";
    }

    function escapeHtml(text) {
        return String(text)
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;");
    }

    function sortActions(actions) {
        return actions.slice().sort(function (a, b) {
            return Number(b.confidence) - Number(a.confidence);
        });
    }

    async function extractErrorMessage(response) {
        try {
            const payload = await response.json();
            if (payload && payload.detail) {
                if (typeof payload.detail === "string") return payload.detail;
                return JSON.stringify(payload.detail);
            }
        } catch (err) {
            // Không phải JSON
        }
        return response.statusText || "Unexpected server error.";
    }

    // ======================================================================
    // TAB 1: LOGIC VIDEO ANALYSIS (Bảo toàn 100% logic cũ)
    // ======================================================================
    function revokePreview() {
        if (objectUrl) {
            URL.revokeObjectURL(objectUrl);
            objectUrl = null;
        }
        videoPreview.removeAttribute("src");
        videoPreview.load();
    }

    function setHint(message) {
        formHint.textContent = message || "";
    }

    function setProcessing(processing) {
        isProcessing = processing;
        analyzeBtn.disabled = processing || !selectedFile;
        browseBtn.disabled = processing;
        clearBtn.disabled = processing;
        dropzone.setAttribute("aria-disabled", processing ? "true" : "false");
        analyzeLabel.textContent = processing ? "Analyzing…" : "Analyze video";
    }

    function renderEmpty() {
        resultBody.innerHTML = [
            '<div class="empty-state" id="empty-state">',
            '  <div class="empty-icon" aria-hidden="true"></div>',
            "  <h3>No analysis yet</h3>",
            "  <p>Select a video and run analysis to see hazard status, confidence, and detected workplace actions.</p>",
            "</div>"
        ].join("");
    }

    function renderLoading() {
        resultBody.innerHTML = [
            '<div class="loading-state">',
            '  <div class="spinner" aria-hidden="true"></div>',
            "  <div>",
            "    <strong>Running hazard detection</strong>",
            "    <p class=\"no-actions\">The model is analyzing sampled frames. This can take a moment.</p>",
            "  </div>",
            "</div>"
        ].join("");
    }

    function renderError(title, message) {
        const safeTitle = escapeHtml(title);
        const safeMessage = escapeHtml(message);
        resultBody.innerHTML = [
            '<div class="error-card" role="alert">',
            "  <h3>" + safeTitle + "</h3>",
            "  <p>" + safeMessage + "</p>",
            "</div>"
        ].join("");
    }

    function buildResultCardHtml(data, subtitle) {
        const isHazard = data.status === "Hazard";
        const bannerClass = isHazard ? "hazard" : "safe";
        const statusLabel = isHazard ? "HAZARD DETECTED" : "SAFE";
        const copy = isHazard
            ? "An unsafe workplace condition was detected. Review the actions below and follow site safety procedure."
            : "No hazard was detected. Workplace actions detected by the model are listed below.";

        const actions = Array.isArray(data.detected_actions)
            ? sortActions(data.detected_actions)
            : [];

        const actionsHtml = actions.length
            ? '<ul class="action-list">' +
              actions.map(function (item) {
                  const name = escapeHtml(item.action);
                  const pct = formatPercent(item.confidence);
                  const width = Math.max(0, Math.min(100, Number(item.confidence) * 100));
                  return [
                      '<li class="action-item">',
                      '  <div class="action-item-head">',
                      "    <span>" + name + "</span>",
                      "    <span>" + pct + "</span>",
                      "  </div>",
                      '  <div class="meter"><div class="meter-fill ' + bannerClass + '" style="width:' + width + '%"></div></div>',
                      "</li>"
                  ].join("");
              }).join("") +
              "</ul>"
            : '<p class="no-actions">No actions crossed the detection threshold.</p>';

        const confPct = formatPercent(data.hazard_confidence);
        const confWidth = Math.max(0, Math.min(100, Number(data.hazard_confidence) * 100));

        return [
            '<article class="result-card">',
            '  <div class="result-banner ' + bannerClass + '">',
            '    <p class="status-kicker">' + escapeHtml(subtitle || "Model status") + "</p>",
            '    <h3 class="status-title">' + statusLabel + "</h3>",
            '    <p class="status-copy">' + copy + "</p>",
            "  </div>",
            '  <div class="confidence-block">',
            '    <div class="confidence-row">',
            "      <span>Hazard confidence</span>",
            "      <strong>" + confPct + "</strong>",
            "    </div>",
            '    <div class="meter"><div class="meter-fill ' + bannerClass + '" style="width:' + confWidth + '%"></div></div>',
            "  </div>",
            '  <div class="actions-block">',
            "    <h3>Detected actions</h3>",
            actionsHtml,
            "  </div>",
            "</article>"
        ].join("");
    }

    function renderResult(data) {
        resultBody.innerHTML = buildResultCardHtml(data, "Video analysis result");
    }

    function resetSelection() {
        selectedFile = null;
        fileInput.value = "";
        fileMeta.hidden = true;
        previewWrap.hidden = true;
        previewFallback.hidden = true;
        clearBtn.hidden = true;
        revokePreview();
        setHint("");
        setProcessing(false);
        renderEmpty();
    }

    function showPreview(file) {
        revokePreview();
        objectUrl = URL.createObjectURL(file);
        previewWrap.hidden = false;
        previewFallback.hidden = true;
        videoPreview.hidden = false;
        videoPreview.src = objectUrl;

        videoPreview.onerror = function () {
            videoPreview.hidden = true;
            previewFallback.hidden = false;
        };
        videoPreview.onloadeddata = function () {
            videoPreview.hidden = false;
            previewFallback.hidden = true;
        };
    }

    function applySelectedFile(file) {
        if (!file) return;

        if (!isAllowedVideo(file)) {
            setHint("Only .mp4, .avi, and .mov files are supported.");
            return;
        }

        selectedFile = file;
        fileNameEl.textContent = file.name;
        fileSizeEl.textContent = formatBytes(file.size);
        fileMeta.hidden = false;
        clearBtn.hidden = false;
        setHint("");
        analyzeBtn.disabled = false;
        showPreview(file);
        renderEmpty();
    }

    async function analyzeVideo() {
        if (!selectedFile || isProcessing) return;

        setProcessing(true);
        setHint("");
        renderLoading();

        const formData = new FormData();
        formData.append("file", selectedFile);

        try {
            const response = await fetch(PREDICT_URL, {
                method: "POST",
                body: formData
            });

            if (!response.ok) {
                const message = await extractErrorMessage(response);
                const title = response.status === 400
                    ? "Invalid request"
                    : response.status === 500
                        ? "Server error"
                        : "Request failed (" + response.status + ")";
                renderError(title, message);
                return;
            }

            const data = await response.json();
            if (!data || typeof data.status === "undefined") {
                renderError("Unexpected response", "The server did not return a valid prediction result.");
                return;
            }

            renderResult(data);
        } catch (err) {
            renderError(
                "Network error",
                "Could not reach the analysis service. Check that the backend is running and try again."
            );
        } finally {
            setProcessing(false);
        }
    }

    function onDropzoneActivate(event) {
        if (isProcessing) return;
        if (event.target === browseBtn) return;
        fileInput.click();
    }

    dropzone.addEventListener("click", onDropzoneActivate);
    dropzone.addEventListener("keydown", function (event) {
        if (event.key === "Enter" || event.key === " ") {
            event.preventDefault();
            onDropzoneActivate(event);
        }
    });

    browseBtn.addEventListener("click", function (event) {
        event.stopPropagation();
        if (!isProcessing) fileInput.click();
    });

    ["dragenter", "dragover"].forEach(function (type) {
        dropzone.addEventListener(type, function (event) {
            event.preventDefault();
            dropzone.classList.add("is-dragover");
        });
    });

    ["dragleave", "drop"].forEach(function (type) {
        dropzone.addEventListener(type, function (event) {
            event.preventDefault();
            dropzone.classList.remove("is-dragover");
        });
    });

    dropzone.addEventListener("drop", function (event) {
        if (isProcessing) return;
        const file = event.dataTransfer.files && event.dataTransfer.files[0];
        applySelectedFile(file);
    });

    fileInput.addEventListener("change", function () {
        applySelectedFile(fileInput.files && fileInput.files[0]);
    });

    analyzeBtn.addEventListener("click", analyzeVideo);
    clearBtn.addEventListener("click", resetSelection);

    // ======================================================================
    // TAB 2: LOGIC LIVE CAMERA (Webcam + Warm-up + Temporal Smoothing)
    // ======================================================================
    function setLiveBadge(type, label) {
        liveStatusBadge.className = "status-badge badge-" + type;
        liveStatusBadge.textContent = label;
    }

    function setLiveHint(message, isError) {
        liveHint.textContent = message || "";
        liveHint.style.color = isError ? "var(--hazard)" : "var(--text-muted)";
    }

    function renderLiveEmpty() {
        liveResultBody.innerHTML = [
            '<div class="empty-state" id="live-empty-state">',
            '  <div class="empty-icon" aria-hidden="true"></div>',
            "  <h3>Waiting for live stream</h3>",
            "  <p>Start the camera to monitor workplace hazards and view real-time classification results.</p>",
            "</div>"
        ].join("");
    }

    function renderLiveResult(data, subtitleText) {
        const subtitle = subtitleText || "Live window";
        liveResultBody.innerHTML = buildResultCardHtml(data, subtitle);
    }

    function renderLiveError(title, message) {
        const safeTitle = escapeHtml(title);
        const safeMessage = escapeHtml(message);
        liveResultBody.innerHTML = [
            '<div class="error-card" role="alert">',
            "  <h3>" + safeTitle + "</h3>",
            "  <p>" + safeMessage + "</p>",
            "</div>"
        ].join("");
    }

    // Chụp frame từ video hiện tại sang JPEG Blob (giữ nguyên tỷ lệ gốc của video)
    function captureFrameBlob() {
        return new Promise(function (resolve) {
            if (!liveVideo || liveVideo.videoWidth === 0 || liveVideo.videoHeight === 0) {
                return resolve(null);
            }
            const vw = liveVideo.videoWidth || 640;
            const vh = liveVideo.videoHeight || 480;
            const targetHeight = 240;
            const targetWidth = Math.round((vw / vh) * targetHeight);
            liveCanvas.width = targetWidth;
            liveCanvas.height = targetHeight;
            const ctx = liveCanvas.getContext("2d");
            if (!ctx) return resolve(null);

            ctx.drawImage(liveVideo, 0, 0, targetWidth, targetHeight);
            liveCanvas.toBlob(
                function (blob) {
                    resolve(blob);
                },
                "image/jpeg",
                0.85
            );
        });
    }

    // Gửi 16 frames lên backend với Temporal Smoothing
    async function triggerLiveInference(framesToSend, sessionId) {
        if (isLiveAnalyzing) return;
        if (sessionId !== currentSessionId || !cameraActive) return;

        isLiveAnalyzing = true;
        setLiveBadge("analyzing", "Analyzing...");
        liveStreamState.textContent = "Forward pass in progress...";

        const formData = new FormData();
        for (let i = 0; i < framesToSend.length; i++) {
            formData.append("frames", framesToSend[i], "frame_" + i + ".jpg");
        }

        try {
            const response = await fetch(LIVE_PREDICT_URL, {
                method: "POST",
                body: formData
            });

            // Session guard: nếu camera đã dừng hoặc session thay đổi trong lúc request chạy
            if (sessionId !== currentSessionId || !cameraActive) {
                console.log("[LiveCamera] Discarding response from outdated session:", sessionId);
                return;
            }

            if (response.status === 429) {
                // Model đang bận (429) -> Không tính là Safe hoặc Hazard, bảo toàn bộ đếm smoothing
                console.log("[LiveCamera] HTTP 429 received (model busy). Consecutive Hazard count preserved at: " + consecutiveHazardCount);
                setLiveBadge("busy", "Model Busy");
                liveStreamState.textContent = "Model busy (HTTP 429), retrying next window...";
                setLiveHint("GPU/Model is currently processing another task. Skipping this window.", false);
                return;
            }

            if (!response.ok) {
                const errMsg = await extractErrorMessage(response);
                console.log("[LiveCamera] HTTP error (" + response.status + "): " + errMsg + ". Consecutive Hazard count preserved at: " + consecutiveHazardCount);
                liveStreamState.textContent = "Error (" + response.status + ")";
                renderLiveError("Live Analysis Error", errMsg);
                return;
            }

            const data = await response.json();
            if (sessionId !== currentSessionId || !cameraActive) return;

            const rawStatus = data.status; // "Safe" hoặc "Hazard"
            const confidence = data.hazard_confidence;

            console.log("[LiveCamera] Window analysis finished.");
            console.log("[LiveCamera] -> Raw backend prediction: status=" + rawStatus + ", confidence=" + confidence);

            // ==============================================================
            // Temporal Smoothing Logic:
            // Chỉ cảnh báo Hazard khi có ít nhất 2 cửa sổ liên tiếp dự đoán Hazard.
            // ==============================================================
            if (rawStatus === "Hazard") {
                consecutiveHazardCount++;
            } else {
                consecutiveHazardCount = 0;
            }

            console.log(
                "[LiveCamera] -> Consecutive Hazard count: " + consecutiveHazardCount + 
                " / " + REQUIRED_CONSECUTIVE_HAZARDS + " required"
            );

            let smoothedStatus = "Safe";
            if (consecutiveHazardCount >= REQUIRED_CONSECUTIVE_HAZARDS) {
                smoothedStatus = "Hazard";
            } else {
                smoothedStatus = "Safe";
            }
            lastSmoothedStatus = smoothedStatus;

            console.log("[LiveCamera] -> Displayed status (smoothed): " + smoothedStatus);

            const displayData = {
                status: smoothedStatus,
                hazard_confidence: confidence,
                detected_actions: data.detected_actions || []
            };

            const now = new Date();
            const timeStr = now.toLocaleTimeString();

            // Hiển thị thông điệp ngữ cảnh rõ ràng
            let subtitle = "Live window (" + timeStr + ")";
            if (rawStatus === "Hazard" && smoothedStatus === "Safe") {
                subtitle = "Live window (" + timeStr + ") - Single hazard window (1/" + REQUIRED_CONSECUTIVE_HAZARDS + "), verifying...";
                setLiveHint("Potential hazard detected in 1 window. Verifying next window before triggering alert.", false);
            } else if (smoothedStatus === "Hazard") {
                subtitle = "Live window (" + timeStr + ") - CONFIRMED HAZARD (" + consecutiveHazardCount + " consecutive windows)";
                setLiveHint("Warning: Workplace hazard confirmed across multiple consecutive windows!", true);
            } else {
                setLiveHint("", false);
            }

            renderLiveResult(displayData, subtitle);
            liveStreamState.textContent = "Result updated: " + smoothedStatus + " at " + timeStr;
        } catch (err) {
            if (sessionId !== currentSessionId || !cameraActive) return;
            console.log("[LiveCamera] Network error: " + err.message + ". Consecutive Hazard count preserved at: " + consecutiveHazardCount);
            liveStreamState.textContent = "Network error";
            setLiveHint("Network error while sending live frames to backend.", true);
        } finally {
            isLiveAnalyzing = false;
            if (cameraActive && sessionId === currentSessionId && !isWarmingUp) {
                setLiveBadge("streaming", "Monitoring");
            }
        }
    }

    // Vòng lặp lấy mẫu frames
    async function sampleLiveFrame(sessionId) {
        if (!cameraActive || sessionId !== currentSessionId || isWarmingUp) return;

        const blob = await captureFrameBlob();
        if (!blob || !cameraActive || sessionId !== currentSessionId || isWarmingUp) return;

        frameBuffer.push(blob);
        liveBufferCounter.textContent = frameBuffer.length + " / " + LIVE_NUM_FRAMES + " frames";
        console.log("[LiveCamera] Collected frame: " + frameBuffer.length + " / " + LIVE_NUM_FRAMES);

        // Khi buffer đạt đủ 16 frames
        if (frameBuffer.length >= LIVE_NUM_FRAMES) {
            const snapshot = frameBuffer.slice(0, LIVE_NUM_FRAMES);
            
            // Giữ lại sliding window (trượt 8 frames để cửa sổ kế tiếp chỉ cần 8 frame mới = ~2.6s)
            frameBuffer = frameBuffer.slice(8);

            // Gửi snapshot nếu không đang bận inference
            if (!isLiveAnalyzing) {
                triggerLiveInference(snapshot, sessionId);
            } else {
                liveStreamState.textContent = "Analyzing previous window...";
                console.log("[LiveCamera] Previous inference still in progress, waiting for next cycle.");
            }
        }
    }

    async function startCamera() {
        if (cameraActive) return;

        setLiveHint("");
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            setLiveHint("Browser does not support camera access (getUserMedia). Please use Chrome, Edge, or Firefox.", true);
            return;
        }

        try {
            currentSessionId++;
            const thisSession = currentSessionId;

            setLiveBadge("streaming", "Connecting...");
            liveStreamState.textContent = "Requesting camera permission...";

            cameraStream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: { ideal: 640 },
                    height: { ideal: 480 },
                    facingMode: "user"
                },
                audio: false
            });

            // Nếu người dùng đã nhấn Stop trong lúc chờ quyền
            if (thisSession !== currentSessionId) {
                cameraStream.getTracks().forEach(function (t) { t.stop(); });
                cameraStream = null;
                return;
            }

            cameraActive = true;
            isWarmingUp = true;
            consecutiveHazardCount = 0;
            lastSmoothedStatus = "Safe";
            frameBuffer = [];
            renderLiveEmpty();

            liveVideo.srcObject = cameraStream;
            liveVideo.style.display = "block";
            livePlaceholder.style.display = "none";

            liveStartBtn.disabled = true;
            liveStopBtn.disabled = false;

            // Warm-up UI
            setLiveBadge("warming", "Warming Up");
            liveStreamState.textContent = "Camera warming up (stabilizing image)...";
            liveBufferCounter.textContent = "Stabilizing (2.5s)...";
            setLiveHint("Camera is warming up to stabilize auto-exposure and focus.", false);

            console.log("[LiveCamera] Camera started (Session #" + thisSession + ") at " + new Date().toISOString() + ". Warming up for " + (WARMUP_DURATION_MS / 1000) + "s...");

            // Bắt đầu timer warm-up; chỉ sau khi warm-up xong mới kích hoạt lấy mẫu frame
            warmupTimer = setTimeout(function () {
                if (thisSession !== currentSessionId || !cameraActive) return;
                isWarmingUp = false;
                warmupTimer = null;

                console.log("[LiveCamera] Warm-up complete at " + new Date().toISOString() + ". Starting frame sampling at 3 FPS...");
                setLiveBadge("streaming", "Monitoring");
                liveStreamState.textContent = "Collecting frames (3 FPS)...";
                liveBufferCounter.textContent = "0 / " + LIVE_NUM_FRAMES + " frames";
                setLiveHint("", false);

                // Bắt đầu timer lấy mẫu theo LIVE_SAMPLE_INTERVAL_MS (333ms / 3 FPS)
                liveSampleTimer = setInterval(function () {
                    sampleLiveFrame(thisSession);
                }, LIVE_SAMPLE_INTERVAL_MS);
            }, WARMUP_DURATION_MS);

        } catch (err) {
            console.error("Camera access error:", err);
            cameraActive = false;
            currentSessionId++;
            setLiveBadge("idle", "Camera Off");
            liveStreamState.textContent = "Camera unavailable";

            if (err.name === "NotAllowedError" || err.name === "PermissionDeniedError") {
                setLiveHint("Camera permission was denied. Please allow camera access in browser settings.", true);
            } else if (err.name === "NotFoundError" || err.name === "DevicesNotFoundError") {
                setLiveHint("No camera device found on this system.", true);
            } else if (err.name === "NotReadableError" || err.name === "TrackStartError") {
                setLiveHint("Camera is already in use by another application.", true);
            } else {
                setLiveHint("Could not access camera: " + (err.message || err.name), true);
            }
        }
    }

    function stopCamera() {
        console.log("[LiveCamera] Camera stopped (Session #" + currentSessionId + ") at " + new Date().toISOString());
        cameraActive = false;
        currentSessionId++; // Vô hiệu hóa mọi request pending của session vừa rồi

        if (warmupTimer) {
            clearTimeout(warmupTimer);
            warmupTimer = null;
        }

        if (liveSampleTimer) {
            clearInterval(liveSampleTimer);
            liveSampleTimer = null;
        }

        if (cameraStream) {
            cameraStream.getTracks().forEach(function (track) {
                track.stop();
            });
            cameraStream = null;
        }

        if (liveVideo) {
            liveVideo.srcObject = null;
            liveVideo.style.display = "none";
        }

        // Reset toàn bộ trạng thái warm-up và smoothing
        isWarmingUp = false;
        frameBuffer = [];
        isLiveAnalyzing = false;
        consecutiveHazardCount = 0;
        lastSmoothedStatus = "Safe";

        // Xóa hoàn toàn trạng thái Hazard cũ trên giao diện
        renderLiveEmpty();
        setLiveHint("", false);

        livePlaceholder.style.display = "flex";
        liveStartBtn.disabled = false;
        liveStopBtn.disabled = true;
        setLiveBadge("idle", "Camera Off");
        liveBufferCounter.textContent = "0 / " + LIVE_NUM_FRAMES + " frames";
        liveStreamState.textContent = "Stopped";
    }

    liveStartBtn.addEventListener("click", startCamera);
    liveStopBtn.addEventListener("click", stopCamera);

    // ======================================================================
    // TAB SWITCHING MANAGEMENT
    // ======================================================================
    function switchTab(target) {
        if (target === "video") {
            // Khi chuyển sang tab Video: dừng camera để giải phóng tài nguyên
            if (cameraActive) {
                stopCamera();
            }
            tabBtnVideo.classList.add("active");
            tabBtnVideo.setAttribute("aria-selected", "true");
            tabBtnLive.classList.remove("active");
            tabBtnLive.setAttribute("aria-selected", "false");

            tabContentVideo.classList.add("active");
            tabContentVideo.hidden = false;
            tabContentLive.classList.remove("active");
            tabContentLive.hidden = true;
        } else if (target === "live") {
            tabBtnLive.classList.add("active");
            tabBtnLive.setAttribute("aria-selected", "true");
            tabBtnVideo.classList.remove("active");
            tabBtnVideo.setAttribute("aria-selected", "false");

            tabContentLive.classList.add("active");
            tabContentLive.hidden = false;
            tabContentVideo.classList.remove("active");
            tabContentVideo.hidden = true;
        }
    }

    tabBtnVideo.addEventListener("click", function () {
        switchTab("video");
    });

    tabBtnLive.addEventListener("click", function () {
        switchTab("live");
    });

    // Dọn dẹp media streams khi người dùng đóng trang hoặc reload
    window.addEventListener("beforeunload", function () {
        stopCamera();
    });

    // Khởi tạo trạng thái ban đầu
    renderEmpty();
    renderLiveEmpty();
})();
