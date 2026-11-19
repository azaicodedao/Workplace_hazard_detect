(function () {
    "use strict";

    const PREDICT_URL = "/predict/";
    const PREDICT_TIMELINE_URL = "/predict/timeline/";
    const PREDICT_TIMELINE_STREAM_URL = "/predict/timeline/stream";
    const LIVE_PREDICT_URL = "/predict/live/";
    const CAMERA_SESSIONS_URL = "/api/camera-sessions";
    const ALLOWED_EXTENSIONS = [".mp4", ".avi", ".mov"];
    const UPLOAD_WINDOW_SECONDS = 5;
    const UPLOAD_STRIDE_SECONDS = 4;
    const LIVE_FRAME_COUNT = 16;
    const LIVE_WINDOW_MS = 5000;
    const LIVE_STRIDE_MS = 4000;
    const LIVE_SAMPLE_INTERVAL_MS = LIVE_WINDOW_MS / (LIVE_FRAME_COUNT - 1);
    const LIVE_CAPTURE_MAX_HEIGHT = 360;
    const LIVE_JPEG_QUALITY = 0.92;
    const LIVE_REQUEST_TIMEOUT_MS = 25000;
    const CAMERA_WARMUP_MS = 2500;
    const POSE_FRAME_INTERVAL_MS = 100;
    const POSE_CAPTURE_MAX_WIDTH = 640;
    const POSE_JPEG_QUALITY = 0.82;
    const POSE_SKELETON = [
        [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],
        [5, 11], [6, 12], [11, 12], [11, 13], [13, 15], [12, 14], [14, 16]
    ];

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
    const poseCaptureCanvas = byId("pose-capture-canvas");
    const poseOverlay = byId("pose-overlay");
    const poseTrackingState = byId("pose-tracking-state");
    const poseAlert = byId("pose-alert");
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
    const liveCurrentWindow = byId("live-current-window");
    const liveAnalysisWindow = byId("live-analysis-window");
    const liveLastResultWindow = byId("live-last-result-window");
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
        frameBuffer: [],
        nextWindowDueAt: null,
        windowNumber: 0,
        poseSocket: null,
        poseTimer: null,
        poseAwaitingResult: false,
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

    function formatClock(epochMilliseconds) {
        return new Date(epochMilliseconds).toLocaleTimeString("vi-VN", {
            hour12: false,
            hour: "2-digit",
            minute: "2-digit",
            second: "2-digit"
        });
    }

    function formatDelay(milliseconds) {
        return (Math.max(0, milliseconds) / 1000).toFixed(2) + "s";
    }

    function formatVideoTime(seconds) {
        const safeSeconds = Math.max(0, Number(seconds) || 0);
        const totalTenths = Math.round(safeSeconds * 10);
        const wholeSeconds = Math.floor(totalTenths / 10);
        const hours = Math.floor(wholeSeconds / 3600);
        const minutes = Math.floor((wholeSeconds % 3600) / 60);
        const remaining = wholeSeconds % 60;
        const fraction = totalTenths % 10;
        const base = (hours ? String(hours).padStart(2, "0") + ":" : "") +
            String(minutes).padStart(2, "0") + ":" + String(remaining).padStart(2, "0");
        return fraction ? base + "." + fraction : base;
    }

    function estimateCoarseWindowCount(durationSeconds) {
        const duration = Number(durationSeconds);
        if (!Number.isFinite(duration) || duration <= 0) return null;
        if (duration <= UPLOAD_WINDOW_SECONDS) return 1;
        const lastStart = duration - UPLOAD_WINDOW_SECONDS;
        const regularCount = Math.floor(lastStart / UPLOAD_STRIDE_SECONDS) + 1;
        const strideRatio = lastStart / UPLOAD_STRIDE_SECONDS;
        const endsExactlyOnStride = Math.abs(strideRatio - Math.round(strideRatio)) < 0.001;
        return regularCount + (endsExactlyOnStride ? 0 : 1);
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
        const observation = data.observation && typeof data.observation === "object"
            ? data.observation
            : null;
        const rawStatus = observation ? String(observation.raw_status || "Unknown") : "";
        const rawHazard = rawStatus.toLowerCase() === "hazard";
        const observationHtml = observation
            ? '<div class="result-observation-grid">' +
                '<div class="result-observation-item"><span>Raw model result</span><strong class="' +
                    (rawHazard ? "raw-hazard" : "raw-safe") + '">' + escapeHtml(rawStatus.toUpperCase()) + '</strong></div>' +
                '<div class="result-observation-item"><span>Alert decision</span><strong>' +
                    escapeHtml(observation.alert_decision || "-") + '</strong></div>' +
                '<div class="result-observation-item"><span>Captured interval</span><strong>' +
                    escapeHtml(observation.captured_interval || "-") + '</strong></div>' +
                '<div class="result-observation-item"><span>Delay after capture</span><strong>' +
                    escapeHtml(observation.delay_after_capture || "-") + '</strong></div>' +
              '</div>'
            : "";
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
            observationHtml,
            '  <div class="confidence-block">',
            '    <div class="confidence-row"><span>Hazard confidence</span><strong>' + formatPercent(data.hazard_confidence) + "</strong></div>",
            '    <div class="meter"><div class="meter-fill ' + (hazard ? "hazard" : "safe") + '" style="width:' +
                Math.max(0, Math.min(100, Number(data.hazard_confidence || 0) * 100)) + '%"></div></div>',
            "  </div>",
            '  <div class="actions-block"><h3>Detected actions</h3>' + actionHtml + "</div>",
            "</article>"
        ].join("");
    }

    function buildTimelineHtml(data) {
        const status = String(data.status || "Unknown");
        const hazard = status.toLowerCase() === "hazard";
        const events = Array.isArray(data.events) ? data.events : [];
        const windows = Array.isArray(data.windows) ? data.windows : [];
        const poseAnalysis = data.pose_analysis && typeof data.pose_analysis === "object"
            ? data.pose_analysis
            : { available: false, events: [] };
        const poseEvents = Array.isArray(poseAnalysis.events) ? poseAnalysis.events : [];
        const scan = data.scan || {};
        const eventHtml = events.length
            ? events.map((event) => {
                const actions = Array.isArray(event.detected_actions) ? event.detected_actions : [];
                const actionText = actions.length
                    ? actions.slice(0, 4).map((item) => escapeHtml(item.action)).join(", ")
                    : "No action label";
                return '<button type="button" class="timeline-event" data-seek="' +
                    Number(event.peak_time || 0) + '">' +
                    '<span class="timeline-event-number">Event #' + Number(event.event_id) + '</span>' +
                    '<strong>' + formatVideoTime(event.start_time) + " – " + formatVideoTime(event.end_time) + '</strong>' +
                    '<span>Peak ' + formatVideoTime(event.peak_time) + " · " +
                        formatPercent(event.hazard_confidence) + '</span>' +
                    '<small>' + actionText + '</small>' +
                    '<em>Play this section</em>' +
                    '</button>';
            }).join("")
            : '<p class="no-actions">No hazardous interval was detected.</p>';
        const poseEventHtml = poseEvents.length
            ? poseEvents.map((event) => (
                '<button type="button" class="timeline-event pose-fall-event" data-seek="' +
                Number(event.peak_time || event.start_time || 0) + '">' +
                '<span class="timeline-event-number">Fall #' + Number(event.event_id) +
                    ' · Person #' + Number(event.track_id) + '</span>' +
                '<strong>' + formatVideoTime(event.start_time) + " – " +
                    formatVideoTime(event.end_time) + '</strong>' +
                '<span>Peak ' + formatVideoTime(event.peak_time) + " · " +
                    formatPercent(event.confidence) + '</span>' +
                '<small>Pose state: ' + escapeHtml(event.final_state || "FALLING") + '</small>' +
                '<em>Play fall moment</em>' +
                '</button>'
            )).join("")
            : poseAnalysis.available
                ? '<p class="no-actions">YOLO Pose did not detect a fall in this video.</p>'
                : '<p class="no-actions">Pose analysis was unavailable' +
                    (poseAnalysis.error ? ': ' + escapeHtml(poseAnalysis.error) : '.') + '</p>';
        const poseMeta = poseAnalysis.available
            ? Number(poseAnalysis.processed_frames || 0) + ' sampled frames · ' +
                Number(poseAnalysis.effective_fps || 0).toFixed(1) + ' FPS'
            : 'Pose service unavailable';
        const windowRows = windows.map((window) => {
            const windowHazard = String(window.status).toLowerCase() === "hazard";
            return '<tr class="' + (windowHazard ? "window-hazard" : "") + '">' +
                '<td>' + escapeHtml(window.phase) + '</td>' +
                '<td>' + formatVideoTime(window.start_time) + " – " + formatVideoTime(window.end_time) + '</td>' +
                '<td>' + escapeHtml(window.status) + '</td>' +
                '<td>' + formatPercent(window.hazard_confidence) + '</td>' +
                '</tr>';
        }).join("");

        return [
            '<article class="result-card timeline-result">',
            '  <div class="result-banner ' + (hazard ? "hazard" : "safe") + '">',
            '    <p class="status-kicker">Temporal video analysis</p>',
            '    <h3 class="status-title">' + escapeHtml(status.toUpperCase()) + '</h3>',
            '    <p class="status-copy">Video duration ' + formatVideoTime(data.video_duration) +
                ' · ' + Number(scan.total_inferences || 0) + ' GPU inferences</p>',
            '  </div>',
            '  <div class="confidence-block">',
            '    <div class="confidence-row"><span>Highest hazard confidence</span><strong>' +
                formatPercent(data.hazard_confidence) + '</strong></div>',
            '    <div class="meter"><div class="meter-fill ' + (hazard ? "hazard" : "safe") +
                '" style="width:' + Math.max(0, Math.min(100, Number(data.hazard_confidence || 0) * 100)) + '%"></div></div>',
            '  </div>',
            '  <div class="timeline-block pose-timeline-block"><h3>Fall detection (YOLO Pose)</h3>',
            '    <p class="timeline-section-meta">' + escapeHtml(poseMeta) + '</p>',
            '    <div class="timeline-events">' + poseEventHtml + '</div>',
            '  </div>',
            '  <div class="timeline-block"><h3>VideoMAE hazard time ranges</h3><div class="timeline-events">' +
                eventHtml + '</div></div>',
            '  <details class="timeline-windows"><summary>View ' + windows.length +
                ' analyzed windows</summary><div class="timeline-table-wrap"><table><thead><tr>' +
                '<th>Phase</th><th>Time</th><th>Result</th><th>Confidence</th>' +
                '</tr></thead><tbody>' + windowRows + '</tbody></table></div></details>',
            '</article>'
        ].join("");
    }

    function buildTimelineProgressHtml(state) {
        const windows = state.windows || [];
        const hazardCount = windows.filter((item) => String(item.status).toLowerCase() === "hazard").length;
        const windowHtml = windows.length
            ? windows.map((window, index) => {
                const hazard = String(window.status).toLowerCase() === "hazard";
                const actions = Array.isArray(window.detected_actions) ? window.detected_actions : [];
                const actionText = actions.length ? escapeHtml(actions[0].action) : "No action label";
                const midpoint = (Number(window.start_time) + Number(window.end_time)) / 2;
                return '<button type="button" class="timeline-event ' + (hazard ? "" : "timeline-event-safe") +
                    '" data-seek="' + midpoint + '">' +
                    '<span class="timeline-event-number">Window #' + (index + 1) + '</span>' +
                    '<strong>' + formatVideoTime(window.start_time) + " – " + formatVideoTime(window.end_time) + '</strong>' +
                    '<span>' + escapeHtml(String(window.status).toUpperCase()) + " · " +
                        formatPercent(window.hazard_confidence) + '</span>' +
                    '<small>' + actionText + '</small>' +
                    '<em>Play window</em></button>';
            }).join("")
            : '<p class="no-actions">Waiting for the first GPU result...</p>';
        const stageName = "Coarse scan";
        return [
            '<article class="result-card timeline-result">',
            '  <div class="result-banner ' + (hazardCount ? "hazard" : "safe") + '">',
            '    <p class="status-kicker">Streaming timeline analysis</p>',
            '    <h3 class="status-title">ANALYZING</h3>',
            '    <p class="status-copy">' + stageName + ': ' + Number(state.completed || 0) +
                ' / ' + Number(state.total || 0) + ' · ' + hazardCount + ' hazard window(s)</p>',
            '  </div>',
            '  <div class="timeline-block"><h3>Results received so far</h3>',
            '    <div class="timeline-events">' + windowHtml + '</div>',
            '  </div>',
            '</article>'
        ].join("");
    }

    function bindTimelineSeekButtons() {
        resultBody.querySelectorAll("[data-seek]").forEach((button) => {
            button.addEventListener("click", () => {
                const peakTime = Number(button.dataset.seek || 0);
                videoPreview.currentTime = Math.max(0, peakTime - UPLOAD_WINDOW_SECONDS / 2);
                videoPreview.play().catch(() => {});
                videoPreview.scrollIntoView({ behavior: "smooth", block: "center" });
            });
        });
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
    // Upload mode: one streaming result per overlapping coarse window.
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
        renderLoading(resultBody, "Scanning video timeline");
        const videoDuration = Number(videoPreview.duration);
        const coarseWindowCount = estimateCoarseWindowCount(videoDuration);
        formHint.textContent = coarseWindowCount
            ? "Video " + formatVideoTime(videoDuration) + ": scanning " + coarseWindowCount +
                " overlapping window" + (coarseWindowCount === 1 ? "" : "s") + "."
            : "Scanning five-second windows with one-second overlap.";
        const body = new FormData();
        body.append("file", selectedFile);
        try {
            const response = await fetch(PREDICT_TIMELINE_STREAM_URL, { method: "POST", body });
            if (!response.ok) throw new Error(await extractErrorMessage(response));
            if (!response.body) throw new Error("This browser does not support streamed responses.");

            const state = {
                windows: [],
                stage: "coarse",
                completed: 0,
                total: coarseWindowCount || 0
            };
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = "";
            let completedResult = null;

            async function handleStreamEvent(event) {
                if (event.type === "started") {
                    state.total = Number(event.coarse_windows || 0);
                    state.completed = 0;
                    resultBody.innerHTML = buildTimelineProgressHtml(state);
                } else if (event.type === "window") {
                    state.stage = String(event.stage || state.stage);
                    state.completed = Number(event.completed || 0);
                    state.total = Number(event.total || 0);
                    state.windows.push(event.window);
                    resultBody.innerHTML = buildTimelineProgressHtml(state);
                    bindTimelineSeekButtons();
                    formHint.textContent = "Received " + event.stage + " window " +
                        state.completed + "/" + state.total + ": " +
                        formatVideoTime(event.window.start_time) + "–" +
                        formatVideoTime(event.window.end_time) + ".";
                } else if (event.type === "pose_complete") {
                    const pose = event.pose_analysis || {};
                    const falls = Array.isArray(pose.events) ? pose.events.length : 0;
                    formHint.textContent = pose.available
                        ? "Pose analysis complete: " + falls + " fall event" + (falls === 1 ? "" : "s") + "."
                        : "VideoMAE complete; pose analysis was unavailable.";
                } else if (event.type === "complete") {
                    completedResult = event.result;
                } else if (event.type === "error") {
                    throw new Error(event.detail || "Timeline analysis failed.");
                }
            }

            while (true) {
                const chunk = await reader.read();
                buffer += decoder.decode(chunk.value || new Uint8Array(), { stream: !chunk.done });
                const lines = buffer.split("\n");
                buffer = lines.pop() || "";
                for (const line of lines) {
                    if (line.trim()) await handleStreamEvent(JSON.parse(line));
                }
                if (chunk.done) break;
            }
            if (buffer.trim()) await handleStreamEvent(JSON.parse(buffer));
            if (!completedResult) throw new Error("The analysis stream ended before completion.");

            resultBody.innerHTML = buildTimelineHtml(completedResult);
            bindTimelineSeekButtons();
            formHint.textContent = "Analysis complete. Click an event to play the detected section.";
        } catch (error) {
            renderError(resultBody, "Video analysis failed", error.message);
            formHint.textContent = "";
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
            // Preserve detail without upscaling a lower-resolution camera source.
            // VideoMAE performs its own final resize/crop to 224x224.
            const scale = Math.min(1, LIVE_CAPTURE_MAX_HEIGHT / liveVideo.videoHeight);
            const targetWidth = Math.max(1, Math.round(liveVideo.videoWidth * scale));
            const targetHeight = Math.max(1, Math.round(liveVideo.videoHeight * scale));
            liveCanvas.width = targetWidth;
            liveCanvas.height = targetHeight;
            const context = liveCanvas.getContext("2d");
            if (!context) return resolve(null);
            context.drawImage(liveVideo, 0, 0, targetWidth, targetHeight);
            liveCanvas.toBlob(resolve, "image/jpeg", LIVE_JPEG_QUALITY);
        });
    }

    function clearPoseOverlay() {
        const context = poseOverlay.getContext("2d");
        if (!context) return;
        poseOverlay.width = Math.max(1, poseOverlay.clientWidth || 1);
        poseOverlay.height = Math.max(1, poseOverlay.clientHeight || 1);
        context.clearRect(0, 0, poseOverlay.width, poseOverlay.height);
    }

    function drawPoseResults(payload) {
        const sourceWidth = Number(payload.frame_width || 0);
        const sourceHeight = Number(payload.frame_height || 0);
        const displayWidth = Math.max(1, poseOverlay.clientWidth);
        const displayHeight = Math.max(1, poseOverlay.clientHeight);
        poseOverlay.width = displayWidth;
        poseOverlay.height = displayHeight;
        const context = poseOverlay.getContext("2d");
        if (!context || !sourceWidth || !sourceHeight) return;
        context.clearRect(0, 0, displayWidth, displayHeight);

        // Match the preview's object-fit: cover geometry.
        const scale = Math.max(displayWidth / sourceWidth, displayHeight / sourceHeight);
        const offsetX = (displayWidth - sourceWidth * scale) / 2;
        const offsetY = (displayHeight - sourceHeight * scale) / 2;
        const mapPoint = (point) => [offsetX + point[0] * scale, offsetY + point[1] * scale];
        const people = Array.isArray(payload.people) ? payload.people : [];

        people.forEach((person) => {
            const state = String(person.state || "STANDING");
            const color = state === "FALLEN" ? "#ff4d4d" : (state === "FALLING" ? "#f0b429" : "#2fbf71");
            const box = Array.isArray(person.box) ? person.box : [];
            const keypoints = Array.isArray(person.keypoints) ? person.keypoints : [];
            if (box.length === 4) {
                const topLeft = mapPoint([box[0], box[1]]);
                const bottomRight = mapPoint([box[2], box[3]]);
                context.strokeStyle = color;
                context.lineWidth = 2;
                context.strokeRect(
                    topLeft[0], topLeft[1],
                    bottomRight[0] - topLeft[0], bottomRight[1] - topLeft[1]
                );
                const label = "ID " + person.track_id + " · " + state + " · " +
                    Math.round(Number(person.fall_score || 0) * 100) + "%";
                context.font = "12px sans-serif";
                const labelWidth = context.measureText(label).width + 10;
                const labelY = Math.max(16, topLeft[1]);
                context.fillStyle = color;
                context.fillRect(topLeft[0], labelY - 16, labelWidth, 17);
                context.fillStyle = "#081018";
                context.fillText(label, topLeft[0] + 5, labelY - 4);
            }

            context.strokeStyle = color;
            context.fillStyle = color;
            context.lineWidth = 2;
            POSE_SKELETON.forEach(([from, to]) => {
                const first = keypoints[from];
                const second = keypoints[to];
                if (!first || !second || Number(first[2]) < 0.25 || Number(second[2]) < 0.25) return;
                const p1 = mapPoint(first);
                const p2 = mapPoint(second);
                context.beginPath();
                context.moveTo(p1[0], p1[1]);
                context.lineTo(p2[0], p2[1]);
                context.stroke();
            });
            keypoints.forEach((point) => {
                if (Number(point[2]) < 0.25) return;
                const mapped = mapPoint(point);
                context.beginPath();
                context.arc(mapped[0], mapped[1], 2.5, 0, Math.PI * 2);
                context.fill();
            });
        });

        const falling = people.filter((person) => ["FALLING", "FALLEN"].includes(String(person.state)));
        if (falling.length) {
            const fallen = falling.some((person) => String(person.state) === "FALLEN");
            poseAlert.hidden = false;
            poseAlert.classList.toggle("pose-alert-fallen", fallen);
            poseAlert.textContent = (fallen ? "Fallen posture detected" : "Possible fall") + ": " +
                falling.map((person) => "Person #" + person.track_id).join(", ");
        } else {
            poseAlert.hidden = true;
            poseAlert.classList.remove("pose-alert-fallen");
            poseAlert.textContent = "";
        }
        poseTrackingState.textContent = people.length + " person(s) tracked · " +
            String(payload.device || "unknown device");
    }

    async function sendPoseFrame(revision) {
        const socket = live.poseSocket;
        if (
            !live.active || live.revision !== revision || live.poseAwaitingResult ||
            !socket || socket.readyState !== WebSocket.OPEN ||
            !liveVideo.videoWidth || !liveVideo.videoHeight
        ) return;

        const scale = Math.min(1, POSE_CAPTURE_MAX_WIDTH / liveVideo.videoWidth);
        const width = Math.max(1, Math.round(liveVideo.videoWidth * scale));
        const height = Math.max(1, Math.round(liveVideo.videoHeight * scale));
        poseCaptureCanvas.width = width;
        poseCaptureCanvas.height = height;
        const context = poseCaptureCanvas.getContext("2d");
        if (!context) return;
        context.drawImage(liveVideo, 0, 0, width, height);
        live.poseAwaitingResult = true;
        const blob = await new Promise((resolve) => {
            poseCaptureCanvas.toBlob(resolve, "image/jpeg", POSE_JPEG_QUALITY);
        });
        if (
            !blob || !live.active || live.revision !== revision ||
            live.poseSocket !== socket || socket.readyState !== WebSocket.OPEN
        ) {
            live.poseAwaitingResult = false;
            return;
        }
        socket.send(await blob.arrayBuffer());
    }

    function stopPoseTracking() {
        if (live.poseTimer) clearInterval(live.poseTimer);
        live.poseTimer = null;
        const socket = live.poseSocket;
        live.poseSocket = null;
        live.poseAwaitingResult = false;
        if (socket && socket.readyState < WebSocket.CLOSING) socket.close();
        clearPoseOverlay();
        poseTrackingState.textContent = "Pose tracking inactive";
        poseAlert.hidden = true;
        poseAlert.classList.remove("pose-alert-fallen");
        poseAlert.textContent = "";
    }

    function startPoseTracking(revision) {
        stopPoseTracking();
        poseTrackingState.textContent = "Connecting to YOLO Pose service...";
        const socket = new WebSocket(websocketUrl("/ws/pose"));
        live.poseSocket = socket;
        socket.onopen = () => {
            if (live.poseSocket !== socket || live.revision !== revision) return;
            poseTrackingState.textContent = "YOLO Pose connected; loading tracker...";
        };
        socket.onmessage = (event) => {
            if (live.poseSocket !== socket || live.revision !== revision) return;
            try {
                const payload = JSON.parse(event.data);
                if (payload.type === "pose_ready") {
                    live.poseAwaitingResult = false;
                    poseTrackingState.textContent = "YOLO Pose ready · " + payload.model + " · " + payload.device;
                    if (!live.poseTimer) {
                        live.poseTimer = setInterval(
                            () => void sendPoseFrame(revision),
                            POSE_FRAME_INTERVAL_MS
                        );
                    }
                } else if (payload.type === "pose_result") {
                    live.poseAwaitingResult = false;
                    drawPoseResults(payload);
                } else if (payload.type === "pose_error") {
                    live.poseAwaitingResult = false;
                    poseTrackingState.textContent = payload.detail || "YOLO Pose error";
                }
            } catch (error) {
                live.poseAwaitingResult = false;
                poseTrackingState.textContent = "Invalid YOLO Pose response: " + error.message;
            }
        };
        socket.onerror = () => {
            if (live.poseSocket === socket) poseTrackingState.textContent = "YOLO Pose connection failed";
        };
        socket.onclose = () => {
            if (live.poseSocket !== socket) return;
            live.poseSocket = null;
            live.poseAwaitingResult = false;
            if (live.poseTimer) clearInterval(live.poseTimer);
            live.poseTimer = null;
            if (live.active && live.revision === revision) {
                poseTrackingState.textContent = "YOLO Pose disconnected. Start pose_service/run.py on port 8001.";
            }
        };
    }

    function buildLatestRollingWindow(windowNumber) {
        if (live.frameBuffer.length < LIVE_FRAME_COUNT) return null;
        const newest = live.frameBuffer[live.frameBuffer.length - 1];
        const windowStart = newest.capturedAt - LIVE_WINDOW_MS;
        if (live.frameBuffer[0].capturedAt > windowStart + LIVE_SAMPLE_INTERVAL_MS) return null;

        const selected = [];
        let minimumIndex = 0;
        for (let sampleIndex = 0; sampleIndex < LIVE_FRAME_COUNT; sampleIndex += 1) {
            const targetTime = windowStart + sampleIndex * LIVE_SAMPLE_INTERVAL_MS;
            let bestIndex = minimumIndex;
            let bestDistance = Number.POSITIVE_INFINITY;
            const lastAllowedIndex = live.frameBuffer.length - (LIVE_FRAME_COUNT - sampleIndex);
            for (let index = minimumIndex; index <= lastAllowedIndex; index += 1) {
                const distance = Math.abs(live.frameBuffer[index].capturedAt - targetTime);
                if (distance <= bestDistance) {
                    bestDistance = distance;
                    bestIndex = index;
                } else {
                    break;
                }
            }
            selected.push(live.frameBuffer[bestIndex]);
            minimumIndex = bestIndex + 1;
        }

        return {
            frames: selected.map((entry) => entry.blob),
            durationMs: LIVE_WINDOW_MS,
            number: windowNumber,
            startedWallClock: newest.wallClock - LIVE_WINDOW_MS,
            completedWallClock: newest.wallClock,
            completedAt: newest.capturedAt
        };
    }

    async function analyzeLiveWindow(windowData, revision) {
        if (!live.active || revision !== live.revision) return;
        live.analyzing = true;
        const requestStartedAt = performance.now();
        const body = new FormData();
        windowData.frames.forEach((frame, index) => body.append("frames", frame, "frame_" + index + ".jpg"));
        body.append("source_type", live.source);
        body.append("captured_duration_ms", String(windowData.durationMs));
        body.append("window_id", String(windowData.number));
        liveAnalysisWindow.textContent = "#" + windowData.number + " - Uploading / GPU";
        liveStreamState.textContent = "Model is processing window #" + windowData.number + ".";
        const abortController = new AbortController();
        const requestTimeout = setTimeout(
            () => abortController.abort(),
            LIVE_REQUEST_TIMEOUT_MS
        );

        try {
            const response = await fetch(LIVE_PREDICT_URL, {
                method: "POST",
                body,
                signal: abortController.signal
            });
            if (!live.active || revision !== live.revision) return;
            if (response.status === 429) {
                liveAnalysisWindow.textContent = "#" + windowData.number + " - GPU busy";
                live.nextWindowDueAt = performance.now() + 1000;
                setLiveHint("GPU is busy. No stale window was queued; the latest buffer will retry.", false);
                return;
            }
            if (!response.ok) throw new Error(await extractErrorMessage(response));
            const data = await response.json();
            const receivedAt = performance.now();
            const delayAfterCapture = receivedAt - windowData.completedAt;
            const requestDuration = receivedAt - requestStartedAt;
            const rawHazard = String(data.status).toLowerCase() === "hazard";
            const displayData = Object.assign({}, data);
            const alertDecision = rawHazard
                ? "Hazard detected in this window"
                : "Safe in this window";
            const capturedInterval = formatClock(windowData.startedWallClock) + " - " +
                formatClock(windowData.completedWallClock);
            displayData.observation = {
                raw_status: rawHazard ? "Hazard" : "Safe",
                alert_decision: alertDecision,
                captured_interval: capturedInterval,
                delay_after_capture: formatDelay(delayAfterCapture) +
                    " (request " + formatDelay(requestDuration) + ")"
            };
            const subtitle = "Window #" + windowData.number + " | captured " + capturedInterval +
                " | returned +" + formatDelay(delayAfterCapture);
            liveAnalysisWindow.textContent = "#" + windowData.number + " - Complete in " +
                formatDelay(requestDuration);
            liveLastResultWindow.textContent = "#" + windowData.number + " - Raw " +
                (rawHazard ? "HAZARD" : "SAFE") + " - +" + formatDelay(delayAfterCapture);
            liveResultBody.innerHTML = buildResultCardHtml(displayData, subtitle);
            setLiveHint(rawHazard
                ? "A hazard was detected in this window."
                : "", false);
        } catch (error) {
            if (live.active && revision === live.revision) {
                const timedOut = error && error.name === "AbortError";
                liveAnalysisWindow.textContent = "#" + windowData.number +
                    (timedOut ? " - Timed out" : " - Failed");
                renderError(
                    liveResultBody,
                    timedOut ? "Live analysis timed out" : "Live analysis failed",
                    timedOut
                        ? "Window #" + windowData.number +
                            " exceeded 25 seconds. Continuing with the newest window."
                        : error.message
                );
            }
        } finally {
            clearTimeout(requestTimeout);
            if (live.revision === revision) live.analyzing = false;
        }
    }

    async function rollingCaptureLoop(revision) {
        const captureStartedAt = performance.now();
        let captureIndex = 0;
        while (live.active && revision === live.revision) {
            if (captureIndex > 0 && !await waitUntil(
                captureStartedAt + captureIndex * LIVE_SAMPLE_INTERVAL_MS,
                revision
            )) break;
            if (!live.active || revision !== live.revision) break;

            const blob = await captureFrameBlob();
            if (!blob) break;
            const capturedAt = performance.now();
            const wallClock = Date.now();
            live.frameBuffer.push({ blob, capturedAt, wallClock });
            const retentionStart = capturedAt - LIVE_WINDOW_MS - LIVE_SAMPLE_INTERVAL_MS * 3;
            while (live.frameBuffer.length && live.frameBuffer[0].capturedAt < retentionStart) {
                live.frameBuffer.shift();
            }

            const coverageMs = live.frameBuffer.length > 1
                ? capturedAt - live.frameBuffer[0].capturedAt
                : 0;
            const readyPercent = Math.min(100, coverageMs / LIVE_WINDOW_MS * 100);
            liveBufferCounter.textContent = live.frameBuffer.length + " buffered frames · " +
                (Math.min(LIVE_WINDOW_MS, coverageMs) / 1000).toFixed(1) + " / 5.0s";
            progressBar.style.width = readyPercent + "%";
            liveCurrentWindow.textContent = coverageMs >= LIVE_WINDOW_MS - LIVE_SAMPLE_INTERVAL_MS
                ? "Rolling buffer ready"
                : "Filling rolling buffer";

            if (live.nextWindowDueAt === null) {
                live.nextWindowDueAt = capturedAt + Math.max(0, LIVE_WINDOW_MS - coverageMs);
            }

            if (capturedAt >= live.nextWindowDueAt) {
                if (live.analyzing) {
                    liveStreamState.textContent = "GPU busy; keeping only the latest five-second buffer.";
                } else {
                    const windowNumber = live.windowNumber + 1;
                    const windowData = buildLatestRollingWindow(windowNumber);
                    if (windowData) {
                        live.windowNumber = windowNumber;
                        live.nextWindowDueAt = capturedAt + LIVE_STRIDE_MS;
                        setLiveBadge("streaming", "Rolling live");
                        liveStreamState.textContent = "Sending newest rolling window #" + windowNumber + ".";
                        void analyzeLiveWindow(windowData, revision);
                    }
                }
            } else if (!live.analyzing) {
                const untilNext = Math.max(0, live.nextWindowDueAt - capturedAt);
                liveStreamState.textContent = coverageMs < LIVE_WINDOW_MS - LIVE_SAMPLE_INTERVAL_MS
                    ? "Building the first five-second rolling buffer."
                    : "GPU idle; next rolling window in " + (untilNext / 1000).toFixed(1) + "s.";
            }
            captureIndex += 1;
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
        startPoseTracking(revision);
        liveStartBtn.disabled = true;
        liveStopBtn.disabled = false;
        setLiveBadge("warming", "Warming up");
        liveStreamState.textContent = "Stabilizing exposure and focus";
        liveCurrentWindow.textContent = "Waiting for warm-up";
        liveAnalysisWindow.textContent = "No window";
        liveLastResultWindow.textContent = "No result";
        liveBufferCounter.textContent = "Warm-up 2.5s";
        const ready = await waitUntil(performance.now() + CAMERA_WARMUP_MS, revision);
        if (!ready) return;
        live.frameBuffer = [];
        live.nextWindowDueAt = null;
        void rollingCaptureLoop(revision);
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
        stopPoseTracking();
        live.active = false;
        cancelWaiters();
        if (live.stream) live.stream.getTracks().forEach((track) => track.stop());
        live.stream = null;
        liveVideo.srcObject = null;
        liveVideo.style.display = "none";
        livePlaceholder.style.display = "flex";
        live.frameBuffer = [];
        live.nextWindowDueAt = null;
        live.analyzing = false;
        progressBar.style.width = "0%";
        liveBufferCounter.textContent = "0 / 16 frames";
        liveCurrentWindow.textContent = "No active window";
        liveAnalysisWindow.textContent = "No window";
        liveLastResultWindow.textContent = "No result";
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
            ? "Scan the QR code to stream a phone camera into a five-second rolling buffer."
            : "Preview stays live while a 360p rolling buffer sends 16 frames when the GPU is idle.";
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
