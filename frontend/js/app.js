(function () {
    "use strict";

    const PREDICT_URL = "/predict/";
    const ALLOWED_EXTENSIONS = [".mp4", ".avi", ".mov"];

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

    let selectedFile = null;
    let objectUrl = null;
    let isProcessing = false;

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

    function renderResult(data) {
        const isHazard = data.status === "Hazard";
        const bannerClass = isHazard ? "hazard" : "safe";
        const statusLabel = isHazard ? "HAZARD DETECTED" : "SAFE";
        const copy = isHazard
            ? "An unsafe workplace condition was detected in this clip. Review the actions below and follow site safety procedure."
            : "No hazard was detected in this clip. Detected workplace actions are listed below when available.";

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

        resultBody.innerHTML = [
            '<article class="result-card">',
            '  <div class="result-banner ' + bannerClass + '">',
            '    <p class="status-kicker">Model status</p>',
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

    async function extractErrorMessage(response) {
        try {
            const payload = await response.json();
            if (payload && payload.detail) {
                if (typeof payload.detail === "string") return payload.detail;
                return JSON.stringify(payload.detail);
            }
        } catch (err) {
            // Response was not JSON; fall through to status text.
        }
        return response.statusText || "Unexpected server error.";
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

    renderEmpty();
})();
