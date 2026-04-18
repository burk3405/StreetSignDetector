// ============================================================================
// GLOBAL STATE
// ============================================================================
const state = {
    videoStream: null,      // <img> element for OAK-D MJPEG / uploaded photo
    webcamVideo: null,      // <video> element for browser webcam
    canvas: null,
    canvasCtx: null,
    isVideoReady: false,
    confidenceThreshold: 0.5,
    darkMode: false,
    detections: [],
    fps: 0,
    frameCount: 0,
    lastTime: Date.now(),
    isDetecting: false,
    videoWidth: 1280,
    videoHeight: 720,
    lastLoggedDetections: {},
    maxConsoleLines: 50,
    // Camera mode: 'oak_d' | 'webcam' | 'photo'
    cameraMode: 'webcam',
    photoMode: false,
    photoDetections: [],
    webcamStream: null,     // MediaStream from getUserMedia
    offscreenCanvas: null,  // For capturing webcam frames to send to backend
    offscreenCtx: null,
};

// ============================================================================
// INITIALIZATION
// ============================================================================

async function initialize() {
    console.log('🚀 Initializing YOLOv8 Object Detection App');

    try {
        setupDOM();
        setupEventListeners();
        await setupCamera();
        updateStatus('Ready to detect', true);
        runDetectionLoop();
    } catch (error) {
        console.error('❌ Initialization failed:', error);
        showError(error.message || 'Failed to initialize the app');
        updateStatus('Error occurred', false);
    }
}

function setupDOM() {
    state.videoStream = document.getElementById('videoStream');
    state.webcamVideo = document.getElementById('webcamVideo');
    state.canvas = document.getElementById('canvas');
    state.canvasCtx = state.canvas.getContext('2d', { willReadFrequently: true });

    if (!state.canvas || !state.canvasCtx) {
        throw new Error('Required DOM elements not found');
    }

    // Offscreen canvas for webcam frame capture
    state.offscreenCanvas = document.createElement('canvas');
    state.offscreenCtx = state.offscreenCanvas.getContext('2d');

    // Restore dark mode preference
    const darkModePreference = localStorage.getItem('darkMode') === 'true';
    if (darkModePreference) {
        toggleDarkMode(true);
    }
}

function setupEventListeners() {
    // Confidence threshold slider
    const confidenceSlider = document.getElementById('confidenceSlider');
    confidenceSlider.addEventListener('input', (e) => {
        state.confidenceThreshold = parseFloat(e.target.value);
        document.getElementById('thresholdValue').textContent =
            state.confidenceThreshold.toFixed(2);
        if (state.photoMode) {
            redetectUploadedPhoto();
        }
    });

    // Dark mode toggle
    document.getElementById('darkModeToggle').addEventListener('click', () => {
        toggleDarkMode(!state.darkMode);
    });

    // Virtual camera buttons
    const vcStart = document.getElementById('virtualCameraStartBtn');
    if (vcStart) vcStart.addEventListener('click', startVirtualCamera);
    const vcStop = document.getElementById('virtualCameraStopBtn');
    if (vcStop) vcStop.addEventListener('click', stopVirtualCamera);

    // Resize
    window.addEventListener('resize', () => {
        if (state.isVideoReady) resizeCanvasToVideoSize();
    });

    document.addEventListener('visibilitychange', () => {
        console.log(document.hidden ? '📴 Page hidden' : '📱 Page visible');
    });

    // Console clear
    const clearBtn = document.getElementById('clearConsoleBtn');
    if (clearBtn) clearBtn.addEventListener('click', clearConsoleLog);

    // Photo upload
    const photoInput = document.getElementById('photoFileInput');
    if (photoInput) photoInput.addEventListener('change', handlePhotoUpload);

    const clearPhotoBtn = document.getElementById('clearPhotoBtn');
    if (clearPhotoBtn) clearPhotoBtn.addEventListener('click', clearUploadedPhoto);
}

// ============================================================================
// CAMERA SETUP — graceful OAK-D → webcam fallback
// ============================================================================

async function setupCamera() {
    console.log('📷 Checking camera availability...');

    try {
        // Check if OAK-D is available
        const response = await fetch('/api/oak-d/status');
        const status = await response.json();
        console.log('📷 OAK-D status:', status);

        if (status.initialized) {
            // OAK-D is connected — use MJPEG stream
            console.log('✅ OAK-D detected, using MJPEG stream');
            state.cameraMode = 'oak_d';
            const [w, h] = status.resolution || [1280, 720];
            state.videoWidth = w;
            state.videoHeight = h;

            // Show the <img> element with MJPEG stream
            state.videoStream.src = `/video_feed?t=${Date.now()}`;
            state.videoStream.style.display = 'block';
            state.webcamVideo.style.display = 'none';
            state.isVideoReady = true;

            addConsoleLog('📷 OAK-D camera connected', 'info-sign');
            resizeCanvasToVideoSize();
            return;
        }
    } catch (e) {
        console.warn('⚠️  Could not check OAK-D status:', e.message);
    }

    // Fall back to browser webcam
    console.log('⚠️  OAK-D not available, falling back to browser webcam');
    await setupBrowserWebcam();
}

async function setupBrowserWebcam() {
    state.cameraMode = 'webcam';

    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'environment' },
            audio: false
        });

        state.webcamStream = stream;
        state.webcamVideo.srcObject = stream;
        state.webcamVideo.style.display = 'block';
        state.videoStream.style.display = 'none';

        await new Promise((resolve) => {
            state.webcamVideo.onloadedmetadata = () => {
                state.videoWidth = state.webcamVideo.videoWidth;
                state.videoHeight = state.webcamVideo.videoHeight;
                state.offscreenCanvas.width = state.videoWidth;
                state.offscreenCanvas.height = state.videoHeight;
                resolve();
            };
        });

        state.isVideoReady = true;
        console.log(`✅ Browser webcam ready: ${state.videoWidth}x${state.videoHeight}`);
        addConsoleLog('📷 Using browser webcam (no OAK-D detected)', 'info-sign');
        resizeCanvasToVideoSize();
    } catch (error) {
        console.error('❌ Webcam access failed:', error);
        state.isVideoReady = true; // Still allow photo uploads
        state.videoWidth = 1280;
        state.videoHeight = 720;
        addConsoleLog('⚠️ No camera available — use photo upload', 'warning-sign');
        resizeCanvasToVideoSize();
    }
}

function resizeCanvasToVideoSize() {
    if (!state.isVideoReady) return;
    const container = state.canvas.parentElement;
    const rect = container.getBoundingClientRect();
    state.canvas.width = rect.width;
    state.canvas.height = rect.height;
}

// ============================================================================
// MAIN DETECTION LOOP
// ============================================================================

async function runDetectionLoop() {
    if (document.hidden || state.photoMode) {
        requestAnimationFrame(runDetectionLoop);
        return;
    }

    if (!state.isDetecting && state.isVideoReady) {
        state.isDetecting = true;
        try {
            await detectObjects();
            drawDetections();
            logDetections();
            updateFPS();
        } catch (error) {
            console.error('❌ Detection error:', error);
        }
        state.isDetecting = false;
    }

    requestAnimationFrame(runDetectionLoop);
}

// ============================================================================
// DETECTION
// ============================================================================

async function detectObjects() {
    try {
        if (state.cameraMode === 'oak_d') {
            await detectViaOakD();
        } else if (state.cameraMode === 'webcam') {
            await detectViaWebcam();
        }
    } catch (error) {
        console.error('Detection error:', error);
        state.detections = [];
    }
}

/**
 * OAK-D: server captures frame + runs detection
 */
async function detectViaOakD() {
    const response = await fetch(
        `/api/oak-d/detect?confidence=${state.confidenceThreshold}&include_spatial=true`
    );
    if (!response.ok) throw new Error(`Backend error: ${response.status}`);
    const result = await response.json();
    if (!result.success) throw new Error(result.error || 'Detection failed');

    state.detections = result.detections.map(det => {
        const bbox = det.bbox;
        const d = {
            class: det.class,
            score: det.confidence,
            confidence: det.confidence,
            classId: det.classId,
            sign_category: det.sign_category,
            bbox: [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]
        };
        if (det.spatial) { d.spatial = det.spatial; d.distance = det.distance; }
        return d;
    });
}

/**
 * Browser webcam: capture frame, send base64 to /api/detect
 */
async function detectViaWebcam() {
    if (!state.webcamVideo || state.webcamVideo.readyState < 2) {
        state.detections = [];
        return;
    }

    // Draw current video frame to offscreen canvas
    state.offscreenCtx.drawImage(state.webcamVideo, 0, 0,
        state.videoWidth, state.videoHeight);

    // Convert to base64 JPEG
    const dataUrl = state.offscreenCanvas.toDataURL('image/jpeg', 0.8);
    const b64 = dataUrl.split(',')[1];

    const response = await fetch('/api/detect', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            image: b64,
            confidence: state.confidenceThreshold
        })
    });

    if (!response.ok) throw new Error(`Backend error: ${response.status}`);
    const result = await response.json();
    if (!result.success) throw new Error(result.error || 'Detection failed');

    const fw = result.frame_width || state.videoWidth;
    const fh = result.frame_height || state.videoHeight;

    state.detections = result.detections.map(det => {
        const [x1, y1, x2, y2] = det.bbox;
        return {
            class: det.class,
            score: det.confidence,
            confidence: det.confidence,
            classId: det.classId,
            sign_category: det.sign_category,
            // Normalize to 0-1 (the /api/detect endpoint returns pixel coords)
            bbox: [x1 / fw, y1 / fh, (x2 - x1) / fw, (y2 - y1) / fh]
        };
    });
}

// ============================================================================
// DRAWING
// ============================================================================

/** Convert hex color like '#FF6600' to 'rgba(255,102,0,alpha)' */
function hexToRgba(hex, alpha) {
    const r = parseInt(hex.slice(1, 3), 16);
    const g = parseInt(hex.slice(3, 5), 16);
    const b = parseInt(hex.slice(5, 7), 16);
    return `rgba(${r},${g},${b},${alpha})`;
}

/**
 * Compute the actual rendered position & size of the image/video inside its
 * container when object-fit: contain is used.  Returns {offsetX, offsetY,
 * renderWidth, renderHeight}.
 */
function getContainedRect() {
    const canvasW = state.canvas.width;
    const canvasH = state.canvas.height;
    const videoW = state.videoWidth;
    const videoH = state.videoHeight;

    if (!videoW || !videoH) return { offsetX: 0, offsetY: 0, renderWidth: canvasW, renderHeight: canvasH };

    const containerAspect = canvasW / canvasH;
    const imageAspect = videoW / videoH;

    let renderWidth, renderHeight;
    if (imageAspect > containerAspect) {
        // Image is wider than container → letterbox top/bottom
        renderWidth = canvasW;
        renderHeight = canvasW / imageAspect;
    } else {
        // Image is taller than container → pillarbox left/right
        renderHeight = canvasH;
        renderWidth = canvasH * imageAspect;
    }

    const offsetX = (canvasW - renderWidth) / 2;
    const offsetY = (canvasH - renderHeight) / 2;

    return { offsetX, offsetY, renderWidth, renderHeight };
}

function drawDetections() {
    const canvas = state.canvas;
    const ctx = state.canvasCtx;
    const canvasWidth = canvas.width;
    const canvasHeight = canvas.height;

    ctx.clearRect(0, 0, canvasWidth, canvasHeight);
    if (state.detections.length === 0) return;

    // Compute where the image actually renders (accounting for object-fit: contain)
    const { offsetX, offsetY, renderWidth, renderHeight } = getContainedRect();

    state.detections.forEach(detection => {
        drawBoundingBox(detection, offsetX, offsetY, renderWidth, renderHeight, ctx, canvasWidth, canvasHeight);
    });
}

function drawBoundingBox(detection, offsetX, offsetY, renderWidth, renderHeight, ctx, canvasWidth, canvasHeight) {
    const [x, y, width, height] = detection.bbox;

    // bbox values are normalized 0-1 — map them into the rendered image area
    const scaledX = offsetX + x * renderWidth;
    const scaledY = offsetY + y * renderHeight;
    const scaledWidth = width * renderWidth;
    const scaledHeight = height * renderHeight;

    const clampedX = Math.max(0, scaledX);
    const clampedY = Math.max(0, scaledY);
    const clampedWidth = Math.min(canvasWidth - clampedX, scaledWidth);
    const clampedHeight = Math.min(canvasHeight - clampedY, scaledHeight);

    // Determine box color — use sign_category when available, otherwise color by class
    let boxColor;
    const category = detection.sign_category || detection.category;
    if (category === 'regulatory') {
        boxColor = '#FF0000';   // Red for regulatory signs
    } else if (category === 'warning') {
        boxColor = '#FFFF00';   // Yellow for warning signs
    } else if (category === 'informational') {
        boxColor = '#00BFFF';   // Blue for informational signs
    } else {
        // Color by COCO object type for general detection
        const cls = (detection.class || '').toLowerCase();
        if (cls.includes('person') || cls.includes('pedestrian')) {
            boxColor = '#FF6600';       // Orange for people
        } else if (cls.includes('car') || cls.includes('truck') || cls.includes('bus') || cls.includes('motorcycle') || cls.includes('bicycle')) {
            boxColor = '#00FF00';       // Green for vehicles
        } else if (cls.includes('stop sign')) {
            boxColor = '#FF0000';       // Red for stop signs
        } else if (cls.includes('traffic light')) {
            boxColor = '#FFFF00';       // Yellow for traffic lights
        } else {
            boxColor = '#00FFFF';       // Cyan for everything else
        }
    }

    ctx.strokeStyle = boxColor;
    ctx.lineWidth = 3;
    ctx.strokeRect(clampedX, clampedY, clampedWidth, clampedHeight);

    const signClass = detection.class || 'Unknown';
    const confidence = detection.score || detection.confidence || 0;
    let label = `${signClass} — ${(confidence * 100).toFixed(1)}%`;

    if (detection.spatial && detection.spatial.z) {
        const distanceM = detection.spatial.z;
        const distanceFt = (distanceM * 3.28084).toFixed(1);
        label += ` | ${distanceM.toFixed(1)}m (${distanceFt}ft)`;
    } else if (detection.depth_mm) {
        const distanceM = (detection.depth_mm / 1000).toFixed(1);
        const distanceFt = (distanceM * 3.28084).toFixed(1);
        label += ` | ${distanceM}m (${distanceFt}ft)`;
    }

    const fontSize = 14;
    const padding = 6;
    const labelX = clampedX;
    const labelY = Math.max(fontSize + padding * 2, clampedY);

    ctx.font = `${fontSize}px Arial`;
    const textMetrics = ctx.measureText(label);
    const labelWidth = textMetrics.width + padding * 2;
    const labelHeight = fontSize + padding * 2;

    // Background fill matching the box color (semi-transparent)
    ctx.fillStyle = hexToRgba(boxColor, 0.8);
    ctx.fillRect(labelX, labelY - labelHeight, labelWidth, labelHeight);

    // Text color — dark text on yellow, white on everything else
    ctx.fillStyle = (boxColor === '#FFFF00') ? '#000000' : '#FFFFFF';
    ctx.textBaseline = 'top';
    ctx.fillText(label, labelX + padding, labelY - labelHeight + padding);

    if (detection.spatial && (detection.spatial.x !== 0 || detection.spatial.y !== 0)) {
        drawSpatialIndicator(detection, clampedX, clampedY, clampedWidth, clampedHeight, ctx);
    }
}

function drawSpatialIndicator(detection, boxX, boxY, boxW, boxH, ctx) {
    const centerX = boxX + boxW / 2;
    const centerY = boxY + boxH / 2;
    ctx.strokeStyle = '#FF00FF';
    ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(centerX, centerY - 8); ctx.lineTo(centerX, centerY + 8); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(centerX - 8, centerY); ctx.lineTo(centerX + 8, centerY); ctx.stroke();
}

// ============================================================================
// DETECTION LOG
// ============================================================================

function logDetections() {
    if (state.detections.length === 0) return;

    state.detections.forEach(detection => {
        const detectionKey = `${detection.class}-${detection.confidence.toFixed(2)}`;
        if (!state.lastLoggedDetections[detectionKey] ||
            Date.now() - state.lastLoggedDetections[detectionKey] > 1000) {

            const now = new Date();
            const timestamp = now.toLocaleTimeString('en-US', {
                hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit'
            });

            let distance = 'Unknown';
            if (detection.spatial && detection.spatial.z) {
                const m = detection.spatial.z.toFixed(1);
                const ft = (detection.spatial.z * 3.28084).toFixed(1);
                distance = `${m}m (${ft}ft)`;
            }

            const conf = (detection.confidence * 100).toFixed(0);
            const action = getActionRecommendation(detection.class);
            const category = getSignCategory(detection.class);
            const message = `${timestamp} ${detection.class} (${conf}% confident) ${distance}: ${action}`;
            addConsoleLog(message, category);
            state.lastLoggedDetections[detectionKey] = Date.now();
        }
    });
}

function getActionRecommendation(signName) {
    const s = signName.toLowerCase();
    if (s.includes('stop')) return '⚠️ STOP: Apply brakes immediately';
    if (s.includes('do not enter') || s.includes('no entry')) return '⚠️ DO NOT ENTER: Turn around';
    if (s.includes('yield')) return '⚠️ YIELD: Reduce speed';
    if (s.includes('speed') || s.includes('mph')) return '📋 Adjust speed to limit';
    if (s.includes('one way')) return '➡️ Travel in indicated direction only';
    if (s.includes('pedestrian') || s.includes('school') || s.includes('crossing')) return '👥 Caution: Pedestrians ahead';
    if (s.includes('curve') || s.includes('turn')) return '🔄 Prepare for turn';
    if (s.includes('merge')) return '↔️ Merge carefully';
    if (s.includes('construction') || s.includes('work')) return '⚠️ Construction zone';
    return '💡 New sign detected';
}

function getSignCategory(signName) {
    const s = signName.toLowerCase();
    if (s.includes('stop') || s.includes('do not') || s.includes('no entry') || s.includes('no passing')) return 'stop-sign';
    if (s.includes('warning') || s.includes('caution') || s.includes('curve') || s.includes('pedestrian') ||
        s.includes('school') || s.includes('construction') || s.includes('merge') || s.includes('railroad')) return 'warning-sign';
    return 'info-sign';
}

function addConsoleLog(message, category = 'detection') {
    const logContainer = document.getElementById('detectionLog');
    if (!logContainer) return;

    const logLine = document.createElement('div');
    logLine.className = `console-line ${category}`;
    logLine.textContent = message;
    logContainer.appendChild(logLine);

    const welcomeLine = logContainer.querySelector('.welcome');
    if (welcomeLine && category !== 'detection') welcomeLine.remove();

    const lines = logContainer.querySelectorAll('.console-line');
    if (lines.length > state.maxConsoleLines) {
        for (let i = 0; i < lines.length - state.maxConsoleLines; i++) lines[i].remove();
    }

    setTimeout(() => { logContainer.scrollTop = logContainer.scrollHeight; }, 10);
}

function clearConsoleLog() {
    const logContainer = document.getElementById('detectionLog');
    if (!logContainer) return;
    logContainer.innerHTML = '<div class="console-line welcome">🟢 Console cleared • Ready for new detections...</div>';
    state.lastLoggedDetections = {};
}

function updateFPS() {
    state.frameCount++;
    const currentTime = Date.now();
    const elapsed = currentTime - state.lastTime;
    if (elapsed >= 1000) {
        state.fps = Math.round((state.frameCount * 1000) / elapsed);
        document.getElementById('fpsValue').textContent = state.fps;
        state.frameCount = 0;
        state.lastTime = currentTime;
    }
}

// ============================================================================
// UI UTILITIES
// ============================================================================

function toggleDarkMode(enable) {
    state.darkMode = enable;
    document.body.classList.toggle('dark-mode', enable);
    document.getElementById('darkModeToggle').classList.toggle('active', enable);
    localStorage.setItem('darkMode', enable.toString());
}

function updateStatus(message, isReady) {
    document.getElementById('statusText').textContent = message;
    document.getElementById('statusDot').classList.toggle('ready', isReady);
}

function showError(message) {
    const el = document.getElementById('errorMessage');
    el.textContent = message;
    el.classList.add('show');
    setTimeout(() => el.classList.remove('show'), 5000);
}

// ============================================================================
// VIRTUAL CAMERA CONTROL
// ============================================================================

async function startVirtualCamera() {
    try {
        const response = await fetch('/api/virtual-camera/start', {
            method: 'POST', headers: { 'Content-Type': 'application/json' }
        });
        const data = await response.json();
        if (!response.ok) {
            let errorMsg = data.error || 'Failed to start virtual camera';
            if (data.instructions && Array.isArray(data.instructions))
                errorMsg += '\n\n' + data.instructions.join('\n');
            showError(errorMsg);
            return;
        }
        showSuccessMessage('✅ Virtual camera activated!');
        updateVirtualCameraUI(true);
    } catch (error) {
        showError('Failed to start virtual camera: ' + error.message);
    }
}

async function stopVirtualCamera() {
    try {
        const response = await fetch('/api/virtual-camera/stop', {
            method: 'POST', headers: { 'Content-Type': 'application/json' }
        });
        if (!response.ok) { const e = await response.json(); throw new Error(e.error); }
        showSuccessMessage('✅ Virtual camera deactivated');
        updateVirtualCameraUI(false);
    } catch (error) {
        showError('Failed to stop virtual camera: ' + error.message);
    }
}

function updateVirtualCameraUI(isRunning) {
    const startBtn = document.getElementById('virtualCameraStartBtn');
    const stopBtn = document.getElementById('virtualCameraStopBtn');
    const statusLabel = document.getElementById('virtualCameraStatus');
    if (startBtn) startBtn.disabled = isRunning;
    if (stopBtn) stopBtn.disabled = !isRunning;
    if (statusLabel) {
        statusLabel.textContent = isRunning ? '🟢 Running' : '🔴 Inactive';
        statusLabel.className = isRunning ? 'status-running' : 'status-inactive';
    }
}

function showSuccessMessage(message) {
    const el = document.createElement('div');
    el.textContent = message;
    el.style.cssText = `
        position: fixed; top: 60px; left: 50%; transform: translateX(-50%);
        background: #4CAF50; color: white; padding: 16px 24px; border-radius: 8px;
        z-index: 1000; box-shadow: 0 2px 8px rgba(0,0,0,0.2); font-size: 14px;
    `;
    document.body.appendChild(el);
    setTimeout(() => el.remove(), 4000);
}

async function checkVirtualCameraStatus() {
    try {
        const response = await fetch('/api/virtual-camera/status');
        if (response.status === 412) return;
        if (!response.ok) return;
        const data = await response.json();
        updateVirtualCameraUI(data.running);
    } catch (error) {
        console.debug('Could not check virtual camera status:', error);
    }
}

// ============================================================================
// PHOTO UPLOAD
// ============================================================================

async function handlePhotoUpload(event) {
    const file = event.target.files[0];
    if (!file) return;

    console.log(`📸 Uploading photo: ${file.name} (${(file.size / 1024).toFixed(1)} KB)`);

    try {
        const formData = new FormData();
        formData.append('photo', file);
        formData.append('confidence', state.confidenceThreshold.toString());

        const response = await fetch('/api/upload-photo', {
            method: 'POST',
            body: formData
        });

        // Guard against non-JSON responses (e.g. Flask HTML error pages)
        const contentType = response.headers.get('content-type') || '';
        if (!contentType.includes('application/json')) {
            const text = await response.text();
            console.error('Server returned non-JSON:', text.substring(0, 200));
            throw new Error('Server error — check that the backend is running and model is loaded');
        }

        const result = await response.json();

        if (!response.ok || !result.success) {
            throw new Error(result.error || 'Upload failed');
        }

        console.log(`✅ Photo processed: ${result.width}x${result.height}, ${result.count} detections`);

        // Enter photo mode
        state.photoMode = true;
        state.videoWidth = result.width;
        state.videoHeight = result.height;

        // Hide webcam video, show <img> with uploaded photo
        state.webcamVideo.style.display = 'none';
        state.videoStream.style.display = 'block';
        state.videoStream.src = `/api/uploaded-photo?t=${Date.now()}`;

        // Parse detections (bbox is normalized [x1,y1,x2,y2] → convert to [x,y,w,h])
        state.detections = result.detections.map(det => {
            const bbox = det.bbox;
            return {
                class: det.class,
                score: det.confidence,
                confidence: det.confidence,
                classId: det.classId,
                sign_category: det.sign_category,
                bbox: [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]
            };
        });
        state.photoDetections = state.detections;

        // Wait for image to load then draw bounding boxes
        state.videoStream.onload = () => {
            state.isVideoReady = true;
            resizeCanvasToVideoSize();
            drawDetections();
            logDetections();
        };

        updatePhotoModeUI(true);
        updateStatus(`Photo: ${result.count} detections`, true);
        addConsoleLog(`📸 Photo uploaded: ${file.name} — ${result.count} detection(s) found`, 'info-sign');

    } catch (error) {
        console.error('❌ Photo upload error:', error);
        showError('Photo upload failed: ' + error.message);
    }

    event.target.value = '';
}

async function clearUploadedPhoto() {
    try { await fetch('/api/clear-photo', { method: 'POST' }); } catch (e) { /* ok */ }

    state.photoMode = false;
    state.photoDetections = [];
    state.detections = [];

    // Clear canvas
    state.canvasCtx.clearRect(0, 0, state.canvas.width, state.canvas.height);

    // Restore the correct display element based on camera mode
    if (state.cameraMode === 'oak_d') {
        state.videoStream.src = `/video_feed?t=${Date.now()}`;
        state.videoStream.style.display = 'block';
        state.webcamVideo.style.display = 'none';
    } else {
        state.videoStream.style.display = 'none';
        state.webcamVideo.style.display = 'block';
    }

    updatePhotoModeUI(false);
    updateStatus('Ready to detect', true);
    addConsoleLog('📷 Returned to camera feed', 'welcome');
    resizeCanvasToVideoSize();
}

async function redetectUploadedPhoto() {
    if (!state.photoMode) return;
    try {
        const response = await fetch(`/api/uploaded-photo/detect?confidence=${state.confidenceThreshold}`);
        if (!response.ok) return;
        const result = await response.json();
        if (!result.success) return;

        state.detections = result.detections.map(det => {
            const bbox = det.bbox;
            return {
                class: det.class,
                score: det.confidence,
                confidence: det.confidence,
                classId: det.classId,
                sign_category: det.sign_category,
                bbox: [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]
            };
        });
        state.photoDetections = state.detections;
        drawDetections();
        updateStatus(`Photo: ${result.count} detections`, true);
    } catch (error) {
        console.error('Photo re-detection error:', error);
    }
}

function updatePhotoModeUI(isPhotoMode) {
    const uploadBtn = document.getElementById('photoUploadBtn');
    const clearBtn = document.getElementById('clearPhotoBtn');
    const modeLabel = document.getElementById('photoModeLabel');
    if (uploadBtn) uploadBtn.style.display = isPhotoMode ? 'none' : '';
    if (clearBtn) clearBtn.style.display = isPhotoMode ? 'inline-block' : 'none';
    if (modeLabel) modeLabel.style.display = isPhotoMode ? 'inline-block' : 'none';
}

// ============================================================================
// STARTUP
// ============================================================================

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        initialize();
        checkVirtualCameraStatus();
    });
} else {
    initialize();
    checkVirtualCameraStatus();
}
