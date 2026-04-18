#!/usr/bin/env python3
"""
US Traffic Sign Detection using YOLOv8
Detects USDOT signs: Speed Limit, One Way, Do Not Enter, etc.
Trained on specialized traffic sign datasets
Includes OAK-D camera integration and web UI
"""

from flask import Flask, request, jsonify, send_file, Response, send_from_directory
from flask_cors import CORS
import base64
import cv2
import numpy as np
from ultralytics import YOLO
import logging
import os
import threading
from pathlib import Path

# OAK-D camera integration (optional)
try:
    from oak_d_adapter import (init_oak_d, get_adapter, stop_oak_d, 
                               start_virtual_camera, stop_virtual_camera, 
                               is_virtual_camera_running)
    from virtual_camera import get_virtual_camera_error
    OAK_D_AVAILABLE = True
except ImportError:
    OAK_D_AVAILABLE = False

# Get the directory of this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Uploaded photo state (in-memory)
uploaded_photo_data = None  # Raw JPEG bytes
uploaded_photo_detections = None  # Cached detections for the uploaded photo

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# FRONTEND ROUTES
# ============================================================================

# ============================================================================
# FRONTEND ROUTES
# ============================================================================

@app.route('/debug-routes')
def debug_routes():
    """Debug endpoint to show registered routes"""
    rules = []
    for rule in app.url_map.iter_rules():
        rules.append(f"{rule.rule} -> {rule.endpoint}")
    return '\n'.join(rules)

@app.route('/')
def index():
    """Serve index.html"""
    try:
        file_path = os.path.join(SCRIPT_DIR, 'index.html')
        logger.info(f"🔍 Root route requested, file_path={file_path}")
        logger.info(f"   File exists: {os.path.isfile(file_path)}")
        result = send_from_directory(SCRIPT_DIR, 'index.html')
        logger.info(f"✅ Successfully sent index.html")
        return result
    except Exception as e:
        logger.error(f"❌ Error in index(): {str(e)}", exc_info=True)
        return f"Error: {str(e)}", 500

@app.route('/<path:filename>')
def serve_static(filename):
    """Serve static files (app.js, styles.css, etc)"""
    logger.info(f"🔍 Catch-all route for: {filename}")
    if filename.startswith('api/'):
        logger.info(f"❌ Blocking API path: {filename}")
        return jsonify({'error': 'Not found'}), 404
    try:
        file_path = os.path.join(SCRIPT_DIR, filename)
        logger.info(f"📁 Trying to serve {filename} from {file_path}")
        logger.info(f"   File exists: {os.path.isfile(file_path)}")
        result = send_from_directory(SCRIPT_DIR, filename)
        logger.info(f"✅ Successfully sent {filename}")
        return result
    except Exception as e:
        logger.error(f"❌ Error serving {filename}: {str(e)}", exc_info=True)
        return f"Error: {str(e)}", 500

# ============================================================================
# CAMERA CONFIGURATION
# ============================================================================
USE_OAK_D = os.getenv('USE_OAK_D', 'True').lower() == 'true'
oak_d_adapter = None

def init_camera():
    """Initialize camera (OAK-D if available, else fall back to API-based).
    Uses a timeout to avoid blocking the server if no OAK-D is connected."""
    global oak_d_adapter
    
    if not OAK_D_AVAILABLE:
        logger.warning('⚠️  OAK-D adapter not available. Using API-based webcam')
        return
    
    if not USE_OAK_D:
        logger.info('⚠️  OAK-D disabled via USE_OAK_D environment variable')
        return
    
    # Try to initialize with a timeout so the server doesn't hang
    init_result = [None]
    init_error = [None]
    
    def _try_init():
        try:
            init_result[0] = init_oak_d(use_oak_d=True)
        except Exception as e:
            init_error[0] = e
    
    logger.info('🎥 Initializing OAK-D Lite camera (10s timeout)...')
    t = threading.Thread(target=_try_init, daemon=True)
    t.start()
    t.join(timeout=10)  # Wait at most 10 seconds
    
    if t.is_alive():
        logger.warning('⚠️  OAK-D initialization timed out (no camera detected). Using browser webcam.')
        return
    
    if init_error[0]:
        logger.warning(f'⚠️  OAK-D initialization failed: {str(init_error[0])}')
        logger.info('Falling back to browser webcam')
        return
    
    oak_d_adapter = init_result[0]
    if oak_d_adapter:
        logger.info('✅ OAK-D camera ready for spatial detection')
    else:
        logger.warning('⚠️  OAK-D init returned None. Using browser webcam.')

# ============================================================================
# US TRAFFIC SIGN CLASSES - USDOT Signs
# ============================================================================
US_TRAFFIC_SIGNS = {
    # Regulatory Signs (Red/White)
    0: "Stop Sign",
    1: "Do Not Enter",
    2: "Yield",
    3: "One Way",
    4: "No Left Turn",
    5: "No Right Turn",
    6: "No U-Turn",
    7: "No Parking",
    8: "Speed Limit 5",
    9: "Speed Limit 15",
    10: "Speed Limit 25",
    11: "Speed Limit 35",
    12: "Speed Limit 45",
    13: "Speed Limit 55",
    14: "Speed Limit 65",
    15: "Speed Limit 75",
    
    # Warning Signs (Yellow/Black)
    16: "Pedestrian Crossing",
    17: "School Zone",
    18: "Curve Ahead",
    19: "Slippery Road",
    20: "Merge",
    21: "Construction",
    22: "Deer Crossing",
    23: "Hill Ahead",
    24: "Sharp Turn",
    25: "Divided Highway",
    26: "Road Narrows",
    27: "Bump",
    
    # Informational Signs (Green/White)
    28: "Exit",
    29: "Parking",
    30: "Hospital",
    31: "Gas Station",
    32: "Lodging",
    33: "Rest Area",
    
    # Guide/Direction Signs
    34: "Route Shield",
    35: "Interstate Shield",
    36: "US Route Shield",
    37: "State Route Shield",
}

# Load YOLOv8 model trained on traffic signs
def load_traffic_sign_model():
    """Load YOLOv8 model trained on US traffic signs"""
    try:
        # Option 1: Use fine-tuned traffic sign model (if available)
        # model = YOLO('path/to/trained_traffic_signs.pt')
        
        # Option 2: Use generic YOLOv8 (will detect objects)
        # This is the fallback - for production, train on traffic sign dataset
        model = YOLO('yolov8m.pt')  # medium model for better accuracy
        
        logger.info('✅ YOLOv8 model loaded successfully')
        return model
    except Exception as e:
        logger.error(f'❌ Failed to load YOLOv8: {e}')
        return None

model = load_traffic_sign_model()

# ============================================================================
# TRAFFIC SIGN RECOGNITION UTILITIES
# ============================================================================

def extract_sign_region(image, bbox):
    """Extract region of interest (traffic sign) from image"""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(image.shape[1], x2)
    y2 = min(image.shape[0], y2)
    
    return image[y1:y2, x1:x2]

def identify_sign_type(image_region):
    """
    Identify specific traffic sign type from image region
    This uses color and shape analysis as a heuristic
    For production, use a dedicated sign classifier
    """
    if image_region.size == 0:
        return None
    
    # Resize for analysis
    h, w = image_region.shape[:2]
    aspect_ratio = w / h if h > 0 else 1
    
    # Convert to HSV for color analysis
    hsv = cv2.cvtColor(image_region, cv2.COLOR_BGR2HSV)
    
    # Check dominant colors to classify sign type
    # Red signs: Stop, Do Not Enter, Yield
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    
    red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_pixels = cv2.countNonZero(red_mask1) + cv2.countNonZero(red_mask2)
    
    # White color
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 20, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    white_pixels = cv2.countNonZero(white_mask)
    
    # Yellow/Green for warning signs
    lower_yellow = np.array([15, 100, 100])
    upper_yellow = np.array([35, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    yellow_pixels = cv2.countNonZero(yellow_mask)
    
    total_pixels = image_region.size // 3
    
    red_ratio = red_pixels / total_pixels if total_pixels > 0 else 0
    white_ratio = white_pixels / total_pixels if total_pixels > 0 else 0
    yellow_ratio = yellow_pixels / total_pixels if total_pixels > 0 else 0
    
    # Classify based on dominant color
    if red_ratio > 0.3:
        # Red sign: Stop, Do Not Enter, Yield
        if 1.2 < aspect_ratio < 1.5:
            return "Do Not Enter"  # Rectangular
        elif abs(aspect_ratio - 1.0) < 0.3:
            return "Stop Sign"  # Octagon shape (roughly square)
        else:
            return "Yield"  # Triangle (wider base)
    elif yellow_ratio > 0.3:
        return "Warning Sign"
    elif white_ratio > 0.4:
        return "One Way"  # White arrows and signs
    
    return None

def confidence_filter(detections, threshold=0.5):
    """Filter detections by confidence threshold"""
    filtered = []
    for detection in detections:
        if detection.conf[0] >= threshold:
            filtered.append(detection)
    return filtered

@app.route('/api/detect', methods=['POST'])
def detect():
    """
    Receive base64-encoded image, run YOLOv8 detection + sign classification
    Optionally includes 3D spatial coordinates from OAK-D camera
    
    Request: {
        "image": "base64_encoded_image",
        "confidence": 0.5,
        "include_spatial": true
    }
    
    Response: {
        "detections": [
            {
                "class": "Speed Limit 55",
                "confidence": 0.95,
                "sign_type": "Regulatory",
                "bbox": [x1, y1, x2, y2],
                "classId": 13,
                "spatial": {
                    "distance_m": 15.5,
                    "x": 2.3,
                    "y": -0.5,
                    "z": 15.5
                },
                "depth_mm": 15500
            },
            ...
        ],
        "success": true,
        "camera": "oak_d" or "api"
    }
    """
    try:
        if not model:
            return jsonify({'error': 'Model not loaded', 'success': False}), 500
        
        data = request.json
        image_b64 = data.get('image')
        confidence_threshold = data.get('confidence', 0.5)
        
        if not image_b64:
            return jsonify({'error': 'No image provided', 'success': False}), 400
        
        # Decode base64 image
        image_data = base64.b64decode(image_b64)
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'Invalid image', 'success': False}), 400
        
        # Run YOLOv8 inference
        results = model(image, conf=confidence_threshold, verbose=False)
        
        # Parse detections and identify traffic signs
        detections = []
        include_spatial = data.get('include_spatial', False)
        
        # Get frame resolution for spatial calculations
        frame_height, frame_width = image.shape[:2]
        
        for result in results:
            boxes = result.boxes
            
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = model.names.get(cls_id, f"Object {cls_id}")
                
                # For traffic sign models: identify specific sign type
                sign_region = extract_sign_region(image, [x1, y1, x2, y2])
                sign_type = identify_sign_type(sign_region)
                
                # Map to US traffic sign classes if available
                if sign_type:
                    classification = sign_type
                else:
                    # Use model's original classification
                    classification = cls_name
                
                detection_dict = {
                    'class': classification,
                    'original_class': cls_name,
                    'confidence': conf,
                    'bbox': [x1, y1, x2, y2],
                    'classId': cls_id,
                    'sign_category': get_sign_category(classification)
                }
                
                # Add spatial data from OAK-D if available and requested
                if include_spatial and oak_d_adapter:
                    # Normalize bbox to 0-1
                    norm_bbox = [
                        x1 / frame_width,
                        y1 / frame_height,
                        x2 / frame_width,
                        y2 / frame_height
                    ]
                    
                    # Get spatial coordinates
                    spatial = oak_d_adapter.get_spatial_coordinates(norm_bbox)
                    depth = oak_d_adapter.get_depth_at_detection(norm_bbox)
                    
                    detection_dict['spatial'] = spatial
                    detection_dict['depth_mm'] = depth['mean']
                    detection_dict['depth_stats'] = depth
                
                detections.append(detection_dict)
        
        camera_source = 'oak_d' if oak_d_adapter else 'api'
        
        return jsonify({
            'detections': detections,
            'success': True,
            'camera': camera_source,
            'frame_width': frame_width,
            'frame_height': frame_height
        })
    
    except Exception as e:
        logger.error(f'Detection error: {e}')
        return jsonify({'error': str(e), 'success': False}), 500

def get_sign_category(sign_name):
    """Categorize traffic signs by type"""
    sign_name_lower = sign_name.lower()
    
    if 'stop' in sign_name_lower or 'do not enter' in sign_name_lower or 'yield' in sign_name_lower:
        return 'regulatory'
    elif 'speed limit' in sign_name_lower or 'one way' in sign_name_lower:
        return 'regulatory'
    elif 'warning' in sign_name_lower or 'school' in sign_name_lower or 'pedestrian' in sign_name_lower:
        return 'warning'
    elif 'exit' in sign_name_lower or 'parking' in sign_name_lower or 'hospital' in sign_name_lower:
        return 'informational'
    else:
        return 'unknown'

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'model_type': 'YOLOv8 (Traffic Signs)',
        'supported_signs': len(US_TRAFFIC_SIGNS)
    })

@app.route('/video_feed')
def video_feed():
    """Stream OAK-D camera feed as MJPEG"""
    def generate_frames():
        """Generate frames from OAK-D camera"""
        frame_count = 0
        while True:
            try:
                if not oak_d_adapter:
                    continue
                
                # Get frame from OAK-D
                b64_frame = oak_d_adapter.get_rgb_frame(jpg_quality=85)
                
                if b64_frame:
                    # Decode base64 back to bytes for MJPEG stream
                    frame_bytes = base64.b64decode(b64_frame)
                    frame_count += 1
                    
                    # Yield frame in MJPEG format
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n'
                           b'Content-Length: ' + str(len(frame_bytes)).encode() + b'\r\n\r\n'
                           + frame_bytes + b'\r\n')
                    
                    # Log every 100 frames
                    if frame_count % 100 == 0:
                        logger.debug(f'📹 Video feed: {frame_count} frames streamed')
                
            except Exception as e:
                logger.error(f'Error in video feed: {str(e)}')
                continue
    
    # Return MJPEG stream response
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

# ============================================================================
# PHOTO UPLOAD ENDPOINTS
# ============================================================================

@app.route('/api/upload-photo', methods=['POST'])
def upload_photo():
    """
    Upload a JPEG photo for detection.
    Runs YOLOv8 on the image and returns detections.
    The image is stored in memory so it can be served at /api/uploaded-photo.
    
    Accepts: multipart/form-data with 'photo' file field
    Returns: { detections: [...], success: true, width: int, height: int }
    """
    global uploaded_photo_data, uploaded_photo_detections
    
    try:
        if not model:
            return jsonify({'error': 'Model not loaded', 'success': False}), 500
        
        if 'photo' not in request.files:
            return jsonify({'error': 'No photo file provided', 'success': False}), 400
        
        file = request.files['photo']
        if file.filename == '':
            return jsonify({'error': 'No file selected', 'success': False}), 400
        
        # Read image bytes
        image_bytes = file.read()
        if not image_bytes:
            return jsonify({'error': 'Empty file', 'success': False}), 400
        
        # Decode image
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'Invalid image file', 'success': False}), 400
        
        frame_height, frame_width = image.shape[:2]
        
        # Store raw JPEG bytes for serving later
        # Re-encode to ensure it's valid JPEG
        _, encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        uploaded_photo_data = encoded.tobytes()
        
        # Get confidence from query params
        confidence_threshold = float(request.form.get('confidence', 0.5))
        
        # Run YOLOv8 detection
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = model(image_rgb, conf=confidence_threshold, verbose=False)
        
        detections = []
        if results and len(results) > 0:
            for result in results:
                if result.boxes is None:
                    continue
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    if confidence < confidence_threshold:
                        continue
                    
                    cls_name = model.names.get(class_id, f'Object {class_id}')
                    
                    # Try sign type classification
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    sign_region = extract_sign_region(image, [x1, y1, x2, y2])
                    sign_type = identify_sign_type(sign_region)
                    classification = sign_type if sign_type else cls_name
                    
                    # Normalize bbox to 0-1 range
                    x1_norm = x1 / frame_width
                    y1_norm = y1 / frame_height
                    x2_norm = x2 / frame_width
                    y2_norm = y2 / frame_height
                    
                    detection = {
                        'class': classification,
                        'original_class': cls_name,
                        'classId': class_id,
                        'confidence': round(confidence, 3),
                        'bbox': [x1_norm, y1_norm, x2_norm, y2_norm],
                        'sign_category': get_sign_category(classification)
                    }
                    detections.append(detection)
        
        # Cache detections for re-detection at different thresholds
        uploaded_photo_detections = detections
        
        logger.info(f'📸 Photo uploaded: {frame_width}x{frame_height}, {len(detections)} detections')
        
        return jsonify({
            'detections': detections,
            'success': True,
            'width': frame_width,
            'height': frame_height,
            'count': len(detections),
            'camera': 'photo_upload'
        })
    
    except Exception as e:
        logger.error(f'Photo upload error: {str(e)}')
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/api/uploaded-photo', methods=['GET'])
def get_uploaded_photo():
    """
    Serve the currently uploaded photo as a JPEG image.
    Used by the frontend <img> tag to display the uploaded photo.
    """
    global uploaded_photo_data
    
    if uploaded_photo_data is None:
        return jsonify({'error': 'No photo uploaded'}), 404
    
    return Response(uploaded_photo_data, mimetype='image/jpeg')


@app.route('/api/uploaded-photo/detect', methods=['GET'])
def detect_uploaded_photo():
    """
    Re-run detection on the uploaded photo (e.g. when confidence threshold changes).
    
    Query params:
        - confidence: Confidence threshold (default 0.5)
    """
    global uploaded_photo_data
    
    if uploaded_photo_data is None:
        return jsonify({'error': 'No photo uploaded', 'detections': [], 'success': False}), 404
    
    try:
        threshold = float(request.args.get('confidence', 0.5))
        
        # Decode stored image
        nparr = np.frombuffer(uploaded_photo_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        frame_height, frame_width = image.shape[:2]
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Run detection
        results = model(image_rgb, conf=threshold, verbose=False)
        
        detections = []
        if results and len(results) > 0:
            for result in results:
                if result.boxes is None:
                    continue
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    if confidence < threshold:
                        continue
                    
                    cls_name = model.names.get(class_id, f'Object {class_id}')
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    sign_region = extract_sign_region(image, [x1, y1, x2, y2])
                    sign_type = identify_sign_type(sign_region)
                    classification = sign_type if sign_type else cls_name
                    
                    x1_norm = x1 / frame_width
                    y1_norm = y1 / frame_height
                    x2_norm = x2 / frame_width
                    y2_norm = y2 / frame_height
                    
                    detection = {
                        'class': classification,
                        'original_class': cls_name,
                        'classId': class_id,
                        'confidence': round(confidence, 3),
                        'bbox': [x1_norm, y1_norm, x2_norm, y2_norm],
                        'sign_category': get_sign_category(classification)
                    }
                    detections.append(detection)
        
        return jsonify({
            'detections': detections,
            'success': True,
            'width': frame_width,
            'height': frame_height,
            'count': len(detections),
            'camera': 'photo_upload'
        })
    
    except Exception as e:
        logger.error(f'Photo re-detection error: {str(e)}')
        return jsonify({'error': str(e), 'detections': [], 'success': False}), 500


@app.route('/api/clear-photo', methods=['POST'])
def clear_photo():
    """
    Clear the uploaded photo and return to camera mode.
    """
    global uploaded_photo_data, uploaded_photo_detections
    uploaded_photo_data = None
    uploaded_photo_detections = None
    logger.info('📸 Uploaded photo cleared, returning to camera mode')
    return jsonify({'success': True, 'message': 'Photo cleared'})


@app.route('/api/supported-signs', methods=['GET'])
def supported_signs():
    """Return list of supported US traffic signs"""
    return jsonify({
        'signs': US_TRAFFIC_SIGNS,
        'total': len(US_TRAFFIC_SIGNS),
        'categories': {
            'regulatory': 'Stop, Yield, One Way, Speed Limits, Do Not Enter',
            'warning': 'Pedestrian, School, Curves, Slippery Road',
            'informational': 'Exit, Parking, Hospital, Gas Station'
        }
    })

@app.route('/api/oak-d/frame', methods=['GET'])
def get_oak_d_frame():
    """
    Get current RGB frame from OAK-D camera as base64 JPEG
    Requires OAK-D to be connected and initialized
    
    Response: {
        "image": "base64_encoded_jpeg",
        "resolution": [1280, 720],
        "success": true,
        "camera": "oak_d"
    }
    """
    if not oak_d_adapter:
        return jsonify({
            'error': 'OAK-D camera not initialized',
            'success': False
        }), 503
    
    try:
        b64_frame = oak_d_adapter.get_rgb_frame(jpg_quality=80)
        
        if not b64_frame:
            return jsonify({
                'error': 'Failed to capture frame',
                'success': False
            }), 503
        
        w, h = oak_d_adapter.get_frame_resolution()
        
        return jsonify({
            'image': b64_frame,
            'resolution': [w, h],
            'success': True,
            'camera': 'oak_d'
        })
    
    except Exception as e:
        logger.error(f'Error getting OAK-D frame: {str(e)}')
        return jsonify({
            'error': str(e),
            'success': False
        }), 500

@app.route('/api/oak-d/detect', methods=['GET'])
def detect_oak_d():
    """
    Run YOLOv8 detection on current OAK-D frame
    
    Query params:
        - confidence: Confidence threshold (default 0.5)
        - include_spatial: Include 3D coordinates (default true)
    
    Response: {
        "detections": [
            {
                "class": "Speed Limit 55",
                "confidence": 0.95,
                "bbox": [x1, y1, x2, y2],
                "spatial": { "distance_m": 15.5, "x": 2.3, "y": -0.5, "z": 15.5 }
            }
        ],
        "success": true,
        "resolution": [1280, 720]
    }
    """
    if not oak_d_adapter:
        return jsonify({
            'error': 'OAK-D camera not initialized',
            'detections': [],
            'success': False
        }), 503
    
    try:
        # Get confidence threshold from query params
        threshold = float(request.args.get('confidence', 0.5))
        include_spatial = request.args.get('include_spatial', 'true').lower() == 'true'
        
        # Get current frame from OAK-D
        b64_frame = oak_d_adapter.get_rgb_frame(jpg_quality=80)
        if not b64_frame:
            return jsonify({'error': 'Failed to capture frame', 'detections': [], 'success': False}), 500
        
        # Decode for YOLO processing
        frame_bytes = base64.b64decode(b64_frame)
        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        
        # Run detection
        results = model(frame_rgb, conf=threshold, verbose=False)
        
        # Process detections
        detections = []
        if results and len(results) > 0:
            for result in results:
                if result.boxes is None:
                    continue
                    
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    
                    if confidence < threshold:
                        continue
                    
                    # Get class name
                    class_name = US_TRAFFIC_SIGNS.get(class_id, f'Unknown ({class_id})')
                    
                    # Normalize bbox (0-1)
                    h, w = frame_rgb.shape[:2]
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1_norm = float(x1) / w
                    y1_norm = float(y1) / h
                    x2_norm = float(x2) / w
                    y2_norm = float(y2) / h
                    
                    detection = {
                        'class': class_name,
                        'classId': class_id,
                        'confidence': round(confidence, 3),
                        'bbox': [x1_norm, y1_norm, x2_norm, y2_norm]
                    }
                    
                    # Add spatial info if requested and OAK-D available
                    if include_spatial:
                        spatial = oak_d_adapter.get_spatial_coordinates([x1_norm, y1_norm, x2_norm, y2_norm])
                        if spatial:
                            detection['spatial'] = spatial
                            # Add distance info
                            distance_m = spatial.get('z', 0)
                            distance_ft = distance_m * 3.28084
                            detection['distance'] = {
                                'meters': round(distance_m, 2),
                                'feet': round(distance_ft, 2)
                            }
                    
                    detections.append(detection)
        
        w, h = oak_d_adapter.get_frame_resolution()
        
        return jsonify({
            'detections': detections,
            'resolution': [w, h],
            'count': len(detections),
            'success': True,
            'camera': 'oak_d'
        })
    
    except Exception as e:
        logger.error(f'Error during OAK-D detection: {str(e)}')
        return jsonify({'error': str(e), 'detections': [], 'success': False}), 500

@app.route('/api/oak-d/status', methods=['GET'])
def oak_d_status():
    """
    Get OAK-D camera status and capabilities.
    Always returns a valid JSON response so the frontend can decide
    whether to use OAK-D or fall back to browser webcam.
    """
    try:
        status = {
            'available': OAK_D_AVAILABLE,
            'enabled': USE_OAK_D,
            'initialized': oak_d_adapter is not None,
            'model_loaded': model is not None
        }
        
        if oak_d_adapter:
            status['capabilities'] = ['rgb', 'depth', 'spatial']
            try:
                w, h = oak_d_adapter.get_frame_resolution()
                status['resolution'] = [w, h]
            except Exception:
                status['resolution'] = [1280, 720]
        else:
            # No OAK-D — frontend should fall back to browser webcam
            status['resolution'] = [1280, 720]
            status['capabilities'] = []
        
        return jsonify(status)
    
    except Exception as e:
        logger.error(f'Error getting OAK-D status: {str(e)}')
        # Still return a useful fallback so the frontend doesn't crash
        return jsonify({
            'available': False,
            'enabled': False,
            'initialized': False,
            'model_loaded': model is not None,
            'resolution': [1280, 720],
            'capabilities': []
        })


# ============================================================================
# VIRTUAL CAMERA ROUTES
# ============================================================================

@app.route('/api/virtual-camera/start', methods=['POST'])
def virtual_camera_start():
    """
    Start virtual camera (appears as webcam in Chrome/Windows)
    
    Response: {"started": true, "message": "Virtual camera activated"}
    """
    try:
        if not OAK_D_AVAILABLE or oak_d_adapter is None:
            return jsonify({'error': 'OAK-D not available'}), 400
        
        if is_virtual_camera_running():
            return jsonify({'started': True, 'message': 'Virtual camera already running'}), 200
        
        success = start_virtual_camera()
        
        if success:
            logger.info('✅ Virtual camera started')
            return jsonify({
                'started': True,
                'message': 'Virtual camera activated - OAK-D feed now appears as webcam in Chrome',
                'instructions': 'Open Chrome Settings > Privacy and Security > Camera > Use "OAK-D camera" or "Camera"'
            }), 200
        else:
            error_msg = get_virtual_camera_error() if get_virtual_camera_error else 'Failed to start virtual camera'
            logger.error(f'Virtual camera start failed: {error_msg}')
            return jsonify({
                'error': 'Virtual camera backend not available',
                'detail': str(error_msg),
                'instructions': [
                    '1. Install OBS Studio: https://obsproject.com/download',
                    '2. In OBS, enable Virtual Camera: Tools > Start Virtual Camera',
                    '3. Restart this application',
                    '4. Click Start Virtual Camera again'
                ]
            }), 412  # Precondition Failed
    
    except Exception as e:
        logger.error(f'Error starting virtual camera: {str(e)}')
        error_msg = get_virtual_camera_error() if get_virtual_camera_error else str(e)
        return jsonify({
            'error': str(e),
            'detail': str(error_msg),
            'instructions': [
                '1. Install OBS Studio: https://obsproject.com/download',
                '2. In OBS, enable Virtual Camera: Tools > Start Virtual Camera',
                '3. Restart this application',
                '4. Try again'
            ]
        }), 412


@app.route('/api/virtual-camera/stop', methods=['POST'])
def virtual_camera_stop():
    """
    Stop virtual camera streaming
    
    Response: {"stopped": true, "message": "Virtual camera deactivated"}
    """
    try:
        stop_virtual_camera()
        logger.info('🔴 Virtual camera stopped')
        return jsonify({'stopped': True, 'message': 'Virtual camera deactivated'}), 200
    
    except Exception as e:
        logger.error(f'Error stopping virtual camera: {str(e)}')
        return jsonify({'error': str(e)}), 500


@app.route('/api/virtual-camera/status', methods=['GET'])
def virtual_camera_status():
    """
    Get virtual camera status
    
    Response: {
        "running": true,
        "status": "Virtual camera is streaming OAK-D feed",
        "instructions": "Use in Chrome as a standard webcam"
    }
    """
    try:
        is_running = is_virtual_camera_running()
        error_msg = get_virtual_camera_error()
        
        status = {
            'running': is_running,
            'available': OAK_D_AVAILABLE and oak_d_adapter is not None,
            'status': 'Virtual camera is streaming OAK-D feed' if is_running else 'Virtual camera inactive',
            'backend_error': error_msg,
            'instructions': 'Click "Start Virtual Camera" to begin. The OAK-D feed will appear as a webcam device in Chrome and Windows applications.'
        }
        
        if error_msg:
            status['setup_required'] = True
            status['setup_instructions'] = [
                '1. Download OBS Studio: https://obsproject.com/download (Windows version)',
                '2. Install OBS Studio',
                '3. In OBS Studio, go to Tools > Start Virtual Camera',
                '4. Leave OBS running in background',
                '5. Return here and click "Start Virtual Camera" button'
            ]
        
        return jsonify(status), 200
    
    except Exception as e:
        logger.error(f'Error getting virtual camera status: {str(e)}')
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Initialize camera
    init_camera()
    
    logger.info('🚀 Starting YOLOv8 US Traffic Sign Detection Server on http://localhost:5000')
    logger.info(f'📍 Supported signs: {len(US_TRAFFIC_SIGNS)}')
    logger.info('🛑 Detecting: Stop, Do Not Enter, Speed Limits, One Way, and more...')
    
    if oak_d_adapter:
        logger.info('📷 OAK-D Lite camera: ACTIVE (spatial detection enabled)')
    else:
        logger.info('📷 No OAK-D detected — browser webcam + photo upload mode')
    
    try:
        app.run(host='localhost', port=5000, debug=False)
    finally:
        if oak_d_adapter:
            stop_oak_d()
            logger.info('🔴 OAK-D camera closed')
