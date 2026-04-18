#!/usr/bin/env python3
"""
OAK-D Adapter for Flask Server
Bridges OAK-D camera to existing detection API
Provides 3D spatial coordinates for traffic signs
"""

from oak_d_integration import OAKDCamera
from virtual_camera import VirtualCamera
import logging
import base64
import cv2
import numpy as np
from typing import Optional, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OAKDAdapter:
    """
    Adapter to use OAK-D camera with existing Flask detection server
    Replaces USB webcam with OAK-D Lite
    Includes virtual camera streaming to Windows/Chrome
    """
    
    def __init__(self, use_oak_d: bool = True, rgb_resolution: Tuple[int, int] = (1280, 720)):
        """
        Initialize camera adapter
        
        Args:
            use_oak_d: Use OAK-D if True, else use webcam
            rgb_resolution: Resolution for RGB stream
        """
        self.use_oak_d = use_oak_d
        self.camera = None
        self.frame_cache = None
        self.virtual_camera = None
        
        if use_oak_d:
            try:
                self.camera = OAKDCamera(rgb_resolution=rgb_resolution, spatial_detection=True)
                self.camera.start()
                logger.info("✅ OAK-D camera initialized")
            except Exception as e:
                logger.error(f"❌ Failed to initialize OAK-D: {str(e)}")
                logger.warning("⚠️  Falling back to webcam")
                self.use_oak_d = False
    
    def get_rgb_frame(self, jpg_quality: int = 80) -> Optional[str]:
        """
        Get current RGB frame as base64 JPEG
        
        Returns:
            Base64 encoded JPEG string or None
        """
        if not self.use_oak_d:
            return None
        
        try:
            frame_data = self.camera.get_frame(timeout=0.5)
            if frame_data is None:
                return None
            
            # Cache for spatial queries
            self.frame_cache = frame_data
            
            # Get RGB frame
            rgb = frame_data['rgb']
            
            # Convert RGB to BGR for OpenCV
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            
            # Encode as JPEG
            success, buffer = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, jpg_quality])
            if not success:
                logger.error("Failed to encode frame")
                return None
            
            # Convert to base64
            b64_string = base64.b64encode(buffer).decode('utf-8')
            return b64_string
        
        except Exception as e:
            logger.error(f"Error getting RGB frame: {str(e)}")
            return None
    
    def get_frame_resolution(self) -> Tuple[int, int]:
        """Get RGB frame resolution (width, height)"""
        if self.camera:
            return self.camera.rgb_resolution
        return (1280, 720)
    
    def get_depth_at_detection(self, bbox: list) -> dict:
        """
        Get depth information for a detection bounding box
        
        Args:
            bbox: [x1, y1, x2, y2] normalized coordinates (0-1)
            
        Returns:
            Dictionary with depth stats: min, max, mean, median (in mm)
        """
        if not self.use_oak_d or self.frame_cache is None:
            return {'min': 0, 'max': 0, 'mean': 0, 'median': 0, 'distance_m': 0}
        
        try:
            # Denormalize coordinates
            w, h = self.get_frame_resolution()
            x1 = int(bbox[0] * w)
            y1 = int(bbox[1] * h)
            x2 = int(bbox[2] * w)
            y2 = int(bbox[3] * h)
            
            # Get depth stats
            depth_stats = self.camera.get_depth_in_bbox(self.frame_cache, x1, y1, x2, y2)
            
            # Convert mm to meters
            depth_stats['distance_m'] = depth_stats['mean'] / 1000.0
            depth_stats['distance_ft'] = depth_stats['mean'] / 304.8
            
            return depth_stats
        
        except Exception as e:
            logger.error(f"Error getting depth at detection: {str(e)}")
            return {'min': 0, 'max': 0, 'mean': 0, 'median': 0, 'distance_m': 0}
    
    def get_spatial_coordinates(self, bbox: list) -> dict:
        """
        Get 3D spatial coordinates (X, Y, Z) for detection
        Z = depth (distance from camera)
        X, Y = left/right, up/down relative to center
        
        Args:
            bbox: [x1, y1, x2, y2] normalized coordinates (0-1)
            
        Returns:
            Dictionary with x, y, z coordinates in meters
        """
        if not self.use_oak_d or self.frame_cache is None:
            return {'x': 0, 'y': 0, 'z': 0}
        
        try:
            # Get resolution
            w, h = self.get_frame_resolution()
            
            # Denormalize
            x1 = int(bbox[0] * w)
            y1 = int(bbox[1] * h)
            x2 = int(bbox[2] * w)
            y2 = int(bbox[3] * h)
            
            # Get center of bounding box
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            # Get depth
            depth_mm = self.camera.get_depth_at_xy(self.frame_cache, center_x, center_y)
            depth_m = depth_mm / 1000.0
            
            # Estimate X, Y based on camera intrinsics (approximation for OAK-D)
            # Approximate FOV: 70 degrees horizontal, 55 degrees vertical
            fov_h = 70  # degrees
            fov_v = 55  # degrees
            
            # Calculate X (left-right offset)
            x_angle = (center_x - w / 2) / w * (fov_h / 2)
            x_m = depth_m * np.tan(np.radians(x_angle))
            
            # Calculate Y (up-down offset)
            y_angle = (center_y - h / 2) / h * (fov_v / 2)
            y_m = depth_m * np.tan(np.radians(y_angle))
            
            return {
                'x': round(x_m, 2),
                'y': round(y_m, 2),
                'z': round(depth_m, 2),
                'distance_m': round(depth_m, 2)
            }
        
        except Exception as e:
            logger.error(f"Error getting spatial coordinates: {str(e)}")
            return {'x': 0, 'y': 0, 'z': 0}
    
    def stop(self):
        """Stop camera stream"""
        if self.virtual_camera:
            self.stop_virtual_camera()
        if self.camera:
            self.camera.stop()
            logger.info("Camera stopped")
    
    def get_raw_frame(self) -> Optional[np.ndarray]:
        """
        Get current RGB frame as numpy array (BGR format)
        
        Returns:
            BGR numpy array or None
        """
        if not self.use_oak_d:
            return None
        
        try:
            frame_data = self.camera.get_frame(timeout=0.5)
            if frame_data is None:
                return None
            
            # Cache for spatial queries
            self.frame_cache = frame_data
            
            # Get RGB frame and convert to BGR
            rgb = frame_data['rgb']
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            
            return bgr
        
        except Exception as e:
            logger.error(f"Error getting raw frame: {str(e)}")
            return None
    
    def start_virtual_camera(self) -> bool:
        """
        Start virtual camera streaming
        
        Returns:
            True if started successfully, False otherwise
        """
        if not self.use_oak_d:
            logger.warning("⚠️  OAK-D not available, cannot start virtual camera")
            return False
        
        if self.virtual_camera and self.virtual_camera.is_running:
            logger.warning("⚠️  Virtual camera already running")
            return True
        
        try:
            w, h = self.get_frame_resolution()
            self.virtual_camera = VirtualCamera(width=w, height=h, fps=30)
            
            # Define frame source function that returns BGR frames
            def frame_source():
                frame = self.get_raw_frame()
                return frame
            
            self.virtual_camera.start(frame_source)
            logger.info("✅ Virtual camera started - appears as webcam in Chrome/Windows")
            return True
        
        except Exception as e:
            logger.error(f"❌ Failed to start virtual camera: {str(e)}")
            return False
    
    def stop_virtual_camera(self):
        """Stop virtual camera streaming"""
        if self.virtual_camera:
            self.virtual_camera.stop()
            self.virtual_camera = None
            logger.info("🔴 Virtual camera stopped")
    
    def is_virtual_camera_running(self) -> bool:
        """Check if virtual camera is running"""
        return self.virtual_camera is not None and self.virtual_camera.is_running


# Global adapter instance
_adapter = None


def init_oak_d(use_oak_d: bool = True) -> Optional[OAKDAdapter]:
    """Initialize global OAK-D adapter. Returns None if OAK-D is not available."""
    global _adapter
    _adapter = OAKDAdapter(use_oak_d=use_oak_d)
    # If the adapter fell back to non-OAK-D mode, treat as not available
    if not _adapter.use_oak_d:
        logger.warning('⚠️  OAK-D adapter created but no camera available')
        _adapter = None
        return None
    return _adapter


def get_adapter() -> Optional[OAKDAdapter]:
    """Get global adapter instance"""
    return _adapter


def get_oak_d_frame() -> Optional[str]:
    """Get current frame as base64 JPEG"""
    if _adapter:
        return _adapter.get_rgb_frame()
    return None


def get_depth_for_detection(bbox: list) -> dict:
    """Get depth info for detection"""
    if _adapter:
        return _adapter.get_depth_at_detection(bbox)
    return {'min': 0, 'max': 0, 'mean': 0, 'median': 0, 'distance_m': 0}


def get_spatial_for_detection(bbox: list) -> dict:
    """Get spatial coords for detection"""
    if _adapter:
        return _adapter.get_spatial_coordinates(bbox)
    return {'x': 0, 'y': 0, 'z': 0}


def stop_oak_d():
    """Stop OAK-D camera"""
    global _adapter
    if _adapter:
        _adapter.stop()
        _adapter = None


def start_virtual_camera() -> bool:
    """Start virtual camera"""
    if _adapter:
        return _adapter.start_virtual_camera()
    return False


def stop_virtual_camera():
    """Stop virtual camera"""
    if _adapter:
        _adapter.stop_virtual_camera()


def is_virtual_camera_running() -> bool:
    """Check if virtual camera is running"""
    if _adapter:
        return _adapter.is_virtual_camera_running()
    return False


if __name__ == "__main__":
    # Test adapter
    logger.info("Testing OAK-D Adapter")
    
    adapter = init_oak_d(use_oak_d=True)
    
    # Get a few frames
    for i in range(10):
        b64_frame = adapter.get_rgb_frame()
        if b64_frame:
            logger.info(f"✅ Frame {i+1}: {len(b64_frame)} bytes")
            
            # Test depth query
            test_bbox = [0.25, 0.25, 0.75, 0.75]  # Center 50%
            depth = adapter.get_depth_at_detection(test_bbox)
            spatial = adapter.get_spatial_coordinates(test_bbox)
            
            logger.info(f"   Depth: {depth['mean']}mm ({depth['distance_m']:.2f}m)")
            logger.info(f"   Spatial: X={spatial['x']:.2f}m, Y={spatial['y']:.2f}m, Z={spatial['z']:.2f}m")
        else:
            logger.warning(f"Frame {i+1}: Failed to get frame")
    
    adapter.stop()
    logger.info("Test complete")
