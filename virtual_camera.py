#!/usr/bin/env python3
"""
Virtual Camera for OAK-D Lite
Creates a virtual webcam that streams OAK-D feed
Chrome and other apps can use this as a standard camera

NOTE: On Windows, this requires one of:
1. OBS Virtual Camera (free, install OBS Studio)
2. ManyCam (commercial)
3. WebcamXP (commercial)
4. VirtualDub with Dshow bridge

Alternatively, use Chrome's screen/tab sharing feature:
1. Open http://localhost:5000 in Chrome
2. Click "Start Virtual Camera"
3. In any app that asks for camera permission, choose "This Tab"
"""

import logging
import cv2
import numpy as np
import threading
from typing import Optional, Callable

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import pyvirtualcam, but provide graceful fallback
try:
    import pyvirtualcam
    PYVIRTUALCAM_AVAILABLE = True
except ImportError:
    PYVIRTUALCAM_AVAILABLE = False
    logger.warning("⚠️  pyvirtualcam not installed. Virtual camera disabled. Install with: pip install pyvirtualcam")


class VirtualCamera:
    """
    Virtual camera that streams frames to a virtual device
    Appears as standard webcam to Chrome and other applications
    """
    
    def __init__(self, width: int = 1280, height: int = 720, fps: int = 30):
        """
        Initialize virtual camera
        
        Args:
            width: Frame width
            height: Frame height
            fps: Frames per second
        """
        self.width = width
        self.height = height
        self.fps = fps
        self.camera = None
        self.is_running = False
        self.thread = None
        self.frame_source: Optional[Callable] = None
        self.current_frame = None
        self._lock = threading.Lock()
        self.backend_available = False
        
    def start(self, frame_source: Callable):
        """
        Start virtual camera streaming
        
        Args:
            frame_source: Callable that returns frames, signature: () -> np.ndarray (BGR format)
        """
        if self.is_running:
            logger.warning("⚠️  Virtual camera already running")
            return
        
        if not PYVIRTUALCAM_AVAILABLE:
            logger.error("❌ pyvirtualcam not available")
            return
        
        self.frame_source = frame_source
        self.is_running = True
        
        try:
            # Try to create virtual camera
            self.camera = pyvirtualcam.Camera(
                width=self.width,
                height=self.height,
                fps=self.fps,
                fmt=pyvirtualcam.PixelFormat.BGR
            )
            self.backend_available = True
            logger.info(f"✅ Virtual camera created: {self.width}x{self.height} @ {self.fps}fps")
            logger.info("💡 Virtual camera backend AVAILABLE - OAK-D feed is now a webcam device")
            
            # Start streaming thread
            self.thread = threading.Thread(target=self._stream_loop, daemon=True)
            self.thread.start()
            logger.info("🟢 Virtual camera streaming started")
            
        except RuntimeError as e:
            # Backend not available (OBS not installed, etc)
            error_msg = str(e)
            logger.error(f"❌ Virtual camera backend not available: {error_msg}")
            logger.warning("⚠️  SETUP REQUIRED - Install one of:")
            logger.warning("   1. OBS Virtual Camera: https://obsproject.com/download (choose OBS Studio)")
            logger.warning("   2. Then restart this application")
            logger.warning("   3. OR use Chrome's built-in screen/tab sharing instead")
            
            self.is_running = False
            self.backend_available = False
            raise RuntimeError(f"Virtual camera backend unavailable. {error_msg}. Install OBS Virtual Camera.")
        
        except Exception as e:
            logger.error(f"❌ Unexpected error creating virtual camera: {str(e)}")
            self.is_running = False
            self.backend_available = False
            raise
    
    def stop(self):
        """Stop virtual camera streaming"""
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=2)
        if self.camera:
            self.camera.close()
        logger.info("🔴 Virtual camera stopped")
    
    def _stream_loop(self):
        """Frame streaming loop"""
        frame_count = 0
        while self.is_running:
            try:
                if self.frame_source is None:
                    continue
                
                # Get frame from OAK-D
                frame = self.frame_source()
                
                if frame is None:
                    continue
                
                # Ensure frame is in correct format and size
                frame = self._prepare_frame(frame)
                
                with self._lock:
                    self.current_frame = frame.copy()
                
                # Send to virtual camera
                self.camera.send(frame)
                frame_count += 1
                
                # Log status every 30 frames
                if frame_count % 30 == 0:
                    logger.debug(f"📹 Virtual camera: {frame_count} frames sent")
                    
            except Exception as e:
                logger.error(f"❌ Error in virtual camera stream: {str(e)}")
                continue
    
    def _prepare_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Prepare frame for virtual camera
        
        Args:
            frame: Input frame (any size, any format)
            
        Returns:
            Prepared frame (target resolution, BGR format)
        """
        # Ensure uint8
        if frame.dtype != np.uint8:
            if frame.max() > 1:
                frame = np.uint8(frame)
            else:
                frame = np.uint8(frame * 255)
        
        # Convert grayscale to BGR if needed
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.shape[2] == 4:  # BGRA to BGR
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        elif frame.shape[2] == 3 and frame.shape[0:2] != (self.height, self.width):
            # Need to check if it's RGB instead of BGR
            pass
        
        # Resize to target resolution if needed
        if frame.shape[0:2] != (self.height, self.width):
            frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        
        return frame
    
    def get_current_frame(self) -> Optional[np.ndarray]:
        """Get the current frame being streamed"""
        with self._lock:
            if self.current_frame is not None:
                return self.current_frame.copy()
            return None


class VirtualCameraManager:
    """Manages virtual camera lifecycle"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.virtual_camera: Optional[VirtualCamera] = None
            self.initialized = True
            self.backend_error = None
    
    def start_virtual_camera(self, frame_source: Callable, width: int = 1280, height: int = 720, fps: int = 30):
        """
        Start the virtual camera
        
        Args:
            frame_source: Function that returns BGR frames
            width: Frame width
            height: Frame height
            fps: Frames per second
        """
        if self.virtual_camera and self.virtual_camera.is_running:
            logger.warning("⚠️  Virtual camera already running")
            return
        
        try:
            self.virtual_camera = VirtualCamera(width=width, height=height, fps=fps)
            self.virtual_camera.start(frame_source)
            self.backend_error = None
        except RuntimeError as e:
            self.backend_error = str(e)
            logger.error(f"Failed to start virtual camera: {self.backend_error}")
            raise
    
    def stop_virtual_camera(self):
        """Stop the virtual camera"""
        if self.virtual_camera:
            self.virtual_camera.stop()
            self.virtual_camera = None
    
    def is_running(self) -> bool:
        """Check if virtual camera is running"""
        return self.virtual_camera is not None and self.virtual_camera.is_running
    
    def get_backend_error(self) -> Optional[str]:
        """Get any backend errors"""
        return self.backend_error


# Global instance
_virtual_camera_manager = VirtualCameraManager()


def start_virtual_camera(frame_source: Callable, width: int = 1280, height: int = 720, fps: int = 30):
    """Start virtual camera"""
    _virtual_camera_manager.start_virtual_camera(frame_source, width, height, fps)


def stop_virtual_camera():
    """Stop virtual camera"""
    _virtual_camera_manager.stop_virtual_camera()


def is_virtual_camera_running() -> bool:
    """Check if virtual camera is running"""
    return _virtual_camera_manager.is_running()


def get_virtual_camera_error() -> Optional[str]:
    """Get any virtual camera backend errors"""
    return _virtual_camera_manager.get_backend_error()
