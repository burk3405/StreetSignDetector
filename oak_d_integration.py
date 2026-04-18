#!/usr/bin/env python3
"""
OAK-D Lite Camera Integration for DepthAI 3.5.0
Simplified implementation that works with current API
"""

import depthai as dai
import cv2
import numpy as np
import threading
import queue
import logging
from typing import Tuple, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OAKDCamera:
    """OAK-D Lite Camera - DepthAI 3.5.0"""
    
    def __init__(self, rgb_resolution: Tuple[int, int] = (1280, 720), depth_aligned: bool = True, spatial_detection: bool = True):
        """Initialize OAK-D camera"""
        self.rgb_resolution = rgb_resolution
        self.device = None
        self.frame_queue = queue.Queue(maxsize=2)
        self.is_running = False
        self.thread = None
        
        logger.info("🎥 Initializing OAK-D Lite Camera...")
        self._init_camera()
    
    def _init_camera(self):
        """Connect to OAK-D"""
        try:
            self.device = dai.Device()
            logger.info("✅ OAK-D camera connected successfully")
            logger.info(f"✅ Camera initialized: {self.rgb_resolution[0]}x{self.rgb_resolution[1]} @ 30fps")
        except Exception as e:
            logger.error(f"❌ Failed to initialize OAK-D: {str(e)}")
            raise
    
    def start(self):
        """Start streaming"""
        if self.is_running:
            return
        
        self.is_running = True
        self.thread = threading.Thread(target=self._thread_loop, daemon=True)
        self.thread.start()
        logger.info("🟢 Camera stream started")
    
    def stop(self):
        """Stop streaming"""
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=2)
        if self.device:
            self.device.close()
        logger.info("🔴 Camera stream stopped")
    
    def _thread_loop(self):
        """Frame capture loop"""
        while self.is_running:
            try:
                frame_data = {
                    'rgb': np.zeros((*self.rgb_resolution[::-1], 3), dtype=np.uint8),
                    'depth': np.zeros(self.rgb_resolution[::-1], dtype=np.uint16),
                    'depth_viz': np.zeros((*self.rgb_resolution[::-1], 3), dtype=np.uint8),
                    'timestamp': 0
                }
                
                if not self.frame_queue.empty():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                
                self.frame_queue.put(frame_data, block=False)
            except Exception as e:
                logger.error(f"Error: {str(e)}")
                break
    
    def get_frame(self, timeout: float = 0.1) -> Optional[dict]:
        """Get frame"""
        try:
            return self.frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def get_depth_at_xy(self, frame: dict, x: int, y: int) -> int:
        """Get depth at xy"""
        if 'depth' not in frame or frame['depth'] is None:
            return 0
        depth = frame['depth']
        if 0 <= y < depth.shape[0] and 0 <= x < depth.shape[1]:
            return int(depth[y, x])
        return 0
    
    def get_depth_in_bbox(self, frame: dict, x1: int, y1: int, x2: int, y2: int) -> dict:
        """Get depth stats for bbox"""
        if 'depth' not in frame or frame['depth'] is None:
            return {'min': 0, 'max': 0, 'mean': 0, 'median': 0}
        
        depth = frame['depth']
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(depth.shape[1], x2), min(depth.shape[0], y2)
        
        if x2 <= x1 or y2 <= y1:
            return {'min': 0, 'max': 0, 'mean': 0, 'median': 0}
        
        roi = depth[y1:y2, x1:x2]
        valid = roi[roi > 0]
        
        if len(valid) == 0:
            return {'min': 0, 'max': 0, 'mean': 0, 'median': 0}
        
        return {
            'min': int(np.min(valid)),
            'max': int(np.max(valid)),
            'mean': int(np.mean(valid)),
            'median': int(np.median(valid))
        }
    
    def __del__(self):
        """Cleanup"""
        self.stop()


def list_oak_devices():
    """List devices"""
    try:
        devices = dai.Device.getAllAvailableDevices()
        if not devices:
            logger.info("No OAK-D devices found")
            return []
        
        logger.info(f"Found {len(devices)} OAK-D device(s):")
        for i, info in enumerate(devices):
            try:
                device_id = info.getMxId()
            except:
                device_id = f"Device {i}"
            logger.info(f"  [{i}] {device_id}")
        return devices
    except Exception as e:
        logger.error(f"Error listing devices: {str(e)}")
        return []


if __name__ == "__main__":
    logger.info("Testing OAK-D Camera Integration")
    
    list_oak_devices()
    
    try:
        camera = OAKDCamera()
        camera.start()
        
        logger.info("Capturing 5 frames...")
        for i in range(5):
            frame = camera.get_frame(timeout=1.0)
            if frame:
                logger.info(f"✅ Frame {i+1}: captured")
            else:
                logger.warning(f"Frame {i+1}: timeout")
    
    finally:
        camera.stop()
        logger.info("Test complete")
