"""
Screen Capture Module for Clash Royale
Handles real-time frame capture from Android emulator
"""

import cv2
import numpy as np
import time
import threading
from typing import Optional, Tuple, Callable
from dataclasses import dataclass
from queue import Queue, Empty
import logging

from ..emulator.adb_controller import ADBController

logger = logging.getLogger(__name__)


@dataclass
class CaptureConfig:
    """Configuration for screen capture"""
    fps: int = 30
    resolution: Tuple[int, int] = (1280, 720)
    buffer_size: int = 10
    save_frames: bool = False
    frame_save_path: str = "data/frames/"


class ScreenCapture:
    """Real-time screen capture from emulator"""
    
    def __init__(self, adb_controller: ADBController, config: CaptureConfig):
        self.adb = adb_controller
        self.config = config
        self.frame_queue = Queue(maxsize=config.buffer_size)
        self.is_capturing = False
        self.capture_thread = None
        self.frame_count = 0
        self.fps_counter = 0
        self.last_fps_time = time.time()
        
    def start_capture(self) -> bool:
        """Start continuous frame capture"""
        if self.is_capturing:
            logger.warning("Capture already running")
            return True
            
        if not self.adb.is_connected:
            logger.error("ADB not connected")
            return False
            
        self.is_capturing = True
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        
        logger.info(f"Started screen capture at {self.config.fps} FPS")
        return True
    
    def stop_capture(self):
        """Stop frame capture"""
        self.is_capturing = False
        if self.capture_thread:
            self.capture_thread.join(timeout=2.0)
        
        # Clear remaining frames
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except Empty:
                break
                
        logger.info("Stopped screen capture")
    
    def get_latest_frame(self, timeout: float = 0.1) -> Optional[np.ndarray]:
        """Get the most recent frame"""
        try:
            # Get the latest frame, discard older ones
            frame = None
            while not self.frame_queue.empty():
                try:
                    frame = self.frame_queue.get_nowait()
                except Empty:
                    break
            
            # If no frames in queue, wait for new one
            if frame is None:
                frame = self.frame_queue.get(timeout=timeout)
                
            return frame
        except Empty:
            return None
        except Exception as e:
            logger.error(f"Error getting frame: {e}")
            return None
    
    def get_current_fps(self) -> float:
        """Get current capture FPS"""
        current_time = time.time()
        if current_time - self.last_fps_time >= 1.0:
            fps = self.fps_counter / (current_time - self.last_fps_time)
            self.fps_counter = 0
            self.last_fps_time = current_time
            return fps
        return 0.0
    
    def _capture_loop(self):
        """Main capture loop running in separate thread"""
        frame_interval = 1.0 / self.config.fps
        last_capture_time = 0
        
        while self.is_capturing:
            current_time = time.time()
            
            # Maintain target FPS
            if current_time - last_capture_time < frame_interval:
                time.sleep(0.001)  # Small sleep to prevent busy waiting
                continue
            
            try:
                # Capture frame via ADB
                temp_path = f"/tmp/cr_frame_{self.frame_count}.png"
                if self.adb.capture_screen(temp_path):
                    # Load and process frame
                    frame = cv2.imread(temp_path)
                    if frame is not None:
                        # Resize if needed
                        if frame.shape[:2] != self.config.resolution[::-1]:
                            frame = cv2.resize(frame, self.config.resolution)
                        
                        # Add frame to queue (non-blocking)
                        try:
                            self.frame_queue.put_nowait(frame)
                            self.frame_count += 1
                            self.fps_counter += 1
                            
                            # Save frame if enabled
                            if self.config.save_frames:
                                save_path = f"{self.config.frame_save_path}frame_{self.frame_count:06d}.png"
                                cv2.imwrite(save_path, frame)
                                
                        except:
                            # Queue is full, skip this frame
                            pass
                    
                last_capture_time = current_time
                
            except Exception as e:
                logger.error(f"Capture error: {e}")
                time.sleep(0.1)  # Brief pause on error
    
    def save_frame(self, frame: np.ndarray, filename: str) -> bool:
        """Save a single frame to disk"""
        try:
            cv2.imwrite(filename, frame)
            return True
        except Exception as e:
            logger.error(f"Error saving frame: {e}")
            return False
    
    def get_frame_info(self, frame: np.ndarray) -> dict:
        """Get information about a frame"""
        if frame is None:
            return {}
            
        return {
            "shape": frame.shape,
            "dtype": frame.dtype,
            "size_bytes": frame.nbytes,
            "channels": frame.shape[2] if len(frame.shape) == 3 else 1,
            "timestamp": time.time()
        }


class FrameProcessor:
    """Process captured frames for ML pipeline"""
    
    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        self.target_size = target_size
        
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame for neural network input"""
        if frame is None:
            return None
            
        try:
            # Resize to target size
            processed = cv2.resize(frame, self.target_size)
            
            # Convert BGR to RGB
            processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
            
            # Normalize to [0, 1]
            processed = processed.astype(np.float32) / 255.0
            
            # Add batch dimension and transpose to CHW format
            processed = np.transpose(processed, (2, 0, 1))
            processed = np.expand_dims(processed, axis=0)
            
            return processed
            
        except Exception as e:
            logger.error(f"Frame preprocessing error: {e}")
            return None
    
    def extract_game_region(self, frame: np.ndarray) -> np.ndarray:
        """Extract the main game area from full screen"""
        if frame is None:
            return None
            
        # Clash Royale game area is typically centered
        # These coordinates may need adjustment based on emulator setup
        height, width = frame.shape[:2]
        
        # Calculate game area (assuming 16:9 aspect ratio game in center)
        game_width = min(width, int(height * 16/9))
        game_height = min(height, int(width * 9/16))
        
        x_start = (width - game_width) // 2
        y_start = (height - game_height) // 2
        
        game_region = frame[y_start:y_start+game_height, x_start:x_start+game_width]
        return game_region
    
    def detect_ui_elements(self, frame: np.ndarray) -> dict:
        """Detect UI elements positions (elixir bar, cards, etc.)"""
        # This will be implemented with template matching or trained models
        # For now, return estimated positions based on typical UI layout
        
        height, width = frame.shape[:2]
        
        ui_elements = {
            "elixir_bar": {
                "x": width // 2 - 50,
                "y": height - 100,
                "width": 100,
                "height": 20
            },
            "card_slots": [
                {"x": width//2 - 150 + i*75, "y": height - 60, "width": 60, "height": 80}
                for i in range(4)
            ],
            "next_card": {
                "x": width - 80,
                "y": height - 60,
                "width": 60,
                "height": 80
            }
        }
        
        return ui_elements
