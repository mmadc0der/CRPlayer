#!/usr/bin/env python3
"""
Data Collector Consumer Implementation
Separate module for data collection functionality.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
import json
import time
from datetime import datetime

from core.stream_pipeline import StreamConsumer, FrameData


class DataCollectorConsumer(StreamConsumer):
    """Consumer that saves frames to disk for training data with metadata."""
    
    def __init__(self, consumer_id: str, stream_buffer, 
                 output_dir: str = "collected_data",
                 session_name: Optional[str] = None,
                 game_name: str = "unknown_game",
                 save_format: str = "jpg",
                 quality: int = 85,
                 sample_rate: int = 1,
                 sparsity_mode: str = "uniform",
                 max_frames: Optional[int] = None):
        """
        Initialize data collector consumer.
        
        Args:
            consumer_id: Unique consumer identifier
            stream_buffer: Shared stream buffer
            output_dir: Directory to save frames
            session_name: Optional session name
            game_name: Name of the game being recorded
            save_format: Image format (jpg/png)
            quality: JPEG quality (1-100)
            sample_rate: Save every Nth frame (1 = all frames, 10 = every 10th frame)
            sparsity_mode: Sampling mode ('uniform', 'random', 'adaptive')
            max_frames: Maximum frames to save (None = unlimited)
        """
        super().__init__(consumer_id, stream_buffer)
        
        self.output_dir = Path(output_dir)
        self.save_format = save_format.lower()
        self.quality = quality
        self.sample_rate = sample_rate
        self.sparsity_mode = sparsity_mode
        self.max_frames = max_frames
        
        # Sparsity tracking
        self.random_threshold = 1.0 / sample_rate if sparsity_mode == "random" else None
        self.adaptive_counter = 0
        
        # Session management
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_id = session_name or f"session_{timestamp}"
        self.game_name = game_name
        
        # Create session directory
        self.session_dir = self.output_dir / self.session_id
        self.session_dir.mkdir(parents=True, exist_ok=True)
        
        # Counters
        self.frame_count = 0
        self.saved_count = 0
        
        # Session metadata
        self.session_metadata = {
            "session_id": self.session_id,
            "game_name": game_name,
            "consumer_id": consumer_id,
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "save_format": save_format,
            "quality": quality,
            "sample_rate": sample_rate,
            "frames_processed": 0,
            "frames_saved": 0,
            "frames": []
        }
        
        print(f"[DATA] Data collector initialized: {self.session_id}")
        print(f"[DATA] Output directory: {self.session_dir}")
        print(f"[DATA] Sample rate: 1/{sample_rate} frames ({sparsity_mode} mode)")
        if max_frames:
            print(f"[DATA] Max frames limit: {max_frames}")
    
    def process_frame(self, frame_data: FrameData) -> bool:
        """Save frame to disk with metadata."""
        try:
            self.frame_count += 1
            
            # Check max frames limit
            if self.max_frames and self.saved_count >= self.max_frames:
                return True
            
            # Apply sparsity sampling
            if not self._should_save_frame():
                return True
            
            # Convert tensor to numpy
            frame_np = frame_data.tensor.cpu().numpy()
            frame_np = np.transpose(frame_np, (1, 2, 0))  # CHW to HWC
            frame_np = (frame_np * 255).astype(np.uint8)
            
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
            
            # Generate filename
            frame_filename = f"frame_{self.saved_count:06d}.{self.save_format}"
            frame_path = self.session_dir / frame_filename
            
            # Save frame
            if self.save_format == "jpg":
                cv2.imwrite(str(frame_path), frame_bgr, 
                           [cv2.IMWRITE_JPEG_QUALITY, self.quality])
            else:  # png
                cv2.imwrite(str(frame_path), frame_bgr)
            
            # Update metadata
            frame_metadata = {
                "frame_id": frame_data.frame_id,
                "saved_id": self.saved_count,
                "filename": frame_filename,
                "timestamp": frame_data.timestamp,
                "pts": frame_data.pts,
                "shape": list(frame_data.tensor.shape),
                "game_state": None,  # To be annotated later
                "importance": None,  # To be annotated later
                "metadata": frame_data.metadata
            }
            
            self.session_metadata["frames"].append(frame_metadata)
            self.saved_count += 1
            
            # Print progress
            if self.saved_count % 100 == 0:
                elapsed = time.time() - self.start_time if hasattr(self, 'start_time') else 0
                fps = self.saved_count / elapsed if elapsed > 0 else 0
                size_mb = self._get_session_size()
                
                print(f"[DATA] Saved {self.saved_count} frames | "
                      f"FPS: {fps:.1f} | Size: {size_mb:.1f}MB")
            
            return True
            
        except Exception as e:
            print(f"[ERROR] Data collector error: {e}")
            return True  # Continue on error
    
    def _should_save_frame(self) -> bool:
        """Determine if current frame should be saved based on sparsity mode."""
        if self.sparsity_mode == "uniform":
            # Standard uniform sampling - every Nth frame
            return self.frame_count % self.sample_rate == 0
        
        elif self.sparsity_mode == "random":
            # Random sampling with probability 1/sample_rate
            import random
            return random.random() < self.random_threshold
        
        elif self.sparsity_mode == "adaptive":
            # Adaptive sampling - denser at start, sparser later
            self.adaptive_counter += 1
            
            # Start with higher frequency, then reduce
            if self.saved_count < 100:
                # First 100 frames: every frame
                return True
            elif self.saved_count < 500:
                # Next 400 frames: every 2nd frame
                return self.adaptive_counter % 2 == 0
            elif self.saved_count < 1000:
                # Next 500 frames: every 5th frame
                return self.adaptive_counter % 5 == 0
            else:
                # After 1000 frames: use specified sample_rate
                return self.adaptive_counter % self.sample_rate == 0
        
        else:
            # Default to uniform if unknown mode
            return self.frame_count % self.sample_rate == 0
    
    def start(self):
        """Start data collection with timing."""
        self.start_time = time.time()
        super().start()
        print(f"[DATA] Started collecting data for session: {self.session_id}")
    
    def stop(self):
        """Stop data collection and save metadata."""
        super().stop()
        
        # Finalize session metadata
        self.session_metadata["end_time"] = datetime.now().isoformat()
        self.session_metadata["frames_processed"] = self.frame_count
        self.session_metadata["frames_saved"] = self.saved_count
        
        # Save session metadata
        metadata_path = self.session_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(self.session_metadata, f, indent=2)
        
        # Print final stats
        elapsed = time.time() - self.start_time if hasattr(self, 'start_time') else 0
        avg_fps = self.saved_count / elapsed if elapsed > 0 else 0
        size_mb = self._get_session_size()
        
        print(f"[DATA] Collection completed!")
        print(f"[DATA] Session: {self.session_id}")
        print(f"[DATA] Frames processed: {self.frame_count}")
        print(f"[DATA] Frames saved: {self.saved_count}")
        print(f"[DATA] Effective sample rate: 1/{self.frame_count/self.saved_count:.1f}" if self.saved_count > 0 else "[DATA] No frames saved")
        print(f"[DATA] Duration: {elapsed:.1f}s")
        print(f"[DATA] Average FPS: {avg_fps:.1f}")
        print(f"[DATA] Total size: {size_mb:.1f}MB")
        print(f"[DATA] Metadata: {metadata_path}")
    
    def _get_session_size(self) -> float:
        """Get session size in MB."""
        total_size = sum(f.stat().st_size for f in self.session_dir.rglob('*') if f.is_file())
        return total_size / (1024 * 1024)
    
    def create_annotation_template(self) -> str:
        """Create annotation template for the collected data."""
        annotation_template = {
            "session_id": self.session_id,
            "game_name": self.game_name,
            "annotation_info": {
                "game_states": {
                    "menu": "Main menu, settings, character selection",
                    "loading": "Loading screens, transitions",
                    "battle": "Active gameplay, combat",
                    "final": "Results, game over, victory/defeat screens"
                },
                "importance_levels": {
                    0: "Low - static screen, no action",
                    1: "Medium - some activity, UI changes", 
                    2: "High - important action, key moments"
                }
            },
            "frames": []
        }
        
        # Sample frames for annotation (max 200 to keep manageable)
        sample_rate = max(1, len(self.session_metadata["frames"]) // 200)
        
        for i, frame in enumerate(self.session_metadata["frames"]):
            if i % sample_rate == 0:
                annotation_template["frames"].append({
                    "frame_id": frame["frame_id"],
                    "saved_id": frame["saved_id"],
                    "filename": frame["filename"],
                    "timestamp": frame["timestamp"],
                    "game_state": None,  # To be filled
                    "importance": None,  # To be filled
                    "notes": ""
                })
        
        # Save annotation template
        annotation_path = self.output_dir / "annotations" / f"{self.session_id}_template.json"
        annotation_path.parent.mkdir(exist_ok=True)
        
        with open(annotation_path, 'w') as f:
            json.dump(annotation_template, f, indent=2)
        
        print(f"[DATA] Annotation template created: {annotation_path}")
        print(f"[DATA] Frames to annotate: {len(annotation_template['frames'])}")
        
        return str(annotation_path)


class MonitoringConsumer(StreamConsumer):
    """Consumer that monitors stream performance."""
    
    def __init__(self, consumer_id: str, stream_buffer, 
                 stats_interval: float = 5.0):
        super().__init__(consumer_id, stream_buffer)
        self.stats_interval = stats_interval
        self.last_stats_time = time.time()
        self.frame_count = 0
        
    def process_frame(self, frame_data: FrameData) -> bool:
        """Monitor stream performance."""
        self.frame_count += 1
        
        # Print stats at specified interval
        current_time = time.time()
        if current_time - self.last_stats_time >= self.stats_interval:
            elapsed = current_time - self.last_stats_time
            fps = self.frame_count / elapsed
            
            # Get buffer stats
            buffer_stats = self.stream_buffer.get_stats()
            
            print(f"[MONITOR] FPS: {fps:.1f} | Buffer: {buffer_stats['buffer_size']} frames | "
                  f"Memory: {buffer_stats['memory_usage_mb']:.1f}MB | "
                  f"Consumers: {buffer_stats['active_consumers']}")
            
            self.frame_count = 0
            self.last_stats_time = current_time
        
        return True
