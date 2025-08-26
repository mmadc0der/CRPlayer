#!/usr/bin/env python3
"""
Data Collection System for Game State Classification
Captures and stores frames with metadata for later annotation and training.
"""

import os
import json
import time
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import torch
import cv2
import numpy as np
from streamer.android_stream_gpu import GPUAndroidStreamer


class GameDataCollector:
    """Collects and stores game frames with metadata for ML training."""
    
    def __init__(self, 
                 output_dir: str = "game_data",
                 max_fps: int = 30,  # Lower FPS for data collection to save storage
                 max_size: int = 800,
                 save_format: str = "jpg",  # jpg for smaller files, png for quality
                 quality: int = 85):  # JPEG quality
        """
        Initialize data collector.
        
        Args:
            output_dir: Directory to save collected data
            max_fps: Frame rate for data collection
            max_size: Maximum frame size
            save_format: Image format (jpg/png)
            quality: JPEG quality (1-100)
        """
        self.output_dir = Path(output_dir)
        self.max_fps = max_fps
        self.max_size = max_size
        self.save_format = save_format.lower()
        self.quality = quality
        
        # Create directory structure
        self.setup_directories()
        
        # Collection state
        self.is_collecting = False
        self.frame_count = 0
        self.session_id = None
        self.session_metadata = {}
        
        # Streamer
        self.streamer = None
        
    def setup_directories(self):
        """Create necessary directories for data storage."""
        directories = [
            self.output_dir,
            self.output_dir / "frames",
            self.output_dir / "sessions",
            self.output_dir / "annotations"
        ]
        
        for dir_path in directories:
            dir_path.mkdir(parents=True, exist_ok=True)
            
        print(f"Data collection directories created in: {self.output_dir}")
    
    def start_session(self, session_name: Optional[str] = None, 
                     game_name: str = "unknown_game",
                     notes: str = "") -> str:
        """
        Start a new data collection session.
        
        Args:
            session_name: Custom session name
            game_name: Name of the game being played
            notes: Additional notes about the session
            
        Returns:
            Session ID
        """
        if self.is_collecting:
            print("Collection already in progress. Stop current session first.")
            return self.session_id
            
        # Generate session ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_id = session_name or f"session_{timestamp}"
        
        # Create session directory
        session_dir = self.output_dir / "sessions" / self.session_id
        session_dir.mkdir(exist_ok=True)
        
        # Initialize session metadata
        self.session_metadata = {
            "session_id": self.session_id,
            "game_name": game_name,
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "notes": notes,
            "frame_count": 0,
            "fps": self.max_fps,
            "resolution": f"{self.max_size}x{int(self.max_size * 9/16)}",  # Assuming 16:9
            "format": self.save_format,
            "frames": []
        }
        
        # Initialize streamer
        self.streamer = GPUAndroidStreamer(
            max_fps=self.max_fps,
            max_size=self.max_size,
            video_codec="h264",
            bit_rate="12M",
            use_gpu=True,
            buffer_size=5  # Small buffer for data collection
        )
        
        print(f"Started data collection session: {self.session_id}")
        print(f"Game: {game_name}")
        print(f"Output: {session_dir}")
        
        return self.session_id
    
    def frame_callback(self, tensor: torch.Tensor, pts: int, timestamp: float):
        """Callback function to save frames during collection."""
        if not self.is_collecting:
            return
            
        try:
            # Convert tensor to numpy array (CPU)
            frame_np = tensor.cpu().numpy()
            
            # Convert from CHW to HWC format
            frame_np = np.transpose(frame_np, (1, 2, 0))
            
            # Convert from [0,1] to [0,255] and uint8
            frame_np = (frame_np * 255).astype(np.uint8)
            
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
            
            # Generate filename
            frame_filename = f"frame_{self.frame_count:06d}.{self.save_format}"
            frame_path = self.output_dir / "sessions" / self.session_id / frame_filename
            
            # Save frame
            if self.save_format == "jpg":
                cv2.imwrite(str(frame_path), frame_bgr, 
                           [cv2.IMWRITE_JPEG_QUALITY, self.quality])
            else:  # png
                cv2.imwrite(str(frame_path), frame_bgr)
            
            # Update metadata
            frame_metadata = {
                "frame_id": self.frame_count,
                "filename": frame_filename,
                "timestamp": timestamp,
                "pts": pts,
                "shape": list(tensor.shape),
                "game_state": None,  # To be annotated later
                "importance": None,  # To be annotated later
                "objects": []  # To be annotated later
            }
            
            self.session_metadata["frames"].append(frame_metadata)
            self.frame_count += 1
            
            # Print progress every 100 frames
            if self.frame_count % 100 == 0:
                elapsed = time.time() - self.start_time
                fps = self.frame_count / elapsed
                print(f"Collected {self.frame_count} frames | {fps:.1f} FPS | "
                      f"Size: {self._get_session_size():.1f}MB")
                
        except Exception as e:
            print(f"Error saving frame {self.frame_count}: {e}")
    
    def start_collection(self):
        """Start collecting frames."""
        if not self.session_id:
            print("No active session. Call start_session() first.")
            return
            
        if self.is_collecting:
            print("Collection already in progress.")
            return
            
        self.is_collecting = True
        self.frame_count = 0
        self.start_time = time.time()
        
        # Start streaming with frame callback
        self.streamer.start_streaming(frame_callback=self.frame_callback)
        
        print("Data collection started. Press Ctrl+C to stop.")
        print("Play your game normally - all frames will be captured.")
    
    def stop_collection(self):
        """Stop collecting frames and save session metadata."""
        if not self.is_collecting:
            print("No collection in progress.")
            return
            
        self.is_collecting = False
        
        # Stop streamer
        if self.streamer:
            self.streamer.stop_streaming()
        
        # Finalize session metadata
        self.session_metadata["end_time"] = datetime.now().isoformat()
        self.session_metadata["frame_count"] = self.frame_count
        
        # Save session metadata
        metadata_path = self.output_dir / "sessions" / self.session_id / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(self.session_metadata, f, indent=2)
        
        elapsed = time.time() - self.start_time
        avg_fps = self.frame_count / elapsed if elapsed > 0 else 0
        size_mb = self._get_session_size()
        
        print(f"\nData collection completed!")
        print(f"Session: {self.session_id}")
        print(f"Frames collected: {self.frame_count}")
        print(f"Duration: {elapsed:.1f}s")
        print(f"Average FPS: {avg_fps:.1f}")
        print(f"Total size: {size_mb:.1f}MB")
        print(f"Metadata saved: {metadata_path}")
    
    def _get_session_size(self) -> float:
        """Get current session size in MB."""
        if not self.session_id:
            return 0.0
            
        session_dir = self.output_dir / "sessions" / self.session_id
        total_size = sum(f.stat().st_size for f in session_dir.rglob('*') if f.is_file())
        return total_size / (1024 * 1024)  # Convert to MB
    
    def list_sessions(self):
        """List all collected sessions."""
        sessions_dir = self.output_dir / "sessions"
        if not sessions_dir.exists():
            print("No sessions found.")
            return
            
        sessions = []
        for session_dir in sessions_dir.iterdir():
            if session_dir.is_dir():
                metadata_path = session_dir / "metadata.json"
                if metadata_path.exists():
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    sessions.append(metadata)
        
        if not sessions:
            print("No sessions found.")
            return
            
        print(f"\nFound {len(sessions)} sessions:")
        print("-" * 80)
        for session in sorted(sessions, key=lambda x: x.get('start_time', '')):
            start_time = session.get('start_time', 'Unknown')[:19].replace('T', ' ')
            frame_count = session.get('frame_count', 0)
            game_name = session.get('game_name', 'Unknown')
            session_id = session.get('session_id', 'Unknown')
            
            print(f"{session_id:20} | {game_name:15} | {frame_count:6} frames | {start_time}")
    
    def create_annotation_template(self, session_id: str):
        """Create annotation template for a session."""
        session_dir = self.output_dir / "sessions" / session_id
        metadata_path = session_dir / "metadata.json"
        
        if not metadata_path.exists():
            print(f"Session {session_id} not found.")
            return
            
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Create simplified annotation template
        annotation_template = {
            "session_id": session_id,
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
            },
            "frames": []
        }
        
        # Sample every Nth frame for annotation (to make it manageable)
        sample_rate = max(1, len(metadata["frames"]) // 200)  # Max 200 frames to annotate
        
        for i, frame in enumerate(metadata["frames"]):
            if i % sample_rate == 0:
                annotation_template["frames"].append({
                    "frame_id": frame["frame_id"],
                    "filename": frame["filename"],
                    "timestamp": frame["timestamp"],
                    "game_state": None,  # To be filled
                    "importance": None,  # To be filled
                    "notes": ""
                })
        
        # Save annotation template
        annotation_path = self.output_dir / "annotations" / f"{session_id}_template.json"
        with open(annotation_path, 'w') as f:
            json.dump(annotation_template, f, indent=2)
        
        print(f"Annotation template created: {annotation_path}")
        print(f"Frames to annotate: {len(annotation_template['frames'])}")
        print("Edit this file to add game_state and importance labels.")


def main():
    """Interactive data collection interface."""
    collector = GameDataCollector()
    
    print("=== Game Data Collector ===")
    print("Commands:")
    print("  start <game_name> [notes] - Start new session")
    print("  collect - Begin frame collection")
    print("  stop - Stop collection")
    print("  list - List all sessions")
    print("  annotate <session_id> - Create annotation template")
    print("  quit - Exit")
    print()
    
    try:
        while True:
            command = input("> ").strip().split()
            if not command:
                continue
                
            cmd = command[0].lower()
            
            if cmd == "start":
                game_name = command[1] if len(command) > 1 else "unknown_game"
                notes = " ".join(command[2:]) if len(command) > 2 else ""
                collector.start_session(game_name=game_name, notes=notes)
                
            elif cmd == "collect":
                collector.start_collection()
                
            elif cmd == "stop":
                collector.stop_collection()
                
            elif cmd == "list":
                collector.list_sessions()
                
            elif cmd == "annotate":
                if len(command) > 1:
                    collector.create_annotation_template(command[1])
                else:
                    print("Usage: annotate <session_id>")
                    
            elif cmd in ["quit", "exit", "q"]:
                if collector.is_collecting:
                    collector.stop_collection()
                break
                
            else:
                print("Unknown command. Type 'quit' to exit.")
                
    except KeyboardInterrupt:
        print("\nStopping collection...")
        if collector.is_collecting:
            collector.stop_collection()
        print("Goodbye!")


if __name__ == "__main__":
    main()
