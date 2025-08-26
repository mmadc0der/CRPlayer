#!/usr/bin/env python3
"""
Pipeline Script Parser and Executor
Allows defining pipeline configurations in script files.
"""

import json
try:
    import yaml
except ImportError:
    print("Warning: PyYAML not installed. YAML support disabled.")
    yaml = None
from typing import Dict, List, Any, Optional
from pathlib import Path
import time
import threading

from pipeline_demo import PipelineManager


class PipelineScript:
    """Parser and executor for pipeline script files."""
    
    def __init__(self, manager: PipelineManager):
        self.manager = manager
        self.config = {}
        self.auto_start = True
        self.duration = None
        self.wait_for_completion = False
    
    def load_from_file(self, script_path: str) -> bool:
        """Load pipeline configuration from file."""
        script_path = Path(script_path)
        
        if not script_path.exists():
            print(f"[SCRIPT] Error: Script file not found: {script_path}")
            return False
        
        try:
            with open(script_path, 'r') as f:
                if script_path.suffix.lower() in ['.yaml', '.yml']:
                    self.config = yaml.safe_load(f)
                elif script_path.suffix.lower() == '.json':
                    self.config = json.load(f)
                else:
                    # Try to parse as simple command format
                    self.config = self._parse_simple_format(f.read())
            
            print(f"[SCRIPT] Loaded configuration from: {script_path}")
            return True
            
        except Exception as e:
            print(f"[SCRIPT] Error loading script: {e}")
            return False
    
    def _parse_simple_format(self, content: str) -> Dict[str, Any]:
        """Parse simple command-based script format."""
        config = {
            "consumers": [],
            "settings": {}
        }
        
        for line in content.strip().split('\n'):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split()
            if not parts:
                continue
            
            cmd = parts[0].lower()
            
            if cmd == "add" and len(parts) >= 3:
                consumer_type = parts[1]
                consumer_id = parts[2]
                
                # Parse config arguments
                consumer_config = {"type": consumer_type, "id": consumer_id}
                for arg in parts[3:]:
                    if '=' in arg:
                        key, value = arg.split('=', 1)
                        # Try to convert to appropriate type
                        try:
                            if value.lower() in ['true', 'false']:
                                value = value.lower() == 'true'
                            elif value.isdigit():
                                value = int(value)
                            elif '.' in value and value.replace('.', '').isdigit():
                                value = float(value)
                        except:
                            pass  # Keep as string
                        consumer_config[key] = value
                
                config["consumers"].append(consumer_config)
            
            elif cmd == "duration" and len(parts) >= 2:
                config["settings"]["duration"] = int(parts[1])
            
            elif cmd == "auto_start" and len(parts) >= 2:
                config["settings"]["auto_start"] = parts[1].lower() == 'true'
            
            elif cmd == "wait_completion":
                config["settings"]["wait_for_completion"] = True
        
        return config
    
    def execute(self) -> bool:
        """Execute the loaded pipeline configuration."""
        if not self.config:
            print("[SCRIPT] No configuration loaded")
            return False
        
        try:
            # Parse settings
            settings = self.config.get("settings", {})
            self.auto_start = settings.get("auto_start", True)
            self.duration = settings.get("duration", None)
            self.wait_for_completion = settings.get("wait_for_completion", False)
            
            # Add consumers
            consumers_added = 0
            for consumer_config in self.config.get("consumers", []):
                consumer_type = consumer_config.pop("type")
                consumer_id = consumer_config.pop("id")
                
                consumer = self.manager.add_consumer(consumer_type, consumer_id, **consumer_config)
                if consumer:
                    consumers_added += 1
                    print(f"[SCRIPT] Added {consumer_type} '{consumer_id}'")
            
            print(f"[SCRIPT] Added {consumers_added} consumers")
            
            # Start pipeline if auto_start is enabled
            if self.auto_start:
                print("[SCRIPT] Starting pipeline...")
                self.manager.start_pipeline()
                
                # Handle duration or wait for completion
                if self.duration:
                    print(f"[SCRIPT] Running for {self.duration} seconds...")
                    time.sleep(self.duration)
                    print("[SCRIPT] Duration completed, stopping pipeline...")
                    self.manager.stop_pipeline()
                
                elif self.wait_for_completion:
                    print("[SCRIPT] Waiting for completion (Ctrl+C to stop)...")
                    try:
                        self._wait_for_completion()
                    except KeyboardInterrupt:
                        print("\n[SCRIPT] Interrupted, stopping pipeline...")
                        self.manager.stop_pipeline()
                
                else:
                    print("[SCRIPT] Pipeline started. Press Ctrl+C to stop.")
                    # Keep main thread alive
                    try:
                        while True:
                            time.sleep(1)
                    except KeyboardInterrupt:
                        print("\n[SCRIPT] Interrupted, stopping pipeline...")
                        self.manager.stop_pipeline()
            
            return True
            
        except Exception as e:
            print(f"[SCRIPT] Execution error: {e}")
            return False
    
    def _wait_for_completion(self):
        """Wait for all consumers to complete their work."""
        while True:
            time.sleep(1)
            
            # Check if any consumers have max_frames and are done
            all_done = True
            consumers = self.manager.registry.list_consumers()
            
            for consumer_info in consumers:
                if consumer_info['running']:
                    consumer = self.manager.registry.get_consumer(consumer_info['id'])
                    if consumer and hasattr(consumer, 'max_frames') and consumer.max_frames:
                        if hasattr(consumer, 'saved_count') and consumer.saved_count < consumer.max_frames:
                            all_done = False
                            break
                    else:
                        all_done = False
                        break
            
            if all_done:
                print("[SCRIPT] All consumers completed their work")
                self.manager.stop_pipeline()
                break




if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python pipeline_script.py <script_file>     - Run pipeline script")
        print("\nAvailable scripts:")
        print("  example.yaml - Full example with all consumer types")
        print("  prod.yaml    - Production config (data collector + monitor)")
        sys.exit(1)
    
    script_file = sys.argv[1]
    
    # Create pipeline manager
    manager = PipelineManager()
    
    # Load and execute script
    script = PipelineScript(manager)
    if script.load_from_file(script_file):
        try:
            script.execute()
        except KeyboardInterrupt:
            print("\n[SCRIPT] Interrupted")
            if manager.running:
                manager.stop_pipeline()
    else:
        print(f"Failed to load script: {script_file}")
