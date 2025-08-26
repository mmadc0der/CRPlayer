#!/usr/bin/env python3
"""
Pipeline Manager for GPU-Accelerated Android Streaming
Manages streaming pipeline with producer and multiple consumers using registry pattern.
"""

import threading
import time
from typing import Dict, List, Optional

from stream_pipeline import SharedStreamBuffer, StreamProducer
from consumer_registry import ConsumerRegistry, ConsumerBuilder
from android_stream_gpu import GPUAndroidStreamer


class PipelineManager:
    """Manages the complete streaming pipeline with consumer registry."""
    
    def __init__(self):
        # Create shared buffer
        self.buffer = SharedStreamBuffer(max_buffer_size=100)
        
        # Create producer
        self.producer = StreamProducer(self.buffer)
        
        # Consumer registry with move semantics
        self.registry = ConsumerRegistry(self.buffer)
        
        # Pipeline state
        self.running = False
        
    def add_consumer(self, consumer_type: str, consumer_id: str, **kwargs):
        """Add consumer using registry pattern."""
        return self.registry.create_consumer(consumer_type, consumer_id, **kwargs)
    
    def start_consumer(self, consumer_id: str):
        """Start specific consumer."""
        return self.registry.start_consumer(consumer_id)
    
    def stop_consumer(self, consumer_id: str):
        """Stop specific consumer."""
        return self.registry.stop_consumer(consumer_id)
    
    def remove_consumer(self, consumer_id: str):
        """Remove consumer from registry."""
        return self.registry.remove_consumer(consumer_id)
    
    def start_pipeline(self):
        """Start the complete pipeline."""
        if self.running:
            print("Pipeline already running")
            return
        
        print("Starting streaming pipeline...")
        
        # Start producer
        self.producer.start_production()
        
        # Start all registered consumers
        started = self.registry.start_all()
        print(f"Started {started} consumers")
        
        self.running = True
        print("Pipeline running. Use 'stop' command to stop.")
        
    def stop_pipeline(self):
        """Stop the complete pipeline."""
        if not self.running:
            print("Pipeline not running")
            return
        
        print("\nStopping pipeline...")
        
        # Stop producer
        self.producer.stop_production()
        
        # Stop all consumers
        stopped = self.registry.stop_all()
        print(f"Stopped {stopped} consumers")
        
        # Print final stats
        stats = self.buffer.get_stats()
        print(f"\nFinal Statistics:")
        print(f"Frames produced: {stats['frames_produced']}")
        print(f"Frames consumed: {stats['frames_consumed']}")
        print(f"Memory cleanups: {stats['memory_cleanups']}")
        print(f"Peak memory usage: {stats['memory_usage_mb']:.1f}MB")
        
        self.running = False
        
    def get_consumer_builder(self) -> ConsumerBuilder:
        """Get consumer builder for fluent interface."""
        return ConsumerBuilder(self.registry)
    
    def run_script(self, script_path: str) -> bool:
        """Run pipeline from script file."""
        from pipeline_script import PipelineScript
        
        script = PipelineScript(self)
        if script.load_from_file(script_path):
            return script.execute()
        return False
    
    def run_interactive(self):
        """Run interactive pipeline management."""
        print("=== Streaming Pipeline Manager ===")
        print("Commands:")
        print("  start - Start pipeline")
        print("  add <type> <id> [config] - Add consumer (types: data_collector, classifier, monitor)")
        print("  start_consumer <id> - Start specific consumer")
        print("  stop_consumer <id> - Stop specific consumer")
        print("  remove <id> - Remove consumer")
        print("  list - List all consumers")
        print("  script <file> - Run pipeline script")
        print("  stats - Show buffer statistics")
        print("  stop - Stop pipeline")
        print("  quit - Exit")
        print()
        
        try:
            while True:
                command = input("> ").strip().split()
                if not command:
                    continue
                
                cmd = command[0].lower()
                
                if cmd == "start":
                    self.start_pipeline()
                
                elif cmd == "add":
                    if len(command) < 3:
                        print("Usage: add <type> <id> [config_key=value ...]")
                        print("Types: data_collector, classifier, monitor")
                        continue
                    
                    consumer_type = command[1]
                    consumer_id = command[2]
                    
                    # Parse config arguments
                    config = {}
                    for arg in command[3:]:
                        if '=' in arg:
                            key, value = arg.split('=', 1)
                            config[key] = value
                    
                    consumer = self.add_consumer(consumer_type, consumer_id, **config)
                    if consumer:
                        print(f"Added {consumer_type} '{consumer_id}'")
                        if config:
                            print(f"  Config: {config}")
                
                elif cmd == "start_consumer":
                    if len(command) < 2:
                        print("Usage: start_consumer <id>")
                        continue
                    self.start_consumer(command[1])
                
                elif cmd == "stop_consumer":
                    if len(command) < 2:
                        print("Usage: stop_consumer <id>")
                        continue
                    self.stop_consumer(command[1])
                
                elif cmd == "remove":
                    if len(command) < 2:
                        print("Usage: remove <id>")
                        continue
                    self.remove_consumer(command[1])
                
                elif cmd == "list":
                    consumers = self.registry.list_consumers()
                    if consumers:
                        print("\nRegistered Consumers:")
                        for consumer in consumers:
                            status = "RUNNING" if consumer['running'] else "STOPPED"
                            print(f"  {consumer['id']} ({consumer['type']}) - {status} - {consumer['frame_count']} frames")
                    else:
                        print("No consumers registered")
                
                elif cmd == "script":
                    if len(command) < 2:
                        print("Usage: script <file>")
                        continue
                    
                    script_file = command[1]
                    print(f"Running script: {script_file}")
                    success = self.run_script(script_file)
                    if success:
                        print("Script completed successfully")
                    else:
                        print("Script execution failed")
                
                elif cmd == "stats":
                    stats = self.buffer.get_stats()
                    print(f"\nBuffer Statistics:")
                    print(f"  Frames produced: {stats['frames_produced']}")
                    print(f"  Frames consumed: {stats['frames_consumed']}")
                    print(f"  Buffer size: {stats['buffer_size']}")
                    print(f"  Memory usage: {stats['memory_usage_mb']:.1f}MB")
                    print(f"  Active consumers: {stats['active_consumers']}")
                    print(f"  Consumer positions: {stats['consumer_positions']}")
                
                elif cmd == "stop":
                    self.stop_pipeline()
                
                elif cmd in ["quit", "exit", "q"]:
                    if self.running:
                        self.stop_pipeline()
                    break
                
                else:
                    print("Unknown command")
        
        except KeyboardInterrupt:
            if self.running:
                self.stop_pipeline()
            print("\nGoodbye!")


def demo_basic_pipeline():
    """Demonstrate basic pipeline setup."""
    print("=== Basic Pipeline Demo ===")
    
    # Create pipeline
    manager = PipelineManager()
    
    # Add consumers using registry
    manager.add_consumer("monitor", "demo_monitor")
    manager.add_consumer("data_collector", "demo_collector", output_dir="demo_data")
    
    # Start pipeline
    manager.start_pipeline()
    
    try:
        # Run for 30 seconds
        time.sleep(30)
    except KeyboardInterrupt:
        pass
    finally:
        manager.stop_pipeline()


def demo_full_pipeline():
    """Demonstrate full pipeline with all consumers."""
    print("=== Full Pipeline Demo ===")
    
    # Create pipeline
    manager = PipelineManager()
    
    # Add all consumer types using registry
    manager.add_consumer("monitor", "full_monitor")
    manager.add_consumer("data_collector", "full_collector", output_dir="full_demo_data")
    # manager.add_consumer("classifier", "full_classifier", model_path="best_game_state_model.pth")  # Uncomment if model exists
    
    # Start pipeline
    manager.start_pipeline()
    
    try:
        # Run until interrupted
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        manager.stop_pipeline()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        
        if mode == "basic":
            demo_basic_pipeline()
        elif mode == "full":
            demo_full_pipeline()
        elif mode == "interactive":
            manager = PipelineManager()
            manager.run_interactive()
        elif mode == "script":
            if len(sys.argv) < 3:
                print("Usage: python pipeline_demo.py script <script_file>")
            else:
                script_file = sys.argv[2]
                manager = PipelineManager()
                manager.run_script(script_file)
        else:
            print("Usage: python pipeline_demo.py [basic|full|interactive|script <file>]")
    else:
        print("Available modes:")
        print("  basic - Run basic demo")
        print("  full - Run full demo")
        print("  interactive - Interactive mode")
        print("  script <file> - Run from script file")
        print("\nDefaulting to interactive mode...")
        manager = PipelineManager()
        manager.run_interactive()
