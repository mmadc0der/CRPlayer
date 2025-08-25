"""
NAL Unit Distributor - Zero-copy distribution system for multiple consumers
Replaces the pipeline.py subscriber pattern with async queues and backpressure handling
"""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
from .scrcpy_socket import NALUnit

logger = logging.getLogger(__name__)

class ConsumerType(Enum):
    RL = "rl"
    MONITOR = "monitor" 
    REPLAY = "replay"

@dataclass
class ConsumerConfig:
    """Configuration for a consumer queue"""
    name: str
    consumer_type: ConsumerType
    max_queue_size: int = 30
    drop_non_keyframes: bool = True
    priority: int = 1  # Higher = more important
    callback: Optional[Callable[[NALUnit], None]] = None

@dataclass
class ConsumerStats:
    """Statistics for a consumer"""
    frames_received: int = 0
    frames_dropped: int = 0
    queue_full_events: int = 0
    last_frame_time: float = 0.0
    start_time: float = field(default_factory=time.time)
    
    @property
    def fps(self) -> float:
        elapsed = time.time() - self.start_time
        return self.frames_received / elapsed if elapsed > 0 else 0.0
    
    @property
    def drop_rate(self) -> float:
        total = self.frames_received + self.frames_dropped
        return self.frames_dropped / total if total > 0 else 0.0

class NALDistributor:
    """Zero-copy NAL unit distributor with backpressure handling"""
    
    def __init__(self):
        self.consumers: Dict[str, ConsumerConfig] = {}
        self.queues: Dict[str, asyncio.Queue] = {}
        self.stats: Dict[str, ConsumerStats] = {}
        self._running = False
        self._distribution_task: Optional[asyncio.Task] = None
        
        # Global stats
        self.total_frames = 0
        self.start_time = 0.0
        
    def add_consumer(self, config: ConsumerConfig) -> str:
        """Add a consumer with specified configuration"""
        consumer_id = config.name
        
        if consumer_id in self.consumers:
            raise ValueError(f"Consumer {consumer_id} already exists")
        
        self.consumers[consumer_id] = config
        self.queues[consumer_id] = asyncio.Queue(maxsize=config.max_queue_size)
        self.stats[consumer_id] = ConsumerStats()
        
        logger.info(f"Added consumer: {consumer_id} ({config.consumer_type.value})")
        return consumer_id
    
    def remove_consumer(self, consumer_id: str):
        """Remove a consumer"""
        if consumer_id in self.consumers:
            del self.consumers[consumer_id]
            del self.queues[consumer_id] 
            del self.stats[consumer_id]
            logger.info(f"Removed consumer: {consumer_id}")
    
    async def distribute(self, nal_unit: NALUnit):
        """Distribute NAL unit to all consumers with backpressure handling"""
        if not self._running:
            return
            
        self.total_frames += 1
        
        # Sort consumers by priority (higher first)
        sorted_consumers = sorted(
            self.consumers.items(),
            key=lambda x: x[1].priority,
            reverse=True
        )
        
        for consumer_id, config in sorted_consumers:
            await self._send_to_consumer(consumer_id, config, nal_unit)
    
    async def _send_to_consumer(self, consumer_id: str, config: ConsumerConfig, nal_unit: NALUnit):
        """Send NAL unit to specific consumer with backpressure handling"""
        queue = self.queues[consumer_id]
        stats = self.stats[consumer_id]
        
        try:
            # Try non-blocking put first
            queue.put_nowait(nal_unit)
            stats.frames_received += 1
            stats.last_frame_time = time.time()
            
            # Call callback if provided
            if config.callback:
                try:
                    config.callback(nal_unit)
                except Exception as e:
                    logger.error(f"Consumer {consumer_id} callback error: {e}")
                    
        except asyncio.QueueFull:
            stats.queue_full_events += 1
            
            # Handle backpressure based on consumer type
            if config.consumer_type == ConsumerType.REPLAY:
                # Replay consumer should never drop frames - block if necessary
                await queue.put(nal_unit)
                stats.frames_received += 1
                stats.last_frame_time = time.time()
                
            elif config.drop_non_keyframes and not nal_unit.is_keyframe:
                # Drop non-keyframe for monitor/RL consumers
                stats.frames_dropped += 1
                logger.debug(f"Dropped non-keyframe for {consumer_id}")
                
            else:
                # Drop oldest frame and add new one
                try:
                    dropped = queue.get_nowait()
                    stats.frames_dropped += 1
                    queue.put_nowait(nal_unit)
                    stats.frames_received += 1
                    stats.last_frame_time = time.time()
                    logger.debug(f"Dropped frame for {consumer_id}")
                except asyncio.QueueEmpty:
                    # Queue became empty while we were checking
                    queue.put_nowait(nal_unit)
                    stats.frames_received += 1
                    stats.last_frame_time = time.time()
    
    async def get_nal_unit(self, consumer_id: str, timeout: float = 0.1) -> Optional[NALUnit]:
        """Get next NAL unit for specific consumer"""
        if consumer_id not in self.queues:
            raise ValueError(f"Unknown consumer: {consumer_id}")
        
        try:
            return await asyncio.wait_for(
                self.queues[consumer_id].get(),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            return None
    
    def get_nal_unit_nowait(self, consumer_id: str) -> Optional[NALUnit]:
        """Get next NAL unit for specific consumer (non-blocking)"""
        if consumer_id not in self.queues:
            raise ValueError(f"Unknown consumer: {consumer_id}")
        
        try:
            return self.queues[consumer_id].get_nowait()
        except asyncio.QueueEmpty:
            return None
    
    def start(self):
        """Start the distributor"""
        self._running = True
        self.start_time = time.time()
        logger.info("NAL distributor started")
    
    def stop(self):
        """Stop the distributor"""
        self._running = False
        logger.info("NAL distributor stopped")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get distributor and consumer statistics"""
        elapsed = time.time() - self.start_time if self.start_time > 0 else 0
        global_fps = self.total_frames / elapsed if elapsed > 0 else 0
        
        consumer_stats = {}
        for consumer_id, stats in self.stats.items():
            consumer_stats[consumer_id] = {
                'frames_received': stats.frames_received,
                'frames_dropped': stats.frames_dropped,
                'queue_full_events': stats.queue_full_events,
                'fps': stats.fps,
                'drop_rate': stats.drop_rate,
                'queue_size': self.queues[consumer_id].qsize(),
                'max_queue_size': self.consumers[consumer_id].max_queue_size
            }
        
        return {
            'total_frames': self.total_frames,
            'elapsed_time': elapsed,
            'global_fps': global_fps,
            'consumers': consumer_stats,
            'running': self._running
        }
    
    def print_stats(self):
        """Print formatted statistics"""
        stats = self.get_stats()
        
        print(f"\n=== NAL Distributor Stats ===")
        print(f"Total frames: {stats['total_frames']}")
        print(f"Global FPS: {stats['global_fps']:.1f}")
        print(f"Elapsed: {stats['elapsed_time']:.1f}s")
        print(f"Running: {stats['running']}")
        
        for consumer_id, consumer_stats in stats['consumers'].items():
            print(f"\n{consumer_id}:")
            print(f"  Received: {consumer_stats['frames_received']}")
            print(f"  Dropped: {consumer_stats['frames_dropped']}")
            print(f"  FPS: {consumer_stats['fps']:.1f}")
            print(f"  Drop rate: {consumer_stats['drop_rate']:.1%}")
            print(f"  Queue: {consumer_stats['queue_size']}/{consumer_stats['max_queue_size']}")


# Consumer factory functions
def create_rl_consumer(name: str = "rl_consumer") -> ConsumerConfig:
    """Create RL consumer configuration"""
    return ConsumerConfig(
        name=name,
        consumer_type=ConsumerType.RL,
        max_queue_size=30,  # Allow some buffering for training
        drop_non_keyframes=False,  # RL needs all frames
        priority=3  # Highest priority
    )

def create_monitor_consumer(name: str = "monitor_consumer") -> ConsumerConfig:
    """Create monitoring consumer configuration"""
    return ConsumerConfig(
        name=name,
        consumer_type=ConsumerType.MONITOR,
        max_queue_size=10,  # Small buffer for real-time display
        drop_non_keyframes=True,  # Can drop frames for display
        priority=2  # Medium priority
    )

def create_replay_consumer(name: str = "replay_consumer") -> ConsumerConfig:
    """Create replay consumer configuration"""
    return ConsumerConfig(
        name=name,
        consumer_type=ConsumerType.REPLAY,
        max_queue_size=100,  # Large buffer for recording
        drop_non_keyframes=False,  # Never drop frames for replay
        priority=1  # Lower priority but never drops
    )


async def main():
    """Test the NAL distributor"""
    logging.basicConfig(level=logging.INFO)
    
    distributor = NALDistributor()
    
    # Add test consumers
    rl_id = distributor.add_consumer(create_rl_consumer())
    monitor_id = distributor.add_consumer(create_monitor_consumer())
    replay_id = distributor.add_consumer(create_replay_consumer())
    
    distributor.start()
    
    # Simulate NAL units
    async def simulate_frames():
        for i in range(100):
            nal_unit = NALUnit(
                data=b'\x00\x00\x00\x01' + bytes([i % 256]) * 1000,
                pts=i * 33333333,  # ~30 FPS
                size=1004,
                timestamp=time.time(),
                is_keyframe=(i % 30 == 0)  # Keyframe every 30 frames
            )
            await distributor.distribute(nal_unit)
            await asyncio.sleep(0.033)  # 30 FPS
    
    # Simulate consumers
    async def consume_frames(consumer_id: str, name: str):
        count = 0
        while count < 50:
            nal_unit = await distributor.get_nal_unit(consumer_id, timeout=1.0)
            if nal_unit:
                count += 1
                if count % 10 == 0:
                    print(f"{name}: received frame {count}")
    
    try:
        # Run simulation
        await asyncio.gather(
            simulate_frames(),
            consume_frames(rl_id, "RL"),
            consume_frames(monitor_id, "Monitor"),
            consume_frames(replay_id, "Replay")
        )
    finally:
        distributor.stop()
        distributor.print_stats()


if __name__ == "__main__":
    asyncio.run(main())
