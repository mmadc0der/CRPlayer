"""
Socket-based Pipeline - Complete replacement for FIFO-based pipeline
Direct scrcpy socket connection with zero-copy NAL distribution
"""

import asyncio
import logging
import time
from typing import Optional, Dict, Any
from inference.scrcpy_socket import ScrcpySocketDemux, NALUnit
from inference.nal_distributor import (
    NALDistributor, 
    create_rl_consumer, 
    create_monitor_consumer, 
    create_replay_consumer
)

logger = logging.getLogger(__name__)

class SocketStreamPipeline:
    """Main pipeline using direct scrcpy socket connection"""
    
    def __init__(self, adb_port: int = 27183, device_id: Optional[str] = None):
        self.adb_port = adb_port
        self.device_id = device_id
        
        # Core components
        self.demux = ScrcpySocketDemux(adb_port, device_id)
        self.distributor = NALDistributor()
        
        # Pipeline state
        self._running = False
        self._stream_task: Optional[asyncio.Task] = None
        self._stats_task: Optional[asyncio.Task] = None
        
        # Statistics
        self.start_time = 0.0
        
    async def start(self) -> bool:
        """Start the socket pipeline"""
        if self._running:
            logger.warning("Pipeline already running")
            return True
        
        try:
            # Connect to scrcpy socket
            if not await self.demux.connect():
                logger.error("Failed to connect to scrcpy socket")
                return False
            
            # Start distributor
            self.distributor.start()
            
            # Start streaming task
            self._running = True
            self.start_time = time.time()
            self._stream_task = asyncio.create_task(self._stream_loop())
            self._stats_task = asyncio.create_task(self._stats_loop())
            
            logger.info("Socket pipeline started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start pipeline: {e}")
            await self.stop()
            return False
    
    async def stop(self):
        """Stop the socket pipeline"""
        if not self._running:
            return
        
        logger.info("Stopping socket pipeline...")
        self._running = False
        
        # Cancel tasks
        if self._stream_task:
            self._stream_task.cancel()
            try:
                await self._stream_task
            except asyncio.CancelledError:
                pass
        
        if self._stats_task:
            self._stats_task.cancel()
            try:
                await self._stats_task
            except asyncio.CancelledError:
                pass
        
        # Stop components
        self.distributor.stop()
        await self.demux.disconnect()
        
        logger.info("Socket pipeline stopped")
    
    async def _stream_loop(self):
        """Main streaming loop"""
        try:
            async for nal_unit in self.demux.stream_nal_units():
                if not self._running:
                    break
                
                # Distribute to all consumers
                await self.distributor.distribute(nal_unit)
                
        except asyncio.CancelledError:
            logger.info("Stream loop cancelled")
        except Exception as e:
            logger.error(f"Error in stream loop: {e}")
            self._running = False
    
    async def _stats_loop(self):
        """Periodic statistics reporting"""
        try:
            while self._running:
                await asyncio.sleep(5.0)  # Report every 5 seconds
                
                if self._running:
                    self._print_stats()
                    
        except asyncio.CancelledError:
            logger.info("Stats loop cancelled")
    
    def add_rl_consumer(self, name: str = "rl_consumer") -> str:
        """Add RL consumer"""
        config = create_rl_consumer(name)
        return self.distributor.add_consumer(config)
    
    def add_monitor_consumer(self, name: str = "monitor_consumer") -> str:
        """Add monitoring consumer"""
        config = create_monitor_consumer(name)
        return self.distributor.add_consumer(config)
    
    def add_replay_consumer(self, name: str = "replay_consumer") -> str:
        """Add replay consumer"""
        config = create_replay_consumer(name)
        return self.distributor.add_consumer(config)
    
    async def get_nal_unit(self, consumer_id: str, timeout: float = 0.1) -> Optional[NALUnit]:
        """Get NAL unit for specific consumer"""
        return await self.distributor.get_nal_unit(consumer_id, timeout)
    
    def get_nal_unit_nowait(self, consumer_id: str) -> Optional[NALUnit]:
        """Get NAL unit for specific consumer (non-blocking)"""
        return self.distributor.get_nal_unit_nowait(consumer_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive pipeline statistics"""
        demux_stats = self.demux.get_stats()
        distributor_stats = self.distributor.get_stats()
        
        elapsed = time.time() - self.start_time if self.start_time > 0 else 0
        
        return {
            'pipeline': {
                'running': self._running,
                'elapsed_time': elapsed,
                'start_time': self.start_time
            },
            'demux': demux_stats,
            'distributor': distributor_stats
        }
    
    def _print_stats(self):
        """Print formatted statistics"""
        stats = self.get_stats()
        
        print(f"\n=== Socket Pipeline Stats ===")
        print(f"Running: {stats['pipeline']['running']}")
        print(f"Elapsed: {stats['pipeline']['elapsed_time']:.1f}s")
        
        print(f"\nDemux:")
        demux = stats['demux']
        print(f"  Frames: {demux['frames_received']}")
        print(f"  FPS: {demux['fps']:.1f}")
        print(f"  Bitrate: {demux['bitrate_bps']/1000000:.1f} Mbps")
        print(f"  Keyframes: {demux['keyframes']}")
        
        print(f"\nDistributor:")
        dist = stats['distributor']
        print(f"  Total frames: {dist['total_frames']}")
        print(f"  Global FPS: {dist['global_fps']:.1f}")
        
        for consumer_id, consumer_stats in dist['consumers'].items():
            print(f"\n  {consumer_id}:")
            print(f"    Received: {consumer_stats['frames_received']}")
            print(f"    Dropped: {consumer_stats['frames_dropped']}")
            print(f"    FPS: {consumer_stats['fps']:.1f}")
            print(f"    Queue: {consumer_stats['queue_size']}/{consumer_stats['max_queue_size']}")


async def main():
    """Test the complete socket pipeline"""
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    device_id = sys.argv[1] if len(sys.argv) > 1 else None
    
    pipeline = SocketStreamPipeline(device_id=device_id)
    
    try:
        # Start pipeline
        if not await pipeline.start():
            print("Failed to start pipeline")
            return
        
        # Add consumers
        rl_id = pipeline.add_rl_consumer()
        monitor_id = pipeline.add_monitor_consumer()
        replay_id = pipeline.add_replay_consumer()
        
        print("Pipeline started with 3 consumers!")
        print("Press Ctrl+C to stop\n")
        
        # Simple consumer test
        async def test_consumer(consumer_id: str, name: str):
            frame_count = 0
            while pipeline._running:
                nal_unit = await pipeline.get_nal_unit(consumer_id, timeout=1.0)
                if nal_unit:
                    frame_count += 1
                    if frame_count % 30 == 0:  # Every 30 frames
                        print(f"{name}: received {frame_count} frames")
        
        # Run consumers
        await asyncio.gather(
            test_consumer(rl_id, "RL"),
            test_consumer(monitor_id, "Monitor"), 
            test_consumer(replay_id, "Replay"),
            return_exceptions=True
        )
        
    except KeyboardInterrupt:
        print("\nStopping pipeline...")
    finally:
        await pipeline.stop()
        
        # Final stats
        stats = pipeline.get_stats()
        print(f"\nFinal Stats:")
        print(f"Demux frames: {stats['demux']['frames_received']}")
        print(f"Distributor frames: {stats['distributor']['total_frames']}")


if __name__ == "__main__":
    asyncio.run(main())
