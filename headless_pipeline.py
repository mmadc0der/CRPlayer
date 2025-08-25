#!/usr/bin/env python3
"""
Complete Headless Scrcpy Pipeline - Integrated solution for zero-copy video streaming
"""

import asyncio
import logging
import time
from typing import Optional
from scrcpy_headless import ScrcpyHeadlessManager
from inference.scrcpy_socket import ScrcpySocketDemux
from inference.nal_distributor import (
    NALDistributor, 
    create_rl_consumer, 
    create_monitor_consumer, 
    create_replay_consumer
)

logger = logging.getLogger(__name__)

class HeadlessScrcpyPipeline:
    """Complete pipeline with headless scrcpy and zero-copy distribution"""
    
    def __init__(self, device_id: Optional[str] = None):
        self.device_id = device_id
        
        # Core components
        self.scrcpy_manager = ScrcpyHeadlessManager(device_id)
        self.socket_demux = ScrcpySocketDemux()
        self.distributor = NALDistributor()
        
        # Pipeline state
        self._running = False
        self._stream_task: Optional[asyncio.Task] = None
        self._stats_task: Optional[asyncio.Task] = None
        
        # Override SCID detection to use our fixed SCID
        self.socket_demux._get_scrcpy_scid = self._get_fixed_scid
        
    async def _get_fixed_scid(self) -> str:
        """Return the fixed SCID from headless manager"""
        return self.scrcpy_manager.scid
    
    async def start(self) -> bool:
        """Start the complete headless pipeline"""
        if self._running:
            logger.warning("Pipeline already running")
            return True
        
        try:
            # Start headless scrcpy
            logger.info("Starting headless scrcpy...")
            if not await self.scrcpy_manager.start():
                logger.error("Failed to start headless scrcpy")
                return False
            
            # Wait for scrcpy to be ready
            await asyncio.sleep(2)
            
            # Connect socket demux
            logger.info("Connecting to scrcpy socket...")
            if not await self.socket_demux.connect():
                logger.error("Failed to connect to scrcpy socket")
                await self.scrcpy_manager.stop()
                return False
            
            # Start distributor
            self.distributor.start()
            
            # Start pipeline tasks
            self._running = True
            self._stream_task = asyncio.create_task(self._stream_loop())
            self._stats_task = asyncio.create_task(self._stats_loop())
            
            logger.info("Headless scrcpy pipeline started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start pipeline: {e}")
            await self.stop()
            return False
    
    async def stop(self):
        """Stop the complete pipeline"""
        if not self._running:
            return
        
        logger.info("Stopping headless scrcpy pipeline...")
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
        await self.socket_demux.disconnect()
        await self.scrcpy_manager.stop()
        
        logger.info("Headless scrcpy pipeline stopped")
    
    async def _stream_loop(self):
        """Main streaming loop"""
        try:
            logger.info("Starting video stream processing...")
            
            async for nal_unit in self.socket_demux.stream_nal_units():
                if not self._running:
                    break
                
                # Distribute NAL unit to consumers
                await self.distributor.distribute_nal_unit(nal_unit)
                
        except asyncio.CancelledError:
            logger.info("Stream loop cancelled")
        except Exception as e:
            logger.error(f"Error in stream loop: {e}")
            raise
    
    async def _stats_loop(self):
        """Statistics reporting loop"""
        try:
            while self._running:
                await asyncio.sleep(5)  # Report every 5 seconds
                
                # Get stats from components
                socket_stats = self.socket_demux.get_stats()
                distributor_stats = self.distributor.get_stats()
                
                logger.info(f"Pipeline Stats - "
                           f"Frames: {socket_stats.get('frames_received', 0)}, "
                           f"FPS: {socket_stats.get('fps', 0):.1f}, "
                           f"Bitrate: {socket_stats.get('bitrate_kbps', 0):.0f} kbps, "
                           f"Consumers: {len(distributor_stats.get('consumers', {}))}")
                
        except asyncio.CancelledError:
            logger.info("Stats loop cancelled")
        except Exception as e:
            logger.error(f"Error in stats loop: {e}")
    
    # Consumer management methods
    async def add_rl_consumer(self, name: str = "rl_training"):
        """Add RL training consumer"""
        consumer = create_rl_consumer(name)
        await self.distributor.add_consumer(consumer)
        logger.info(f"Added RL consumer: {name}")
        return consumer
    
    async def add_monitor_consumer(self, name: str = "monitor"):
        """Add monitoring consumer"""
        consumer = create_monitor_consumer(name)
        await self.distributor.add_consumer(consumer)
        logger.info(f"Added monitor consumer: {name}")
        return consumer
    
    async def add_replay_consumer(self, name: str = "replay"):
        """Add replay consumer"""
        consumer = create_replay_consumer(name)
        await self.distributor.add_consumer(consumer)
        logger.info(f"Added replay consumer: {name}")
        return consumer
    
    def get_stats(self):
        """Get comprehensive pipeline statistics"""
        return {
            'scrcpy_running': self.scrcpy_manager.is_running(),
            'socket_connected': self.socket_demux._connected,
            'distributor_running': self.distributor._running,
            'socket_stats': self.socket_demux.get_stats(),
            'distributor_stats': self.distributor.get_stats()
        }

# Test the complete pipeline
async def test_headless_pipeline():
    """Test the complete headless pipeline"""
    pipeline = HeadlessScrcpyPipeline()
    
    try:
        # Start pipeline
        if await pipeline.start():
            logger.info("Pipeline started successfully!")
            
            # Add consumers
            await pipeline.add_monitor_consumer("test_monitor")
            await pipeline.add_rl_consumer("test_rl")
            
            # Run for 30 seconds
            logger.info("Running pipeline for 30 seconds...")
            await asyncio.sleep(30)
            
            # Print final stats
            stats = pipeline.get_stats()
            logger.info(f"Final stats: {stats}")
            
        else:
            logger.error("Failed to start pipeline")
    
    finally:
        await pipeline.stop()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    asyncio.run(test_headless_pipeline())
