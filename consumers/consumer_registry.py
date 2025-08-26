#!/usr/bin/env python3
"""
Consumer Registry System
Manages registration and lifecycle of stream consumers with move semantics.
"""

from typing import Dict, List, Type, Any, Optional, Callable
from abc import ABC, abstractmethod
import threading
import time

from core.stream_pipeline import StreamConsumer, SharedStreamBuffer


class ConsumerFactory(ABC):
    """Abstract factory for creating consumers."""
    
    @abstractmethod
    def create_consumer(self, consumer_id: str, stream_buffer: SharedStreamBuffer, 
                       **kwargs) -> StreamConsumer:
        """Create a consumer instance."""
        pass
    
    @abstractmethod
    def get_consumer_type(self) -> str:
        """Get consumer type name."""
        pass


class DataCollectorFactory(ConsumerFactory):
    """Factory for data collector consumers."""
    
    def create_consumer(self, consumer_id: str, stream_buffer: SharedStreamBuffer, 
                       **kwargs) -> StreamConsumer:
        from .data_collector_consumer import DataCollectorConsumer
        return DataCollectorConsumer(consumer_id, stream_buffer, **kwargs)
    
    def get_consumer_type(self) -> str:
        return "data_collector"


class ClassifierFactory(ConsumerFactory):
    """Factory for classifier consumers."""
    
    def create_consumer(self, consumer_id: str, stream_buffer: SharedStreamBuffer, 
                       **kwargs) -> StreamConsumer:
        from .classifier_consumer import ClassifierConsumer
        return ClassifierConsumer(consumer_id, stream_buffer, **kwargs)
    
    def get_consumer_type(self) -> str:
        return "classifier"


class MonitoringFactory(ConsumerFactory):
    """Factory for monitoring consumers."""
    
    def create_consumer(self, consumer_id: str, stream_buffer: SharedStreamBuffer, 
                       **kwargs) -> StreamConsumer:
        from .data_collector_consumer import MonitoringConsumer
        return MonitoringConsumer(consumer_id, stream_buffer, **kwargs)
    
    def get_consumer_type(self) -> str:
        return "monitor"


class ConsumerRegistry:
    """Registry for managing stream consumers with move semantics."""
    
    def __init__(self, stream_buffer: SharedStreamBuffer):
        self.stream_buffer = stream_buffer
        self.consumers: Dict[str, StreamConsumer] = {}
        self.factories: Dict[str, ConsumerFactory] = {}
        self.lock = threading.RLock()
        
        # Register built-in factories
        self._register_builtin_factories()
    
    def _register_builtin_factories(self):
        """Register built-in consumer factories."""
        self.register_factory(DataCollectorFactory())
        self.register_factory(ClassifierFactory())
        self.register_factory(MonitoringFactory())
    
    def register_factory(self, factory: ConsumerFactory):
        """Register a consumer factory."""
        with self.lock:
            self.factories[factory.get_consumer_type()] = factory
            print(f"[REGISTRY] Registered factory: {factory.get_consumer_type()}")
    
    def create_consumer(self, consumer_type: str, consumer_id: str, 
                       **kwargs) -> Optional[StreamConsumer]:
        """Create and register a consumer using factory pattern."""
        with self.lock:
            if consumer_id in self.consumers:
                print(f"[REGISTRY] Consumer {consumer_id} already exists")
                return None
            
            if consumer_type not in self.factories:
                print(f"[REGISTRY] Unknown consumer type: {consumer_type}")
                print(f"[REGISTRY] Available types: {list(self.factories.keys())}")
                return None
            
            try:
                factory = self.factories[consumer_type]
                consumer = factory.create_consumer(consumer_id, self.stream_buffer, **kwargs)
                
                # Register consumer
                self.consumers[consumer_id] = consumer
                
                print(f"[REGISTRY] Created consumer: {consumer_id} ({consumer_type})")
                return consumer
                
            except Exception as e:
                print(f"[REGISTRY] Failed to create consumer {consumer_id}: {e}")
                return None
    
    def start_consumer(self, consumer_id: str) -> bool:
        """Start a registered consumer."""
        with self.lock:
            if consumer_id not in self.consumers:
                print(f"[REGISTRY] Consumer {consumer_id} not found")
                return False
            
            try:
                consumer = self.consumers[consumer_id]
                consumer.start()
                print(f"[REGISTRY] Started consumer: {consumer_id}")
                return True
                
            except Exception as e:
                print(f"[REGISTRY] Failed to start consumer {consumer_id}: {e}")
                return False
    
    def stop_consumer(self, consumer_id: str) -> bool:
        """Stop a registered consumer."""
        with self.lock:
            if consumer_id not in self.consumers:
                print(f"[REGISTRY] Consumer {consumer_id} not found")
                return False
            
            try:
                consumer = self.consumers[consumer_id]
                consumer.stop()
                print(f"[REGISTRY] Stopped consumer: {consumer_id}")
                return True
                
            except Exception as e:
                print(f"[REGISTRY] Failed to stop consumer {consumer_id}: {e}")
                return False
    
    def remove_consumer(self, consumer_id: str) -> bool:
        """Remove a consumer from registry (with move semantics)."""
        with self.lock:
            if consumer_id not in self.consumers:
                print(f"[REGISTRY] Consumer {consumer_id} not found")
                return False
            
            try:
                consumer = self.consumers[consumer_id]
                
                # Stop consumer if running
                if consumer.is_running():
                    consumer.stop()
                
                # Remove from registry (move semantics - consumer is no longer managed)
                removed_consumer = self.consumers.pop(consumer_id)
                
                print(f"[REGISTRY] Removed consumer: {consumer_id}")
                return True
                
            except Exception as e:
                print(f"[REGISTRY] Failed to remove consumer {consumer_id}: {e}")
                return False
    
    def get_consumer(self, consumer_id: str) -> Optional[StreamConsumer]:
        """Get a consumer by ID."""
        with self.lock:
            return self.consumers.get(consumer_id)
    
    def list_consumers(self) -> List[Dict[str, Any]]:
        """List all registered consumers."""
        with self.lock:
            consumer_list = []
            for consumer_id, consumer in self.consumers.items():
                consumer_info = {
                    'id': consumer_id,
                    'type': type(consumer).__name__,
                    'running': consumer.is_running(),
                    'frame_count': getattr(consumer, 'frame_count', 0)
                }
                consumer_list.append(consumer_info)
            return consumer_list
    
    def start_all(self) -> int:
        """Start all registered consumers."""
        with self.lock:
            started = 0
            for consumer_id in list(self.consumers.keys()):
                if self.start_consumer(consumer_id):
                    started += 1
            return started
    
    def stop_all(self) -> int:
        """Stop all registered consumers."""
        with self.lock:
            stopped = 0
            for consumer_id in list(self.consumers.keys()):
                if self.stop_consumer(consumer_id):
                    stopped += 1
            return stopped
    
    def cleanup_inactive(self) -> int:
        """Remove inactive consumers."""
        with self.lock:
            inactive_consumers = []
            
            for consumer_id, consumer in self.consumers.items():
                if not consumer.is_running() and hasattr(consumer, '_thread'):
                    if not consumer._thread.is_alive():
                        inactive_consumers.append(consumer_id)
            
            for consumer_id in inactive_consumers:
                self.remove_consumer(consumer_id)
            
            if inactive_consumers:
                print(f"[REGISTRY] Cleaned up {len(inactive_consumers)} inactive consumers")
            
            return len(inactive_consumers)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        with self.lock:
            stats = {
                'total_consumers': len(self.consumers),
                'running_consumers': sum(1 for c in self.consumers.values() if c.is_running()),
                'available_types': list(self.factories.keys()),
                'consumers': self.list_consumers()
            }
            return stats


class ConsumerBuilder:
    """Builder pattern for creating consumers with fluent interface."""
    
    def __init__(self, registry: ConsumerRegistry):
        self.registry = registry
        self.consumer_type = None
        self.consumer_id = None
        self.kwargs = {}
    
    def of_type(self, consumer_type: str) -> 'ConsumerBuilder':
        """Set consumer type."""
        self.consumer_type = consumer_type
        return self
    
    def with_id(self, consumer_id: str) -> 'ConsumerBuilder':
        """Set consumer ID."""
        self.consumer_id = consumer_id
        return self
    
    def with_config(self, **kwargs) -> 'ConsumerBuilder':
        """Set consumer configuration."""
        self.kwargs.update(kwargs)
        return self
    
    def build(self) -> Optional[StreamConsumer]:
        """Build and register the consumer."""
        if not self.consumer_type or not self.consumer_id:
            raise ValueError("Consumer type and ID must be specified")
        
        return self.registry.create_consumer(
            self.consumer_type, self.consumer_id, **self.kwargs
        )
    
    def build_and_start(self) -> Optional[StreamConsumer]:
        """Build, register, and start the consumer."""
        consumer = self.build()
        if consumer:
            self.registry.start_consumer(self.consumer_id)
        return consumer
