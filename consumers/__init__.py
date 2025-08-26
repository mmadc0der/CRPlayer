"""
Consumers Module - Consumer Registry and Distribution

Contains the consumer registry system and various consumer implementations
for distributing processed frames to different components (collectors, classifiers, agents).
"""

from .consumer_registry import ConsumerRegistry
from .data_collector_consumer import DataCollectorConsumer
from .classifier_consumer import ClassifierConsumer

__all__ = ['ConsumerRegistry', 'DataCollectorConsumer', 'ClassifierConsumer']
