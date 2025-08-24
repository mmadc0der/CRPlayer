"""
CRPlayer Dashboard Module

Provides WebSocket server subscriber for pipeline integration.
"""

from .server import WebSocketDashboardSubscriber, create_dashboard_subscriber

__all__ = ['WebSocketDashboardSubscriber', 'create_dashboard_subscriber']