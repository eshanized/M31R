# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
M31R Training Dashboard - Real-time visualization.

This module provides a web-based dashboard for monitoring training in real-time.

Note: This module requires 'fastapi' and 'uvicorn' to be installed:
    pip install fastapi uvicorn
"""

# Check if FastAPI is available
try:
    import fastapi
    import uvicorn

    DASHBOARD_AVAILABLE = True
except ImportError:
    DASHBOARD_AVAILABLE = False

# Only export dashboard components if dependencies are available
if DASHBOARD_AVAILABLE:
    from m31r.dashboard.server import (
        app,
        broadcast_log,
        broadcast_metrics,
        metrics_data,
    )

    __all__ = [
        "app",
        "broadcast_log",
        "broadcast_metrics",
        "metrics_data",
        "DASHBOARD_AVAILABLE",
    ]
else:
    # Dummy exports when dashboard not available
    app = None
    metrics_data = None

    async def broadcast_metrics(metrics: dict) -> None:
        """No-op when dashboard not available."""
        pass

    async def broadcast_log(log_entry: dict) -> None:
        """No-op when dashboard not available."""
        pass

    __all__ = [
        "app",
        "broadcast_log",
        "broadcast_metrics",
        "metrics_data",
        "DASHBOARD_AVAILABLE",
    ]
