# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
M31R Training Dashboard - Real-time visualization server.

Provides a web-based dashboard for monitoring training in real-time.
Uses FastAPI for the backend and WebSocket for live updates.

Features:
- Real-time metrics streaming via WebSocket
- Live log tailing
- Interactive charts (loss, learning rate, throughput)
- Training progress tracking
- Multi-run comparison
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse
    from fastapi.staticfiles import StaticFiles

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

    # Create dummy classes for type hints
    class FastAPI:
        def __init__(self, *args, **kwargs):
            pass

    class WebSocket:
        pass

    class WebSocketDisconnect(Exception):
        pass

    class HTMLResponse:
        def __init__(self, content, *args, **kwargs):
            self.content = content

    class StaticFiles:
        def __init__(self, *args, **kwargs):
            pass

    class WebSocket:
        pass

    class WebSocketDisconnect:
        pass

    class HTMLResponse:
        pass

    class StaticFiles:
        pass


from m31r.logging.logger import get_logger

logger: logging.Logger = get_logger(__name__)

if not FASTAPI_AVAILABLE:
    logger.warning(
        "FastAPI not installed. Dashboard will not be available. "
        "Install with: pip install fastapi uvicorn"
    )


@dataclass
class MetricsData:
    """Container for all dashboard metrics."""

    steps: list[int] = field(default_factory=list)
    losses: list[float] = field(default_factory=list)
    learning_rates: list[float] = field(default_factory=list)
    grad_norms: list[float] = field(default_factory=list)
    tokens_per_sec: list[float] = field(default_factory=list)
    memory_mb: list[float] = field(default_factory=list)
    tokens_seen: list[int] = field(default_factory=list)
    timestamps: list[float] = field(default_factory=list)

    # Metadata
    current_step: int = 0
    max_steps: int = 0
    start_time: float = field(default_factory=time.time)

    def add_step(self, metrics: dict[str, Any]) -> None:
        """Add a new step's metrics."""
        self.steps.append(metrics.get("step", 0))
        self.losses.append(metrics.get("loss", 0.0))
        self.learning_rates.append(metrics.get("lr", 0.0))
        self.grad_norms.append(metrics.get("grad_norm", 0.0))
        self.tokens_per_sec.append(metrics.get("tokens_per_sec", 0.0))
        self.memory_mb.append(metrics.get("memory_mb", 0.0))
        self.tokens_seen.append(metrics.get("tokens_seen", 0))
        self.timestamps.append(asyncio.get_event_loop().time())

        self.current_step = metrics.get("step", 0)

    def get_summary(self) -> dict[str, Any]:
        """Get current summary for dashboard."""
        if not self.steps:
            return {
                "current_step": 0,
                "max_steps": self.max_steps,
                "progress_pct": 0.0,
                "latest_loss": 0.0,
                "avg_tokens_per_sec": 0.0,
                "elapsed_time": 0.0,
            }

        elapsed = self.timestamps[-1] - self.start_time if self.timestamps else 0
        progress = (self.current_step / self.max_steps * 100) if self.max_steps > 0 else 0

        return {
            "current_step": self.current_step,
            "max_steps": self.max_steps,
            "progress_pct": round(progress, 2),
            "latest_loss": round(self.losses[-1], 6) if self.losses else 0.0,
            "avg_tokens_per_sec": round(sum(self.tokens_per_sec) / len(self.tokens_per_sec), 1)
            if self.tokens_per_sec
            else 0.0,
            "elapsed_time": round(elapsed, 1),
        }


class ConnectionManager:
    """Manage WebSocket connections for real-time updates."""

    def __init__(self) -> None:
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(
            "Dashboard client connected", extra={"total_clients": len(self.active_connections)}
        )

    def disconnect(self, websocket: WebSocket) -> None:
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logger.info(
                "Dashboard client disconnected",
                extra={"total_clients": len(self.active_connections)},
            )

    async def broadcast(self, message: dict[str, Any]) -> None:
        """Broadcast message to all connected clients."""
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                disconnected.append(connection)

        # Clean up disconnected clients
        for conn in disconnected:
            self.disconnect(conn)


class LogTailer:
    """Tail training log files for real-time updates."""

    def __init__(self, log_file: Path | None = None) -> None:
        self.log_file = log_file
        self.log_buffer: deque[dict[str, Any]] = deque(maxlen=1000)
        self._running = False
        self._task: asyncio.Task | None = None

    async def start(self, callback: callable) -> None:
        """Start tailing log file."""
        if self._running or not self.log_file:
            return

        self._running = True
        self._task = asyncio.create_task(self._tail_loop(callback))
        logger.info("Log tailer started", extra={"log_file": str(self.log_file)})

    async def _tail_loop(self, callback: callable) -> None:
        """Main tailing loop."""
        if not self.log_file:
            return

        # For now, just watch for new log entries from stdout
        # In production, this would tail an actual log file
        while self._running:
            await asyncio.sleep(1)

    def stop(self) -> None:
        """Stop tailing."""
        self._running = False
        if self._task:
            self._task.cancel()

    def add_log_entry(self, entry: dict[str, Any]) -> None:
        """Add a log entry to the buffer."""
        self.log_buffer.append(entry)


# Global state
metrics_data = MetricsData()
connection_manager = ConnectionManager()
log_tailer = LogTailer()

# Create FastAPI app
app = FastAPI(title="M31R Training Dashboard", version="1.0.0")

# Try to mount static files
try:
    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
except Exception as e:
    logger.warning("Could not mount static files", extra={"error": str(e)})


@app.get("/", response_class=HTMLResponse)
async def get_dashboard() -> str:
    """Serve the main dashboard HTML."""
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>M31R Training Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0f172a;
            color: #e2e8f0;
            line-height: 1.6;
        }
        
        .header {
            background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
            padding: 1.5rem 2rem;
            border-bottom: 2px solid #3b82f6;
        }
        
        .header h1 {
            font-size: 1.8rem;
            background: linear-gradient(90deg, #60a5fa, #a78bfa);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        .header .subtitle {
            color: #94a3b8;
            font-size: 0.9rem;
            margin-top: 0.25rem;
        }
        
        .container {
            max-width: 1600px;
            margin: 0 auto;
            padding: 2rem;
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2rem;
        }
        
        .stat-card {
            background: #1e293b;
            border-radius: 12px;
            padding: 1.5rem;
            border: 1px solid #334155;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        
        .stat-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 40px rgba(0,0,0,0.3);
        }
        
        .stat-card .label {
            color: #94a3b8;
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        
        .stat-card .value {
            font-size: 2rem;
            font-weight: 700;
            margin-top: 0.5rem;
        }
        
        .stat-card .value.loss { color: #f87171; }
        .stat-card .value.lr { color: #60a5fa; }
        .stat-card .value.speed { color: #34d399; }
        .stat-card .value.progress { color: #a78bfa; }
        
        .charts-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 2rem;
            margin-bottom: 2rem;
        }
        
        .chart-container {
            background: #1e293b;
            border-radius: 12px;
            padding: 1.5rem;
            border: 1px solid #334155;
        }
        
        .chart-container h3 {
            margin-bottom: 1rem;
            color: #e2e8f0;
            font-size: 1.1rem;
        }
        
        .chart-wrapper {
            position: relative;
            height: 300px;
        }
        
        .logs-container {
            background: #1e293b;
            border-radius: 12px;
            border: 1px solid #334155;
            overflow: hidden;
        }
        
        .logs-header {
            background: #334155;
            padding: 1rem 1.5rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .logs-header h3 {
            font-size: 1rem;
        }
        
        .connection-status {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-size: 0.85rem;
        }
        
        .connection-status .dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #ef4444;
            transition: background 0.3s;
        }
        
        .connection-status.connected .dot {
            background: #22c55e;
        }
        
        .logs-content {
            padding: 1rem;
            font-family: 'Monaco', 'Menlo', monospace;
            font-size: 0.85rem;
            max-height: 400px;
            overflow-y: auto;
        }
        
        .log-entry {
            padding: 0.5rem;
            border-left: 3px solid transparent;
            margin-bottom: 0.25rem;
        }
        
        .log-entry:hover {
            background: #334155;
        }
        
        .log-entry.info { border-left-color: #3b82f6; }
        .log-entry.warning { border-left-color: #f59e0b; }
        .log-entry.error { border-left-color: #ef4444; }
        .log-entry.training { border-left-color: #10b981; }
        
        .log-time {
            color: #64748b;
            margin-right: 1rem;
        }
        
        .log-level {
            display: inline-block;
            padding: 0.1rem 0.4rem;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 600;
            margin-right: 1rem;
        }
        
        .log-level.INFO { background: #1e3a8a; color: #60a5fa; }
        .log-level.WARNING { background: #78350f; color: #fbbf24; }
        .log-level.ERROR { background: #7f1d1d; color: #f87171; }
        
        .progress-bar {
            width: 100%;
            height: 8px;
            background: #334155;
            border-radius: 4px;
            overflow: hidden;
            margin-top: 0.5rem;
        }
        
        .progress-bar .fill {
            height: 100%;
            background: linear-gradient(90deg, #3b82f6, #8b5cf6);
            transition: width 0.5s ease;
        }
        
        @media (max-width: 768px) {
            .charts-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🔬 M31R Training Dashboard</h1>
        <div class="subtitle">Real-time training visualization & monitoring</div>
    </div>
    
    <div class="container">
        <!-- Stats Grid -->
        <div class="stats-grid">
            <div class="stat-card">
                <div class="label">Current Step</div>
                <div class="value progress" id="current-step">0 / 0</div>
                <div class="progress-bar">
                    <div class="fill" id="progress-fill" style="width: 0%"></div>
                </div>
            </div>
            
            <div class="stat-card">
                <div class="label">Loss</div>
                <div class="value loss" id="current-loss">0.0000</div>
            </div>
            
            <div class="stat-card">
                <div class="label">Learning Rate</div>
                <div class="value lr" id="current-lr">0.0000</div>
            </div>
            
            <div class="stat-card">
                <div class="label">Throughput</div>
                <div class="value speed" id="current-speed">0 tok/s</div>
            </div>
        </div>
        
        <!-- Charts Grid -->
        <div class="charts-grid">
            <div class="chart-container">
                <h3>📉 Loss Curve</h3>
                <div class="chart-wrapper">
                    <canvas id="lossChart"></canvas>
                </div>
            </div>
            
            <div class="chart-container">
                <h3>📊 Learning Rate Schedule</h3>
                <div class="chart-wrapper">
                    <canvas id="lrChart"></canvas>
                </div>
            </div>
            
            <div class="chart-container">
                <h3>⚡ Tokens/Second</h3>
                <div class="chart-wrapper">
                    <canvas id="speedChart"></canvas>
                </div>
            </div>
            
            <div class="chart-container">
                <h3>📈 Gradient Norm</h3>
                <div class="chart-wrapper">
                    <canvas id="gradChart"></canvas>
                </div>
            </div>
        </div>
        
        <!-- Live Logs -->
        <div class="logs-container">
            <div class="logs-header">
                <h3>📜 Live Training Logs</h3>
                <div class="connection-status" id="connection-status">
                    <span class="dot"></span>
                    <span>Disconnected</span>
                </div>
            </div>
            <div class="logs-content" id="logs-content">
                <div class="log-entry info">
                    <span class="log-time">--:--:--</span>
                    <span class="log-level INFO">INFO</span>
                    <span>Waiting for training to start...</span>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        // Chart.js default configuration
        Chart.defaults.color = '#94a3b8';
        Chart.defaults.borderColor = '#334155';
        Chart.defaults.backgroundColor = 'rgba(59, 130, 246, 0.1)';
        
        // Initialize charts
        const lossChart = new Chart(document.getElementById('lossChart'), {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Loss',
                    data: [],
                    borderColor: '#f87171',
                    backgroundColor: 'rgba(248, 113, 113, 0.1)',
                    borderWidth: 2,
                    tension: 0.4,
                    pointRadius: 0,
                    pointHoverRadius: 4,
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    intersect: false,
                    mode: 'index',
                },
                plugins: {
                    legend: {
                        display: false,
                    },
                },
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Step',
                        },
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Loss',
                        },
                        beginAtZero: true,
                    },
                },
            },
        });
        
        const lrChart = new Chart(document.getElementById('lrChart'), {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Learning Rate',
                    data: [],
                    borderColor: '#60a5fa',
                    backgroundColor: 'rgba(96, 165, 250, 0.1)',
                    borderWidth: 2,
                    tension: 0.4,
                    pointRadius: 0,
                    pointHoverRadius: 4,
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false,
                    },
                },
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Step',
                        },
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Learning Rate',
                        },
                    },
                },
            },
        });
        
        const speedChart = new Chart(document.getElementById('speedChart'), {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Tokens/sec',
                    data: [],
                    borderColor: '#34d399',
                    backgroundColor: 'rgba(52, 211, 153, 0.1)',
                    borderWidth: 2,
                    tension: 0.4,
                    pointRadius: 0,
                    pointHoverRadius: 4,
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false,
                    },
                },
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Step',
                        },
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Tokens/sec',
                        },
                        beginAtZero: true,
                    },
                },
            },
        });
        
        const gradChart = new Chart(document.getElementById('gradChart'), {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Gradient Norm',
                    data: [],
                    borderColor: '#a78bfa',
                    backgroundColor: 'rgba(167, 139, 250, 0.1)',
                    borderWidth: 2,
                    tension: 0.4,
                    pointRadius: 0,
                    pointHoverRadius: 4,
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false,
                    },
                },
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Step',
                        },
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Gradient Norm',
                        },
                        beginAtZero: true,
                    },
                },
            },
        });
        
        // WebSocket connection
        let ws = null;
        let reconnectInterval = null;
        
        function connect() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${protocol}//${window.location.host}/ws`);
            
            ws.onopen = () => {
                console.log('Connected to dashboard');
                updateConnectionStatus(true);
                if (reconnectInterval) {
                    clearInterval(reconnectInterval);
                    reconnectInterval = null;
                }
            };
            
            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                handleMessage(data);
            };
            
            ws.onclose = () => {
                console.log('Disconnected from dashboard');
                updateConnectionStatus(false);
                // Try to reconnect every 3 seconds
                if (!reconnectInterval) {
                    reconnectInterval = setInterval(connect, 3000);
                }
            };
            
            ws.onerror = (error) => {
                console.error('WebSocket error:', error);
            };
        }
        
        function updateConnectionStatus(connected) {
            const status = document.getElementById('connection-status');
            if (connected) {
                status.classList.add('connected');
                status.querySelector('span:last-child').textContent = 'Connected';
            } else {
                status.classList.remove('connected');
                status.querySelector('span:last-child').textContent = 'Disconnected';
            }
        }
        
        function handleMessage(data) {
            if (data.type === 'metrics') {
                updateMetrics(data.payload);
            } else if (data.type === 'log') {
                addLogEntry(data.payload);
            } else if (data.type === 'summary') {
                updateSummary(data.payload);
            }
        }
        
        function updateMetrics(metrics) {
            const steps = metrics.steps || [];
            
            // Update all charts
            lossChart.data.labels = steps;
            lossChart.data.datasets[0].data = metrics.losses || [];
            lossChart.update('none');
            
            lrChart.data.labels = steps;
            lrChart.data.datasets[0].data = metrics.learning_rates || [];
            lrChart.update('none');
            
            speedChart.data.labels = steps;
            speedChart.data.datasets[0].data = metrics.tokens_per_sec || [];
            speedChart.update('none');
            
            gradChart.data.labels = steps;
            gradChart.data.datasets[0].data = metrics.grad_norms || [];
            gradChart.update('none');
        }
        
        function updateSummary(summary) {
            document.getElementById('current-step').textContent = 
                `${summary.current_step} / ${summary.max_steps}`;
            document.getElementById('progress-fill').style.width = 
                `${summary.progress_pct}%`;
            document.getElementById('current-loss').textContent = 
                summary.latest_loss.toFixed(4);
            document.getElementById('current-lr').textContent = 
                summary.latest_lr ? summary.latest_lr.toExponential(2) : '0.00e+00';
            document.getElementById('current-speed').textContent = 
                `${summary.avg_tokens_per_sec} tok/s`;
        }
        
        function addLogEntry(log) {
            const logsContent = document.getElementById('logs-content');
            const entry = document.createElement('div');
            entry.className = `log-entry ${log.level?.toLowerCase() || 'info'}`;
            
            const time = new Date().toLocaleTimeString();
            entry.innerHTML = `
                <span class="log-time">${time}</span>
                <span class="log-level ${log.level}">${log.level}</span>
                <span>${log.message}</span>
            `;
            
            logsContent.insertBefore(entry, logsContent.firstChild);
            
            // Keep only last 100 log entries
            while (logsContent.children.length > 100) {
                logsContent.removeChild(logsContent.lastChild);
            }
        }
        
        // Start connection
        connect();
    </script>
</body>
</html>
    """
    return html_content


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """WebSocket endpoint for real-time updates."""
    await connection_manager.connect(websocket)

    try:
        # Send initial data
        await websocket.send_json(
            {
                "type": "metrics",
                "payload": {
                    "steps": metrics_data.steps,
                    "losses": metrics_data.losses,
                    "learning_rates": metrics_data.learning_rates,
                    "grad_norms": metrics_data.grad_norms,
                    "tokens_per_sec": metrics_data.tokens_per_sec,
                    "memory_mb": metrics_data.memory_mb,
                },
            }
        )

        await websocket.send_json({"type": "summary", "payload": metrics_data.get_summary()})

        # Keep connection alive
        while True:
            await asyncio.sleep(1)

    except WebSocketDisconnect:
        connection_manager.disconnect(websocket)


async def broadcast_metrics(metrics: dict[str, Any]) -> None:
    """Broadcast metrics to all connected clients."""
    metrics_data.add_step(metrics)

    await connection_manager.broadcast(
        {
            "type": "metrics",
            "payload": {
                "steps": metrics_data.steps,
                "losses": metrics_data.losses,
                "learning_rates": metrics_data.learning_rates,
                "grad_norms": metrics_data.grad_norms,
                "tokens_per_sec": metrics_data.tokens_per_sec,
                "memory_mb": metrics_data.memory_mb,
            },
        }
    )

    await connection_manager.broadcast(
        {
            "type": "summary",
            "payload": {
                **metrics_data.get_summary(),
                "latest_lr": metrics.get("lr", 0.0),
            },
        }
    )


async def broadcast_log(log_entry: dict[str, Any]) -> None:
    """Broadcast log entry to all connected clients."""
    await connection_manager.broadcast({"type": "log", "payload": log_entry})
