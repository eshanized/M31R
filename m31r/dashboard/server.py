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
        if not self.log_file or not self.log_file.exists():
            # Wait for file to appear
            while self._running and (not self.log_file or not self.log_file.exists()):
                await asyncio.sleep(1)
        
        if not self.log_file:
            return

        try:
            with open(self.log_file, "r", encoding="utf-8") as f:
                # Go to end of file to tail new entries
                # f.seek(0, 2)
                # Actually, for the dashboard we might want to read from start to catch up
                # if we started late. Let's read from start.
                f.seek(0, 0)
                
                while self._running:
                    line = f.readline()
                    if not line:
                        await asyncio.sleep(0.1)
                        continue
                        
                    try:
                        entry = json.loads(line)
                        # Enrich with metrics if present
                        if "loss" in entry and "step" in entry:
                            metrics_data.add_step(entry)
                        
                        # Add to log buffer
                        self.add_log_entry(entry)
                        
                        # Broadcast
                        if callback:
                            await callback(entry)
                            
                    except json.JSONDecodeError:
                        continue
                        
        except Exception as e:
            logger.error(f"Tail loop failed: {e}")

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
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Fira+Code:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        :root {
            --bg-primary: #0a0e1a;
            --bg-secondary: #111827;
            --bg-tertiary: #1f2937;
            --bg-card: rgba(31, 41, 55, 0.7);
            --text-primary: #f9fafb;
            --text-secondary: #9ca3af;
            --text-muted: #6b7280;
            --accent-blue: #3b82f6;
            --accent-cyan: #06b6d4;
            --accent-purple: #8b5cf6;
            --accent-pink: #ec4899;
            --accent-green: #10b981;
            --accent-yellow: #f59e0b;
            --accent-red: #ef4444;
            --border-color: rgba(75, 85, 99, 0.4);
            --glow-blue: 0 0 20px rgba(59, 130, 246, 0.3);
            --glow-cyan: 0 0 20px rgba(6, 182, 212, 0.3);
            --glow-purple: 0 0 20px rgba(139, 92, 246, 0.3);
        }
        
        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: linear-gradient(135deg, var(--bg-primary) 0%, #0f172a 50%, var(--bg-secondary) 100%);
            background-attachment: fixed;
            color: var(--text-primary);
            line-height: 1.6;
            min-height: 100vh;
        }
        
        .header {
            background: linear-gradient(135deg, rgba(17, 24, 39, 0.95) 0%, rgba(31, 41, 55, 0.95) 100%);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            padding: 2rem 2.5rem;
            border-bottom: 1px solid var(--border-color);
            box-shadow: 0 4px 30px rgba(0, 0, 0, 0.3);
            position: relative;
            overflow: hidden;
        }
        
        .header::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 2px;
            background: linear-gradient(90deg, var(--accent-blue), var(--accent-cyan), var(--accent-purple), var(--accent-pink));
            animation: gradient-shift 8s ease infinite;
            background-size: 300% 100%;
        }
        
        @keyframes gradient-shift {
            0%, 100% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
        }
        
        .header h1 {
            font-size: 2.2rem;
            font-weight: 700;
            background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 50%, #c084fc 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            letter-spacing: -0.02em;
            font-family: 'Fira Code', monospace;
        }
        
        .header .subtitle {
            color: var(--text-secondary);
            font-size: 1rem;
            margin-top: 0.5rem;
            font-weight: 400;
            font-family: 'Fira Code', monospace;
        }
        
        .container {
            max-width: 1800px;
            margin: 0 auto;
            padding: 2.5rem;
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2.5rem;
        }
        
        .stat-card {
            background: var(--bg-card);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border-radius: 16px;
            padding: 1.75rem;
            border: 1px solid var(--border-color);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            overflow: hidden;
        }
        
        .stat-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 2px;
            background: linear-gradient(90deg, transparent, currentColor, transparent);
            opacity: 0;
            transition: opacity 0.3s;
        }
        
        .stat-card:hover {
            transform: translateY(-4px);
            box-shadow: var(--glow-blue);
            border-color: rgba(59, 130, 246, 0.3);
        }
        
        .stat-card:hover::before {
            opacity: 1;
        }
        
        .stat-card .label {
            color: var(--text-secondary);
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            font-weight: 600;
            font-family: 'Fira Code', monospace;
        }
        
        .stat-card .value {
            font-size: 2.5rem;
            font-weight: 700;
            margin-top: 0.75rem;
            font-family: 'Fira Code', monospace;
            letter-spacing: -0.02em;
        }
        
        .stat-card .value.loss { 
            color: var(--accent-red);
            text-shadow: 0 0 20px rgba(239, 68, 68, 0.3);
        }
        .stat-card .value.lr { 
            color: var(--accent-cyan);
            text-shadow: 0 0 20px rgba(6, 182, 212, 0.3);
        }
        .stat-card .value.speed { 
            color: var(--accent-green);
            text-shadow: 0 0 20px rgba(16, 185, 129, 0.3);
        }
        .stat-card .value.progress { 
            color: var(--accent-purple);
            text-shadow: 0 0 20px rgba(139, 92, 246, 0.3);
        }
        
        .progress-bar {
            width: 100%;
            height: 6px;
            background: rgba(75, 85, 99, 0.3);
            border-radius: 3px;
            overflow: hidden;
            margin-top: 1rem;
            position: relative;
        }
        
        .progress-bar .fill {
            height: 100%;
            background: linear-gradient(90deg, var(--accent-blue), var(--accent-purple));
            transition: width 0.5s cubic-bezier(0.4, 0, 0.2, 1);
            box-shadow: 0 0 10px rgba(59, 130, 246, 0.5);
            position: relative;
        }
        
        .progress-bar .fill::after {
            content: '';
            position: absolute;
            top: 0;
            right: 0;
            bottom: 0;
            width: 20px;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3));
            animation: shimmer 2s infinite;
        }
        
        @keyframes shimmer {
            0% { transform: translateX(0); }
            100% { transform: translateX(20px); }
        }
        
        .charts-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2.5rem;
        }
        
        .chart-container {
            background: var(--bg-card);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border-radius: 16px;
            padding: 1.75rem;
            border: 1px solid var(--border-color);
            transition: all 0.3s ease;
        }
        
        .chart-container:hover {
            border-color: rgba(139, 92, 246, 0.3);
            box-shadow: var(--glow-purple);
        }
        
        .chart-container h3 {
            margin-bottom: 1.25rem;
            color: var(--text-primary);
            font-size: 1.1rem;
            font-weight: 600;
            font-family: 'Fira Code', monospace;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .chart-wrapper {
            position: relative;
            height: 320px;
        }
        
        .logs-container {
            background: var(--bg-card);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border-radius: 16px;
            border: 1px solid var(--border-color);
            overflow: hidden;
            transition: all 0.3s ease;
        }
        
        .logs-container:hover {
            border-color: rgba(6, 182, 212, 0.3);
            box-shadow: var(--glow-cyan);
        }
        
        .logs-header {
            background: linear-gradient(135deg, rgba(31, 41, 55, 0.9) 0%, rgba(17, 24, 39, 0.9) 100%);
            padding: 1.25rem 1.75rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--border-color);
        }
        
        .logs-header h3 {
            font-size: 1.1rem;
            font-weight: 600;
            font-family: 'Fira Code', monospace;
            color: var(--text-primary);
        }
        
        .connection-status {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            font-size: 0.9rem;
            font-family: 'Fira Code', monospace;
            padding: 0.5rem 1rem;
            background: rgba(0, 0, 0, 0.2);
            border-radius: 20px;
            border: 1px solid var(--border-color);
        }
        
        .connection-status .dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background: var(--accent-red);
            box-shadow: 0 0 10px var(--accent-red);
            transition: all 0.3s ease;
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        
        .connection-status.connected .dot {
            background: var(--accent-green);
            box-shadow: 0 0 10px var(--accent-green);
            animation: none;
        }
        
        .logs-content {
            padding: 1.25rem;
            font-family: 'Fira Code', monospace;
            font-size: 0.9rem;
            max-height: 450px;
            overflow-y: auto;
            background: rgba(10, 14, 26, 0.5);
        }
        
        .logs-content::-webkit-scrollbar {
            width: 8px;
        }
        
        .logs-content::-webkit-scrollbar-track {
            background: rgba(31, 41, 55, 0.3);
        }
        
        .logs-content::-webkit-scrollbar-thumb {
            background: var(--border-color);
            border-radius: 4px;
        }
        
        .logs-content::-webkit-scrollbar-thumb:hover {
            background: var(--text-muted);
        }
        
        .log-entry {
            padding: 0.75rem 1rem;
            border-left: 3px solid transparent;
            margin-bottom: 0.5rem;
            border-radius: 0 8px 8px 0;
            transition: all 0.2s ease;
            animation: slideIn 0.3s ease;
        }
        
        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateX(-10px);
            }
            to {
                opacity: 1;
                transform: translateX(0);
            }
        }
        
        .log-entry:hover {
            background: rgba(59, 130, 246, 0.1);
            transform: translateX(4px);
        }
        
        .log-entry.info { 
            border-left-color: var(--accent-blue);
            background: rgba(59, 130, 246, 0.05);
        }
        .log-entry.warning { 
            border-left-color: var(--accent-yellow);
            background: rgba(245, 158, 11, 0.05);
        }
        .log-entry.error { 
            border-left-color: var(--accent-red);
            background: rgba(239, 68, 68, 0.05);
        }
        .log-entry.training { 
            border-left-color: var(--accent-green);
            background: rgba(16, 185, 129, 0.05);
        }
        
        .log-time {
            color: var(--text-muted);
            margin-right: 1rem;
            font-size: 0.85rem;
        }
        
        .log-level {
            display: inline-block;
            padding: 0.2rem 0.6rem;
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
