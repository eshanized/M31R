# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
Dashboard command handler for M31R CLI.
"""

import argparse
import logging
import webbrowser
from pathlib import Path

from m31r.cli.exit_codes import CONFIG_ERROR, RUNTIME_ERROR, SUCCESS
from m31r.cli.commands import _load_and_bootstrap
from m31r.logging.logger import get_logger

logger: logging.Logger = get_logger(__name__)


def handle_dashboard(args: argparse.Namespace) -> int:
    """Start the training visualization dashboard."""
    exit_code, config, logger = _load_and_bootstrap(args, "dashboard")
    if exit_code != SUCCESS:
        return exit_code

    # Check if dashboard dependencies are available
    try:
        import fastapi
        import uvicorn
    except ImportError:
        logger.error(
            "Dashboard requires 'fastapi' and 'uvicorn'. Install with: pip install fastapi uvicorn"
        )
        return CONFIG_ERROR

    from m31r.dashboard.server import app

    try:
        import fastapi
        import uvicorn
    except ImportError:
        logger.error(
            "Dashboard requires 'fastapi' and 'uvicorn'. Install with: pip install fastapi uvicorn"
        )
        return CONFIG_ERROR

    from m31r.dashboard.server import app

    try:
        host = getattr(args, "host", "127.0.0.1")
        port = getattr(args, "port", 8080)

        logger.info(
            "Starting dashboard server",
            extra={
                "host": host,
                "port": port,
                "dry_run": args.dry_run,
            },
        )

        if args.dry_run:
            logger.info("Dry run — would start dashboard", extra={"host": host, "port": port})
            return SUCCESS

        # Open browser if requested
        if getattr(args, "open", False):
            url = f"http://{host}:{port}"
            logger.info("Opening browser", extra={"url": url})
            webbrowser.open(url)

        # Start the dashboard server
        logger.info("Dashboard server starting...", extra={"url": f"http://{host}:{port}"})
        uvicorn.run(
            app,
            host=host,
            port=port,
            log_level="info",
        )

        return SUCCESS

    except Exception as err:
        logger.error("Dashboard failed", extra={"error": str(err)}, exc_info=True)
        return RUNTIME_ERROR
