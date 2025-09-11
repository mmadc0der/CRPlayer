#!/usr/bin/env python3
"""
Web-based Annotation Tool for Game State Markup
Flask web application for in-place annotation via browser.
"""

from flask import Flask, render_template
from pathlib import Path
import argparse
import threading
import atexit
import os
import logging

from logging_setup import setup_logging, install_flask_request_hooks

from core.session_manager import SessionManager
from api import create_annotation_api
from db.connection import get_connection
from db.schema import init_db
from db.indexer import run_indexer_loop

# Ensure static/templates resolve regardless of where the process is started
BASE_DIR = Path(__file__).parent.resolve()
app = Flask(
  __name__,
  static_folder=str(BASE_DIR / "static"),
  static_url_path="/annotation/static",
  template_folder=str(BASE_DIR / "templates"),
)

# Initialize logging as early as possible
setup_logging(app_debug=None)
install_flask_request_hooks(app)
logger = logging.getLogger("annotation.app")

# Configure Flask to work behind reverse proxy with URL prefix
app.config["APPLICATION_ROOT"] = "/annotation"

# Handle X-Script-Name header for proper URL generation
from flask import request


@app.before_request
def set_script_name():
  if "X-Script-Name" in request.headers:
    app.config["APPLICATION_ROOT"] = request.headers["X-Script-Name"]


# Bootstrap services and register blueprint
session_manager = SessionManager()

# Register API blueprint at /api/* for backward compatibility
api_bp = create_annotation_api(session_manager, name="annotation_api")
app.register_blueprint(api_bp)

# Initialize SQLite schema (idempotent)
try:
  _conn = get_connection()
  try:
    init_db(_conn)
  finally:
    _conn.close()
except Exception:
  # Keep app running even if DB init fails; API can still operate in file-only mode
  logger.exception("DB initialization skipped due to error")

# -------------------- Background Indexer --------------------
_indexer_stop_event = threading.Event()


def _start_indexer():
  try:
    t = threading.Thread(
      target=run_indexer_loop,
      args=(Path(session_manager.data_root), ),
      kwargs={
        "interval_s": 5.0,
        "jitter_s": 1.0,
        "stop_event": _indexer_stop_event,
      },
      daemon=True,
    )
    t.start()
    logger.info("Background indexer thread started")
  except Exception:
    logger.exception("Background indexer failed to start")


def _stop_indexer():
  try:
    _indexer_stop_event.set()
  except Exception:
    logger.warning("Failed to signal indexer stop", exc_info=True)


def _should_start_indexer() -> bool:
  try:
    # Don't start indexer during testing
    if app.config.get("TESTING"):
      return False
    # If debug reloader is active, only start in the reloader main process
    if app.debug:
      return os.environ.get("WERKZEUG_RUN_MAIN") == "true"
    return True
  except Exception:
    return True


if _should_start_indexer():
  _start_indexer()
atexit.register(_stop_indexer)


@app.route("/")
def index():
  """Main annotation interface (primary access point)."""
  return render_template("index.html")


@app.route("/annotation/")
def annotation_index():
  """Main annotation interface at /annotation/ path (for sub-path mounting)."""
  return render_template("index.html")


## API routes are provided by the annotation_api blueprint

## Legacy /api/load_session removed in favor of stateless endpoints

## Legacy /api/frame/<idx> removed; use GET /api/frame?session_path&project_name&idx

## Legacy /api/image/<idx> removed; use GET /api/image?session_path&idx

## Legacy /api/save_annotation removed; use POST /api/save_annotation with DTO

## Legacy /api/export removed (will be reintroduced stateless in API layer when needed)

## Legacy /api/stats removed; can be re-added stateless via blueprint if needed

## Legacy /api/save_categories removed; to be implemented in blueprint if required

## Legacy /api/projects GET removed

## Legacy /api/projects POST removed

## Legacy /api/datasets removed

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Web-based Annotation Tool")
  parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
  parser.add_argument("--port", type=int, default=5000, help="Port to bind to")
  parser.add_argument("--debug", action="store_true", help="Enable debug mode")

  args = parser.parse_args()

  setup_logging(app_debug=bool(args.debug))
  logger.info("Game State Annotation Tool")
  logger.info("%s", "=" * 40)
  logger.info("Starting web server at http://%s:%s", args.host, args.port)
  logger.info("Open this URL in your browser to start annotating!")

  app.run(host=args.host, port=args.port, debug=args.debug)
