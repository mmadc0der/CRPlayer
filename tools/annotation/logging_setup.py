import logging
import logging.config
import os
import sys
import time
import uuid
from typing import Any, Dict

try:
  from pythonjsonlogger import jsonlogger  # type: ignore
  _HAS_JSON_LOGGER = True
except Exception:
  _HAS_JSON_LOGGER = False

try:
  # Python 3.11+ has task locals for logging; we will use contextvars for request correlation
  import contextvars
  _request_id_ctx: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")
except Exception:
  _request_id_ctx = None  # type: ignore


class RequestContextFilter(logging.Filter):
  """Injects request-scoped fields into log records.

	Adds request_id if available. Ensures missing attributes exist to avoid format KeyError.
	"""

  def filter(self, record: logging.LogRecord) -> bool:  # noqa: D401
    try:
      setattr(record, "request_id", _request_id_ctx.get() if _request_id_ctx else "-")
    except Exception:
      setattr(record, "request_id", "-")
    # Ensure funcName is present (standard), but avoid filename/lineno exposure by not formatting them
    return True


def generate_request_id(header_value: str | None = None) -> str:
  if header_value:
    return header_value.strip()[:128] or str(uuid.uuid4())
  return str(uuid.uuid4())


def _build_text_formatter(dev_mode: bool) -> logging.Formatter:
  if dev_mode:
    fmt = "%(asctime)s %(levelname)s %(name)s:%(funcName)s [req=%(request_id)s] - %(message)s"
  else:
    fmt = "%(asctime)s %(levelname)s %(name)s:%(funcName)s [req=%(request_id)s] - %(message)s"
  datefmt = "%Y-%m-%dT%H:%M:%S%z"
  return logging.Formatter(fmt=fmt, datefmt=datefmt)


def _build_json_formatter() -> logging.Formatter:
  if _HAS_JSON_LOGGER:
    return jsonlogger.JsonFormatter(
      '%(asctime)s %(levelname)s %(name)s %(funcName)s %(message)s %(request_id)s',
      json_ensure_ascii=False,
    )
  # Fallback to text if json logger not available
  return _build_text_formatter(dev_mode=False)


def setup_logging(app_debug: bool | None = None) -> None:
  """Configure root logging for the annotation tool.

	Environment variables:
	- ANNOTATION_LOG_LEVEL: DEBUG, INFO, WARNING, ERROR (default INFO; DEBUG if app_debug True)
	- ANNOTATION_LOG_JSON: 1 to enable JSON logs (default 0)
	- ANNOTATION_LOG_FILE: path to log file (optional; stdout by default)
	- ANNOTATION_LOG_WORKER_PREFIX: prefix to include in logger name (optional)
	"""
  level_name = os.getenv("ANNOTATION_LOG_LEVEL")
  if not level_name and app_debug:
    level_name = "DEBUG"
  level = getattr(logging, (level_name or "INFO").upper(), logging.INFO)

  use_json = os.getenv("ANNOTATION_LOG_JSON", "0").strip() in ("1", "true", "TRUE")
  log_file = os.getenv("ANNOTATION_LOG_FILE")
  dev_mode = bool(app_debug)

  handlers: Dict[str, Dict[str, Any]] = {}
  formatters: Dict[str, Dict[str, Any]] = {}
  filters: Dict[str, Dict[str, Any]] = {
    "request_context": {
      "()": RequestContextFilter
    },
  }

  if use_json and _HAS_JSON_LOGGER:
    formatters["json"] = {"()": _build_json_formatter}
    formatter_name = "json"
  else:
    formatters["text"] = {"()": _build_text_formatter, "dev_mode": dev_mode}
    formatter_name = "text"

  if log_file:
    handlers["file"] = {
      "class": "logging.handlers.RotatingFileHandler",
      "filename": log_file,
      "maxBytes": 10 * 1024 * 1024,
      "backupCount": 3,
      "formatter": formatter_name,
      "filters": ["request_context"],
      "encoding": "utf-8",
    }
    root_handlers = ["file"]
  else:
    handlers["stdout"] = {
      "class": "logging.StreamHandler",
      "stream": sys.stdout,
      "formatter": formatter_name,
      "filters": ["request_context"],
    }
    root_handlers = ["stdout"]

  config: Dict[str, Any] = {
    "version": 1,
    "disable_existing_loggers": False,
    "filters": filters,
    "formatters": formatters,
    "handlers": handlers,
    "root": {
      "level": level,
      "handlers": root_handlers,
    },
    "loggers": {
      # Reduce noise from werkzeug unless debugging
      "werkzeug": {
        "level": "WARNING" if level > logging.DEBUG else "INFO"
      },
      # SQLite can be noisy; keep at WARNING
      "sqlite3": {
        "level": "WARNING"
      },
    },
  }

  logging.config.dictConfig(config)


def install_flask_request_hooks(app) -> None:
  """Attach Flask hooks for request correlation and access logging.

	- Assigns a request_id at the start of each request (from X-Request-Id or generated)
	- Logs a concise access line at INFO on completion
	- Adds timing information at DEBUG
	"""
  from flask import g, request

  @app.before_request
  def _start_request_timer():
    try:
      g._start_time = time.perf_counter()
      req_id = generate_request_id(request.headers.get("X-Request-Id"))
      set_request_id(req_id)
    except Exception:
      logging.getLogger("annotation.access").debug("failed to start request timer", exc_info=True)

  @app.after_request
  def _log_access(response):
    try:
      logger = logging.getLogger("annotation.access")
      duration_ms = None
      try:
        start = getattr(g, "_start_time", None)
        if start is not None:
          duration_ms = int((time.perf_counter() - start) * 1000)
      except Exception:
        logging.getLogger("annotation.access").debug("failed to compute duration", exc_info=True)
      logger.info(
        "%s %s -> %s (%sms)",
        request.method,
        request.full_path if request.query_string else request.path,
        response.status_code,
        str(duration_ms) if duration_ms is not None else "-",
      )
      return response
    except Exception:
      logging.getLogger("annotation.access").debug("access logging failed", exc_info=True)
      return response


def set_request_id(request_id: str) -> None:
  try:
    if _request_id_ctx is not None:
      _request_id_ctx.set(request_id)
  except Exception:
    logging.getLogger("annotation.app").debug("failed to set request_id", exc_info=True)
