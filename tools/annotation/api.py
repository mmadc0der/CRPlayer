from __future__ import annotations

from flask import Blueprint, request, jsonify, send_file, Response
from pathlib import Path
import logging
import json
from datetime import datetime
from typing import Dict, Any

from core.session_manager import SessionManager
from services.session_service import SessionService
from services.annotation_service import AnnotationService
from services.settings_service import SettingsService
from services.dataset_service import DatasetService
from services.download_service import DownloadService
from services.autolabel_service import AutoLabelService, AutoLabelUnavailable
from dto import (
  FrameQuery,
  ImageQuery,
  ErrorResponse,
  SaveRegressionRequest,
  SaveSingleLabelRequest,
  SaveMultilabelRequest,
)
from db.connection import get_connection
from db.connection import get_db_path
from db.indexer import reindex_sessions
from db.projects import (
  list_projects as db_list_projects,
  create_project as db_create_project,
  update_project as db_update_project,
  delete_project as db_delete_project,
  list_datasets as db_list_datasets,
  create_dataset as db_create_dataset,
  get_dataset as db_get_dataset,
  dataset_progress as db_dataset_progress,
  get_dataset_by_name as db_get_dataset_by_name,
  update_dataset as db_update_dataset,
  delete_dataset as db_delete_dataset,
)
import sqlite3
from db.datasets import enroll_session_frames as db_enroll_session_frames
from db.classes import (
  list_dataset_classes as db_list_dataset_classes,
  sync_dataset_classes as db_sync_dataset_classes,
  get_or_create_dataset_class as db_get_or_create_dataset_class,
)


def create_annotation_api(session_manager: SessionManager, name: str = "annotation_api") -> Blueprint:
  bp = Blueprint(name, __name__)
  log = logging.getLogger(f"annotation.api.{name}")

  @bp.before_request
  def _bp_log_request():
    try:
      log.debug("request %s %s qs=%s", request.method, request.path, request.query_string)
    except Exception:
      log.debug("failed to log request", exc_info=True)

  @bp.after_request
  def _bp_log_response(resp):
    try:
      log.debug("response %s %s -> %s", request.method, request.path, resp.status_code)
    except Exception:
      log.debug("failed to log response", exc_info=True)
    return resp

  session_service = SessionService(session_manager)
  annotation_service = AnnotationService(session_manager)
  dataset_service = DatasetService(session_manager)
  settings_service = SettingsService()
  download_service = DownloadService(session_manager)
  autolabel_service = AutoLabelService()

  @bp.route("/api/target_types", methods=["GET"])
  def api_list_target_types():
    try:
      conn = get_connection()
      rows = conn.execute("SELECT id, name FROM target_types ORDER BY id").fetchall()
      return jsonify([{"id": int(r["id"]), "name": r["name"]} for r in rows])
    except Exception as e:
      err = ErrorResponse(code="target_types_error", message="Failed to list target types", details={"error": str(e)})
      return jsonify(err.model_dump()), 500

  # Removed duplicate api_list_projects function - keeping the one at line 187

  # -------------------- Read annotation for a frame --------------------
  @bp.route("/api/annotations/frame", methods=["GET"])
  def api_get_annotation_for_frame():
    """Return annotation info for a given frame (by session + idx) within a dataset.

        Query params:
          - session_id: external session identifier
          - dataset_id: dataset numeric ID
          - idx: zero-based frame index within the session
        Response includes unified annotation row with effective_settings merged
        from dataset-session settings and per-frame override settings.
        """
    try:
      session_id = request.args.get("session_id", "")
      dataset_id = int(request.args.get("dataset_id", "0"))
      idx = int(request.args.get("idx", "-1"))
      if not session_id or dataset_id <= 0 or idx < 0:
        err = ErrorResponse(code="bad_request", message="session_id, dataset_id and idx are required")
        return jsonify(err.model_dump()), 400
    except Exception as e:
      err = ErrorResponse(code="bad_request", message="Invalid query parameters", details={"error": str(e)})
      return jsonify(err.model_dump()), 400

    try:
      # Resolve frame_id from idx
      frame = session_service.get_frame_by_idx(session_id, idx)
      frame_id = str(frame["frame_id"])
      ann = annotation_service.get_annotation_db(session_id, dataset_id, frame_id)
      return jsonify({"annotation": ann})
    except Exception as e:
      err = ErrorResponse(code="annotation_error", message="Failed to fetch annotation", details={"error": str(e)})
      return jsonify(err.model_dump()), 500

  @bp.route("/api/sessions", methods=["GET"])
  def discover_sessions():
    try:
      sessions = session_manager.discover_sessions()
      return jsonify(sessions)
    except Exception as e:
      err = ErrorResponse(code="sessions_error", message="Failed to discover sessions", details={"error": str(e)})
      return jsonify(err.model_dump()), 500

  @bp.route("/api/projects/<int:project_id>", methods=["PUT"])
  def api_update_project(project_id: int):
    try:
      payload = request.get_json(force=True) or {}
      name = payload.get("name")
      description = payload.get("description")
      conn = get_connection()
      # ensure exists
      exists = conn.execute("SELECT 1 FROM projects WHERE id = ?", (project_id, )).fetchone()
      if not exists:
        err = ErrorResponse(code="not_found", message="Project not found")
        return jsonify(err.model_dump()), 404
      db_update_project(conn, project_id, name, description)
      conn.commit()
      row = conn.execute("SELECT id, name, description, created_at FROM projects WHERE id = ?",
                         (project_id, )).fetchone()
      return jsonify(dict(row))
    except sqlite3.IntegrityError as e:
      err = ErrorResponse(code="conflict", message="Project name already exists", details={"error": str(e)})
      return jsonify(err.model_dump()), 409
    except Exception as e:
      err = ErrorResponse(code="update_project_error", message="Failed to update project", details={"error": str(e)})
      return jsonify(err.model_dump()), 500

  @bp.route("/api/projects/<int:project_id>", methods=["DELETE"])
  def api_delete_project(project_id: int):
    try:
      conn = get_connection()
      # Optional force delete: when true, rely on FK ON DELETE CASCADE from datasets -> projects
      force = str(request.args.get("force", "0")).lower() in ("1", "true", "yes")
      if not force:
        # Enforce no datasets under project before delete
        cnt = conn.execute("SELECT COUNT(1) FROM datasets WHERE project_id = ?", (project_id, )).fetchone()[0]
        if cnt:
          err = ErrorResponse(code="conflict", message="Project has datasets; delete them first")
          return jsonify(err.model_dump()), 409
      n = db_delete_project(conn, project_id)
      conn.commit()
      return jsonify({"deleted": int(n)})
    except Exception as e:
      err = ErrorResponse(code="delete_project_error", message="Failed to delete project", details={"error": str(e)})
      return jsonify(err.model_dump()), 500

  @bp.route("/api/reindex", methods=["POST"])
  def reindex():
    """Scan data directories and populate SQLite with sessions/frames."""
    try:
      conn = get_connection()
      try:
        log.info("reindex start db_path=%s", get_db_path())
      except Exception:
        log.debug("failed to log db path", exc_info=True)
      summary = reindex_sessions(conn, Path(session_manager.data_root))
      log.info("reindex summary sessions=%s frames=%s", summary.get("sessions_indexed"), summary.get("frames_indexed"))
      return jsonify({
        "ok": True,
        "summary": summary,
      })
    except Exception as e:
      log.exception("reindex_error")
      err = ErrorResponse(code="reindex_error", message="Failed to reindex", details={"error": str(e)})
      return jsonify(err.model_dump()), 500

  # -------------------- Projects & Datasets CRUD --------------------
  @bp.route("/api/projects", methods=["GET"])
  def api_list_projects():
    try:
      conn = get_connection()
      return jsonify(db_list_projects(conn))
    except Exception as e:
      err = ErrorResponse(code="projects_error", message="Failed to list projects", details={"error": str(e)})
      return jsonify(err.model_dump()), 500

  @bp.route("/api/projects", methods=["POST"])
  def api_create_project():
    try:
      payload = request.get_json(force=True) or {}
      name = payload.get("name")
      description = payload.get("description")
      if not name:
        err = ErrorResponse(code="bad_request", message="name is required")
        return jsonify(err.model_dump()), 400
      conn = get_connection()
      try:
        pid = db_create_project(conn, name, description)
        conn.commit()
        # Fetch the created project with created_at timestamp
        cur = conn.execute("SELECT id, name, description, created_at FROM projects WHERE id = ?", (pid, ))
        row = cur.fetchone()
        if row:
          return jsonify(dict(row)), 201
        else:
          return jsonify({"id": pid, "name": name, "description": description}), 201
      except sqlite3.IntegrityError:
        # Unique constraint on projects.name; return existing
        # Fetch existing row id
        cur = conn.execute("SELECT id, name, description, created_at FROM projects WHERE name = ?", (name, ))
        row = cur.fetchone()
        if row:
          return jsonify(dict(row)), 200
        raise
      except Exception:
        conn.rollback()
        raise
    except Exception as e:
      err = ErrorResponse(code="create_project_error", message="Failed to create project", details={"error": str(e)})
      return jsonify(err.model_dump()), 500

  @bp.route("/api/projects/<int:project_id>/datasets", methods=["GET"])
  def api_list_datasets(project_id: int):
    try:
      conn = get_connection()
      return jsonify(db_list_datasets(conn, project_id))
    except Exception as e:
      err = ErrorResponse(code="datasets_error", message="Failed to list datasets", details={"error": str(e)})
      return jsonify(err.model_dump()), 500

  @bp.route("/api/datasets/<int:dataset_id>", methods=["GET"])
  def api_get_dataset(dataset_id: int):
    try:
      conn = get_connection()
      d = db_get_dataset(conn, dataset_id)
      if not d:
        err = ErrorResponse(code="not_found", message="Dataset not found")
        return jsonify(err.model_dump()), 404
      return jsonify(d)
    except Exception as e:
      err = ErrorResponse(code="dataset_error", message="Failed to get dataset", details={"error": str(e)})
      return jsonify(err.model_dump()), 500

  @bp.route("/api/projects/<int:project_id>/datasets", methods=["POST"])
  def api_create_dataset(project_id: int):
    try:
      payload = request.get_json(force=True) or {}
      name = payload.get("name")
      description = payload.get("description")
      target_type_id = payload.get("target_type_id")
      if not name or target_type_id is None:
        err = ErrorResponse(code="bad_request", message="name and target_type_id are required")
        return jsonify(err.model_dump()), 400
      conn = get_connection()
      # Ensure the parent project exists; otherwise, the insert will raise a FK error
      proj_row = conn.execute("SELECT id FROM projects WHERE id = ?", (project_id, )).fetchone()
      if not proj_row:
        err = ErrorResponse(code="not_found", message="Project not found", details={"project_id": project_id})
        return jsonify(err.model_dump()), 404
      try:
        ttid = int(target_type_id)
        # Validate target type exists in DB
        exists = conn.execute("SELECT 1 FROM target_types WHERE id = ?", (ttid, )).fetchone()
        if not exists:
          err = ErrorResponse(code="bad_request", message="unknown target_type_id", details={"target_type_id": ttid})
          return jsonify(err.model_dump()), 400
        did = db_create_dataset(conn, project_id, name, description, ttid)
        conn.commit()
        # Fetch full dataset row to include target_type_name
        created = db_get_dataset(conn, did)
        if created:
          return jsonify(created), 201
        return (
          jsonify({
            "id": did,
            "project_id": project_id,
            "name": name,
            "description": description,
            "target_type_id": ttid,
          }),
          201,
        )
      except sqlite3.IntegrityError:
        # Unique(project_id, name) exists: return existing entry as 200
        existing = db_get_dataset_by_name(conn, project_id, name)
        if existing:
          return jsonify(existing), 200
        raise
    except Exception as e:
      err = ErrorResponse(code="create_dataset_error", message="Failed to create dataset", details={"error": str(e)})
      return jsonify(err.model_dump()), 500

  @bp.route("/api/datasets/<int:dataset_id>", methods=["PUT"])
  def api_update_dataset(dataset_id: int):
    try:
      payload = request.get_json(force=True) or {}
      name = payload.get("name")
      description = payload.get("description")
      ttid = payload.get("target_type_id")
      ttid_val = None
      if ttid is not None:
        ttid_val = int(ttid)
        # Validate target type exists in DB
        conn = get_connection()
        tt_exists = conn.execute("SELECT 1 FROM target_types WHERE id = ?", (ttid_val, )).fetchone()
        if not tt_exists:
          err = ErrorResponse(code="bad_request",
                              message="unknown target_type_id",
                              details={"target_type_id": ttid_val})
          return jsonify(err.model_dump()), 400
      conn = get_connection()
      existing = db_get_dataset(conn, dataset_id)
      if not existing:
        err = ErrorResponse(code="not_found", message="Dataset not found")
        return jsonify(err.model_dump()), 404
      db_update_dataset(conn, dataset_id, name=name, description=description, target_type_id=ttid_val)
      conn.commit()
      updated = db_get_dataset(conn, dataset_id)
      return jsonify(updated)
    except sqlite3.IntegrityError as e:
      err = ErrorResponse(code="conflict", message="Dataset name already exists in project", details={"error": str(e)})
      return jsonify(err.model_dump()), 409
    except Exception as e:
      err = ErrorResponse(code="update_dataset_error", message="Failed to update dataset", details={"error": str(e)})
      return jsonify(err.model_dump()), 500

  @bp.route("/api/datasets/<int:dataset_id>", methods=["DELETE"])
  def api_delete_dataset(dataset_id: int):
    try:
      conn = get_connection()
      # Optional force delete: when true, rely on FK ON DELETE CASCADE from dependents -> datasets
      force = str(request.args.get("force", "0")).lower() in ("1", "true", "yes")
      if not force:
        cnt = conn.execute("SELECT COUNT(1) FROM annotations WHERE dataset_id = ?", (dataset_id, )).fetchone()[0]
        if cnt:
          err = ErrorResponse(code="conflict", message="Dataset has annotations; cannot delete")
          return jsonify(err.model_dump()), 409
      n = db_delete_dataset(conn, dataset_id)
      conn.commit()
      return jsonify({"deleted": int(n)})
    except Exception as e:
      err = ErrorResponse(code="delete_dataset_error", message="Failed to delete dataset", details={"error": str(e)})
      return jsonify(err.model_dump()), 500

  # Enroll session frames into dataset (creates annotations rows with status=unlabeled)
  @bp.route("/api/datasets/<int:dataset_id>/enroll_session", methods=["POST"])
  def api_enroll_session(dataset_id: int):
    try:
      payload = request.get_json(force=True) or {}
      session_id = payload.get("session_id")
      settings = payload.get("settings")  # optional default settings_json
      if not session_id:
        err = ErrorResponse(code="bad_request", message="session_id is required")
        return jsonify(err.model_dump()), 400
      conn = get_connection()
      # Ensure dataset exists
      d = db_get_dataset(conn, dataset_id)
      if not d:
        err = ErrorResponse(code="not_found", message="Dataset not found")
        return jsonify(err.model_dump()), 404
      # If baseline settings include categories, ensure dataset_classes exist in DB with stable ordering
      try:
        if isinstance(settings, dict):
          cats = settings.get("categories")
          if isinstance(cats, list) and len(cats) > 0:
            # Sync classes regardless of target type for forward-compat; safe no-op if duplicates
            db_sync_dataset_classes(conn, dataset_id, [str(c) for c in cats])
      except Exception:
        # Do not fail enrollment on class sync issues; continue
        log.warning("class sync failed during enrollment", exc_info=True)
      summary = db_enroll_session_frames(conn, dataset_id, session_id, settings)
      return jsonify({"ok": True, "summary": summary})
    except ValueError as ve:
      err = ErrorResponse(code="bad_request", message=str(ve))
      return jsonify(err.model_dump()), 400
    except Exception as e:
      err = ErrorResponse(code="enroll_error", message="Failed to enroll session", details={"error": str(e)})
      return jsonify(err.model_dump()), 500

  @bp.route("/api/datasets/<int:dataset_id>/progress", methods=["GET"])
  def api_dataset_progress(dataset_id: int):
    try:
      conn = get_connection()
      # Base progress
      base = db_dataset_progress(conn, dataset_id)
      # Enrich with dataset meta
      d = db_get_dataset(conn, dataset_id)
      name = d.get("name") if d else None
      try:
        row = conn.execute("SELECT COUNT(1) FROM dataset_classes WHERE dataset_id = ?", (dataset_id, )).fetchone()
        classes_count = int(row[0]) if row and row[0] is not None else 0
      except Exception:
        log.debug("failed to count dataset classes in progress", exc_info=True)
        classes_count = 0
      # Per-session breakdown
      sess_rows = conn.execute(
        """
                SELECT s.session_id AS session_id,
                       COUNT(1) AS total,
                       SUM(CASE WHEN a.status = 'labeled' THEN 1 ELSE 0 END) AS labeled
                FROM annotations a
                JOIN frames f   ON f.id = a.frame_id
                JOIN sessions s ON s.id = f.session_id
                WHERE a.dataset_id = ?
                GROUP BY s.session_id
                ORDER BY s.session_id
                """,
        (dataset_id, ),
      ).fetchall()
      sessions = []
      for r in sess_rows or []:
        sid = str(r["session_id"] if isinstance(r, dict) else r[0])
        total = int(r["total"] if isinstance(r, dict) else r[1])
        labeled = int((r["labeled"] if isinstance(r, dict) else r[2]) or 0)
        sessions.append({
          "session_id": sid,
          "total": total,
          "labeled": labeled,
          "unlabeled": int(max(0, total - labeled)),
        })
      out = {
        # Back-compat keys
        "total": int(base.get("total") or 0),
        "labeled": int(base.get("labeled") or base.get("annotated") or 0),
        "unlabeled": int(max(0,
                             int(base.get("total") or 0) - int(base.get("labeled") or base.get("annotated") or 0))),
        # Enriched
        "name": name,
        "classes_count": int(classes_count),
        "sessions": sessions,
      }
      return jsonify(out)
    except Exception as e:
      err = ErrorResponse(code="progress_error", message="Failed to get dataset progress", details={"error": str(e)})
      return jsonify(err.model_dump()), 500

  @bp.route("/api/projects/<int:project_id>/progress", methods=["GET"])
  def api_project_progress(project_id: int):
    """Aggregate progress across all datasets under a project.

        Returns:
          { total, labeled, unlabeled, datasets?: [ { id, name, classes_count, total, labeled, unlabeled } ] }
        """
    try:
      conn = get_connection()
      # Total annotations across datasets in the project
      total_row = conn.execute(
        """
                SELECT COUNT(1) AS cnt
                FROM annotations a
                JOIN datasets d ON d.id = a.dataset_id
                WHERE d.project_id = ?
                """,
        (project_id, ),
      ).fetchone()
      labeled_row = conn.execute(
        """
                SELECT COUNT(1) AS cnt
                FROM annotations a
                JOIN datasets d ON d.id = a.dataset_id
                WHERE d.project_id = ? AND a.status = 'labeled'
                """,
        (project_id, ),
      ).fetchone()
      total = int(total_row[0] if total_row and total_row[0] is not None else 0)
      labeled = int(labeled_row[0] if labeled_row and labeled_row[0] is not None else 0)
      result = {
        "total": total,
        "labeled": labeled,
        "unlabeled": int(max(0, total - labeled)),
      }
      # Enrich with per-dataset summaries
      try:
        ds_list = db_list_datasets(conn, project_id)
        datasets = []
        for d in ds_list:
          did = int(d["id"])
          prog = db_dataset_progress(conn, did)
          try:
            row = conn.execute("SELECT COUNT(1) FROM dataset_classes WHERE dataset_id = ?", (did, )).fetchone()
            classes_count = int(row[0]) if row and row[0] is not None else 0
          except Exception:
            log.debug("failed to count classes for dataset %s", did, exc_info=True)
            classes_count = 0
          datasets.append({
            "id":
            did,
            "name":
            d.get("name"),
            "classes_count":
            classes_count,
            "total":
            int(prog.get("total") or 0),
            "labeled":
            int(prog.get("labeled") or prog.get("annotated") or 0),
            "unlabeled":
            int(max(
              0,
              int(prog.get("total") or 0) - int(prog.get("labeled") or prog.get("annotated") or 0),
            )),
          })
        result["datasets"] = datasets
      except Exception:
        log.warning("failed to build project datasets summary", exc_info=True)
      return jsonify(result)
    except Exception as e:
      err = ErrorResponse(code="progress_error", message="Failed to get project progress", details={"error": str(e)})
      return jsonify(err.model_dump()), 500

  # Enrollments inspection helpers
  @bp.route("/api/datasets/<int:dataset_id>/enrollments", methods=["GET"])
  def api_list_enrollments(dataset_id: int):
    """Return list of external session_ids enrolled in the dataset.

        We infer enrollment from presence of rows in annotations for frames belonging
        to a session within the given dataset. The annotations table does not carry
        a session_id column, so we join through frames -> sessions.
        """
    try:
      conn = get_connection()
      rows = conn.execute(
        """
                SELECT DISTINCT s.session_id
                FROM annotations a
                JOIN frames f   ON f.id = a.frame_id
                JOIN sessions s ON s.id = f.session_id
                WHERE a.dataset_id = ?
                """,
        (dataset_id, ),
      ).fetchall()
      out = [str(r[0]) for r in rows if r and r[0] is not None]
      return jsonify(out)
    except Exception as e:
      err = ErrorResponse(code="enrollments_error", message="Failed to list enrollments", details={"error": str(e)})
      return jsonify(err.model_dump()), 500

  @bp.route("/api/datasets/<int:dataset_id>/sessions/<session_id>/enrolled", methods=["GET"])
  def api_is_enrolled(dataset_id: int, session_id: str):
    """Return whether the given session is enrolled in the dataset."""
    try:
      conn = get_connection()
      row = conn.execute(
        """
                SELECT 1
                FROM annotations a
                JOIN frames f   ON f.id = a.frame_id
                JOIN sessions s ON s.id = f.session_id
                WHERE a.dataset_id = ? AND s.session_id = ?
                LIMIT 1
                """,
        (dataset_id, session_id),
      ).fetchone()
      return jsonify({"enrolled": bool(row is not None)})
    except Exception as e:
      err = ErrorResponse(code="enrollment_check_error",
                          message="Failed to check enrollment",
                          details={"error": str(e)})
      return jsonify(err.model_dump()), 500

  @bp.route("/api/datasets/<int:dataset_id>/labeled", methods=["GET"])
  def api_dataset_labeled(dataset_id: int):
    try:
      # Use service to enrich with frame_path_rel based on metadata and resolvers
      rows = dataset_service.fetch_labeled(dataset_id)
      return jsonify(rows)
    except Exception as e:
      err = ErrorResponse(code="labeled_error", message="Failed to list labeled items", details={"error": str(e)})
      return jsonify(err.model_dump()), 500

  # -------------------- Dataset download (export) --------------------
  # NOTE: The lightweight CSV/JSONL download endpoint implemented earlier has
  # been removed to avoid duplicate route registration with the advanced
  # export endpoint below. Keep only one implementation to prevent Flask
  # endpoint overwrite assertions during tests.

  @bp.route("/api/datasets/<int:dataset_id>/sessions/<session_id>/unlabeled_indices", methods=["GET"])
  def api_unlabeled_indices(dataset_id: int, session_id: str):
    """Return zero-based frame indices (ordered by frame_id) that are not labeled
       for the given dataset-session pair, plus simple counts.

       This allows the frontend to navigate only unlabeled frames with a single call.
    """
    try:
      # Fetch all rows (ordered by frame_id) for the session with annotation status
      # Important: Order frames the same way the /api/frame index navigation does so indices align 1:1.
      conn = get_connection()
      q = conn.execute(
        ("""
          SELECT f.id          AS frame_db_id,
                 f.frame_id    AS frame_id,
                 f.ts_ms       AS ts_ms,
                 a.status      AS status
            FROM frames f
            JOIN sessions s ON s.id = f.session_id
       LEFT JOIN annotations a
              ON a.frame_id = f.id AND a.dataset_id = ?
           WHERE s.session_id = ?
        ORDER BY COALESCE(f.ts_ms, f.frame_id), f.frame_id
          """),
        (int(dataset_id), str(session_id)),
      )
      rows = [dict(r) for r in q.fetchall()]
      indices: list[int] = []
      labeled_count = 0
      total = len(rows)
      for i, r in enumerate(rows):
        status = str((r.get("status") or "")).lower()
        if status == "labeled":
          labeled_count += 1
        else:
          indices.append(i)
      out = {
        "indices": indices,
        "total": total,
        "labeled": labeled_count,
        "unlabeled": int(max(0, total - labeled_count)),
      }
      return jsonify(out)
    except FileNotFoundError as e:
      err = ErrorResponse(code="not_found", message=str(e))
      return jsonify(err.model_dump()), 404
    except Exception as e:
      err = ErrorResponse(code="unlabeled_error", message="Failed to list unlabeled indices", details={"error": str(e)})
      return jsonify(err.model_dump()), 500

  @bp.route("/api/datasets/<int:dataset_id>/classes", methods=["GET"])
  def api_list_dataset_classes(dataset_id: int):
    try:
      conn = get_connection()
      # Ensure dataset exists
      d = db_get_dataset(conn, dataset_id)
      if not d:
        err = ErrorResponse(code="not_found", message="Dataset not found")
        return jsonify(err.model_dump()), 404
      rows = db_list_dataset_classes(conn, dataset_id)
      return jsonify(rows)
    except Exception as e:
      err = ErrorResponse(code="classes_error", message="Failed to list dataset classes", details={"error": str(e)})
      return jsonify(err.model_dump()), 500

  @bp.route("/api/frame", methods=["GET"])
  def get_frame():
    try:
      try:
        from flask import request as _rq

        log.debug("frame request path=%s full_path=%s qs=%s", _rq.path, _rq.full_path, _rq.query_string)
      except Exception:
        log.debug("failed to log frame request attributes", exc_info=True)
      q = FrameQuery(session_id=request.args.get("session_id", ""), idx=int(request.args.get("idx", "-1")))
      dataset_id_raw = request.args.get("dataset_id")
      dataset_id = (int(dataset_id_raw) if (dataset_id_raw is not None and str(dataset_id_raw).strip() != "") else None)
    except Exception as e:
      err = ErrorResponse(code="bad_request", message="Invalid query parameters", details={"error": str(e)})
      return jsonify(err.model_dump()), 400

    try:
      sid = (q.session_id or "").strip()
      frame = session_service.get_frame_by_idx(sid, q.idx)
      # Include annotation when dataset_id is provided
      if dataset_id and dataset_id > 0:
        try:
          frame_id = str(frame["frame_id"])
          ann = annotation_service.get_annotation_db(sid, int(dataset_id), frame_id)
        except Exception:
          log.exception("failed to fetch annotation for frame during /api/frame")
          ann = None
        return jsonify({"frame": frame, "annotation": ann})
      # Backward compatibility: only frame when no dataset_id
      return jsonify({"frame": frame})
    except IndexError as e:
      err = ErrorResponse(code="not_found", message=str(e))
      return jsonify(err.model_dump()), 404
    except FileNotFoundError as e:
      err = ErrorResponse(code="not_found", message=str(e))
      return jsonify(err.model_dump()), 404
    except Exception as e:
      log.exception("frame_error")
      err = ErrorResponse(code="frame_error", message="Failed to fetch frame", details={"error": str(e)})
      return jsonify(err.model_dump()), 500

  @bp.route("/api/image", methods=["GET"])
  def get_image():
    try:
      q = ImageQuery(session_id=request.args.get("session_id", ""), idx=int(request.args.get("idx", "-1")))
    except Exception as e:
      err = ErrorResponse(code="bad_request", message="Invalid query parameters", details={"error": str(e)})
      return jsonify(err.model_dump()), 400

    try:
      abs_path, _frame = session_service.get_frame_for_image(q.session_id, q.idx)
      # Enable client/proxy caching for immutable frames and 304 handling on revalidation
      resp = send_file(abs_path, conditional=True, etag=True, max_age=60 * 60 * 24 * 365)
      try:
        resp.headers["Cache-Control"] = "public, max-age=31536000, immutable"
        resp.headers.setdefault("Accept-Ranges", "bytes")
      except Exception:
        pass
      return resp
    except FileNotFoundError as e:
      err = ErrorResponse(code="not_found", message=str(e))
      return jsonify(err.model_dump()), 404
    except Exception as e:
      log.exception("image_error")
      err = ErrorResponse(code="image_error", message="Failed to load image", details={"error": str(e)})
      return jsonify(err.model_dump()), 500

  # Legacy filesystem save endpoint removed in favor of DB-backed endpoints above

  # -------------------- DB-backed annotation write endpoints --------------------
  @bp.route("/api/annotations/regression", methods=["POST"])
  def api_save_regression():
    try:
      payload = SaveRegressionRequest.model_validate(request.get_json())
    except Exception as e:
      err = ErrorResponse(code="bad_request", message="Invalid payload", details={"error": str(e)})
      return jsonify(err.model_dump()), 400
    try:
      frame_id = payload.frame_id
      if frame_id is None and payload.frame_idx is not None:
        frame = session_service.get_frame_by_idx(payload.session_id, int(payload.frame_idx))
        frame_id = str(frame["frame_id"])
      elif frame_id is None:
        err = ErrorResponse(code="bad_request", message="Either frame_id or frame_idx must be provided")
        return jsonify(err.model_dump()), 400

      res = annotation_service.save_regression(
        session_id=payload.session_id,
        dataset_id=payload.dataset_id,
        frame_id=str(frame_id),
        value=payload.value,
        override_settings=payload.override_settings,
      )
      return jsonify({"success": True, "annotation": res})
    except ValueError as e:
      err = ErrorResponse(code="validation_error", message=str(e))
      return jsonify(err.model_dump()), 400
    except FileNotFoundError as e:
      err = ErrorResponse(code="not_found", message=str(e))
      return jsonify(err.model_dump()), 404
    except Exception as e:
      err = ErrorResponse(code="save_error", message="Failed to save regression", details={"error": str(e)})
      return jsonify(err.model_dump()), 500

  @bp.route("/api/annotations/single_label", methods=["POST"])
  def api_save_single_label():
    try:
      payload = SaveSingleLabelRequest.model_validate(request.get_json())
    except Exception as e:
      err = ErrorResponse(code="bad_request", message="Invalid payload", details={"error": str(e)})
      return jsonify(err.model_dump()), 400
    try:
      # Ensure dataset target_type matches single-label; if not, auto-align to SingleLabelClassification (1)
      try:
        conn = get_connection()
        d = db_get_dataset(conn, int(payload.dataset_id))
        if d and d.get("target_type_name") != "SingleLabelClassification":
          db_update_dataset(conn, int(payload.dataset_id), target_type_id=1)
          conn.commit()
        conn.close()
      except Exception:
        # Non-fatal; DB triggers will still protect integrity
        log.warning("failed to align dataset target_type to single-label; continuing", exc_info=True)

      frame_id = payload.frame_id
      if frame_id is None and payload.frame_idx is not None:
        frame = session_service.get_frame_by_idx(payload.session_id, int(payload.frame_idx))
        frame_id = str(frame["frame_id"])
      elif frame_id is None:
        err = ErrorResponse(code="bad_request", message="Either frame_id or frame_idx must be provided")
        return jsonify(err.model_dump()), 400

      # Resolve class_id if only category_name was provided
      class_id = payload.class_id
      if class_id is None and payload.category_name:
        try:
          conn = get_connection()
          cls = db_get_or_create_dataset_class(conn, int(payload.dataset_id), str(payload.category_name))
          conn.commit()
          class_id = int(cls["id"])
          conn.close()
        except Exception:
          log.warning("failed to resolve class_id from category_name; continuing", exc_info=True)
          class_id = None
      if class_id is None:
        err = ErrorResponse(code="bad_request", message="class_id or category_name must resolve to a class")
        return jsonify(err.model_dump()), 400
      res = annotation_service.save_single_label(
        session_id=payload.session_id,
        dataset_id=payload.dataset_id,
        frame_id=str(frame_id),
        class_id=int(class_id),
        override_settings=payload.override_settings,
      )
      return jsonify({"success": True, "annotation": res})
    except ValueError as e:
      err = ErrorResponse(code="validation_error", message=str(e))
      return jsonify(err.model_dump()), 400
    except FileNotFoundError as e:
      err = ErrorResponse(code="not_found", message=str(e))
      return jsonify(err.model_dump()), 404
    except Exception as e:
      err = ErrorResponse(code="save_error", message="Failed to save single label", details={"error": str(e)})
      return jsonify(err.model_dump()), 500

  @bp.route("/api/annotations/multilabel", methods=["POST"])
  def api_save_multilabel():
    try:
      payload = SaveMultilabelRequest.model_validate(request.get_json())
    except Exception as e:
      err = ErrorResponse(code="bad_request", message="Invalid payload", details={"error": str(e)})
      return jsonify(err.model_dump()), 400
    try:
      frame_id = payload.frame_id
      if frame_id is None and payload.frame_idx is not None:
        frame = session_service.get_frame_by_idx(payload.session_id, int(payload.frame_idx))
        frame_id = str(frame["frame_id"])
      # Resolve any provided category_names into class_ids
      class_ids = list(payload.class_ids or [])
      if getattr(payload, "category_names", None):
        try:
          conn = get_connection()
          for nm in payload.category_names or []:
            if not nm:
              continue
            cls = db_get_or_create_dataset_class(conn, int(payload.dataset_id), str(nm))
            class_ids.append(int(cls["id"]))
          # Deduplicate while preserving order
          seen = set()
          dedup: list[int] = []
          for cid in class_ids:
            if cid in seen:
              continue
            seen.add(cid)
            dedup.append(int(cid))
          class_ids = dedup
          conn.commit()
        except Exception:
          log.warning("failed to resolve some category_names to class_ids; continuing", exc_info=True)
      res = annotation_service.save_multilabel(
        session_id=payload.session_id,
        dataset_id=payload.dataset_id,
        frame_id=str(frame_id),
        class_ids=class_ids,
        override_settings=payload.override_settings,
      )
      return jsonify(res)
    except Exception as e:
      err = ErrorResponse(code="save_error", message="Failed to save multilabel", details={"error": str(e)})
      return jsonify(err.model_dump()), 500

  @bp.route("/api/annotations/batch", methods=["POST"])
  def api_save_batch_annotations():
    """Save multiple annotations in a single request.

    Payload:
      { session_id, dataset_id, annotations: [{ frame_idx, category_name?, class_id?, value_real?, class_ids? }] }

    Response:
      { saved: count, errors: count, results: [{ frame_idx, success: bool, error?: string }] }
    """
    try:
      payload = request.get_json(force=True) or {}
      session_id = str(payload.get("session_id") or "").strip()
      dataset_id = int(payload.get("dataset_id") or 0)
      annotations = payload.get("annotations", [])

      if not session_id or dataset_id <= 0 or not annotations or not isinstance(annotations, list):
        err = ErrorResponse(code="bad_request", message="session_id, dataset_id and annotations array are required")
        return jsonify(err.model_dump()), 400

      if len(annotations) > 100:  # Limit batch size to prevent memory issues
        err = ErrorResponse(code="bad_request", message="Batch size limited to 100 annotations")
        return jsonify(err.model_dump()), 400

    except Exception as e:
      err = ErrorResponse(code="bad_request", message="Invalid payload", details={"error": str(e)})
      return jsonify(err.model_dump()), 400

    try:
      results = []
      saved_count = 0
      error_count = 0

      # Process each annotation
      for ann in annotations:
        try:
          frame_idx = int(ann.get("frame_idx", -1))
          if frame_idx < 0:
            results.append({"frame_idx": frame_idx, "success": False, "error": "Invalid frame_idx"})
            error_count += 1
            continue

          # Resolve frame_id from frame_idx
          frame = session_service.get_frame_by_idx(session_id, frame_idx)
          frame_id = str(frame["frame_id"])

          # Determine annotation type based on payload
          if ann.get("category_name") or ann.get("class_id"):
            # Single label annotation
            category_name = ann.get("category_name")
            class_id = ann.get("class_id")

            # Resolve class_id if only category_name provided
            if class_id is None and category_name:
              try:
                conn = get_connection()
                cls = db_get_or_create_dataset_class(conn, int(dataset_id), str(category_name))
                conn.commit()
                class_id = int(cls["id"])
                conn.close()
              except Exception:
                log.warning("failed to resolve class_id from category_name in batch; continuing", exc_info=True)
                class_id = None

            if class_id is None:
              results.append({
                "frame_idx": frame_idx,
                "success": False,
                "error": "class_id or category_name must resolve to a class"
              })
              error_count += 1
              continue

            res = annotation_service.save_single_label(session_id=session_id,
                                                       dataset_id=dataset_id,
                                                       frame_id=frame_id,
                                                       class_id=int(class_id),
                                                       override_settings=None)

          elif ann.get("value_real") is not None:
            # Regression annotation
            value_real = float(ann.get("value_real"))
            res = annotation_service.save_regression(session_id=session_id,
                                                     dataset_id=dataset_id,
                                                     frame_id=frame_id,
                                                     value=value_real,
                                                     override_settings=None)

          elif ann.get("class_ids"):
            # Multi-label annotation
            class_ids = ann.get("class_ids", [])
            if not isinstance(class_ids, list):
              class_ids = []

            # Resolve any category_names to class_ids
            resolved_class_ids = []
            if ann.get("category_names"):
              try:
                conn = get_connection()
                for nm in ann.get("category_names", []):
                  if not nm:
                    continue
                  cls = db_get_or_create_dataset_class(conn, int(dataset_id), str(nm))
                  resolved_class_ids.append(int(cls["id"]))
                conn.commit()
                conn.close()
              except Exception:
                log.warning("failed to resolve category_names to class_ids in batch; continuing", exc_info=True)

            # Combine with provided class_ids
            all_class_ids = list(class_ids) + resolved_class_ids
            # Deduplicate while preserving order
            seen = set()
            dedup_class_ids = []
            for cid in all_class_ids:
              if cid not in seen:
                seen.add(cid)
                dedup_class_ids.append(int(cid))

            if not dedup_class_ids:
              results.append({"frame_idx": frame_idx, "success": False, "error": "No valid class_ids provided"})
              error_count += 1
              continue

            res = annotation_service.save_multilabel(session_id=session_id,
                                                     dataset_id=dataset_id,
                                                     frame_id=frame_id,
                                                     class_ids=dedup_class_ids,
                                                     override_settings=None)

          else:
            results.append({"frame_idx": frame_idx, "success": False, "error": "No valid annotation data provided"})
            error_count += 1
            continue

          results.append({"frame_idx": frame_idx, "success": True})
          saved_count += 1

        except Exception as e:
          log.warning("Failed to save annotation for frame_idx %s: %s", ann.get("frame_idx"), e)
          results.append({"frame_idx": ann.get("frame_idx"), "success": False, "error": str(e)})
          error_count += 1

      return jsonify({"saved": saved_count, "errors": error_count, "results": results})

    except Exception as e:
      err = ErrorResponse(code="batch_save_error",
                          message="Failed to save batch annotations",
                          details={"error": str(e)})
      return jsonify(err.model_dump()), 500

  @bp.route("/api/gpu/status", methods=["GET"])
  def api_gpu_status():
    """Check GPU availability and configuration."""
    try:
      import torch
      gpu_info = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "current_device": torch.cuda.current_device() if torch.cuda.is_available() else None,
      }

      if gpu_info["cuda_available"]:
        devices = []
        for i in range(gpu_info["device_count"]):
          props = torch.cuda.get_device_properties(i)
          devices.append({
            "id": i,
            "name": props.name,
            "total_memory_gb": props.total_memory / (1024**3),
            "major": props.major,
            "minor": props.minor,
          })
        gpu_info["devices"] = devices
        gpu_info["cuda_version"] = torch.version.cuda
        gpu_info["cudnn_version"] = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None

      return jsonify({
        "gpu_available": gpu_info["cuda_available"],
        "gpu_count": gpu_info["device_count"],
        "gpu_info": gpu_info,
        "torch_version": torch.__version__,
      })
    except Exception as e:
      return jsonify({"gpu_available": False, "error": str(e), "torch_version": "unknown"}), 500

  # -------------------- Autolabel (inference) --------------------
  @bp.route("/api/annotations/autolabel", methods=["POST"])
  def api_autolabel_frame():
    """Run model inference for current frame and return predicted class mapping.

    Payload:
      { session_id, dataset_id, frame_idx, confidence_threshold? }

    Response:
      { prediction: { index, category_name, confidence, meets_threshold }, mapped: { class_id?, category_name }, annotation_saved?: bool }
    """
    try:
      payload = request.get_json(force=True) or {}
      session_id = str(payload.get("session_id") or "").strip()
      dataset_id = int(payload.get("dataset_id") or 0)
      frame_idx = int(payload.get("frame_idx") if payload.get("frame_idx") is not None else -1)
      confidence_threshold = float(payload.get("confidence_threshold") or 0.9)
      if not session_id or dataset_id <= 0 or frame_idx < 0:
        err = ErrorResponse(code="bad_request", message="session_id, dataset_id and frame_idx are required")
        return jsonify(err.model_dump()), 400
    except Exception as e:
      err = ErrorResponse(code="bad_request", message="Invalid payload", details={"error": str(e)})
      return jsonify(err.model_dump()), 400

    try:
      # Resolve frame path and dataset classes
      frame = session_service.get_frame_by_idx(session_id, frame_idx)
      abs_path, _ = session_service.get_frame_for_image(session_id, frame_idx)
      # Get dataset details and classes (ordered)
      conn = get_connection()
      d = db_get_dataset(conn, int(dataset_id))
      if not d:
        err = ErrorResponse(code="not_found", message="Dataset not found")
        return jsonify(err.model_dump()), 404
      rows = db_list_dataset_classes(conn, int(dataset_id))
      class_names = [str(r.get("name")) for r in rows]
      # Fallback: if no classes stored yet, infer from any existing annotations
      if not class_names:
        class_names = []
      target_type_name = d.get("target_type_name") or "SingleLabelClassification"

      pred = autolabel_service.predict_single(Path(abs_path), target_type_name, class_names, confidence_threshold)

      # Map to class_id, creating class if it doesn't exist
      mapped: Dict[str, Any] = {"category_name": pred.get("category_name")}
      class_id = None
      try:
        name_to_id = {str(r.get("name")): int(r.get("id")) for r in rows}
        class_id = name_to_id.get(str(pred.get("category_name")))

        # If class doesn't exist, create it on-the-fly
        if class_id is None:
          predicted_name = str(pred.get("category_name"))
          if predicted_name and predicted_name.strip():
            try:
              new_cls = db_get_or_create_dataset_class(conn, int(dataset_id), predicted_name)
              conn.commit()
              class_id = int(new_cls["id"])
              log.info("Created new class '%s' (id=%s) for dataset %s during autolabel", predicted_name, class_id,
                       dataset_id)
            except Exception as e:
              log.warning("Failed to create new class '%s' during autolabel: %s", predicted_name, e)
              # Continue without creating class - prediction will still be returned

        if class_id is not None:
          mapped["class_id"] = int(class_id)
      except Exception:
        pass

      return jsonify({
        "prediction": pred,
        "mapped": mapped,
      })
    except AutoLabelUnavailable as e:
      # 501 Not Implemented indicates optional feature missing
      err = ErrorResponse(code="autolabel_unavailable", message=str(e))
      return jsonify(err.model_dump()), 501
    except FileNotFoundError as e:
      err = ErrorResponse(code="not_found", message=str(e))
      return jsonify(err.model_dump()), 404
    except Exception as e:
      err = ErrorResponse(code="autolabel_error", message="Failed to autolabel frame", details={"error": str(e)})
      return jsonify(err.model_dump()), 500

  @bp.route("/api/annotations/autolabel_session", methods=["POST"])
  def api_autolabel_session():
    """Autolabel an entire session with progress updates and batch processing.

    Payload:
      { session_id, dataset_id, confidence_threshold?=0.9, dry_run?=false, limit?, batch_size?=10, progress_callback?=false }

    Returns summary with counts and per-class tallies. If progress_callback=true, returns progress updates.
    """
    try:
      payload = request.get_json(force=True) or {}
      session_id = str(payload.get("session_id") or "").strip()
      dataset_id = int(payload.get("dataset_id") or 0)
      confidence_threshold = float(payload.get("confidence_threshold") or 0.9)
      dry_run = bool(payload.get("dry_run") or False)
      limit = payload.get("limit")
      batch_size = int(payload.get("batch_size") or 10)
      progress_callback = bool(payload.get("progress_callback") or False)
      n_limit = int(limit) if (limit is not None and str(limit).strip() != "") else None

      if not session_id or dataset_id <= 0:
        err = ErrorResponse(code="bad_request", message="session_id and dataset_id are required")
        return jsonify(err.model_dump()), 400

      if batch_size > 50:  # Prevent excessive memory usage
        batch_size = 50

    except Exception as e:
      err = ErrorResponse(code="bad_request", message="Invalid payload", details={"error": str(e)})
      return jsonify(err.model_dump()), 400

    try:
      # Fetch dataset and classes
      conn = get_connection()
      d = db_get_dataset(conn, int(dataset_id))
      if not d:
        err = ErrorResponse(code="not_found", message="Dataset not found")
        return jsonify(err.model_dump()), 404
      rows = db_list_dataset_classes(conn, int(dataset_id))
      class_names = [str(r.get("name")) for r in rows]
      name_to_id = {str(r.get("name")): int(r.get("id")) for r in rows}
      target_type_name = d.get("target_type_name") or "SingleLabelClassification"

      # Count total frames first
      total_frames = 0
      try:
        idx = 0
        while True:
          try:
            session_service.get_frame_for_image(session_id, idx)
            total_frames += 1
            idx += 1
            if n_limit is not None and total_frames >= n_limit:
              break
          except IndexError:
            break
          except Exception:
            idx += 1
      except Exception:
        total_frames = 0  # Fallback if we can't determine total

      # Initialize progress tracking
      processed = 0
      saved = 0
      errors = 0
      per_class: Dict[str, int] = {}
      batch_annotations = []

      # Process frames in batches
      idx = 0
      while True:
        if n_limit is not None and processed >= n_limit:
          break

        try:
          abs_path, _frame = session_service.get_frame_for_image(session_id, idx)
        except IndexError:
          break
        except Exception:
          errors += 1
          idx += 1
          continue

        try:
          pred = autolabel_service.predict_single(Path(abs_path), target_type_name, class_names, confidence_threshold)
          processed += 1

          if pred.get("meets_threshold"):
            cat = str(pred.get("category_name"))
            per_class[cat] = per_class.get(cat, 0) + 1

            if not dry_run:
              # Get or create class
              cls_id = name_to_id.get(cat)
              if cls_id is None:
                try:
                  new_cls = db_get_or_create_dataset_class(conn, int(dataset_id), cat)
                  conn.commit()
                  cls_id = int(new_cls["id"])
                  name_to_id[cat] = cls_id
                  class_names.append(cat)  # Update class names for future predictions
                  log.info("Created new class '%s' (id=%s) during session autolabel", cat, cls_id)
                except Exception as e:
                  log.warning("Failed to create new class '%s' during session autolabel: %s", cat, e)
                  errors += 1
                  idx += 1
                  continue

              # Add to batch for later saving
              frame = session_service.get_frame_by_idx(session_id, idx)
              batch_annotations.append({"frame_idx": idx, "category_name": cat, "class_id": cls_id})

              # Save batch if it reaches the batch size
              if len(batch_annotations) >= batch_size:
                try:
                  # Use the batch endpoint directly
                  batch_payload = {"session_id": session_id, "dataset_id": dataset_id, "annotations": batch_annotations}
                  # Call batch save function directly
                  results = []
                  batch_saved = 0
                  batch_errors = 0

                  for ann in batch_annotations:
                    try:
                      frame = session_service.get_frame_by_idx(session_id, ann["frame_idx"])
                      annotation_service.save_single_label(session_id=session_id,
                                                           dataset_id=dataset_id,
                                                           frame_id=str(frame["frame_id"]),
                                                           class_id=int(ann["class_id"]),
                                                           override_settings=None)
                      results.append({"frame_idx": ann["frame_idx"], "success": True})
                      batch_saved += 1
                    except Exception as e:
                      log.warning("Failed to save annotation for frame_idx %s: %s", ann["frame_idx"], e)
                      results.append({"frame_idx": ann["frame_idx"], "success": False, "error": str(e)})
                      batch_errors += 1

                  saved += batch_saved
                  errors += batch_errors
                  batch_annotations = []
                except Exception as e:
                  log.error("Batch save failed: %s", e)
                  errors += len(batch_annotations)
                  batch_annotations = []

                # Send progress update if requested
                if progress_callback and hasattr(request, 'is_json') and request.is_json:
                  progress_data = {
                    "progress": {
                      "processed": processed,
                      "total": total_frames,
                      "saved": saved,
                      "errors": errors,
                      "percentage": (processed / total_frames) * 100 if total_frames > 0 else 0
                    },
                    "per_class": per_class,
                    "completed": False
                  }
                  # Note: In a real streaming implementation, you'd use Server-Sent Events or WebSockets
                  # For now, we'll just log the progress
                  log.info("Autolabel progress: %s/%s (%.1f%%)", processed, total_frames,
                           progress_data["progress"]["percentage"])

        except AutoLabelUnavailable as e:
          err = ErrorResponse(code="autolabel_unavailable", message=str(e))
          return jsonify(err.model_dump()), 501
        except Exception as e:
          log.warning("Error processing frame %s: %s", idx, e)
          errors += 1
        finally:
          idx += 1

      # Save remaining batch
      if len(batch_annotations) > 0 and not dry_run:
        try:
          for ann in batch_annotations:
            try:
              frame = session_service.get_frame_by_idx(session_id, ann["frame_idx"])
              annotation_service.save_single_label(session_id=session_id,
                                                   dataset_id=dataset_id,
                                                   frame_id=str(frame["frame_id"]),
                                                   class_id=int(ann["class_id"]),
                                                   override_settings=None)
              saved += 1
            except Exception as e:
              log.warning("Failed to save final annotation for frame_idx %s: %s", ann["frame_idx"], e)
              errors += 1
        except Exception as e:
          log.error("Final batch save failed: %s", e)
          errors += len(batch_annotations)

      # Send final progress update
      if progress_callback:
        log.info("Autolabel completed: %s processed, %s saved, %s errors", processed, saved, errors)

      return jsonify({
        "processed":
        int(processed),
        "saved":
        int(saved),
        "errors":
        int(errors),
        "total_frames":
        int(total_frames),
        "per_class":
        per_class,
        "dry_run":
        bool(dry_run),
        "classes_created":
        len([k for k in name_to_id.keys() if k not in [str(r.get("name")) for r in rows]])
      })

    except Exception as e:
      err = ErrorResponse(code="autolabel_error", message="Failed to autolabel session", details={"error": str(e)})
      return jsonify(err.model_dump()), 500

  # -------------------- Dataset-session settings --------------------
  @bp.route("/api/datasets/<int:dataset_id>/sessions/<session_id>/settings", methods=["PUT"])
  def api_upsert_dataset_session_settings(dataset_id: int, session_id: str):
    try:
      payload = request.get_json(force=True) or {}
      settings = payload.get("settings")
      if not isinstance(settings, dict):
        err = ErrorResponse(code="bad_request", message="settings (object) is required")
        return jsonify(err.model_dump()), 400
      settings_service.upsert_dataset_session_settings(dataset_id, session_id, settings)
      return jsonify({"ok": True})
    except Exception as e:
      err = ErrorResponse(code="settings_error", message="Failed to upsert settings", details={"error": str(e)})
      return jsonify(err.model_dump()), 500

  @bp.route("/api/datasets/<int:dataset_id>/sessions/<session_id>/settings", methods=["GET"])
  def api_get_dataset_session_settings(dataset_id: int, session_id: str):
    try:
      s = settings_service.get_dataset_session_settings(dataset_id, session_id)
      return jsonify({"settings": s})
    except Exception as e:
      err = ErrorResponse(code="settings_error", message="Failed to get settings", details={"error": str(e)})
      return jsonify(err.model_dump()), 500

  @bp.route("/api/datasets/<int:dataset_id>/sessions/<session_id>/settings", methods=["DELETE"])
  def api_delete_dataset_session_settings(dataset_id: int, session_id: str):
    try:
      settings_service.clear_dataset_session_settings(dataset_id, session_id)
      return jsonify({"ok": True})
    except Exception as e:
      err = ErrorResponse(code="settings_error", message="Failed to delete settings", details={"error": str(e)})
      return jsonify(err.model_dump()), 500

  # -------------------- Dataset Download Endpoint --------------------
  @bp.route("/api/datasets/<int:dataset_id>/download", methods=["GET"])
  def api_download_dataset(dataset_id: int):
    """
    Download a complete labeled dataset in various formats.
    
    Query parameters:
      - format: Export format ('json', 'csv', 'coco') - default: 'json'
      - include_images: Include image paths/data ('true'/'false') - default: 'false'  
      - sessions: Comma-separated list of session IDs to include (optional)
      - status: Comma-separated list of statuses to include - default: 'labeled'
      - limit: Maximum number of records to export (optional)
      - offset: Number of records to skip for pagination (optional)
      - download: Return as file attachment ('true'/'false') - default: 'false'
    """
    try:
      # Parse query parameters
      format_type = request.args.get("format", "json").lower()
      if format_type not in ["json", "csv", "coco", "jsonl"]:
        err = ErrorResponse(code="bad_request", message="Invalid format. Must be 'json', 'csv', or 'coco'")
        return jsonify(err.model_dump()), 400

      # Backward-compatible lightweight CSV/JSONL paths expected by tests
      if format_type in ("csv", "jsonl"):
        try:
          # Ensure dataset exists
          conn = get_connection()
          d = db_get_dataset(conn, int(dataset_id))
          if not d:
            err = ErrorResponse(code="not_found", message="Dataset not found")
            return jsonify(err.model_dump()), 404
        finally:
          try:
            conn.close()
          except Exception:
            pass

        rows = dataset_service.fetch_labeled(int(dataset_id))

        base_name = (str(request.args.get("filename") or "") or f"dataset_{int(dataset_id)}_labeled").strip()

        if format_type == "csv":
          import io
          import csv as _csv

          fieldnames = [
            "session_id",
            "frame_id",
            "value_real",
            "single_label_class_id",
            "multilabel_class_ids_csv",
            "frame_path_rel",
          ]
          sio = io.StringIO()
          writer = _csv.DictWriter(sio, fieldnames=fieldnames)
          writer.writeheader()
          for r in rows:
            writer.writerow({
              "session_id": r.get("session_id"),
              "frame_id": r.get("frame_id"),
              "value_real": r.get("value_real"),
              "single_label_class_id": r.get("single_label_class_id"),
              "multilabel_class_ids_csv": r.get("multilabel_class_ids_csv"),
              "frame_path_rel": r.get("frame_path_rel"),
            })
          data = sio.getvalue()
          from flask import Response as _Resp
          resp = _Resp(data, mimetype="text/csv; charset=utf-8")
          filename = f"{base_name}.csv"
          try:
            resp.headers["Content-Disposition"] = f"attachment; filename=\"{filename}\""
          except Exception:
            pass
          return resp
        else:
          # jsonl
          import io
          import json as _json
          sio = io.StringIO()
          for r in rows:
            sio.write(_json.dumps(r, ensure_ascii=False))
            sio.write("\n")
          data = sio.getvalue()
          from flask import Response as _Resp
          resp = _Resp(data, mimetype="application/x-ndjson; charset=utf-8")
          filename = f"{base_name}.jsonl"
          try:
            resp.headers["Content-Disposition"] = f"attachment; filename=\"{filename}\""
          except Exception:
            pass
          return resp

      # Advanced export flow (JSON/CSV/COCO via DownloadService)
      include_images = request.args.get("include_images", "false").lower() in ("true", "1", "yes")
      download_as_file = request.args.get("download", "false").lower() in ("true", "1", "yes")

      # Parse session filter
      session_filter = None
      sessions_param = request.args.get("sessions", "").strip()
      if sessions_param:
        session_filter = [s.strip() for s in sessions_param.split(",") if s.strip()]

      # Parse status filter
      status_filter = None
      status_param = request.args.get("status", "labeled").strip()
      if status_param:
        status_filter = [s.strip() for s in status_param.split(",") if s.strip()]

      # Parse pagination
      limit = None
      offset = None
      try:
        if request.args.get("limit"):
          limit = int(request.args.get("limit"))
          if limit <= 0:
            raise ValueError("Limit must be positive")
      except ValueError:
        err = ErrorResponse(code="bad_request", message="Invalid limit parameter")
        return jsonify(err.model_dump()), 400

      try:
        if request.args.get("offset"):
          offset = int(request.args.get("offset"))
          if offset < 0:
            raise ValueError("Offset must be non-negative")
      except ValueError:
        err = ErrorResponse(code="bad_request", message="Invalid offset parameter")
        return jsonify(err.model_dump()), 400

      # Pre-check dataset existence to return 404 consistently
      try:
        _conn_chk = get_connection()
        if not db_get_dataset(_conn_chk, int(dataset_id)):
          err = ErrorResponse(code="not_found", message="Dataset not found")
          return jsonify(err.model_dump()), 404
      finally:
        try:
          _conn_chk.close()
        except Exception:
          pass

      export_result = download_service.export_dataset(
        dataset_id=dataset_id,
        format_type=format_type,
        include_images=include_images,
        session_filter=session_filter,
        status_filter=status_filter,
        limit=limit,
        offset=offset,
      )

      if download_as_file:
        from flask import Response

        dataset_name = export_result["metadata"]["dataset"]["name"]
        safe_name = "".join(c for c in dataset_name if c.isalnum() or c in (' ', '-', '_')).strip()
        safe_name = safe_name.replace(' ', '_')
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')

        if format_type == "csv":
          filename = f"{safe_name}_dataset_{timestamp}.csv"
          content = export_result["data"]
          mimetype = "text/csv"
        elif format_type == "coco":
          filename = f"{safe_name}_dataset_{timestamp}_coco.json"
          content = json.dumps(export_result["data"], indent=2, ensure_ascii=False)
          mimetype = "application/json"
        else:  # json
          filename = f"{safe_name}_dataset_{timestamp}.json"
          content = json.dumps(export_result, indent=2, ensure_ascii=False)
          mimetype = "application/json"

        return Response(content,
                        mimetype=mimetype,
                        headers={
                          "Content-Disposition": f"attachment; filename={filename}",
                          "Content-Type": mimetype,
                        })

      return jsonify(export_result)

    except ValueError as e:
      err = ErrorResponse(code="validation_error", message=str(e))
      return jsonify(err.model_dump()), 400
    except Exception as e:
      log.exception("dataset_download_error")
      err = ErrorResponse(code="download_error", message="Failed to download dataset", details={"error": str(e)})
      return jsonify(err.model_dump()), 500

  return bp
