from __future__ import annotations

from flask import Blueprint, request, jsonify, send_file
from pathlib import Path
from typing import Optional

from core.session_manager import SessionManager
from services.session_service import SessionService
from services.annotation_service import AnnotationService
from services.settings_service import SettingsService
from services.dataset_service import DatasetService
from core.path_resolver import resolve_frame_absolute_path
from dto import (
    FrameQuery,
    ImageQuery,
    ErrorResponse,
    SaveRegressionRequest,
    SaveSingleLabelRequest,
    SaveMultilabelRequest,
)
from db.connection import get_connection
from db.schema import init_db
from db.indexer import reindex_sessions
from db.projects import (
    list_projects as db_list_projects,
    create_project as db_create_project,
    list_datasets as db_list_datasets,
    create_dataset as db_create_dataset,
    get_dataset as db_get_dataset,
    dataset_progress as db_dataset_progress,
    get_dataset_by_name as db_get_dataset_by_name,
)
import sqlite3
from db.datasets import enroll_session_frames as db_enroll_session_frames, list_labeled as db_list_labeled


def create_annotation_api(session_manager: SessionManager) -> Blueprint:
    bp = Blueprint('annotation_api', __name__)

    session_service = SessionService(session_manager)
    annotation_service = AnnotationService(session_manager)
    dataset_service = DatasetService(session_manager)
    settings_service = SettingsService()

    @bp.route('/api/sessions', methods=['GET'])
    def discover_sessions():
        try:
            sessions = session_manager.discover_sessions()
            return jsonify(sessions)
        except Exception as e:
            err = ErrorResponse(code='sessions_error', message='Failed to discover sessions', details={'error': str(e)})
            return jsonify(err.dict()), 500

    @bp.route('/api/reindex', methods=['POST'])
    def reindex():
        """Scan data directories and populate SQLite with sessions/frames."""
        try:
            conn = get_connection()
            init_db(conn)
            summary = reindex_sessions(conn, Path(session_manager.data_root))
            print(f"[api][reindex] {summary}", flush=True)
            return jsonify({
                'ok': True,
                'summary': summary,
            })
        except Exception as e:
            err = ErrorResponse(code='reindex_error', message='Failed to reindex', details={'error': str(e)})
            return jsonify(err.dict()), 500

    # -------------------- Projects & Datasets CRUD --------------------
    @bp.route('/api/projects', methods=['GET'])
    def api_list_projects():
        try:
            conn = get_connection()
            init_db(conn)
            return jsonify(db_list_projects(conn))
        except Exception as e:
            err = ErrorResponse(code='projects_error', message='Failed to list projects', details={'error': str(e)})
            return jsonify(err.dict()), 500

    @bp.route('/api/projects', methods=['POST'])
    def api_create_project():
        try:
            payload = request.get_json(force=True) or {}
            name = payload.get('name')
            description = payload.get('description')
            if not name:
                err = ErrorResponse(code='bad_request', message='name is required')
                return jsonify(err.dict()), 400
            conn = get_connection()
            init_db(conn)
            try:
                pid = db_create_project(conn, name, description)
                conn.commit()
                return jsonify({'id': pid, 'name': name, 'description': description}), 201
            except sqlite3.IntegrityError:
                # Unique constraint on projects.name; return existing
                # Fetch existing row id
                cur = conn.execute("SELECT id, name, description, created_at FROM projects WHERE name = ?", (name,))
                row = cur.fetchone()
                if row:
                    return jsonify({'id': int(row['id']), 'name': row['name'], 'description': row['description']}), 200
                raise
        except Exception as e:
            err = ErrorResponse(code='create_project_error', message='Failed to create project', details={'error': str(e)})
            return jsonify(err.dict()), 500

    @bp.route('/api/projects/<int:project_id>/datasets', methods=['GET'])
    def api_list_datasets(project_id: int):
        try:
            conn = get_connection()
            init_db(conn)
            return jsonify(db_list_datasets(conn, project_id))
        except Exception as e:
            err = ErrorResponse(code='datasets_error', message='Failed to list datasets', details={'error': str(e)})
            return jsonify(err.dict()), 500

    @bp.route('/api/projects/<int:project_id>/datasets', methods=['POST'])
    def api_create_dataset(project_id: int):
        try:
            payload = request.get_json(force=True) or {}
            name = payload.get('name')
            description = payload.get('description')
            target_type_id = payload.get('target_type_id')
            if not name or target_type_id is None:
                err = ErrorResponse(code='bad_request', message='name and target_type_id are required')
                return jsonify(err.dict()), 400
            conn = get_connection()
            init_db(conn)
            # Ensure the parent project exists; otherwise, the insert will raise a FK error
            proj_row = conn.execute("SELECT id FROM projects WHERE id = ?", (project_id,)).fetchone()
            if not proj_row:
                err = ErrorResponse(code='not_found', message='Project not found', details={'project_id': project_id})
                return jsonify(err.dict()), 404
            try:
                ttid = int(target_type_id)
                if ttid not in (0, 1, 2):
                    err = ErrorResponse(code='bad_request', message='target_type_id must be 0, 1, or 2')
                    return jsonify(err.dict()), 400
                did = db_create_dataset(conn, project_id, name, description, ttid)
                conn.commit()
                return jsonify({'id': did, 'project_id': project_id, 'name': name, 'description': description, 'target_type_id': ttid}), 201
            except sqlite3.IntegrityError:
                # Unique(project_id, name) exists: return existing entry as 200
                existing = db_get_dataset_by_name(conn, project_id, name)
                if existing:
                    return jsonify(existing), 200
                raise
        except Exception as e:
            err = ErrorResponse(code='create_dataset_error', message='Failed to create dataset', details={'error': str(e)})
            return jsonify(err.dict()), 500

    # Enroll session frames into dataset (creates annotations rows with status=unlabeled)
    @bp.route('/api/datasets/<int:dataset_id>/enroll_session', methods=['POST'])
    def api_enroll_session(dataset_id: int):
        try:
            payload = request.get_json(force=True) or {}
            session_id = payload.get('session_id')
            settings = payload.get('settings')  # optional default settings_json
            if not session_id:
                err = ErrorResponse(code='bad_request', message='session_id is required')
                return jsonify(err.dict()), 400
            conn = get_connection()
            init_db(conn)
            # Ensure dataset exists
            d = db_get_dataset(conn, dataset_id)
            if not d:
                err = ErrorResponse(code='not_found', message='Dataset not found')
                return jsonify(err.dict()), 404
            summary = db_enroll_session_frames(conn, dataset_id, session_id, settings)
            return jsonify({'ok': True, 'summary': summary})
        except ValueError as ve:
            err = ErrorResponse(code='bad_request', message=str(ve))
            return jsonify(err.dict()), 400
        except Exception as e:
            err = ErrorResponse(code='enroll_error', message='Failed to enroll session', details={'error': str(e)})
            return jsonify(err.dict()), 500

    @bp.route('/api/datasets/<int:dataset_id>/progress', methods=['GET'])
    def api_dataset_progress(dataset_id: int):
        try:
            conn = get_connection()
            init_db(conn)
            return jsonify(db_dataset_progress(conn, dataset_id))
        except Exception as e:
            err = ErrorResponse(code='progress_error', message='Failed to get dataset progress', details={'error': str(e)})
            return jsonify(err.dict()), 500

    @bp.route('/api/datasets/<int:dataset_id>/labeled', methods=['GET'])
    def api_dataset_labeled(dataset_id: int):
        try:
            # Use service to enrich with frame_path_rel based on metadata and resolvers
            rows = dataset_service.fetch_labeled(dataset_id)
            return jsonify(rows)
        except Exception as e:
            err = ErrorResponse(code='labeled_error', message='Failed to list labeled items', details={'error': str(e)})
            return jsonify(err.dict()), 500

    @bp.route('/api/frame', methods=['GET'])
    def get_frame():
        try:
            q = FrameQuery(session_id=request.args.get('session_id', ''),
                           project_name=request.args.get('project_name', 'default'),
                           idx=int(request.args.get('idx', '-1')))
        except Exception as e:
            err = ErrorResponse(code='bad_request', message='Invalid query parameters', details={'error': str(e)})
            return jsonify(err.dict()), 400

        try:
            # Debug: log incoming params and DB frame count
            try:
                print(f"[api][frame] session_id={q.session_id} idx={q.idx}", flush=True)
                sid_db = session_manager.get_session_db_id(q.session_id)
                if sid_db is not None:
                    conn = get_connection()
                    init_db(conn)
                    total = conn.execute("SELECT COUNT(1) FROM frames WHERE session_id = ?", (sid_db,)).fetchone()[0]
                    print(f"[api][frame] db_count={total}", flush=True)
                else:
                    print("[api][frame] session not in DB", flush=True)
            except Exception:
                pass
            frame = session_service.get_frame_by_idx(q.session_id, q.idx)
            # Deprecated filesystem annotation removed; return only frame
            return jsonify({'frame': frame})
        except IndexError as e:
            err = ErrorResponse(code='not_found', message=str(e))
            return jsonify(err.dict()), 404
        except FileNotFoundError as e:
            err = ErrorResponse(code='not_found', message=str(e))
            return jsonify(err.dict()), 404
        except Exception as e:
            err = ErrorResponse(code='frame_error', message='Failed to fetch frame', details={'error': str(e)})
            return jsonify(err.dict()), 500

    @bp.route('/api/image', methods=['GET'])
    def get_image():
        try:
            q = ImageQuery(session_id=request.args.get('session_id', ''),
                           idx=int(request.args.get('idx', '-1')))
        except Exception as e:
            err = ErrorResponse(code='bad_request', message='Invalid query parameters', details={'error': str(e)})
            return jsonify(err.dict()), 400

        try:
            abs_path, _frame = session_service.get_frame_for_image(q.session_id, q.idx)
            return send_file(abs_path)
        except FileNotFoundError as e:
            err = ErrorResponse(code='not_found', message=str(e))
            return jsonify(err.dict()), 404
        except Exception as e:
            err = ErrorResponse(code='image_error', message='Failed to load image', details={'error': str(e)})
            return jsonify(err.dict()), 500

    # Legacy filesystem save endpoint removed in favor of DB-backed endpoints above

    # -------------------- DB-backed annotation write endpoints --------------------
    @bp.route('/api/annotations/regression', methods=['POST'])
    def api_save_regression():
        try:
            payload = SaveRegressionRequest.parse_obj(request.get_json())
        except Exception as e:
            err = ErrorResponse(code='bad_request', message='Invalid payload', details={'error': str(e)})
            return jsonify(err.dict()), 400
        try:
            frame_id = payload.frame_id
            if frame_id is None and payload.frame_idx is not None:
                frame = session_service.get_frame_by_idx(payload.session_id, int(payload.frame_idx))
                frame_id = str(frame['frame_id'])
            res = annotation_service.save_regression(
                session_id=payload.session_id,
                dataset_id=payload.dataset_id,
                frame_id=str(frame_id),
                value=payload.value,
                override_settings=payload.override_settings,
            )
            return jsonify(res)
        except Exception as e:
            err = ErrorResponse(code='save_error', message='Failed to save regression', details={'error': str(e)})
            return jsonify(err.dict()), 500

    @bp.route('/api/annotations/single_label', methods=['POST'])
    def api_save_single_label():
        try:
            payload = SaveSingleLabelRequest.parse_obj(request.get_json())
        except Exception as e:
            err = ErrorResponse(code='bad_request', message='Invalid payload', details={'error': str(e)})
            return jsonify(err.dict()), 400
        try:
            frame_id = payload.frame_id
            if frame_id is None and payload.frame_idx is not None:
                frame = session_service.get_frame_by_idx(payload.session_id, int(payload.frame_idx))
                frame_id = str(frame['frame_id'])
            res = annotation_service.save_single_label(
                session_id=payload.session_id,
                dataset_id=payload.dataset_id,
                frame_id=str(frame_id),
                class_id=payload.class_id,
                override_settings=payload.override_settings,
            )
            return jsonify(res)
        except Exception as e:
            err = ErrorResponse(code='save_error', message='Failed to save single label', details={'error': str(e)})
            return jsonify(err.dict()), 500

    @bp.route('/api/annotations/multilabel', methods=['POST'])
    def api_save_multilabel():
        try:
            payload = SaveMultilabelRequest.parse_obj(request.get_json())
        except Exception as e:
            err = ErrorResponse(code='bad_request', message='Invalid payload', details={'error': str(e)})
            return jsonify(err.dict()), 400
        try:
            frame_id = payload.frame_id
            if frame_id is None and payload.frame_idx is not None:
                frame = session_service.get_frame_by_idx(payload.session_id, int(payload.frame_idx))
                frame_id = str(frame['frame_id'])
            res = annotation_service.save_multilabel(
                session_id=payload.session_id,
                dataset_id=payload.dataset_id,
                frame_id=str(frame_id),
                class_ids=payload.class_ids,
                override_settings=payload.override_settings,
            )
            return jsonify(res)
        except Exception as e:
            err = ErrorResponse(code='save_error', message='Failed to save multilabel', details={'error': str(e)})
            return jsonify(err.dict()), 500

    # -------------------- Dataset-session settings --------------------
    @bp.route('/api/datasets/<int:dataset_id>/sessions/<session_id>/settings', methods=['PUT'])
    def api_upsert_dataset_session_settings(dataset_id: int, session_id: str):
        try:
            payload = request.get_json(force=True) or {}
            settings = payload.get('settings')
            if not isinstance(settings, dict):
                err = ErrorResponse(code='bad_request', message='settings (object) is required')
                return jsonify(err.dict()), 400
            settings_service.upsert_dataset_session_settings(dataset_id, session_id, settings)
            return jsonify({'ok': True})
        except Exception as e:
            err = ErrorResponse(code='settings_error', message='Failed to upsert settings', details={'error': str(e)})
            return jsonify(err.dict()), 500

    @bp.route('/api/datasets/<int:dataset_id>/sessions/<session_id>/settings', methods=['GET'])
    def api_get_dataset_session_settings(dataset_id: int, session_id: str):
        try:
            s = settings_service.get_dataset_session_settings(dataset_id, session_id)
            return jsonify({'settings': s})
        except Exception as e:
            err = ErrorResponse(code='settings_error', message='Failed to get settings', details={'error': str(e)})
            return jsonify(err.dict()), 500

    @bp.route('/api/datasets/<int:dataset_id>/sessions/<session_id>/settings', methods=['DELETE'])
    def api_delete_dataset_session_settings(dataset_id: int, session_id: str):
        try:
            settings_service.clear_dataset_session_settings(dataset_id, session_id)
            return jsonify({'ok': True})
        except Exception as e:
            err = ErrorResponse(code='settings_error', message='Failed to delete settings', details={'error': str(e)})
            return jsonify(err.dict()), 500

    return bp
