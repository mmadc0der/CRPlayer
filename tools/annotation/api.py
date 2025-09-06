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
from db.connection import get_db_path
from db.schema import init_db
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
from db.datasets import enroll_session_frames as db_enroll_session_frames, list_labeled as db_list_labeled
from db.classes import (
    list_dataset_classes as db_list_dataset_classes,
    sync_dataset_classes as db_sync_dataset_classes,
    get_or_create_dataset_class as db_get_or_create_dataset_class,
)


def create_annotation_api(session_manager: SessionManager, name: str = 'annotation_api') -> Blueprint:
    bp = Blueprint(name, __name__)

    session_service = SessionService(session_manager)
    annotation_service = AnnotationService(session_manager)
    dataset_service = DatasetService(session_manager)
    settings_service = SettingsService()

    @bp.route('/api/target_types', methods=['GET'])
    def api_list_target_types():
        try:
            conn = get_connection()
            rows = conn.execute("SELECT id, name FROM target_types ORDER BY id").fetchall()
            return jsonify([{'id': int(r['id']), 'name': r['name']} for r in rows])
        except Exception as e:
            err = ErrorResponse(code='target_types_error', message='Failed to list target types', details={'error': str(e)})
            return jsonify(err.model_dump()), 500

    # Removed duplicate api_list_projects function - keeping the one at line 187

    # -------------------- Read annotation for a frame --------------------
    @bp.route('/api/annotations/frame', methods=['GET'])
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
            session_id = request.args.get('session_id', '')
            dataset_id = int(request.args.get('dataset_id', '0'))
            idx = int(request.args.get('idx', '-1'))
            if not session_id or dataset_id <= 0 or idx < 0:
                err = ErrorResponse(code='bad_request', message='session_id, dataset_id and idx are required')
                return jsonify(err.model_dump()), 400
        except Exception as e:
            err = ErrorResponse(code='bad_request', message='Invalid query parameters', details={'error': str(e)})
            return jsonify(err.model_dump()), 400

        try:
            # Resolve frame_id from idx
            frame = session_service.get_frame_by_idx(session_id, idx)
            frame_id = str(frame['frame_id'])
            ann = annotation_service.get_annotation_db(session_id, dataset_id, frame_id)
            return jsonify({'annotation': ann})
        except Exception as e:
            err = ErrorResponse(code='annotation_error', message='Failed to fetch annotation', details={'error': str(e)})
            return jsonify(err.model_dump()), 500

    @bp.route('/api/sessions', methods=['GET'])
    def discover_sessions():
        try:
            sessions = session_manager.discover_sessions()
            return jsonify(sessions)
        except Exception as e:
            err = ErrorResponse(code='sessions_error', message='Failed to discover sessions', details={'error': str(e)})
            return jsonify(err.model_dump()), 500

    @bp.route('/api/projects/<int:project_id>', methods=['PUT'])
    def api_update_project(project_id: int):
        try:
            payload = request.get_json(force=True) or {}
            name = payload.get('name')
            description = payload.get('description')
            conn = get_connection()
            # ensure exists
            exists = conn.execute("SELECT 1 FROM projects WHERE id = ?", (project_id,)).fetchone()
            if not exists:
                err = ErrorResponse(code='not_found', message='Project not found')
                return jsonify(err.model_dump()), 404
            db_update_project(conn, project_id, name, description)
            conn.commit()
            row = conn.execute("SELECT id, name, description, created_at FROM projects WHERE id = ?", (project_id,)).fetchone()
            return jsonify(dict(row))
        except sqlite3.IntegrityError as e:
            err = ErrorResponse(code='conflict', message='Project name already exists', details={'error': str(e)})
            return jsonify(err.model_dump()), 409
        except Exception as e:
            err = ErrorResponse(code='update_project_error', message='Failed to update project', details={'error': str(e)})
            return jsonify(err.model_dump()), 500

    @bp.route('/api/projects/<int:project_id>', methods=['DELETE'])
    def api_delete_project(project_id: int):
        try:
            conn = get_connection()
            # Optional force delete: when true, rely on FK ON DELETE CASCADE from datasets -> projects
            force = str(request.args.get('force', '0')).lower() in ('1', 'true', 'yes')
            if not force:
                # Enforce no datasets under project before delete
                cnt = conn.execute("SELECT COUNT(1) FROM datasets WHERE project_id = ?", (project_id,)).fetchone()[0]
                if cnt:
                    err = ErrorResponse(code='conflict', message='Project has datasets; delete them first')
                    return jsonify(err.model_dump()), 409
            n = db_delete_project(conn, project_id)
            conn.commit()
            return jsonify({'deleted': int(n)})
        except Exception as e:
            err = ErrorResponse(code='delete_project_error', message='Failed to delete project', details={'error': str(e)})
            return jsonify(err.model_dump()), 500

    @bp.route('/api/reindex', methods=['POST'])
    def reindex():
        """Scan data directories and populate SQLite with sessions/frames."""
        try:
            conn = get_connection()
            try:
                print(f"[api][reindex] db_path={get_db_path()}", flush=True)
            except Exception:
                pass
            summary = reindex_sessions(conn, Path(session_manager.data_root))
            print(f"[api][reindex] {summary}", flush=True)
            return jsonify({
                'ok': True,
                'summary': summary,
            })
        except Exception as e:
            err = ErrorResponse(code='reindex_error', message='Failed to reindex', details={'error': str(e)})
            return jsonify(err.model_dump()), 500

    # -------------------- Projects & Datasets CRUD --------------------
    @bp.route('/api/projects', methods=['GET'])
    def api_list_projects():
        try:
            conn = get_connection()
            return jsonify(db_list_projects(conn))
        except Exception as e:
            err = ErrorResponse(code='projects_error', message='Failed to list projects', details={'error': str(e)})
            return jsonify(err.model_dump()), 500

    @bp.route('/api/projects', methods=['POST'])
    def api_create_project():
        try:
            payload = request.get_json(force=True) or {}
            name = payload.get('name')
            description = payload.get('description')
            if not name:
                err = ErrorResponse(code='bad_request', message='name is required')
                return jsonify(err.model_dump()), 400
            conn = get_connection()
            try:
                pid = db_create_project(conn, name, description)
                conn.commit()
                # Fetch the created project with created_at timestamp
                cur = conn.execute("SELECT id, name, description, created_at FROM projects WHERE id = ?", (pid,))
                row = cur.fetchone()
                if row:
                    return jsonify(dict(row)), 201
                else:
                    return jsonify({'id': pid, 'name': name, 'description': description}), 201
            except sqlite3.IntegrityError:
                # Unique constraint on projects.name; return existing
                # Fetch existing row id
                cur = conn.execute("SELECT id, name, description, created_at FROM projects WHERE name = ?", (name,))
                row = cur.fetchone()
                if row:
                    return jsonify(dict(row)), 200
                raise
            except Exception as e:
                conn.rollback()
                raise
        except Exception as e:
            err = ErrorResponse(code='create_project_error', message='Failed to create project', details={'error': str(e)})
            return jsonify(err.model_dump()), 500

    @bp.route('/api/projects/<int:project_id>/datasets', methods=['GET'])
    def api_list_datasets(project_id: int):
        try:
            conn = get_connection()
            return jsonify(db_list_datasets(conn, project_id))
        except Exception as e:
            err = ErrorResponse(code='datasets_error', message='Failed to list datasets', details={'error': str(e)})
            return jsonify(err.model_dump()), 500

    @bp.route('/api/datasets/<int:dataset_id>', methods=['GET'])
    def api_get_dataset(dataset_id: int):
        try:
            conn = get_connection()
            d = db_get_dataset(conn, dataset_id)
            if not d:
                err = ErrorResponse(code='not_found', message='Dataset not found')
                return jsonify(err.model_dump()), 404
            return jsonify(d)
        except Exception as e:
            err = ErrorResponse(code='dataset_error', message='Failed to get dataset', details={'error': str(e)})
            return jsonify(err.model_dump()), 500

    @bp.route('/api/projects/<int:project_id>/datasets', methods=['POST'])
    def api_create_dataset(project_id: int):
        try:
            payload = request.get_json(force=True) or {}
            name = payload.get('name')
            description = payload.get('description')
            target_type_id = payload.get('target_type_id')
            if not name or target_type_id is None:
                err = ErrorResponse(code='bad_request', message='name and target_type_id are required')
                return jsonify(err.model_dump()), 400
            conn = get_connection()
            # Ensure the parent project exists; otherwise, the insert will raise a FK error
            proj_row = conn.execute("SELECT id FROM projects WHERE id = ?", (project_id,)).fetchone()
            if not proj_row:
                err = ErrorResponse(code='not_found', message='Project not found', details={'project_id': project_id})
                return jsonify(err.model_dump()), 404
            try:
                ttid = int(target_type_id)
                # Validate target type exists in DB
                exists = conn.execute("SELECT 1 FROM target_types WHERE id = ?", (ttid,)).fetchone()
                if not exists:
                    err = ErrorResponse(code='bad_request', message='unknown target_type_id', details={'target_type_id': ttid})
                    return jsonify(err.model_dump()), 400
                did = db_create_dataset(conn, project_id, name, description, ttid)
                conn.commit()
                # Fetch full dataset row to include target_type_name
                created = db_get_dataset(conn, did)
                if created:
                    return jsonify(created), 201
                return jsonify({'id': did, 'project_id': project_id, 'name': name, 'description': description, 'target_type_id': ttid}), 201
            except sqlite3.IntegrityError:
                # Unique(project_id, name) exists: return existing entry as 200
                existing = db_get_dataset_by_name(conn, project_id, name)
                if existing:
                    return jsonify(existing), 200
                raise
        except Exception as e:
            err = ErrorResponse(code='create_dataset_error', message='Failed to create dataset', details={'error': str(e)})
            return jsonify(err.model_dump()), 500

    @bp.route('/api/datasets/<int:dataset_id>', methods=['PUT'])
    def api_update_dataset(dataset_id: int):
        try:
            payload = request.get_json(force=True) or {}
            name = payload.get('name')
            description = payload.get('description')
            ttid = payload.get('target_type_id')
            ttid_val = None
            if ttid is not None:
                ttid_val = int(ttid)
                # Validate target type exists in DB
                conn = get_connection()
                tt_exists = conn.execute("SELECT 1 FROM target_types WHERE id = ?", (ttid_val,)).fetchone()
                if not tt_exists:
                    err = ErrorResponse(code='bad_request', message='unknown target_type_id', details={'target_type_id': ttid_val})
                    return jsonify(err.model_dump()), 400
            conn = get_connection()
            existing = db_get_dataset(conn, dataset_id)
            if not existing:
                err = ErrorResponse(code='not_found', message='Dataset not found')
                return jsonify(err.model_dump()), 404
            db_update_dataset(conn, dataset_id, name=name, description=description, target_type_id=ttid_val)
            conn.commit()
            updated = db_get_dataset(conn, dataset_id)
            return jsonify(updated)
        except sqlite3.IntegrityError as e:
            err = ErrorResponse(code='conflict', message='Dataset name already exists in project', details={'error': str(e)})
            return jsonify(err.model_dump()), 409
        except Exception as e:
            err = ErrorResponse(code='update_dataset_error', message='Failed to update dataset', details={'error': str(e)})
            return jsonify(err.model_dump()), 500

    @bp.route('/api/datasets/<int:dataset_id>', methods=['DELETE'])
    def api_delete_dataset(dataset_id: int):
        try:
            conn = get_connection()
            # Optional force delete: when true, rely on FK ON DELETE CASCADE from dependents -> datasets
            force = str(request.args.get('force', '0')).lower() in ('1', 'true', 'yes')
            if not force:
                cnt = conn.execute("SELECT COUNT(1) FROM annotations WHERE dataset_id = ?", (dataset_id,)).fetchone()[0]
                if cnt:
                    err = ErrorResponse(code='conflict', message='Dataset has annotations; cannot delete')
                    return jsonify(err.model_dump()), 409
            n = db_delete_dataset(conn, dataset_id)
            conn.commit()
            return jsonify({'deleted': int(n)})
        except Exception as e:
            err = ErrorResponse(code='delete_dataset_error', message='Failed to delete dataset', details={'error': str(e)})
            return jsonify(err.model_dump()), 500

    # Enroll session frames into dataset (creates annotations rows with status=unlabeled)
    @bp.route('/api/datasets/<int:dataset_id>/enroll_session', methods=['POST'])
    def api_enroll_session(dataset_id: int):
        try:
            payload = request.get_json(force=True) or {}
            session_id = payload.get('session_id')
            settings = payload.get('settings')  # optional default settings_json
            if not session_id:
                err = ErrorResponse(code='bad_request', message='session_id is required')
                return jsonify(err.model_dump()), 400
            conn = get_connection()
            # Ensure dataset exists
            d = db_get_dataset(conn, dataset_id)
            if not d:
                err = ErrorResponse(code='not_found', message='Dataset not found')
                return jsonify(err.model_dump()), 404
            # If baseline settings include categories, ensure dataset_classes exist in DB with stable ordering
            try:
                if isinstance(settings, dict):
                    cats = settings.get('categories')
                    if isinstance(cats, list) and len(cats) > 0:
                        # Sync classes regardless of target type for forward-compat; safe no-op if duplicates
                        db_sync_dataset_classes(conn, dataset_id, [str(c) for c in cats])
            except Exception:
                # Do not fail enrollment on class sync issues; continue
                pass
            summary = db_enroll_session_frames(conn, dataset_id, session_id, settings)
            return jsonify({'ok': True, 'summary': summary})
        except ValueError as ve:
            err = ErrorResponse(code='bad_request', message=str(ve))
            return jsonify(err.model_dump()), 400
        except Exception as e:
            err = ErrorResponse(code='enroll_error', message='Failed to enroll session', details={'error': str(e)})
            return jsonify(err.model_dump()), 500

    @bp.route('/api/datasets/<int:dataset_id>/progress', methods=['GET'])
    def api_dataset_progress(dataset_id: int):
        try:
            conn = get_connection()
            # Base progress
            base = db_dataset_progress(conn, dataset_id)
            # Enrich with dataset meta
            d = db_get_dataset(conn, dataset_id)
            name = d.get('name') if d else None
            try:
                row = conn.execute("SELECT COUNT(1) FROM dataset_classes WHERE dataset_id = ?", (dataset_id,)).fetchone()
                classes_count = int(row[0]) if row and row[0] is not None else 0
            except Exception:
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
                (dataset_id,),
            ).fetchall()
            sessions = []
            for r in sess_rows or []:
                sid = str(r['session_id'] if isinstance(r, dict) else r[0])
                total = int(r['total'] if isinstance(r, dict) else r[1])
                labeled = int((r['labeled'] if isinstance(r, dict) else r[2]) or 0)
                sessions.append({
                    'session_id': sid,
                    'total': total,
                    'labeled': labeled,
                    'unlabeled': int(max(0, total - labeled)),
                })
            out = {
                # Back-compat keys
                'total': int(base.get('total') or 0),
                'labeled': int(base.get('labeled') or base.get('annotated') or 0),
                'unlabeled': int(max(0, int(base.get('total') or 0) - int(base.get('labeled') or base.get('annotated') or 0))),
                # Enriched
                'name': name,
                'classes_count': int(classes_count),
                'sessions': sessions,
            }
            return jsonify(out)
        except Exception as e:
            err = ErrorResponse(code='progress_error', message='Failed to get dataset progress', details={'error': str(e)})
            return jsonify(err.model_dump()), 500

    @bp.route('/api/projects/<int:project_id>/progress', methods=['GET'])
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
                (project_id,),
            ).fetchone()
            labeled_row = conn.execute(
                """
                SELECT COUNT(1) AS cnt
                FROM annotations a
                JOIN datasets d ON d.id = a.dataset_id
                WHERE d.project_id = ? AND a.status = 'labeled'
                """,
                (project_id,),
            ).fetchone()
            total = int(total_row[0] if total_row and total_row[0] is not None else 0)
            labeled = int(labeled_row[0] if labeled_row and labeled_row[0] is not None else 0)
            result = {
                'total': total,
                'labeled': labeled,
                'unlabeled': int(max(0, total - labeled)),
            }
            # Enrich with per-dataset summaries
            try:
                ds_list = db_list_datasets(conn, project_id)
                datasets = []
                for d in ds_list:
                    did = int(d['id'])
                    prog = db_dataset_progress(conn, did)
                    try:
                        row = conn.execute("SELECT COUNT(1) FROM dataset_classes WHERE dataset_id = ?", (did,)).fetchone()
                        classes_count = int(row[0]) if row and row[0] is not None else 0
                    except Exception:
                        classes_count = 0
                    datasets.append({
                        'id': did,
                        'name': d.get('name'),
                        'classes_count': classes_count,
                        'total': int(prog.get('total') or 0),
                        'labeled': int(prog.get('labeled') or prog.get('annotated') or 0),
                        'unlabeled': int(max(0, int(prog.get('total') or 0) - int(prog.get('labeled') or prog.get('annotated') or 0))),
                    })
                result['datasets'] = datasets
            except Exception:
                pass
            return jsonify(result)
        except Exception as e:
            err = ErrorResponse(code='progress_error', message='Failed to get project progress', details={'error': str(e)})
            return jsonify(err.model_dump()), 500

    # Enrollments inspection helpers
    @bp.route('/api/datasets/<int:dataset_id>/enrollments', methods=['GET'])
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
                (dataset_id,),
            ).fetchall()
            out = [str(r[0]) for r in rows if r and r[0] is not None]
            return jsonify(out)
        except Exception as e:
            err = ErrorResponse(code='enrollments_error', message='Failed to list enrollments', details={'error': str(e)})
            return jsonify(err.model_dump()), 500

    @bp.route('/api/datasets/<int:dataset_id>/sessions/<session_id>/enrolled', methods=['GET'])
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
            return jsonify({'enrolled': bool(row is not None)})
        except Exception as e:
            err = ErrorResponse(code='enrollment_check_error', message='Failed to check enrollment', details={'error': str(e)})
            return jsonify(err.model_dump()), 500

    @bp.route('/api/datasets/<int:dataset_id>/labeled', methods=['GET'])
    def api_dataset_labeled(dataset_id: int):
        try:
            # Use service to enrich with frame_path_rel based on metadata and resolvers
            rows = dataset_service.fetch_labeled(dataset_id)
            return jsonify(rows)
        except Exception as e:
            err = ErrorResponse(code='labeled_error', message='Failed to list labeled items', details={'error': str(e)})
            return jsonify(err.model_dump()), 500

    @bp.route('/api/datasets/<int:dataset_id>/classes', methods=['GET'])
    def api_list_dataset_classes(dataset_id: int):
        try:
            conn = get_connection()
            # Ensure dataset exists
            d = db_get_dataset(conn, dataset_id)
            if not d:
                err = ErrorResponse(code='not_found', message='Dataset not found')
                return jsonify(err.model_dump()), 404
            rows = db_list_dataset_classes(conn, dataset_id)
            return jsonify(rows)
        except Exception as e:
            err = ErrorResponse(code='classes_error', message='Failed to list dataset classes', details={'error': str(e)})
            return jsonify(err.model_dump()), 500

    @bp.route('/api/frame', methods=['GET'])
    def get_frame():
        try:
            try:
                from flask import request as _rq
                print(f"[api][frame] request.path={_rq.path} full_path={_rq.full_path} query_string={_rq.query_string}", flush=True)
            except Exception:
                pass
            q = FrameQuery(session_id=request.args.get('session_id', ''),
                           idx=int(request.args.get('idx', '-1')))
            dataset_id_raw = request.args.get('dataset_id')
            dataset_id = int(dataset_id_raw) if (dataset_id_raw is not None and str(dataset_id_raw).strip() != '') else None
        except Exception as e:
            err = ErrorResponse(code='bad_request', message='Invalid query parameters', details={'error': str(e)})
            return jsonify(err.model_dump()), 400

        try:
            sid = (q.session_id or '').strip()
            frame = session_service.get_frame_by_idx(sid, q.idx)
            # Include annotation when dataset_id is provided
            if dataset_id and dataset_id > 0:
                try:
                    frame_id = str(frame['frame_id'])
                    ann = annotation_service.get_annotation_db(sid, int(dataset_id), frame_id)
                except Exception:
                    ann = None
                return jsonify({'frame': frame, 'annotation': ann})
            # Backward compatibility: only frame when no dataset_id
            return jsonify({'frame': frame})
        except IndexError as e:
            err = ErrorResponse(code='not_found', message=str(e))
            return jsonify(err.model_dump()), 404
        except FileNotFoundError as e:
            err = ErrorResponse(code='not_found', message=str(e))
            return jsonify(err.model_dump()), 404
        except Exception as e:
            err = ErrorResponse(code='frame_error', message='Failed to fetch frame', details={'error': str(e)})
            return jsonify(err.model_dump()), 500

    @bp.route('/api/image', methods=['GET'])
    def get_image():
        try:
            q = ImageQuery(session_id=request.args.get('session_id', ''),
                           idx=int(request.args.get('idx', '-1')))
        except Exception as e:
            err = ErrorResponse(code='bad_request', message='Invalid query parameters', details={'error': str(e)})
            return jsonify(err.model_dump()), 400

        try:
            abs_path, _frame = session_service.get_frame_for_image(q.session_id, q.idx)
            return send_file(abs_path)
        except FileNotFoundError as e:
            err = ErrorResponse(code='not_found', message=str(e))
            return jsonify(err.model_dump()), 404
        except Exception as e:
            err = ErrorResponse(code='image_error', message='Failed to load image', details={'error': str(e)})
            return jsonify(err.model_dump()), 500

    # Legacy filesystem save endpoint removed in favor of DB-backed endpoints above

    # -------------------- DB-backed annotation write endpoints --------------------
    @bp.route('/api/annotations/regression', methods=['POST'])
    def api_save_regression():
        try:
            payload = SaveRegressionRequest.model_validate(request.get_json())
        except Exception as e:
            err = ErrorResponse(code='bad_request', message='Invalid payload', details={'error': str(e)})
            return jsonify(err.model_dump()), 400
        try:
            frame_id = payload.frame_id
            if frame_id is None and payload.frame_idx is not None:
                frame = session_service.get_frame_by_idx(payload.session_id, int(payload.frame_idx))
                frame_id = str(frame['frame_id'])
            elif frame_id is None:
                err = ErrorResponse(code='bad_request', message='Either frame_id or frame_idx must be provided')
                return jsonify(err.model_dump()), 400
            
            res = annotation_service.save_regression(
                session_id=payload.session_id,
                dataset_id=payload.dataset_id,
                frame_id=str(frame_id),
                value=payload.value,
                override_settings=payload.override_settings,
            )
            return jsonify({'success': True, 'annotation': res})
        except ValueError as e:
            err = ErrorResponse(code='validation_error', message=str(e))
            return jsonify(err.model_dump()), 400
        except FileNotFoundError as e:
            err = ErrorResponse(code='not_found', message=str(e))
            return jsonify(err.model_dump()), 404
        except Exception as e:
            err = ErrorResponse(code='save_error', message='Failed to save regression', details={'error': str(e)})
            return jsonify(err.model_dump()), 500

    @bp.route('/api/annotations/single_label', methods=['POST'])
    def api_save_single_label():
        try:
            payload = SaveSingleLabelRequest.model_validate(request.get_json())
        except Exception as e:
            err = ErrorResponse(code='bad_request', message='Invalid payload', details={'error': str(e)})
            return jsonify(err.model_dump()), 400
        try:
            # Ensure dataset target_type matches single-label; if not, auto-align to SingleLabelClassification (1)
            try:
                conn = get_connection()
                d = db_get_dataset(conn, int(payload.dataset_id))
                if d and d.get('target_type_name') != 'SingleLabelClassification':
                    db_update_dataset(conn, int(payload.dataset_id), target_type_id=1)
                    conn.commit()
                conn.close()
            except Exception:
                # Non-fatal; DB triggers will still protect integrity
                pass
            
            frame_id = payload.frame_id
            if frame_id is None and payload.frame_idx is not None:
                frame = session_service.get_frame_by_idx(payload.session_id, int(payload.frame_idx))
                frame_id = str(frame['frame_id'])
            elif frame_id is None:
                err = ErrorResponse(code='bad_request', message='Either frame_id or frame_idx must be provided')
                return jsonify(err.model_dump()), 400
                
            # Resolve class_id if only category_name was provided
            class_id = payload.class_id
            if class_id is None and payload.category_name:
                try:
                    conn = get_connection()
                    cls = db_get_or_create_dataset_class(conn, int(payload.dataset_id), str(payload.category_name))
                    conn.commit()
                    class_id = int(cls['id'])
                    conn.close()
                except Exception:
                    class_id = None
            if class_id is None:
                err = ErrorResponse(code='bad_request', message='class_id or category_name must resolve to a class')
                return jsonify(err.model_dump()), 400
            res = annotation_service.save_single_label(
                session_id=payload.session_id,
                dataset_id=payload.dataset_id,
                frame_id=str(frame_id),
                class_id=int(class_id),
                override_settings=payload.override_settings,
            )
            return jsonify({'success': True, 'annotation': res})
        except ValueError as e:
            err = ErrorResponse(code='validation_error', message=str(e))
            return jsonify(err.model_dump()), 400
        except FileNotFoundError as e:
            err = ErrorResponse(code='not_found', message=str(e))
            return jsonify(err.model_dump()), 404
        except Exception as e:
            err = ErrorResponse(code='save_error', message='Failed to save single label', details={'error': str(e)})
            return jsonify(err.model_dump()), 500

    @bp.route('/api/annotations/multilabel', methods=['POST'])
    def api_save_multilabel():
        try:
            payload = SaveMultilabelRequest.model_validate(request.get_json())
        except Exception as e:
            err = ErrorResponse(code='bad_request', message='Invalid payload', details={'error': str(e)})
            return jsonify(err.model_dump()), 400
        try:
            frame_id = payload.frame_id
            if frame_id is None and payload.frame_idx is not None:
                frame = session_service.get_frame_by_idx(payload.session_id, int(payload.frame_idx))
                frame_id = str(frame['frame_id'])
            # Resolve any provided category_names into class_ids
            class_ids = list(payload.class_ids or [])
            if getattr(payload, 'category_names', None):
                try:
                    conn = get_connection()
                    for nm in payload.category_names or []:
                        if not nm:
                            continue
                        cls = db_get_or_create_dataset_class(conn, int(payload.dataset_id), str(nm))
                        class_ids.append(int(cls['id']))
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
                    pass
            res = annotation_service.save_multilabel(
                session_id=payload.session_id,
                dataset_id=payload.dataset_id,
                frame_id=str(frame_id),
                class_ids=class_ids,
                override_settings=payload.override_settings,
            )
            return jsonify(res)
        except Exception as e:
            err = ErrorResponse(code='save_error', message='Failed to save multilabel', details={'error': str(e)})
            return jsonify(err.model_dump()), 500

    # -------------------- Dataset-session settings --------------------
    @bp.route('/api/datasets/<int:dataset_id>/sessions/<session_id>/settings', methods=['PUT'])
    def api_upsert_dataset_session_settings(dataset_id: int, session_id: str):
        try:
            payload = request.get_json(force=True) or {}
            settings = payload.get('settings')
            if not isinstance(settings, dict):
                err = ErrorResponse(code='bad_request', message='settings (object) is required')
                return jsonify(err.model_dump()), 400
            settings_service.upsert_dataset_session_settings(dataset_id, session_id, settings)
            return jsonify({'ok': True})
        except Exception as e:
            err = ErrorResponse(code='settings_error', message='Failed to upsert settings', details={'error': str(e)})
            return jsonify(err.model_dump()), 500

    @bp.route('/api/datasets/<int:dataset_id>/sessions/<session_id>/settings', methods=['GET'])
    def api_get_dataset_session_settings(dataset_id: int, session_id: str):
        try:
            s = settings_service.get_dataset_session_settings(dataset_id, session_id)
            return jsonify({'settings': s})
        except Exception as e:
            err = ErrorResponse(code='settings_error', message='Failed to get settings', details={'error': str(e)})
            return jsonify(err.model_dump()), 500

    @bp.route('/api/datasets/<int:dataset_id>/sessions/<session_id>/settings', methods=['DELETE'])
    def api_delete_dataset_session_settings(dataset_id: int, session_id: str):
        try:
            settings_service.clear_dataset_session_settings(dataset_id, session_id)
            return jsonify({'ok': True})
        except Exception as e:
            err = ErrorResponse(code='settings_error', message='Failed to delete settings', details={'error': str(e)})
            return jsonify(err.model_dump()), 500

    return bp
