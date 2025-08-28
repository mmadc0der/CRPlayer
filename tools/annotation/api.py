from __future__ import annotations

from flask import Blueprint, request, jsonify, send_file
from pathlib import Path
from typing import Optional

from core.session_manager import SessionManager
from services.session_service import SessionService
from services.annotation_service import AnnotationService
from services.dataset_service import DatasetService
from core.path_resolver import resolve_frame_absolute_path
from dto import FrameQuery, ImageQuery, SaveAnnotationRequest, ErrorResponse
from db.connection import get_connection
from db.schema import init_db
from db.indexer import reindex_sessions


def create_annotation_api(session_manager: SessionManager) -> Blueprint:
    bp = Blueprint('annotation_api', __name__)

    session_service = SessionService(session_manager)
    annotation_service = AnnotationService(session_manager)
    dataset_service = DatasetService(session_manager)

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
            return jsonify({
                'ok': True,
                'summary': summary,
            })
        except Exception as e:
            err = ErrorResponse(code='reindex_error', message='Failed to reindex', details={'error': str(e)})
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
            frame = session_service.get_frame_by_idx(q.session_id, q.idx)
            # best-effort: include current annotation if exists
            ann = annotation_service.get_annotation(q.session_id, q.project_name, str(frame['frame_id']))
            return jsonify({'frame': frame, 'annotation': ann})
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
            info = session_manager.find_session_by_id(q.session_id)
            if not info:
                err = ErrorResponse(code='not_found', message='Session not found', details={'session_id': q.session_id})
                return jsonify(err.dict()), 404
            session_dir = Path(info['session_dir'])
            frames = info['metadata'].get('frames', [])
            if q.idx < 0 or q.idx >= len(frames):
                err = ErrorResponse(code='not_found', message=f'frame_idx {q.idx} out of range')
                return jsonify(err.dict()), 404
            frame = frames[q.idx]
            abs_path = resolve_frame_absolute_path(session_manager, session_dir, info['metadata']['session_id'], frame['filename'])
            if not abs_path:
                from pathlib import Path as _P
                abs_path = resolve_frame_absolute_path(session_manager, session_dir, info['metadata']['session_id'], _P(frame['filename']).name)
            if not abs_path or not abs_path.exists():
                err = ErrorResponse(code='not_found', message='Image not found', details={'frame_idx': q.idx})
                return jsonify(err.dict()), 404
            return send_file(abs_path)
        except FileNotFoundError as e:
            err = ErrorResponse(code='not_found', message=str(e))
            return jsonify(err.dict()), 404
        except Exception as e:
            err = ErrorResponse(code='image_error', message='Failed to load image', details={'error': str(e)})
            return jsonify(err.dict()), 500

    @bp.route('/api/save_annotation', methods=['POST'])
    def save_annotation():
        try:
            payload = SaveAnnotationRequest.parse_obj(request.get_json())
        except Exception as e:
            err = ErrorResponse(code='bad_request', message='Invalid payload', details={'error': str(e)})
            return jsonify(err.dict()), 400

        try:
            frame_id: Optional[str] = payload.frame_id
            if frame_id is None and payload.frame_idx is not None:
                frame = session_service.get_frame_by_idx(payload.session_id, int(payload.frame_idx))
                frame_id = str(frame['frame_id'])
            saved = annotation_service.save_annotation(
                session_id=payload.session_id,
                project_name=payload.project_name,
                frame_id=str(frame_id),
                annotations=payload.annotations,
                confidence=payload.confidence,
            )
            return jsonify({'saved': saved})
        except IndexError as e:
            err = ErrorResponse(code='not_found', message=str(e))
            return jsonify(err.dict()), 404
        except FileNotFoundError as e:
            err = ErrorResponse(code='not_found', message=str(e))
            return jsonify(err.dict()), 404
        except Exception as e:
            err = ErrorResponse(code='save_error', message='Failed to save annotation', details={'error': str(e)})
            return jsonify(err.dict()), 500

    return bp
