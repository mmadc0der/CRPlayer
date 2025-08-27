#!/usr/bin/env python3
"""
Web-based Annotation Tool for Game State Markup
Flask web application for in-place annotation via browser.
"""

from flask import Flask, render_template, request, jsonify, send_file
import json
import os
from pathlib import Path
from typing import Dict, List, Optional
import argparse

from core.session_manager import SessionManager
from core.annotation_store import AnnotationStore
from core.dataset_builder import DatasetBuilder

app = Flask(__name__)

# Global state - refactored to use new architecture
session_manager = SessionManager()
annotation_store = AnnotationStore(session_manager)
dataset_builder = DatasetBuilder(session_manager)

# Legacy compatibility
session_data = None
session_dir = None
current_project_name = "default_classification"


@app.route('/')
def index():
    """Main annotation interface."""
    return render_template('index.html')


@app.route('/api/sessions')
def discover_sessions():
    """Discover available annotation sessions."""
    try:
        sessions = session_manager.discover_sessions()
        print(f"[DEBUG] Total sessions found: {len(sessions)}")
        return jsonify(sessions)
        
    except Exception as e:
        print(f"[ERROR] Exception in discover_sessions: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/load_session', methods=['POST'])
def load_session():
    """Load a specific session."""
    global session_data, session_dir, current_project_name
    
    data = request.get_json()
    session_path = data['session_path']
    project_name = data.get('project_name', current_project_name)
    
    try:
        # Load session using new architecture
        session_info = session_manager.load_session(session_path)
        session_data = session_info['metadata']
        session_dir = Path(session_info['session_dir'])
        
        # Load or create project
        project = annotation_store.load_session_project(session_path, project_name)
        current_project_name = project_name
        
        return jsonify({
            'success': True,
            'session_id': session_info['session_id'],
            'frames_count': len(session_info['frames']),
            'annotated_count': len(project.annotations),
            'project_name': project_name,
            'annotation_type': project.annotation_type,
            'categories': project.categories,
            'hotkeys': project.hotkeys
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/frame/<int:frame_idx>')
def get_frame(frame_idx):
    """Get frame information and annotation."""
    global session_data
    
    if not session_data or frame_idx >= len(session_data['frames']):
        return jsonify({'error': 'Invalid frame index'}), 400
    
    frame_info = session_data['frames'][frame_idx]
    frame_id = str(frame_info['frame_id'])
    
    # Get existing annotation using new architecture (by frame_id)
    ann_obj = annotation_store.get_annotation(frame_id)
    if ann_obj:
        annotation = {
            'category': ann_obj.annotations.get('category', ''),
            'notes': ann_obj.annotations.get('notes', '')
        }
    else:
        annotation = {
            'category': '',
            'notes': ''
        }
    
    # Get project stats
    stats = annotation_store.get_project_stats(len(session_data['frames']))
    
    return jsonify({
        'frame_info': frame_info,
        'annotation': annotation,
        'total_frames': len(session_data['frames']),
        'annotated_frames': stats.get('annotated_frames', 0)
    })


@app.route('/api/image/<int:frame_idx>')
def serve_image(frame_idx):
    """Serve frame image."""
    global session_data, session_dir
    
    print(f"[DEBUG] Image request for frame_idx: {frame_idx}")
    print(f"[DEBUG] Session data loaded: {session_data is not None}")
    print(f"[DEBUG] Session dir: {session_dir}")
    
    if not session_data:
        print(f"[ERROR] No session data loaded")
        return "No session loaded", 404
        
    if frame_idx >= len(session_data['frames']):
        print(f"[ERROR] Frame index {frame_idx} out of range (max: {len(session_data['frames'])-1})")
        return "Frame index out of range", 404
    
    frame_info = session_data['frames'][frame_idx]
    print(f"[DEBUG] Frame info: {frame_info}")
    
    # Check if we need to look in frames subdirectory
    image_path = session_dir / frame_info['filename']
    frames_path = session_dir / 'frames' / frame_info['filename']
    
    print(f"[DEBUG] Checking image path: {image_path}")
    print(f"[DEBUG] Image exists at main path: {image_path.exists()}")
    print(f"[DEBUG] Checking frames subdir: {frames_path}")
    print(f"[DEBUG] Image exists at frames path: {frames_path.exists()}")
    
    # Try frames subdirectory first (new structure)
    if frames_path.exists():
        print(f"[DEBUG] Serving from frames subdir: {frames_path}")
        return send_file(frames_path)
    elif image_path.exists():
        print(f"[DEBUG] Serving from main dir: {image_path}")
        return send_file(image_path)
    else:
        print(f"[ERROR] Image not found at either location")
        return "Image not found", 404


@app.route('/api/save_annotation', methods=['POST'])
def save_annotation():
    """Save frame annotation."""
    global session_data
    
    data = request.get_json()
    frame_idx = data['frame_idx']
    
    try:
        # Map index to frame_id and save using new architecture
        frame_info = session_data['frames'][frame_idx]
        frame_id = str(frame_info['frame_id'])
        
        annotation_data = {
            'category': data.get('category', ''),
            'notes': data.get('notes', '')
        }
        
        annotation_store.save_annotation(frame_id, annotation_data)
        stats = annotation_store.get_project_stats(len(session_data['frames']))
        
        return jsonify({
            'success': True,
            'annotated_count': stats.get('annotated_frames', 0)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/export', methods=['POST'])
def export_dataset():
    """Export annotated dataset using new architecture."""
    global session_data, session_dir, current_project_name
    
    if not session_data:
        return jsonify({'error': 'No session loaded'}), 400
    
    data = request.get_json()
    dataset_id = data.get('dataset_id', f"{current_project_name}_{session_data['session_id']}")
    
    try:
        # Create dataset using new architecture
        session_id = session_data['session_id']
        manifest = dataset_builder.create_dataset(
            dataset_id=dataset_id,
            project_name=current_project_name,
            session_ids=[session_id]
        )
        
        export_path = str(dataset_builder.datasets_dir / manifest.dataset_id)
        return jsonify({
            'success': True,
            'dataset_id': manifest.dataset_id,
            'exported_count': len(manifest.samples),
            'statistics': manifest.get_statistics(),
            'export_path': export_path
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/stats')
def get_stats():
    """Get annotation statistics."""
    global session_data
    
    if not session_data:
        return jsonify({'error': 'No session loaded'}), 400
    
    try:
        # Get stats using new architecture
        stats = annotation_store.get_project_stats(len(session_data['frames']))
        
        # Get category distribution
        state_counts = {}
        if annotation_store.current_project:
            for annotation in annotation_store.current_project.annotations.values():
                # Count by saved 'category'
                state = annotation.annotations.get('category', 'unknown')
                if state and state != 'unknown':
                    state_counts[state] = state_counts.get(state, 0) + 1
        
        return jsonify({
            'total_frames': stats.get('total_frames', 0),
            'annotated_frames': stats.get('annotated_frames', 0),
            'progress_percent': stats.get('progress_percent', 0),
            'state_counts': state_counts
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# New API endpoints for project management
@app.route('/api/save_categories', methods=['POST'])
def save_categories():
    """Save categories and hotkeys for a project."""
    data = request.get_json()
    session_path = data['session_path']
    project_name = data.get('project_name', 'default')
    categories = data.get('categories', [])
    hotkeys = data.get('hotkeys', {})
    
    try:
        # Load the project and update categories
        project = annotation_store.load_session_project(session_path, project_name)
        project.categories = categories
        project.hotkeys = hotkeys
        
        # Save the updated project
        annotation_store.save_project(session_path, project_name, project)
        
        return jsonify({'success': True})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/projects', methods=['GET'])
def list_projects():
    """List all projects in current session."""
    global session_dir
    
    if not session_dir:
        return jsonify({'error': 'No session loaded'}), 400
    
    try:
        projects = session_manager._load_session_projects(session_dir)
        project_list = []
        
        for name, project in projects.items():
            project_list.append({
                'name': name,
                'annotation_type': project.annotation_type,
                'categories': project.categories,
                'annotated_count': len(project.annotations),
                'created_at': project.created_at.isoformat() if project.created_at else None,
                'updated_at': project.updated_at.isoformat() if project.updated_at else None
            })
        
        return jsonify(project_list)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/projects', methods=['POST'])
def create_project():
    """Create a new annotation project."""
    global session_dir
    
    if not session_dir:
        return jsonify({'error': 'No session loaded'}), 400
    
    data = request.get_json()
    project_name = data.get('project_name')
    annotation_type = data.get('annotation_type', 'classification')
    
    if not project_name:
        return jsonify({'error': 'Project name required'}), 400
    
    try:
        project = session_manager.create_project(str(session_dir), project_name, annotation_type)
        
        return jsonify({
            'success': True,
            'project': {
                'name': project.name,
                'annotation_type': project.annotation_type,
                'categories': project.categories,
                'created_at': project.created_at.isoformat() if project.created_at else None
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/datasets', methods=['GET'])
def list_datasets():
    """List all available datasets."""
    try:
        datasets = dataset_builder.list_datasets()
        return jsonify(datasets)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Web-based Annotation Tool')
    parser.add_argument('--host', default='127.0.0.1', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    print("🎮 Game State Annotation Tool")
    print("=" * 40)
    print(f"Starting web server at http://{args.host}:{args.port}")
    print("Open this URL in your browser to start annotating!")
    print()
    
    app.run(host=args.host, port=args.port, debug=args.debug)
