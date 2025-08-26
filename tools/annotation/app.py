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


app = Flask(__name__)

# Global state
session_data = None
session_dir = None
annotations = {}


@app.route('/')
def index():
    """Main annotation interface."""
    return render_template('index.html')


@app.route('/api/sessions')
def discover_sessions():
    """Discover available annotation sessions."""
    try:
        sessions = []
        
        # Search directories for sessions (new structure + legacy)
        search_dirs = [
            'data/raw',
            'data/annotated',
            'production_data',
            'collected_data', 
            'markup_data',
            'sparse_data',
            '.'
        ]
        
        print(f"[DEBUG] Searching for sessions in: {search_dirs}")
        print(f"[DEBUG] Current working directory: {Path.cwd()}")
        
        for search_dir in search_dirs:
            search_path = Path(search_dir)
            print(f"[DEBUG] Checking directory: {search_path} (exists: {search_path.exists()})")
            
            if search_path.exists():
                for item in search_path.iterdir():
                    if item.is_dir():
                        print(f"[DEBUG] Found directory: {item}")
                        metadata_file = item / "metadata.json"
                        print(f"[DEBUG] Looking for metadata: {metadata_file} (exists: {metadata_file.exists()})")
                        
                        if metadata_file.exists():
                            try:
                                with open(metadata_file, 'r') as f:
                                    metadata = json.load(f)
                                
                                session_info = {
                                    'path': str(item),
                                    'session_id': metadata.get('session_id', item.name),
                                    'game_name': metadata.get('game_name', 'unknown'),
                                    'frames_count': len(metadata.get('frames', [])),
                                    'start_time': metadata.get('start_time', 'unknown')
                                }
                                sessions.append(session_info)
                                print(f"[DEBUG] Added session: {session_info}")
                                
                            except Exception as e:
                                print(f"[ERROR] Error reading metadata from {metadata_file}: {e}")
                                continue
        
        print(f"[DEBUG] Total sessions found: {len(sessions)}")
        return jsonify(sessions)
        
    except Exception as e:
        print(f"[ERROR] Exception in discover_sessions: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/load_session', methods=['POST'])
def load_session():
    """Load a specific session."""
    global session_data, session_dir, annotations
    
    data = request.get_json()
    session_path = Path(data['session_path'])
    
    if not session_path.exists():
        return jsonify({'error': 'Session path does not exist'}), 400
    
    metadata_file = session_path / "metadata.json"
    if not metadata_file.exists():
        return jsonify({'error': 'No metadata.json found'}), 400
    
    try:
        # Load session metadata
        with open(metadata_file, 'r') as f:
            session_data = json.load(f)
        
        session_dir = session_path
        
        # Load existing annotations if they exist
        annotations_file = session_path / "annotations.json"
        if annotations_file.exists():
            with open(annotations_file, 'r') as f:
                annotations = json.load(f)
        else:
            annotations = {}
        
        return jsonify({
            'success': True,
            'session_id': session_data['session_id'],
            'frames_count': len(session_data['frames']),
            'annotated_count': len(annotations)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/frame/<int:frame_idx>')
def get_frame(frame_idx):
    """Get frame information and annotation."""
    global session_data, annotations
    
    if not session_data or frame_idx >= len(session_data['frames']):
        return jsonify({'error': 'Invalid frame index'}), 400
    
    frame_info = session_data['frames'][frame_idx]
    frame_id = str(frame_info['frame_id'])
    
    # Get existing annotation if available
    annotation = annotations.get(frame_id, {
        'game_state': 'menu',
        'importance': 1,
        'notes': ''
    })
    
    return jsonify({
        'frame_info': frame_info,
        'annotation': annotation,
        'total_frames': len(session_data['frames']),
        'annotated_frames': len(annotations)
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


@app.route('/api/annotate', methods=['POST'])
def save_annotation():
    """Save frame annotation."""
    global session_data, annotations, session_dir
    
    data = request.get_json()
    frame_idx = data['frame_idx']
    
    if not session_data or frame_idx >= len(session_data['frames']):
        return jsonify({'error': 'Invalid frame index'}), 400
    
    frame_info = session_data['frames'][frame_idx]
    frame_id = str(frame_info['frame_id'])
    
    # Save annotation
    annotations[frame_id] = {
        'game_state': data['game_state'],
        'importance': data['importance'],
        'notes': data['notes'],
        'annotated_at': frame_idx
    }
    
    # Auto-save to file
    try:
        annotations_file = session_dir / "annotations.json"
        with open(annotations_file, 'w') as f:
            json.dump(annotations, f, indent=2)
        
        return jsonify({
            'success': True,
            'annotated_count': len(annotations)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/export', methods=['POST'])
def export_dataset():
    """Export annotated dataset."""
    global session_data, annotations, session_dir
    
    if not session_data or not annotations:
        return jsonify({'error': 'No data to export'}), 400
    
    data = request.get_json()
    export_path = Path(data['export_path'])
    
    try:
        # Create dataset structure
        game_states = ['menu', 'loading', 'battle', 'final']
        for state in game_states:
            (export_path / state).mkdir(parents=True, exist_ok=True)
        
        # Copy annotated images
        exported_count = 0
        import shutil
        
        for frame_info in session_data['frames']:
            frame_id = str(frame_info['frame_id'])
            if frame_id in annotations:
                annotation = annotations[frame_id]
                state = annotation['game_state']
                
                src_path = session_dir / frame_info['filename']
                dst_path = export_path / state / frame_info['filename']
                
                if src_path.exists():
                    shutil.copy2(src_path, dst_path)
                    exported_count += 1
        
        # Save dataset metadata
        dataset_info = {
            'source_session': session_data['session_id'],
            'exported_frames': exported_count,
            'classes': game_states,
            'export_date': session_data.get('end_time', 'unknown')
        }
        
        with open(export_path / "dataset_info.json", 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        return jsonify({
            'success': True,
            'exported_count': exported_count,
            'export_path': str(export_path)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/stats')
def get_stats():
    """Get annotation statistics."""
    global session_data, annotations
    
    if not session_data:
        return jsonify({'error': 'No session loaded'}), 400
    
    total_frames = len(session_data['frames'])
    annotated_frames = len(annotations)
    
    # Count by state
    state_counts = {}
    for annotation in annotations.values():
        state = annotation.get('game_state', 'unknown')
        state_counts[state] = state_counts.get(state, 0) + 1
    
    return jsonify({
        'total_frames': total_frames,
        'annotated_frames': annotated_frames,
        'progress_percent': (annotated_frames / total_frames * 100) if total_frames > 0 else 0,
        'state_counts': state_counts
    })


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
