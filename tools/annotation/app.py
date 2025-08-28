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
from api import create_annotation_api

app = Flask(__name__)

# Bootstrap services and register blueprint
session_manager = SessionManager()
app.register_blueprint(create_annotation_api(session_manager))


@app.route('/')
def index():
    """Main annotation interface."""
    return render_template('index.html')


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Web-based Annotation Tool')
    parser.add_argument('--host', default='127.0.0.1', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    print("Game State Annotation Tool")
    print("=" * 40)
    print(f"Starting web server at http://{args.host}:{args.port}")
    print("Open this URL in your browser to start annotating!")
    print()
    
    app.run(host=args.host, port=args.port, debug=args.debug)
