"""
Dataset download service for exporting labeled annotation data.

This service provides comprehensive dataset export functionality with support for:
- Multiple export formats (JSON, CSV, COCO-style)
- All annotation types (regression, single-label, multi-label)
- Complete metadata and session information
- File path resolution and validation
- Filtering and pagination options
"""

from __future__ import annotations

import csv
import json
import logging
import sqlite3
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from core.session_manager import SessionManager
from db.connection import get_connection
from db.projects import get_dataset
from db.repository import get_session_db_id


class DownloadService:
  """Service for downloading and exporting labeled datasets."""

  def __init__(self, session_manager: SessionManager):
    self.session_manager = session_manager
    self.log = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

  def export_dataset(
    self,
    dataset_id: int,
    format_type: str = "json",
    include_images: bool = False,
    session_filter: Optional[List[str]] = None,
    status_filter: Optional[List[str]] = None,
    limit: Optional[int] = None,
    offset: Optional[int] = None,
  ) -> Dict[str, Any]:
    """
    Export a complete labeled dataset in the specified format.
    
    Args:
      dataset_id: ID of the dataset to export
      format_type: Export format ('json', 'csv', 'coco')
      include_images: Whether to include image data/paths
      session_filter: List of session IDs to include (None = all)
      status_filter: List of annotation statuses to include (default: ['labeled'])
      limit: Maximum number of records to export
      offset: Number of records to skip (for pagination)
        
    Returns:
      Dictionary containing:
      - metadata: Dataset and export information
      - data: Exported annotation data
      - statistics: Export statistics
      - format: Export format used
    """
    try:
      conn = get_connection()

      # Get dataset metadata
      dataset_info = get_dataset(conn, dataset_id)
      if not dataset_info:
        raise ValueError(f"Dataset {dataset_id} not found")

      # Set default status filter
      if status_filter is None:
        status_filter = ["labeled"]

      # Fetch labeled data with comprehensive information
      labeled_data = self._fetch_comprehensive_data(conn, dataset_id, session_filter, status_filter, limit, offset)

      # Get dataset classes for reference
      classes_info = self._get_dataset_classes(conn, dataset_id)

      # Get export statistics
      stats = self._get_export_statistics(conn, dataset_id, session_filter, status_filter)

      # Build comprehensive metadata
      metadata = self._build_export_metadata(
        dataset_info, classes_info, stats, {
          "format_type": format_type,
          "include_images": include_images,
          "session_filter": session_filter,
          "status_filter": status_filter,
          "limit": limit,
          "offset": offset,
        })

      # Format the data according to requested format
      if format_type.lower() == "csv":
        formatted_data = self._format_as_csv(labeled_data, metadata)
      elif format_type.lower() == "coco":
        formatted_data = self._format_as_coco(labeled_data, metadata)
      else:  # Default to JSON
        formatted_data = self._format_as_json(labeled_data, metadata, include_images)

      return {
        "metadata": metadata,
        "data": formatted_data,
        "statistics": stats,
        "format": format_type.lower(),
        "exported_at": datetime.utcnow().isoformat() + "Z",
      }

    except Exception as e:
      self.log.error(f"Failed to export dataset {dataset_id}: {e}", exc_info=True)
      raise
    finally:
      if 'conn' in locals():
        conn.close()

  def _fetch_comprehensive_data(
    self,
    conn: sqlite3.Connection,
    dataset_id: int,
    session_filter: Optional[List[str]],
    status_filter: List[str],
    limit: Optional[int],
    offset: Optional[int],
  ) -> List[Dict[str, Any]]:
    """Fetch comprehensive labeled data with all metadata."""

    # Build dynamic query with filters
    where_conditions = ["a.dataset_id = ?"]
    params = [dataset_id]

    if session_filter:
      placeholders = ",".join("?" * len(session_filter))
      where_conditions.append(f"s.session_id IN ({placeholders})")
      params.extend(session_filter)

    if status_filter:
      placeholders = ",".join("?" * len(status_filter))
      where_conditions.append(f"a.status IN ({placeholders})")
      params.extend(status_filter)

    where_clause = " AND ".join(where_conditions)

    # Comprehensive query joining all relevant tables
    query = f"""
    SELECT 
      -- Dataset and frame identification
      a.dataset_id,
      s.session_id,
      f.frame_id,
      f.ts_ms,
      a.status,
      a.created_at as annotation_created_at,
      a.updated_at as annotation_updated_at,
      
      -- Session metadata
      s.root_path as session_root_path,
      s.metadata_json as session_metadata,
      
      -- Annotation payloads
      r.value_real as regression_value,
      c.class_id as single_label_class_id,
      
      -- Settings
      a.settings_json as frame_settings,
      dss.settings_json as session_settings,
      
      -- Dataset classes for single label
      dc_single.name as single_label_class_name,
      dc_single.idx as single_label_class_index,
      
      -- Multi-label data (aggregated)
      GROUP_CONCAT(
        CASE WHEN al.class_id IS NOT NULL 
        THEN al.class_id || ':' || dc_multi.name || ':' || COALESCE(dc_multi.idx, -1)
        END, '|'
      ) as multilabel_data
        
    FROM annotations a
    JOIN frames f ON f.id = a.frame_id
    JOIN sessions s ON s.id = f.session_id
    
    -- Join annotation payloads
    LEFT JOIN regression_annotations r ON r.dataset_id = a.dataset_id AND r.frame_id = a.frame_id
    LEFT JOIN classification_annotations c ON c.dataset_id = a.dataset_id AND c.frame_id = a.frame_id
    LEFT JOIN annotation_labels al ON al.dataset_id = a.dataset_id AND al.frame_id = a.frame_id
    
    -- Join dataset classes
    LEFT JOIN dataset_classes dc_single ON dc_single.id = c.class_id
    LEFT JOIN dataset_classes dc_multi ON dc_multi.id = al.class_id
    
    -- Join settings
    LEFT JOIN dataset_session_settings dss ON dss.dataset_id = a.dataset_id AND dss.session_id = f.session_id
    
    WHERE {where_clause}
    GROUP BY a.dataset_id, a.frame_id
    ORDER BY s.session_id, COALESCE(f.ts_ms, 0), f.frame_id
    """

    if limit is not None:
      query += f" LIMIT {int(limit)}"
    if offset is not None:
      query += f" OFFSET {int(offset)}"

    cursor = conn.execute(query, params)
    rows = cursor.fetchall()

    # Process results into comprehensive format
    results = []
    for row in rows:
      row_dict = dict(row)

      # Parse JSON fields
      session_metadata = {}
      if row_dict.get("session_metadata"):
        try:
          session_metadata = json.loads(row_dict["session_metadata"])
        except Exception:
          pass

      frame_settings = {}
      if row_dict.get("frame_settings"):
        try:
          frame_settings = json.loads(row_dict["frame_settings"])
        except Exception:
          pass

      session_settings = {}
      if row_dict.get("session_settings"):
        try:
          session_settings = json.loads(row_dict["session_settings"])
        except Exception:
          pass

      # Process multi-label data
      multilabel_classes = []
      if row_dict.get("multilabel_data"):
        for item in row_dict["multilabel_data"].split("|"):
          if item and ":" in item:
            parts = item.split(":", 2)
            if len(parts) >= 3:
              class_id, class_name, class_idx = parts
              multilabel_classes.append({
                "class_id": int(class_id),
                "class_name": class_name,
                "class_index": int(class_idx) if class_idx != "-1" else None,
              })

      # Resolve file path from session metadata
      frame_path = None
      filename = None
      if session_metadata.get("frames"):
        for frame_meta in session_metadata["frames"]:
          if str(frame_meta.get("frame_id")) == str(row_dict["frame_id"]):
            filename = frame_meta.get("filename")
            break

      if filename and row_dict.get("session_root_path"):
        frame_path = str(Path(row_dict["session_root_path"]) / filename)

      # Build comprehensive record
      record = {
        # Identifiers
        "dataset_id": row_dict["dataset_id"],
        "session_id": row_dict["session_id"],
        "frame_id": row_dict["frame_id"],
        "timestamp_ms": row_dict["ts_ms"],

        # Paths and files
        "session_root_path": row_dict["session_root_path"],
        "frame_filename": filename,
        "frame_path": frame_path,

        # Annotation data
        "status": row_dict["status"],
        "annotation_created_at": row_dict["annotation_created_at"],
        "annotation_updated_at": row_dict["annotation_updated_at"],

        # Labels based on type
        "regression_value": row_dict["regression_value"],
        "single_label": {
          "class_id": row_dict["single_label_class_id"],
          "class_name": row_dict["single_label_class_name"],
          "class_index": row_dict["single_label_class_index"],
        } if row_dict["single_label_class_id"] is not None else None,
        "multilabel_classes": multilabel_classes,

        # Settings and metadata
        "frame_settings": frame_settings,
        "session_settings": session_settings,
        "session_metadata": session_metadata,
      }

      results.append(record)

    return results

  def _get_dataset_classes(self, conn: sqlite3.Connection, dataset_id: int) -> List[Dict[str, Any]]:
    """Get all classes defined for the dataset."""
    cursor = conn.execute(
      """
      SELECT id, name, idx
      FROM dataset_classes
      WHERE dataset_id = ?
      ORDER BY COALESCE(idx, id), id
      """, (dataset_id, ))

    return [{"id": row[0], "name": row[1], "index": row[2]} for row in cursor.fetchall()]

  def _get_export_statistics(
    self,
    conn: sqlite3.Connection,
    dataset_id: int,
    session_filter: Optional[List[str]],
    status_filter: List[str],
  ) -> Dict[str, Any]:
    """Get comprehensive export statistics."""

    # Base statistics
    base_stats = conn.execute(
      """
      SELECT 
        COUNT(*) as total_annotations,
        COUNT(DISTINCT f.session_id) as unique_sessions,
        COUNT(CASE WHEN a.status = 'labeled' THEN 1 END) as labeled_count,
        COUNT(CASE WHEN a.status = 'unlabeled' THEN 1 END) as unlabeled_count,
        COUNT(CASE WHEN a.status = 'skipped' THEN 1 END) as skipped_count
      FROM annotations a
      JOIN frames f ON f.id = a.frame_id
      WHERE a.dataset_id = ?
      """, (dataset_id, )).fetchone()

    # Annotation type breakdown
    type_stats = conn.execute(
      """
        SELECT 
          COUNT(r.dataset_id) as regression_count,
          COUNT(c.dataset_id) as single_label_count,
          COUNT(DISTINCT al.dataset_id || '-' || al.frame_id) as multilabel_count
        FROM annotations a
        LEFT JOIN regression_annotations r ON r.dataset_id = a.dataset_id AND r.frame_id = a.frame_id
        LEFT JOIN classification_annotations c ON c.dataset_id = a.dataset_id AND c.frame_id = a.frame_id  
        LEFT JOIN annotation_labels al ON al.dataset_id = a.dataset_id AND al.frame_id = a.frame_id
        WHERE a.dataset_id = ? AND a.status = 'labeled'
      """, (dataset_id, )).fetchone()

    # Session breakdown
    session_stats = conn.execute(
      """
        SELECT 
          s.session_id,
          COUNT(*) as total_frames,
          COUNT(CASE WHEN a.status = 'labeled' THEN 1 END) as labeled_frames
        FROM annotations a
        JOIN frames f ON f.id = a.frame_id
        JOIN sessions s ON s.id = f.session_id
        WHERE a.dataset_id = ?
        GROUP BY s.session_id
        ORDER BY s.session_id
      """, (dataset_id, )).fetchall()

    return {
      "total_annotations":
      base_stats[0],
      "unique_sessions":
      base_stats[1],
      "status_breakdown": {
        "labeled": base_stats[2],
        "unlabeled": base_stats[3],
        "skipped": base_stats[4],
      },
      "annotation_type_breakdown": {
        "regression": type_stats[0],
        "single_label": type_stats[1],
        "multilabel": type_stats[2],
      },
      "session_breakdown": [{
        "session_id": row[0],
        "total_frames": row[1],
        "labeled_frames": row[2],
      } for row in session_stats],
    }

  def _build_export_metadata(
    self,
    dataset_info: Dict[str, Any],
    classes_info: List[Dict[str, Any]],
    stats: Dict[str, Any],
    export_params: Dict[str, Any],
  ) -> Dict[str, Any]:
    """Build comprehensive export metadata."""

    return {
      "dataset": {
        "id": dataset_info["id"],
        "name": dataset_info["name"],
        "description": dataset_info.get("description"),
        "project_id": dataset_info["project_id"],
        "target_type_id": dataset_info["target_type_id"],
        "target_type_name": dataset_info.get("target_type_name"),
        "created_at": dataset_info.get("created_at"),
      },
      "classes": classes_info,
      "export_parameters": export_params,
      "statistics": stats,
      "export_info": {
        "exported_at": datetime.utcnow().isoformat() + "Z",
        "exporter": "CRPlayer Annotation Tool",
        "version": "1.0",
      },
      "schema_version": "1.0",
    }

  def _format_as_json(self, data: List[Dict[str, Any]], metadata: Dict[str, Any],
                      include_images: bool) -> Dict[str, Any]:
    """Format data as comprehensive JSON."""

    # Optionally remove image paths if not requested
    if not include_images:
      for record in data:
        record.pop("frame_path", None)
        record.pop("session_root_path", None)

    return {
      "metadata": metadata,
      "annotations": data,
    }

  def _format_as_csv(self, data: List[Dict[str, Any]], metadata: Dict[str, Any]) -> str:
    """Format data as CSV string."""

    if not data:
      return ""

    output = StringIO()

    # Flatten the data structure for CSV
    flattened_data = []
    for record in data:
      flat_record = {
        "dataset_id": record["dataset_id"],
        "session_id": record["session_id"],
        "frame_id": record["frame_id"],
        "timestamp_ms": record["timestamp_ms"],
        "status": record["status"],
        "frame_filename": record["frame_filename"],
        "frame_path": record["frame_path"],
        "annotation_created_at": record["annotation_created_at"],
        "annotation_updated_at": record["annotation_updated_at"],
        "regression_value": record["regression_value"],
      }

      # Handle single label
      if record["single_label"]:
        flat_record.update({
          "single_label_class_id": record["single_label"]["class_id"],
          "single_label_class_name": record["single_label"]["class_name"],
          "single_label_class_index": record["single_label"]["class_index"],
        })

      # Handle multilabel (as JSON string in CSV)
      if record["multilabel_classes"]:
        flat_record["multilabel_classes_json"] = json.dumps(record["multilabel_classes"])

      # Settings as JSON strings
      if record["frame_settings"]:
        flat_record["frame_settings_json"] = json.dumps(record["frame_settings"])
      if record["session_settings"]:
        flat_record["session_settings_json"] = json.dumps(record["session_settings"])

      flattened_data.append(flat_record)

    if flattened_data:
      writer = csv.DictWriter(output, fieldnames=flattened_data[0].keys())
      writer.writeheader()
      writer.writerows(flattened_data)

    return output.getvalue()

  def _format_as_coco(self, data: List[Dict[str, Any]], metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Format data as COCO-style JSON (adapted for frame annotation)."""

    # Build COCO-style structure
    coco_data = {
      "info": {
        "description": f"CRPlayer Dataset: {metadata['dataset']['name']}",
        "version": metadata["export_info"]["version"],
        "year": datetime.now().year,
        "contributor": "CRPlayer Annotation Tool",
        "date_created": metadata["export_info"]["exported_at"],
      },
      "licenses": [{
        "id": 1,
        "name": "Unknown",
        "url": "",
      }],
      "categories": [],
      "images": [],
      "annotations": [],
    }

    # Build categories from dataset classes
    for cls_info in metadata["classes"]:
      coco_data["categories"].append({
        "id": cls_info["id"],
        "name": cls_info["name"],
        "supercategory": "frame_content",
      })

    # Process annotations
    annotation_id = 1
    for record in data:
      # Add image entry
      image_id = f"{record['session_id']}_{record['frame_id']}"
      image_entry = {
        "id": image_id,
        "width": None,  # Could be extracted from session metadata if available
        "height": None,
        "file_name": record["frame_filename"] or f"{record['frame_id']}.jpg",
        "session_id": record["session_id"],
        "frame_id": record["frame_id"],
        "timestamp_ms": record["timestamp_ms"],
      }
      coco_data["images"].append(image_entry)

      # Add annotations based on type
      if record["regression_value"] is not None:
        coco_data["annotations"].append({
          "id": annotation_id,
          "image_id": image_id,
          "category_id": None,
          "regression_value": record["regression_value"],
          "annotation_type": "regression",
        })
        annotation_id += 1

      if record["single_label"]:
        coco_data["annotations"].append({
          "id": annotation_id,
          "image_id": image_id,
          "category_id": record["single_label"]["class_id"],
          "annotation_type": "single_label",
        })
        annotation_id += 1

      for ml_class in record["multilabel_classes"]:
        coco_data["annotations"].append({
          "id": annotation_id,
          "image_id": image_id,
          "category_id": ml_class["class_id"],
          "annotation_type": "multilabel",
        })
        annotation_id += 1

    return coco_data
