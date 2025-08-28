from __future__ import annotations

import sqlite3

SCHEMA_SQL = r"""
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS target_types (
  id    INTEGER PRIMARY KEY,
  name  TEXT NOT NULL UNIQUE
);

INSERT OR IGNORE INTO target_types(id, name) VALUES
  (0,'Regression'),
  (1,'SingleLabelClassification'),
  (2,'MultiLabelClassification');

CREATE TABLE IF NOT EXISTS projects (
  id          INTEGER PRIMARY KEY,
  name        TEXT NOT NULL UNIQUE,
  description TEXT,
  created_at  TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS datasets (
  id               INTEGER PRIMARY KEY,
  project_id       INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  name             TEXT NOT NULL,
  description      TEXT,
  target_type_id   INTEGER NOT NULL REFERENCES target_types(id),
  created_at       TEXT DEFAULT CURRENT_TIMESTAMP,
  UNIQUE(project_id, name)
);

CREATE TABLE IF NOT EXISTS sessions (
  id            INTEGER PRIMARY KEY,
  session_id    TEXT NOT NULL UNIQUE,
  root_path     TEXT NOT NULL,
  metadata_json TEXT
);

CREATE TABLE IF NOT EXISTS frames (
  id          INTEGER PRIMARY KEY,
  session_id  INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
  frame_id    TEXT NOT NULL,
  ts_ms       INTEGER,
  UNIQUE(session_id, frame_id)
);

CREATE INDEX IF NOT EXISTS idx_frames_session_frame ON frames(session_id, frame_id);

CREATE TABLE IF NOT EXISTS dataset_classes (
  id          INTEGER PRIMARY KEY,
  dataset_id  INTEGER NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
  name        TEXT NOT NULL,
  idx         INTEGER,
  UNIQUE(dataset_id, name)
);

CREATE INDEX IF NOT EXISTS idx_dataset_classes_dataset ON dataset_classes(dataset_id);

CREATE TABLE IF NOT EXISTS annotations (
  dataset_id   INTEGER NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
  frame_id     INTEGER NOT NULL REFERENCES frames(id) ON DELETE CASCADE,
  status       TEXT NOT NULL DEFAULT 'unlabeled'
               CHECK (status IN ('unlabeled','labeled','skipped')),
  created_at   TEXT DEFAULT CURRENT_TIMESTAMP,
  updated_at   TEXT,
  PRIMARY KEY (dataset_id, frame_id)
);

CREATE INDEX IF NOT EXISTS idx_annotations_frame ON annotations(frame_id);
CREATE INDEX IF NOT EXISTS idx_annotations_dataset ON annotations(dataset_id);

CREATE TABLE IF NOT EXISTS regression_annotations (
  dataset_id INTEGER NOT NULL,
  frame_id   INTEGER NOT NULL,
  value_real REAL NOT NULL,
  PRIMARY KEY (dataset_id, frame_id),
  FOREIGN KEY (dataset_id, frame_id)
    REFERENCES annotations(dataset_id, frame_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS classification_annotations (
  dataset_id INTEGER NOT NULL,
  frame_id   INTEGER NOT NULL,
  class_id   INTEGER NOT NULL
             REFERENCES dataset_classes(id) ON DELETE RESTRICT,
  PRIMARY KEY (dataset_id, frame_id),
  FOREIGN KEY (dataset_id, frame_id)
    REFERENCES annotations(dataset_id, frame_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS annotation_labels (
  dataset_id   INTEGER NOT NULL,
  frame_id     INTEGER NOT NULL,
  class_id     INTEGER NOT NULL
               REFERENCES dataset_classes(id) ON DELETE CASCADE,
  PRIMARY KEY (dataset_id, frame_id, class_id),
  FOREIGN KEY (dataset_id, frame_id)
    REFERENCES annotations(dataset_id, frame_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_ann_labels_dataset ON annotation_labels(dataset_id);
CREATE INDEX IF NOT EXISTS idx_ann_labels_frame   ON annotation_labels(frame_id);

CREATE TRIGGER IF NOT EXISTS trg_check_regression_dataset
BEFORE INSERT ON regression_annotations
WHEN (
  (SELECT t.name
     FROM datasets d
     JOIN target_types t ON t.id = d.target_type_id
    WHERE d.id = NEW.dataset_id) != 'Regression'
)
BEGIN
  SELECT RAISE(ABORT, 'Dataset target_type must be Regression for regression_annotations');
END;

CREATE TRIGGER IF NOT EXISTS trg_check_classification_dataset
BEFORE INSERT ON classification_annotations
WHEN (
  (SELECT t.name
     FROM datasets d
     JOIN target_types t ON t.id = d.target_type_id
    WHERE d.id = NEW.dataset_id) != 'SingleLabelClassification'
)
BEGIN
  SELECT RAISE(ABORT, 'Dataset target_type must be SingleLabelClassification for classification_annotations');
END;

CREATE TRIGGER IF NOT EXISTS trg_check_multilabel_dataset
BEFORE INSERT ON annotation_labels
WHEN (
  (SELECT t.name
     FROM datasets d
     JOIN target_types t ON t.id = d.target_type_id
    WHERE d.id = NEW.dataset_id) != 'MultiLabelClassification'
)
BEGIN
  SELECT RAISE(ABORT, 'Dataset target_type must be MultiLabelClassification for annotation_labels');
END;

CREATE TRIGGER IF NOT EXISTS trg_no_regression_if_classification
BEFORE INSERT ON regression_annotations
WHEN EXISTS (
  SELECT 1 FROM classification_annotations c
  WHERE c.dataset_id = NEW.dataset_id AND c.frame_id = NEW.frame_id
)
BEGIN
  SELECT RAISE(ABORT, 'Cannot insert regression when classification exists for same (dataset_id, frame_id)');
END;

CREATE TRIGGER IF NOT EXISTS trg_no_classification_if_regression
BEFORE INSERT ON classification_annotations
WHEN EXISTS (
  SELECT 1 FROM regression_annotations r
  WHERE r.dataset_id = NEW.dataset_id AND r.frame_id = NEW.frame_id
)
BEGIN
  SELECT RAISE(ABORT, 'Cannot insert classification when regression exists for same (dataset_id, frame_id)');
END;

CREATE TRIGGER IF NOT EXISTS trg_no_multilabel_if_regression_or_single
BEFORE INSERT ON annotation_labels
WHEN EXISTS (
  SELECT 1 FROM regression_annotations r
  WHERE r.dataset_id = NEW.dataset_id AND r.frame_id = NEW.frame_id
) OR EXISTS (
  SELECT 1 FROM classification_annotations c
  WHERE c.dataset_id = NEW.dataset_id AND c.frame_id = NEW.frame_id
)
BEGIN
  SELECT RAISE(ABORT, 'Cannot insert multilabel when regression or single-label exists for same (dataset_id, frame_id)');
END;

CREATE TRIGGER IF NOT EXISTS trg_classification_class_belongs_to_dataset
BEFORE INSERT ON classification_annotations
WHEN NOT EXISTS (
  SELECT 1 FROM dataset_classes dc
  WHERE dc.id = NEW.class_id AND dc.dataset_id = NEW.dataset_id
)
BEGIN
  SELECT RAISE(ABORT, 'classification_annotations.class_id must belong to the same dataset');
END;

CREATE TRIGGER IF NOT EXISTS trg_multilabel_class_belongs_to_dataset
BEFORE INSERT ON annotation_labels
WHEN NOT EXISTS (
  SELECT 1 FROM dataset_classes dc
  WHERE dc.id = NEW.class_id AND dc.dataset_id = NEW.dataset_id
)
BEGIN
  SELECT RAISE(ABORT, 'annotation_labels.class_id must belong to the same dataset');
END;

CREATE TRIGGER IF NOT EXISTS trg_annotations_touch_updated_at
AFTER UPDATE ON annotations
BEGIN
  UPDATE annotations SET updated_at = CURRENT_TIMESTAMP
  WHERE dataset_id = NEW.dataset_id AND frame_id = NEW.frame_id;
END;

CREATE VIEW IF NOT EXISTS annotations_view AS
SELECT
  a.dataset_id,
  a.frame_id,
  a.status,
  a.created_at,
  a.updated_at,
  r.value_real,
  c.class_id AS single_label_class_id,
  group_concat(al.class_id) AS multilabel_class_ids_csv
FROM annotations a
LEFT JOIN regression_annotations r
       ON r.dataset_id = a.dataset_id AND r.frame_id = a.frame_id
LEFT JOIN classification_annotations c
       ON c.dataset_id = a.dataset_id AND c.frame_id = a.frame_id
LEFT JOIN annotation_labels al
       ON al.dataset_id = a.dataset_id AND al.frame_id = a.frame_id
GROUP BY
  a.dataset_id, a.frame_id, a.status, a.created_at, a.updated_at, r.value_real, c.class_id;
"""


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA_SQL)
    conn.commit()
