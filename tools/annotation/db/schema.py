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
  created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS datasets (
  id               INTEGER PRIMARY KEY,
  project_id       INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  name             TEXT NOT NULL,
  description      TEXT,
  target_type_id   INTEGER NOT NULL REFERENCES target_types(id),
  created_at       DATETIME DEFAULT CURRENT_TIMESTAMP,
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
  ts_ms       INTEGER CHECK (ts_ms IS NULL OR ts_ms >= 0),
  UNIQUE(session_id, frame_id)
);

CREATE INDEX IF NOT EXISTS idx_frames_session_frame ON frames(session_id, frame_id);

CREATE TABLE IF NOT EXISTS dataset_classes (
  id          INTEGER PRIMARY KEY,
  dataset_id  INTEGER NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
  name        TEXT NOT NULL,
  idx         INTEGER CHECK (idx IS NULL OR idx >= 0),
  UNIQUE(dataset_id, name)
);

CREATE INDEX IF NOT EXISTS idx_dataset_classes_dataset ON dataset_classes(dataset_id);
-- Ensure dataset + class id pair is unique to support composite FK references
CREATE UNIQUE INDEX IF NOT EXISTS ux_dataset_classes_dataset_id_id
  ON dataset_classes(dataset_id, id);
-- Enforce stable class ordering per dataset
CREATE UNIQUE INDEX IF NOT EXISTS ux_dataset_classes_dataset_id_idx
  ON dataset_classes(dataset_id, idx);

CREATE TABLE IF NOT EXISTS annotations (
  dataset_id   INTEGER NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
  frame_id     INTEGER NOT NULL REFERENCES frames(id) ON DELETE CASCADE,
  status       TEXT NOT NULL DEFAULT 'unlabeled' CHECK (status IN ('unlabeled','labeled','skipped')),
  settings_json TEXT CHECK (settings_json IS NULL OR json_valid(settings_json)), 
  created_at   DATETIME DEFAULT CURRENT_TIMESTAMP,
  updated_at   DATETIME,
  PRIMARY KEY (dataset_id, frame_id)
)
WITHOUT ROWID;

CREATE INDEX IF NOT EXISTS idx_annotations_frame ON annotations(frame_id);
CREATE INDEX IF NOT EXISTS idx_annotations_dataset ON annotations(dataset_id);
-- Partial index to speed up labeled lookups per dataset
CREATE INDEX IF NOT EXISTS idx_annotations_labeled
ON annotations(dataset_id, frame_id)
WHERE status = 'labeled';

CREATE TABLE IF NOT EXISTS regression_annotations (
  dataset_id INTEGER NOT NULL,
  frame_id   INTEGER NOT NULL,
  value_real REAL NOT NULL,
  PRIMARY KEY (dataset_id, frame_id),
  FOREIGN KEY (dataset_id, frame_id)
    REFERENCES annotations(dataset_id, frame_id) ON DELETE CASCADE
)
WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS classification_annotations (
  dataset_id INTEGER NOT NULL,
  frame_id   INTEGER NOT NULL,
  class_id   INTEGER NOT NULL
             REFERENCES dataset_classes(id) ON DELETE RESTRICT,
  PRIMARY KEY (dataset_id, frame_id),
  FOREIGN KEY (dataset_id, frame_id)
    REFERENCES annotations(dataset_id, frame_id) ON DELETE CASCADE,
  FOREIGN KEY (dataset_id, class_id)
    REFERENCES dataset_classes(dataset_id, id) ON DELETE RESTRICT
)
WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS annotation_labels (
  dataset_id   INTEGER NOT NULL,
  frame_id     INTEGER NOT NULL,
  class_id     INTEGER NOT NULL
               REFERENCES dataset_classes(id) ON DELETE CASCADE,
  PRIMARY KEY (dataset_id, frame_id, class_id),
  FOREIGN KEY (dataset_id, frame_id)
    REFERENCES annotations(dataset_id, frame_id) ON DELETE CASCADE,
  FOREIGN KEY (dataset_id, class_id)
    REFERENCES dataset_classes(dataset_id, id) ON DELETE CASCADE
)
WITHOUT ROWID;

-- Baseline settings per (dataset, session)
CREATE TABLE IF NOT EXISTS dataset_session_settings (
  dataset_id    INTEGER NOT NULL REFERENCES datasets(id)  ON DELETE CASCADE,
  session_id    INTEGER NOT NULL REFERENCES sessions(id)  ON DELETE CASCADE,
  settings_json TEXT CHECK (settings_json IS NULL OR json_valid(settings_json)),
  created_at    DATETIME DEFAULT CURRENT_TIMESTAMP,
  updated_at    DATETIME,
  PRIMARY KEY (dataset_id, session_id)
)
WITHOUT ROWID;

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

-- Triggers enforcing class belongs to dataset are replaced by composite FKs above

CREATE TRIGGER IF NOT EXISTS trg_annotations_touch_updated_at
AFTER UPDATE ON annotations
BEGIN
  UPDATE annotations SET updated_at = CURRENT_TIMESTAMP
  WHERE dataset_id = NEW.dataset_id AND frame_id = NEW.frame_id;
END;

CREATE TRIGGER IF NOT EXISTS trg_ds_settings_touch_updated_at
AFTER UPDATE ON dataset_session_settings
BEGIN
  UPDATE dataset_session_settings SET updated_at = CURRENT_TIMESTAMP
  WHERE dataset_id = NEW.dataset_id AND session_id = NEW.session_id;
END;

-- View: all annotations with payloads, labeled-only (deterministic multilabel order)
CREATE VIEW IF NOT EXISTS annotations_view AS
WITH ordered_labels AS (
  SELECT al.dataset_id, al.frame_id, al.class_id
  FROM annotation_labels al
  JOIN dataset_classes dc ON dc.id = al.class_id AND dc.dataset_id = al.dataset_id
  ORDER BY dc.idx, al.class_id
), labels AS (
  SELECT dataset_id, frame_id, group_concat(class_id) AS multilabel_class_ids_csv
  FROM ordered_labels
  GROUP BY dataset_id, frame_id
)
SELECT
  a.dataset_id,
  a.frame_id,
  a.status,
  a.created_at,
  a.updated_at,
  a.settings_json,
  r.value_real,
  c.class_id AS single_label_class_id,
  l.multilabel_class_ids_csv
FROM annotations a
LEFT JOIN regression_annotations r
       ON r.dataset_id = a.dataset_id AND r.frame_id = a.frame_id
LEFT JOIN classification_annotations c
       ON c.dataset_id = a.dataset_id AND c.frame_id = a.frame_id
LEFT JOIN labels l
       ON l.dataset_id = a.dataset_id AND l.frame_id = a.frame_id
WHERE a.status = 'labeled'
GROUP BY
  a.dataset_id, a.frame_id, a.status, a.created_at, a.updated_at, a.settings_json, r.value_real, c.class_id;

-- View: dataset labeled items with session/frame identity
CREATE VIEW IF NOT EXISTS dataset_labeled_view AS
SELECT
  d.id AS dataset_id,
  s.session_id AS session_id,
  f.frame_id AS frame_id,
  av.value_real,
  av.single_label_class_id,
  av.multilabel_class_ids_csv
FROM annotations_view av
JOIN annotations a ON a.dataset_id = av.dataset_id AND a.frame_id = av.frame_id
JOIN frames f ON f.id = a.frame_id
JOIN sessions s ON s.id = f.session_id
JOIN datasets d ON d.id = a.dataset_id;
"""


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA_SQL)
    # Backfill: add settings_json if missing
    try:
        cur = conn.execute("PRAGMA table_info(annotations);")
        cols = [row[1] for row in cur.fetchall()]
        if 'settings_json' not in cols:
            conn.execute("ALTER TABLE annotations ADD COLUMN settings_json TEXT;")
    except Exception:
        pass
    conn.commit()
