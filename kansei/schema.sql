CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    checkpoint TEXT,
    decoder TEXT,
    training_config TEXT,
    notes TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS candidates (
    candidate_id TEXT PRIMARY KEY,
    run_id TEXT,
    wav_path TEXT NOT NULL,
    source_id TEXT,
    target_id TEXT,
    category TEXT,
    conversion_intensity REAL,
    legacy_b1_delta_scale REAL,
    tau REAL,
    wet REAL,
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

CREATE TABLE IF NOT EXISTS metrics (
    candidate_id TEXT,
    metric_name TEXT,
    value REAL,
    PRIMARY KEY (candidate_id, metric_name),
    FOREIGN KEY (candidate_id) REFERENCES candidates(candidate_id)
);

CREATE TABLE IF NOT EXISTS comparisons (
    comparison_id TEXT PRIMARY KEY,
    axis TEXT NOT NULL,
    candidate_a TEXT NOT NULL,
    candidate_b TEXT NOT NULL,
    winner TEXT NOT NULL,
    tags TEXT,
    user_id TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (candidate_a) REFERENCES candidates(candidate_id),
    FOREIGN KEY (candidate_b) REFERENCES candidates(candidate_id)
);

CREATE TABLE IF NOT EXISTS preference_scores (
    candidate_id TEXT,
    axis TEXT,
    score REAL,
    uncertainty REAL,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (candidate_id, axis),
    FOREIGN KEY (candidate_id) REFERENCES candidates(candidate_id)
);
