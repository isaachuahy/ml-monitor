CREATE TABLE IF NOT EXISTS model_versions (
    id SERIAL PRIMARY KEY,
    version VARCHAR(50) NOT NULL,
    filepath VARCHAR(255) NOT NULL,
    is_active BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metrics_json JSONB
);

-- Initialise with v1.0.0 model
INSERT INTO model_versions (version, filepath, is_active, metrics_json) VALUES 
('v1.0.0', '/app/models/model_v1.0.0.pkl', TRUE, '{"accuracy": 0.85, "f1_score": 0.85}')
ON CONFLICT DO NOTHING;