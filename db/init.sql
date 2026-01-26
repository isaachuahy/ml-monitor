-- Create the predictions table
CREATE TABLE IF NOT EXISTS predictions (
    request_id UUID PRIMARY KEY,
    model_version VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    input_data JSONB NOT NULL,
    prediction_prob FLOAT NOT NULL,
    prediction_class INTEGER NOT NULL,
    latency_ms FLOAT NOT NULL
);

-- Create the ground truth table. I kept the request_id as a foreign key to the predictions table to ensure data integrity.
CREATE TABLE IF NOT EXISTS ground_truth (
    request_id UUID PRIMARY KEY REFERENCES predictions(request_id),
    actual_class INTEGER NOT NULL,
    labeled_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create the metrics table
CREATE TABLE IF NOT EXISTS metrics (
    id SERIAL PRIMARY KEY,
    metric_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metric_name VARCHAR(50) NOT NULL,
    metric_value FLOAT NOT NULL,
    model_version VARCHAR(20) NOT NULL,
    window_start TIMESTAMP NOT NULL,
    window_end TIMESTAMP NOT NULL
);