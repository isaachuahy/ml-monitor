# ML Model Monitoring System

A production-ready ML monitoring platform that tracks model performance, detects data drift, manages model versions, and provides real-time observability for machine learning models in production.

## Features

- **Real-time Inference API** - FastAPI-based REST API with hot-reloading model support
- **Model Versioning** - Semantic versioning (v1.0.0, v1.0.1, etc.) with automatic version management
- **Performance Monitoring** - Tracks accuracy, F1-score, and other metrics over time
- **Data Drift Detection** - Kolmogorov-Smirnov (KS) test for detecting distribution shifts
- **Automated Retraining** - Scheduled model retraining with candidate model generation
- **Live Dashboard** - Streamlit dashboard with real-time metrics visualization
- **Alerting** - Discord webhook integration for model alerts
- **Database-backed** - PostgreSQL for persistent storage and querying

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Client    │────▶│  API Service │────▶│  PostgreSQL │
│  (HTTP)     │     │  (FastAPI)   │     │   Database   │
└─────────────┘     └──────────────┘     └─────────────┘
                           │                     ▲
                           │                     │
                           ▼                     │
                    ┌──────────────┐            │
                    │ Model Files  │            │
                    │  (Pickle)    │            │
                    └──────────────┘            │
                                                │
┌─────────────┐     ┌──────────────┐           │
│  Dashboard  │────▶│ Eval Worker  │───────────┘
│ (Streamlit)  │     │ (Scheduler)  │
└─────────────┘     └──────────────┘
                           │
                           ├── Retraining
                           ├── Metrics Computation
                           ├── Drift Detection
                           └── Ground Truth Simulation
```

### Components

1. **API Service** (`api/`) - FastAPI inference service with hot-reloading models
2. **Eval Worker** (`eval/`) - Background worker for monitoring, retraining, and drift detection
3. **Dashboard** (`dashboard/`) - Streamlit web UI for visualization
4. **Database** (`db/`) - PostgreSQL with schema for predictions, metrics, and model versions

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Python 3.9+ (for local development)

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd ml-monitor
   ```

2. **Configure environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your database credentials and Discord webhook URL
   ```

3. **Start all services**
   ```bash
   docker-compose up -d
   ```

4. **Initialize database schema**
   ```bash
   # The database is automatically initialized with init.sql
   # For model_versions table, run:
   docker-compose exec db psql -U ml_user -d ml_monitor < db/update_v2.sql
   ```

5. **Access the services**
   - **API**: http://localhost:8000
   - **Dashboard**: http://localhost:8501
   - **API Docs**: http://localhost:8000/docs

## Usage

### Making Predictions

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "income": 75000,
    "debt": 15000,
    "credit_score": 720
  }'
```

Response:
```json
{
  "request_id": "uuid-here",
  "prediction_prob": 0.85,
  "prediction_class": 1,
  "model_version": "v1.0.2"
}
```

### Model Retraining

Models are automatically retrained by the scheduler, or you can trigger manually:

```bash
docker-compose run --rm \
  -e DATABASE_URL=postgresql://ml_user:password@db:5432/ml_monitor \
  eval_worker python -m eval.retrain
```

This creates a new candidate model with an incremented patch version (e.g., `v1.0.2` → `v1.0.3`).

### Activating a New Model Version

1. **Check available versions**
   ```bash
   docker-compose exec db psql -U ml_user -d ml_monitor -c \
     "SELECT version, is_active, created_at FROM model_versions ORDER BY created_at DESC;"
   ```

2. **Activate a version**
   ```bash
   docker-compose exec db psql -U ml_user -d ml_monitor -c \
     "UPDATE model_versions SET is_active = FALSE WHERE is_active = TRUE;
      UPDATE model_versions SET is_active = TRUE WHERE version = 'v1.0.3';
      SELECT version, is_active FROM model_versions;"
   ```

3. **API automatically reloads** - The API polls every 30 seconds and will load the new active model automatically.

## Model Versioning

The system uses **semantic versioning** (MAJOR.MINOR.PATCH):

- **PATCH** (v1.0.0 → v1.0.1): Automated retraining with same architecture
- **MINOR** (v1.0.0 → v1.1.0): New features, significant improvements
- **MAJOR** (v1.0.0 → v2.0.0): Breaking changes, architecture changes

### Version Management

- **Automatic**: Retraining increments PATCH version automatically
- **Manual**: Use SQL or `increment_version()` function for MAJOR/MINOR bumps

```python
from eval.db_utils import increment_version

# Increment patch (automatic in retrain)
increment_version("v1.0.0", "patch")  # → "v1.0.1"

# Increment minor (manual)
increment_version("v1.0.0", "minor")  # → "v1.1.0"

# Increment major (manual)
increment_version("v1.0.0", "major")  # → "v2.0.0"
```

## Monitoring Features

### Performance Metrics

The system tracks:
- **Accuracy** - Model classification accuracy
- **F1-Score** - Harmonic mean of precision and recall
- **Custom metrics** - Extensible metric system

Metrics are computed over sliding windows and stored in the `metrics` table.

### Data Drift Detection

Uses Kolmogorov-Smirnov (KS) test to detect distribution shifts:
- Compares training data distribution vs. production data
- P-value < 0.05 indicates significant drift
- Alerts sent via Discord when drift detected

### Dashboard Views

The Streamlit dashboard shows:
- **Performance Over Time** - Accuracy and F1-score trends
- **Drift Detection** - KS-test p-values with threshold line
- **Prediction Distribution** - Histogram of prediction probabilities
- **Recent Traffic** - Latest prediction logs

## API Endpoints

### `GET /health`
Health check endpoint.

**Response:**
```json
{
  "status": "healthy"
}
```

### `POST /predict`
Make a prediction request.

**Request Body:**
```json
{
  "income": 75000.0,
  "debt": 15000.0,
  "credit_score": 720
}
```

**Response:**
```json
{
  "request_id": "uuid",
  "prediction_prob": 0.85,
  "prediction_class": 1,
  "model_version": "v1.0.2"
}
```

## Database Schema

### Tables

- **`predictions`** - Stores all prediction requests and responses
- **`ground_truth`** - Stores actual labels (for evaluation)
- **`metrics`** - Stores computed metrics over time windows
- **`model_versions`** - Tracks model versions, filepaths, and metadata

### Key Queries

**Get latest metrics:**
```sql
SELECT metric_name, metric_value, window_end 
FROM metrics 
WHERE metric_name IN ('accuracy', 'f1_score')
ORDER BY window_end DESC 
LIMIT 10;
```

**Get active model:**
```sql
SELECT version, filepath, metrics_json 
FROM model_versions 
WHERE is_active = TRUE;
```

## Development

### Project Structure

```
ml-monitor/
├── api/                 # FastAPI inference service
│   ├── app.py          # Main API application
│   ├── schemas.py      # Pydantic models
│   └── Dockerfile.api
├── eval/               # Evaluation and monitoring workers
│   ├── retrain.py      # Model retraining
│   ├── compute_metrics.py
│   ├── drift.py        # Drift detection
│   ├── scheduler.py    # Job scheduler
│   └── db_utils.py     # Database utilities
├── dashboard/          # Streamlit dashboard
│   └── app.py
├── db/                 # Database migrations
│   ├── init.sql
│   └── update_v2.sql
├── models/             # Model artifacts (gitignored)
├── tests/              # Unit and integration tests
│   ├── __init__.py
│   ├── conftest.py             # Pytest fixtures, path setup
│   ├── test_db_utils_retry.py  # Retry logic unit tests
│   ├── test_e2e_db_writes.py   # DB write integration tests
│   └── run_retry_tests.py      # Standalone retry tests (no pytest)
└── docker-compose.yml
```

### Testing

Unit and integration tests cover DB write retry behavior and end-to-end writes.

**Unit tests (no database):** retry logic, backoff, exhausted retries, non-retryable exceptions.

```bash
# With pytest (from repo root, after pip install -r requirements.txt)
pytest tests/test_db_utils_retry.py -v

# Or standalone (no pytest)
python tests/run_retry_tests.py
```

**Integration tests (require running Postgres and DATABASE_URL):** real DB writes and retry path.

```bash
# Start DB, then run (set DATABASE_URL to match .env)
docker-compose up -d db
export DATABASE_URL=postgresql://ml_user:YOUR_PASSWORD@localhost:5433/ml_monitor
# Apply schema if needed: docker-compose exec db psql -U ml_user -d ml_monitor -f /docker-entrypoint-initdb.d/init.sql
# and update_v2 for model_versions
pytest tests/ -v

# Skip integration tests when DB is not available
pytest tests/ -v -m "not integration"
```

**See retry behavior in the logs:** pytest captures logs by default. E.g. to see the "DB write failed (attempt X/Y), retrying in..." messages when a test deliberately triggers retries:

```bash
docker-compose run --rm eval_worker python -m pytest tests/test_db_utils_retry.py -v --log-cli-level=WARNING
```

Run the test that forces two retries then success:

```bash
docker-compose run --rm eval_worker python -m pytest tests/test_db_utils_retry.py::TestRetryDbWrite::test_succeeds_after_two_operational_errors -v --log-cli-level=WARNING
```

You should see two warning lines like: `DB write failed (attempt 1/5), retrying in 0.0s: connection lost`.

### Running Locally

1. **Start database only**
   ```bash
   docker-compose up -d db
   ```

2. **Run API locally**
   ```bash
   export DATABASE_URL=postgresql://ml_user:password@localhost:5433/ml_monitor
   cd api
   uvicorn app:app --reload
   ```

3. **Run eval worker locally**
   ```bash
   export DATABASE_URL=postgresql://ml_user:password@localhost:5433/ml_monitor
   python -m eval.scheduler
   ```

### Adding New Metrics

1. Add metric computation in `eval/compute_metrics.py`
2. Store in `metrics` table with appropriate `metric_name`
3. Dashboard will automatically display if named correctly

### Adding New Models

1. Train and save model as pickle file in `models/` directory
2. Insert into `model_versions` table:
   ```sql
   INSERT INTO model_versions (version, filepath, is_active, metrics_json)
   VALUES ('v1.0.0', '/app/models/model_v1.0.0.pkl', TRUE, '{"accuracy": 0.85}');
   ```
3. API will automatically load it on next poll cycle

## Environment Variables

Required environment variables (see `.env.example`):

- `POSTGRES_USER` - Database username
- `POSTGRES_PASSWORD` - Database password
- `POSTGRES_DB` - Database name
- `DATABASE_URL` - Full PostgreSQL connection string
- `DISCORD_WEBHOOK_URL` - Discord webhook for alerts (optional)

## Notes

- **Model Hot-Reloading**: API polls database every 30 seconds for new active models
- **Scheduler Frequency**: Jobs run every 30 seconds (demo mode). Adjust in `eval/scheduler.py` for production
- **Model Storage**: Models stored as pickle files in `./models/` directory (mounted as volume)
- **Database Port**: PostgreSQL exposed on port `5433` (to avoid conflicts with local PostgreSQL)

## Troubleshooting

### API not loading models
- Check `models/` directory exists and is mounted
- Verify model file exists at path specified in database
- Check API logs: `docker-compose logs api`

### Database connection errors
- Verify `.env` file has correct `DATABASE_URL`
- Check database container is running: `docker-compose ps db`
- Test connection: `docker-compose exec db psql -U ml_user -d ml_monitor`

### Dashboard not updating
- Check database connection in dashboard logs
- Verify scheduler is running: `docker-compose logs eval_worker`
- Ensure metrics are being computed (check `metrics` table)

## Author

Isaac Hua
