FROM python:3.12-slim

WORKDIR /app

# Inference and verification use the same image so pickle loads against the training versions.
RUN --mount=type=secret,id=proxy_ca,required=false \
    if [ -f /run/secrets/proxy_ca ]; then export PIP_CERT=/run/secrets/proxy_ca; fi; \
    pip install --no-cache-dir \
    fastapi==0.128.0 uvicorn==0.34.0 pydantic==2.12.0 asyncpg==0.32.0 psycopg2-binary==2.9.11 \
    requests==2.32.5 numpy==1.26.4 pandas==2.2.3 scipy==1.13.1 scikit-learn==1.6.1

COPY api/*.py api/
COPY eval/*.py eval/
COPY replay/scenario.py replay/
