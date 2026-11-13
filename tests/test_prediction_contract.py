"""Public prediction HTTP contract tests without an HTTP client dependency."""
import asyncio
import json

import api.app as api


def request_prediction(payload, headers=()):
    body = json.dumps(payload).encode()
    messages = [{"type": "http.request", "body": body, "more_body": False}]
    sent = []

    async def receive():
        if messages:
            return messages.pop(0)
        await asyncio.Event().wait()

    async def send(message):
        sent.append(message)

    async def request():
        scope = {
            "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
            "method": "POST", "scheme": "http", "path": "/predict",
            "raw_path": b"/predict", "query_string": b"", "server": ("test", 80),
            "client": ("test", 1234),
            "headers": [(b"content-type", b"application/json"), *headers],
        }
        await api.app(scope, receive, send)

    asyncio.run(request())
    start = next(message for message in sent if message["type"] == "http.response.start")
    response_body = b"".join(message.get("body", b"") for message in sent if message["type"] == "http.response.body")
    return start["status"], dict(start["headers"]), json.loads(response_body)


def test_database_failure_does_not_return_success(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://localhost:1/unavailable")
    monkeypatch.setattr("eval.db_utils._write_prediction", lambda payload: (_ for _ in ()).throw(RuntimeError("failed")))
    status, _, _ = request_prediction({"income": 50000, "debt": 1000, "credit_score": 700})
    assert status == 503


def test_headers_and_input_validation(monkeypatch):
    class Store:
        async def get_or_create(self, key, inputs, infer):
            return {"request_id": "8ef51668-3c37-45a2-8b12-a639950b392c",
                    "prediction_prob": 0.2, "prediction_class": 0, "model_version": "v1"}

    monkeypatch.setattr(api, "prediction_store", Store())
    payload = {"income": 50000, "debt": 1000, "credit_score": 700}
    status, headers, body = request_prediction(payload, [(b"idempotency-key", b"caller:one")])
    assert status == 200
    assert headers[b"idempotency-key"] == b"caller:one"
    assert set(body) == {"request_id", "prediction_prob", "prediction_class", "model_version"}
    assert request_prediction(payload)[1][b"idempotency-key"].startswith(b"server:")
    assert request_prediction(payload, [(b"idempotency-key", b"bad key")])[0] == 422
    assert request_prediction(payload, [(b"idempotency-key", b"a"), (b"idempotency-key", b"a")])[0] == 422
    assert request_prediction({**payload, "income": "Infinity"})[0] == 422
    assert request_prediction({**payload, "debt": "NaN"})[0] == 422


def test_conflict_and_unavailable_echo_key(monkeypatch):
    from api.prediction_store import PredictionConflict, PredictionUnavailable

    class Store:
        error = PredictionConflict

        async def get_or_create(self, key, inputs, infer):
            raise self.error()

    store = Store()
    monkeypatch.setattr(api, "prediction_store", store)
    payload = {"income": 50000, "debt": 1000, "credit_score": 700}
    for error, expected in [(PredictionConflict, 409), (PredictionUnavailable, 503)]:
        store.error = error
        status, headers, _ = request_prediction(payload, [(b"idempotency-key", b"retry-key")])
        assert status == expected
        assert headers[b"idempotency-key"] == b"retry-key"
