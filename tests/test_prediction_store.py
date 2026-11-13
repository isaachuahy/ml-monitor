import asyncio
import json

import asyncpg
import pytest

from api.prediction_store import PredictionConflict, PredictionStore, PredictionUnavailable

INPUTS = {"income": 50000.0, "debt": 1000.0, "credit_score": 700}
RESULT = {"model_version": "v1", "prediction_prob": 0.2, "prediction_class": 0}


class Connection:
    def __init__(self, existing=None):
        self.existing = existing
        self.commit_gate = asyncio.Event()
        self.commit_started = asyncio.Event()
        self.closed = False

    async def execute(self, sql):
        if sql == "COMMIT":
            self.commit_started.set()
            await self.commit_gate.wait()
        return "OK"

    async def fetchrow(self, sql, *params):
        if sql.startswith("SELECT"):
            return self.existing
        if sql.startswith("INSERT"):
            return (params[0], params[1], params[2], params[3], params[4])

    def terminate(self):
        self.closed = True


def test_success_waits_for_commit():
    async def scenario():
        conn = Connection()
        async def connect(*args, **kwargs):
            return conn
        store = PredictionStore(connect=connect)
        task = asyncio.create_task(store.get_or_create("key", INPUTS, lambda: RESULT))
        await conn.commit_started.wait()
        assert not task.done()
        conn.commit_gate.set()
        result = await task
        assert result["model_version"] == "v1"
        assert conn.closed
    asyncio.run(scenario())


def test_replay_uses_existing_result_without_inference_and_detects_conflict():
    async def scenario():
        row = ("8ef51668-3c37-45a2-8b12-a639950b3922", "old", json.dumps(INPUTS), 0.4, 1)
        async def connect(*args, **kwargs):
            return Connection(row)
        store = PredictionStore(connect=connect)
        def infer():
            raise AssertionError("replay must not infer")
        assert (await store.get_or_create("key", INPUTS, infer))["model_version"] == "old"
        with pytest.raises(PredictionConflict):
            await store.get_or_create("key", {**INPUTS, "debt": 2}, infer)
    asyncio.run(scenario())


def test_jsonb_expanded_number_matches_validated_input():
    async def scenario():
        inputs = {**INPUTS, "income": 1e23}
        stored = '{"income": 100000000000000000000000, "debt": 1000.0, "credit_score": 700}'
        row = ("8ef51668-3c37-45a2-8b12-a639950b3922", "v1", stored, 0.4, 1)
        async def connect(*args, **kwargs):
            return Connection(row)
        store = PredictionStore(connect=connect)
        def infer():
            raise AssertionError("replay must not infer")
        assert (await store.get_or_create("key", inputs, infer))["model_version"] == "v1"
        with pytest.raises(PredictionConflict):
            await store.get_or_create("key", {**inputs, "income": 1.0000000000000001e23}, infer)
    asyncio.run(scenario())


def test_retry_only_known_transient_failures():
    async def scenario(error, expected):
        calls, delays = [], []
        async def connect(*args, **kwargs):
            calls.append(1)
            raise error
        async def sleep(delay):
            delays.append(delay)
        store = PredictionStore(connect=connect, sleep=sleep, uniform=lambda a, b: b)
        with pytest.raises(PredictionUnavailable):
            await store.get_or_create("key", INPUTS, lambda: RESULT)
        assert len(calls) == expected
        assert delays == ([0.1, 0.2] if expected == 3 else [])
    asyncio.run(scenario(ConnectionRefusedError("connection refused"), 3))
    asyncio.run(scenario(asyncpg.InvalidPasswordError("password authentication failed"), 1))
    asyncio.run(scenario(asyncpg.UndefinedTableError("missing"), 1))


def test_retry_delays_stop_at_deadline():
    async def scenario():
        now = 0.0
        attempts, delays, jitter_bounds = [], [], []

        async def connect(*args, **kwargs):
            attempts.append(now)
            raise ConnectionRefusedError("connection refused")

        async def sleep(delay):
            nonlocal now
            delays.append(delay)
            now += delay

        def uniform(lower, upper):
            jitter_bounds.append((lower, upper))
            return upper

        store = PredictionStore(connect=connect, clock=lambda: now, sleep=sleep,
                                uniform=uniform, timeout=0.15)
        with pytest.raises(PredictionUnavailable):
            await store.get_or_create("key", INPUTS, lambda: RESULT)
        assert attempts == pytest.approx([0.0, 0.1])
        assert jitter_bounds == [(0, 0.1), (0, 0.2)]
        assert delays == pytest.approx([0.1, 0.05])
        assert now == pytest.approx(0.15)
    asyncio.run(scenario())


def test_cancellation_during_connect_releases_capacity():
    async def scenario():
        started = asyncio.Event()
        calls = []
        conn = Connection()
        conn.commit_gate.set()

        async def connect(*args, **kwargs):
            calls.append(1)
            if len(calls) == 1:
                started.set()
                await asyncio.Event().wait()
            return conn

        store = PredictionStore(connect=connect, capacity=1)
        task = asyncio.create_task(store.get_or_create("key", INPUTS, lambda: RESULT))
        await asyncio.wait_for(started.wait(), timeout=1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert len(calls) == 1
        result = await store.get_or_create("next", INPUTS, lambda: RESULT)
        assert result["model_version"] == "v1"
        assert len(calls) == 2
        assert conn.closed
    asyncio.run(scenario())


def test_capacity_and_close_before_cancel():
    async def scenario():
        conn = Connection()
        async def connect(*args, **kwargs):
            return conn
        store = PredictionStore(connect=connect, capacity=1)
        first = asyncio.create_task(store.get_or_create("key", INPUTS, lambda: RESULT))
        await conn.commit_started.wait()
        with pytest.raises(PredictionUnavailable):
            await store.get_or_create("other", INPUTS, lambda: RESULT)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        assert conn.closed
        await asyncio.sleep(0)
        assert not store.active
    asyncio.run(scenario())


def test_ambiguous_commit_reuses_candidate_and_returns_committed_row():
    async def scenario():
        rows = []
        inferred = []
        class AmbiguousConnection(Connection):
            async def fetchrow(self, sql, *params):
                if sql.startswith("INSERT"):
                    rows.append((params[0], params[1], params[2], params[3], params[4]))
                    return rows[-1]
                if sql.startswith("SELECT"):
                    return rows[-1] if rows else None

            async def execute(self, sql):
                if sql == "COMMIT":
                    raise ConnectionResetError("connection reset")
                return "OK"

        async def connect(*args, **kwargs):
            return AmbiguousConnection()
        async def sleep(delay):
            pass
        def infer():
            inferred.append(1)
            return RESULT
        result = await PredictionStore(connect=connect, sleep=sleep).get_or_create("key", INPUTS, infer)
        assert result["request_id"] == str(rows[0][0])
        assert len(rows) == len(inferred) == 1
    asyncio.run(scenario())


def test_unfinished_cancelled_worker_keeps_capacity():
    async def scenario():
        release = asyncio.Event()
        cancelled = asyncio.Event()
        conn = Connection()
        async def stubborn_execute(sql):
            if sql != "COMMIT":
                return "OK"
            conn.commit_started.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                assert conn.closed
                cancelled.set()
                await release.wait()
        conn.execute = stubborn_execute
        async def connect(*args, **kwargs):
            return conn
        store = PredictionStore(connect=connect, capacity=1, timeout=0.02, cleanup_grace=0.02)
        with pytest.raises(PredictionUnavailable):
            await store.get_or_create("key", INPUTS, lambda: RESULT)
        assert cancelled.is_set()
        assert len(store.active) == 1
        with pytest.raises(PredictionUnavailable):
            await store.get_or_create("next", INPUTS, lambda: RESULT)
        release.set()
        await asyncio.wait(store.active, timeout=1)
        await asyncio.sleep(0)
        assert not store.active
    asyncio.run(scenario())
