"""Bounded, idempotent persistence for prediction requests."""
import asyncio
import json
import logging
import os
import random
import time
import uuid
from dataclasses import dataclass
from decimal import Decimal

import asyncpg

logger = logging.getLogger(__name__)


class PredictionConflict(Exception):
    """The key belongs to a prediction with different inputs."""


class PredictionUnavailable(Exception):
    """The prediction could not be confirmed; retry with the same key."""


@dataclass
class _Operation:
    connection: object = None
    loop: object = None


class _TrackingProtocol(asyncio.Protocol):
    def __init__(self, protocol, transports):
        self.protocol = protocol
        self.transports = transports

    def connection_made(self, transport):
        self.transports.add(transport)
        self.protocol.connection_made(transport)

    def data_received(self, data):
        self.protocol.data_received(data)

    def connection_lost(self, error):
        self.protocol.connection_lost(error)

    def eof_received(self):
        return self.protocol.eof_received()

    def pause_writing(self):
        self.protocol.pause_writing()

    def resume_writing(self):
        self.protocol.resume_writing()


class _ConnectionLoop:
    # asyncpg closes a pending TLS transport gracefully on cancellation; keep
    # access to it so a stalled TLS shutdown can be aborted immediately.
    def __init__(self, loop):
        self.loop = loop
        self.transports = set()

    def __getattr__(self, name):
        return getattr(self.loop, name)

    def abort(self, *, closing_only=False):
        for transport in tuple(self.transports):
            if not closing_only or transport.is_closing():
                transport.abort()
                self.transports.discard(transport)

    async def create_connection(self, protocol_factory, *args, **kwargs):
        self.abort(closing_only=True)
        if kwargs.get("ssl"):
            factory = protocol_factory
            protocol_factory = lambda: _TrackingProtocol(factory(), self.transports)
        transport, protocol = await self.loop.create_connection(protocol_factory, *args, **kwargs)
        self.transports.add(transport)
        if isinstance(protocol, _TrackingProtocol):
            protocol = protocol.protocol
        return transport, protocol

    async def create_unix_connection(self, *args, **kwargs):
        self.abort(closing_only=True)
        transport, protocol = await self.loop.create_unix_connection(*args, **kwargs)
        self.transports.add(transport)
        return transport, protocol

    async def start_tls(self, transport, *args, **kwargs):
        upgraded = await self.loop.start_tls(transport, *args, **kwargs)
        if upgraded is not None:
            self.transports.discard(transport)
            self.transports.add(upgraded)
        return upgraded


def _retryable(error):
    state = getattr(error, "sqlstate", None)
    if state:
        return state.startswith("08") or state in {
            "40001", "40P01", "55P03", "57014", "57P01", "57P02", "57P03"
        }
    return isinstance(error, (ConnectionError, TimeoutError))


class PredictionStore:
    """Store one committed prediction per key and return existing matching results."""

    def __init__(self, *, connect=None, clock=time.monotonic, sleep=asyncio.sleep,
                 uniform=random.uniform, timeout=8, cleanup_grace=1, capacity=16):
        self.connect = connect or asyncpg.connect
        self.clock = clock
        self.sleep = sleep
        self.uniform = uniform
        self.timeout = timeout
        self.cleanup_grace = cleanup_grace
        self.capacity = capacity
        self.active = set()

    async def get_or_create(self, key, inputs, infer):
        """Return the committed result for key, or raise conflict or unavailable."""
        if len(self.active) >= self.capacity:
            raise PredictionUnavailable()
        operation = _Operation()
        deadline = self.clock() + self.timeout
        task = asyncio.create_task(self._run(operation, key, inputs, infer, deadline))
        self.active.add(task)
        task.add_done_callback(self._finished)
        try:
            done, _ = await asyncio.wait({task}, timeout=self.timeout)
            if done:
                return task.result()
        except asyncio.CancelledError:
            await self._stop(task, operation)
            raise
        await self._stop(task, operation)
        raise PredictionUnavailable()

    def _finished(self, task):
        self.active.discard(task)
        if not task.cancelled():
            task.exception()

    async def _stop(self, task, operation):
        if operation.connection is not None:
            operation.connection.terminate()
        if operation.loop is not None:
            operation.loop.abort()
        task.cancel()
        await asyncio.wait({task}, timeout=self.cleanup_grace)

    async def _run(self, operation, key, inputs, infer, deadline):
        candidate = None
        operation.loop = _ConnectionLoop(asyncio.get_running_loop())
        for attempt in range(3):
            if self.clock() >= deadline:
                raise PredictionUnavailable()
            try:
                operation.connection = await self.connect(
                    os.getenv("DATABASE_URL"), loop=operation.loop, timeout=2, command_timeout=1,
                    server_settings={
                        "application_name": "ml_prediction_store",
                        "statement_timeout": "1000",
                        "lock_timeout": "500",
                        "idle_in_transaction_session_timeout": "2000",
                        "synchronous_commit": "on",
                    },
                )
                operation.loop.abort(closing_only=True)
                operation.loop.transports.clear()
                conn = operation.connection
                await conn.execute("BEGIN ISOLATION LEVEL READ COMMITTED")
                row = await self._select(conn, key)
                if row is not None:
                    return self._matched(row, inputs)
                if candidate is None:
                    started = self.clock()
                    result = infer()
                    candidate = (uuid.uuid4(), result["model_version"],
                                 result["prediction_prob"], result["prediction_class"],
                                 (self.clock() - started) * 1000)
                row = await conn.fetchrow(
                    "INSERT INTO predictions "
                    "(request_id, model_version, input_data, prediction_prob, prediction_class, latency_ms, idempotency_key) "
                    "VALUES ($1, $2, $3::jsonb, $4, $5, $6, $7) "
                    "ON CONFLICT (idempotency_key) DO NOTHING "
                    "RETURNING request_id, model_version, input_data, prediction_prob, prediction_class",
                    candidate[0], candidate[1], json.dumps(inputs), candidate[2], candidate[3], candidate[4], key,
                )
                if row is None:
                    row = await self._select(conn, key)
                    if row is None:
                        raise PredictionUnavailable()
                    return self._matched(row, inputs)
                await conn.execute("COMMIT")
                return self._matched(row, inputs)
            except (asyncpg.PostgresError, asyncpg.InterfaceError, OSError, TimeoutError, ValueError) as error:
                logger.warning("Prediction persistence failed at attempt %d (SQLSTATE %s)",
                               attempt + 1, getattr(error, "sqlstate", None))
                if not _retryable(error) or attempt == 2:
                    raise PredictionUnavailable() from None
            finally:
                if operation.connection is not None:
                    operation.connection.terminate()
                    operation.connection = None
                operation.loop.abort()
            delay = min(self.uniform(0, 0.1 * 2 ** attempt), max(0, deadline - self.clock()))
            await self.sleep(delay)
        raise PredictionUnavailable()

    @staticmethod
    async def _select(conn, key):
        return await conn.fetchrow(
            "SELECT request_id, model_version, input_data, prediction_prob, prediction_class "
            "FROM predictions WHERE idempotency_key = $1", key,
        )

    @staticmethod
    def _matched(row, inputs):
        request_id, version, stored_inputs, probability, prediction_class = row
        stored = json.loads(stored_inputs, parse_int=Decimal, parse_float=Decimal)
        requested = json.loads(json.dumps(inputs), parse_int=Decimal, parse_float=Decimal)
        if stored != requested:
            raise PredictionConflict()
        return {"request_id": str(request_id), "model_version": version,
                "prediction_prob": probability, "prediction_class": prediction_class}
