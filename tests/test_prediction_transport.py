import asyncio
import ssl
import struct

import asyncpg

import pytest

from api.prediction_store import PredictionStore, PredictionUnavailable, _ConnectionLoop


INPUTS = {"income": 50000.0, "debt": 1000.0, "credit_score": 700}
RESULT = {"model_version": "v1", "prediction_prob": 0.2, "prediction_class": 0}


def test_stalled_handshakes_drain_and_release_capacity(monkeypatch):
    async def scenario():
        peers = []
        handlers = set()
        accepted = asyncio.Queue()

        async def handle(reader, writer):
            task = asyncio.current_task()
            handlers.add(task)
            drained = asyncio.Event()
            peers.append((writer, drained))
            try:
                await reader.read(1024)
                await accepted.put(drained)
                while await reader.read(1024):
                    pass
                drained.set()
            finally:
                writer.close()
                handlers.discard(task)

        server = await asyncio.start_server(handle, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        monkeypatch.setenv("DATABASE_URL", f"postgresql://user:password@127.0.0.1:{port}/db?sslmode=disable")
        store = PredictionStore(timeout=0.05, cleanup_grace=0.05, capacity=1)
        request = None
        try:
            for cancel in (False, True, False, True):
                request = asyncio.create_task(store.get_or_create("key", INPUTS, lambda: RESULT))
                drained = await asyncio.wait_for(accepted.get(), 1)
                assert len(store.active) == 1
                with pytest.raises(PredictionUnavailable):
                    await store.get_or_create("overflow", INPUTS, lambda: RESULT)
                if cancel:
                    request.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await request
                else:
                    with pytest.raises(PredictionUnavailable):
                        await request
                await asyncio.wait_for(drained.wait(), 0.5)
                await asyncio.sleep(0)
                assert not store.active
        finally:
            server.close()
            if request is not None and not request.done():
                request.cancel()
                await asyncio.wait({request}, timeout=0.5)
            for writer, _ in peers:
                writer.transport.abort()
            pending = tuple(handlers)
            for handler in pending:
                handler.cancel()
            if pending:
                await asyncio.wait(pending, timeout=0.5)
            await asyncio.wait_for(server.wait_closed(), 0.5)

    asyncio.run(scenario())


# A fixed, test-only key lets the local TLS peer run without external tools.
TLS_KEY = '-----BEGIN PRIVATE KEY-----\nMIIEvwIBADANBgkqhkiG9w0BAQEFAASCBKkwggSlAgEAAoIBAQCnxApVcr57iwry\nCXPlJ6HR/jl5h88xjWw0Fa6BUM/PdzcbGzpKlox5JajsnxQnaYIGVjQOHyIhdIpu\nq5s2tYkOwmfRcNDGv45UJIBP6YAlgmxXnf5Xw/rnHMwBkQ8fbWQEeeNBXs5tLC0P\nthSh9LuRFDCfSZeNrl7IJh8tqQJ7KrjS7aMAp+8JEjFlY1zuqbVkeqieymMaPxNt\nCtj0EaIpYQh9M9USrdXLykaUkfloagEZHhMfCG4Bs83PcLol1q8qKOKIOrb3VC7o\nqIZNWZNstPylnwX5ApYafC4fcC3xwP2m+N0sR4yqHBRnvVynISOoR2gWAebR873C\n1cQv7kAbAgMBAAECggEAFo65hPvEuP/dHOZAexN1IrM4LysfpN4LKoWvzRt8lbzF\n4BMtJ1JO2QtDR22sXBBHt2pn3W46PJ1a776LRDPsldt18idlKxQeoxm1Vwlifain\nwLAxvsNPPTwWAi/rBKLEzAVHlUaWIv/Jn08wPwaVIdDBKOnQQvCrNeZKUu1itAS2\n0JRGNJvVfvWjGoS+J4PQQ3Ifc8NV6mx25sPiGrjdsI0iasAqlWXjBjWenj8qcIuE\njc4IjAMf3bbUCNY2HMln4uJ0ECUmXp4BFlAEDZdchAu7spMCseiWdtI8v52O1WsN\nTZko/7eCELfbP+Pswsn/xqlxrFi7GxsowrW24woswQKBgQDSV+6gze/ly1vnTG4T\nuUYfkqdGh1jKl6EZ78pdfAroLs6NyNazDSHJZEaHfqZfONyP1T1i7qDCr6GoanpM\nSK63KdorKJGzANV0UZd6VihMhC6uFI66RAKX/C/S6MzQ9vfZbqo5HTl9BJxHCZ0j\nLZm+3tO5fkyjs1lwgEXjq7hWQwKBgQDMLjYCSeaSvzCX80/oDbSz8DO/0HpDKh2l\nQ00B17Frf8cItepmTguhJOkNpOrE12mAzFf2bwyRYemn5wwS54Mrln4omSWDrdse\n3a/yWpCextFiBaesmG8wr01MZsvZABOMySS7RjzXDEj0TcPSjeabgBdKBeq7g0lt\nR0Eo4+7NSQKBgQC6eI+o6MhazExeIaiYPbCHvugiI0y41XFRX/3joFq++PfORBTp\nyKOCOS45YjK1yd/mduEjum60CSVj+3kN9YVGBYlOp3l62ithIhAKuSSiqLAFTV42\njKJMAJqY/B7OxZJ1pKCtX1YemZ69by1qCTv6NOo30Fq66P1KOb07l3kkpwKBgQCG\nBPsMa4d94SK7Skizz3NB/LNq/yORPG6zEEbWo8ZgAkIZdJbDygyWHE0g2JQE2kP0\n96vxJJkCQFuHCwAA1Jdjn+d1LHjUQuSmEOsxT8LHhUbgaUfS4UdylE5PjNXWvNcY\nPNrwVb0Pho/IU9bqOJwai2w9+T5HEHi80t1xQpN3iQKBgQDGNN41cbOCeAce3gKg\neJoazvuNOdW9WZuQxCNHndqL0z+bUkQdKLBFgPeYR9cRHQbZCgd5i+xSpvEyp+zd\nmw4jetv/RB/uMlVNjFoy9cSrR7k1nbeJ2QjNJZ/dATprpy5Je8iazeT0nyCVD9kZ\ndk95ZUCprw/1u9LT3Qrgw+1MUA==\n-----END PRIVATE KEY-----\n'
TLS_CERT = '-----BEGIN CERTIFICATE-----\nMIIDCTCCAfGgAwIBAgIUQoDhenvB+LYuzaElD+0K/Un01GMwDQYJKoZIhvcNAQEL\nBQAwFDESMBAGA1UEAwwJbG9jYWxob3N0MB4XDTI2MTExMjIzMzQ1MFoXDTM2MTEw\nOTIzMzQ1MFowFDESMBAGA1UEAwwJbG9jYWxob3N0MIIBIjANBgkqhkiG9w0BAQEF\nAAOCAQ8AMIIBCgKCAQEAp8QKVXK+e4sK8glz5Seh0f45eYfPMY1sNBWugVDPz3c3\nGxs6SpaMeSWo7J8UJ2mCBlY0Dh8iIXSKbqubNrWJDsJn0XDQxr+OVCSAT+mAJYJs\nV53+V8P65xzMAZEPH21kBHnjQV7ObSwtD7YUofS7kRQwn0mXja5eyCYfLakCeyq4\n0u2jAKfvCRIxZWNc7qm1ZHqonspjGj8TbQrY9BGiKWEIfTPVEq3Vy8pGlJH5aGoB\nGR4THwhuAbPNz3C6JdavKijiiDq291Qu6KiGTVmTbLT8pZ8F+QKWGnwuH3At8cD9\npvjdLEeMqhwUZ71cpyEjqEdoFgHm0fO9wtXEL+5AGwIDAQABo1MwUTAdBgNVHQ4E\nFgQUTF6l2Zv7GtY4jWog8Is0aHHmDjMwHwYDVR0jBBgwFoAUTF6l2Zv7GtY4jWog\n8Is0aHHmDjMwDwYDVR0TAQH/BAUwAwEB/zANBgkqhkiG9w0BAQsFAAOCAQEAEgZy\nS5EzSHM/sWG4+TfnZVoBYaAmm2IZMulqsSlyUznPyl8ZyFhh0iyAzIs8rcqW4mcP\nqJqSkUmxjHfNyMMV1ap3WTx6ePQ+c99cma0sqDqXf7znlTxEG9zWvMhYR1/nd77v\ngil0Sdi5nrTzkFTP+gXNwqQluH3MUt3WtTAd/ia0SGIaIdI7rP5Y/U0d/3XrwNCE\nWibHACUI4qVrkM7zqoS7EIpA8vEZQcPxluHB+kc3CZi4TMVYCiYBMmreqnrhtMM/\nLG1HAABd0wZWqMhslwx27LEcGeZaI5kfRqW8jXhSx2DjLbPUvXWYo99uphpU41bf\nkODhg3Mm35FIEupEaQ==\n-----END CERTIFICATE-----\n'


@pytest.mark.parametrize("direct", [False, True])
def test_tls_auth_stalls_abort_raw_transport(monkeypatch, tmp_path, direct):
    key_path = tmp_path / "key.pem"
    cert_path = tmp_path / "cert.pem"
    key_path.write_text(TLS_KEY)
    cert_path.write_text(TLS_CERT)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(cert_path, key_path)

    async def scenario():
        peers = []
        handlers = set()
        accepted = asyncio.Queue()

        async def handle(reader, writer):
            handlers.add(asyncio.current_task())
            drained = asyncio.Event()
            peers.append(writer)
            try:
                if not direct:
                    assert len(await reader.readexactly(8)) == 8
                    writer.write(b"S")
                    await writer.drain()
                incoming, outgoing = ssl.MemoryBIO(), ssl.MemoryBIO()
                tls = context.wrap_bio(incoming, outgoing, server_side=True)
                established = False
                while not established:
                    data = await reader.read(16384)
                    if not data:
                        return
                    incoming.write(data)
                    try:
                        tls.do_handshake()
                        established = True
                    except ssl.SSLWantReadError:
                        pass
                    writer.write(outgoing.read())
                    await writer.drain()
                while True:
                    try:
                        startup = tls.read(16384)
                        if startup:
                            break
                    except ssl.SSLWantReadError:
                        data = await reader.read(16384)
                        if not data:
                            return
                        incoming.write(data)
                await accepted.put(drained)
                while await reader.read(16384):
                    pass
                drained.set()
            finally:
                writer.close()
                handlers.discard(asyncio.current_task())

        server = await asyncio.start_server(handle, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        dsn = f"postgresql://user:password@localhost:{port}/db?sslmode=verify-full&sslrootcert={cert_path}"
        if direct:
            dsn += "&sslnegotiation=direct"
        monkeypatch.setenv("DATABASE_URL", dsn)
        store = PredictionStore(timeout=0.2, cleanup_grace=0.05, capacity=1)
        request = None
        race_fired = asyncio.Event()
        if direct:
            loop = asyncio.get_running_loop()
            create_connection = loop.create_connection

            class CancelOnConnection(asyncio.Protocol):
                def __init__(self, protocol):
                    self.protocol = protocol

                def connection_made(self, transport):
                    self.protocol.connection_made(transport)
                    race_fired.set()
                    request.cancel()

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

            async def intercept(factory, *args, **kwargs):
                if kwargs.get("ssl"):
                    return await create_connection(lambda: CancelOnConnection(factory()), *args, **kwargs)
                return await create_connection(factory, *args, **kwargs)

            monkeypatch.setattr(loop, "create_connection", intercept)
        try:
            for cancel in ((True,) * 4 if direct else (False, True, False, True)):
                race_fired.clear()
                request = asyncio.create_task(store.get_or_create("key", INPUTS, lambda: RESULT))
                drained = await asyncio.wait_for(accepted.get(), 1)
                if direct:
                    assert race_fired.is_set()
                else:
                    assert len(store.active) == 1
                    with pytest.raises(PredictionUnavailable):
                        await store.get_or_create("overflow", INPUTS, lambda: RESULT)
                if cancel:
                    if not direct:
                        request.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await request
                else:
                    with pytest.raises(PredictionUnavailable):
                        await request
                await asyncio.wait_for(drained.wait(), 0.5)
                await asyncio.sleep(0)
                assert not store.active
        finally:
            server.close()
            if request is not None and not request.done():
                request.cancel()
                await asyncio.wait({request}, timeout=0.5)
            for writer in peers:
                writer.transport.abort()
            pending = tuple(handlers)
            for handler in pending:
                handler.cancel()
            if pending:
                await asyncio.wait(pending, timeout=0.5)
            await asyncio.wait_for(server.wait_closed(), 0.5)

    asyncio.run(scenario())



def test_direct_tls_connection_and_query(monkeypatch, tmp_path):
    key_path = tmp_path / "key.pem"
    cert_path = tmp_path / "cert.pem"
    key_path.write_text(TLS_KEY)
    cert_path.write_text(TLS_CERT)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(cert_path, key_path)
    context.set_alpn_protocols(["postgresql"])

    def message(kind, payload):
        return kind + struct.pack("!I", len(payload) + 4) + payload

    async def scenario():
        handlers = set()
        peers = []

        async def handle(reader, writer):
            handlers.add(asyncio.current_task())
            peers.append(writer)
            try:
                size = struct.unpack("!I", await reader.readexactly(4))[0]
                await reader.readexactly(size - 4)
                writer.write(message(b"R", struct.pack("!I", 0)))
                writer.write(message(b"S", b"server_version\x0016.0\x00"))
                writer.write(message(b"S", b"client_encoding\x00UTF8\x00"))
                writer.write(message(b"K", struct.pack("!II", 123, 456)))
                writer.write(message(b"Z", b"I"))
                await writer.drain()
                assert await reader.readexactly(1) == b"Q"
                size = struct.unpack("!I", await reader.readexactly(4))[0]
                assert await reader.readexactly(size - 4) == b"SELECT 1\x00"
                writer.write(message(b"C", b"SELECT 1\x00") + message(b"Z", b"I"))
                await writer.drain()
                await reader.read()
            finally:
                writer.close()
                handlers.discard(asyncio.current_task())

        server = await asyncio.start_server(handle, "127.0.0.1", 0, ssl=context)
        port = server.sockets[0].getsockname()[1]
        dsn = f"postgresql://user:password@localhost:{port}/db?sslmode=verify-full&sslrootcert={cert_path}&sslnegotiation=direct"
        tracked_loop = _ConnectionLoop(asyncio.get_running_loop())
        conn = None
        try:
            conn = await asyncpg.connect(dsn, loop=tracked_loop, timeout=1)
            assert await conn.execute("SELECT 1") == "SELECT 1"
        finally:
            if conn is not None:
                conn.terminate()
            tracked_loop.abort()
            server.close()
            for writer in peers:
                writer.transport.abort()
            pending = tuple(handlers)
            for handler in pending:
                handler.cancel()
            if pending:
                await asyncio.wait(pending, timeout=0.5)
            await asyncio.wait_for(server.wait_closed(), 0.5)

    asyncio.run(scenario())
