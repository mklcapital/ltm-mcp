"""The listen port is the DNS-rebinding allowlist, and a non-Bearer scheme
does not authenticate.

No Postgres, Ollama, or rerank model. The HTTP server is never started.
"""

import asyncio
import json
import os
import unittest
from unittest.mock import patch

import server


TOKEN = "unit-test-bearer"


class App:
    def __init__(self):
        self.scopes = []

    async def __call__(self, scope, receive, send):
        self.scopes.append(scope)
        await send({"type": "http.response.start", "status": 204, "headers": []})
        await send({"type": "http.response.body", "body": b""})


def run(gate, scope):
    sent = []

    async def send(message):
        sent.append(message)

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    asyncio.run(gate(scope, receive, send))
    return sent


def status_of(sent):
    return next(msg["status"] for msg in sent if msg["type"] == "http.response.start")


def body_of(sent):
    return b"".join(
        msg.get("body", b"") for msg in sent if msg["type"] == "http.response.body")


class ListenPortTests(unittest.TestCase):
    def test_the_listen_port_is_the_allowlist_port(self):
        # Hardcoding :8322 in the allowlist while HTTP_PORT moves would let a
        # rebinding client through, or block the port the server actually bound.
        saved = (server.HTTP_PORT, server.HTTP_HOST)
        settings = server.mcp.settings
        saved_settings = (
            settings.stateless_http,
            settings.host,
            settings.port,
            settings.transport_security,
        )
        server.HTTP_PORT = 9999
        server.HTTP_HOST = "127.0.0.1"
        calls = {}

        def fake_run(app, **kwargs):
            calls["kwargs"] = kwargs

        try:
            os.environ.pop("LTM_PUBLIC_HOST", None)
            with patch.object(server, "_load_or_create_token", return_value=TOKEN):
                with patch.object(server.mcp, "streamable_http_app", return_value=object()):
                    with patch("uvicorn.run", fake_run):
                        with self.assertLogs("ltm", level="INFO") as captured:
                            server.run_http()
            self.assertNotIn(TOKEN, "\n".join(captured.output))
            security = server.mcp.settings.transport_security
            self.assertEqual(
                security.allowed_hosts,
                ["127.0.0.1", "127.0.0.1:9999", "localhost", "localhost:9999"],
            )
            self.assertNotIn("127.0.0.1:8322", security.allowed_hosts)
            self.assertNotIn("localhost:8322", security.allowed_hosts)
            self.assertEqual(security.allowed_origins, [])
            self.assertEqual(calls["kwargs"]["host"], "127.0.0.1")
            self.assertEqual(calls["kwargs"]["port"], 9999)
            self.assertNotEqual(calls["kwargs"]["port"], 8322)
            self.assertIs(calls["kwargs"]["access_log"], False)
            self.assertNotIn(TOKEN, json.dumps(calls["kwargs"]))
        finally:
            server.HTTP_PORT, server.HTTP_HOST = saved
            (
                settings.stateless_http,
                settings.host,
                settings.port,
                settings.transport_security,
            ) = saved_settings


class BasicAuthTests(unittest.TestCase):
    def test_basic_and_a_stuck_scheme_are_rejected_and_healthz_stays_open(self):
        # `Basic <token>` and `Bearer<token>` (no space) must not compare equal.
        # POST /healthz stays the open probe and must not echo the header.
        cases = (
            {"type": "http", "path": "/mcp", "method": "POST",
             "headers": [(b"authorization", b"Basic " + TOKEN.encode())]},
            {"type": "http", "path": "/mcp", "method": "GET",
             "headers": [(b"authorization", b"Bearer" + TOKEN.encode())]},
        )
        for scope in cases:
            app = App()
            sent = run(server._BearerGate(app, TOKEN), scope)
            self.assertEqual(status_of(sent), 401, scope["path"])
            self.assertEqual(body_of(sent), b'{"error":"unauthorized"}')
            self.assertEqual(app.scopes, [])
            self.assertNotIn(TOKEN.encode(), body_of(sent))

        app = App()
        sent = run(server._BearerGate(app, TOKEN), {
            "type": "http",
            "path": "/healthz",
            "method": "POST",
            "headers": [(b"authorization", b"Basic " + TOKEN.encode())],
        })
        self.assertEqual(status_of(sent), 200)
        self.assertEqual(body_of(sent), b"ok")
        self.assertEqual(app.scopes, [])
        self.assertNotIn(TOKEN.encode(), body_of(sent))


if __name__ == "__main__":
    unittest.main()
