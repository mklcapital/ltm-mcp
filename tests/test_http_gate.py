"""Bearer gate and token-file permissions. No database or network."""

import asyncio
import os
import tempfile
import unittest

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
    return b"".join(msg.get("body", b"") for msg in sent if msg["type"] == "http.response.body")


def header_map(sent):
    start = next(msg for msg in sent if msg["type"] == "http.response.start")
    return {k.decode().lower(): v.decode() for k, v in start.get("headers", [])}


class BearerGateTests(unittest.TestCase):
    def setUp(self):
        self.app = App()
        self.gate = server._BearerGate(self.app, TOKEN)

    def http(self, path, headers=None):
        return run(self.gate, {
            "type": "http",
            "path": path,
            "headers": headers or [],
        })

    def test_healthz_is_open_and_does_not_reach_the_app(self):
        sent = self.http("/healthz", [(b"authorization", b"Bearer wrong")])
        self.assertEqual(status_of(sent), 200)
        self.assertEqual(body_of(sent), b"ok")
        self.assertEqual(self.app.scopes, [])
        self.assertNotIn(TOKEN.encode(), body_of(sent))

    def test_rejects_missing_wrong_and_non_bearer_credentials(self):
        cases = (
            "/mcp",
            [],
            "/mcp",
            [(b"authorization", b"Bearer wrong-token")],
            "/mcp",
            [(b"authorization", b"bearer " + TOKEN.encode())],
            "/mcp",
            [(b"authorization", b"Bearer ")],
            "/t/" + TOKEN + "-extra/mcp",
            [],
        )
        for path, headers in zip(cases[0::2], cases[1::2]):
            app = App()
            sent = run(server._BearerGate(app, TOKEN), {
                "type": "http",
                "path": path,
                "headers": headers,
            })
            self.assertEqual(status_of(sent), 401, path)
            self.assertEqual(body_of(sent), b'{"error":"unauthorized"}')
            self.assertEqual(header_map(sent).get("www-authenticate"), "Bearer")
            self.assertNotIn(TOKEN.encode(), body_of(sent))
            self.assertEqual(app.scopes, [])

    def test_authorization_bearer_is_accepted(self):
        sent = self.http("/mcp", [(b"Authorization", b"Bearer  " + TOKEN.encode())])
        self.assertEqual(status_of(sent), 204)
        self.assertEqual(self.app.scopes[0]["path"], "/mcp")
        self.assertNotIn(TOKEN, self.app.scopes[0]["path"])

    def test_path_token_is_stripped_before_the_app_sees_the_request(self):
        original = "/t/%s/mcp/messages" % TOKEN
        scope = {
            "type": "http",
            "path": original,
            "headers": [(b"authorization", b"Bearer wrong-token")],
        }
        sent = run(self.gate, scope)
        self.assertEqual(status_of(sent), 204)
        self.assertEqual(scope["path"], original)
        seen = self.app.scopes[0]
        self.assertEqual(seen["path"], "/mcp/messages")
        self.assertEqual(seen["raw_path"], b"/mcp/messages")
        self.assertNotIn(TOKEN, seen["path"])
        self.assertNotIn(TOKEN.encode(), seen["raw_path"])

    def test_non_http_events_reach_the_app(self):
        scope = {"type": "lifespan"}
        sent = run(self.gate, scope)
        self.assertEqual(status_of(sent), 204)
        self.assertIs(self.app.scopes[0], scope)


class TokenFileTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.addCleanup(setattr, server, "TOKEN_FILE", server.TOKEN_FILE)
        self.path = os.path.join(self.tmp.name, "nested", "http-token")
        server.TOKEN_FILE = self.path

    def test_new_token_is_private_and_not_logged(self):
        with self.assertLogs("ltm", level="INFO") as captured:
            tok = server._load_or_create_token()
        self.assertTrue(tok)
        self.assertNotIn(tok, "\n".join(captured.output))
        self.assertTrue(any("minted new bearer token" in line for line in captured.output))
        self.assertIn(self.path, "\n".join(captured.output))
        self.assertEqual(os.stat(self.path).st_mode & 0o777, 0o600)
        self.assertEqual(os.stat(os.path.dirname(self.path)).st_mode & 0o777, 0o700)
        with open(self.path, encoding="utf-8") as fh:
            self.assertEqual(fh.read(), tok + "\n")
        self.assertEqual(server._load_or_create_token(), tok)

    def test_existing_token_is_stripped_and_not_rewritten_or_logged(self):
        os.makedirs(os.path.dirname(self.path))
        with open(self.path, "w", encoding="utf-8") as fh:
            fh.write("  kept-token \n")
        with self.assertNoLogs("ltm", level="INFO"):
            self.assertEqual(server._load_or_create_token(), "kept-token")
        with open(self.path, encoding="utf-8") as fh:
            self.assertEqual(fh.read(), "  kept-token \n")


if __name__ == "__main__":
    unittest.main()
