"""Auth channel boundaries and document-id parsing. No database or network."""

import asyncio
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
    return b"".join(msg.get("body", b"") for msg in sent if msg["type"] == "http.response.body")


def doc_row(text):
    return (
        7, "disk", "/vault", "Taxes/a.pdf", "application/pdf", "lease", "Lease",
        None, 100, 1, len(text), "pdftotext", 0, None, text,
    )


class Conn:
    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    def run(self, sql, **params):
        self.calls.append((sql, params))
        if "WHERE id = :v" in sql and params.get("v") == 7:
            return self.rows
        return []


class AuthChannelTests(unittest.TestCase):
    def setUp(self):
        self.app = App()
        self.gate = server._BearerGate(self.app, TOKEN)

    def test_a_query_string_token_does_not_authorize(self):
        sent = run(self.gate, {
            "type": "http",
            "path": "/mcp",
            "query_string": b"token=" + TOKEN.encode() + b"&key=" + TOKEN.encode(),
            "headers": [],
        })
        self.assertEqual(status_of(sent), 401)
        self.assertEqual(body_of(sent), b'{"error":"unauthorized"}')
        self.assertEqual(self.app.scopes, [])
        self.assertNotIn(TOKEN.encode(), body_of(sent))

    def test_a_wrong_bearer_does_not_hide_a_valid_path_token(self):
        sent = run(self.gate, {
            "type": "http",
            "path": "/t/" + TOKEN + "/mcp",
            "headers": [(b"authorization", b"Bearer not-the-token")],
        })
        self.assertEqual(status_of(sent), 204)
        self.assertEqual(len(self.app.scopes), 1)
        seen = self.app.scopes[0]
        self.assertEqual(seen["path"], "/mcp")
        self.assertEqual(seen["raw_path"], b"/mcp")
        self.assertNotIn(TOKEN, seen["path"])
        self.assertNotIn("not-the-token", seen["path"])


class DocumentIdTests(unittest.TestCase):
    def fetch(self, key):
        conn = Conn([doc_row("only-body")])
        with patch.object(server, "db", lambda: conn):
            out = server.ltm_get_document(key)
        return out, conn

    def test_padding_and_leading_zeros_stay_an_id_lookup(self):
        for key in ("  7\t", "007", "7"):
            out, conn = self.fetch(key)
            self.assertTrue(out["found"], key)
            self.assertEqual(out["text"], "only-body", key)
            self.assertEqual(len(conn.calls), 1, key)
            self.assertIn("WHERE id = :v", conn.calls[0][0])
            self.assertEqual(conn.calls[0][1]["v"], 7)
            self.assertIsInstance(conn.calls[0][1]["v"], int)
            self.assertNotIn("ILIKE", conn.calls[0][0])

    def test_a_nearby_token_is_not_parsed_as_an_id(self):
        out, conn = self.fetch("  7a ")
        self.assertFalse(out["found"])
        self.assertNotIn("only-body", str(out))
        self.assertGreaterEqual(len(conn.calls), 1)
        for sql, params in conn.calls:
            self.assertNotIsInstance(params.get("v"), int)
            self.assertNotEqual(params.get("v"), 7)
            self.assertNotIn("7a", sql)
        self.assertEqual(conn.calls[0][1]["v"], "7a")
        self.assertIn("ILIKE :v", conn.calls[1][0])
        self.assertEqual(conn.calls[1][1]["v"], "%7a%")


if __name__ == "__main__":
    unittest.main()
