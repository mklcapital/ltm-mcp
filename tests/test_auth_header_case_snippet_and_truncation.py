"""Authorization header case, snippet budget, and the 201-character cutoff.

No Postgres, Ollama, or rerank model. The HTTP server is never started.
"""

import asyncio
import json
import unittest
from unittest.mock import patch

import server


TOKEN = "unit-test-bearer"
SUFFIX = "SUFFIX-MUST-DROP"


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


def header_map(sent):
    raw = next(msg["headers"] for msg in sent if msg["type"] == "http.response.start")
    return {k.lower(): v for k, v in raw}


class FakeConn:
    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    def run(self, sql, **params):
        self.calls.append((sql, params))
        if sql.lstrip().upper().startswith("SELECT"):
            return self.rows
        return []


class Resp:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return json.dumps({"embedding": [0.1]}).encode()


class AuthHeaderCaseTests(unittest.TestCase):
    def test_authorization_header_name_is_case_insensitive(self):
        # HTTP header names are case-insensitive. A wrong spelling must not
        # authorize, and the 401 must advertise Bearer without echoing the token.
        app = App()
        gate = server._BearerGate(app, TOKEN)

        sent = run(gate, {
            "type": "http",
            "path": "/mcp",
            "headers": [(b"AuThOrIzAtIoN", b"Bearer " + TOKEN.encode())],
        })
        self.assertEqual(status_of(sent), 204)
        self.assertEqual(len(app.scopes), 1)
        self.assertEqual(app.scopes[0]["path"], "/mcp")

        sent = run(gate, {
            "type": "http",
            "path": "/mcp",
            "headers": [(b"Authorisation", b"Bearer " + TOKEN.encode())],
        })
        self.assertEqual(status_of(sent), 401)
        self.assertEqual(header_map(sent).get(b"www-authenticate"), b"Bearer")
        body = b"".join(
            msg.get("body", b"") for msg in sent
            if msg["type"] == "http.response.body")
        self.assertEqual(body, b'{"error":"unauthorized"}')
        self.assertNotIn(TOKEN.encode(), body)
        self.assertEqual(len(app.scopes), 1)


class SnippetBudgetTests(unittest.TestCase):
    def test_internal_whitespace_does_not_consume_the_snippet_budget(self):
        # The cap applies after whitespace is collapsed. A gap wider than
        # SNIPPET_CHARS must not hide the word that follows it.
        gap = " \t\n" * 120
        self.assertGreater(len("KEEP" + gap), server.SNIPPET_CHARS)
        chunk = "KEEP" + gap + "MID" + ("x" * 400)
        self.assertNotIn("MID", chunk[:server.SNIPPET_CHARS])

        conn = FakeConn([(
            1, 10, "lease", "Title", "disk", "/vault", "Taxes/a.pdf",
            None, "pdftotext", 0.9, chunk,
        )])

        def urlopen(req, timeout=None):
            return Resp()

        with patch.object(server.urllib.request, "urlopen", urlopen), patch.multiple(
            server,
            db=lambda: conn,
            hnsw_ready=lambda _con: False,
            rerank_available=lambda: False,
        ):
            out = server.ltm_search(query="where is the lease", k=1, rerank=False)

        snippet = out["results"][0]["snippet"]
        self.assertEqual(len(snippet), server.SNIPPET_CHARS)
        self.assertTrue(snippet.startswith("KEEP MID"))
        self.assertNotIn("\n", snippet)
        self.assertNotIn("\t", snippet)
        self.assertEqual(snippet.count(" "), 1)
        self.assertNotIn(gap, snippet)
        self.assertNotIn("KEEP" + gap[:8], snippet)


class TruncationBoundaryTests(unittest.TestCase):
    def test_two_hundred_one_characters_drop_the_suffix(self):
        # max_chars=200 keeps a 200-character body whole. One more character
        # must set truncated and drop the suffix, without rewriting char_count.
        text = ("A" * 200) + SUFFIX
        self.assertEqual(len(text), 200 + len(SUFFIX))
        conn = FakeConn([(
            22, "disk", "/vault", "Taxes/a.pdf", "application/pdf", "lease",
            "Title", "2024-05-06", 4096, 3, 99999, "pdftotext", 1,
            "2024-05-07T00:00:00", text,
        )])
        with patch.object(server, "db", lambda: conn):
            out = server.ltm_get_document("22", max_chars=200)

        self.assertTrue(out["found"])
        self.assertTrue(out["truncated"])
        self.assertEqual(out["text"], "A" * 200)
        self.assertNotIn(SUFFIX, out["text"])
        self.assertEqual(len(out["text"]), 200)
        self.assertEqual(out["char_count"], 99999)
        self.assertEqual(out["doc_id"], 22)
        self.assertEqual(len(conn.calls), 1)
        sql, params = conn.calls[0]
        self.assertIn("WHERE id = :v", sql)
        self.assertNotIn("ILIKE", sql)
        self.assertEqual(params["v"], 22)
        self.assertNotIn(SUFFIX, sql)


if __name__ == "__main__":
    unittest.main()
