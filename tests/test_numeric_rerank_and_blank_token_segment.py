"""Numeric rerank flags follow truthiness, and an empty /t/ segment is not a token.

No Postgres, Ollama, or rerank model. The HTTP server is never started.
"""

import asyncio
import json
import unittest
from unittest.mock import patch

import server


ONLY = "ONLY_SNIPPET"
TOKEN = "unit-test-bearer"


def search_row(text, cosine=0.9):
    return (
        1, 10, "lease", "Title", "disk", "/vault", "a.pdf",
        "2024-01-01", "pdftotext", cosine, text,
    )


class ScriptedConn:
    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    def run(self, sql, **params):
        self.calls.append((sql, params))
        if "SELECT c.id" in sql:
            return self.rows
        return []


class App:
    def __init__(self):
        self.scopes = []

    async def __call__(self, scope, receive, send):
        self.scopes.append(scope)
        await send({"type": "http.response.start", "status": 204, "headers": []})
        await send({"type": "http.response.body", "body": b""})


def run_gate(gate, path):
    sent = []

    async def send(message):
        sent.append(message)

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    asyncio.run(gate({"type": "http", "path": path, "headers": []}, receive, send))
    return sent


def status_of(sent):
    return next(msg["status"] for msg in sent if msg["type"] == "http.response.start")


def body_of(sent):
    return b"".join(
        msg.get("body", b"") for msg in sent if msg["type"] == "http.response.body")


class NumericRerankTests(unittest.TestCase):
    def search(self, rerank, calls):
        conn = ScriptedConn([search_row(ONLY)])

        def run_rerank(query, texts, top_k):
            calls.append((query, texts, top_k))
            return [0], [4.0]

        with patch.multiple(
            server,
            db=lambda: conn,
            embed_query=lambda _text: "[0.100000]",
            hnsw_ready=lambda _con: False,
            rerank_available=lambda: True,
            run_rerank=run_rerank,
        ):
            out = server.ltm_search("find the clause", k=2, rerank=rerank)
        select = [params for sql, params in conn.calls if "SELECT c.id" in sql]
        self.assertEqual(len(select), 1)
        self.assertFalse(any(sql.startswith("SET ") for sql, _params in conn.calls))
        return out, select[0]

    def test_zero_does_not_rerank_and_two_does(self):
        # bool(0) is false and bool(2) is true. 0 must not be "requested but
        # unavailable", and 2 must not be forwarded as the integer 2.
        skipped = []
        out, params = self.search(0, skipped)
        self.assertEqual(skipped, [])
        self.assertEqual(params["k"], 2)
        self.assertNotEqual(params["k"], server.RERANK_CANDIDATES)
        self.assertFalse(out["reranked"])
        self.assertNotIn("note", out)
        self.assertNotIn("rerank_score", out["results"][0])
        self.assertEqual(out["results"][0]["snippet"], ONLY)
        self.assertEqual(
            out["confidence"],
            "high (cosine only — enable rerank for a calibrated verdict)",
        )

        calls = []
        out, params = self.search(2, calls)
        self.assertEqual(calls, [("find the clause", [ONLY], 2)])
        self.assertEqual(params["k"], server.RERANK_CANDIDATES)
        self.assertTrue(out["reranked"])
        self.assertNotIn("note", out)
        self.assertEqual(out["results"][0]["rerank_score"], 4.0)
        self.assertEqual(out["confidence"], "medium")
        self.assertEqual(out["top_score"], {"rerank_score": 4.0})
        rendered = json.dumps(out)
        self.assertNotIn("unavailable", rendered)
        self.assertIn(ONLY, rendered)


class BlankTokenSegmentTests(unittest.TestCase):
    def test_an_empty_path_segment_is_not_the_token(self):
        # /t//<token>/mcp partitions to an empty first segment. It must not
        # authorize, and a later slash after a real token must not be collapsed.
        app = App()
        gate = server._BearerGate(app, TOKEN)
        for path in ("/t//" + TOKEN + "/mcp", "/t//mcp", "/t///" + TOKEN + "/mcp"):
            sent = run_gate(gate, path)
            self.assertEqual(status_of(sent), 401, path)
            self.assertNotIn(TOKEN.encode(), body_of(sent))
            self.assertNotIn(b"ok", body_of(sent))
        self.assertEqual(app.scopes, [])

        sent = run_gate(gate, "/t/" + TOKEN + "//mcp")
        self.assertEqual(status_of(sent), 204)
        self.assertEqual(body_of(sent), b"")
        self.assertEqual(len(app.scopes), 1)
        self.assertEqual(app.scopes[0]["path"], "//mcp")
        self.assertNotIn(TOKEN, app.scopes[0]["path"])
        self.assertNotIn(TOKEN.encode(), app.scopes[0]["raw_path"])


if __name__ == "__main__":
    unittest.main()
