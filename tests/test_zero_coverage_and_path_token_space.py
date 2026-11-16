"""Zero embeddings stay a percentage, and a path token is not stripped.

No Postgres, Ollama, or rerank model. The HTTP server is never started.
"""

import asyncio
import json
import unittest
from unittest.mock import patch

import server


TOKEN = "unit-test-bearer"


class ScriptedConn:
    def __init__(self, script):
        self.script = script
        self.calls = []

    def run(self, sql, **params):
        self.calls.append((sql, params))
        for prefix, rows in self.script:
            if sql.startswith(prefix):
                return rows
        raise AssertionError(sql)


def stats(counts):
    conn = ScriptedConn([
        ("SELECT count(*) FROM documents", [(counts[0],)]),
        ("SELECT count(*) FROM doc_chunks", [(counts[1],)]),
        ("SELECT count(*) FROM emb_mxbai", [(counts[2],)]),
        ("SELECT COALESCE", []),
        ("SELECT pg_size_pretty", [("1 MB", "2 MB", "3 MB", "9 MB")]),
    ])

    def urlopen(_url, timeout=0):
        raise TimeoutError("SECRET-PROBE-MESSAGE")

    with patch.multiple(
        server,
        db=lambda: conn,
        hnsw_ready=lambda _con: False,
        rerank_available=lambda: False,
    ), patch.object(server.urllib.request, "urlopen", urlopen):
        out = server.ltm_stats()
    return out, conn


class ZeroCoverageTests(unittest.TestCase):
    def test_zero_embeddings_are_a_percentage_and_extra_ones_are_not_capped(self):
        # n/a is only for an empty chunk table. A stalled embedder with chunks
        # left over is 0.00%, and a surplus of embeddings is not clipped to 100%.
        out, conn = stats((2, 4, 0))
        self.assertEqual(out["documents"], 2)
        self.assertEqual(out["chunks"], 4)
        self.assertEqual(out["mxbai_embeddings"], 0)
        self.assertEqual(out["embedding_coverage"], "0.00%")
        self.assertNotEqual(out["embedding_coverage"], "n/a")

        out, _conn = stats((1, 2, 5))
        self.assertEqual(out["documents"], 1)
        self.assertEqual(out["chunks"], 2)
        self.assertEqual(out["mxbai_embeddings"], 5)
        self.assertEqual(out["embedding_coverage"], "250.00%")
        self.assertNotIn(out["embedding_coverage"], ("n/a", "100.00%", "100%"))
        rendered = json.dumps(out)
        self.assertNotIn("SECRET-PROBE-MESSAGE", rendered)
        self.assertEqual(out["ollama"], "unreachable (TimeoutError)")

        count_sqls = [sql for sql, _params in conn.calls if sql.startswith("SELECT count")]
        self.assertEqual(count_sqls, [
            "SELECT count(*) FROM documents",
            "SELECT count(*) FROM doc_chunks",
            "SELECT count(*) FROM emb_mxbai",
        ])
        for sql in count_sqls:
            self.assertNotIn("/", sql)
            self.assertNotIn("0.00", sql)
            self.assertNotIn("250", sql)


class App:
    def __init__(self):
        self.scopes = []

    async def __call__(self, scope, receive, send):
        self.scopes.append(scope)
        await send({"type": "http.response.start", "status": 204, "headers": []})
        await send({"type": "http.response.body", "body": b""})


def run_gate(gate, path, headers):
    sent = []

    async def send(message):
        sent.append(message)

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    asyncio.run(gate(
        {"type": "http", "path": path, "headers": headers}, receive, send))
    return sent


def status_of(sent):
    return next(msg["status"] for msg in sent if msg["type"] == "http.response.start")


def body_of(sent):
    return b"".join(
        msg.get("body", b"") for msg in sent if msg["type"] == "http.response.body")


class PathTokenSpaceTests(unittest.TestCase):
    def test_bearer_strips_a_trailing_space_and_the_path_token_does_not(self):
        app = App()
        gate = server._BearerGate(app, TOKEN)

        sent = run_gate(
            gate, "/mcp", [(b"authorization", f"Bearer {TOKEN} ".encode())])
        self.assertEqual(status_of(sent), 204)
        self.assertEqual(body_of(sent), b"")
        self.assertEqual(len(app.scopes), 1)
        self.assertEqual(app.scopes[0]["path"], "/mcp")
        self.assertNotIn(TOKEN, app.scopes[0]["path"])

        sent = run_gate(gate, "/t/" + TOKEN + " /mcp", [])
        self.assertEqual(status_of(sent), 401)
        self.assertEqual(body_of(sent), b'{"error":"unauthorized"}')
        start = next(msg for msg in sent if msg["type"] == "http.response.start")
        headers = {k.lower(): v for k, v in start["headers"]}
        self.assertEqual(headers[b"www-authenticate"], b"Bearer")
        self.assertNotIn(TOKEN.encode(), body_of(sent))
        self.assertEqual(len(app.scopes), 1)
        self.assertNotIn(TOKEN + " ", app.scopes[0]["path"])


if __name__ == "__main__":
    unittest.main()
