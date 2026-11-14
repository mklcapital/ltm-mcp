"""A plus in a path token is not decoded, and the search text never reaches SQL.

No Postgres, Ollama, or rerank model. The HTTP server is never started.
"""

import asyncio
import json
import unittest
from unittest.mock import patch

import server


TOKEN = "unit+test-bearer"
QUERY = "  '; DROP TABLE documents;--  "
SNIPPET = "clause-text-must-stay"


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


class FakeConn:
    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    def run(self, sql, **params):
        self.calls.append((sql, params))
        if sql.lstrip().upper().startswith("SELECT"):
            return self.rows
        return []


class PathPlusTests(unittest.TestCase):
    def test_a_plus_in_the_path_token_is_not_percent_decoded(self):
        # The path segment is compared raw. Decoding %2B would authorize a
        # different token than the one on disk, and would also let a bearer
        # plus-token and an encoded path token drift apart.
        app = App()
        gate = server._BearerGate(app, TOKEN)

        sent = run(gate, {
            "type": "http",
            "path": "/t/" + TOKEN + "/mcp",
            "headers": [],
        })
        self.assertEqual(status_of(sent), 204)
        self.assertEqual(len(app.scopes), 1)
        self.assertEqual(app.scopes[0]["path"], "/mcp")
        self.assertEqual(app.scopes[0]["raw_path"], b"/mcp")
        self.assertNotIn(TOKEN.encode(), app.scopes[0]["raw_path"])

        for encoded in ("unit%2Btest-bearer", "unit%2btest-bearer"):
            sent = run(gate, {
                "type": "http",
                "path": "/t/" + encoded + "/mcp",
                "headers": [],
            })
            self.assertEqual(status_of(sent), 401, encoded)
            body = b"".join(
                msg.get("body", b"") for msg in sent
                if msg["type"] == "http.response.body")
            self.assertEqual(body, b'{"error":"unauthorized"}')
            self.assertNotIn(TOKEN.encode(), body)
        self.assertEqual(len(app.scopes), 1)

        sent = run(gate, {
            "type": "http",
            "path": "/mcp",
            "headers": [(b"authorization", b"Bearer " + TOKEN.encode())],
        })
        self.assertEqual(status_of(sent), 204)
        self.assertEqual(app.scopes[-1]["path"], "/mcp")
        self.assertNotIn(b"%2B", app.scopes[-1]["path"].encode())


class QueryTextTests(unittest.TestCase):
    def test_the_search_query_is_embedded_and_not_interpolated(self):
        # The natural-language query is a prompt, not a SQL fragment. Spaces
        # stay, and a statement hidden in the query cannot reach the database.
        conn = FakeConn([(
            1, 10, "lease", "Title", "disk", "/vault", "Taxes/a.pdf",
            None, "pdftotext", 0.9, SNIPPET,
        )])
        seen = {}

        class Resp:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return json.dumps({"embedding": [0.1]}).encode()

        def urlopen(req, timeout=None):
            seen["prompt"] = json.loads(req.data.decode())
            seen["timeout"] = timeout
            return Resp()

        with patch.object(server.urllib.request, "urlopen", urlopen), patch.multiple(
            server,
            db=lambda: conn,
            hnsw_ready=lambda _con: False,
            rerank_available=lambda: False,
        ):
            out = server.ltm_search(query=QUERY, k=1, rerank=False)

        self.assertEqual(out["query"], QUERY)
        self.assertEqual(seen["prompt"]["model"], server.EMBED_MODEL)
        self.assertEqual(seen["prompt"]["prompt"], server.QUERY_PREFIX + QUERY)
        self.assertEqual(seen["prompt"]["prompt"].count(server.QUERY_PREFIX), 1)
        self.assertEqual(seen["timeout"], 60)
        self.assertEqual(out["results"][0]["snippet"], SNIPPET)
        self.assertNotIn("DROP", out["results"][0]["snippet"])
        self.assertFalse(out["reranked"])
        self.assertEqual(len(conn.calls), 1)
        sql, params = conn.calls[0]
        self.assertNotIn("DROP", sql)
        self.assertNotIn(QUERY, sql)
        self.assertNotIn("documents;--", sql)
        self.assertNotIn("SET ", sql)
        self.assertEqual(params["k"], 1)
        self.assertEqual(params["e"], "[0.100000]")
        self.assertNotIn("dt", params)
        self.assertNotIn("pc", params)


if __name__ == "__main__":
    unittest.main()
