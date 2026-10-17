"""One-sided search filters, null document text, and auth-gate exactness.

No Postgres, Ollama, or rerank model. The HTTP server is never started.
"""

import asyncio
import json
import unittest
from unittest.mock import patch

import server


TOKEN = "unit-test-bearer"


class FakeConn:
    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    def run(self, sql, **params):
        self.calls.append((sql, params))
        if sql.lstrip().upper().startswith("SELECT"):
            return self.rows
        return []


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


class FilterScopeTests(unittest.TestCase):
    def search(self, **kwargs):
        conn = FakeConn([])

        def embed(_text):
            return "[0.100000]"

        with patch.multiple(
            server,
            db=lambda: conn,
            embed_query=embed,
            hnsw_ready=lambda _con: True,
            rerank_available=lambda: False,
        ):
            out = server.ltm_search(**kwargs)
        return out, conn

    def test_either_filter_alone_keeps_iterative_scan_and_only_its_predicate(self):
        # HNSW under-returns when a WHERE is present. `or` must not become `and`:
        # a single filter still has to scan until LIMIT is filled, and must not
        # add the other predicate.
        out, conn = self.search(query="taxes", k=4, rerank=False, doc_type="lease")
        self.assertEqual(out["results"], [])
        sets = [sql for sql, _params in conn.calls if sql.startswith("SET")]
        self.assertIn("SET hnsw.iterative_scan = relaxed_order", sets)
        select_sql, params = next(
            (sql, p) for sql, p in conn.calls if sql.startswith("SELECT"))
        self.assertEqual(params["dt"], "lease")
        self.assertNotIn("pc", params)
        self.assertIn("ILIKE :dt", select_sql)
        self.assertNotIn(":pc", select_sql)
        self.assertNotIn("lease", select_sql)

        _out, conn = self.search(query="taxes", k=4, rerank=False, path_contains="Taxes")
        sets = [sql for sql, _params in conn.calls if sql.startswith("SET")]
        self.assertIn("SET hnsw.iterative_scan = relaxed_order", sets)
        select_sql, params = next(
            (sql, p) for sql, p in conn.calls if sql.startswith("SELECT"))
        self.assertEqual(params["pc"], "%Taxes%")
        self.assertNotIn("dt", params)
        self.assertIn("ILIKE :pc", select_sql)
        self.assertNotIn(":dt", select_sql)
        self.assertNotIn("Taxes", select_sql)


class DocumentNullTests(unittest.TestCase):
    def test_null_text_is_empty_and_ocr_null_is_false(self):
        conn = FakeConn([(
            7, "disk", None, None, "application/pdf", None, "Title",
            None, 0, 0, None, "pdftotext", None, None, None,
        )])

        with patch.object(server, "db", lambda: conn):
            out = server.ltm_get_document("7")
        self.assertTrue(out["found"])
        self.assertEqual(out["text"], "")
        self.assertFalse(out["truncated"])
        self.assertIs(out["ocr"], False)
        self.assertIsNone(out["doc_date"])
        self.assertIsNone(out["ingested_at"])
        self.assertEqual(out["path"], "/")
        self.assertEqual(out["doc_type"], "untyped")
        self.assertNotEqual(out["text"], "None")
        self.assertNotIn("None", json.dumps(out["text"]))


class AuthExactnessTests(unittest.TestCase):
    def test_a_token_prefix_does_not_authorize(self):
        prefix = TOKEN[:-1]
        app = App()
        gate = server._BearerGate(app, TOKEN)
        sent = run(gate, {
            "type": "http",
            "path": "/mcp",
            "headers": [(b"authorization", b"Bearer " + prefix.encode())],
        })
        self.assertEqual(status_of(sent), 401)
        self.assertEqual(app.scopes, [])

        sent = run(gate, {
            "type": "http",
            "path": "/t/" + prefix + "/mcp",
            "headers": [],
        })
        self.assertEqual(status_of(sent), 401)
        self.assertEqual(app.scopes, [])
        body = b"".join(
            msg.get("body", b"") for msg in sent if msg["type"] == "http.response.body")
        self.assertNotIn(TOKEN.encode(), body)

    def test_surrounding_whitespace_on_the_bearer_still_authorizes(self):
        app = App()
        sent = run(server._BearerGate(app, TOKEN), {
            "type": "http",
            "path": "/mcp",
            "headers": [(b"authorization", b"Bearer  " + TOKEN.encode() + b"  ")],
        })
        self.assertEqual(status_of(sent), 204)
        self.assertEqual(len(app.scopes), 1)
        self.assertEqual(app.scopes[0]["path"], "/mcp")

    def test_a_non_http_scope_is_forwarded_without_a_401(self):
        app = App()
        scope = {"type": "lifespan"}
        sent = run(server._BearerGate(app, TOKEN), scope)
        self.assertEqual(app.scopes, [scope])
        self.assertEqual(status_of(sent), 204)


if __name__ == "__main__":
    unittest.main()
