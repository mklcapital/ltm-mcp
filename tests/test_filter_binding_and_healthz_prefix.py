"""Filter values stay bound, and /healthz is open only as an exact path.

No Postgres, Ollama, or rerank model. The HTTP server is never started.
"""

import asyncio
import json
import unittest
from unittest.mock import patch

import server


TOKEN = "unit-test-bearer"
EVIL_TYPE = "lease' OR '1'='1"
EVIL_PATH = "Taxes'; DROP TABLE documents;--"


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


class ScriptedConn:
    def __init__(self):
        self.calls = []

    def run(self, sql, **params):
        self.calls.append((sql, params))
        return []


class FilterBindingTests(unittest.TestCase):
    def test_quotes_and_statement_text_stay_in_parameters(self):
        conn = ScriptedConn()
        with patch.multiple(
            server,
            db=lambda: conn,
            embed_query=lambda _text: "[0.100000]",
            hnsw_ready=lambda _con: False,
        ):
            out = server.ltm_search(
                "rent roll",
                k=4,
                rerank=False,
                doc_type=EVIL_TYPE,
                path_contains=EVIL_PATH,
            )

        rendered = json.dumps(out)
        self.assertEqual(out["query"], "rent roll")
        self.assertEqual(out["results"], [])
        self.assertIn("no matches", out["confidence"])
        self.assertNotIn(EVIL_TYPE, rendered)
        self.assertNotIn(EVIL_PATH, rendered)
        self.assertNotIn("DROP", rendered)

        selects = [(sql, params) for sql, params in conn.calls if "SELECT c.id" in sql]
        self.assertEqual(len(selects), 1)
        sql, params = selects[0]
        self.assertIn("d.doc_type ILIKE :dt", sql)
        self.assertIn("ILIKE :pc", sql)
        self.assertIn(" AND ", sql)
        self.assertNotIn(" OR ", sql)
        self.assertNotIn(EVIL_TYPE, sql)
        self.assertNotIn(EVIL_PATH, sql)
        self.assertNotIn("DROP", sql)
        self.assertEqual(params["dt"], EVIL_TYPE)
        self.assertEqual(params["pc"], "%" + EVIL_PATH + "%")
        self.assertEqual(params["k"], 4)
        self.assertFalse(any(sql.startswith("SET ") for sql, _params in conn.calls))


class HealthzPrefixTests(unittest.TestCase):
    def test_query_string_survives_the_path_token_rewrite(self):
        app = App()
        gate = server._BearerGate(app, TOKEN)
        query = b"q=1&token=not-the-bearer"
        scope = {
            "type": "http",
            "path": "/t/%s/mcp" % TOKEN,
            "query_string": query,
            "headers": [],
        }
        sent = run(gate, scope)
        self.assertEqual(status_of(sent), 204)
        self.assertEqual(scope["path"], "/t/%s/mcp" % TOKEN)
        seen = app.scopes[0]
        self.assertEqual(seen["path"], "/mcp")
        self.assertEqual(seen["raw_path"], b"/mcp")
        self.assertEqual(seen["query_string"], query)
        self.assertNotIn(TOKEN, seen["path"])
        self.assertNotIn(TOKEN.encode(), seen["raw_path"])
        self.assertNotIn(TOKEN.encode(), seen["query_string"])

    def test_a_trailing_bearer_word_does_not_authorize(self):
        app = App()
        gate = server._BearerGate(app, TOKEN)
        sent = run(gate, {
            "type": "http",
            "path": "/mcp",
            "headers": [(b"authorization", b"Bearer " + TOKEN.encode() + b" extra")],
        })
        self.assertEqual(status_of(sent), 401)
        self.assertEqual(body_of(sent), b'{"error":"unauthorized"}')
        self.assertNotIn(TOKEN.encode(), body_of(sent))
        self.assertEqual(app.scopes, [])

    def test_healthz_under_the_token_prefix_is_not_the_open_probe(self):
        app = App()
        gate = server._BearerGate(app, TOKEN)

        denied = run(gate, {"type": "http", "path": "/t/nope/healthz", "headers": []})
        self.assertEqual(status_of(denied), 401)
        self.assertNotEqual(body_of(denied), b"ok")
        self.assertNotIn(TOKEN.encode(), body_of(denied))
        self.assertEqual(app.scopes, [])

        opened = run(gate, {
            "type": "http",
            "path": "/healthz",
            "headers": [(b"authorization", b"Bearer wrong")],
        })
        self.assertEqual(status_of(opened), 200)
        self.assertEqual(body_of(opened), b"ok")
        self.assertEqual(app.scopes, [])

        forwarded = run(gate, {
            "type": "http",
            "path": "/t/%s/healthz" % TOKEN,
            "headers": [],
        })
        self.assertEqual(status_of(forwarded), 204)
        self.assertEqual(body_of(forwarded), b"")
        self.assertEqual(app.scopes[0]["path"], "/healthz")
        self.assertNotIn(TOKEN, app.scopes[0]["path"])


if __name__ == "__main__":
    unittest.main()
