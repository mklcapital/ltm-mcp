"""Search filters keep surrounding spaces, and a case-flipped token is rejected.

No Postgres, Ollama, or rerank model. The HTTP server is never started.
"""

import asyncio
import unittest
from unittest.mock import patch

import server


TOKEN = "unit-test-bearer"


class ScriptedConn:
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


def body_of(sent):
    return b"".join(
        msg.get("body", b"") for msg in sent if msg["type"] == "http.response.body")


class FilterWhitespaceTests(unittest.TestCase):
    def test_search_filters_keep_surrounding_spaces(self):
        # get_document strips its path. Search filters must not: trimming
        # " lease " to "lease" would match a different set of documents.
        conn = ScriptedConn([])
        with patch.multiple(
            server,
            db=lambda: conn,
            embed_query=lambda _text: "[0.100000]",
            hnsw_ready=lambda _con: False,
            rerank_available=lambda: False,
        ):
            out = server.ltm_search(
                "taxes",
                k=1,
                rerank=False,
                doc_type=" lease ",
                path_contains=" Taxes ",
            )

        select_sql, select = next(
            (sql, params) for sql, params in conn.calls if sql.startswith("SELECT"))
        self.assertEqual(select["dt"], " lease ")
        self.assertNotEqual(select["dt"], "lease")
        self.assertEqual(select["pc"], "% Taxes %")
        self.assertNotIn("%Taxes%", select["pc"])
        self.assertIn(" AND ", select_sql)
        self.assertNotIn(" OR ", select_sql)
        self.assertEqual(out["results"], [])
        self.assertIn("no matches", out["confidence"])
        self.assertNotIn("lease", out["confidence"])


class TokenCaseTests(unittest.TestCase):
    def test_a_case_flipped_token_does_not_authorize(self):
        app = App()
        gate = server._BearerGate(app, TOKEN)
        folded = TOKEN.swapcase()
        self.assertNotEqual(folded, TOKEN)
        cases = (
            ("/mcp", [(b"authorization", b"Bearer " + folded.encode())]),
            ("/t/" + folded + "/mcp", []),
            ("/healthz", [(b"authorization", b"Bearer " + folded.encode())]),
        )
        for path, headers in cases:
            app.scopes.clear()
            sent = run(gate, {
                "type": "http",
                "path": path,
                "headers": headers,
                "query_string": b"token=" + TOKEN.encode(),
            })
            body = body_of(sent)
            self.assertNotIn(TOKEN.encode(), body, path)
            self.assertNotIn(folded.encode(), body, path)
            if path == "/healthz":
                self.assertEqual(status_of(sent), 200, path)
                self.assertEqual(body, b"ok")
                self.assertEqual(app.scopes, [], path)
            else:
                self.assertEqual(status_of(sent), 401, path)
                self.assertEqual(body, b'{"error":"unauthorized"}')
                self.assertEqual(app.scopes, [], path)


if __name__ == "__main__":
    unittest.main()
