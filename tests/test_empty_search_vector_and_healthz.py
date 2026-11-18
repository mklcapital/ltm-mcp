"""Empty-search confidence, a hostile embed vector, and an open health check.

No Postgres, Ollama, or rerank model. The HTTP server is never started.
"""

import json
import unittest
from unittest.mock import patch

import server


DROP = "[1.0); DROP TABLE emb_mxbai;--"
SNIPPET = "CLAUSE-STAYS"


class ScriptedConn:
    def __init__(self, script):
        self.script = script
        self.calls = []

    def run(self, sql, **params):
        self.calls.append((sql, params))
        for prefix, rows in self.script:
            if sql.startswith(prefix):
                return rows
        if sql.startswith("SET "):
            return []
        raise AssertionError(sql)


def search_row():
    return (
        7, 22, "lease", "Title", "disk", "/vault", "a.pdf", None,
        "pdftotext", 0.90, SNIPPET,
    )


class EmptySearchTests(unittest.TestCase):
    def test_no_rows_uses_the_narrow_filter_confidence(self):
        # The model is told to stop when nothing matched. The empty-result
        # index string is the short one: it must not grow the "slower" or
        # "fast approximate search" wording used on other paths.
        unfiltered = ScriptedConn([("SELECT c.id", [])])
        filtered = ScriptedConn([("SELECT c.id", [])])
        rerank_calls = []

        def rerank(*args):
            rerank_calls.append(args)
            raise AssertionError("rerank must not run when there are no rows")

        with patch.multiple(
            server,
            db=lambda: unfiltered,
            embed_query=lambda _text: "[0.100000]",
            hnsw_ready=lambda _con: False,
            rerank_available=lambda: True,
            run_rerank=rerank,
        ):
            empty = server.ltm_search("where is the lease", k=4, rerank=True)

        self.assertEqual(empty["query"], "where is the lease")
        self.assertEqual(empty["results"], [])
        self.assertEqual(
            empty["confidence"],
            "LOW — no matches (filters may be too narrow)",
        )
        self.assertEqual(empty["hnsw_index"], "not built yet (seq scan)")
        self.assertNotIn("slower", empty["hnsw_index"])
        self.assertNotIn("ready", empty["hnsw_index"])
        self.assertNotIn("note", empty)
        self.assertEqual(rerank_calls, [])
        self.assertFalse(any(sql.startswith("SET ") for sql, _p in unfiltered.calls))

        with patch.multiple(
            server,
            db=lambda: filtered,
            embed_query=lambda _text: "[0.100000]",
            hnsw_ready=lambda _con: True,
            rerank_available=lambda: True,
            run_rerank=rerank,
        ):
            narrowed = server.ltm_search(
                "where is the lease", k=4, rerank=True,
                doc_type="lease", path_contains="Taxes",
            )

        self.assertEqual(narrowed["results"], [])
        self.assertEqual(
            narrowed["confidence"],
            "LOW — no matches (filters may be too narrow)",
        )
        self.assertEqual(narrowed["hnsw_index"], "ready")
        self.assertNotIn("sequential", narrowed["hnsw_index"])
        self.assertNotIn("10 s", narrowed["hnsw_index"])
        select = [c for c in filtered.calls if c[0].startswith("SELECT c.id")]
        self.assertEqual(len(select), 1)
        sql, params = select[0]
        self.assertIn(" AND ", sql)
        self.assertNotIn(" OR ", sql)
        self.assertEqual(params["dt"], "lease")
        self.assertEqual(params["pc"], "%Taxes%")
        self.assertNotIn("lease", sql)
        self.assertNotIn("Taxes", sql)
        sets = [sql for sql, _params in filtered.calls if sql.startswith("SET ")]
        self.assertEqual(sets, [
            "SET hnsw.ef_search = 200",
            "SET hnsw.iterative_scan = relaxed_order",
        ])
        self.assertEqual(rerank_calls, [])


class EmbedVectorTests(unittest.TestCase):
    def test_embed_vector_is_bound_and_not_spliced_into_sql(self):
        conn = ScriptedConn([("SELECT c.id", [search_row()])])

        with patch.multiple(
            server,
            db=lambda: conn,
            embed_query=lambda _text: DROP,
            hnsw_ready=lambda _con: False,
            rerank_available=lambda: False,
        ):
            out = server.ltm_search("clause", k=1, rerank=False)

        select = [c for c in conn.calls if "SELECT c.id" in c[0]]
        self.assertEqual(len(select), 1)
        sql, params = select[0]
        self.assertIn("CAST(:e AS vector)", sql)
        self.assertNotIn("DROP", sql)
        self.assertNotIn("emb_mxbai;--", sql)
        self.assertEqual(params["e"], DROP)
        self.assertEqual(params["k"], 1)
        self.assertEqual(out["results"][0]["snippet"], SNIPPET)
        rendered = json.dumps(out)
        self.assertNotIn("DROP TABLE", rendered)
        self.assertFalse(any(sql.startswith("SET ") for sql, _p in conn.calls))


class HealthzBearerTests(unittest.IsolatedAsyncioTestCase):
    async def test_wrong_bearer_on_healthz_stays_open_and_echoes_nothing(self):
        # /healthz is checked before auth. A stale or hostile bearer, and a
        # token sitting in the query string, must not change the body.
        seen = {}

        async def app(scope, receive, send):
            seen["called"] = scope.get("path")

        gate = server._BearerGate(app, "real-token-value")
        messages = []

        async def send(msg):
            messages.append(msg)

        async def receive():
            return {"type": "http.request", "body": b""}

        scope = {
            "type": "http",
            "path": "/healthz",
            "headers": [(b"authorization", b"Bearer wrong-secret")],
            "query_string": b"token=real-token-value",
        }
        await gate(scope, receive, send)

        self.assertNotIn("called", seen)
        self.assertEqual(messages[0]["status"], 200)
        self.assertEqual(messages[1]["body"], b"ok")
        blob = messages[1]["body"]
        for header in messages[0]["headers"]:
            blob += b" ".join(header)
        self.assertNotIn(b"wrong-secret", blob)
        self.assertNotIn(b"real-token-value", blob)
        self.assertNotIn(b"Bearer", blob)
