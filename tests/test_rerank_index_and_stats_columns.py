"""A bad rerank index falls back, and corpus sizes stay on the right tables.

No Postgres, Ollama, or rerank model. The HTTP server is never started.
"""

import json
import unittest
from unittest.mock import patch

import server


ALPHA = "ALPHA_COSINE_FIRST"
BETA = "BETA_SECOND_CHUNK"


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


def search_row(chunk_id, cosine, text):
    return (
        chunk_id, 10 + chunk_id, "lease", "Title", "disk", "/vault",
        "a.pdf", None, "pdftotext", cosine, text,
    )


class RerankIndexTests(unittest.TestCase):
    def tearDown(self):
        server._rr_proc = None

    def test_an_out_of_range_index_keeps_cosine_order(self):
        # The worker can return an index past the candidate list. That must
        # not crash the search, must not apply a partial reorder, and must
        # not put the exception text or either chunk into the note.
        rows = [
            search_row(1, 0.99126, ALPHA),
            search_row(2, 0.80, BETA),
        ]
        conn = ScriptedConn([("SELECT c.id", rows)])
        seen = {}

        def rerank(query, texts, top_k):
            seen["query"] = query
            seen["texts"] = list(texts)
            seen["top_k"] = top_k
            return [99, 0], [9.0, 1.0]

        with patch.multiple(
            server,
            db=lambda: conn,
            embed_query=lambda _text: "[0.100000]",
            hnsw_ready=lambda _con: False,
            rerank_available=lambda: True,
            run_rerank=rerank,
        ):
            out = server.ltm_search("find the clause", k=2, rerank=True)

        self.assertEqual(seen["query"], "find the clause")
        self.assertEqual(seen["texts"], [ALPHA, BETA])
        self.assertEqual(seen["top_k"], 2)
        select = [params for sql, params in conn.calls if "SELECT c.id" in sql]
        self.assertEqual(len(select), 1)
        self.assertEqual(select[0]["k"], server.RERANK_CANDIDATES)
        self.assertNotEqual(select[0]["k"], 2)
        self.assertFalse(any(sql.startswith("SET ") for sql, _params in conn.calls))

        self.assertEqual([hit["snippet"] for hit in out["results"]], [ALPHA, BETA])
        self.assertEqual([hit["chunk_id"] for hit in out["results"]], [1, 2])
        self.assertFalse(out["reranked"])
        self.assertNotIn("rerank_score", out["results"][0])
        self.assertEqual(out["results"][0]["cosine"], 0.9913)
        self.assertEqual(
            out["confidence"],
            "high (cosine only — enable rerank for a calibrated verdict)",
        )
        self.assertEqual(out["top_score"], {"cosine": 0.9913})
        self.assertEqual(
            out["note"],
            "rerank failed (IndexError) — cosine order only",
        )
        rendered = json.dumps(out)
        self.assertNotIn("list index", rendered)
        self.assertNotIn("9.0", rendered)
        self.assertNotIn(ALPHA, out["note"])
        self.assertNotIn(BETA, out["note"])


class StatsColumnTests(unittest.TestCase):
    def test_blank_types_are_grouped_and_sizes_stay_in_column_order(self):
        # Swapping pg_size_pretty columns would point an operator at the
        # wrong table. Blank doc types are grouped in SQL, not in Python.
        conn = ScriptedConn([
            ("SELECT count(*) FROM documents", [(3,)]),
            ("SELECT count(*) FROM doc_chunks", [(2,)]),
            ("SELECT count(*) FROM emb_mxbai", [(2,)]),
            ("SELECT COALESCE", [("untyped", 4)]),
            ("SELECT pg_size_pretty", [(
                "DOC-SIZE", "CHUNK-SIZE", "EMB-SIZE", "DB-SIZE",
            )]),
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

        by_type = [
            sql for sql, _params in conn.calls if sql.startswith("SELECT COALESCE")
        ]
        self.assertEqual(by_type, [
            "SELECT COALESCE(NULLIF(doc_type, ''), 'untyped'), count(*) "
            "FROM documents GROUP BY 1 ORDER BY 2 DESC",
        ])
        sizes = [
            sql for sql, _params in conn.calls if sql.startswith("SELECT pg_size_pretty")
        ]
        self.assertEqual(sizes, [
            "SELECT pg_size_pretty(pg_total_relation_size('documents')), "
            "pg_size_pretty(pg_total_relation_size('doc_chunks')), "
            "pg_size_pretty(pg_total_relation_size('emb_mxbai')), "
            "pg_size_pretty(pg_database_size(current_database()))",
        ])
        self.assertEqual(out["documents_by_type"], {"untyped": 4})
        self.assertNotIn("", out["documents_by_type"])
        self.assertEqual(out["table_sizes"], {
            "documents": "DOC-SIZE",
            "doc_chunks": "CHUNK-SIZE",
            "emb_mxbai": "EMB-SIZE",
            "database": "DB-SIZE",
        })
        self.assertEqual(
            out["reranker"],
            "unavailable (set LTM_RERANK_PYTHON / LTM_RERANK_SCRIPT)",
        )
        self.assertIs(out["read_only"], True)
        rendered = json.dumps(out)
        self.assertNotIn("SECRET-PROBE-MESSAGE", rendered)
        self.assertNotIn("DOC-SIZE", by_type[0])
        self.assertNotIn("untyped", sizes[0])
