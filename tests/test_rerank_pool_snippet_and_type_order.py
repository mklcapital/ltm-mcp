"""Rerank that cannot run must not widen the fetch, snippets drop
whitespace before the cap, and corpus-type counts keep query order.

No Postgres, Ollama, or rerank model. The HTTP server is never started.
"""

import unittest
from unittest.mock import patch

import server


class ScriptedConn:
    def __init__(self, rows_for_sql):
        self.rows_for_sql = rows_for_sql
        self.calls = []

    def run(self, sql, **params):
        self.calls.append((sql, params))
        for prefix, rows in self.rows_for_sql:
            if sql.startswith(prefix):
                return rows
        if sql.startswith("SELECT"):
            return []
        return []


class UnavailableRerankTests(unittest.TestCase):
    def test_unavailable_rerank_does_not_fetch_the_candidate_pool(self):
        # rerank=True with no worker must search at k, not the 50-candidate
        # pool. Widening the fetch pulls extra chunk text for a rerank that
        # will not happen.
        row = (
            11, 22, "lease", "Title", "disk", "/vault", "a.pdf",
            None, "pdftotext", 0.8, "snippet body",
        )
        conn = ScriptedConn([("SELECT", [row])])

        def fail_rerank(*_args):
            raise AssertionError("rerank worker must not be called")

        with patch.multiple(
            server,
            db=lambda: conn,
            embed_query=lambda _text: "[0.100000]",
            hnsw_ready=lambda _con: False,
            rerank_available=lambda: False,
            run_rerank=fail_rerank,
        ):
            out = server.ltm_search("taxes", k=4, rerank=True)

        select_sql, select = next(
            (sql, params) for sql, params in conn.calls if sql.startswith("SELECT"))
        self.assertEqual(select["k"], 4)
        self.assertNotEqual(select["k"], server.RERANK_CANDIDATES)
        self.assertNotIn("WHERE", select_sql)
        self.assertFalse(any(sql.startswith("SET") for sql, _params in conn.calls))
        self.assertFalse(out["reranked"])
        self.assertIn("reranker unavailable", out["note"])
        self.assertNotIn("rerank_score", out["results"][0])
        self.assertEqual(out["results"][0]["snippet"], "snippet body")
        self.assertEqual(len(conn.calls), 1)


class SnippetBudgetTests(unittest.TestCase):
    def test_snippet_budget_starts_after_whitespace_collapses(self):
        # Leading newlines must not consume the snippet cap. The model only
        # sees this prefix; collapsing after the slice would hide the text.
        text = ("\n" * 80) + "SENTINEL" + ("x" * 400)
        out = server.clean_snippet(text, limit=20)
        self.assertEqual(out, ("SENTINEL" + ("x" * 400))[:20])
        self.assertTrue(out.startswith("SENTINEL"))
        self.assertNotIn("\n", out)
        self.assertEqual(len(out), 20)


class TypeOrderTests(unittest.TestCase):
    def test_type_counts_keep_the_query_order(self):
        # appraisal is the larger count and the earlier name. The SQL already
        # ordered by count. Python must not re-sort alphabetically or by count.
        conn = ScriptedConn([
            ("SELECT count(*) FROM documents", [(4,)]),
            ("SELECT count(*) FROM doc_chunks", [(4,)]),
            ("SELECT count(*) FROM emb_mxbai", [(4,)]),
            ("SELECT COALESCE", [("tax return", 2), ("appraisal", 9)]),
            ("SELECT pg_size_pretty", [("1 MB", "2 MB", "3 MB", "9 MB")]),
        ])

        def urlopen(_url, timeout=0):
            raise TimeoutError("offline")

        with patch.multiple(
            server,
            db=lambda: conn,
            hnsw_ready=lambda _con: False,
            rerank_available=lambda: False,
        ), patch.object(server.urllib.request, "urlopen", urlopen):
            out = server.ltm_stats()

        self.assertEqual(
            list(out["documents_by_type"]),
            ["tax return", "appraisal"],
        )
        self.assertEqual(out["documents_by_type"]["appraisal"], 9)
        self.assertEqual(out["documents_by_type"]["tax return"], 2)
        grouped = next(sql for sql, _params in conn.calls if sql.startswith("SELECT COALESCE"))
        self.assertIn("ORDER BY 2 DESC", grouped)
        self.assertNotIn("appraisal", grouped)
        self.assertNotIn("tax return", grouped)


if __name__ == "__main__":
    unittest.main()
