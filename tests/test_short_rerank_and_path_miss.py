"""A short rerank order is the whole result, and a total path miss returns no text.

No Postgres, Ollama, or rerank model. The HTTP server is never started.
"""

import json
import unittest
from unittest.mock import patch

import server


SNIPPET_A = "KEEP-COSINE-FIRST-SHOULD-DROP"
SNIPPET_B = "RERANK-ONLY-WINNER"
MISSING = "Taxes/missing.pdf"


class ScriptedConn:
    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    def run(self, sql, **params):
        self.calls.append((sql, params))
        if "SELECT c.id" in sql:
            return self.rows
        return []


def search_row(snippet, doc_type, cosine, chunk_id):
    return (
        chunk_id, 7, doc_type, "Title", "disk", "", "", None,
        "pdftotext", cosine, snippet,
    )


class ShortRerankTests(unittest.TestCase):
    def test_a_short_rerank_order_does_not_backfill_cosine_hits(self):
        # The worker's order is the result set. Omitting an index must not
        # append that leftover hit, and an empty doc_type still displays as
        # untyped rather than an empty string.
        rows = [
            search_row(SNIPPET_A, "lease", 0.91, 11),
            search_row(SNIPPET_B, "", 0.70, 22),
        ]
        conn = ScriptedConn(rows)
        seen = {}

        def rerank(query, texts, top_k):
            seen["rerank"] = (query, list(texts), top_k)
            return [1], [4.0]

        with patch.multiple(
            server,
            db=lambda: conn,
            embed_query=lambda _text: "[0.700000]",
            hnsw_ready=lambda _con: False,
            rerank_available=lambda: True,
            run_rerank=rerank,
        ):
            out = server.ltm_search("find the clause", k=2, rerank=True)

        self.assertEqual(seen["rerank"][0], "find the clause")
        self.assertEqual(seen["rerank"][1], [SNIPPET_A, SNIPPET_B])
        self.assertEqual(seen["rerank"][2], 2)
        self.assertTrue(out["reranked"])
        self.assertNotIn("note", out)
        self.assertEqual(out["confidence"], "medium")
        self.assertEqual(out["top_score"], {"rerank_score": 4.0})
        self.assertEqual(len(out["results"]), 1)
        hit = out["results"][0]
        self.assertEqual(hit["snippet"], SNIPPET_B)
        self.assertEqual(hit["chunk_id"], 22)
        self.assertEqual(hit["doc_type"], "untyped")
        self.assertEqual(hit["path"], "/")
        self.assertEqual(hit["rerank_score"], 4.0)
        self.assertEqual(hit["title"], "Title")
        rendered = json.dumps(out)
        self.assertNotIn(SNIPPET_A, rendered)
        self.assertNotIn('"lease"', rendered)
        select = [params for sql, params in conn.calls if "SELECT c.id" in sql]
        self.assertEqual(len(select), 1)
        self.assertEqual(select[0]["k"], server.RERANK_CANDIDATES)
        self.assertNotEqual(select[0]["k"], 2)
        self.assertFalse(any(sql.startswith("SET ") for sql, _params in conn.calls))


class PathMissTests(unittest.TestCase):
    def test_a_total_path_miss_returns_no_text(self):
        conn = ScriptedConn([])
        with patch.object(server, "db", lambda: conn):
            out = server.ltm_get_document(MISSING)

        self.assertEqual(
            out,
            {"found": False,
             "error": "no document matches '%s' (id, path, or fragment)" % MISSING},
        )
        self.assertNotIn("text", out)
        self.assertNotIn("ambiguous", out)
        self.assertEqual(len(conn.calls), 2)
        exact_sql, exact_params = conn.calls[0]
        self.assertIn("WHERE relpath = :v", exact_sql)
        self.assertNotIn("ILIKE", exact_sql)
        self.assertEqual(exact_params["v"], MISSING)
        frag_sql, frag_params = conn.calls[1]
        self.assertIn("ILIKE :v", frag_sql)
        self.assertIn("ORDER BY char_count DESC NULLS LAST LIMIT 10", frag_sql)
        self.assertEqual(frag_params["v"], "%" + MISSING + "%")
        self.assertNotIn(MISSING, frag_sql)


if __name__ == "__main__":
    unittest.main()
