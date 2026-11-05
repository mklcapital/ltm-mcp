"""Blank ids, LIKE metacharacters, rerank confidence, and coverage rounding.

No Postgres, Ollama, or rerank model. The HTTP server is never started.
"""

import json
import unittest
from unittest.mock import patch

import server


SECRET = "FULL_TEXT_SHOULD_NOT_LEAK"


def doc_row(text, doc_id=7, relpath="Taxes/a.pdf"):
    return (
        doc_id, "disk", "/vault", relpath, "application/pdf", "lease", "Lease",
        None, 100, 1, len(text), "pdftotext", 0, None, text,
    )


class ScriptedConn:
    """First matching rule wins. A later rule is the leak we must not reach."""

    def __init__(self, rules):
        self.rules = rules
        self.calls = []

    def run(self, sql, **params):
        self.calls.append((sql, params))
        for needle, rows in self.rules:
            if needle in sql:
                return rows
        return []


class BlankLookupTests(unittest.TestCase):
    def test_whitespace_only_id_is_a_blank_lookup_not_a_space_fragment(self):
        # strip() runs before the lookup. Three spaces must not be searched
        # as a fragment, and the empty result must not return document text.
        conn = ScriptedConn([
            ("relpath = :v", []),
            ("ILIKE", [doc_row(SECRET, 1, "a.pdf"), doc_row(SECRET, 2, "b.pdf")]),
        ])
        with patch.object(server, "db", lambda: conn):
            out = server.ltm_get_document("   \t  ")
        self.assertFalse(out["found"])
        self.assertTrue(out.get("ambiguous"))
        self.assertNotIn("text", out)
        self.assertNotIn(SECRET, json.dumps(out))
        self.assertEqual(len(conn.calls), 2)
        exact_sql, exact_params = conn.calls[0]
        self.assertIn("relpath = :v", exact_sql)
        self.assertNotIn("WHERE id = :v", exact_sql)
        self.assertEqual(exact_params["v"], "")
        self.assertNotIsInstance(exact_params["v"], int)
        frag_sql, frag_params = conn.calls[1]
        self.assertIn("ILIKE", frag_sql)
        self.assertEqual(frag_params["v"], "%%")
        self.assertNotIn("   ", frag_sql)
        self.assertNotIn("\t", frag_sql)


class LikeMetacharacterTests(unittest.TestCase):
    def test_percent_and_underscore_stay_bound_parameters(self):
        # `%` and `_` are LIKE metacharacters. They stay inside the bound
        # value. The SQL text must not gain them by interpolation.
        conn = ScriptedConn([
            ("relpath = :v", []),
            ("ILIKE", []),
        ])
        with patch.object(server, "db", lambda: conn):
            out = server.ltm_get_document("100%_")
        self.assertFalse(out["found"])
        self.assertNotIn("ambiguous", out)
        self.assertNotIn(SECRET, json.dumps(out))
        exact_sql, exact_params = conn.calls[0]
        self.assertEqual(exact_params["v"], "100%_")
        self.assertNotIn("100%", exact_sql)
        frag_sql, frag_params = conn.calls[1]
        self.assertEqual(frag_params["v"], "%100%_%")
        self.assertNotIn("100%", frag_sql)
        self.assertNotIn("ILIKE '%", frag_sql)


class RerankTopScoreTests(unittest.TestCase):
    def test_confidence_follows_the_first_rerank_score_not_the_max(self):
        # A later 9.0 must not upgrade a top score of 1.0 to high.
        rows = [
            (1, 10, "lease", "A", "disk", "/vault", "a.pdf", None, "pdftotext", 0.9, "FIRST SNIPPET"),
            (2, 11, "lease", "B", "disk", "/vault", "b.pdf", None, "pdftotext", 0.99, "SECOND SNIPPET"),
        ]
        conn = ScriptedConn([("SELECT c.id", rows)])

        def rerank(query, texts, top_k):
            self.assertEqual(query, "q")
            self.assertEqual(texts, ["FIRST SNIPPET", "SECOND SNIPPET"])
            self.assertEqual(top_k, 2)
            return [0, 1], [1.0, 9.0]

        with patch.multiple(
            server,
            db=lambda: conn,
            embed_query=lambda _text: "[0.100000]",
            hnsw_ready=lambda _con: False,
            rerank_available=lambda: True,
            run_rerank=rerank,
        ):
            out = server.ltm_search("q", k=2, rerank=True)

        self.assertTrue(out["confidence"].startswith("LOW"), out["confidence"])
        self.assertNotIn("cosine only", out["confidence"])
        self.assertEqual(out["top_score"], {"rerank_score": 1.0})
        self.assertEqual(out["reranked"], True)
        self.assertEqual(
            [item["rerank_score"] for item in out["results"]],
            [1.0, 9.0],
        )
        self.assertEqual(out["results"][0]["snippet"], "FIRST SNIPPET")
        self.assertEqual(out["results"][1]["snippet"], "SECOND SNIPPET")
        self.assertFalse(any("SET " in sql for sql, _params in conn.calls))


class CoverageRoundingTests(unittest.TestCase):
    def test_one_embedding_in_three_chunks_is_33_33_percent(self):
        # Integer division would report 0% and hide two missing embeddings.
        conn = ScriptedConn([
            ("SELECT count(*) FROM documents", [[4]]),
            ("count(*) FROM doc_chunks", [[3]]),
            ("count(*) FROM emb_mxbai", [[1]]),
            ("GROUP BY", [("lease", 4)]),
            ("pg_size_pretty", [("1 MB", "2 MB", "3 MB", "4 MB")]),
        ])

        class Resp:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return json.dumps({"models": [{"name": "other"}]}).encode()

        with patch.object(server, "db", lambda: conn), \
             patch.object(server, "hnsw_ready", lambda _con: False), \
             patch.object(server.urllib.request, "urlopen", lambda *args, **kwargs: Resp()):
            out = server.ltm_stats()

        self.assertEqual(out["embedding_coverage"], "33.33%")
        self.assertEqual(out["documents"], 4)
        self.assertEqual(out["chunks"], 3)
        self.assertEqual(out["mxbai_embeddings"], 1)
        self.assertNotIn(SECRET, json.dumps(out))


if __name__ == "__main__":
    unittest.main()
