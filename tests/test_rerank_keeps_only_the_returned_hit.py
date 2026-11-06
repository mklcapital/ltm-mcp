"""Rerank may score a wide candidate pool, but only k hits are returned.

No Postgres, Ollama, or rerank model. The HTTP server is never started.
"""

import json
import unittest
from unittest.mock import patch

import server


LOSER = "LOSER_CHUNK_MUST_NOT_LEAK"
OTHER = "OTHER_CHUNK_MUST_NOT_LEAK"
WINNER = "WINNER_SNIPPET"


class ScriptedConn:
    def __init__(self, rules):
        self.rules = rules
        self.calls = []

    def run(self, sql, **params):
        self.calls.append((sql, params))
        for needle, rows in self.rules:
            if needle in sql:
                return rows
        return []


class RerankSliceTests(unittest.TestCase):
    def test_only_the_reordered_winner_is_returned(self):
        # The winner is the worst cosine hit. Slicing to k before rerank would
        # drop it, and returning the whole pool would leak the other chunks.
        rows = [
            (1, 10, "lease", "A", "disk", "/vault", "a.pdf", None, "pdftotext", 0.99, LOSER),
            (2, 11, "lease", "B", "disk", "/vault", "b.pdf", None, "pdftotext", 0.98, OTHER),
            (3, 12, "lease", "C", "disk", "/vault", "c.pdf", None, "pdftotext", 0.10, WINNER),
        ]
        conn = ScriptedConn([("SELECT c.id", rows)])

        def rerank(query, texts, top_k):
            self.assertEqual(query, "find the clause")
            self.assertEqual(texts, [LOSER, OTHER, WINNER])
            self.assertEqual(top_k, 1)
            return [2, 0, 1], [9.0, 1.0, 0.5]

        with patch.multiple(
            server,
            db=lambda: conn,
            embed_query=lambda _text: "[0.100000]",
            hnsw_ready=lambda _con: False,
            rerank_available=lambda: True,
            run_rerank=rerank,
        ):
            out = server.ltm_search("find the clause", k=1, rerank=True)

        self.assertEqual(len(out["results"]), 1)
        hit = out["results"][0]
        self.assertEqual(hit["chunk_id"], 3)
        self.assertEqual(hit["snippet"], WINNER)
        self.assertEqual(hit["rerank_score"], 9.0)
        self.assertEqual(hit["cosine"], 0.1)
        self.assertEqual(out["confidence"], "high")
        self.assertEqual(out["top_score"], {"rerank_score": 9.0})
        self.assertTrue(out["reranked"])
        rendered = json.dumps(out)
        self.assertNotIn(LOSER, rendered)
        self.assertNotIn(OTHER, rendered)
        select = [params for sql, params in conn.calls if "SELECT c.id" in sql]
        self.assertEqual(len(select), 1)
        self.assertEqual(select[0]["k"], server.RERANK_CANDIDATES)
        self.assertNotEqual(select[0]["k"], 1)
        self.assertFalse(any(sql.startswith("SET ") for sql, _params in conn.calls))


if __name__ == "__main__":
    unittest.main()
