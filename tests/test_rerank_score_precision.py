"""Rerank confidence uses the raw score, not a rounded cosine-style display.

No Postgres, Ollama, or rerank model.
"""

import unittest
from unittest.mock import patch

import server


class ScriptedConn:
    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    def run(self, sql, **params):
        self.calls.append((sql, params))
        if sql.lstrip().upper().startswith("SELECT"):
            return self.rows
        return []


class RerankScorePrecisionTests(unittest.TestCase):
    def test_a_score_just_below_high_stays_medium(self):
        # round(5.99995, 4) is 6.0, which is the high gate. Rounding before
        # the comparison would report a calibrated high verdict for a medium hit.
        raw_cosine = 0.85495
        score = 5.99995
        self.assertEqual(round(score, 4), 6.0)
        self.assertLess(score, server.HIGH_RERANK)
        self.assertGreaterEqual(score, server.LOW_RERANK)
        row = (
            1, 7, "lease", "TITLE", "SRC", "/ROOT", "a.pdf",
            None, "pdftotext", raw_cosine, "SNIPPET-SENTINEL",
        )
        conn = ScriptedConn([row])

        def run_rerank(query, texts, top_k):
            self.assertEqual(query, "lease renewal")
            self.assertEqual(texts, ["SNIPPET-SENTINEL"])
            self.assertEqual(top_k, 8)
            return [0], [score]

        with patch.object(server, "db", lambda: conn), \
             patch.object(server, "embed_query", return_value="[0.100000]"), \
             patch.object(server, "hnsw_ready", return_value=False), \
             patch.object(server, "rerank_available", return_value=True), \
             patch.object(server, "run_rerank", side_effect=run_rerank):
            out = server.ltm_search("lease renewal", k=8, rerank=True)

        self.assertEqual(out["confidence"], "medium")
        self.assertNotIn("cosine only", out["confidence"])
        self.assertNotIn("high", out["confidence"])
        self.assertNotIn("SNIPPET-SENTINEL", out["confidence"])
        item = out["results"][0]
        self.assertEqual(item["rerank_score"], score)
        self.assertNotEqual(item["rerank_score"], round(score, 4))
        self.assertEqual(out["top_score"], {"rerank_score": score})
        self.assertEqual(item["cosine"], round(raw_cosine, 4))
        self.assertNotEqual(item["cosine"], raw_cosine)
        self.assertEqual(item["snippet"], "SNIPPET-SENTINEL")
        self.assertNotIn("note", out)


if __name__ == "__main__":
    unittest.main()
