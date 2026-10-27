"""Wide document ids stay exact, and confidence uses the raw gate.

No Postgres, Ollama, or rerank model. The HTTP server is never started.
"""

import json
import unittest
from unittest.mock import patch

import server


SECRET = "FULL_TEXT_SHOULD_NOT_LEAK"
SNIPPET = "quarterly lease snippet"


def doc_row(text, doc_id=7, relpath="Taxes/a.pdf"):
    return (
        doc_id, "disk", "/vault", relpath, "application/pdf", "lease", "Lease",
        None, 100, 1, len(text), "pdftotext", 0, None, text,
    )


def search_row(cosine, text=SNIPPET):
    return (
        1, 7, "lease", "Lease", "disk", "/vault", "Taxes/a.pdf",
        None, "pdftotext", cosine, text,
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


class WideDocumentIdTests(unittest.TestCase):
    def test_a_digit_string_past_int64_does_not_alias_another_document(self):
        # int() is unbounded. Truncating to 64 bits, or giving up and fragment-
        # searching, would return a different document's text.
        key = "9" * 40
        full = int(key)
        self.assertGreater(full, 2**63 - 1)

        def run(sql, **params):
            conn.calls.append((sql, params))
            if "WHERE id = :v" in sql:
                if params.get("v") != full:
                    return [doc_row(SECRET, doc_id=7)]
                return []
            if "ILIKE" in sql:
                return [doc_row(SECRET, doc_id=7)]
            return []

        conn = ScriptedConn([])
        conn.run = run
        with patch.object(server, "db", lambda: conn):
            out = server.ltm_get_document(key)

        self.assertFalse(out["found"])
        self.assertNotIn("ambiguous", out)
        self.assertNotIn(SECRET, json.dumps(out))
        self.assertEqual(len(conn.calls), 1)
        exact_sql, exact_params = conn.calls[0]
        self.assertIn("WHERE id = :v", exact_sql)
        self.assertNotIn("ILIKE", exact_sql)
        self.assertNotIn(key, exact_sql)
        self.assertEqual(exact_params["v"], full)
        self.assertIsInstance(exact_params["v"], int)
        self.assertNotEqual(exact_params["v"], 7)
        self.assertNotEqual(exact_params["v"], 9)


class ConfidenceEdgeTests(unittest.TestCase):
    def search(self, cosine, rerank_score=None):
        conn = ScriptedConn([("SELECT", [search_row(cosine)])])

        def embed(_text):
            return "[0.100000]"

        patches = {
            "db": lambda: conn,
            "embed_query": embed,
            "hnsw_ready": lambda _con: False,
            "rerank_available": lambda: rerank_score is not None,
        }
        if rerank_score is not None:
            def fake_rerank(_query, _texts, _top_k):
                return [0], [rerank_score]
            patches["run_rerank"] = fake_rerank
        with patch.multiple(server, **patches):
            return server.ltm_search(
                "taxes", k=4, rerank=rerank_score is not None)

    def test_exact_cosine_thresholds_use_the_raw_score(self):
        # The gate is `>=`, on the raw score. Rounding is display-only.
        low = self.search(server.LOW_COSINE)
        self.assertEqual(
            low["confidence"],
            "medium (cosine only — enable rerank for a calibrated verdict)",
        )
        self.assertEqual(low["top_score"]["cosine"], round(server.LOW_COSINE, 4))
        self.assertEqual(low["results"][0]["snippet"], SNIPPET)
        self.assertNotIn("LOW", low["confidence"])

        high = self.search(server.HIGH_COSINE)
        self.assertEqual(
            high["confidence"],
            "high (cosine only — enable rerank for a calibrated verdict)",
        )
        self.assertEqual(high["results"][0]["snippet"], SNIPPET)
        self.assertNotIn(SECRET, json.dumps(high))

    def test_rerank_score_overrides_cosine_at_the_boundary(self):
        # A strong cosine must not upgrade a weak rerank, and the reverse.
        high = self.search(0.1, rerank_score=server.HIGH_RERANK)
        self.assertEqual(high["confidence"], "high")
        self.assertEqual(high["results"][0]["snippet"], SNIPPET)
        self.assertNotIn("cosine only", high["confidence"])

        medium = self.search(0.1, rerank_score=server.LOW_RERANK)
        self.assertEqual(medium["confidence"], "medium")

        weak = self.search(0.9, rerank_score=server.LOW_RERANK - 0.001)
        self.assertEqual(weak["confidence"], "LOW — answer may not be in the corpus")
        self.assertNotIn("cosine only", weak["confidence"])
        self.assertEqual(weak["results"][0]["cosine"], round(0.9, 4))
        self.assertEqual(weak["results"][0]["snippet"], SNIPPET)


if __name__ == "__main__":
    unittest.main()
