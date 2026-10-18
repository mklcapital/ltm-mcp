"""Missing ids, blank paths, and empty searches must not leak document text.

No Postgres, Ollama, or rerank model. The HTTP server is never started.
"""

import json
import unittest
from unittest.mock import patch

import server


SECRET = "FULL_TEXT_SHOULD_NOT_LEAK"
NO_MATCHES = "LOW — no matches (filters may be too narrow)"


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


class DocumentMissTests(unittest.TestCase):
    def fetch(self, rules, key):
        conn = ScriptedConn(rules)
        with patch.object(server, "db", lambda: conn):
            out = server.ltm_get_document(key)
        return out, conn

    def test_a_missing_numeric_id_does_not_fragment_search(self):
        out, conn = self.fetch(
            [("ILIKE", [doc_row(SECRET)])],
            "42",
        )
        self.assertFalse(out["found"])
        self.assertNotIn("ambiguous", out)
        self.assertNotIn(SECRET, json.dumps(out))
        self.assertEqual(len(conn.calls), 1)
        sql, params = conn.calls[0]
        self.assertIn("WHERE id = :v", sql)
        self.assertNotIn("ILIKE", sql)
        self.assertEqual(params["v"], 42)
        self.assertIsInstance(params["v"], int)

    def test_a_blank_path_does_not_return_document_text(self):
        out, conn = self.fetch(
            [("ILIKE", [
                doc_row(SECRET, doc_id=1, relpath="a.pdf"),
                doc_row(SECRET, doc_id=2, relpath="b.pdf"),
            ])],
            " \t ",
        )
        self.assertFalse(out["found"])
        self.assertTrue(out["ambiguous"])
        self.assertNotIn(SECRET, json.dumps(out))
        self.assertEqual([item["doc_id"] for item in out["candidates"]], [1, 2])
        self.assertNotIn("text", out["candidates"][0])
        self.assertEqual(len(conn.calls), 2)
        self.assertNotIn("ILIKE", conn.calls[0][0])
        self.assertEqual(conn.calls[0][1]["v"], "")
        self.assertIn("ILIKE :v", conn.calls[1][0])
        self.assertEqual(conn.calls[1][1]["v"], "%%")
        self.assertNotIn(SECRET, conn.calls[1][0])


class EmptySearchTests(unittest.TestCase):
    def test_an_empty_result_does_not_call_the_reranker(self):
        conn = ScriptedConn([])
        seen = {}

        def embed(text):
            seen["embed"] = text
            return "[0.100000]"

        def rerank(query, texts, top_k):
            seen["rerank"] = (query, texts, top_k)
            return [0], [9.0]

        with patch.multiple(
            server,
            db=lambda: conn,
            embed_query=embed,
            hnsw_ready=lambda _con: False,
            rerank_available=lambda: True,
            run_rerank=rerank,
        ):
            out = server.ltm_search(query="taxes", k=8, rerank=True)

        self.assertEqual(seen["embed"], "taxes")
        self.assertNotIn("rerank", seen)
        self.assertEqual(out["results"], [])
        self.assertEqual(out["confidence"], NO_MATCHES)
        self.assertNotIn("note", out)
        self.assertNotIn("reranked", out)
        self.assertNotIn(SECRET, json.dumps(out))
        select = [sql for sql, _params in conn.calls if sql.lstrip().upper().startswith("SELECT")]
        self.assertEqual(len(select), 1)


if __name__ == "__main__":
    unittest.main()
