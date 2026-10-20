"""Exact-match ambiguity and rerank errors must not leak document text.

No Postgres, Ollama, or rerank model. The HTTP server is never started.
"""

import json
import unittest
from unittest.mock import patch

import server


SECRET = "FULL_TEXT_SHOULD_NOT_LEAK"
NOTE_LEAK = "RERANK_EXCEPTION_TEXT_SHOULD_NOT_LEAK"
HIGH_COSINE = "high (cosine only — enable rerank for a calibrated verdict)"


def search_row(chunk_id, cosine, text):
    return (
        chunk_id, 10, "tax return", "Title", "disk", "/vault", "Taxes/2024.pdf",
        "2024-01-01", "pdftotext", cosine, text,
    )


def doc_row(text, doc_id, relpath="Taxes/a.pdf"):
    return (
        doc_id, "disk", "/vault", relpath, "application/pdf", "lease", "Lease",
        None, 100, 1, len(text), "pdftotext", 0, None, text,
    )


class ScriptedConn:
    """First matching rule wins. A later rule is the query we must not reach."""

    def __init__(self, rules):
        self.rules = rules
        self.calls = []

    def run(self, sql, **params):
        self.calls.append((sql, params))
        for needle, rows in self.rules:
            if needle in sql:
                return rows
        return []


class ExactMatchTests(unittest.TestCase):
    def fetch(self, rules, key):
        conn = ScriptedConn(rules)
        with patch.object(server, "db", lambda: conn):
            out = server.ltm_get_document(key)
        return out, conn

    def test_two_exact_paths_do_not_return_text_or_fall_through(self):
        out, conn = self.fetch(
            [
                ("WHERE relpath", [
                    doc_row(SECRET, 1),
                    doc_row(SECRET + "-sibling", 2),
                ]),
                ("ILIKE", [doc_row("FRAGMENT_SHOULD_NOT_BE_FETCHED", 9, "other.pdf")]),
            ],
            "Taxes/a.pdf",
        )
        self.assertFalse(out["found"])
        self.assertTrue(out["ambiguous"])
        self.assertEqual([item["doc_id"] for item in out["candidates"]], [1, 2])
        self.assertNotIn("text", out["candidates"][0])
        rendered = json.dumps(out)
        self.assertNotIn(SECRET, rendered)
        self.assertNotIn("FRAGMENT_SHOULD_NOT_BE_FETCHED", rendered)
        self.assertEqual(len(conn.calls), 1)
        self.assertNotIn("ILIKE", conn.calls[0][0])
        self.assertEqual(conn.calls[0][1]["v"], "Taxes/a.pdf")

    def test_plus_and_decimal_ids_are_not_integer_lookups(self):
        # int("+7") and int("7.0") are 7, but the id gate is str.isdigit.
        # A wider parser must not start returning document 7 for these keys.
        for key in ("+7", "7.0"):
            with self.subTest(key=key):
                out, conn = self.fetch(
                    [
                        ("WHERE id", [doc_row(SECRET, 7)]),
                        ("ILIKE", [doc_row("fragment-hit", 9, "notes/7.0.pdf")]),
                    ],
                    key,
                )
                self.assertTrue(out["found"])
                self.assertEqual(out["doc_id"], 9)
                self.assertEqual(out["text"], "fragment-hit")
                self.assertNotIn(SECRET, json.dumps(out))
                self.assertTrue(all("WHERE id" not in sql for sql, _params in conn.calls))
                self.assertNotIn(7, [params.get("v") for _sql, params in conn.calls])
                self.assertIsInstance(conn.calls[0][1]["v"], str)


class RerankNoteTests(unittest.TestCase):
    def test_a_rerank_exception_message_is_not_returned(self):
        conn = ScriptedConn([])
        # Any SELECT returns the cosine hit. The exception text must not replace it.
        conn.rules = [("SELECT", [search_row(1, 0.9, "kept clause")])]

        def embed(text):
            return "[0.100000]"

        def rerank(query, texts, top_k):
            self.assertEqual(texts, ["kept clause"])
            raise RuntimeError(NOTE_LEAK)

        with patch.multiple(
            server,
            db=lambda: conn,
            embed_query=embed,
            hnsw_ready=lambda _con: False,
            rerank_available=lambda: True,
            run_rerank=rerank,
        ):
            out = server.ltm_search(query="taxes", k=2, rerank=True)

        self.assertFalse(out["reranked"])
        self.assertEqual(out["confidence"], HIGH_COSINE)
        self.assertEqual(out["results"][0]["snippet"], "kept clause")
        self.assertNotIn("rerank_score", out["results"][0])
        self.assertIn("RuntimeError", out["note"])
        self.assertIn("cosine order only", out["note"])
        self.assertNotIn(NOTE_LEAK, out["note"])
        self.assertNotIn(NOTE_LEAK, json.dumps(out))


if __name__ == "__main__":
    unittest.main()
