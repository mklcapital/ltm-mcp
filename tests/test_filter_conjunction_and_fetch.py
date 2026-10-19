"""Filter conjunction, exact-path trimming, and capped document fetch.

No Postgres, Ollama, or rerank model. The HTTP server is never started.
"""

import json
import unittest
from unittest.mock import patch

import server


SECRET = "FULL_TEXT_SHOULD_NOT_LEAK"


def doc_row(text, doc_id=7, relpath="Taxes/a.pdf", chars=None):
    return (
        doc_id, "disk", "/vault", relpath, "application/pdf", "lease", "Lease",
        None, 100, 1, len(text) if chars is None else chars, "pdftotext", 0,
        None, text,
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


class FilterConjunctionTests(unittest.TestCase):
    def test_both_filters_are_anded_and_bound(self):
        doc_type = "lease"
        path = "Taxes"
        conn = ScriptedConn([])

        def embed(text):
            return "[0.100000]"

        with patch.multiple(
            server,
            db=lambda: conn,
            embed_query=embed,
            hnsw_ready=lambda _con: False,
            rerank_available=lambda: False,
        ):
            out = server.ltm_search(
                query="taxes", k=4, rerank=False,
                doc_type=doc_type, path_contains=path,
            )

        self.assertEqual(out["results"], [])
        select_sql, params = next(
            (sql, p) for sql, p in conn.calls if sql.lstrip().upper().startswith("SELECT"))
        self.assertIn(" AND ", select_sql)
        self.assertNotIn(" OR ", select_sql)
        self.assertEqual(params["dt"], doc_type)
        self.assertEqual(params["pc"], "%Taxes%")
        self.assertNotIn(doc_type, select_sql)
        self.assertNotIn(path, select_sql)
        self.assertNotIn(SECRET, json.dumps(out))


class DocumentFetchEdgesTests(unittest.TestCase):
    def fetch(self, rules, key, max_chars=20000):
        conn = ScriptedConn(rules)
        with patch.object(server, "db", lambda: conn):
            out = server.ltm_get_document(key, max_chars=max_chars)
        return out, conn

    def test_zero_id_stays_an_integer_lookup(self):
        out, conn = self.fetch(
            [
                ("WHERE id", [doc_row("id-zero-body", doc_id=0)]),
                ("ILIKE", [doc_row(SECRET)]),
            ],
            "0",
        )
        self.assertTrue(out["found"])
        self.assertEqual(out["doc_id"], 0)
        self.assertEqual(out["text"], "id-zero-body")
        self.assertNotIn(SECRET, json.dumps(out))
        self.assertEqual(len(conn.calls), 1)
        sql, params = conn.calls[0]
        self.assertIn("WHERE id = :v", sql)
        self.assertNotIn("ILIKE", sql)
        self.assertEqual(params["v"], 0)
        self.assertIsInstance(params["v"], int)

    def test_surrounding_space_on_an_exact_path_does_not_fragment(self):
        out, conn = self.fetch(
            [
                ("relpath =", [doc_row("exact-body")]),
                ("ILIKE", [doc_row(SECRET, doc_id=9, relpath="other.pdf")]),
            ],
            "  Taxes/a.pdf  ",
        )
        self.assertTrue(out["found"])
        self.assertEqual(out["text"], "exact-body")
        self.assertNotIn(SECRET, json.dumps(out))
        self.assertNotIn("ambiguous", out)
        self.assertEqual(len(conn.calls), 1)
        sql, params = conn.calls[0]
        self.assertIn("relpath = :v", sql)
        self.assertNotIn("ILIKE", sql)
        self.assertEqual(params["v"], "Taxes/a.pdf")

    def test_fragment_search_is_capped_and_ordered_by_length(self):
        out, conn = self.fetch([("ILIKE", [])], "Taxes")
        self.assertFalse(out["found"])
        self.assertNotIn("ambiguous", out)
        self.assertEqual(len(conn.calls), 2)
        exact_sql = conn.calls[0][0]
        frag_sql, frag_params = conn.calls[1]
        self.assertNotIn("LIMIT 10", exact_sql)
        self.assertIn("ORDER BY char_count DESC NULLS LAST", frag_sql)
        self.assertIn("LIMIT 10", frag_sql)
        self.assertEqual(frag_params["v"], "%Taxes%")
        self.assertNotIn("Taxes", frag_sql)

    def test_truncation_keeps_the_stored_char_count(self):
        text = "y" * 250
        out, _conn = self.fetch(
            [("WHERE id", [doc_row(text, chars=99999)])],
            "7",
            max_chars=200,
        )
        self.assertTrue(out["truncated"])
        self.assertEqual(out["text"], "y" * 200)
        self.assertEqual(out["char_count"], 99999)
        self.assertNotEqual(out["char_count"], len(out["text"]))

        exact = "z" * 200
        out, _conn = self.fetch(
            [("WHERE id", [doc_row(exact, chars=200)])],
            "7",
            max_chars=200,
        )
        self.assertFalse(out["truncated"])
        self.assertEqual(out["text"], exact)
        self.assertEqual(out["char_count"], 200)


if __name__ == "__main__":
    unittest.main()
