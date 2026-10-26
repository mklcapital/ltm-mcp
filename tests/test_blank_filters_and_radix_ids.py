"""Blank filters stay filters, and radix-like ids are not document lookups.

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


class BlankFilterTests(unittest.TestCase):
    def test_empty_strings_still_constrain_the_search(self):
        # "" is not "no filter". Dropping it would scan the whole corpus.
        conn = ScriptedConn([
            ("ILIKE :pc", []),
            ("SELECT", [doc_row(SECRET)]),
        ])

        def embed(text):
            return "[0.100000]"

        with patch.multiple(
            server,
            db=lambda: conn,
            embed_query=embed,
            hnsw_ready=lambda _con: True,
            rerank_available=lambda: False,
        ):
            out = server.ltm_search(
                query="taxes", k=4, rerank=False,
                doc_type="", path_contains="",
            )

        self.assertEqual(out["results"], [])
        self.assertNotIn(SECRET, json.dumps(out))
        scans = [sql for sql, _params in conn.calls if "iterative_scan" in sql]
        self.assertEqual(scans, ["SET hnsw.iterative_scan = relaxed_order"])
        select_sql, params = next(
            (sql, p) for sql, p in conn.calls
            if sql.lstrip().upper().startswith("SELECT"))
        self.assertIn("d.doc_type ILIKE :dt", select_sql)
        self.assertIn("(d.root || '/' || d.relpath) ILIKE :pc", select_sql)
        self.assertIn(" AND ", select_sql)
        self.assertNotIn(" OR ", select_sql)
        self.assertEqual(params["dt"], "")
        self.assertEqual(params["pc"], "%%")
        self.assertEqual(params["k"], 4)
        self.assertNotIn(SECRET, select_sql)


class RadixIdTests(unittest.TestCase):
    def fetch(self, key):
        conn = ScriptedConn([
            ("WHERE id = :v", [doc_row(SECRET, doc_id=7)]),
            ("ILIKE", []),
        ])
        with patch.object(server, "db", lambda: conn):
            out = server.ltm_get_document(key)
        return out, conn

    def test_hex_and_exponent_ids_do_not_return_document_seven(self):
        # int("0x7", 0) and float("7e0") are 7. isdigit() must not accept them.
        for key in ("0x7", "7e0"):
            out, conn = self.fetch(key)
            self.assertFalse(out["found"], key)
            self.assertNotIn("ambiguous", out)
            self.assertNotIn(SECRET, json.dumps(out))
            self.assertEqual(len(conn.calls), 2, key)
            exact_sql, exact_params = conn.calls[0]
            self.assertNotIn("WHERE id = :v", exact_sql)
            self.assertIn("relpath = :v", exact_sql)
            self.assertEqual(exact_params["v"], key)
            self.assertNotIsInstance(exact_params["v"], int)
            self.assertNotIn("WHERE id = :v", conn.calls[1][0])
            self.assertTrue(all(params.get("v") != 7 for _sql, params in conn.calls))


if __name__ == "__main__":
    unittest.main()
