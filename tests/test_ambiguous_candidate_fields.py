"""Ambiguous document matches identify candidates without returning text.

No Postgres or Ollama. Two exact hits stay in driver order, and a zero
char_count is preserved.
"""

import json
import unittest
from unittest.mock import patch

import server


LEAK_ONE = "LEAK-ONE-FULL-TEXT"
LEAK_TWO = "LEAK-TWO-FULL-TEXT"


def doc_row(text, doc_id, root, relpath, doc_type, title, char_count):
    return (
        doc_id, "disk", root, relpath, "application/pdf", doc_type, title,
        None, 4096, 3, char_count, "pdftotext", 1, None, text,
    )


class AmbiguousCandidateTests(unittest.TestCase):
    def test_candidates_keep_identity_and_omit_text(self):
        key = "Taxes/a.pdf"
        conn = type("Conn", (), {})()
        conn.calls = []

        def run(sql, **params):
            conn.calls.append((sql, params))
            if "WHERE id =" in sql:
                raise AssertionError("a path must not be queried as an integer id")
            if "ILIKE" in sql:
                raise AssertionError("two exact hits must not fall through to a fragment")
            return [
                doc_row(LEAK_ONE, 9, None, None, None, None, None),
                doc_row(LEAK_TWO, 10, "/vault", "Taxes/a.pdf", "", "Lease", 0),
            ]

        conn.run = run
        with patch.object(server, "db", lambda: conn):
            out = server.ltm_get_document(key)

        dumped = json.dumps(out)
        self.assertFalse(out["found"])
        self.assertTrue(out["ambiguous"])
        self.assertNotIn("text", out)
        self.assertNotIn(LEAK_ONE, dumped)
        self.assertNotIn(LEAK_TWO, dumped)
        self.assertNotIn("None", dumped)
        self.assertEqual(len(out["candidates"]), 2)
        self.assertEqual(out["candidates"][0], {
            "doc_id": 9,
            "path": "/",
            "doc_type": "untyped",
            "title": None,
            "char_count": None,
        })
        self.assertEqual(out["candidates"][1], {
            "doc_id": 10,
            "path": "/vault/Taxes/a.pdf",
            "doc_type": "untyped",
            "title": "Lease",
            "char_count": 0,
        })
        self.assertEqual(len(conn.calls), 1)
        sql, params = conn.calls[0]
        self.assertIn("relpath = :v", sql)
        self.assertNotIn("ILIKE", sql)
        self.assertEqual(params["v"], key)
        self.assertNotIn(key, sql)


if __name__ == "__main__":
    unittest.main()
