"""A NUL inside a document id must not be stripped into a different id.

No database or network.
"""

import unittest
from unittest.mock import patch

import server

SECRET = "only-body-must-not-leak"


def doc_row(text):
    return (
        7, "disk", "/vault", "Taxes/a.pdf", "application/pdf", "lease", "Lease",
        None, 100, 1, len(text), "pdftotext", 0, None, text,
    )


class Conn:
    def __init__(self):
        self.calls = []

    def run(self, sql, **params):
        self.calls.append((sql, params))
        if "WHERE id = :v" in sql and params.get("v") == 7:
            return [doc_row(SECRET)]
        return []


class NulDocumentIdTests(unittest.TestCase):
    def fetch(self, key):
        conn = Conn()
        with patch.object(server, "db", lambda: conn):
            out = server.ltm_get_document(key)
        return out, conn

    def test_a_nul_does_not_truncate_the_id_into_a_different_document(self):
        for key in ("7\x00", "7\x00extra", "\x007"):
            out, conn = self.fetch(key)
            self.assertFalse(out["found"], repr(key))
            self.assertNotIn("ambiguous", out)
            rendered = str(out)
            self.assertNotIn(SECRET, rendered, repr(key))
            self.assertGreaterEqual(len(conn.calls), 2, repr(key))
            for sql, params in conn.calls:
                self.assertNotIsInstance(params.get("v"), int, repr(key))
                self.assertNotEqual(params.get("v"), 7, repr(key))
                self.assertNotIn("\x00", sql)
                self.assertNotIn(SECRET, sql)
                self.assertNotIn("WHERE id = :v", sql)
            self.assertEqual(conn.calls[0][1]["v"], key)
            self.assertIn("relpath = :v", conn.calls[0][0])
            self.assertIn("ILIKE :v", conn.calls[1][0])
            self.assertEqual(conn.calls[1][1]["v"], "%" + key + "%")
            self.assertNotEqual(conn.calls[1][1]["v"], "%7%")


if __name__ == "__main__":
    unittest.main()
