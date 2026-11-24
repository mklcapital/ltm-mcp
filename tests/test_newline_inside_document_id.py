"""A newline inside a document id is not an integer primary key.

`str.strip` removes surrounding whitespace, not a newline between digits.
That key must stay a path lookup and must not return document 7.
"""

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import server


class NewlineInsideDocumentIdTest(unittest.TestCase):
    def test_internal_newline_is_not_document_seven(self):
        key = "7\nextra"
        secret = "DOC-7-FULL-TEXT-SHOULD-NOT-LEAK"
        original = server.db

        class Con:
            def __init__(self):
                self.calls = []

            def run(self, sql, **params):
                self.calls.append((sql, params))
                if "WHERE id = :v" in sql:
                    return [(7, "src", "ROOT", "secret.pdf", "application/pdf",
                             "lease", "Secret", None, 10, 1, 99, "pdf", 0,
                             None, secret)]
                return []

        con = Con()
        server.db = lambda: con
        try:
            out = server.ltm_get_document(key)
        finally:
            server.db = original

        self.assertEqual(len(con.calls), 2)
        exact_sql, exact_params = con.calls[0]
        like_sql, like_params = con.calls[1]
        self.assertNotIn("WHERE id = :v", exact_sql)
        self.assertNotIn("WHERE id = :v", like_sql)
        self.assertIn("relpath = :v", exact_sql)
        self.assertNotIn("ILIKE", exact_sql)
        self.assertIn("ILIKE", like_sql)
        self.assertEqual(exact_params["v"], key)
        self.assertEqual(like_params["v"], "%" + key + "%")
        self.assertNotEqual(exact_params["v"], 7)
        self.assertNotIn("\n", exact_sql)
        self.assertNotIn("\n", like_sql)
        self.assertNotIn(secret, exact_sql)
        self.assertNotIn(secret, like_sql)
        self.assertNotIn("7", exact_sql)

        self.assertFalse(out["found"])
        self.assertNotIn("ambiguous", out)
        self.assertNotIn("text", out)
        self.assertEqual(
            out["error"],
            "no document matches '7\\nextra' (id, path, or fragment)",
        )
        self.assertNotIn("\n", out["error"])
        blob = json.dumps(out)
        self.assertNotIn(secret, blob)
        self.assertNotIn("secret.pdf", blob)


if __name__ == "__main__":
    unittest.main()
