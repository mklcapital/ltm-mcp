"""A case-mismatched path is not an exact id, then ILIKE may return one hit.

No Postgres, Ollama, or rerank model.
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
    def __init__(self, rules):
        self.rules = rules
        self.calls = []

    def run(self, sql, **params):
        self.calls.append((sql, params))
        for needle, rows in self.rules:
            if needle in sql:
                return rows
        return []


class PathCaseFallbackTests(unittest.TestCase):
    def test_case_mismatch_uses_one_fragment_hit_and_two_hits_stay_ambiguous(self):
        # Exact `=` is case-sensitive. The fallback ILIKE is not folded here;
        # one hit returns that text, two hits must not.
        key = "taxes/A.PDF"
        unique = ScriptedConn([
            ("ILIKE", [doc_row("fragment-hit", 9, "Taxes/A.PDF")]),
            ("WHERE id =", [doc_row(SECRET, 7)]),
        ])
        with patch.object(server, "db", lambda: unique):
            out = server.ltm_get_document(key)

        self.assertTrue(out["found"])
        self.assertEqual(out["doc_id"], 9)
        self.assertEqual(out["text"], "fragment-hit")
        self.assertNotIn(SECRET, json.dumps(out))
        self.assertEqual(len(unique.calls), 2)
        exact_sql, exact_params = unique.calls[0]
        frag_sql, frag_params = unique.calls[1]
        self.assertNotIn("ILIKE", exact_sql)
        self.assertNotIn("WHERE id", exact_sql)
        self.assertIn("relpath = :v", exact_sql)
        self.assertEqual(exact_params["v"], key)
        self.assertNotIn(key.lower(), exact_sql)
        self.assertNotIn(key, exact_sql)
        self.assertIn("ILIKE", frag_sql)
        self.assertIn("LIMIT 10", frag_sql)
        self.assertEqual(frag_params["v"], "%" + key + "%")
        self.assertNotEqual(frag_params["v"], "%" + key.lower() + "%")
        self.assertNotIn(key, frag_sql)

        ambiguous = ScriptedConn([
            ("ILIKE", [
                doc_row("SECRET-ONE", 9, "Taxes/A.PDF"),
                doc_row("SECRET-TWO", 10, "archive/Taxes/A.PDF"),
            ]),
        ])
        with patch.object(server, "db", lambda: ambiguous):
            out = server.ltm_get_document(key)

        self.assertFalse(out["found"])
        self.assertTrue(out["ambiguous"])
        self.assertNotIn("text", out)
        dumped = json.dumps(out)
        self.assertNotIn("SECRET-ONE", dumped)
        self.assertNotIn("SECRET-TWO", dumped)
        self.assertEqual([c["doc_id"] for c in out["candidates"]], [9, 10])
        self.assertEqual(len(ambiguous.calls), 2)
        self.assertNotIn("ILIKE", ambiguous.calls[0][0])
        self.assertEqual(ambiguous.calls[0][1]["v"], key)
        self.assertEqual(ambiguous.calls[1][1]["v"], "%" + key + "%")


if __name__ == "__main__":
    unittest.main()
