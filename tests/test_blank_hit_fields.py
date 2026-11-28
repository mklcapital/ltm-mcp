"""A search hit must not invent a type, a title, or a snippet from blanks.

No Postgres, Ollama, or rerank model. The HTTP server is never started.
"""

import json
import unittest
from unittest.mock import patch

import server


class ScriptedConn:
    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    def run(self, sql, **params):
        self.calls.append((sql, params))
        if sql.startswith("SELECT"):
            return self.rows
        return []


class BlankHitFieldTests(unittest.TestCase):
    def test_whitespace_type_and_blank_text_stay_blank(self):
        # A whitespace doc_type is not missing. Null title/source must not
        # become the string "None". A whitespace chunk is not a snippet.
        row = (
            11,
            22,
            "  ",
            None,
            None,
            None,
            "rel/file.pdf",
            None,
            None,
            0.5,
            "\n\t   \n",
        )
        conn = ScriptedConn([row])
        with patch.multiple(
            server,
            db=lambda: conn,
            embed_query=lambda _text: "[1.0); DROP TABLE emb_mxbai;--",
            hnsw_ready=lambda _con: False,
            rerank_available=lambda: False,
        ):
            out = server.ltm_search("taxes", k=1, rerank=False)

        self.assertEqual(len(out["results"]), 1)
        hit = out["results"][0]
        self.assertEqual(hit["doc_type"], "  ")
        self.assertNotEqual(hit["doc_type"], "untyped")
        self.assertIsNone(hit["title"])
        self.assertIsNone(hit["source"])
        self.assertIsNone(hit["doc_date"])
        self.assertIsNone(hit["extraction"])
        self.assertEqual(hit["path"], "/rel/file.pdf")
        self.assertEqual(hit["snippet"], "")
        self.assertNotIn("\n", hit["snippet"])
        self.assertNotIn("\t", hit["snippet"])
        self.assertEqual(hit["cosine"], 0.5)
        self.assertEqual(hit["chunk_id"], 11)
        self.assertEqual(hit["doc_id"], 22)
        self.assertNotIn("rerank_score", hit)
        self.assertFalse(out["reranked"])
        self.assertEqual(
            out["confidence"],
            "LOW — answer may not be in the corpus (cosine only)",
        )
        self.assertEqual(out["hnsw_index"], "not built yet (seq scan, slower)")
        self.assertEqual(out["query"], "taxes")
        dumped = json.dumps(out)
        self.assertNotIn("None", dumped)
        self.assertNotIn("DROP TABLE", dumped)
        self.assertNotIn("\n", hit["path"])

        selects = [(sql, params) for sql, params in conn.calls if sql.startswith("SELECT")]
        self.assertEqual(len(selects), 1)
        sql, params = selects[0]
        self.assertIn("left(c.text, %d)" % server.RERANK_TEXT_CHARS, sql)
        self.assertNotIn("SET ", sql)
        self.assertNotIn("DROP", sql)
        self.assertEqual(params["k"], 1)
        self.assertEqual(params["e"], "[1.0); DROP TABLE emb_mxbai;--")
        self.assertNotIn("[1.0)", sql)
        sets = [sql for sql, _params in conn.calls if sql.startswith("SET")]
        self.assertEqual(sets, [])


if __name__ == "__main__":
    unittest.main()
