"""The path returned by search is an exact document lookup, and query
embeddings keep six rounded decimals.

No Postgres, Ollama, or rerank model. The HTTP server is never started.
"""

import json
import unittest
from unittest.mock import patch

import server


DOCUMENT_SELECT = (
    "SELECT id, source_label, root, relpath, mime, doc_type, title, "
    "doc_date, bytes, chunk_count, char_count, text_method, ocr, "
    "ingested_at, text FROM documents "
)


class ScriptedConn:
    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    def run(self, sql, **params):
        self.calls.append((sql, params))
        if sql.lstrip().upper().startswith("SELECT"):
            return self.rows
        return []


class SearchPathLookupTests(unittest.TestCase):
    def test_the_path_search_returns_is_an_exact_lookup(self):
        # ltm_search builds path as root + "/" + relpath. Fetching that string
        # must hit the exact predicate. Dropping the root half turns it into
        # a fragment search, which can refuse to return the text.
        text = "BODY-SENTINEL"
        row = (
            22,
            "SRC-SENTINEL",
            "/ROOT",
            "Taxes/a.pdf",
            "application/pdf",
            "lease",
            "TITLE-SENTINEL",
            "2024-05-06",
            10,
            1,
            len(text),
            "pdftotext",
            0,
            None,
            text,
        )
        full = "/ROOT/Taxes/a.pdf"
        conn = ScriptedConn([row])
        with patch.object(server, "db", lambda: conn):
            out = server.ltm_get_document(full)

        self.assertEqual(len(conn.calls), 1)
        sql, params = conn.calls[0]
        self.assertEqual(sql, DOCUMENT_SELECT + "WHERE relpath = :v OR (root || '/' || relpath) = :v")
        self.assertNotIn("ILIKE", sql)
        self.assertEqual(params["v"], full)
        self.assertNotIn("%", params["v"])
        self.assertTrue(out["found"])
        self.assertFalse(out["truncated"])
        self.assertEqual(out["text"], text)
        self.assertEqual(out["path"], full)
        self.assertNotEqual(out["path"], "Taxes/a.pdf")
        self.assertNotIn(text, out["title"])


class EmbedRoundingTests(unittest.TestCase):
    def test_query_vectors_round_to_six_decimals(self):
        # pgvector sees this text. Truncation, or printing a tiny negative as
        # 0.000000, would search a different point than the one we embed.
        class Resp:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return b'{"embedding":[1.23456789,-4e-07,9.9999996]}'

        def urlopen(req, timeout=0):
            self.assertEqual(timeout, 60)
            body = json.loads(req.data.decode())
            self.assertEqual(body["model"], server.EMBED_MODEL)
            self.assertTrue(body["prompt"].endswith("lease renewal"))
            return Resp()

        with patch.object(server.urllib.request, "urlopen", urlopen):
            out = server.embed_query("lease renewal")

        self.assertEqual(out, "[1.234568,-0.000000,10.000000]")
        self.assertNotIn("1.23456789", out)
        self.assertNotIn("9.999999", out)
        self.assertNotEqual(out, "[1.234568,0.000000,10.000000]")


if __name__ == "__main__":
    unittest.main()
