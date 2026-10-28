"""Search hits and fetched documents keep each column in its own field.

No Postgres, Ollama, or rerank model. The HTTP server is never started.
The fake row is built in SELECT order, and the SQL text is checked too, so
the query and the unpack cannot drift apart.
"""

import unittest
from unittest.mock import patch

import server


SEARCH_SELECT = (
    "SELECT c.id, d.id, d.doc_type, d.title, d.source_label, "
    "d.root, d.relpath, d.doc_date, d.text_method, "
    "1 - (e.embedding <=> CAST(:e AS vector)) AS cosine, "
    "left(c.text, %d) " % server.RERANK_TEXT_CHARS
)

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
        if sql.startswith("SELECT"):
            return self.rows
        return []


class ResultFieldMappingTests(unittest.TestCase):
    def test_a_search_hit_keeps_each_column_in_its_field(self):
        # Distinct sentinels. A swapped title/source, or a dropped doc_date,
        # would still look like a successful search.
        row = (
            11,
            22,
            "lease",
            "TITLE-SENTINEL",
            "SRC-SENTINEL",
            "/ROOT",
            "rel/file.pdf",
            "2024-05-06",
            "pdftotext-method",
            0.8012,
            "snippet body",
        )
        conn = ScriptedConn([row])
        with patch.multiple(
            server,
            db=lambda: conn,
            embed_query=lambda _text: "[0.100000]",
            hnsw_ready=lambda _con: False,
            rerank_available=lambda: False,
        ):
            out = server.ltm_search("taxes", k=4, rerank=False)

        self.assertEqual(len(out["results"]), 1)
        hit = out["results"][0]
        self.assertEqual(hit["chunk_id"], 11)
        self.assertEqual(hit["doc_id"], 22)
        self.assertEqual(hit["doc_type"], "lease")
        self.assertEqual(hit["title"], "TITLE-SENTINEL")
        self.assertEqual(hit["source"], "SRC-SENTINEL")
        self.assertEqual(hit["path"], "/ROOT/rel/file.pdf")
        self.assertEqual(hit["doc_date"], "2024-05-06")
        self.assertEqual(hit["extraction"], "pdftotext-method")
        self.assertEqual(hit["cosine"], 0.8012)
        self.assertEqual(hit["snippet"], "snippet body")
        self.assertNotEqual(hit["title"], hit["source"])
        self.assertNotEqual(hit["doc_date"], hit["extraction"])
        self.assertNotIn("TITLE-SENTINEL", hit["snippet"])
        self.assertNotIn("SRC-SENTINEL", hit["snippet"])

        select_sql = next(sql for sql, _params in conn.calls if sql.startswith("SELECT"))
        self.assertIn(SEARCH_SELECT, select_sql)
        self.assertLess(select_sql.index("d.title"), select_sql.index("d.source_label"))
        self.assertLess(select_sql.index("d.doc_date"), select_sql.index("d.text_method"))
        self.assertLess(select_sql.index("d.text_method"), select_sql.index("AS cosine"))

    def test_a_fetched_document_keeps_each_column_in_its_field(self):
        text = "BODY-SENTINEL"
        row = (
            22,
            "SRC-SENTINEL",
            "/ROOT",
            "rel/file.pdf",
            "application/pdf",
            "lease",
            "TITLE-SENTINEL",
            "2024-05-06",
            4096,
            3,
            99999,
            "pdftotext-method",
            1,
            "2024-05-07T00:00:00",
            text,
        )
        conn = ScriptedConn([row])
        with patch.object(server, "db", lambda: conn):
            out = server.ltm_get_document("22")

        self.assertTrue(out["found"])
        self.assertFalse(out["truncated"])
        self.assertEqual(out["doc_id"], 22)
        self.assertEqual(out["source"], "SRC-SENTINEL")
        self.assertEqual(out["path"], "/ROOT/rel/file.pdf")
        self.assertEqual(out["mime"], "application/pdf")
        self.assertEqual(out["doc_type"], "lease")
        self.assertEqual(out["title"], "TITLE-SENTINEL")
        self.assertEqual(out["doc_date"], "2024-05-06")
        self.assertEqual(out["bytes"], 4096)
        self.assertEqual(out["chunk_count"], 3)
        self.assertEqual(out["char_count"], 99999)
        self.assertNotEqual(out["char_count"], out["bytes"])
        self.assertEqual(out["extraction"], "pdftotext-method")
        self.assertIs(out["ocr"], True)
        self.assertEqual(out["ingested_at"], "2024-05-07T00:00:00")
        self.assertEqual(out["text"], text)
        self.assertNotEqual(out["text"], out["title"])
        self.assertNotEqual(out["text"], out["mime"])
        self.assertNotIn("BODY-SENTINEL", out["title"])

        self.assertEqual(len(conn.calls), 1)
        sql, params = conn.calls[0]
        self.assertEqual(sql, DOCUMENT_SELECT + "WHERE id = :v")
        self.assertNotIn("ILIKE", sql)
        self.assertEqual(params["v"], 22)
        self.assertIsInstance(params["v"], int)


if __name__ == "__main__":
    unittest.main()
