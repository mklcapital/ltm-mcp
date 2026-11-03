"""OCR is a boolean flag, and the embed model must match by prefix.

No Postgres or Ollama. Stats and document fetches run against a scripted
connection and a fake tags response.
"""

import io
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
        return self.rows


def doc_row(ocr, nbytes, chunks):
    return (
        7, "disk", "/vault", "Taxes/a.pdf", None, "lease", "Lease",
        None, nbytes, chunks, 4, "pdftotext", ocr, None, "body",
    )


class OcrFlagTests(unittest.TestCase):
    def fetch(self, row):
        conn = ScriptedConn([row])
        with patch.object(server, "db", lambda: conn):
            out = server.ltm_get_document("7")
        self.assertEqual(conn.calls[0][1]["v"], 7)
        return out

    def test_zero_is_false_and_any_other_count_is_true(self):
        # `ocr == 1` would reject a driver that reports 2, and `str(ocr)` would
        # store "0" / "None" instead of a boolean the client can trust.
        off = self.fetch(doc_row(0, None, None))
        self.assertIs(off["ocr"], False)
        self.assertIsNone(off["bytes"])
        self.assertIsNone(off["chunk_count"])
        self.assertIsNone(off["mime"])
        self.assertEqual(off["text"], "body")
        self.assertNotIn("None", json.dumps(off["bytes"]))

        on = self.fetch(doc_row(2, 4096, 3))
        self.assertIs(on["ocr"], True)
        self.assertNotEqual(on["ocr"], 2)
        self.assertEqual(on["bytes"], 4096)
        self.assertEqual(on["chunk_count"], 3)


class ModelPrefixTests(unittest.TestCase):
    def stats(self, names):
        conn = ScriptedConn([])

        class Resp(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

        def urlopen(url, timeout=0):
            self.assertEqual(timeout, 5)
            self.assertTrue(str(url).endswith("/api/tags"))
            return Resp(json.dumps({"models": [{"name": n} for n in names]}).encode())

        def run(sql, **_params):
            conn.calls.append(sql)
            if sql.startswith("SELECT count(*) FROM documents"):
                return [(1,)]
            if sql.startswith("SELECT count(*) FROM doc_chunks"):
                return [(1,)]
            if sql.startswith("SELECT count(*) FROM emb_mxbai"):
                return [(1,)]
            if sql.startswith("SELECT COALESCE"):
                return []
            if sql.startswith("SELECT pg_size_pretty"):
                return [("1 MB", "2 MB", "3 MB", "9 MB")]
            return []

        conn.run = run
        with patch.multiple(
            server,
            db=lambda: conn,
            hnsw_ready=lambda _con: False,
            rerank_available=lambda: False,
        ), patch.object(server.urllib.request, "urlopen", urlopen):
            return server.ltm_stats()

    def test_a_suffix_lookalike_is_not_the_embed_model(self):
        # `in` would accept a model that only ends with the id. Tags such as
        # `mxbai-embed-large:latest` still have to count, so the check is a prefix.
        lookalike = "vendor-" + server.EMBED_MODEL
        out = self.stats([lookalike])
        self.assertIn("NOT pulled", out["ollama"])
        self.assertNotIn("present)", out["ollama"])
        self.assertNotIn(lookalike, out["ollama"])

        exact = self.stats([server.EMBED_MODEL])
        self.assertEqual(exact["ollama"], "ok (" + server.EMBED_MODEL + " present)")
        self.assertNotIn("NOT pulled", exact["ollama"])
