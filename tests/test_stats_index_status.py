"""Corpus stats must say whether search is on HNSW or a sequential scan.

No Postgres or Ollama. The status string is what an operator uses to tell a
fast index from the ~10s fallback, so a loose "ready" substring is not enough.
"""

import json
import unittest
from unittest.mock import patch

import server


READY = "ready — fast approximate search"
SEQUENTIAL = (
    "not built yet — searches fall back to sequential "
    "scan (~10 s each); auto-build pending"
)


class ScriptedConn:
    def __init__(self, index_rows):
        self.index_rows = index_rows
        self.calls = []

    def run(self, sql, **params):
        self.calls.append((sql, params))
        if sql.startswith("SELECT count(*) FROM documents"):
            return [(2,)]
        if sql.startswith("SELECT count(*) FROM doc_chunks"):
            return [(4,)]
        if sql.startswith("SELECT count(*) FROM emb_mxbai"):
            return [(4,)]
        if sql.startswith("SELECT COALESCE"):
            return []
        if sql.startswith("SELECT pg_size_pretty"):
            return [("1 MB", "2 MB", "3 MB", "9 MB")]
        if sql.startswith("SELECT idx.indisvalid"):
            return self.index_rows
        raise AssertionError(sql)


class IndexStatusTests(unittest.TestCase):
    def setUp(self):
        self._saved = server._hnsw_ready
        server._hnsw_ready = False

    def tearDown(self):
        server._hnsw_ready = self._saved

    def stats(self, index_rows):
        conn = ScriptedConn(index_rows)

        def urlopen(_url, timeout=0):
            self.assertEqual(timeout, 5)
            raise TimeoutError("probe down")

        with patch.object(server, "db", lambda: conn), patch.object(
            server.urllib.request, "urlopen", urlopen
        ):
            out = server.ltm_stats()
        return out, conn

    def test_a_missing_index_names_the_sequential_scan(self):
        out, conn = self.stats([])
        self.assertEqual(out["hnsw_index"], SEQUENTIAL)
        self.assertNotIn("ready", out["hnsw_index"])
        self.assertEqual(out["embedding_coverage"], "100.00%")
        self.assertEqual(out["ollama"], "unreachable (TimeoutError)")
        self.assertNotIn("probe down", json.dumps(out))
        indexed = [sql for sql, _params in conn.calls if "indisvalid" in sql]
        self.assertEqual(len(indexed), 1)
        self.assertIn("emb_mxbai", indexed[0])
        self.assertIn("amname = 'hnsw'", indexed[0])

    def test_a_valid_index_is_named_ready_and_not_a_seq_scan(self):
        out, conn = self.stats([(True,)])
        self.assertEqual(out["hnsw_index"], READY)
        self.assertNotIn("sequential", out["hnsw_index"])
        self.assertNotIn("10 s", out["hnsw_index"])
        indexed = [sql for sql, _params in conn.calls if "indisvalid" in sql]
        self.assertEqual(len(indexed), 1)
        self.assertNotIn("probe down", json.dumps(out))
