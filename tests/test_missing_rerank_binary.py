"""A configured rerank path that is not on disk must not spawn a worker.

No Postgres, Ollama, or rerank model. The HTTP server is never started.
"""

import json
import os
import tempfile
import unittest
from unittest.mock import patch

import server


SNIPPET = "COSINE-SNIPPET-KEEP"
MEDIUM_COSINE = "medium (cosine only — enable rerank for a calibrated verdict)"


class ScriptedConn:
    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    def run(self, sql, **params):
        self.calls.append((sql, params))
        if "SELECT c.id" in sql:
            return self.rows
        return []


def search_row():
    return (
        1, 7, "lease", "Title", "disk", "/vault", "a.pdf", None,
        "pdftotext", 0.80, SNIPPET,
    )


class MissingRerankBinaryTests(unittest.TestCase):
    def setUp(self):
        self._python = server.RERANK_PYTHON
        self._script = server.RERANK_SCRIPT

    def tearDown(self):
        server.RERANK_PYTHON = self._python
        server.RERANK_SCRIPT = self._script
        server._rr_proc = None

    def test_present_files_are_available_and_a_missing_file_does_not_spawn(self):
        # Both paths have to exist. A set-but-missing interpreter or script is
        # the same as unset: cosine only, and nothing is executed.
        with tempfile.TemporaryDirectory() as tmp:
            python = os.path.join(tmp, "python")
            script = os.path.join(tmp, "rerank.py")
            for path in (python, script):
                with open(path, "w", encoding="utf-8") as fh:
                    fh.write("# not executed\n")
            server.RERANK_PYTHON = python
            server.RERANK_SCRIPT = script
            self.assertTrue(server.rerank_available())

            server.RERANK_SCRIPT = os.path.join(tmp, "missing-script.py")
            self.assertFalse(server.rerank_available())
            server.RERANK_SCRIPT = script
            server.RERANK_PYTHON = os.path.join(tmp, "missing-python")
            self.assertFalse(server.rerank_available())
            server.RERANK_PYTHON = ""
            server.RERANK_SCRIPT = script
            self.assertFalse(server.rerank_available())

        missing_py = "/no/such/rerank-python-DO-NOT-SPAWN"
        missing_script = "/no/such/rerank-script-DO-NOT-SPAWN"
        conn = ScriptedConn([search_row()])
        with patch.object(server.subprocess, "Popen") as popen, patch.multiple(
            server,
            db=lambda: conn,
            embed_query=lambda _text: "[0.800000]",
            hnsw_ready=lambda _con: False,
            RERANK_PYTHON=missing_py,
            RERANK_SCRIPT=missing_script,
        ):
            out = server.ltm_search("find the clause", k=2, rerank=True)

        popen.assert_not_called()
        self.assertFalse(out["reranked"])
        self.assertEqual(out["confidence"], MEDIUM_COSINE)
        self.assertNotIn("rerank_score", out["top_score"])
        self.assertEqual(out["results"][0]["snippet"], SNIPPET)
        self.assertNotIn("rerank_score", out["results"][0])
        self.assertIn("not found", out["note"])
        self.assertNotIn(SNIPPET, out["note"])
        rendered = json.dumps(out)
        self.assertNotIn(missing_py, rendered)
        self.assertNotIn(missing_script, rendered)
        select = [params for sql, params in conn.calls if "SELECT c.id" in sql]
        self.assertEqual(len(select), 1)
        self.assertEqual(select[0]["k"], 2)
        self.assertNotEqual(select[0]["k"], server.RERANK_CANDIDATES)
        self.assertFalse(any(sql.startswith("SET ") for sql, _params in conn.calls))
