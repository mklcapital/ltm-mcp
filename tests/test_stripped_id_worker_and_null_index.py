"""Document-id whitespace, a null HNSW flag, and the rerank worker launch.

A trailing newline still strips to an integer primary key. A NULL or false
indisvalid flag must not stick. The worker is started unbuffered, without a
shell, and without mutating this process's environment.
"""

import os
import subprocess
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import server


class StrippedDocumentIdTest(unittest.TestCase):
    def test_surrounding_newlines_stay_an_integer_lookup(self):
        secret = "DOC-7-FULL-TEXT-SHOULD-RETURN"
        original = server.db

        class Con:
            def __init__(self):
                self.calls = []

            def run(self, sql, **params):
                self.calls.append((sql, params))
                return [(7, "src", "ROOT", "a.pdf", "application/pdf", "lease",
                         "Lease", None, 10, 1, 99, "pdf", 0, None, secret)]

        for key in ("7\n", "\n7", "\r\n7\r\n", "7\r"):
            con = Con()
            server.db = lambda con=con: con
            try:
                out = server.ltm_get_document(key)
            finally:
                server.db = original

            self.assertEqual(len(con.calls), 1, key)
            sql, params = con.calls[0]
            self.assertIn("WHERE id = :v", sql)
            self.assertNotIn("ILIKE", sql)
            self.assertNotIn("relpath =", sql)
            self.assertEqual(params["v"], 7)
            self.assertIsInstance(params["v"], int)
            self.assertNotIn("\n", sql)
            self.assertNotIn("\r", sql)
            self.assertNotIn(key, sql)
            self.assertNotIn(secret, sql)
            self.assertTrue(out["found"])
            self.assertFalse(out.get("ambiguous", False))
            self.assertEqual(out["doc_id"], 7)
            self.assertEqual(out["text"], secret)
            self.assertEqual(out["path"], "ROOT/a.pdf")


class NullHnswIndexTest(unittest.TestCase):
    def test_null_or_false_index_does_not_stick(self):
        original = server._hnsw_ready
        server._hnsw_ready = False

        class Con:
            def __init__(self, flags):
                self.flags = list(flags)
                self.sql = []

            def run(self, sql, **_params):
                self.sql.append(sql)
                return [(self.flags.pop(0),)]

        con = Con([None, False, None])
        try:
            self.assertFalse(server.hnsw_ready(con))
            self.assertFalse(server._hnsw_ready)
            self.assertFalse(server.hnsw_ready(con))
            self.assertFalse(server._hnsw_ready)
            self.assertFalse(server.hnsw_ready(con))
            self.assertFalse(server._hnsw_ready)
        finally:
            server._hnsw_ready = original

        self.assertEqual(len(con.sql), 3)
        for sql in con.sql:
            self.assertIn("indisvalid", sql)
            self.assertIn("emb_mxbai", sql)
            self.assertIn("hnsw", sql)
            self.assertNotIn("None", sql)


class RerankWorkerLaunchTest(unittest.TestCase):
    def test_worker_is_unbuffered_and_does_not_mutate_environ(self):
        original_popen = server.subprocess.Popen
        original_read = server._rr_read_line
        original_py = server.RERANK_PYTHON
        original_script = server.RERANK_SCRIPT
        original_proc = server._rr_proc
        server.RERANK_PYTHON = sys.executable
        server.RERANK_SCRIPT = os.path.abspath(server.__file__)
        server._rr_proc = None
        seen = {}

        def popen(cmd, **kwargs):
            seen["cmd"] = list(cmd)
            seen["kwargs"] = kwargs

            class Proc:
                def poll(self):
                    return None

                def kill(self):
                    raise AssertionError("ready worker must not be killed")

            return Proc()

        server.subprocess.Popen = popen
        server._rr_read_line = lambda proc, timeout: {"ready": True}
        before = os.environ.get("TQDM_DISABLE")
        try:
            proc = server._rr_worker()
        finally:
            server.subprocess.Popen = original_popen
            server._rr_read_line = original_read
            server.RERANK_PYTHON = original_py
            server.RERANK_SCRIPT = original_script
            server._rr_proc = None if original_proc is None else original_proc
            after = os.environ.get("TQDM_DISABLE")

        self.assertEqual(after, before)
        self.assertIsNotNone(proc)
        cmd = seen["cmd"]
        kwargs = seen["kwargs"]
        self.assertEqual(cmd[0], sys.executable)
        self.assertEqual(cmd[1], "-u")
        self.assertEqual(cmd[2], "-c")
        self.assertEqual(len(cmd), 4)
        src = cmd[3]
        self.assertIn("from rerank import Reranker", src)
        self.assertNotIn("os.system", src)
        self.assertNotIn("shell=True", src)
        self.assertNotIn("shell", kwargs)
        self.assertIs(kwargs["stdin"], subprocess.PIPE)
        self.assertIs(kwargs["stdout"], subprocess.PIPE)
        self.assertIs(kwargs["stderr"], sys.stderr)
        self.assertIs(kwargs["text"], True)
        self.assertIsNot(kwargs["env"], os.environ)
        self.assertEqual(kwargs["env"]["TQDM_DISABLE"], "1")
