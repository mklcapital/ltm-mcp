"""An empty rerank line is a dead worker. The retry can still succeed.

No Postgres, Ollama, or rerank model. The HTTP server is never started.
"""

import json
import os
import unittest
from unittest.mock import patch

import server


ALPHA = "ALPHA_COSINE_FIRST"
BETA = "BETA_SECOND_CHUNK"


class ScriptedConn:
    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    def run(self, sql, **params):
        self.calls.append((sql, params))
        if "SELECT c.id" in sql:
            return self.rows
        return []


def search_row(chunk_id, cosine, text):
    return (
        chunk_id, 10 + chunk_id, "lease", "Title", "disk", "/vault",
        "a.pdf", None, "pdftotext", cosine, text,
    )


class Pipe:
    def __init__(self):
        self.buf = []

    def write(self, data):
        self.buf.append(data)
        return len(data)

    def flush(self):
        return None


class Proc:
    def __init__(self, line):
        read_fd, write_fd = os.pipe()
        if line:
            os.write(write_fd, line)
        os.close(write_fd)
        self.stdout = os.fdopen(read_fd, "r")
        self.stdin = Pipe()
        self.killed = False

    def poll(self):
        return None

    def kill(self):
        self.killed = True
        if self.stdout is not None:
            self.stdout.close()
            self.stdout = None


class RerankEmptyLineRetryTests(unittest.TestCase):
    def tearDown(self):
        server._rr_proc = None

    def test_empty_stdout_retries_once_and_keeps_the_second_order(self):
        rows = [
            search_row(1, 0.99, ALPHA),
            search_row(2, 0.80, BETA),
        ]
        conn = ScriptedConn(rows)
        procs = []
        lines = [
            b"",
            b'{"order":[1,0],"scores":[4.0,1.0]}\n',
        ]

        def worker():
            proc = Proc(lines[len(procs)])
            procs.append(proc)
            return proc

        with patch.multiple(
            server,
            db=lambda: conn,
            embed_query=lambda _text: "[0.100000]",
            hnsw_ready=lambda _con: False,
            rerank_available=lambda: True,
            _rr_worker=worker,
        ):
            out = server.ltm_search("find the clause", k=2, rerank=True)

        self.assertEqual([hit["snippet"] for hit in out["results"]], [BETA, ALPHA])
        self.assertEqual([hit["chunk_id"] for hit in out["results"]], [2, 1])
        self.assertEqual([hit["rerank_score"] for hit in out["results"]], [4.0, 1.0])
        self.assertTrue(out["reranked"])
        self.assertNotIn("note", out)
        self.assertEqual(out["confidence"], "medium")
        self.assertEqual(out["top_score"], {"rerank_score": 4.0})
        rendered = json.dumps(out)
        self.assertNotIn("exited unexpectedly", rendered)
        self.assertNotIn("RuntimeError", rendered)
        self.assertNotIn("JSONDecodeError", rendered)
        self.assertEqual(len(procs), 2)
        self.assertTrue(procs[0].killed)
        self.assertFalse(procs[1].killed)
        self.assertFalse(any(sql.startswith("SET ") for sql, _params in conn.calls))
        limit = [params["k"] for sql, params in conn.calls if "SELECT c.id" in sql]
        self.assertEqual(limit, [server.RERANK_CANDIDATES])
        self.assertNotEqual(limit, [2])
        for proc in procs:
            sent = json.loads(proc.stdin.buf[0])
            self.assertEqual(sent["query"], "find the clause")
            self.assertEqual(sent["texts"], [ALPHA, BETA])
            self.assertEqual(sent["top_k"], 2)
            self.assertNotIn("exited unexpectedly", proc.stdin.buf[0])
            if proc.stdout is not None:
                proc.stdout.close()
                proc.stdout = None


if __name__ == "__main__":
    unittest.main()
