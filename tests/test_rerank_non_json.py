"""A non-JSON rerank line falls back without returning the worker bytes.

No Postgres, Ollama, or rerank model. The HTTP server is never started.
"""

import json
import os
import unittest
from unittest.mock import patch

import server


ALPHA = "ALPHA_COSINE_FIRST"
BETA = "BETA_SECOND_CHUNK"
LEAK = "LEAKED-NOT-JSON"


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
    """One ready stdout line, so the real ``_rr_read_line`` parses it."""

    def __init__(self, line):
        read_fd, write_fd = os.pipe()
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


class RerankNonJsonTests(unittest.TestCase):
    def tearDown(self):
        server._rr_proc = None

    def test_non_json_worker_line_is_not_returned(self):
        # json.loads puts the bad line in the exception text. The tool result
        # may name the exception type, and must not echo the worker bytes.
        rows = [
            search_row(1, 0.99, ALPHA),
            search_row(2, 0.80, BETA),
        ]
        conn = ScriptedConn(rows)
        procs = []

        def worker():
            proc = Proc((LEAK + "\n").encode())
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

        self.assertEqual([hit["snippet"] for hit in out["results"]], [ALPHA, BETA])
        self.assertEqual([hit["chunk_id"] for hit in out["results"]], [1, 2])
        self.assertFalse(out["reranked"])
        self.assertNotIn("rerank_score", out["results"][0])
        self.assertEqual(
            out["note"],
            "rerank failed (JSONDecodeError) — cosine order only",
        )
        self.assertEqual(
            out["confidence"],
            "high (cosine only — enable rerank for a calibrated verdict)",
        )
        rendered = json.dumps(out)
        self.assertNotIn(LEAK, rendered)
        self.assertNotIn("Expecting value", rendered)
        self.assertEqual(len(procs), 2)
        self.assertTrue(all(proc.killed for proc in procs))
        self.assertFalse(any(sql.startswith("SET ") for sql, _params in conn.calls))
        for proc in procs:
            sent = json.loads(proc.stdin.buf[0])
            self.assertEqual(sent["query"], "find the clause")
            self.assertEqual(sent["texts"], [ALPHA, BETA])
            self.assertEqual(sent["top_k"], 2)
            self.assertNotIn(LEAK, proc.stdin.buf[0])
