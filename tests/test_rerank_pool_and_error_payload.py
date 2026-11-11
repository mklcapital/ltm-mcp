"""Rerank candidate limits and worker error payloads.

No Postgres, Ollama, or rerank model. The HTTP server is never started.
"""

import json
import unittest
from unittest.mock import patch

import server


ALPHA = "ALPHA_COSINE_FIRST"
BETA = "BETA_RERANK_CANDIDATE"
LEAK = "LEAKED-MODEL-PATH"


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
    def __init__(self):
        self.stdin = Pipe()
        self.killed = False

    def kill(self):
        self.killed = True


class RerankPoolTests(unittest.TestCase):
    def tearDown(self):
        server._rr_proc = None

    def test_limit_is_k_when_k_exceeds_the_candidate_pool(self):
        # fetch_k is max(pool, k). A pool smaller than k must not cap the SQL
        # limit, and the worker still reranks only the requested k.
        rows = [search_row(i, 0.9 - i * 0.01, "chunk-%d" % i) for i in range(4)]
        conn = ScriptedConn(rows)
        seen = {}

        def rerank(query, texts, top_k):
            seen["query"] = query
            seen["texts"] = texts
            seen["top_k"] = top_k
            return [0, 1, 2, 3], [4.0, 3.0, 2.0, 1.0]

        with patch.multiple(
            server,
            db=lambda: conn,
            embed_query=lambda _text: "[0.100000]",
            hnsw_ready=lambda _con: False,
            rerank_available=lambda: True,
            run_rerank=rerank,
            RERANK_CANDIDATES=3,
        ):
            out = server.ltm_search("find the clause", k=4, rerank=True)

        self.assertEqual(seen["top_k"], 4)
        self.assertEqual(seen["query"], "find the clause")
        self.assertEqual(len(seen["texts"]), 4)
        select = [params for sql, params in conn.calls if "SELECT c.id" in sql]
        self.assertEqual(len(select), 1)
        self.assertEqual(select[0]["k"], 4)
        self.assertNotEqual(select[0]["k"], 3)
        self.assertEqual(len(out["results"]), 4)
        self.assertTrue(out["reranked"])
        self.assertFalse(any(sql.startswith("SET ") for sql, _params in conn.calls))


class RerankErrorPayloadTests(unittest.TestCase):
    def tearDown(self):
        server._rr_proc = None

    def search(self, responses):
        rows = [
            search_row(1, 0.99, ALPHA),
            search_row(2, 0.80, BETA),
        ]
        conn = ScriptedConn(rows)
        procs = []
        pending = list(responses)

        def worker():
            proc = Proc()
            procs.append(proc)
            return proc

        def read_line(proc, timeout):
            self.assertIs(proc, procs[-1])
            self.assertEqual(timeout, server.RERANK_TIMEOUT_S)
            return pending.pop(0)

        with patch.multiple(
            server,
            db=lambda: conn,
            embed_query=lambda _text: "[0.100000]",
            hnsw_ready=lambda _con: False,
            rerank_available=lambda: True,
            _rr_worker=worker,
            _rr_read_line=read_line,
        ):
            out = server.ltm_search("find the clause", k=2, rerank=True)
        return out, procs

    def test_an_error_payload_is_not_applied_and_is_not_returned(self):
        # The worker can attach a tempting order next to `error`. That order
        # must not reorder hits, and the error string must not reach the tool.
        tempting = {"error": LEAK, "order": [1, 0], "scores": [9.0, 0.1]}
        out, procs = self.search([tempting, dict(tempting)])
        self.assertEqual([hit["snippet"] for hit in out["results"]], [ALPHA, BETA])
        self.assertEqual([hit["chunk_id"] for hit in out["results"]], [1, 2])
        self.assertFalse(out["reranked"])
        self.assertNotIn("rerank_score", out["results"][0])
        self.assertIn("RuntimeError", out["note"])
        self.assertNotIn("error", out["note"])
        rendered = json.dumps(out)
        self.assertNotIn(LEAK, rendered)
        self.assertNotIn("9.0", rendered)
        self.assertEqual(len(procs), 2)
        self.assertTrue(all(proc.killed for proc in procs))
        for proc in procs:
            sent = json.loads(proc.stdin.buf[0])
            self.assertEqual(sent["texts"], [ALPHA, BETA])
            self.assertEqual(sent["top_k"], 2)
            self.assertNotIn(LEAK, proc.stdin.buf[0])

    def test_a_recovered_retry_uses_the_second_payload_only(self):
        out, procs = self.search([
            {"error": LEAK, "order": [0], "scores": [9.0]},
            {"order": [1, 0], "scores": [4.0, 1.0]},
        ])
        self.assertEqual([hit["snippet"] for hit in out["results"]], [BETA, ALPHA])
        self.assertEqual(out["results"][0]["rerank_score"], 4.0)
        self.assertEqual(out["confidence"], "medium")
        self.assertTrue(out["reranked"])
        self.assertNotIn("note", out)
        rendered = json.dumps(out)
        self.assertNotIn(LEAK, rendered)
        self.assertEqual(len(procs), 2)
        self.assertTrue(procs[0].killed)
        self.assertFalse(procs[1].killed)
