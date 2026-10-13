"""Session safety, corpus stats, and the rerank protocol.

No Postgres, Ollama, or rerank model. The worker process is faked: these
tests lock the read-only session, the HNSW cache, stats that must not leak
probe errors, the candidate-pool limit, and one-retry worker handling.
"""

import io
import json
import os
import tempfile
import unittest
from unittest.mock import patch

import server

TOKEN = "unit-test-bearer"


class ScriptedConn:
    def __init__(self, rows_for_sql):
        self.rows_for_sql = rows_for_sql
        self.calls = []

    def run(self, sql, **params):
        self.calls.append((sql, params))
        for prefix, rows in self.rows_for_sql:
            if sql.startswith(prefix):
                return rows
        if sql.startswith("SELECT"):
            return []
        return []


def search_row():
    return (
        1, 10, "lease", "Title", "disk", "/vault", "a.pdf",
        None, "pdftotext", 0.9, "clause text",
    )


class ReadOnlySessionTests(unittest.TestCase):
    def setUp(self):
        self._saved = server._con
        server._con = None

    def tearDown(self):
        server._con = self._saved

    def test_new_and_reconnected_sessions_are_forced_read_only(self):
        created = []

        class Conn:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.calls = []
                self.fail_ping = False
                created.append(self)

            def run(self, sql, **_params):
                self.calls.append(sql)
                if sql == "SELECT 1" and self.fail_ping:
                    raise RuntimeError("password=hunter2 dbname=brain")
                return [(1,)]

        with patch.object(server.pg8000.native, "Connection", Conn):
            first = server.db()
            self.assertIs(first, created[0])
            self.assertEqual(first.kwargs["host"], server.PGHOST)
            self.assertEqual(first.kwargs["port"], server.PGPORT)
            self.assertEqual(first.kwargs["user"], server.PGUSER)
            self.assertEqual(first.kwargs["database"], server.PGDATABASE)
            self.assertNotIn("password", first.kwargs)
            self.assertEqual(
                first.calls,
                [
                    "SET default_transaction_read_only = on",
                    "SET statement_timeout = '180s'",
                ],
            )

            again = server.db()
            self.assertIs(again, first)
            self.assertEqual(first.calls[-1], "SELECT 1")

            first.fail_ping = True
            second = server.db()

        self.assertIs(second, created[1])
        self.assertIsNot(second, first)
        self.assertIs(server._con, second)
        self.assertEqual(
            second.calls[:2],
            [
                "SET default_transaction_read_only = on",
                "SET statement_timeout = '180s'",
            ],
        )
        self.assertNotIn("hunter2", json.dumps(second.calls))


class HnswCacheTests(unittest.TestCase):
    def setUp(self):
        self._saved = server._hnsw_ready
        server._hnsw_ready = False

    def tearDown(self):
        server._hnsw_ready = self._saved

    def test_invalid_or_missing_index_rechecks_and_a_valid_one_sticks(self):
        class Conn:
            def __init__(self, rows):
                self.rows = rows
                self.n = 0
                self.sql = ""

            def run(self, sql, **_params):
                self.n += 1
                self.sql = sql
                return self.rows

        missing = Conn([])
        self.assertFalse(server.hnsw_ready(missing))
        self.assertIn("indisvalid", missing.sql)
        self.assertIn("emb_mxbai", missing.sql)
        self.assertIn("hnsw", missing.sql)
        self.assertFalse(server.hnsw_ready(missing))
        self.assertEqual(missing.n, 2)

        invalid = Conn([(False,), (None,)])
        self.assertFalse(server.hnsw_ready(invalid))
        self.assertEqual(invalid.n, 1)

        mixed = Conn([(False,), (True,)])
        self.assertTrue(server.hnsw_ready(mixed))
        queries = mixed.n
        mixed.rows = [(False,)]
        self.assertTrue(server.hnsw_ready(mixed))
        self.assertEqual(mixed.n, queries)


class StatsTests(unittest.TestCase):
    def stats(self, counts, by_type, urlopen):
        conn = ScriptedConn([
            ("SELECT count(*) FROM documents", [(counts[0],)]),
            ("SELECT count(*) FROM doc_chunks", [(counts[1],)]),
            ("SELECT count(*) FROM emb_mxbai", [(counts[2],)]),
            ("SELECT COALESCE", list(by_type)),
            ("SELECT pg_size_pretty", [("1 MB", "2 MB", "3 MB", "9 MB")]),
        ])
        with patch.multiple(
            server,
            db=lambda: conn,
            hnsw_ready=lambda _con: True,
            rerank_available=lambda: False,
        ), patch.object(server.urllib.request, "urlopen", urlopen):
            out = server.ltm_stats()
        return out, conn

    def test_coverage_grouping_and_probe_errors_hide_the_exception_text(self):
        def urlopen(url, timeout=0):
            self.assertEqual(timeout, 5)
            self.assertTrue(str(url).endswith("/api/tags"))
            self.assertNotIn("/api/embeddings", str(url))
            raise server.urllib.error.URLError("token=SECRET-IN-MESSAGE")

        out, conn = self.stats((4, 3, 1), [("lease", 3), ("untyped", 1)], urlopen)
        self.assertEqual(out["documents"], 4)
        self.assertEqual(out["chunks"], 3)
        self.assertEqual(out["mxbai_embeddings"], 1)
        self.assertEqual(out["embedding_coverage"], "33.33%")
        self.assertEqual(out["documents_by_type"], {"lease": 3, "untyped": 1})
        self.assertEqual(out["table_sizes"]["documents"], "1 MB")
        self.assertEqual(out["table_sizes"]["doc_chunks"], "2 MB")
        self.assertEqual(out["table_sizes"]["emb_mxbai"], "3 MB")
        self.assertEqual(out["table_sizes"]["database"], "9 MB")
        self.assertTrue(out["read_only"])
        self.assertIn("ready", out["hnsw_index"])
        self.assertIn("unavailable", out["reranker"])
        self.assertEqual(out["ollama"], "unreachable (URLError)")
        self.assertNotIn("SECRET", json.dumps(out))

        grouped = next(sql for sql, _params in conn.calls if sql.startswith("SELECT COALESCE"))
        self.assertIn("NULLIF(doc_type, '')", grouped)
        self.assertIn("'untyped'", grouped)

    def test_zero_chunks_report_coverage_as_not_applicable(self):
        def urlopen(_url, timeout=0):
            raise TimeoutError("slow")

        out, _conn = self.stats((0, 0, 5), [], urlopen)
        self.assertEqual(out["embedding_coverage"], "n/a")
        self.assertEqual(out["documents_by_type"], {})
        self.assertEqual(out["ollama"], "unreachable (TimeoutError)")

    def test_ollama_model_presence_matches_the_name_prefix(self):
        seen = {}

        class Resp(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

        def urlopen(url, timeout=0):
            seen["url"] = url
            payload = seen.get("payload")
            return Resp(json.dumps(payload).encode())

        seen["payload"] = {"models": [{"name": server.EMBED_MODEL + ":latest"}]}
        out, _conn = self.stats((1, 1, 1), [], urlopen)
        self.assertEqual(
            out["ollama"], "ok (" + server.EMBED_MODEL + " present)")
        self.assertTrue(str(seen["url"]).endswith("/api/tags"))

        seen["payload"] = {"models": [{"name": "nomic-embed-text"}]}
        out, _conn = self.stats((1, 1, 1), [], urlopen)
        self.assertIn("NOT pulled", out["ollama"])
        self.assertNotIn("present)", out["ollama"])


class CandidatePoolTests(unittest.TestCase):
    def test_rerank_queries_the_candidate_pool_and_asks_for_k(self):
        conn = ScriptedConn([("SELECT", [search_row()])])
        seen = {}

        def rerank(query, texts, top_k):
            seen["rerank"] = (query, texts, top_k)
            return [0], [8.0]

        with patch.multiple(
            server,
            db=lambda: conn,
            embed_query=lambda text: "[0.100000]",
            hnsw_ready=lambda _con: True,
            rerank_available=lambda: True,
            run_rerank=rerank,
        ):
            out = server.ltm_search(
                "find the clause", k=3, rerank=True, doc_type="lease")

        select_sql, select = next(
            (sql, params) for sql, params in conn.calls if sql.startswith("SELECT"))
        self.assertEqual(select["k"], server.RERANK_CANDIDATES)
        self.assertGreater(select["k"], 3)
        self.assertEqual(select["dt"], "lease")
        self.assertEqual(seen["rerank"][2], 3)
        self.assertEqual(seen["rerank"][1], ["clause text"])
        self.assertEqual(out["results"][0]["rerank_score"], 8.0)
        sets = [sql for sql, _params in conn.calls if sql.startswith("SET")]
        self.assertIn("SET hnsw.iterative_scan = relaxed_order", sets)


class BlankTokenTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.addCleanup(setattr, server, "TOKEN_FILE", server.TOKEN_FILE)
        self.path = os.path.join(self.tmp.name, "http-token")
        server.TOKEN_FILE = self.path

    def test_blank_token_file_is_reminted_and_not_logged(self):
        with open(self.path, "w", encoding="utf-8") as fh:
            fh.write(" \n")
        with self.assertLogs("ltm", level="INFO") as captured:
            tok = server._load_or_create_token()
        self.assertTrue(tok.strip())
        self.assertNotEqual(tok, " ")
        self.assertNotIn(tok, "\n".join(captured.output))
        with open(self.path, encoding="utf-8") as fh:
            self.assertEqual(fh.read(), tok + "\n")
        self.assertEqual(server._load_or_create_token(), tok)


class FirstAuthorizationHeaderTests(unittest.TestCase):
    def gate(self, headers):
        app_seen = []

        async def app(scope, receive, send):
            app_seen.append(scope)
            await send({"type": "http.response.start", "status": 204, "headers": []})
            await send({"type": "http.response.body", "body": b""})

        sent = []

        async def send(message):
            sent.append(message)

        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        import asyncio
        asyncio.run(server._BearerGate(app, TOKEN)(
            {"type": "http", "path": "/mcp", "headers": headers},
            receive,
            send,
        ))
        status = next(msg["status"] for msg in sent if msg["type"] == "http.response.start")
        return status, app_seen

    def test_the_first_authorization_header_wins(self):
        denied, seen = self.gate([
            (b"authorization", b"Bearer wrong"),
            (b"authorization", b"Bearer " + TOKEN.encode()),
        ])
        self.assertEqual(denied, 401)
        self.assertEqual(seen, [])

        allowed, seen = self.gate([
            (b"Authorization", b"Bearer " + TOKEN.encode()),
            (b"authorization", b"Bearer wrong"),
        ])
        self.assertEqual(allowed, 204)
        self.assertEqual(seen[0]["path"], "/mcp")


class Proc:
    def __init__(self, exitcode=None):
        self.exitcode = exitcode
        self.killed = False
        self.stdin = io.StringIO()

    def poll(self):
        return self.exitcode

    def kill(self):
        self.killed = True
        self.exitcode = -9


class RerankProtocolTests(unittest.TestCase):
    def setUp(self):
        self._saved = server._rr_proc
        server._rr_proc = None

    def tearDown(self):
        server._rr_proc = self._saved

    def test_a_dead_worker_is_replaced_once_and_a_second_failure_raises(self):
        procs = []

        def worker():
            proc = Proc()
            procs.append(proc)
            return proc

        def read_line(proc, timeout):
            self.assertEqual(timeout, server.RERANK_TIMEOUT_S)
            if proc is procs[0]:
                raise TimeoutError("reranker did not answer within 300s")
            self.assertIn('"query": "lease"', proc.stdin.getvalue())
            self.assertIn('"top_k": 2', proc.stdin.getvalue())
            return {"order": [1, 0], "scores": [9.0, 1.0]}

        with patch.object(server, "_rr_worker", worker), patch.object(
            server, "_rr_read_line", read_line
        ):
            order, scores = server.run_rerank("lease", ["a", "b"], 2)

        self.assertEqual(order, [1, 0])
        self.assertEqual(scores, [9.0, 1.0])
        self.assertTrue(procs[0].killed)
        self.assertFalse(procs[1].killed)

        def always_fail():
            proc = Proc()
            procs.append(proc)
            return proc

        def fail_read(_proc, _timeout):
            raise RuntimeError("reranker worker exited unexpectedly")

        with patch.object(server, "_rr_worker", always_fail), patch.object(
            server, "_rr_read_line", fail_read
        ):
            with self.assertRaises(RuntimeError):
                server.run_rerank("lease", ["a"], 1)

        self.assertEqual(len(procs), 4)
        self.assertTrue(procs[2].killed)
        self.assertTrue(procs[3].killed)
        self.assertIsNone(server._rr_proc)

    def test_an_error_payload_is_retried_then_surfaced(self):
        procs = []

        def worker():
            proc = Proc()
            procs.append(proc)
            return proc

        def read_line(proc, _timeout):
            if proc is procs[0]:
                return {"error": "model exploded"}
            return {"error": "still exploded"}

        with patch.object(server, "_rr_worker", worker), patch.object(
            server, "_rr_read_line", read_line
        ):
            with self.assertRaises(RuntimeError) as caught:
                server.run_rerank("q", ["a"], 1)

        self.assertIn("still exploded", str(caught.exception))
        self.assertEqual(len(procs), 2)
        self.assertTrue(all(proc.killed for proc in procs))

    def test_unready_worker_is_killed_and_a_ready_one_is_reused(self):
        launched = []

        def popen(argv, **kwargs):
            self.assertEqual(argv[1:3], ["-u", "-c"])
            self.assertTrue(kwargs.get("text"))
            self.assertIs(kwargs.get("stdin"), server.subprocess.PIPE)
            self.assertIs(kwargs.get("stdout"), server.subprocess.PIPE)
            proc = Proc()
            launched.append(proc)
            return proc

        replies = [{"ready": False, "error": "no model"}]

        def read_line(_proc, _timeout):
            return replies[0]

        with patch.object(server.subprocess, "Popen", popen), patch.object(
            server, "_rr_read_line", read_line
        ):
            with self.assertRaises(RuntimeError) as caught:
                server._rr_worker()
        self.assertIn("failed to start", str(caught.exception))
        self.assertTrue(launched[0].killed)
        self.assertIsNone(server._rr_proc)

        replies[:] = [{"ready": True}]
        with patch.object(server.subprocess, "Popen", popen), patch.object(
            server, "_rr_read_line", read_line
        ):
            first = server._rr_worker()
            first.exitcode = 0
            second = server._rr_worker()

        self.assertIsNot(first, second)
        self.assertIs(server._rr_proc, second)
        self.assertFalse(second.killed)
        third = server._rr_worker()
        self.assertIs(third, second)
        self.assertEqual(len(launched), 3)


if __name__ == "__main__":
    unittest.main()
