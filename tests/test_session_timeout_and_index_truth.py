"""Session timeout is re-applied, and an integer HNSW flag sticks only when true.

No Postgres, Ollama, or rerank model. The HTTP server is never started.
"""

import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import server


TOKEN = "unit-test-bearer"


class FakeConn:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.runs = []
        self.fail_probe = False

    def run(self, sql, **_params):
        self.runs.append(sql)
        if sql == "SELECT 1" and self.fail_probe:
            self.fail_probe = False
            raise RuntimeError("probe down SECRET-PROBE-MESSAGE")
        return []


class SessionTimeoutTests(unittest.TestCase):
    def setUp(self):
        self._saved = server._con
        server._con = None
        self.made = []

        def connect(**kwargs):
            conn = FakeConn(**kwargs)
            self.made.append(conn)
            return conn

        self._patch = patch.object(server.pg8000.native, "Connection", side_effect=connect)
        self._patch.start()

    def tearDown(self):
        self._patch.stop()
        server._con = self._saved

    def test_statement_timeout_is_set_again_after_reconnect(self):
        first = server.db()
        self.assertIs(first, self.made[0])
        self.assertEqual(
            first.kwargs,
            {"host": "127.0.0.1", "port": 5433, "user": "postgres", "database": "brain"},
        )
        self.assertNotIn("password", first.kwargs)
        self.assertEqual(
            first.runs,
            [
                "SET default_transaction_read_only = on",
                "SET statement_timeout = '180s'",
            ],
        )
        first.fail_probe = True
        second = server.db()
        self.assertIsNot(second, first)
        self.assertEqual(len(self.made), 2)
        self.assertEqual(first.runs[2], "SELECT 1")
        self.assertEqual(
            second.runs,
            [
                "SET default_transaction_read_only = on",
                "SET statement_timeout = '180s'",
            ],
        )
        self.assertNotIn("SECRET-PROBE-MESSAGE", " ".join(second.runs))
        again = server.db()
        self.assertIs(again, second)
        self.assertEqual(second.runs[-1], "SELECT 1")


class QueueConn:
    def __init__(self, batches):
        self.batches = list(batches)
        self.calls = []

    def run(self, sql, **_params):
        self.calls.append(sql)
        if not self.batches:
            return []
        return self.batches.pop(0)


class HnswIntegerTests(unittest.TestCase):
    def setUp(self):
        self._saved = server._hnsw_ready
        server._hnsw_ready = False

    def tearDown(self):
        server._hnsw_ready = self._saved

    def test_integer_zero_rechecks_and_integer_one_sticks(self):
        conn = QueueConn([[(0,)], [(1,)]])
        self.assertFalse(server.hnsw_ready(conn))
        self.assertEqual(len(conn.calls), 1)
        self.assertTrue(server.hnsw_ready(conn))
        self.assertEqual(len(conn.calls), 2)
        self.assertTrue(server.hnsw_ready(conn))
        self.assertEqual(len(conn.calls), 2, "a truthy index must not be re-queried")
        for sql in conn.calls:
            self.assertIn("indisvalid", sql)
            self.assertIn("emb_mxbai", sql)
            self.assertIn("hnsw", sql)
            self.assertNotIn("SECRET", sql)


class BearerInternalSpaceTests(unittest.IsolatedAsyncioTestCase):
    async def test_internal_space_does_not_authorize(self):
        class App:
            def __init__(self):
                self.calls = 0

            async def __call__(self, scope, receive, send):
                self.calls += 1
                self.scope = scope

        app = App()
        gate = server._BearerGate(app, TOKEN)
        sent = []

        async def send(msg):
            sent.append(msg)

        async def receive():
            return {"type": "http.request", "body": b""}

        scope = {
            "type": "http",
            "path": "/mcp",
            "headers": [(b"authorization", b"Bearer unit- test-bearer")],
            "query_string": b"",
        }
        await gate(scope, receive, send)
        self.assertEqual(app.calls, 0)
        self.assertEqual(sent[0]["status"], 401)
        self.assertEqual(sent[1]["body"], b'{"error":"unauthorized"}')
        self.assertNotIn(TOKEN.encode(), sent[1]["body"])

        sent.clear()
        scope = {
            "type": "http",
            "path": "/mcp",
            "headers": [(b"authorization", f"Bearer {TOKEN}".encode())],
            "query_string": b"",
        }
        await gate(scope, receive, send)
        self.assertEqual(app.calls, 1)
        self.assertEqual(sent, [])
        self.assertEqual(app.scope["path"], "/mcp")


if __name__ == "__main__":
    unittest.main()
