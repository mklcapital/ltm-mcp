"""HTTP bind policy, auth-gate failure modes, and search degrade paths.

No Postgres, Ollama, or rerank model. The HTTP server is never actually started.
"""

import asyncio
import json
import os
import unittest
from unittest.mock import patch

import server


TOKEN = "unit-test-bearer"
CHUNK = "CHUNK_TEXT_MUST_NOT_LEAK_INTO_THE_NOTE"


class App:
    def __init__(self):
        self.scopes = []

    async def __call__(self, scope, receive, send):
        self.scopes.append(scope)
        await send({"type": "http.response.start", "status": 204, "headers": []})
        await send({"type": "http.response.body", "body": b""})


def run(gate, scope):
    sent = []

    async def send(message):
        sent.append(message)

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    asyncio.run(gate(scope, receive, send))
    return sent


def status_of(sent):
    return next(msg["status"] for msg in sent if msg["type"] == "http.response.start")


class BearerFailureTests(unittest.TestCase):
    def test_a_nul_in_the_bearer_is_rejected_without_calling_the_app(self):
        # A trailing NUL survives strip() and must not compare equal to the token.
        app = App()
        sent = run(server._BearerGate(app, TOKEN), {
            "type": "http",
            "path": "/mcp",
            "headers": [(b"authorization", b"Bearer " + TOKEN.encode() + b"\x00")],
        })
        self.assertEqual(status_of(sent), 401)
        self.assertEqual(app.scopes, [])
        body = b"".join(
            msg.get("body", b"") for msg in sent if msg["type"] == "http.response.body")
        self.assertNotIn(TOKEN.encode(), body)


class TransportSecurityTests(unittest.TestCase):
    def setUp(self):
        settings = server.mcp.settings
        self._saved = (
            settings.stateless_http,
            settings.host,
            settings.port,
            settings.transport_security,
        )

    def tearDown(self):
        settings = server.mcp.settings
        settings.stateless_http, settings.host, settings.port, settings.transport_security = self._saved

    def _run_http(self, public_host):
        calls = {}

        def fake_run(app, **kwargs):
            calls["app"] = app
            calls["kwargs"] = kwargs

        env = {} if public_host is None else {"LTM_PUBLIC_HOST": public_host}
        with patch.dict(os.environ, env, clear=False):
            if public_host is None:
                os.environ.pop("LTM_PUBLIC_HOST", None)
            with patch.object(server, "_load_or_create_token", return_value=TOKEN):
                with patch.object(server.mcp, "streamable_http_app", return_value=object()):
                    with patch("uvicorn.run", fake_run):
                        server.run_http()
        return calls

    def test_public_host_is_an_origin_allowance_not_a_bind_address(self):
        with self.assertLogs("ltm", level="INFO") as captured:
            calls = self._run_http("  ltm.example.com  ")
        self.assertNotIn(TOKEN, "\n".join(captured.output))
        security = server.mcp.settings.transport_security
        self.assertTrue(security.enable_dns_rebinding_protection)
        self.assertEqual(
            security.allowed_hosts,
            [
                "127.0.0.1",
                "127.0.0.1:%s" % server.HTTP_PORT,
                "localhost",
                "localhost:%s" % server.HTTP_PORT,
                "ltm.example.com",
            ],
        )
        self.assertEqual(security.allowed_origins, ["https://ltm.example.com"])
        self.assertTrue(server.mcp.settings.stateless_http)
        self.assertEqual(calls["kwargs"]["host"], server.HTTP_HOST)
        self.assertEqual(calls["kwargs"]["port"], server.HTTP_PORT)
        self.assertIs(calls["kwargs"]["access_log"], False)
        self.assertNotEqual(calls["kwargs"]["host"], "ltm.example.com")
        self.assertNotIn(TOKEN, json.dumps(calls["kwargs"]))

    def test_blank_public_host_does_not_open_the_allowlist(self):
        calls = self._run_http("   ")
        security = server.mcp.settings.transport_security
        self.assertEqual(
            security.allowed_hosts,
            [
                "127.0.0.1",
                "127.0.0.1:%s" % server.HTTP_PORT,
                "localhost",
                "localhost:%s" % server.HTTP_PORT,
            ],
        )
        self.assertEqual(security.allowed_origins, [])
        self.assertIs(calls["kwargs"]["access_log"], False)

        calls = self._run_http(None)
        security = server.mcp.settings.transport_security
        self.assertEqual(len(security.allowed_hosts), 4)
        self.assertEqual(security.allowed_origins, [])
        self.assertNotIn("", security.allowed_hosts)


class _Conn:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def run(self, sql, **params):
        self.calls.append((sql, params))
        if sql.lstrip().upper().startswith("SELECT"):
            if not self.responses:
                return []
            return self.responses.pop(0)
        return []


def search_row(chunk_id, cosine, text):
    return (
        chunk_id, 10, "lease", "Title", "disk", "/vault", "a.pdf",
        "2024-01-01", "pdftotext", cosine, text,
    )


def doc_row(**overrides):
    fields = {
        "doc_id": 7,
        "src": "disk",
        "root": "/vault",
        "relpath": "Taxes/a.pdf",
        "mime": "application/pdf",
        "dtype": "lease",
        "title": "Lease",
        "doc_date": "2024-04-01",
        "nbytes": 100,
        "chunks": 1,
        "chars": 9,
        "method": "pdftotext",
        "ocr": 0,
        "ingested": None,
        "text": "only-body",
    }
    fields.update(overrides)
    return (
        fields["doc_id"], fields["src"], fields["root"], fields["relpath"],
        fields["mime"], fields["dtype"], fields["title"], fields["doc_date"],
        fields["nbytes"], fields["chunks"], fields["chars"], fields["method"],
        fields["ocr"], fields["ingested"], fields["text"],
    )


class DegradeTests(unittest.TestCase):
    def test_a_corrupt_rerank_order_keeps_cosine_results(self):
        rows = [search_row(1, 0.90, CHUNK), search_row(2, 0.40, "other")]
        conn = _Conn([rows])

        def rerank(_query, _texts, _top_k):
            return [99], [9.0]

        with patch.multiple(
            server,
            db=lambda: conn,
            embed_query=lambda _text: "[0.100000]",
            hnsw_ready=lambda _con: False,
            rerank_available=lambda: True,
            run_rerank=rerank,
        ):
            out = server.ltm_search("lease clause", k=2, rerank=True)

        self.assertFalse(out["reranked"])
        self.assertEqual([hit["chunk_id"] for hit in out["results"]], [1, 2])
        self.assertNotIn("rerank_score", out["results"][0])
        self.assertIn("IndexError", out["note"])
        self.assertNotIn(CHUNK, out["note"])
        self.assertIn(CHUNK, out["results"][0]["snippet"])

    def test_rerank_reader_times_out_instead_of_blocking(self):
        class Proc:
            class Stdout:
                def readline(self):
                    raise AssertionError("readline must not run once the wait has timed out")

            stdout = Stdout()

        seen = {}

        def fake_select(rlist, wlist, xlist, timeout):
            seen["timeout"] = timeout
            seen["watched"] = rlist
            return [], [], []

        with patch("select.select", fake_select):
            with self.assertRaises(TimeoutError) as caught:
                server._rr_read_line(Proc(), 12)

        self.assertEqual(seen["timeout"], 12)
        self.assertIn("12", str(caught.exception))

    def test_unique_fragment_returns_text_and_a_null_root_stays_a_path(self):
        conn = _Conn([[], [doc_row(root=None, relpath="Taxes/a.pdf", ocr=0)]])
        with patch.object(server, "db", lambda: conn):
            out = server.ltm_get_document("Taxes/a")

        self.assertTrue(out["found"])
        self.assertEqual(out["text"], "only-body")
        self.assertEqual(out["path"], "/Taxes/a.pdf")
        self.assertFalse(out["ocr"])
        self.assertIsNone(out["ingested_at"])
        self.assertFalse(out["truncated"])
        self.assertEqual(len(conn.calls), 2)
        self.assertIn("ILIKE :v", conn.calls[1][0])
        self.assertEqual(conn.calls[1][1]["v"], "%Taxes/a%")
        self.assertNotIn("Taxes/a", conn.calls[1][0])


if __name__ == "__main__":
    unittest.main()
