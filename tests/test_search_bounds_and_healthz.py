"""Search bounds and the exact health check. No Postgres, Ollama, or network.

Covers the rerank opt-out, null search fields, snippet truncation, and
health-check paths that must not skip the bearer gate.
"""

import asyncio
import json
import unittest
from unittest.mock import patch

import server

TOKEN = "unit-test-bearer"
HIGH_COSINE = "high (cosine only — enable rerank for a calibrated verdict)"


def search_row(chunk_id, cosine, text, root="/vault", relpath="Taxes/2024.pdf", dtype="tax return"):
    return (
        chunk_id, 10, dtype, "Title", "disk", root, relpath,
        "2024-01-01", "pdftotext", cosine, text,
    )


class FakeConn:
    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    def run(self, sql, **params):
        self.calls.append((sql, params))
        if sql.lstrip().upper().startswith("SELECT"):
            return self.rows
        return []


class SearchBoundTests(unittest.TestCase):
    def search(self, rows, available=False, rerank_impl=None, **kwargs):
        conn = FakeConn(rows)
        seen = {}

        def embed(text):
            seen["embed"] = text
            return "[0.100000]"

        def rerank(query, texts, top_k):
            seen["rerank"] = (query, texts, top_k)
            if rerank_impl is None:
                raise AssertionError("rerank was not expected")
            return rerank_impl(query, texts, top_k)

        with patch.multiple(
            server,
            db=lambda: conn,
            embed_query=embed,
            hnsw_ready=lambda _con: False,
            rerank_available=lambda: available,
            run_rerank=rerank,
        ):
            out = server.ltm_search(**kwargs)
        return out, conn, seen

    def test_rerank_false_does_not_send_chunk_text_when_the_reranker_exists(self):
        secret = "CHUNK_TEXT_MUST_STAY_OUT_OF_RERANK"
        rows = [search_row(1, 0.91, secret)]

        def impl(*_args):
            raise AssertionError("rerank=False must not call the worker")

        out, _conn, seen = self.search(
            rows,
            available=True,
            rerank_impl=impl,
            query="taxes",
            k=8,
            rerank=False,
        )
        self.assertNotIn("rerank", seen)
        self.assertFalse(out["reranked"])
        self.assertNotIn("note", out)
        self.assertEqual(out["confidence"], HIGH_COSINE)
        self.assertNotIn("rerank_score", out["results"][0])
        self.assertEqual(out["results"][0]["snippet"], secret)

    def test_search_snippets_drop_the_tail_of_a_long_chunk(self):
        tail = "TAIL_MUST_NOT_LEAK"
        text = ("y" * (server.SNIPPET_CHARS + 25)) + tail
        out, _conn, seen = self.search(
            [search_row(1, 0.9, text)],
            query="q",
            k=1,
            rerank=False,
        )
        snippet = out["results"][0]["snippet"]
        self.assertEqual(len(snippet), server.SNIPPET_CHARS)
        self.assertNotIn("rerank", seen)
        rendered = json.dumps(out)
        self.assertNotIn(tail, rendered)
        self.assertNotIn(text, rendered)

    def test_null_root_and_chunk_text_do_not_crash_or_stringify_none(self):
        row = (
            1, 10, None, None, None, None, None,
            None, None, 0.5, None,
        )

        def impl(_query, texts, _top_k):
            self.assertEqual(texts, [""])
            return [0], [4.0]

        out, _conn, seen = self.search(
            [row],
            available=True,
            rerank_impl=impl,
            query="q",
            k=1,
            rerank=True,
        )
        hit = out["results"][0]
        self.assertEqual(hit["path"], "/")
        self.assertEqual(hit["snippet"], "")
        self.assertEqual(hit["doc_type"], "untyped")
        self.assertIsNone(hit["doc_date"])
        self.assertNotIn("None", hit["path"])
        self.assertNotIn("None", hit["snippet"])
        self.assertEqual(seen["rerank"][1], [""])
        self.assertEqual(hit["rerank_score"], 4.0)


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


def body_of(sent):
    return next(msg.get("body", b"") for msg in sent if msg["type"] == "http.response.body")


class HealthzBoundaryTests(unittest.TestCase):
    def test_only_the_exact_healthz_path_is_open(self):
        for path in ("/healthz/", "/healthz/extra", "/Healthz"):
            app = App()
            sent = run(server._BearerGate(app, TOKEN), {
                "type": "http",
                "path": path,
                "headers": [],
            })
            self.assertEqual(status_of(sent), 401, path)
            self.assertEqual(body_of(sent), b'{"error":"unauthorized"}')
            self.assertEqual(app.scopes, [], path)
            self.assertNotIn(TOKEN.encode(), body_of(sent))

    def test_a_path_token_with_no_tail_is_rewritten_to_root(self):
        app = App()
        sent = run(server._BearerGate(app, TOKEN), {
            "type": "http",
            "path": "/t/" + TOKEN,
            "headers": [],
        })
        self.assertEqual(status_of(sent), 204)
        self.assertEqual(len(app.scopes), 1)
        seen = app.scopes[0]
        self.assertEqual(seen["path"], "/")
        self.assertEqual(seen["raw_path"], b"/")
        self.assertNotIn(TOKEN, seen["path"])
        self.assertNotIn(TOKEN.encode(), seen["raw_path"])


if __name__ == "__main__":
    unittest.main()
