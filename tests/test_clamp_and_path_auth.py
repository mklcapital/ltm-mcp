"""Negative limits must not flip into suffixes, and a path token must be exact.

No Postgres, Ollama, or rerank model. The HTTP server is never started.
"""

import asyncio
import json
import unittest
from unittest.mock import patch

import server


SECRET = "FULL_TEXT_SHOULD_NOT_LEAK"
TAIL = "TAIL_SHOULD_NOT_BE_THE_ONLY_RESULT"
TOKEN = "unit-test-bearer"


def search_row(chunk_id, cosine, text):
    return (
        chunk_id, 10, "tax return", "Title", "disk", "/vault", "Taxes/2024.pdf",
        "2024-01-01", "pdftotext", cosine, text,
    )


def doc_row(text):
    return (
        7, "disk", "/vault", "Taxes/a.pdf", "application/pdf", "lease", "Lease",
        None, 100, 1, len(text), "pdftotext", 0, None, text,
    )


class ScriptedConn:
    def __init__(self, rules):
        self.rules = rules
        self.calls = []

    def run(self, sql, **params):
        self.calls.append((sql, params))
        for needle, rows in self.rules:
            if needle in sql:
                return rows
        return []


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
    return b"".join(msg.get("body", b"") for msg in sent if msg["type"] == "http.response.body")


class ClampTests(unittest.TestCase):
    def test_a_negative_k_returns_the_first_hit_not_the_tail(self):
        # int is truthy for -4, so `k or 1` would not clamp it. rows[:-4] on a
        # short page is empty; on a longer page it drops the tail and returns
        # the rest. The bound limit and the returned hit must be the first one.
        conn = ScriptedConn([("SELECT", [
            search_row(1, 0.91, "first hit"),
            search_row(2, 0.90, "second hit"),
            search_row(3, 0.80, TAIL),
        ])])
        with patch.multiple(
            server,
            db=lambda: conn,
            embed_query=lambda _text: "[0.100000]",
            hnsw_ready=lambda _con: False,
            rerank_available=lambda: False,
        ):
            out = server.ltm_search(query="taxes", k=-4, rerank=False)

        self.assertEqual([hit["chunk_id"] for hit in out["results"]], [1])
        self.assertEqual(out["results"][0]["snippet"], "first hit")
        rendered = json.dumps(out)
        self.assertNotIn(TAIL, rendered)
        self.assertNotIn("second hit", rendered)
        select = next(params for sql, params in conn.calls if sql.startswith("SELECT"))
        self.assertEqual(select["k"], 1)

    def test_a_negative_max_chars_returns_a_prefix(self):
        # text[:-1] would keep almost the whole document, including the suffix.
        text = "PREFIX" + ("x" * 400) + "SUFFIX_SHOULD_NOT_BE_RETURNED"
        conn = ScriptedConn([("WHERE id", [doc_row(text)])])
        with patch.object(server, "db", lambda: conn):
            out = server.ltm_get_document("7", max_chars=-1)

        self.assertTrue(out["found"])
        self.assertTrue(out["truncated"])
        self.assertEqual(len(out["text"]), 200)
        self.assertTrue(out["text"].startswith("PREFIX"))
        self.assertNotIn("SUFFIX_SHOULD_NOT_BE_RETURNED", out["text"])
        self.assertEqual(out["char_count"], len(text))
        self.assertEqual(conn.calls[0][1]["v"], 7)


class PathAuthTests(unittest.TestCase):
    def test_a_decoy_segment_tab_scheme_or_encoded_slash_does_not_authorize(self):
        app = App()
        gate = server._BearerGate(app, TOKEN)
        cases = (
            "/t/nope/t/" + TOKEN + "/mcp",
            [],
            "/mcp",
            [(b"authorization", b"Bearer\t" + TOKEN.encode())],
            "/t/" + TOKEN + "%2Fmcp",
            [],
        )
        for path, headers in zip(cases[0::2], cases[1::2]):
            app.scopes.clear()
            sent = run(gate, {"type": "http", "path": path, "headers": headers})
            self.assertEqual(status_of(sent), 401, path)
            self.assertEqual(body_of(sent), b'{"error":"unauthorized"}')
            self.assertNotIn(TOKEN.encode(), body_of(sent))
            self.assertEqual(app.scopes, [], path)


if __name__ == "__main__":
    unittest.main()
