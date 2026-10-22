"""Raw cosine confidence, and path tokens that are not stripped.

No Postgres, Ollama, or rerank model. The HTTP server is never started.
"""

import asyncio
import unittest
from unittest.mock import patch

import server


TOKEN = "unit-test-bearer"
SNIPPET = "clause-text-must-stay"


class FakeConn:
    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    def run(self, sql, **params):
        self.calls.append((sql, params))
        if sql.lstrip().upper().startswith("SELECT"):
            return self.rows
        return []


def search_row(cosine):
    return (
        1, 10, "tax return", "Title", "disk", "/vault", "Taxes/2024.pdf",
        "2024-01-01", "pdftotext", cosine, SNIPPET,
    )


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


class ConfidenceRoundingTests(unittest.TestCase):
    def search(self, cosine):
        conn = FakeConn([search_row(cosine)])

        def embed(_text):
            return "[0.100000]"

        with patch.multiple(
            server,
            db=lambda: conn,
            embed_query=embed,
            hnsw_ready=lambda _con: False,
            rerank_available=lambda: False,
        ):
            return server.ltm_search(query="taxes", k=1, rerank=False)

    def test_confidence_uses_the_raw_cosine_not_the_rounded_display(self):
        # 0.77996 displays as 0.78 (the medium gate) but is still below it.
        # 0.85496 displays as 0.855 (the high gate) but is still below it.
        # Comparing the rounded value would overstate confidence.
        low = self.search(0.77996)
        self.assertEqual(low["results"][0]["cosine"], 0.78)
        self.assertEqual(low["top_score"], {"cosine": 0.78})
        self.assertEqual(
            low["confidence"],
            "LOW — answer may not be in the corpus (cosine only)",
        )
        self.assertEqual(low["results"][0]["snippet"], SNIPPET)
        self.assertFalse(low["reranked"])

        mid = self.search(0.85496)
        self.assertEqual(mid["results"][0]["cosine"], 0.855)
        self.assertEqual(mid["top_score"], {"cosine": 0.855})
        self.assertEqual(
            mid["confidence"],
            "medium (cosine only — enable rerank for a calibrated verdict)",
        )
        self.assertNotIn("high (cosine only", mid["confidence"])
        self.assertEqual(mid["results"][0]["snippet"], SNIPPET)


class PathTokenPaddingTests(unittest.TestCase):
    def test_spaces_around_a_path_token_do_not_authorize(self):
        # Bearer values are stripped. The path segment is not: padding it must
        # stay a 401, otherwise a looser compare would accept a different token.
        app = App()
        gate = server._BearerGate(app, TOKEN)
        sent = run(gate, {
            "type": "http",
            "path": "/t/ " + TOKEN + " /mcp",
            "headers": [],
        })
        self.assertEqual(status_of(sent), 401)
        self.assertEqual(app.scopes, [])
        body = b"".join(
            msg.get("body", b"") for msg in sent
            if msg["type"] == "http.response.body")
        self.assertEqual(body, b'{"error":"unauthorized"}')
        self.assertNotIn(TOKEN.encode(), body)
        self.assertNotIn(b"/mcp", body)
