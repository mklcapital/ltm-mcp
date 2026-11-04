"""Path-token traversal and the embed prefix.

No Postgres or Ollama. The bearer gate is an in-process ASGI app, and the
embedder is a scripted urlopen.
"""

import asyncio
import json
import unittest
from unittest.mock import patch

import server

TOKEN = "unit-test-bearer"


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


class PathTraversalTests(unittest.TestCase):
    def test_dotdot_cannot_reach_a_later_token_or_keep_this_one(self):
        # The first segment is the token. `..` does not reparse the path, so a
        # wrong segment cannot climb into a later copy of the real token.
        app = App()
        sent = run(server._BearerGate(app, TOKEN), {
            "type": "http",
            "path": "/t/nope/../t/%s/mcp" % TOKEN,
            "headers": [],
        })
        self.assertEqual(status_of(sent), 401)
        self.assertEqual(body_of(sent), b'{"error":"unauthorized"}')
        self.assertNotIn(TOKEN.encode(), body_of(sent))
        self.assertEqual(app.scopes, [])

        # A valid token is stripped and the tail is forwarded unchanged.
        # Collapsing `..` here would hide where the request is actually routed.
        app = App()
        sent = run(server._BearerGate(app, TOKEN), {
            "type": "http",
            "path": "/t/%s/../../mcp" % TOKEN,
            "headers": [],
        })
        self.assertEqual(status_of(sent), 204)
        seen = app.scopes[0]
        self.assertEqual(seen["path"], "/../../mcp")
        self.assertEqual(seen["raw_path"], b"/../../mcp")
        self.assertNotIn(TOKEN, seen["path"])
        self.assertNotIn(TOKEN.encode(), seen["raw_path"])


class EmbedPrefixTests(unittest.TestCase):
    def prompt_for(self, query):
        seen = {}

        class Resp:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return json.dumps({"embedding": [1]}).encode()

        def urlopen(req, timeout=0):
            self.assertEqual(timeout, 60)
            seen["body"] = json.loads(req.data.decode())
            return Resp()

        with patch.object(server.urllib.request, "urlopen", urlopen):
            out = server.embed_query(query)
        self.assertEqual(out, "[1.000000]")
        self.assertEqual(seen["body"]["model"], server.EMBED_MODEL)
        return seen["body"]["prompt"]

    def test_the_prefix_is_applied_even_when_the_query_is_empty_or_already_prefixed(self):
        # Documents were embedded raw. A blank query, or one that already
        # starts with the model-card prefix, must still be prefixed — not sent raw.
        prefix = server.QUERY_PREFIX
        self.assertEqual(self.prompt_for(""), prefix)
        doubled = self.prompt_for(prefix)
        self.assertEqual(doubled, prefix + prefix)
        self.assertEqual(doubled.count(prefix), 2)
