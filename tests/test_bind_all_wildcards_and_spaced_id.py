"""A widened bind host stays off the allowlist, and lookups do not widen.

No Postgres, Ollama, or rerank model. The HTTP server is never started.
"""

import json
import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import server


TOKEN = "unit-test-bearer"
SECRET = "DOC-7-FULL-TEXT-SHOULD-NOT-LEAK"


def doc_row(text, doc_id=7, relpath="Taxes/a.pdf"):
    return (
        doc_id, "disk", "/vault", relpath, "application/pdf", "lease", "Lease",
        None, 100, 1, len(text), "pdftotext", 0, None, text,
    )


class ScriptedConn:
    def __init__(self, rules):
        self.rules = list(rules)
        self.calls = []

    def run(self, sql, **params):
        self.calls.append((sql, params))
        for needle, rows in self.rules:
            if needle in sql:
                return rows
        return []


class BindAllHostTests(unittest.TestCase):
    def setUp(self):
        settings = server.mcp.settings
        self._saved = (
            settings.stateless_http,
            settings.host,
            settings.port,
            settings.transport_security,
            server.HTTP_HOST,
        )

    def tearDown(self):
        settings = server.mcp.settings
        (
            settings.stateless_http,
            settings.host,
            settings.port,
            settings.transport_security,
            server.HTTP_HOST,
        ) = self._saved

    def test_bind_all_is_not_added_to_the_allowlist(self):
        # LTM_HTTP_HOST=0.0.0.0 listens on every interface. That address must
        # not also become a trusted Host or Origin.
        server.HTTP_HOST = "0.0.0.0"
        calls = {}

        def fake_run(app, **kwargs):
            calls["app"] = app
            calls["kwargs"] = kwargs

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("LTM_PUBLIC_HOST", None)
            with patch.object(server, "_load_or_create_token", return_value=TOKEN):
                with patch.object(server.mcp, "streamable_http_app", return_value=object()):
                    with patch("uvicorn.run", fake_run):
                        server.run_http()

        security = server.mcp.settings.transport_security
        port = server.HTTP_PORT
        self.assertEqual(calls["kwargs"]["host"], "0.0.0.0")
        self.assertEqual(calls["kwargs"]["port"], port)
        self.assertIs(calls["kwargs"]["access_log"], False)
        self.assertEqual(server.mcp.settings.host, "0.0.0.0")
        self.assertEqual(security.allowed_origins, [])
        self.assertEqual(
            security.allowed_hosts,
            [
                "127.0.0.1",
                "127.0.0.1:%s" % port,
                "localhost",
                "localhost:%s" % port,
            ],
        )
        self.assertNotIn("0.0.0.0", security.allowed_hosts)
        self.assertNotIn("0.0.0.0:%s" % port, security.allowed_hosts)
        self.assertNotIn("https://0.0.0.0", security.allowed_origins)
        self.assertNotIn(TOKEN, json.dumps(calls["kwargs"]))
        self.assertNotIn(TOKEN, json.dumps(security.allowed_hosts))


class FilterWildcardTests(unittest.TestCase):
    def test_like_wildcards_are_bound_not_spliced(self):
        conn = ScriptedConn([])
        with patch.object(server, "db", lambda: conn), \
                patch.object(server, "embed_query", lambda _text: "[0.100000]"), \
                patch.object(server, "hnsw_ready", lambda _con: False), \
                patch.object(server, "rerank_available", lambda: False):
            out = server.ltm_search(
                "lease", k=1, rerank=False, doc_type="%", path_contains="_")

        self.assertEqual(len(conn.calls), 1)
        sql, params = conn.calls[0]
        self.assertIn("d.doc_type ILIKE :dt", sql)
        self.assertIn("ILIKE :pc", sql)
        self.assertIn(" AND ", sql)
        self.assertNotIn(" OR ", sql)
        self.assertNotIn("SET ", sql)
        self.assertEqual(params["dt"], "%")
        self.assertEqual(params["pc"], "%_%")
        self.assertEqual(params["k"], 1)
        self.assertNotIn("%", sql)
        self.assertNotIn("'%'", sql)
        self.assertNotIn(params["e"], sql)
        self.assertEqual(out["results"], [])
        self.assertEqual(out["confidence"], "LOW — no matches (filters may be too narrow)")
        self.assertNotIn("%", out["confidence"])
        self.assertNotIn("_", json.dumps(out["confidence"]))


class SpacedDocumentIdTests(unittest.TestCase):
    def test_internal_space_is_not_document_seven(self):
        conn = ScriptedConn([
            ("WHERE id", [doc_row(SECRET)]),
        ])
        with patch.object(server, "db", lambda: conn):
            out = server.ltm_get_document("7 7")

        self.assertFalse(out["found"])
        self.assertNotIn("ambiguous", out)
        self.assertNotIn("text", out)
        self.assertEqual(out["error"], "no document matches '7 7' (id, path, or fragment)")
        self.assertNotIn(SECRET, json.dumps(out))
        self.assertEqual(len(conn.calls), 2)
        exact_sql, exact_params = conn.calls[0]
        frag_sql, frag_params = conn.calls[1]
        self.assertNotIn("WHERE id", exact_sql)
        self.assertNotIn("ILIKE", exact_sql)
        self.assertIn("relpath =", exact_sql)
        self.assertIn("ILIKE", frag_sql)
        self.assertEqual(exact_params["v"], "7 7")
        self.assertIsInstance(exact_params["v"], str)
        self.assertEqual(frag_params["v"], "%7 7%")
        self.assertNotIn("7 7", exact_sql)
        self.assertNotIn("7 7", frag_sql)
        self.assertNotIn(7, [params.get("v") for _sql, params in conn.calls])


if __name__ == "__main__":
    unittest.main()
