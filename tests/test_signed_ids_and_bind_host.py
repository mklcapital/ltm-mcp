"""Signed document ids stay path lookups, and the bind host is not an origin.

No Postgres, Ollama, or rerank model. The HTTP server is never started.
"""

import json
import os
import unittest
from unittest.mock import patch

import server


TOKEN = "unit-test-bearer"
SECRET = "FULL_TEXT_SHOULD_NOT_LEAK"


def doc_row(text, doc_id=7, relpath="Taxes/a.pdf"):
    return (
        doc_id, "disk", "/vault", relpath, "application/pdf", "lease", "Lease",
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


class SignedIdTests(unittest.TestCase):
    def test_signed_and_wrapped_ids_are_not_integer_lookups(self):
        # int("-7") is -7 and a looser parser could take "(7)" as 7. The gate
        # is str.isdigit after strip, so these stay exact path lookups.
        for key in ("-7", "(7)", "7-"):
            with self.subTest(key=key):
                conn = ScriptedConn([
                    ("WHERE id", [doc_row(SECRET, 7)]),
                    ("ILIKE", [doc_row("fragment-hit", 9, "notes/other.pdf")]),
                ])
                with patch.object(server, "db", lambda: conn):
                    out = server.ltm_get_document(key)

                self.assertTrue(out["found"])
                self.assertEqual(out["doc_id"], 9)
                self.assertEqual(out["text"], "fragment-hit")
                self.assertNotIn(SECRET, json.dumps(out))
                self.assertEqual(len(conn.calls), 2)
                exact_sql, exact_params = conn.calls[0]
                frag_sql, frag_params = conn.calls[1]
                self.assertNotIn("WHERE id", exact_sql)
                self.assertNotIn("ILIKE", exact_sql)
                self.assertIn("ILIKE", frag_sql)
                self.assertNotIn(key, exact_sql)
                self.assertNotIn(key, frag_sql)
                self.assertEqual(exact_params["v"], key)
                self.assertIsInstance(exact_params["v"], str)
                self.assertEqual(frag_params["v"], "%" + key + "%")
                self.assertNotIn(7, [params.get("v") for _sql, params in conn.calls])


class BindHostTests(unittest.TestCase):
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

    def test_bind_host_is_not_copied_into_the_allowlist(self):
        # LTM_HTTP_HOST chooses the listen address only. Copying it into
        # allowed_hosts or allowed_origins would widen DNS-rebinding protection.
        server.HTTP_HOST = "127.0.0.2"
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
        self.assertEqual(calls["kwargs"]["host"], "127.0.0.2")
        self.assertEqual(calls["kwargs"]["port"], server.HTTP_PORT)
        self.assertIs(calls["kwargs"]["access_log"], False)
        self.assertEqual(server.mcp.settings.host, "127.0.0.2")
        self.assertEqual(security.allowed_origins, [])
        self.assertEqual(
            security.allowed_hosts,
            [
                "127.0.0.1",
                "127.0.0.1:%s" % server.HTTP_PORT,
                "localhost",
                "localhost:%s" % server.HTTP_PORT,
            ],
        )
        self.assertNotIn("127.0.0.2", security.allowed_hosts)
        self.assertNotIn("0.0.0.0", security.allowed_hosts)
        self.assertNotIn("https://127.0.0.2", security.allowed_origins)
        self.assertNotIn(TOKEN, json.dumps(calls["kwargs"]))


if __name__ == "__main__":
    unittest.main()
