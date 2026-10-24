"""LTM_PUBLIC_HOST is one allowlist entry, not a second origin or a bind address.

No Postgres, Ollama, or a listening HTTP server.
"""

import os
import sys
import types
import unittest
from unittest.mock import patch

import server


TOKEN = "unit-test-bearer"


class PublicHostAllowlistTests(unittest.TestCase):
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
        (
            settings.stateless_http,
            settings.host,
            settings.port,
            settings.transport_security,
        ) = self._saved

    def _run_http(self, public_host):
        calls = {}

        def fake_run(app, **kwargs):
            calls["app"] = app
            calls["kwargs"] = kwargs

        uvicorn = types.ModuleType("uvicorn")
        uvicorn.run = fake_run
        env = {"LTM_PUBLIC_HOST": public_host}
        with patch.dict(os.environ, env, clear=False), patch.dict(sys.modules, {"uvicorn": uvicorn}):
            with patch.object(server, "_load_or_create_token", return_value=TOKEN):
                with patch.object(server.mcp, "streamable_http_app", return_value=object()):
                    with self.assertLogs("ltm", level="INFO") as captured:
                        server.run_http()
        calls["logs"] = "\n".join(captured.output)
        return calls

    def test_a_newline_or_scheme_does_not_add_another_origin(self):
        smuggled = "ltm.example.com\r\nhttps://evil.example"
        calls = self._run_http(smuggled)
        security = server.mcp.settings.transport_security
        loopback = [
            "127.0.0.1",
            "127.0.0.1:%s" % server.HTTP_PORT,
            "localhost",
            "localhost:%s" % server.HTTP_PORT,
        ]
        self.assertEqual(security.allowed_hosts, loopback + [smuggled])
        self.assertEqual(security.allowed_origins, ["https://" + smuggled])
        self.assertNotIn("https://evil.example", security.allowed_origins)
        self.assertNotIn("evil.example", security.allowed_hosts)
        self.assertFalse(any(origin.startswith("http://") for origin in security.allowed_origins))
        self.assertEqual(calls["kwargs"]["host"], "127.0.0.1")
        self.assertNotEqual(calls["kwargs"]["host"], smuggled)
        self.assertIs(calls["kwargs"]["access_log"], False)
        self.assertNotIn(TOKEN, calls["logs"])
        self.assertNotIn(TOKEN, str(calls["kwargs"]))

        calls = self._run_http("http://evil.example")
        security = server.mcp.settings.transport_security
        self.assertEqual(security.allowed_hosts, loopback + ["http://evil.example"])
        self.assertEqual(security.allowed_origins, ["https://http://evil.example"])
        self.assertNotIn("http://evil.example", security.allowed_origins)
        self.assertFalse(any(origin.startswith("http://") for origin in security.allowed_origins))
        self.assertEqual(calls["kwargs"]["host"], server.HTTP_HOST)
        self.assertEqual(calls["kwargs"]["host"], "127.0.0.1")
        self.assertIs(calls["kwargs"]["access_log"], False)
        self.assertNotIn(TOKEN, calls["logs"])


if __name__ == "__main__":
    unittest.main()
