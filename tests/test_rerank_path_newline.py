"""A newline in the rerank script directory stays inside the worker source.

python -c would execute a breakout if the path were concatenated instead of
repr'd. Popen is faked; the model is not loaded.
"""

import ast
import os
import tempfile
import unittest
from unittest.mock import patch

import server


class Proc:
    def __init__(self):
        self.exitcode = None
        self.killed = False

    def poll(self):
        return self.exitcode

    def kill(self):
        self.killed = True
        self.exitcode = -9


class RerankPathNewlineTests(unittest.TestCase):
    def setUp(self):
        self._saved_proc = server._rr_proc
        self._python = server.RERANK_PYTHON
        self._script = server.RERANK_SCRIPT
        server._rr_proc = None

    def tearDown(self):
        server._rr_proc = self._saved_proc
        server.RERANK_PYTHON = self._python
        server.RERANK_SCRIPT = self._script

    def test_newline_backslash_and_braces_stay_inside_the_path_string(self):
        nasty = "dir\\\nimport os\nos.system('echo pwned')\n{0}"
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        folder = os.path.join(tmp.name, nasty)
        os.makedirs(folder)
        script = os.path.join(folder, "rerank.py")
        with open(script, "w", encoding="utf-8"):
            pass
        server.RERANK_PYTHON = "/usr/bin/python3"
        server.RERANK_SCRIPT = script
        expected_dir = os.path.dirname(os.path.abspath(script))

        captured = {}

        def popen(argv, **kwargs):
            captured["argv"] = argv
            captured["kwargs"] = kwargs
            return Proc()

        with patch.object(server.subprocess, "Popen", popen), patch.object(
            server, "_rr_read_line", lambda _proc, _timeout: {"ready": True}
        ):
            proc = server._rr_worker()

        self.assertIs(server._rr_proc, proc)
        self.assertFalse(proc.killed)
        argv = captured["argv"]
        self.assertEqual(argv[:3], ["/usr/bin/python3", "-u", "-c"])
        self.assertEqual(len(argv), 4)
        self.assertIs(captured["kwargs"].get("shell", False), False)
        src = argv[3]
        tree = ast.parse(src)

        modules = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                modules.append(node.module)
        self.assertCountEqual(modules, ["json", "sys", "rerank"])

        attrs = [
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        ]
        self.assertNotIn("system", attrs)
        self.assertIn("insert", attrs)

        constants = [
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        ]
        self.assertIn(expected_dir, constants)
        self.assertNotIn(nasty, src.replace(repr(expected_dir), ""))


if __name__ == "__main__":
    unittest.main()
