"""The rerank worker is started with python -c. The script directory is data.

A path that looks like Python must stay inside the sys.path string. These
tests fake Popen and do not load a model.
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


class WorkerSourceTests(unittest.TestCase):
    def setUp(self):
        self._saved_proc = server._rr_proc
        self._python = server.RERANK_PYTHON
        self._script = server.RERANK_SCRIPT
        server._rr_proc = None

    def tearDown(self):
        server._rr_proc = self._saved_proc
        server.RERANK_PYTHON = self._python
        server.RERANK_SCRIPT = self._script

    def test_script_directory_cannot_escape_the_worker_source(self):
        nasty = "dir'); import os; os.system('echo pwned'); #"
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

        attrs = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                attrs.append(node.func.attr)
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
