"""Rerank protocol and result shaping.

The worker speaks one JSON object per line. A newline inside the query or a
chunk must stay inside that object. A duplicate index in the worker's order
must not surface the other chunk.
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import server


class RerankLineAndDuplicateOrderTest(unittest.TestCase):
    def tearDown(self):
        proc = server._rr_proc
        server._rr_proc = None
        if proc is not None and proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)
        if proc is not None:
            for stream in (proc.stdin, proc.stdout):
                if stream is not None and not stream.closed:
                    stream.close()

    def test_newline_in_query_stays_one_json_line(self):
        query = 'café\n{"query": "pwned"}'
        text = 'chunk"\r\nmore'
        captured = tempfile.NamedTemporaryFile(delete=False)
        captured.close()
        script = (
            "import json,sys\n"
            "line = sys.stdin.readline()\n"
            "open(sys.argv[1], 'w', encoding='utf-8').write(line)\n"
            "sys.stdout.write(json.dumps({'order': [0], 'scores': [4.0]}) + '\\n')\n"
            "sys.stdout.flush()\n"
        )
        proc = subprocess.Popen(
            [sys.executable, "-c", script, captured.name],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
        )
        server._rr_proc = proc
        try:
            order, scores = server.run_rerank(query, [text], 1)
        finally:
            self.tearDown()
        self.assertEqual(order, [0])
        self.assertEqual(scores, [4.0])
        with open(captured.name, encoding="utf-8") as fh:
            line = fh.read()
        os.unlink(captured.name)
        self.assertEqual(line.count("\n"), 1)
        self.assertNotIn("\n", line[:-1])
        payload = json.loads(line)
        self.assertEqual(payload["query"], query)
        self.assertEqual(payload["texts"], [text])
        self.assertEqual(payload["top_k"], 1)
        # One object: the embedded newline and the forged query stay inside the string.
        self.assertEqual(payload["query"].count("\n"), 1)
        self.assertIn('{"query": "pwned"}', payload["query"])

    def test_duplicate_order_index_hides_the_other_chunk(self):
        rows = [
            (1, 10, "lease", "Alpha", "src", "ROOT", "a.pdf", None, "pdf", 0.91, "ALPHA-SNIPPET"),
            (2, 11, "lease", "Beta", "src", "ROOT", "b.pdf", None, "pdf", 0.80, "BETA-SNIPPET-SHOULD-NOT-LEAK"),
        ]

        class Con:
            def __init__(self):
                self.calls = []

            def run(self, sql, **params):
                self.calls.append((sql, params))
                return rows

        con = Con()
        seen = {}

        def fake_rerank(query, texts, top_k):
            seen["query"] = query
            seen["texts"] = list(texts)
            seen["top_k"] = top_k
            return [0, 0], [9.0, 1.0]

        originals = {
            "db": server.db,
            "embed_query": server.embed_query,
            "hnsw_ready": server.hnsw_ready,
            "rerank_available": server.rerank_available,
            "run_rerank": server.run_rerank,
        }
        server.db = lambda: con
        server.embed_query = lambda text: "[0.100000]"
        server.hnsw_ready = lambda _con: False
        server.rerank_available = lambda: True
        server.run_rerank = fake_rerank
        try:
            out = server.ltm_search("where is the lease", k=2, rerank=True)
        finally:
            for name, fn in originals.items():
                setattr(server, name, fn)

        self.assertEqual(len(con.calls), 1)
        sql, params = con.calls[0]
        self.assertNotIn("SET", sql)
        self.assertNotIn("ALPHA-SNIPPET", sql)
        self.assertNotIn("BETA-SNIPPET-SHOULD-NOT-LEAK", sql)
        self.assertEqual(params["k"], server.RERANK_CANDIDATES)
        self.assertNotEqual(params["k"], 2)
        self.assertEqual(seen["texts"], ["ALPHA-SNIPPET", "BETA-SNIPPET-SHOULD-NOT-LEAK"])
        self.assertEqual(seen["top_k"], 2)
        self.assertEqual(seen["query"], "where is the lease")

        self.assertTrue(out["reranked"])
        self.assertEqual(out["confidence"], "high")
        self.assertNotIn("cosine only", out["confidence"])
        self.assertEqual(out["top_score"], {"rerank_score": 9.0})
        self.assertEqual(len(out["results"]), 2)
        self.assertEqual([hit["chunk_id"] for hit in out["results"]], [1, 1])
        self.assertEqual([hit["doc_id"] for hit in out["results"]], [10, 10])
        self.assertEqual([hit["snippet"] for hit in out["results"]], ["ALPHA-SNIPPET", "ALPHA-SNIPPET"])
        self.assertEqual([hit["rerank_score"] for hit in out["results"]], [9.0, 1.0])
        blob = json.dumps(out)
        self.assertNotIn("BETA-SNIPPET-SHOULD-NOT-LEAK", blob)
        self.assertNotIn("b.pdf", blob)


if __name__ == "__main__":
    unittest.main()
