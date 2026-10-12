"""Retrieval contracts that must hold without Postgres or Ollama.

Covers confidence gates, the asymmetric embed prefix, bound SQL filters,
rerank ordering, and document fetch (including ambiguous-match redaction).
"""

import json
import unittest
from unittest.mock import patch

import server

MODEL_CARD_PREFIX = "Represent this sentence for searching relevant passages: "
LOW_RERANK = "LOW — answer may not be in the corpus"
HIGH_COSINE = "high (cosine only — enable rerank for a calibrated verdict)"
MEDIUM_COSINE = "medium (cosine only — enable rerank for a calibrated verdict)"
LOW_COSINE = "LOW — answer may not be in the corpus (cosine only)"
NO_MATCHES = "LOW — no matches (filters may be too narrow)"


class VerdictTests(unittest.TestCase):
    def test_gates_match_the_documented_thresholds(self):
        self.assertEqual(server.QUERY_PREFIX, MODEL_CARD_PREFIX)
        self.assertEqual(server.LOW_RERANK, 3.0)
        self.assertEqual(server.HIGH_RERANK, 6.0)
        self.assertEqual(server.LOW_COSINE, 0.78)
        self.assertEqual(server.HIGH_COSINE, 0.855)

        self.assertEqual(server.verdict_from_rerank(6.0), "high")
        self.assertEqual(server.verdict_from_rerank(5.999), "medium")
        self.assertEqual(server.verdict_from_rerank(3.0), "medium")
        self.assertEqual(server.verdict_from_rerank(2.999), LOW_RERANK)

        self.assertEqual(server.verdict_from_cosine(0.855), HIGH_COSINE)
        self.assertEqual(server.verdict_from_cosine(0.854), MEDIUM_COSINE)
        self.assertEqual(server.verdict_from_cosine(0.78), MEDIUM_COSINE)
        self.assertEqual(server.verdict_from_cosine(0.779), LOW_COSINE)

    def test_clean_snippet_collapses_whitespace_and_limits_length(self):
        self.assertEqual(server.clean_snippet(None), "")
        self.assertEqual(server.clean_snippet("  a \n\n b\t"), "a b")
        self.assertEqual(server.clean_snippet("abcdef", limit=3), "abc")
        self.assertEqual(len(server.clean_snippet("x" * 500)), server.SNIPPET_CHARS)


class EmbedTests(unittest.TestCase):
    def test_query_uses_the_model_card_prefix_and_formats_the_vector(self):
        seen = {}

        class Resp:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return json.dumps({"embedding": [0.1, 2, -1.5]}).encode()

        def urlopen(req, timeout=0):
            seen["timeout"] = timeout
            seen["url"] = req.full_url
            seen["body"] = json.loads(req.data.decode())
            return Resp()

        with patch.object(server.urllib.request, "urlopen", urlopen):
            out = server.embed_query("lease renewal")

        self.assertEqual(seen["url"], server.OLLAMA_EMBED_URL)
        self.assertEqual(seen["timeout"], 60)
        self.assertEqual(seen["body"]["model"], server.EMBED_MODEL)
        self.assertEqual(seen["body"]["prompt"], MODEL_CARD_PREFIX + "lease renewal")
        self.assertEqual(out, "[0.100000,2.000000,-1.500000]")


class FakeConn:
    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    def run(self, sql, **params):
        self.calls.append((sql, params))
        if sql.lstrip().upper().startswith("SELECT"):
            return self.rows
        return []


class ScriptedConn:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def run(self, sql, **params):
        self.calls.append((sql, params))
        if not self.responses:
            return []
        return self.responses.pop(0)


def search_row(chunk_id, cosine, text, root="/vault", relpath="Taxes/2024.pdf", dtype="tax return"):
    return (
        chunk_id, 10, dtype, "Title", "disk", root, relpath,
        "2024-01-01", "pdftotext", cosine, text,
    )


def doc_row(text="hello", **overrides):
    fields = {
        "doc_id": 7,
        "src": "disk",
        "root": "/vault",
        "relpath": "Taxes/a.pdf",
        "mime": "application/pdf",
        "dtype": "lease",
        "title": "Lease",
        "doc_date": "2024-04-01",
        "nbytes": 100,
        "chunks": 2,
        "chars": len(text),
        "method": "pdftotext",
        "ocr": 1,
        "ingested": "2024-04-02",
        "text": text,
    }
    fields.update(overrides)
    return (
        fields["doc_id"], fields["src"], fields["root"], fields["relpath"],
        fields["mime"], fields["dtype"], fields["title"], fields["doc_date"],
        fields["nbytes"], fields["chunks"], fields["chars"], fields["method"],
        fields["ocr"], fields["ingested"], fields["text"],
    )


class SearchTests(unittest.TestCase):
    def search(self, rows, indexed=False, available=False, rerank_impl=None, **kwargs):
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
            hnsw_ready=lambda _con: indexed,
            rerank_available=lambda: available,
            run_rerank=rerank,
        ):
            out = server.ltm_search(**kwargs)
        return out, conn, seen

    def test_filters_are_bound_and_limits_are_clamped(self):
        doc_type = "lease'; DROP TABLE documents; --"
        path = "100%' OR relpath ILIKE '"
        out, conn, seen = self.search(
            [],
            indexed=True,
            query="taxes",
            k=0,
            rerank=False,
            doc_type=doc_type,
            path_contains=path,
        )
        self.assertEqual(seen["embed"], "taxes")
        self.assertEqual(out["results"], [])
        self.assertEqual(out["confidence"], NO_MATCHES)
        self.assertEqual(out["hnsw_index"], "ready")
        self.assertNotIn("rerank", seen)

        sqls = [sql for sql, _params in conn.calls]
        self.assertIn("SET hnsw.ef_search = 200", sqls)
        self.assertIn("SET hnsw.iterative_scan = relaxed_order", sqls)
        for sql, _params in conn.calls:
            self.assertNotIn("DROP", sql)
            self.assertNotIn(doc_type, sql)
            self.assertNotIn(path, sql)

        select = next(params for sql, params in conn.calls if sql.startswith("SELECT"))
        select_sql = next(sql for sql, _params in conn.calls if sql.startswith("SELECT"))
        self.assertEqual(select["dt"], doc_type)
        self.assertEqual(select["pc"], "%" + path + "%")
        self.assertEqual(select["k"], 1)
        self.assertEqual(select["e"], "[0.100000]")
        self.assertIn("ILIKE :dt", select_sql)
        self.assertIn("ILIKE :pc", select_sql)
        self.assertIn("LIMIT :k", select_sql)
        self.assertIn("left(c.text, %d)" % server.RERANK_TEXT_CHARS, select_sql)

        out, conn, _seen = self.search([], query="taxes", k=999, rerank=False)
        select_sql, select = next(
            (sql, params) for sql, params in conn.calls if sql.startswith("SELECT"))
        self.assertEqual(select["k"], server.MAX_K)
        self.assertNotIn("WHERE", select_sql)
        self.assertEqual(out["hnsw_index"], "not built yet (seq scan)")
        self.assertFalse(any(sql.startswith("SET") for sql, _params in conn.calls))

        _out, conn, _seen = self.search([], indexed=True, query="taxes", k=8, rerank=False)
        sets = [sql for sql, _params in conn.calls if sql.startswith("SET")]
        self.assertIn("SET hnsw.ef_search = 200", sets)
        self.assertIn("SET hnsw.iterative_scan = off", sets)
        self.assertNotIn("SET hnsw.iterative_scan = relaxed_order", sets)

    def test_result_count_is_sliced_to_k_even_if_the_query_returns_more(self):
        rows = [search_row(1, 0.9, "one"), search_row(2, 0.2, "two"), search_row(3, 0.1, "three")]
        out, _conn, seen = self.search(rows, query="q", k=1, rerank=False)
        self.assertEqual([hit["chunk_id"] for hit in out["results"]], [1])
        self.assertNotIn("rerank", seen)
        self.assertEqual(out["confidence"], HIGH_COSINE)

    def test_rerank_reorders_hits_and_scores_the_winner(self):
        rows = [
            search_row(1, 0.99, "alpha\nbeta"),
            search_row(2, 0.50, "the clause", dtype=None),
        ]

        def impl(query, texts, top_k):
            self.assertEqual(query, "find the clause")
            self.assertEqual(texts, ["alpha\nbeta", "the clause"])
            self.assertEqual(top_k, 2)
            return [1, 0], [8.0, 4.0]

        out, _conn, seen = self.search(
            rows, available=True, rerank_impl=impl,
            query="find the clause", k=2, rerank=True,
        )
        self.assertEqual(seen["embed"], "find the clause")
        self.assertTrue(out["reranked"])
        self.assertEqual(out["confidence"], "high")
        self.assertEqual(out["top_score"], {"rerank_score": 8.0})
        self.assertEqual(out["hnsw_index"], "not built yet (seq scan, slower)")
        self.assertEqual(out["results"][0]["chunk_id"], 2)
        self.assertEqual(out["results"][0]["rerank_score"], 8.0)
        self.assertEqual(out["results"][0]["cosine"], 0.5)
        self.assertEqual(out["results"][0]["doc_type"], "untyped")
        self.assertEqual(out["results"][0]["path"], "/vault/Taxes/2024.pdf")
        self.assertEqual(out["results"][0]["snippet"], "the clause")
        self.assertEqual(out["results"][1]["chunk_id"], 1)
        self.assertEqual(out["results"][1]["rerank_score"], 4.0)
        self.assertEqual(out["results"][1]["snippet"], "alpha beta")
        self.assertEqual(out["results"][1]["cosine"], 0.99)

    def test_rerank_failure_keeps_cosine_order(self):
        rows = [search_row(1, 0.9, "kept"), search_row(2, 0.2, "other")]

        def impl(*_args):
            raise RuntimeError("boom")

        out, _conn, _seen = self.search(
            rows, available=True, rerank_impl=impl, query="q", k=2, rerank=True,
        )
        self.assertFalse(out["reranked"])
        self.assertEqual([hit["chunk_id"] for hit in out["results"]], [1, 2])
        self.assertNotIn("rerank_score", out["results"][0])
        self.assertIn("RuntimeError", out["note"])
        self.assertIn("cosine order only", out["note"])
        self.assertEqual(out["confidence"], HIGH_COSINE)
        self.assertEqual(out["top_score"], {"cosine": 0.9})

    def test_missing_reranker_is_disclosed(self):
        out, _conn, seen = self.search(
            [search_row(1, 0.8, "hit")],
            available=False,
            query="q",
            rerank=True,
        )
        self.assertFalse(out["reranked"])
        self.assertIn("reranker unavailable", out["note"])
        self.assertNotIn("rerank", seen)
        self.assertEqual(out["confidence"], MEDIUM_COSINE)
        self.assertNotIn("rerank_score", out["results"][0])


class DocumentTests(unittest.TestCase):
    def fetch(self, responses, path_or_id, max_chars=20000):
        conn = ScriptedConn(responses)
        with patch.object(server, "db", lambda: conn):
            out = server.ltm_get_document(path_or_id, max_chars=max_chars)
        return out, conn

    def test_numeric_id_is_bound_as_an_integer(self):
        out, conn = self.fetch(
            [[doc_row(text="body", doc_date=None)]],
            " 007 ",
        )
        self.assertTrue(out["found"])
        self.assertEqual(len(conn.calls), 1)
        self.assertIn("WHERE id = :v", conn.calls[0][0])
        self.assertEqual(conn.calls[0][1]["v"], 7)
        self.assertIsInstance(conn.calls[0][1]["v"], int)
        self.assertEqual(out["text"], "body")
        self.assertEqual(out["path"], "/vault/Taxes/a.pdf")
        self.assertTrue(out["ocr"])
        self.assertIsNone(out["doc_date"])
        self.assertFalse(out["truncated"])

    def test_exact_path_does_not_fall_through_to_a_fragment_search(self):
        _out, conn = self.fetch([[doc_row()]], "Taxes/a.pdf")
        self.assertEqual(len(conn.calls), 1)
        self.assertIn("relpath = :v", conn.calls[0][0])
        self.assertNotIn("ILIKE", conn.calls[0][0])

    def test_ambiguous_fragment_does_not_return_document_text(self):
        secret = "FULL_TEXT_SHOULD_NOT_LEAK"
        rows = [
            doc_row(text=secret, doc_id=1, relpath="a.pdf"),
            doc_row(text=secret, doc_id=2, relpath="b.pdf"),
        ]
        out, conn = self.fetch([[], rows], "Taxes")
        self.assertFalse(out["found"])
        self.assertTrue(out["ambiguous"])
        self.assertNotIn(secret, json.dumps(out))
        self.assertEqual([item["doc_id"] for item in out["candidates"]], [1, 2])
        self.assertIn("ILIKE :v", conn.calls[1][0])
        self.assertEqual(conn.calls[1][1]["v"], "%Taxes%")

    def test_fragment_lookup_binds_the_pattern(self):
        key = "x'; DROP TABLE documents; --"
        out, conn = self.fetch([[], []], key)
        self.assertFalse(out["found"])
        self.assertNotIn("ambiguous", out)
        self.assertNotIn("SELECT", out["error"])
        self.assertEqual(conn.calls[0][1]["v"], key)
        self.assertEqual(conn.calls[1][1]["v"], "%" + key + "%")
        for sql, _params in conn.calls:
            self.assertNotIn("DROP", sql)
            self.assertNotIn(key, sql)

    def test_text_cap_is_clamped(self):
        out, _conn = self.fetch([[doc_row(text="y" * 250)]], "7", max_chars=1)
        self.assertTrue(out["truncated"])
        self.assertEqual(out["text"], "y" * 200)

        out, _conn = self.fetch([[doc_row(text="z" * 500_001)]], "7", max_chars=9_000_000)
        self.assertTrue(out["truncated"])
        self.assertEqual(len(out["text"]), 500_000)
        self.assertTrue(out["text"].startswith("z" * 100))


if __name__ == "__main__":
    unittest.main()
