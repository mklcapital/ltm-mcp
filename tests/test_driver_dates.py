"""Driver date values are stored as strings, not live date objects.

pg8000 returns date and datetime instances. The tool result has to be plain
JSON: str() of a datetime uses a space, not the T from isoformat(). No
Postgres, Ollama, or rerank model. The HTTP server is never started.
"""

import datetime
import unittest
from unittest.mock import patch

import server


class DriverDateTests(unittest.TestCase):
    def test_driver_dates_become_strings(self):
        day = datetime.date(2024, 5, 6)
        stamp = datetime.datetime(2024, 5, 7, 1, 2, 3)
        search_row = (
            11,
            22,
            "lease",
            "TITLE-SENTINEL",
            "SRC-SENTINEL",
            "/ROOT",
            "rel/file.pdf",
            day,
            "pdftotext-method",
            0.8,
            "snippet body",
        )
        search_conn = _ScriptedConn([search_row])
        with patch.multiple(
            server,
            db=lambda: search_conn,
            embed_query=lambda _text: "[0.100000]",
            hnsw_ready=lambda _con: False,
            rerank_available=lambda: False,
        ):
            found = server.ltm_search("taxes", k=4, rerank=False)

        hit = found["results"][0]
        self.assertEqual(hit["doc_date"], "2024-05-06")
        self.assertNotEqual(hit["doc_date"], day.isoformat() + "T00:00:00")
        self.assertNotIsInstance(hit["doc_date"], datetime.date)
        self.assertNotIn("datetime", hit["doc_date"])
        self.assertEqual(hit["extraction"], "pdftotext-method")
        self.assertNotEqual(hit["doc_date"], hit["extraction"])
        self.assertNotIn("TITLE-SENTINEL", hit["doc_date"])

        text = "BODY-SENTINEL"
        doc_row = (
            22,
            "SRC-SENTINEL",
            "/ROOT",
            "rel/file.pdf",
            "application/pdf",
            "lease",
            "TITLE-SENTINEL",
            day,
            4096,
            3,
            12,
            "pdftotext-method",
            0,
            stamp,
            text,
        )
        doc_conn = _ScriptedConn([doc_row])
        with patch.object(server, "db", lambda: doc_conn):
            doc = server.ltm_get_document("22")

        self.assertTrue(doc["found"])
        self.assertEqual(doc["doc_date"], "2024-05-06")
        self.assertEqual(doc["ingested_at"], "2024-05-07 01:02:03")
        self.assertNotIn("T", doc["ingested_at"])
        self.assertNotIn("datetime", doc["ingested_at"])
        self.assertNotIsInstance(doc["ingested_at"], datetime.datetime)
        self.assertEqual(doc["bytes"], 4096)
        self.assertIsInstance(doc["bytes"], int)
        self.assertEqual(doc["text"], text)
        self.assertNotIn(text, doc["ingested_at"])
        self.assertNotIn(text, doc["doc_date"])
        self.assertEqual(doc["extraction"], "pdftotext-method")


class _ScriptedConn:
    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    def run(self, sql, **params):
        self.calls.append((sql, params))
        if sql.startswith("SELECT"):
            return self.rows
        return []


if __name__ == "__main__":
    unittest.main()
