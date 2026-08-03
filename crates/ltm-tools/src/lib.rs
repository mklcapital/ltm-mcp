//! ltm-tools — the three MCP tools (`ltm_search`, `ltm_get_document`,
//! `ltm_stats`), orchestrating a [`Store`], an [`EmbeddingBackend`], and an
//! optional [`Reranker`] from `ltm-core`. Transport-agnostic: returns
//! `serde_json::Value` payloads identical in shape to server.py's, so
//! `ltm-server` only has to expose them.
//!
//! This mirrors `server.py`'s `ltm_search` / `ltm_get_document` / `ltm_stats`
//! behavior exactly (field names, verdict strings, rerank fallback note,
//! empty/no-match shape, get_document Found/Ambiguous/NotFound shapes, and the
//! stats fields). The `ltm-core` traits are blocking; the async transport lives
//! in `ltm-server`.

use ltm_core::{
    clean_snippet, verdict_from_cosine, verdict_from_rerank, Config, DocKey, DocLookup,
    EmbeddingBackend, Error, Hit, Reranker, SearchFilter, Store,
};
use serde_json::{json, Value};

/// Upper bound on `k` (server.py `MAX_K`).
const MAX_K: usize = 50;
/// How many characters a search hit's snippet shows (server.py `SNIPPET_CHARS`).
const SNIPPET_CHARS: usize = 350;
/// How many hits the ChatGPT-compat `search` alias requests (its contract has
/// no `k` argument, so it uses ltm_search's default).
const COMPAT_SEARCH_K: usize = 8;
/// Text cap for the ChatGPT-compat `fetch` alias (no `max_chars` argument in
/// the contract; generous enough for whole documents, still bounded).
const COMPAT_FETCH_MAX_CHARS: usize = 100_000;

pub struct Tools<'a> {
    pub cfg: &'a Config,
    pub store: &'a dyn Store,
    pub embed: &'a dyn EmbeddingBackend,
    pub rerank: &'a dyn Reranker,
}

impl<'a> Tools<'a> {
    /// Semantic search over the corpus. Embeds the query, runs the vector
    /// search, optionally reranks the top candidates, and returns the top-`k`
    /// hits with an honest confidence verdict — matching server.py's shape.
    pub fn ltm_search(
        &self,
        query: &str,
        k: usize,
        rerank: bool,
        filter_doc_type: Option<&str>,
        filter_path_contains: Option<&str>,
    ) -> Value {
        let k = k.clamp(1, MAX_K);

        let use_rerank = rerank && self.rerank.available();
        let mut rerank_note: Option<String> = None;
        if rerank && !use_rerank {
            rerank_note = Some(
                "rerank requested but reranker unavailable (LTM_RERANK_PYTHON / \
                 LTM_RERANK_SCRIPT not set or not found) — cosine order only"
                    .to_string(),
            );
        }

        let fetch_k = if use_rerank {
            self.cfg.rerank_candidates.max(k)
        } else {
            k
        };

        let filter = SearchFilter {
            doc_type: filter_doc_type.map(str::to_string),
            path_contains: filter_path_contains.map(str::to_string),
        };

        let emb = match self.embed.embed_query(query) {
            Ok(e) => e,
            Err(e) => return json!({ "error": e.to_string() }),
        };

        let indexed = self.store.hnsw_ready().unwrap_or(false);

        let mut rows = match self.store.search(&emb, fetch_k, &filter) {
            Ok(r) => r,
            Err(e) => return json!({ "error": e.to_string() }),
        };

        if rows.is_empty() {
            return json!({
                "query": query,
                "results": [],
                "confidence": "LOW — no matches (filters may be too narrow)",
                "hnsw_index": if indexed { "ready" } else { "not built yet (seq scan)" },
            });
        }

        let mut rr_scores: Option<Vec<f64>> = None;
        if use_rerank {
            let texts: Vec<String> = rows.iter().map(|r| r.chunk_text.clone()).collect();
            match self.rerank.rerank(query, &texts, k) {
                Ok((order, scores)) => {
                    // Reorder rows best-first per the reranker's index order.
                    rows = order.iter().filter_map(|&i| rows.get(i).cloned()).collect();
                    rr_scores = Some(scores);
                }
                Err(e) => {
                    // Degrade gracefully: keep cosine order, note the failure.
                    rerank_note = Some(format!(
                        "rerank failed ({}) — cosine order only",
                        error_kind(&e)
                    ));
                }
            }
        }

        rows.truncate(k);

        let hits: Vec<Hit> = rows
            .iter()
            .enumerate()
            .map(|(i, r)| Hit {
                path: format!(
                    "{}/{}",
                    r.root.as_deref().unwrap_or(""),
                    r.relpath.as_deref().unwrap_or("")
                ),
                doc_id: r.doc_id,
                chunk_id: r.chunk_id,
                title: r.title.clone(),
                doc_type: r
                    .doc_type
                    .clone()
                    .filter(|s| !s.is_empty())
                    .unwrap_or_else(|| "untyped".to_string()),
                source: r.source_label.clone(),
                doc_date: r.doc_date.clone(),
                extraction: r.text_method.clone(),
                cosine: round4(r.cosine),
                snippet: clean_snippet(&r.chunk_text, SNIPPET_CHARS),
                rerank_score: rr_scores.as_ref().and_then(|s| s.get(i).copied()),
            })
            .collect();

        let (confidence, top) = match &rr_scores {
            Some(scores) => {
                let top = scores.first().copied().unwrap_or(0.0);
                (verdict_from_rerank(top, self.cfg), json!({ "rerank_score": top }))
            }
            None => {
                let top = rows.first().map(|r| r.cosine).unwrap_or(0.0);
                (
                    verdict_from_cosine(top, self.cfg),
                    json!({ "cosine": round4(top) }),
                )
            }
        };

        let results = serde_json::to_value(&hits).unwrap_or_else(|_| Value::Array(vec![]));

        let mut out = json!({
            "query": query,
            "reranked": rr_scores.is_some(),
            "confidence": confidence,
            "top_score": top,
            "hnsw_index": if indexed {
                "ready"
            } else {
                "not built yet (seq scan, slower)"
            },
            "results": results,
        });
        if let Some(note) = rerank_note {
            out["note"] = Value::String(note);
        }
        out
    }

    /// Fetch a document's full extracted text by id / exact path / fragment,
    /// returning the Found / Ambiguous / NotFound shape from server.py.
    pub fn ltm_get_document(&self, key: &DocKey, max_chars: usize) -> Value {
        let max_chars = max_chars.clamp(200, 500_000);

        match self.store.get_document(key, max_chars) {
            Ok(DocLookup::Found(doc)) => {
                // server.py computes truncation over the full text.
                let char_count = doc.text.chars().count();
                let truncated = char_count > max_chars;
                let text: String = doc.text.chars().take(max_chars).collect();
                json!({
                    "found": true,
                    "doc_id": doc.doc_id,
                    "path": doc.path,
                    "source": doc.source,
                    "mime": doc.mime,
                    "doc_type": doc.doc_type,
                    "title": doc.title,
                    "doc_date": doc.doc_date,
                    "bytes": doc.bytes,
                    "chunk_count": doc.chunk_count,
                    "char_count": doc.char_count,
                    "extraction": doc.extraction,
                    "ocr": doc.ocr,
                    "ingested_at": doc.ingested_at,
                    "truncated": truncated,
                    "text": text,
                })
            }
            Ok(DocLookup::Ambiguous { candidates }) => {
                let candidates: Vec<Value> = candidates
                    .iter()
                    .map(|c| {
                        json!({
                            "doc_id": c.doc_id,
                            "path": c.path,
                            "doc_type": c.doc_type,
                            "title": c.title,
                            "char_count": c.char_count,
                        })
                    })
                    .collect();
                json!({
                    "found": false,
                    "ambiguous": true,
                    "candidates": candidates,
                    "hint": "multiple matches — call again with one doc_id",
                })
            }
            Ok(DocLookup::NotFound) => {
                json!({
                    "found": false,
                    "error": format!(
                        "no document matches {} (id, path, or fragment)",
                        key_repr(key)
                    ),
                })
            }
            Err(e) => json!({ "error": e.to_string() }),
        }
    }

    /// ChatGPT-connector `search` alias. OpenAI's connector / Deep Research
    /// contract requires a tool named exactly `search` taking a single query
    /// string and returning `{"results": [{id, title, text, url}]}`. Wraps
    /// [`Self::ltm_search`] (same read-only pipeline), dedupes hits per
    /// document, and passes the honest `confidence` verdict through.
    pub fn compat_search(&self, query: &str) -> Value {
        let res = self.ltm_search(query, COMPAT_SEARCH_K, true, None, None);
        if res.get("results").is_none() {
            return res; // embed/db error shape — pass through
        }
        let confidence = res.get("confidence").cloned().unwrap_or(Value::Null);
        let mut seen = std::collections::HashSet::new();
        let mut results = Vec::new();
        if let Some(hits) = res.get("results").and_then(Value::as_array) {
            for hit in hits {
                let Some(doc_id) = hit.get("doc_id").and_then(Value::as_i64) else {
                    continue;
                };
                if !seen.insert(doc_id) {
                    continue;
                }
                let title = hit
                    .get("title")
                    .and_then(Value::as_str)
                    .filter(|s| !s.is_empty())
                    .or_else(|| hit.get("path").and_then(Value::as_str))
                    .map(str::to_string)
                    .unwrap_or_else(|| format!("document {doc_id}"));
                results.push(json!({
                    "id": doc_id.to_string(),
                    "title": title,
                    "text": hit.get("snippet").cloned().unwrap_or_else(|| json!("")),
                    "url": format!("ltm://doc/{doc_id}"),
                }));
            }
        }
        json!({ "results": results, "confidence": confidence })
    }

    /// ChatGPT-connector `fetch` alias: full document text by the id a
    /// `search` result returned, shaped `{id, title, text, url, metadata}`.
    /// Ambiguous or missing ids return the `{"error": ...}` shape (the
    /// transport surfaces it as a tool error).
    pub fn compat_fetch(&self, key: &DocKey) -> Value {
        let doc = self.ltm_get_document(key, COMPAT_FETCH_MAX_CHARS);
        if doc.get("found").and_then(Value::as_bool) != Some(true) {
            if doc.get("ambiguous").and_then(Value::as_bool) == Some(true) {
                let cands = doc
                    .get("candidates")
                    .and_then(Value::as_array)
                    .map(|cs| {
                        cs.iter()
                            .map(|c| {
                                format!(
                                    "{}: {}",
                                    c.get("doc_id").and_then(Value::as_i64).unwrap_or(0),
                                    c.get("path").and_then(Value::as_str).unwrap_or("?"),
                                )
                            })
                            .collect::<Vec<_>>()
                            .join("; ")
                    })
                    .unwrap_or_default();
                return json!({
                    "error": format!(
                        "ambiguous id {} — call fetch with one of: {cands}",
                        key_repr(key)
                    ),
                });
            }
            let msg = doc
                .get("error")
                .and_then(Value::as_str)
                .unwrap_or("document lookup failed")
                .to_string();
            return json!({ "error": msg });
        }

        let doc_id = doc.get("doc_id").and_then(Value::as_i64).unwrap_or(0);
        let title = doc
            .get("title")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
            .or_else(|| doc.get("path").and_then(Value::as_str))
            .map(str::to_string)
            .unwrap_or_else(|| format!("document {doc_id}"));
        json!({
            "id": doc_id.to_string(),
            "title": title,
            "text": doc.get("text").cloned().unwrap_or_else(|| json!("")),
            "url": format!("ltm://doc/{doc_id}"),
            "metadata": {
                "path": doc.get("path").cloned().unwrap_or(Value::Null),
                "doc_type": doc.get("doc_type").cloned().unwrap_or(Value::Null),
                "source": doc.get("source").cloned().unwrap_or(Value::Null),
                "doc_date": doc.get("doc_date").cloned().unwrap_or(Value::Null),
                "mime": doc.get("mime").cloned().unwrap_or(Value::Null),
                "extraction": doc.get("extraction").cloned().unwrap_or(Value::Null),
                "ocr": doc.get("ocr").cloned().unwrap_or(Value::Null),
                "char_count": doc.get("char_count").cloned().unwrap_or(Value::Null),
                "truncated": doc.get("truncated").cloned().unwrap_or(Value::Null),
            },
        })
    }

    /// Corpus health: counts, coverage, index status, and embed/rerank health.
    pub fn ltm_stats(&self) -> Value {
        let stats = match self.store.stats() {
            Ok(s) => s,
            Err(e) => return json!({ "error": e.to_string() }),
        };

        let embedding_coverage = if stats.chunks != 0 {
            format!(
                "{:.2}%",
                (stats.mxbai_embeddings as f64 / stats.chunks as f64) * 100.0
            )
        } else {
            "n/a".to_string()
        };

        let hnsw_index = if stats.hnsw_ready {
            "ready — fast approximate search"
        } else {
            "not built yet — searches fall back to sequential scan (~10 s each); \
             auto-build pending"
        };

        let reranker = if self.rerank.available() {
            "available"
        } else {
            "unavailable (set LTM_RERANK_PYTHON / LTM_RERANK_SCRIPT)"
        };

        let documents_by_type =
            serde_json::to_value(&stats.documents_by_type).unwrap_or_else(|_| json!({}));
        let table_sizes = serde_json::to_value(&stats.table_sizes).unwrap_or_else(|_| json!({}));

        json!({
            "documents": stats.documents,
            "chunks": stats.chunks,
            "mxbai_embeddings": stats.mxbai_embeddings,
            "embedding_coverage": embedding_coverage,
            "documents_by_type": documents_by_type,
            "hnsw_index": hnsw_index,
            "embed_model": self.cfg.embed_model,
            "ollama": self.embed.health(),
            "reranker": reranker,
            "table_sizes": table_sizes,
            "read_only": true,
        })
    }
}

/// Round to 4 decimal places (server.py `round(x, 4)`).
fn round4(x: f64) -> f64 {
    (x * 10_000.0).round() / 10_000.0
}

/// A short type name for a rerank failure, standing in for Python's
/// `type(exc).__name__` in the fallback note.
fn error_kind(e: &Error) -> &'static str {
    match e {
        Error::Config(_) => "Config",
        Error::Db(_) => "Db",
        Error::Embed(_) => "Embed",
        Error::Rerank(_) => "Rerank",
        Error::NotFound(_) => "NotFound",
        Error::Other(_) => "Other",
    }
}

/// Reproduce Python's `{key!r}` for the NotFound error (single-quoted).
fn key_repr(key: &DocKey) -> String {
    match key {
        DocKey::Id(n) => format!("'{n}'"),
        DocKey::PathOrFragment(s) => format!("'{s}'"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ltm_core::{ChunkRow, DocCandidate, Document, Stats};
    use std::collections::BTreeMap;
    use std::sync::Mutex;

    // ---- Fake implementations of the frozen traits -----------------------

    #[derive(Default)]
    struct FakeStore {
        rows: Vec<ChunkRow>,
        lookup: Option<DocLookup>,
        stats: Option<Stats>,
        hnsw: bool,
        search_err: bool,
        last_fetch_k: Mutex<Option<usize>>,
    }

    impl Store for FakeStore {
        fn search(
            &self,
            _query_embedding: &[f32],
            fetch_k: usize,
            _filter: &SearchFilter,
        ) -> ltm_core::Result<Vec<ChunkRow>> {
            *self.last_fetch_k.lock().unwrap() = Some(fetch_k);
            if self.search_err {
                return Err(Error::Db("boom".into()));
            }
            Ok(self.rows.clone())
        }

        fn get_document(&self, _key: &DocKey, _max_chars: usize) -> ltm_core::Result<DocLookup> {
            Ok(self.lookup.clone().unwrap_or(DocLookup::NotFound))
        }

        fn stats(&self) -> ltm_core::Result<Stats> {
            self.stats
                .clone()
                .ok_or_else(|| Error::Db("no stats".into()))
        }

        fn hnsw_ready(&self) -> ltm_core::Result<bool> {
            Ok(self.hnsw)
        }
    }

    struct FakeEmbed {
        health: String,
    }
    impl EmbeddingBackend for FakeEmbed {
        fn embed_query(&self, _text: &str) -> ltm_core::Result<Vec<f32>> {
            Ok(vec![0.1, 0.2, 0.3])
        }
        fn describe(&self) -> String {
            "fake(mxbai-embed-large)".to_string()
        }
        fn health(&self) -> String {
            self.health.clone()
        }
    }

    /// Reranker configurable to be unavailable, to succeed with a canned
    /// order/scores, or to fail.
    struct FakeReranker {
        available: bool,
        result: Option<ltm_core::Result<(Vec<usize>, Vec<f64>)>>,
    }
    impl Reranker for FakeReranker {
        fn rerank(
            &self,
            _query: &str,
            _texts: &[String],
            _top_k: usize,
        ) -> ltm_core::Result<(Vec<usize>, Vec<f64>)> {
            match &self.result {
                Some(Ok(v)) => Ok(v.clone()),
                Some(Err(_)) => Err(Error::Rerank("worker died".into())),
                None => Err(Error::Rerank("no result configured".into())),
            }
        }
        fn available(&self) -> bool {
            self.available
        }
    }

    // ---- Fixtures --------------------------------------------------------

    fn cfg() -> Config {
        Config::from_env()
    }

    fn row(chunk_id: i64, doc_id: i64, cosine: f64, text: &str) -> ChunkRow {
        ChunkRow {
            chunk_id,
            doc_id,
            doc_type: Some("lease".into()),
            title: Some(format!("Doc {doc_id}")),
            source_label: Some("src".into()),
            root: Some("/root".into()),
            relpath: Some(format!("path/{doc_id}.txt")),
            doc_date: Some("2024-01-01".into()),
            text_method: Some("pdf".into()),
            cosine,
            chunk_text: text.into(),
        }
    }

    fn unavailable_reranker() -> FakeReranker {
        FakeReranker {
            available: false,
            result: None,
        }
    }

    // ---- ltm_search tests -----------------------------------------------

    #[test]
    fn search_no_rerank_uses_cosine_verdict() {
        let cfg = cfg();
        let store = FakeStore {
            rows: vec![row(1, 10, 0.9, "hello world"), row(2, 11, 0.5, "other")],
            hnsw: true,
            ..Default::default()
        };
        let embed = FakeEmbed {
            health: "ok".into(),
        };
        let rr = unavailable_reranker();
        let tools = Tools {
            cfg: &cfg,
            store: &store,
            embed: &embed,
            rerank: &rr,
        };

        // rerank=false → no note, cosine order.
        let v = tools.ltm_search("q", 8, false, None, None);
        assert_eq!(v["reranked"], json!(false));
        assert_eq!(v["query"], json!("q"));
        assert_eq!(v["hnsw_index"], json!("ready"));
        assert_eq!(v.get("note"), None);
        // 0.9 >= high_cosine (0.855) → high (cosine only ...)
        assert_eq!(
            v["confidence"],
            json!("high (cosine only — enable rerank for a calibrated verdict)")
        );
        assert_eq!(v["top_score"], json!({ "cosine": 0.9 }));
        assert_eq!(v["results"].as_array().unwrap().len(), 2);
        let first = &v["results"][0];
        assert_eq!(first["path"], json!("/root/path/10.txt"));
        assert_eq!(first["doc_type"], json!("lease"));
        assert_eq!(first["cosine"], json!(0.9));
        assert!(first.get("rerank_score").is_none());
        // fetch_k == k when not reranking
        assert_eq!(*store.last_fetch_k.lock().unwrap(), Some(8));
    }

    #[test]
    fn search_rerank_requested_but_unavailable_adds_note() {
        let cfg = cfg();
        let store = FakeStore {
            rows: vec![row(1, 10, 0.80, "hello")],
            hnsw: false,
            ..Default::default()
        };
        let embed = FakeEmbed {
            health: "ok".into(),
        };
        let rr = unavailable_reranker();
        let tools = Tools {
            cfg: &cfg,
            store: &store,
            embed: &embed,
            rerank: &rr,
        };

        let v = tools.ltm_search("q", 5, true, None, None);
        assert_eq!(v["reranked"], json!(false));
        assert_eq!(
            v["note"],
            json!("rerank requested but reranker unavailable (LTM_RERANK_PYTHON / \
                   LTM_RERANK_SCRIPT not set or not found) — cosine order only")
        );
        assert_eq!(v["hnsw_index"], json!("not built yet (seq scan, slower)"));
        // 0.80 is between low(0.78) and high(0.855) → medium cosine
        assert_eq!(
            v["confidence"],
            json!("medium (cosine only — enable rerank for a calibrated verdict)")
        );
        // Not reranking → fetch_k == k
        assert_eq!(*store.last_fetch_k.lock().unwrap(), Some(5));
    }

    #[test]
    fn search_rerank_success_reorders_and_uses_rerank_verdict() {
        let cfg = cfg();
        let store = FakeStore {
            rows: vec![
                row(1, 10, 0.90, "alpha"),
                row(2, 11, 0.80, "bravo"),
                row(3, 12, 0.70, "charlie"),
            ],
            hnsw: true,
            ..Default::default()
        };
        let embed = FakeEmbed {
            health: "ok".into(),
        };
        // Reranker reorders to rows[2], rows[0] with scores 7.0 (>= high 6.0), 2.0
        let rr = FakeReranker {
            available: true,
            result: Some(Ok((vec![2, 0], vec![7.0, 2.0]))),
        };
        let tools = Tools {
            cfg: &cfg,
            store: &store,
            embed: &embed,
            rerank: &rr,
        };

        let v = tools.ltm_search("q", 2, true, None, None);
        assert_eq!(v["reranked"], json!(true));
        assert_eq!(v.get("note"), None);
        assert_eq!(v["confidence"], json!("high"));
        assert_eq!(v["top_score"], json!({ "rerank_score": 7.0 }));

        let results = v["results"].as_array().unwrap();
        assert_eq!(results.len(), 2);
        // First result should be the reranked winner (chunk 3 / doc 12)
        assert_eq!(results[0]["chunk_id"], json!(3));
        assert_eq!(results[0]["rerank_score"], json!(7.0));
        assert_eq!(results[1]["chunk_id"], json!(1));
        assert_eq!(results[1]["rerank_score"], json!(2.0));
        // Reranking → fetch_k == max(rerank_candidates, k)
        assert_eq!(
            *store.last_fetch_k.lock().unwrap(),
            Some(cfg.rerank_candidates.max(2))
        );
    }

    #[test]
    fn search_rerank_failure_falls_back_to_cosine_with_note() {
        let cfg = cfg();
        let store = FakeStore {
            rows: vec![row(1, 10, 0.90, "alpha"), row(2, 11, 0.60, "bravo")],
            hnsw: true,
            ..Default::default()
        };
        let embed = FakeEmbed {
            health: "ok".into(),
        };
        let rr = FakeReranker {
            available: true,
            result: Some(Err(Error::Rerank("dead".into()))),
        };
        let tools = Tools {
            cfg: &cfg,
            store: &store,
            embed: &embed,
            rerank: &rr,
        };

        let v = tools.ltm_search("q", 2, true, None, None);
        assert_eq!(v["reranked"], json!(false));
        assert_eq!(
            v["note"],
            json!("rerank failed (Rerank) — cosine order only")
        );
        // Cosine order preserved; top cosine 0.90 → high cosine verdict
        assert_eq!(
            v["confidence"],
            json!("high (cosine only — enable rerank for a calibrated verdict)")
        );
        assert_eq!(v["top_score"], json!({ "cosine": 0.9 }));
        assert!(v["results"][0].get("rerank_score").is_none());
    }

    #[test]
    fn search_empty_results_shape() {
        let cfg = cfg();
        let store = FakeStore {
            rows: vec![],
            hnsw: false,
            ..Default::default()
        };
        let embed = FakeEmbed {
            health: "ok".into(),
        };
        let rr = unavailable_reranker();
        let tools = Tools {
            cfg: &cfg,
            store: &store,
            embed: &embed,
            rerank: &rr,
        };

        let v = tools.ltm_search("q", 8, true, Some("lease"), None);
        assert_eq!(
            v,
            json!({
                "query": "q",
                "results": [],
                "confidence": "LOW — no matches (filters may be too narrow)",
                "hnsw_index": "not built yet (seq scan)",
            })
        );
    }

    #[test]
    fn search_clamps_k() {
        let cfg = cfg();
        let store = FakeStore {
            rows: vec![row(1, 10, 0.9, "x")],
            hnsw: true,
            ..Default::default()
        };
        let embed = FakeEmbed {
            health: "ok".into(),
        };
        let rr = unavailable_reranker();
        let tools = Tools {
            cfg: &cfg,
            store: &store,
            embed: &embed,
            rerank: &rr,
        };

        // k=0 clamps to 1; not reranking → fetch_k == 1
        tools.ltm_search("q", 0, false, None, None);
        assert_eq!(*store.last_fetch_k.lock().unwrap(), Some(1));
        // k=999 clamps to 50
        tools.ltm_search("q", 999, false, None, None);
        assert_eq!(*store.last_fetch_k.lock().unwrap(), Some(50));
    }

    // ---- ltm_get_document tests -----------------------------------------

    fn document(text: &str) -> Document {
        Document {
            doc_id: 42,
            path: "/root/leases/a.txt".into(),
            source: Some("src".into()),
            mime: Some("text/plain".into()),
            doc_type: "lease".into(),
            title: Some("A Lease".into()),
            doc_date: Some("2024-01-01".into()),
            bytes: Some(123),
            chunk_count: Some(3),
            char_count: Some(text.chars().count() as i64),
            extraction: Some("pdf".into()),
            ocr: false,
            ingested_at: Some("2024-02-02".into()),
            text: text.into(),
        }
    }

    #[test]
    fn get_document_found_not_truncated() {
        let cfg = cfg();
        let store = FakeStore {
            lookup: Some(DocLookup::Found(Box::new(document("short text")))),
            ..Default::default()
        };
        let embed = FakeEmbed {
            health: "ok".into(),
        };
        let rr = unavailable_reranker();
        let tools = Tools {
            cfg: &cfg,
            store: &store,
            embed: &embed,
            rerank: &rr,
        };

        let v = tools.ltm_get_document(&DocKey::Id(42), 20_000);
        assert_eq!(v["found"], json!(true));
        assert_eq!(v["doc_id"], json!(42));
        assert_eq!(v["path"], json!("/root/leases/a.txt"));
        assert_eq!(v["doc_type"], json!("lease"));
        assert_eq!(v["truncated"], json!(false));
        assert_eq!(v["text"], json!("short text"));
        assert_eq!(v["ocr"], json!(false));
    }

    #[test]
    fn get_document_found_truncates_and_clamps_max_chars() {
        let cfg = cfg();
        let full = "x".repeat(1000);
        let store = FakeStore {
            lookup: Some(DocLookup::Found(Box::new(document(&full)))),
            ..Default::default()
        };
        let embed = FakeEmbed {
            health: "ok".into(),
        };
        let rr = unavailable_reranker();
        let tools = Tools {
            cfg: &cfg,
            store: &store,
            embed: &embed,
            rerank: &rr,
        };

        // max_chars below the 200 floor clamps up to 200.
        let v = tools.ltm_get_document(&DocKey::Id(42), 10);
        assert_eq!(v["truncated"], json!(true));
        assert_eq!(v["text"].as_str().unwrap().len(), 200);
    }

    #[test]
    fn get_document_ambiguous_shape() {
        let cfg = cfg();
        let store = FakeStore {
            lookup: Some(DocLookup::Ambiguous {
                candidates: vec![
                    DocCandidate {
                        doc_id: 1,
                        path: "/r/a.txt".into(),
                        doc_type: "lease".into(),
                        title: Some("A".into()),
                        char_count: Some(10),
                    },
                    DocCandidate {
                        doc_id: 2,
                        path: "/r/b.txt".into(),
                        doc_type: "untyped".into(),
                        title: None,
                        char_count: None,
                    },
                ],
            }),
            ..Default::default()
        };
        let embed = FakeEmbed {
            health: "ok".into(),
        };
        let rr = unavailable_reranker();
        let tools = Tools {
            cfg: &cfg,
            store: &store,
            embed: &embed,
            rerank: &rr,
        };

        let v = tools.ltm_get_document(&DocKey::PathOrFragment("a".into()), 20_000);
        assert_eq!(v["found"], json!(false));
        assert_eq!(v["ambiguous"], json!(true));
        assert_eq!(v["hint"], json!("multiple matches — call again with one doc_id"));
        let cands = v["candidates"].as_array().unwrap();
        assert_eq!(cands.len(), 2);
        assert_eq!(cands[0]["doc_id"], json!(1));
        assert_eq!(cands[1]["char_count"], Value::Null);
    }

    #[test]
    fn get_document_not_found_shape() {
        let cfg = cfg();
        let store = FakeStore {
            lookup: Some(DocLookup::NotFound),
            ..Default::default()
        };
        let embed = FakeEmbed {
            health: "ok".into(),
        };
        let rr = unavailable_reranker();
        let tools = Tools {
            cfg: &cfg,
            store: &store,
            embed: &embed,
            rerank: &rr,
        };

        let v = tools.ltm_get_document(&DocKey::PathOrFragment("nope".into()), 20_000);
        assert_eq!(
            v,
            json!({
                "found": false,
                "error": "no document matches 'nope' (id, path, or fragment)",
            })
        );
    }

    // ---- compat (ChatGPT `search`/`fetch` alias) tests -------------------

    #[test]
    fn compat_search_shape_dedupes_per_document() {
        let cfg = cfg();
        // Two chunks of doc 10 and one of doc 11 — doc 10 must appear once.
        let store = FakeStore {
            rows: vec![
                row(1, 10, 0.90, "first chunk of ten"),
                row(2, 10, 0.88, "second chunk of ten"),
                row(3, 11, 0.86, "chunk of eleven"),
            ],
            hnsw: true,
            ..Default::default()
        };
        let embed = FakeEmbed {
            health: "ok".into(),
        };
        let rr = unavailable_reranker();
        let tools = Tools {
            cfg: &cfg,
            store: &store,
            embed: &embed,
            rerank: &rr,
        };

        let v = tools.compat_search("q");
        let results = v["results"].as_array().unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0]["id"], json!("10"));
        assert_eq!(results[0]["title"], json!("Doc 10"));
        assert_eq!(results[0]["text"], json!("first chunk of ten"));
        assert_eq!(results[0]["url"], json!("ltm://doc/10"));
        assert_eq!(results[1]["id"], json!("11"));
        // Honest confidence passed through (0.9 cosine → high, cosine-only).
        assert_eq!(
            v["confidence"],
            json!("high (cosine only — enable rerank for a calibrated verdict)")
        );
    }

    #[test]
    fn compat_search_title_falls_back_to_path() {
        let cfg = cfg();
        let mut r = row(1, 10, 0.9, "x");
        r.title = None;
        let store = FakeStore {
            rows: vec![r],
            hnsw: true,
            ..Default::default()
        };
        let embed = FakeEmbed {
            health: "ok".into(),
        };
        let rr = unavailable_reranker();
        let tools = Tools {
            cfg: &cfg,
            store: &store,
            embed: &embed,
            rerank: &rr,
        };

        let v = tools.compat_search("q");
        assert_eq!(v["results"][0]["title"], json!("/root/path/10.txt"));
    }

    #[test]
    fn compat_fetch_found_shape() {
        let cfg = cfg();
        let store = FakeStore {
            lookup: Some(DocLookup::Found(Box::new(document("full text here")))),
            ..Default::default()
        };
        let embed = FakeEmbed {
            health: "ok".into(),
        };
        let rr = unavailable_reranker();
        let tools = Tools {
            cfg: &cfg,
            store: &store,
            embed: &embed,
            rerank: &rr,
        };

        let v = tools.compat_fetch(&DocKey::Id(42));
        assert_eq!(v["id"], json!("42"));
        assert_eq!(v["title"], json!("A Lease"));
        assert_eq!(v["text"], json!("full text here"));
        assert_eq!(v["url"], json!("ltm://doc/42"));
        assert_eq!(v["metadata"]["path"], json!("/root/leases/a.txt"));
        assert_eq!(v["metadata"]["doc_type"], json!("lease"));
        assert_eq!(v["metadata"]["ocr"], json!(false));
        assert_eq!(v["metadata"]["truncated"], json!(false));
        assert_eq!(v.get("error"), None);
    }

    #[test]
    fn compat_fetch_ambiguous_is_error_listing_candidates() {
        let cfg = cfg();
        let store = FakeStore {
            lookup: Some(DocLookup::Ambiguous {
                candidates: vec![
                    DocCandidate {
                        doc_id: 1,
                        path: "/r/a.txt".into(),
                        doc_type: "lease".into(),
                        title: Some("A".into()),
                        char_count: Some(10),
                    },
                    DocCandidate {
                        doc_id: 2,
                        path: "/r/b.txt".into(),
                        doc_type: "untyped".into(),
                        title: None,
                        char_count: None,
                    },
                ],
            }),
            ..Default::default()
        };
        let embed = FakeEmbed {
            health: "ok".into(),
        };
        let rr = unavailable_reranker();
        let tools = Tools {
            cfg: &cfg,
            store: &store,
            embed: &embed,
            rerank: &rr,
        };

        let v = tools.compat_fetch(&DocKey::PathOrFragment("a".into()));
        let err = v["error"].as_str().unwrap();
        assert!(err.starts_with("ambiguous id 'a'"), "got: {err}");
        assert!(err.contains("1: /r/a.txt"), "got: {err}");
        assert!(err.contains("2: /r/b.txt"), "got: {err}");
    }

    #[test]
    fn compat_fetch_not_found_is_error() {
        let cfg = cfg();
        let store = FakeStore {
            lookup: Some(DocLookup::NotFound),
            ..Default::default()
        };
        let embed = FakeEmbed {
            health: "ok".into(),
        };
        let rr = unavailable_reranker();
        let tools = Tools {
            cfg: &cfg,
            store: &store,
            embed: &embed,
            rerank: &rr,
        };

        let v = tools.compat_fetch(&DocKey::PathOrFragment("nope".into()));
        assert_eq!(
            v["error"],
            json!("no document matches 'nope' (id, path, or fragment)")
        );
    }

    // ---- ltm_stats tests -------------------------------------------------

    fn stats_fixture(chunks: i64, emb: i64, hnsw: bool) -> Stats {
        let mut by_type = BTreeMap::new();
        by_type.insert("lease".to_string(), 5);
        by_type.insert("untyped".to_string(), 2);
        let mut sizes = BTreeMap::new();
        sizes.insert("documents".to_string(), "1 MB".to_string());
        sizes.insert("doc_chunks".to_string(), "2 MB".to_string());
        sizes.insert("emb_mxbai".to_string(), "3 MB".to_string());
        sizes.insert("database".to_string(), "10 MB".to_string());
        Stats {
            documents: 7,
            chunks,
            mxbai_embeddings: emb,
            documents_by_type: by_type,
            hnsw_ready: hnsw,
            table_sizes: sizes,
        }
    }

    #[test]
    fn stats_shape_with_coverage_and_health() {
        let cfg = cfg();
        let store = FakeStore {
            stats: Some(stats_fixture(200, 199, true)),
            ..Default::default()
        };
        let embed = FakeEmbed {
            health: "ok (mxbai-embed-large present)".into(),
        };
        let rr = FakeReranker {
            available: true,
            result: None,
        };
        let tools = Tools {
            cfg: &cfg,
            store: &store,
            embed: &embed,
            rerank: &rr,
        };

        let v = tools.ltm_stats();
        assert_eq!(v["documents"], json!(7));
        assert_eq!(v["chunks"], json!(200));
        assert_eq!(v["mxbai_embeddings"], json!(199));
        assert_eq!(v["embedding_coverage"], json!("99.50%"));
        assert_eq!(v["hnsw_index"], json!("ready — fast approximate search"));
        assert_eq!(v["embed_model"], json!(cfg.embed_model));
        assert_eq!(v["ollama"], json!("ok (mxbai-embed-large present)"));
        assert_eq!(v["reranker"], json!("available"));
        assert_eq!(v["read_only"], json!(true));
        assert_eq!(v["documents_by_type"]["lease"], json!(5));
        assert_eq!(v["table_sizes"]["database"], json!("10 MB"));
    }

    #[test]
    fn stats_zero_chunks_and_unavailable_reranker() {
        let cfg = cfg();
        let store = FakeStore {
            stats: Some(stats_fixture(0, 0, false)),
            ..Default::default()
        };
        let embed = FakeEmbed {
            health: "unreachable (ConnectError)".into(),
        };
        let rr = unavailable_reranker();
        let tools = Tools {
            cfg: &cfg,
            store: &store,
            embed: &embed,
            rerank: &rr,
        };

        let v = tools.ltm_stats();
        assert_eq!(v["embedding_coverage"], json!("n/a"));
        assert_eq!(
            v["hnsw_index"],
            json!("not built yet — searches fall back to sequential scan (~10 s each); \
                   auto-build pending")
        );
        assert_eq!(
            v["reranker"],
            json!("unavailable (set LTM_RERANK_PYTHON / LTM_RERANK_SCRIPT)")
        );
        assert_eq!(v["ollama"], json!("unreachable (ConnectError)"));
    }
}
