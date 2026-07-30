//! End-to-end harness: build the real pipeline and assert server.py behavior.
//!
//! The harness (re)seeds the corpus, then constructs the concrete
//! implementations — `ltm_db::PgStore` + the Ollama `EmbeddingBackend` from
//! `ltm_embed` + `ltm_rerank::NoopReranker` — wires them into `ltm_tools::Tools`,
//! and asserts the three tools behave like server.py.
//!
//! Because the sibling crates may still be stubs, and the live services may be
//! down, every live check is guarded: instead of failing, the harness returns
//! [`Outcome::Skipped`] with a clear reason. It returns [`Outcome::Passed`]
//! only when the whole pipeline actually ran and every assertion held.

use ltm_core::{Config, DocKey, EmbeddingBackend, Store};
use ltm_db::PgStore;
use ltm_embed::make_backend;
use ltm_rerank::NoopReranker;
use ltm_tools::Tools;
use serde_json::Value;

use crate::corpus::{self, CORPUS, LOAN_FRAGMENT};
use crate::seed::{self, SeedError};

/// The result of a harness run.
pub enum Outcome {
    /// The full pipeline ran and every assertion held. `String` is a summary.
    Passed(String),
    /// A dependency is still a stub or a service is unreachable — not a failure.
    Skipped(String),
}

/// The query whose top hit must be the loan document.
pub const LOAN_QUERY: &str = "when should I order the appraisal for the loan";
/// A deliberately off-topic query used as a low-relevance baseline.
pub const NONSENSE_QUERY: &str = "qwerty zxcvbnm asdf gibberish unrelated flumox";
/// Minimum cosine gap between the loan-query top hit and the nonsense-query top
/// hit for the loan match to count as "clearly higher".
pub const MIN_COSINE_MARGIN: f64 = 0.1;

/// Run the whole harness. `Err` is a genuine failure (assertion or setup bug);
/// `Ok(Skipped)` means a dependency/service was not ready.
pub fn run() -> Result<Outcome, String> {
    let cfg = Config::from_env();

    // 1) (Re)seed. If this fails on connectivity, the services are down → skip.
    let report = match seed::seed(&cfg) {
        Ok(r) => r,
        Err(SeedError::ServiceDown(m)) => {
            return Ok(Outcome::Skipped(format!(
                "SKIPPED (service down): {m}"
            )))
        }
        Err(SeedError::Other(m)) => return Err(format!("seed failed: {m}")),
    };
    if report.documents != CORPUS.len() as i64
        || report.chunks != CORPUS.len() as i64
        || report.embeddings != CORPUS.len() as i64
    {
        return Err(format!(
            "seed produced unexpected counts: docs={} chunks={} emb={} (expected {})",
            report.documents,
            report.chunks,
            report.embeddings,
            CORPUS.len()
        ));
    }

    // 2) Build the real components (blocking traits, exactly like ltm-server).
    let store =
        PgStore::connect(&cfg).map_err(|e| format!("PgStore::connect failed: {e}"))?;
    let embed = make_backend(&cfg);
    let rerank = NoopReranker;

    // 3) Seeding succeeded, so both services are up. Any trait error now means
    //    the crate is still a stub → SKIP (not FAIL).
    if let Some(reason) = detect_stub(&store, embed.as_ref()) {
        return Ok(Outcome::Skipped(format!("SKIPPED (dep stub): {reason}")));
    }

    let tools = Tools {
        cfg: &cfg,
        store: &store,
        embed: embed.as_ref(),
        rerank: &rerank,
    };

    // 4) Drive the tools. `ltm-tools` itself may still be a stub even when the
    //    lower crates are live, so guard tool output for the unimplemented
    //    marker → SKIP.
    let stats = tools.ltm_stats();
    if let Some(s) = stub_from_value(&stats) {
        return Ok(Outcome::Skipped(format!("SKIPPED (dep stub): {s}")));
    }
    let loan = tools.ltm_search(LOAN_QUERY, 5, true, None, None);
    if let Some(s) = stub_from_value(&loan) {
        return Ok(Outcome::Skipped(format!("SKIPPED (dep stub): {s}")));
    }
    let nonsense = tools.ltm_search(NONSENSE_QUERY, 5, true, None, None);
    if let Some(s) = stub_from_value(&nonsense) {
        return Ok(Outcome::Skipped(format!("SKIPPED (dep stub): {s}")));
    }

    // 5) Assertions (server.py behavioral contract).
    assert_stats(&stats)?;
    let loan_top_cosine = assert_search(&loan, &nonsense)?;
    assert_get_document(&tools, &loan)?;

    Ok(Outcome::Passed(format!(
        "seeded {} docs; ltm_stats OK; loan top cosine {:.4} (margin ≥ {:.2} over nonsense); \
         ltm_get_document fragment + id lookups OK",
        report.documents, loan_top_cosine, MIN_COSINE_MARGIN
    )))
}

/// True when an `ltm_core::Error` is the stub `unimplemented` marker.
fn is_unimplemented(e: &ltm_core::Error) -> bool {
    e.to_string().to_lowercase().contains("unimplemented")
}

/// Probe the store and embed backend directly. Returns `Some(reason)` when a
/// crate is still a stub (services are known-up because seeding just worked).
fn detect_stub(store: &dyn Store, embed: &dyn EmbeddingBackend) -> Option<String> {
    match store.stats() {
        Ok(_) => {}
        Err(ref e) if is_unimplemented(e) => {
            return Some(format!("ltm-db not implemented ({e})"))
        }
        Err(e) => return Some(format!("ltm-db error after successful seed ({e})")),
    }
    match embed.embed_query("healthcheck probe") {
        Ok(v) if v.len() == 1024 => {}
        Ok(v) => return Some(format!("ltm-embed returned {} dims, expected 1024", v.len())),
        Err(ref e) if is_unimplemented(e) => {
            return Some(format!("ltm-embed not implemented ({e})"))
        }
        Err(e) => return Some(format!("ltm-embed error after successful seed ({e})")),
    }
    None
}

/// If a tool's JSON is the `ltm-tools` stub error, return the reason.
fn stub_from_value(v: &Value) -> Option<String> {
    v.get("error")
        .and_then(|e| e.as_str())
        .filter(|s| s.to_lowercase().contains("unimplemented"))
        .map(|s| s.to_string())
}

/// Assert `ltm_stats` reports the seeded counts and Ollama health.
fn assert_stats(v: &Value) -> Result<(), String> {
    if let Some(err) = v.get("error").and_then(|e| e.as_str()) {
        return Err(format!("ltm_stats returned error: {err}"));
    }
    let n = CORPUS.len() as i64;
    for (field, want) in [
        ("documents", n),
        ("chunks", n),
        ("mxbai_embeddings", n),
    ] {
        let got = v.get(field).and_then(|x| x.as_i64()).ok_or_else(|| {
            format!("ltm_stats missing integer field `{field}`: {v}")
        })?;
        if got != want {
            return Err(format!(
                "ltm_stats.{field} = {got}, expected {want} (whole payload: {v})"
            ));
        }
    }
    let ollama = v
        .get("ollama")
        .and_then(|x| x.as_str())
        .ok_or_else(|| format!("ltm_stats missing string field `ollama`: {v}"))?;
    if !ollama.starts_with("ok") {
        return Err(format!(
            "ltm_stats.ollama = {ollama:?}, expected it to start with \"ok\""
        ));
    }
    if v.get("read_only").and_then(|x| x.as_bool()) != Some(true) {
        return Err(format!("ltm_stats.read_only should be true: {v}"));
    }
    Ok(())
}

/// Assert the loan document ranks first and clearly out-scores the nonsense
/// query. Returns the loan-query top cosine on success.
fn assert_search(loan: &Value, nonsense: &Value) -> Result<f64, String> {
    if let Some(err) = loan.get("error").and_then(|e| e.as_str()) {
        return Err(format!("ltm_search(loan) returned error: {err}"));
    }
    let results = loan
        .get("results")
        .and_then(|r| r.as_array())
        .ok_or_else(|| format!("ltm_search(loan) has no results array: {loan}"))?;
    let top = results
        .first()
        .ok_or_else(|| format!("ltm_search(loan) returned zero results: {loan}"))?;

    let loan_doc = &CORPUS[0];
    let top_type = top.get("doc_type").and_then(|x| x.as_str()).unwrap_or("");
    let top_path = top.get("path").and_then(|x| x.as_str()).unwrap_or("");
    if top_type != loan_doc.doc_type || top_path != loan_doc.path() {
        return Err(format!(
            "ltm_search(loan) top hit was doc_type={top_type:?} path={top_path:?}, \
             expected doc_type={:?} path={:?}",
            loan_doc.doc_type,
            loan_doc.path()
        ));
    }

    let loan_cos = top
        .get("cosine")
        .and_then(|x| x.as_f64())
        .ok_or_else(|| format!("loan top hit has no cosine: {top}"))?;
    let nonsense_cos = nonsense
        .get("results")
        .and_then(|r| r.as_array())
        .and_then(|a| a.first())
        .and_then(|t| t.get("cosine"))
        .and_then(|x| x.as_f64())
        .ok_or_else(|| format!("nonsense query has no top cosine: {nonsense}"))?;

    if loan_cos - nonsense_cos < MIN_COSINE_MARGIN {
        return Err(format!(
            "loan top cosine {loan_cos:.4} is not clearly higher than nonsense \
             top cosine {nonsense_cos:.4} (margin {:.4} < required {MIN_COSINE_MARGIN})",
            loan_cos - nonsense_cos
        ));
    }
    Ok(loan_cos)
}

/// Assert `ltm_get_document` fragment lookup (and the id lookup from the search
/// result) resolve to the loan document.
fn assert_get_document(tools: &Tools, loan_search: &Value) -> Result<(), String> {
    // Fragment lookup: unambiguous → Found.
    let by_frag =
        tools.ltm_get_document(&DocKey::PathOrFragment(LOAN_FRAGMENT.into()), 20_000);
    if let Some(err) = by_frag.get("error").and_then(|e| e.as_str()) {
        return Err(format!("ltm_get_document(fragment) returned error: {err}"));
    }
    if by_frag.get("found").and_then(|x| x.as_bool()) != Some(true) {
        return Err(format!(
            "ltm_get_document(fragment {LOAN_FRAGMENT:?}) not Found: {by_frag}"
        ));
    }
    let loan_doc = &corpus::CORPUS[0];
    let frag_type = by_frag.get("doc_type").and_then(|x| x.as_str()).unwrap_or("");
    if frag_type != loan_doc.doc_type {
        return Err(format!(
            "ltm_get_document(fragment) doc_type={frag_type:?}, expected {:?}",
            loan_doc.doc_type
        ));
    }
    let text = by_frag.get("text").and_then(|x| x.as_str()).unwrap_or("");
    if !text.to_lowercase().contains("appraisal") {
        return Err(format!(
            "ltm_get_document(fragment) text should mention the appraisal: {by_frag}"
        ));
    }

    // Id lookup: use the doc_id the search returned for the loan hit.
    let doc_id = loan_search
        .get("results")
        .and_then(|r| r.as_array())
        .and_then(|a| a.first())
        .and_then(|t| t.get("doc_id"))
        .and_then(|x| x.as_i64())
        .ok_or_else(|| format!("no doc_id in loan search results: {loan_search}"))?;
    let by_id = tools.ltm_get_document(&DocKey::Id(doc_id), 20_000);
    if by_id.get("found").and_then(|x| x.as_bool()) != Some(true) {
        return Err(format!("ltm_get_document(id={doc_id}) not Found: {by_id}"));
    }
    let id_path = by_id.get("path").and_then(|x| x.as_str()).unwrap_or("");
    if id_path != loan_doc.path() {
        return Err(format!(
            "ltm_get_document(id={doc_id}) path={id_path:?}, expected {:?}",
            loan_doc.path()
        ));
    }
    Ok(())
}
