//! Direct Ollama embedding for the seed utility.
//!
//! Why not use `ltm-embed` here? The `EmbeddingBackend` trait only exposes
//! `embed_query`, which (per the mxbai model card and server.py) prepends the
//! asymmetric-retrieval **query** prefix. Corpus documents were embedded RAW
//! (no prefix) at ingestion. To keep the seed faithful to server.py's
//! asymmetric setup — documents raw, queries prefixed — we embed the chunk text
//! directly against Ollama with no prefix. This also makes seeding independent
//! of the (currently stubbed) `ltm-embed` crate.

use ltm_core::Config;

/// Embed `text` RAW (no query prefix) via the configured Ollama endpoint.
/// Returns the raw 1024-dim vector, or an error string on any transport/parse
/// failure (treated by callers as "service down").
pub fn embed_document(cfg: &Config, text: &str) -> Result<Vec<f32>, String> {
    let body = serde_json::json!({
        "model": cfg.embed_model,
        "prompt": text,
    })
    .to_string();

    let resp = ureq::post(&cfg.ollama_url)
        .set("Content-Type", "application/json")
        .send_string(&body)
        .map_err(|e| format!("ollama POST {} failed: {e}", cfg.ollama_url))?;

    let v: serde_json::Value = resp
        .into_json()
        .map_err(|e| format!("ollama response not JSON: {e}"))?;

    let arr = v
        .get("embedding")
        .and_then(|e| e.as_array())
        .ok_or_else(|| format!("ollama response missing .embedding: {v}"))?;

    let emb: Vec<f32> = arr
        .iter()
        .map(|x| x.as_f64().unwrap_or(0.0) as f32)
        .collect();

    if emb.is_empty() {
        return Err("ollama returned an empty embedding".to_string());
    }
    Ok(emb)
}

/// Format an embedding as a pgvector text literal (`[a,b,c]`), matching the
/// 6-decimal formatting server.py uses before `CAST(... AS vector)`.
pub fn to_vector_literal(emb: &[f32]) -> String {
    let mut s = String::with_capacity(emb.len() * 10 + 2);
    s.push('[');
    for (i, x) in emb.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        s.push_str(&format!("{x:.6}"));
    }
    s.push(']');
    s
}
