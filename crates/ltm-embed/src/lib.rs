//! ltm-embed — pluggable embedding backends. Implements
//! [`ltm_core::EmbeddingBackend`].
//!
//! Maintainer requirement: the model backend must be swappable via config —
//! local Ollama today, but OpenRouter / LM Studio / any OpenAI-compatible
//! endpoint / cloud provider without code changes. [`make_backend`] selects the
//! concrete backend from [`Config`].
//!
//! Both backends mirror `server.py`:
//!   - Ollama: POST `{model, prompt: query_prefix + text}` to `ollama_url`,
//!     read `.embedding`; `health()` GETs `{ollama_base}/api/tags`.
//!   - OpenAI-compatible: POST `{model, input: query_prefix + text}` to
//!     `{embed_base_url}/v1/embeddings` with `Authorization: Bearer <key>`
//!     (when a key is configured), read `.data[0].embedding`.

use std::time::Duration;

use ltm_core::{Config, EmbedBackendKind, EmbeddingBackend, Error, Result};
use serde_json::Value;

/// Per-request timeout for embedding calls (matches server.py's 60s).
const EMBED_TIMEOUT: Duration = Duration::from_secs(60);
/// Shorter timeout for health probes (matches server.py's 5s `/api/tags`).
const HEALTH_TIMEOUT: Duration = Duration::from_secs(5);

/// Select and construct the configured embedding backend.
pub fn make_backend(cfg: &Config) -> Box<dyn EmbeddingBackend> {
    match cfg.embed_backend {
        EmbedBackendKind::Ollama => Box::new(OllamaBackend { cfg: cfg.clone() }),
        EmbedBackendKind::OpenAiCompatible => Box::new(OpenAiBackend { cfg: cfg.clone() }),
    }
}

/// Build a `ureq` agent with the given read/connect timeout.
fn agent(timeout: Duration) -> ureq::Agent {
    ureq::AgentBuilder::new()
        .timeout_connect(timeout)
        .timeout_read(timeout)
        .timeout_write(timeout)
        .build()
}

/// Map a `ureq` error into a concise, human-readable label (used for the
/// "unreachable (...)" health strings, mirroring server.py's exception name).
fn err_label(e: &ureq::Error) -> String {
    match e {
        ureq::Error::Status(code, _) => format!("HTTP {code}"),
        ureq::Error::Transport(t) => match t.kind() {
            ureq::ErrorKind::Dns => "DnsError".into(),
            ureq::ErrorKind::ConnectionFailed => "ConnectionError".into(),
            ureq::ErrorKind::Io => "IOError".into(),
            other => format!("{other:?}"),
        },
    }
}

/// Parse a JSON number array into `Vec<f32>` from the value at `field`.
fn parse_embedding(v: &Value, field_path: &[&str]) -> Result<Vec<f32>> {
    let mut cur = v;
    for key in field_path {
        cur = cur
            .get(key)
            .ok_or_else(|| Error::Embed(format!("embedding response missing `{key}`")))?;
    }
    let arr = cur
        .as_array()
        .ok_or_else(|| Error::Embed("embedding field is not an array".into()))?;
    let mut out = Vec::with_capacity(arr.len());
    for x in arr {
        let f = x
            .as_f64()
            .ok_or_else(|| Error::Embed("embedding contains a non-numeric value".into()))?;
        out.push(f as f32);
    }
    if out.is_empty() {
        return Err(Error::Embed("embedding is empty".into()));
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Ollama (local `/api/embeddings`)
// ---------------------------------------------------------------------------

pub struct OllamaBackend {
    cfg: Config,
}

impl OllamaBackend {
    /// Derive the Ollama base URL from the embeddings URL, exactly like
    /// server.py: `OLLAMA_EMBED_URL.rsplit("/api/", 1)[0]`.
    fn base_url(&self) -> String {
        match self.cfg.ollama_url.rsplit_once("/api/") {
            Some((base, _)) => base.to_string(),
            None => self.cfg.ollama_url.clone(),
        }
    }
}

impl EmbeddingBackend for OllamaBackend {
    fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        let prompt = format!("{}{}", self.cfg.query_prefix, text);
        let body = serde_json::json!({
            "model": self.cfg.embed_model,
            "prompt": prompt,
        });
        let resp = agent(EMBED_TIMEOUT)
            .post(&self.cfg.ollama_url)
            .set("Content-Type", "application/json")
            .send_json(body)
            .map_err(|e| Error::Embed(format!("ollama request failed: {e}")))?;
        let json: Value = resp
            .into_json()
            .map_err(|e| Error::Embed(format!("ollama returned invalid JSON: {e}")))?;
        parse_embedding(&json, &["embedding"])
    }

    fn describe(&self) -> String {
        format!("ollama({})", self.cfg.embed_model)
    }

    fn health(&self) -> String {
        let url = format!("{}/api/tags", self.base_url());
        let model = &self.cfg.embed_model;
        match agent(HEALTH_TIMEOUT).get(&url).call() {
            Ok(resp) => match resp.into_json::<Value>() {
                Ok(json) => {
                    let present = json
                        .get("models")
                        .and_then(|m| m.as_array())
                        .map(|models| {
                            models.iter().any(|m| {
                                m.get("name")
                                    .and_then(|n| n.as_str())
                                    .is_some_and(|n| n.starts_with(model.as_str()))
                            })
                        })
                        .unwrap_or(false);
                    if present {
                        format!("ok ({model} present)")
                    } else {
                        format!("ok, but {model} NOT pulled")
                    }
                }
                Err(e) => format!("unreachable ({e})"),
            },
            Err(e) => format!("unreachable ({})", err_label(&e)),
        }
    }
}

// ---------------------------------------------------------------------------
// OpenAI-compatible (`/v1/embeddings`): OpenRouter, LM Studio, on-prem, cloud
// ---------------------------------------------------------------------------

pub struct OpenAiBackend {
    cfg: Config,
}

impl OpenAiBackend {
    /// Configured base URL with any trailing slash trimmed.
    fn base_url(&self) -> Result<String> {
        let base = self.cfg.embed_base_url.as_deref().ok_or_else(|| {
            Error::Config(
                "LTM_EMBED_BASE_URL is required for the OpenAI-compatible backend".into(),
            )
        })?;
        Ok(base.trim_end_matches('/').to_string())
    }
}

impl EmbeddingBackend for OpenAiBackend {
    fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        let url = format!("{}/v1/embeddings", self.base_url()?);
        let input = format!("{}{}", self.cfg.query_prefix, text);
        let body = serde_json::json!({
            "model": self.cfg.embed_model,
            "input": input,
        });
        let mut req = agent(EMBED_TIMEOUT)
            .post(&url)
            .set("Content-Type", "application/json");
        if let Some(key) = self.cfg.embed_api_key.as_deref() {
            req = req.set("Authorization", &format!("Bearer {key}"));
        }
        let resp = req
            .send_json(body)
            .map_err(|e| Error::Embed(format!("openai-compatible request failed: {e}")))?;
        let json: Value = resp
            .into_json()
            .map_err(|e| Error::Embed(format!("openai-compatible returned invalid JSON: {e}")))?;
        // OpenAI-compatible shape: `.data[0].embedding`.
        let first = json
            .get("data")
            .and_then(|d| d.as_array())
            .ok_or_else(|| Error::Embed("openai response missing `data`".into()))?
            .first()
            .ok_or_else(|| Error::Embed("openai `data` array is empty".into()))?;
        parse_embedding(first, &["embedding"])
    }

    fn describe(&self) -> String {
        format!("openai-compatible({})", self.cfg.embed_model)
    }

    fn health(&self) -> String {
        let base = match self.base_url() {
            Ok(b) => b,
            Err(e) => return format!("unreachable ({e})"),
        };
        // Best-effort reachability: probe the `/v1/models` listing.
        let url = format!("{base}/v1/models");
        let mut req = agent(HEALTH_TIMEOUT).get(&url);
        if let Some(key) = self.cfg.embed_api_key.as_deref() {
            req = req.set("Authorization", &format!("Bearer {key}"));
        }
        match req.call() {
            Ok(_) => format!("ok (reachable at {base})"),
            // A 4xx still proves the endpoint is reachable (e.g. auth/route).
            Err(ureq::Error::Status(code, _)) => {
                format!("ok (reachable at {base}, HTTP {code})")
            }
            Err(e) => format!("unreachable ({})", err_label(&e)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ltm_core::Config;

    fn base_cfg() -> Config {
        Config::from_env()
    }

    #[test]
    fn ollama_base_url_derivation() {
        let mut cfg = base_cfg();
        cfg.ollama_url = "http://127.0.0.1:11434/api/embeddings".into();
        let b = OllamaBackend { cfg };
        assert_eq!(b.base_url(), "http://127.0.0.1:11434");
    }

    #[test]
    fn openai_base_url_trims_slash() {
        let mut cfg = base_cfg();
        cfg.embed_base_url = Some("https://openrouter.ai/api/".into());
        let b = OpenAiBackend { cfg };
        assert_eq!(b.base_url().unwrap(), "https://openrouter.ai/api");
    }

    #[test]
    fn parse_embedding_ollama_shape() {
        let v = serde_json::json!({ "embedding": [0.1, 0.2, 0.3] });
        let out = parse_embedding(&v, &["embedding"]).unwrap();
        assert_eq!(out, vec![0.1f32, 0.2, 0.3]);
    }

    #[test]
    fn parse_embedding_rejects_empty() {
        let v = serde_json::json!({ "embedding": [] });
        assert!(parse_embedding(&v, &["embedding"]).is_err());
    }

    /// Live integration test against a local Ollama. Skips gracefully when the
    /// server is unreachable so hermetic CI still passes.
    #[test]
    fn live_ollama_round_trip() {
        let cfg = base_cfg();
        // Only meaningful for the Ollama backend.
        if cfg.embed_backend != EmbedBackendKind::Ollama {
            eprintln!("skipping: configured backend is not Ollama");
            return;
        }
        let backend = OllamaBackend { cfg: cfg.clone() };
        // Probe health first; skip if Ollama is not up.
        let health = backend.health();
        if health.starts_with("unreachable") {
            eprintln!("skipping live_ollama_round_trip: {health}");
            return;
        }
        match backend.embed_query("what is long term memory?") {
            Ok(vec) => {
                assert!(!vec.is_empty(), "embedding should be non-empty");
                // mxbai-embed-large is 1024-dim.
                if cfg.embed_model.starts_with("mxbai-embed-large") {
                    assert_eq!(vec.len(), 1024, "mxbai-embed-large is 1024-dim");
                }
            }
            Err(e) => panic!("live embedding failed despite healthy Ollama: {e}"),
        }
    }
}
