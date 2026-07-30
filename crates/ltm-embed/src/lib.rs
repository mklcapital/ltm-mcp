//! ltm-embed — pluggable embedding backends. Implements
//! [`ltm_core::EmbeddingBackend`].
//!
//! Maintainer requirement: the model backend must be swappable via config —
//! local Ollama today, but OpenRouter / LM Studio / any OpenAI-compatible
//! endpoint / cloud provider without code changes. [`make_backend`] selects the
//! concrete backend from [`Config`].
//!
//! STUB: both backends return `Error::Embed("unimplemented")`. The owning agent
//! implements the HTTP calls (see PORTING.md → ltm-embed):
//!   - Ollama: POST `{model, prompt: QUERY_PREFIX + text}` to `ollama_url`,
//!     read `.embedding`;
//!   - OpenAI-compatible: POST `{model, input}` to `{base_url}/v1/embeddings`
//!     with `Authorization: Bearer <key>`, read `.data[0].embedding`.

use ltm_core::{Config, EmbedBackendKind, EmbeddingBackend, Error, Result};

/// Select and construct the configured embedding backend.
pub fn make_backend(cfg: &Config) -> Box<dyn EmbeddingBackend> {
    match cfg.embed_backend {
        EmbedBackendKind::Ollama => Box::new(OllamaBackend { cfg: cfg.clone() }),
        EmbedBackendKind::OpenAiCompatible => Box::new(OpenAiBackend { cfg: cfg.clone() }),
    }
}

pub struct OllamaBackend {
    #[allow(dead_code)]
    cfg: Config,
}

impl EmbeddingBackend for OllamaBackend {
    fn embed_query(&self, _text: &str) -> Result<Vec<f32>> {
        Err(Error::Embed("ltm-embed::Ollama unimplemented".into()))
    }
    fn describe(&self) -> String {
        format!("ollama({})", self.cfg.embed_model)
    }
    fn health(&self) -> String {
        "unknown (stub)".into()
    }
}

pub struct OpenAiBackend {
    #[allow(dead_code)]
    cfg: Config,
}

impl EmbeddingBackend for OpenAiBackend {
    fn embed_query(&self, _text: &str) -> Result<Vec<f32>> {
        Err(Error::Embed("ltm-embed::OpenAiCompatible unimplemented".into()))
    }
    fn describe(&self) -> String {
        format!("openai-compatible({})", self.cfg.embed_model)
    }
    fn health(&self) -> String {
        "unknown (stub)".into()
    }
}
