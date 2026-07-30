//! ltm-rerank — optional cross-encoder reranker. Implements
//! [`ltm_core::Reranker`].
//!
//! server.py runs a persistent Python worker subprocess (loads
//! mxbai-rerank-base-v2 once, serves JSONL requests over stdin/stdout). The
//! port keeps rerank OPTIONAL and configurable.
//!
//! - [`NoopReranker`] is the default when no reranker is configured
//!   (`available() == false`) → tools fall back to cosine confidence.
//! - [`WorkerReranker`] is a STUB for the owning agent to implement (see
//!   PORTING.md → ltm-rerank): spawn the persistent worker per
//!   `rerank_python`/`rerank_script`, JSONL protocol, wall-clock timeout,
//!   restart-on-death with one retry.

use ltm_core::{Config, Error, Reranker, Result};

/// Always-unavailable reranker: makes `ltm_search` use the cosine gate.
pub struct NoopReranker;

impl Reranker for NoopReranker {
    fn rerank(&self, _q: &str, _texts: &[String], _top_k: usize) -> Result<(Vec<usize>, Vec<f64>)> {
        Err(Error::Rerank("reranker not available".into()))
    }
    fn available(&self) -> bool {
        false
    }
}

pub struct WorkerReranker {
    #[allow(dead_code)]
    cfg: Config,
}

impl WorkerReranker {
    pub fn new(cfg: &Config) -> Self {
        WorkerReranker { cfg: cfg.clone() }
    }
}

impl Reranker for WorkerReranker {
    fn rerank(&self, _q: &str, _texts: &[String], _top_k: usize) -> Result<(Vec<usize>, Vec<f64>)> {
        Err(Error::Rerank("ltm-rerank::WorkerReranker unimplemented".into()))
    }
    fn available(&self) -> bool {
        // Real impl: true only when rerank_python + rerank_script are set/exist.
        self.cfg.rerank_python.is_some() && self.cfg.rerank_script.is_some()
    }
}

/// Choose a reranker from config: the worker when configured, else no-op.
pub fn make_reranker(cfg: &Config) -> Box<dyn Reranker> {
    if cfg.rerank_python.is_some() && cfg.rerank_script.is_some() {
        Box::new(WorkerReranker::new(cfg))
    } else {
        Box::new(NoopReranker)
    }
}
