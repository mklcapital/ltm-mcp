//! ltm-e2e — integration / end-to-end harness for the Rust LTM-MCP port.
//!
//! This crate verifies the whole retrieval pipeline against the **live**
//! services (Postgres + pgvector and Ollama), mirroring how the Python
//! `server.py` was verified:
//!
//!   1. [`seed`] (re)creates the schema and inserts a small loan/lease/tax/
//!      insurance corpus, embedding each chunk RAW via Ollama (1024-dim
//!      `mxbai-embed-large` vectors) — the asymmetric-retrieval setup where
//!      documents are embedded without the query prefix.
//!   2. [`harness`] builds the real `ltm_db::PgStore` + Ollama backend +
//!      `ltm_rerank::NoopReranker`, constructs `ltm_tools::Tools`, and asserts
//!      that `ltm_stats`, `ltm_search`, and `ltm_get_document` behave like
//!      server.py.
//!
//! The sibling crates (`ltm-db`, `ltm-embed`, `ltm-tools`) may still be stubs
//! while they are implemented in parallel. This harness only depends on the
//! FROZEN `ltm-core` contracts, so it **compiles today** and turns **green
//! once the stubs are implemented**. Every live check is guarded so it SKIPS
//! (never fails) when a dependency is still a stub or a service is unreachable.

pub mod corpus;
pub mod embed;
pub mod harness;
pub mod seed;

pub use harness::{run, Outcome};
