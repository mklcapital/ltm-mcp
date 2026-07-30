//! End-to-end tests for the persistent rerank worker subprocess.
//!
//! No real cross-encoder model is installed on the dev box (rerank is
//! optional), so these tests stand in a tiny FAKE `rerank.py` that speaks the
//! same JSONL protocol as the real one: it exposes a `Reranker` class with
//! `_load()` and `rerank(query, texts, top_k)`. The fake returns identity order
//! with simple decreasing scores, and has deliberate hooks to exercise the
//! timeout and restart-on-death paths:
//!
//! - query `"SLEEP"`  → the worker sleeps far longer than the timeout.
//! - query `"DIE_ONCE"` → the FIRST spawned worker exits mid-request; a fresh
//!   worker started for the retry answers normally.
//!
//! The fake tracks how many worker processes have been spawned via a
//! `spawn_count` file next to the script, which also lets us assert the worker
//! is persistent (reused across calls).

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Instant;

use ltm_core::{Config, EmbedBackendKind, Reranker};
use ltm_rerank::{make_reranker, NoopReranker, WorkerReranker};

const FAKE_RERANK_PY: &str = r#"import json, os, sys, time

_DIR = os.path.dirname(os.path.abspath(__file__))
_CNT = os.path.join(_DIR, "spawn_count")


def _next_spawn():
    n = 0
    try:
        with open(_CNT) as f:
            n = int((f.read() or "0").strip() or "0")
    except FileNotFoundError:
        n = 0
    n += 1
    with open(_CNT, "w") as f:
        f.write(str(n))
    return n


class Reranker:
    def __init__(self):
        self.spawn = None

    def _load(self):
        # "Model load": here it just records which worker generation we are.
        self.spawn = _next_spawn()

    def rerank(self, query, texts, top_k=None):
        if query == "SLEEP":
            time.sleep(30)
        if query == "DIE_ONCE" and self.spawn == 1:
            # Simulate the worker dying mid-request on its first generation.
            os._exit(1)
        n = len(texts)
        k = n if top_k is None else min(int(top_k), n)
        order = list(range(n))[:k]
        scores = [float(n - i) for i in order]
        return order, scores
"#;

static TMP_SEQ: AtomicU32 = AtomicU32::new(0);

/// Create a fresh temp dir containing the fake `rerank.py`.
fn fake_script_dir() -> PathBuf {
    let seq = TMP_SEQ.fetch_add(1, Ordering::SeqCst);
    let mut dir = std::env::temp_dir();
    dir.push(format!(
        "ltm_rerank_test_{}_{}_{}",
        std::process::id(),
        seq,
        Instant::now().elapsed().as_nanos()
    ));
    fs::create_dir_all(&dir).expect("create temp dir");
    fs::write(dir.join("rerank.py"), FAKE_RERANK_PY).expect("write fake rerank.py");
    dir
}

/// Absolute path to a `python3` interpreter that exists on disk.
fn python3_path() -> PathBuf {
    let out = Command::new("which")
        .arg("python3")
        .output()
        .expect("run `which python3`");
    let path = String::from_utf8_lossy(&out.stdout).trim().to_string();
    assert!(!path.is_empty(), "python3 not found on PATH");
    let p = PathBuf::from(path);
    assert!(p.exists(), "resolved python3 does not exist: {p:?}");
    p
}

fn base_config() -> Config {
    Config {
        pg_host: "127.0.0.1".into(),
        pg_port: 5433,
        pg_user: "postgres".into(),
        pg_database: "brain".into(),
        embed_backend: EmbedBackendKind::Ollama,
        embed_model: "mxbai-embed-large".into(),
        ollama_url: "http://127.0.0.1:11434/api/embeddings".into(),
        embed_base_url: None,
        embed_api_key: None,
        query_prefix: "Represent this sentence for searching relevant passages: ".into(),
        rerank_python: None,
        rerank_script: None,
        rerank_candidates: 50,
        rerank_timeout_s: 10,
        low_rerank: 3.0,
        high_rerank: 6.0,
        low_cosine: 0.78,
        high_cosine: 0.855,
        http_host: "127.0.0.1".into(),
        http_port: 8322,
        http_token_file: "/tmp/does-not-matter".into(),
        public_host: None,
    }
}

fn configured(dir: &Path, timeout_s: u64) -> Config {
    let mut cfg = base_config();
    cfg.rerank_python = Some(python3_path().to_string_lossy().into_owned());
    cfg.rerank_script = Some(dir.join("rerank.py").to_string_lossy().into_owned());
    cfg.rerank_timeout_s = timeout_s;
    cfg
}

fn spawn_count(dir: &Path) -> u32 {
    fs::read_to_string(dir.join("spawn_count"))
        .ok()
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(0)
}

#[test]
fn available_is_false_when_unconfigured() {
    // Neither python nor script set.
    let cfg = base_config();
    assert!(!WorkerReranker::new(&cfg).available());
    // make_reranker → NoopReranker (also unavailable).
    assert!(!make_reranker(&cfg).available());
    assert!(!NoopReranker.available());

    // Configured but paths don't exist on disk → still unavailable.
    let mut cfg = base_config();
    cfg.rerank_python = Some("/nonexistent/python".into());
    cfg.rerank_script = Some("/nonexistent/rerank.py".into());
    assert!(!WorkerReranker::new(&cfg).available());
}

#[test]
fn available_is_true_when_python_and_script_exist() {
    let dir = fake_script_dir();
    let cfg = configured(&dir, 10);
    assert!(WorkerReranker::new(&cfg).available());
    assert!(make_reranker(&cfg).available());
}

#[test]
fn roundtrip_protocol_and_persistent_worker() {
    let dir = fake_script_dir();
    let rr = WorkerReranker::new(&configured(&dir, 30));

    let texts = vec!["alpha".to_string(), "beta".to_string(), "gamma".to_string()];

    // First request: spawns the worker, runs the JSONL round-trip.
    let (order, scores) = rr.rerank("what is beta", &texts, 3).expect("first rerank");
    assert_eq!(order, vec![0, 1, 2]);
    assert_eq!(scores, vec![3.0, 2.0, 1.0]);
    assert_eq!(spawn_count(&dir), 1, "worker should have spawned exactly once");

    // top_k clamps the returned order/scores.
    let (order2, scores2) = rr.rerank("again", &texts, 2).expect("second rerank");
    assert_eq!(order2, vec![0, 1]);
    assert_eq!(scores2, vec![3.0, 2.0]);

    // Still the SAME worker process (persistent) — no respawn.
    assert_eq!(spawn_count(&dir), 1, "worker must be reused across calls");
}

#[test]
fn times_out_when_worker_does_not_answer() {
    let dir = fake_script_dir();
    // 1s wall-clock timeout; the fake sleeps 30s on the "SLEEP" query.
    let rr = WorkerReranker::new(&configured(&dir, 1));
    let texts = vec!["x".to_string()];

    let start = Instant::now();
    let err = rr.rerank("SLEEP", &texts, 1).expect_err("should time out");
    let msg = err.to_string();
    assert!(msg.contains("did not answer"), "unexpected error: {msg}");

    // Timeout kills + retries once on a fresh worker (which also times out),
    // so two workers were spawned and the whole thing took ~2 timeouts.
    assert_eq!(spawn_count(&dir), 2, "timeout should trigger one retry");
    assert!(
        start.elapsed().as_secs() >= 2,
        "expected ~2 timeout windows, took {:?}",
        start.elapsed()
    );
}

#[test]
fn retries_once_on_worker_death() {
    let dir = fake_script_dir();
    let rr = WorkerReranker::new(&configured(&dir, 30));
    let texts = vec!["a".to_string(), "b".to_string()];

    // The first worker exits mid-request; the retry spawns a fresh worker
    // (generation 2) which answers normally.
    let (order, scores) = rr
        .rerank("DIE_ONCE", &texts, 2)
        .expect("should recover on retry");
    assert_eq!(order, vec![0, 1]);
    assert_eq!(scores, vec![2.0, 1.0]);
    assert_eq!(spawn_count(&dir), 2, "should have respawned exactly once");
}
