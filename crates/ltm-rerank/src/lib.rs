//! ltm-rerank — optional cross-encoder reranker. Implements
//! [`ltm_core::Reranker`].
//!
//! server.py runs a persistent Python worker subprocess (loads
//! mxbai-rerank-base-v2 once, serves JSONL requests over stdin/stdout). The
//! port keeps rerank OPTIONAL and configurable.
//!
//! - [`NoopReranker`] is the default when no reranker is configured
//!   (`available() == false`) → tools fall back to cosine confidence.
//! - [`WorkerReranker`] spawns the persistent Python worker per
//!   `rerank_python`/`rerank_script`, speaks the same JSONL protocol as
//!   server.py's `_RERANK_WORKER_SRC`, enforces a wall-clock timeout, and
//!   restarts-on-death with one retry.

use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::sync::Mutex;
use std::thread;
use std::time::Duration;

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

/// The Python worker program, mirroring server.py's `_RERANK_WORKER_SRC`.
///
/// The literal `{dir}` is replaced with a JSON-encoded (== valid Python string
/// literal) absolute path so the worker can `import rerank.Reranker` from the
/// directory of `rerank_script`. All other `{`/`}` are literal Python dict
/// syntax (this is a plain `str::replace`, not `format!`).
const WORKER_SRC_TEMPLATE: &str = r#"import json, sys
sys.path.insert(0, {dir})
from rerank import Reranker
rr = Reranker()
rr._load()
print(json.dumps({"ready": True}), flush=True)
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        req = json.loads(line)
        order, scores = rr.rerank(req["query"], req["texts"], req.get("top_k"))
        print(json.dumps({"order": order, "scores": scores}), flush=True)
    except Exception as exc:
        print(json.dumps({"error": str(exc)}), flush=True)
"#;

/// A live worker subprocess plus the plumbing to talk to it. Reads happen on a
/// dedicated thread that forwards each JSONL line over a channel; the request
/// path uses `recv_timeout` to get the select-style wall-clock timeout that
/// server.py's `_rr_read_line` gets from `select.select`.
struct Worker {
    child: Child,
    stdin: ChildStdin,
    rx: Receiver<String>,
}

impl Worker {
    /// Spawn `rerank_python -u -c <SRC>`, wait for the `{"ready": true}`
    /// handshake, and return the live worker.
    fn spawn(python: &str, script: &str, timeout: Duration) -> Result<Worker> {
        let script_abs = Path::new(script)
            .canonicalize()
            .unwrap_or_else(|_| Path::new(script).to_path_buf());
        let rerank_dir = script_abs
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| Path::new(".").to_path_buf());
        // JSON-encode the directory: a valid Python string literal.
        let dir_literal = serde_json::to_string(&rerank_dir.to_string_lossy().into_owned())
            .map_err(|e| Error::Rerank(format!("failed to encode rerank dir: {e}")))?;
        let src = WORKER_SRC_TEMPLATE.replace("{dir}", &dir_literal);

        let mut child = Command::new(python)
            .arg("-u")
            .arg("-c")
            .arg(src)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .env("TQDM_DISABLE", "1")
            .spawn()
            .map_err(|e| Error::Rerank(format!("failed to spawn rerank worker: {e}")))?;

        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| Error::Rerank("rerank worker missing stdin".into()))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| Error::Rerank("rerank worker missing stdout".into()))?;

        let (tx, rx) = mpsc::channel::<String>();
        thread::spawn(move || {
            let mut reader = BufReader::new(stdout);
            loop {
                let mut line = String::new();
                match reader.read_line(&mut line) {
                    Ok(0) => break, // EOF: worker exited / stdout closed
                    Ok(_) => {
                        if tx.send(line).is_err() {
                            break; // receiver gone
                        }
                    }
                    Err(_) => break,
                }
            }
        });

        let worker = Worker { child, stdin, rx };

        let hello = worker.read_response(timeout)?;
        let ready = serde_json::from_str::<serde_json::Value>(&hello)
            .ok()
            .and_then(|v| v.get("ready").and_then(|r| r.as_bool()))
            .unwrap_or(false);
        if !ready {
            return Err(Error::Rerank(format!(
                "rerank worker failed to start: {}",
                hello.trim()
            )));
        }
        Ok(worker)
    }

    /// Block up to `timeout` for the next JSONL line from the worker.
    fn read_response(&self, timeout: Duration) -> Result<String> {
        match self.rx.recv_timeout(timeout) {
            Ok(line) => Ok(line),
            Err(RecvTimeoutError::Timeout) => Err(Error::Rerank(format!(
                "reranker did not answer within {}s",
                timeout.as_secs()
            ))),
            Err(RecvTimeoutError::Disconnected) => {
                Err(Error::Rerank("reranker worker exited unexpectedly".into()))
            }
        }
    }

    /// Send one JSONL request and read one JSONL response (with timeout).
    fn request(&mut self, payload: &str, timeout: Duration) -> Result<serde_json::Value> {
        self.stdin
            .write_all(payload.as_bytes())
            .and_then(|_| self.stdin.write_all(b"\n"))
            .and_then(|_| self.stdin.flush())
            .map_err(|e| Error::Rerank(format!("failed to write to rerank worker: {e}")))?;
        let line = self.read_response(timeout)?;
        serde_json::from_str(&line)
            .map_err(|e| Error::Rerank(format!("invalid rerank response {:?}: {e}", line.trim())))
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// Persistent cross-encoder reranker backed by a Python worker subprocess.
///
/// The child handle is guarded by a `Mutex`, since `Reranker` is `Sync` and a
/// shared `&self` may be called concurrently; requests are therefore
/// serialized onto the single worker (matching server.py's global lock).
pub struct WorkerReranker {
    cfg: Config,
    worker: Mutex<Option<Worker>>,
}

impl WorkerReranker {
    pub fn new(cfg: &Config) -> Self {
        WorkerReranker {
            cfg: cfg.clone(),
            worker: Mutex::new(None),
        }
    }

    fn timeout(&self) -> Duration {
        Duration::from_secs(self.cfg.rerank_timeout_s)
    }
}

impl Reranker for WorkerReranker {
    fn rerank(&self, query: &str, texts: &[String], top_k: usize) -> Result<(Vec<usize>, Vec<f64>)> {
        let (python, script) = match (&self.cfg.rerank_python, &self.cfg.rerank_script) {
            (Some(p), Some(s)) => (p.clone(), s.clone()),
            _ => return Err(Error::Rerank("reranker not configured".into())),
        };

        let payload = serde_json::json!({
            "query": query,
            "texts": texts,
            "top_k": top_k,
        })
        .to_string();
        let timeout = self.timeout();

        let mut guard = self
            .worker
            .lock()
            .map_err(|_| Error::Rerank("rerank worker lock poisoned".into()))?;

        // Retry ONCE on a fresh worker if the current one is dead/times out,
        // exactly like server.py's `run_rerank` loop over attempts (1, 2).
        let mut last_err: Option<Error> = None;
        for attempt in 1..=2 {
            let attempt_result = (|| -> Result<serde_json::Value> {
                if guard.is_none() {
                    *guard = Some(Worker::spawn(&python, &script, timeout)?);
                }
                let worker = guard
                    .as_mut()
                    .expect("worker present after spawn");
                worker.request(&payload, timeout)
            })();

            match attempt_result {
                Ok(resp) => {
                    if let Some(err) = resp.get("error").and_then(|e| e.as_str()) {
                        // A worker-side rerank error is a normal response, not a
                        // dead worker: surface it without killing/retrying.
                        return Err(Error::Rerank(err.to_string()));
                    }
                    return parse_ok(&resp);
                }
                Err(err) => {
                    // Drop (kill+reap) the failed worker so the retry — and any
                    // later call — starts fresh.
                    *guard = None;
                    if attempt == 2 {
                        return Err(err);
                    }
                    last_err = Some(err);
                }
            }
        }
        Err(last_err.unwrap_or_else(|| Error::Rerank("rerank failed".into())))
    }

    fn available(&self) -> bool {
        match (&self.cfg.rerank_python, &self.cfg.rerank_script) {
            (Some(python), Some(script)) => {
                Path::new(python).exists() && Path::new(script).exists()
            }
            _ => false,
        }
    }
}

/// Parse a `{"order": [...], "scores": [...]}` response into typed vectors.
fn parse_ok(resp: &serde_json::Value) -> Result<(Vec<usize>, Vec<f64>)> {
    let order = resp
        .get("order")
        .and_then(|v| v.as_array())
        .ok_or_else(|| Error::Rerank("rerank response missing 'order'".into()))?
        .iter()
        .map(|v| {
            v.as_u64()
                .map(|n| n as usize)
                .ok_or_else(|| Error::Rerank("rerank 'order' had a non-index value".into()))
        })
        .collect::<Result<Vec<usize>>>()?;
    let scores = resp
        .get("scores")
        .and_then(|v| v.as_array())
        .ok_or_else(|| Error::Rerank("rerank response missing 'scores'".into()))?
        .iter()
        .map(|v| {
            v.as_f64()
                .ok_or_else(|| Error::Rerank("rerank 'scores' had a non-numeric value".into()))
        })
        .collect::<Result<Vec<f64>>>()?;
    Ok((order, scores))
}

/// Choose a reranker from config: the worker when configured, else no-op.
pub fn make_reranker(cfg: &Config) -> Box<dyn Reranker> {
    if cfg.rerank_python.is_some() && cfg.rerank_script.is_some() {
        Box::new(WorkerReranker::new(cfg))
    } else {
        Box::new(NoopReranker)
    }
}
