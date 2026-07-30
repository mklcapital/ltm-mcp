//! `ltm-e2e` binary.
//!
//! Usage:
//!   ltm-e2e           run the full end-to-end harness (seed + assertions)
//!   ltm-e2e e2e       same as above
//!   ltm-e2e seed      only (re)seed the corpus, print the counts
//!
//! Exit codes: 0 on PASS or SKIP, 1 on FAIL. SKIP (a still-stubbed dependency
//! or an unreachable service) is intentionally not an error so the harness can
//! run in CI before integration is complete.

use std::process::ExitCode;

use ltm_core::Config;
use ltm_e2e::seed::{self, SeedError};
use ltm_e2e::{run, Outcome};

fn main() -> ExitCode {
    let mode = std::env::args().nth(1).unwrap_or_else(|| "e2e".to_string());

    match mode.as_str() {
        "seed" => run_seed(),
        "e2e" => run_harness(),
        other => {
            eprintln!("ltm-e2e: unknown mode {other:?} (expected `e2e` or `seed`)");
            ExitCode::FAILURE
        }
    }
}

fn run_seed() -> ExitCode {
    let cfg = Config::from_env();
    match seed::seed(&cfg) {
        Ok(r) => {
            println!(
                "SEEDED: documents={} chunks={} embeddings={}",
                r.documents, r.chunks, r.embeddings
            );
            ExitCode::SUCCESS
        }
        Err(SeedError::ServiceDown(m)) => {
            println!("SKIPPED (service down): {m}");
            ExitCode::SUCCESS
        }
        Err(SeedError::Other(m)) => {
            eprintln!("FAIL: seed failed: {m}");
            ExitCode::FAILURE
        }
    }
}

fn run_harness() -> ExitCode {
    match run() {
        Ok(Outcome::Passed(summary)) => {
            println!("PASS: {summary}");
            ExitCode::SUCCESS
        }
        Ok(Outcome::Skipped(reason)) => {
            println!("{reason}");
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("FAIL: {e}");
            ExitCode::FAILURE
        }
    }
}
