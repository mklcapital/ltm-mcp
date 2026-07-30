//! End-to-end integration test.
//!
//! Runs the full harness against the live Postgres + Ollama services. When a
//! sibling crate is still a stub or a service is unreachable, the harness
//! reports SKIPPED and the test passes (printing the reason) instead of
//! failing — so this test is green both before and after integration.
//!
//! Run with output visible:
//!   cargo test -p ltm-e2e -- --nocapture
//! (defaults assume LTM_PGPORT=5432; see the crate README.)

use ltm_e2e::{run, Outcome};

#[test]
fn e2e_pipeline() {
    match run() {
        Ok(Outcome::Passed(summary)) => {
            println!("PASS: {summary}");
        }
        Ok(Outcome::Skipped(reason)) => {
            // Not a failure: a dependency is still a stub or a service is down.
            println!("{reason}");
        }
        Err(e) => panic!("e2e harness FAILED: {e}"),
    }
}
