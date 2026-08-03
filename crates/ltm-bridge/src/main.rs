//! ltm-bridge — drive the LTM MCP server from any OpenAI-compatible chat API.
//!
//! OpenRouter, LM Studio's local server, Ollama, Groq, vLLM, and the OpenAI
//! API all speak the same `/v1/chat/completions` dialect — and none of them
//! execute MCP tools for you: the model returns `tool_calls` and it is the
//! client's job to run them. This binary is that client. It connects to the
//! LTM server over MCP streamable HTTP, advertises its tools (`ltm_search` /
//! `ltm_get_document` / `ltm_stats`, plus the `search`/`fetch` aliases in
//! HTTP mode) as OpenAI function definitions, and loops:
//! model -> tool call -> LTM -> model, until the model answers.
//!
//! Usage:
//! ```text
//! # one-shot
//! export OPENAI_BASE_URL=https://openrouter.ai/api/v1
//! export OPENAI_API_KEY=sk-or-...
//! export OPENAI_MODEL=openai/gpt-4o-mini        # any tool-calling model
//! ltm-bridge "which lease covers unit 4B?"
//!
//! # interactive chat
//! ltm-bridge
//! ```
//!
//! Endpoints that work identically (set `OPENAI_BASE_URL` / `OPENAI_MODEL`):
//! ```text
//! OpenRouter   https://openrouter.ai/api/v1      any tool-capable model
//! LM Studio    http://127.0.0.1:1234/v1          local models (LM Studio
//!                                                also has native in-app MCP)
//! Ollama       http://127.0.0.1:11434/v1         local models
//! Groq         https://api.groq.com/openai/v1
//! ```
//!
//! LTM side (env, all optional):
//! ```text
//! LTM_MCP_URL           default http://127.0.0.1:8322/mcp
//! LTM_HTTP_TOKEN        bearer token value, or
//! LTM_HTTP_TOKEN_FILE   default ~/.ltm/http-token
//! ```

use rmcp::model::{CallToolRequestParams, CallToolResult, Tool};
use rmcp::transport::streamable_http_client::{
    StreamableHttpClientTransport, StreamableHttpClientTransportConfig,
};
use rmcp::ServiceExt;
use serde_json::{json, Value};
use std::io::{BufRead, Write as _};

/// Tool-call round-trips allowed per user message.
const MAX_TOOL_ROUNDS: usize = 8;

const SYSTEM: &str = "You can search the user's private Long-Term Memory document corpus \
with the provided tools. Ground answers in tool results and cite document paths. The \
search confidence field is an honest gate: when it says LOW, state that the answer does \
not appear to be in the corpus instead of guessing from weak hits.";

struct LlmApi {
    base: String,
    key: String,
    model: String,
    http: reqwest::Client,
}

impl LlmApi {
    /// One `/chat/completions` call; returns the assistant message object.
    async fn chat(
        &self,
        messages: &[Value],
        tools: &Value,
    ) -> Result<Value, Box<dyn std::error::Error>> {
        let resp = self
            .http
            .post(format!("{}/chat/completions", self.base))
            .bearer_auth(&self.key)
            .json(&json!({ "model": self.model, "messages": messages, "tools": tools }))
            .send()
            .await?;
        let status = resp.status();
        let body = resp.text().await?;
        if !status.is_success() {
            return Err(format!(
                "{} returned HTTP {status}: {}",
                self.base,
                &body[..body.len().min(2000)]
            )
            .into());
        }
        let v: Value = serde_json::from_str(&body)?;
        Ok(v["choices"][0]["message"].clone())
    }
}

fn bearer_token() -> Option<String> {
    if let Ok(tok) = std::env::var("LTM_HTTP_TOKEN") {
        let tok = tok.trim().to_string();
        if !tok.is_empty() {
            return Some(tok);
        }
    }
    let path = std::env::var("LTM_HTTP_TOKEN_FILE").unwrap_or_else(|_| {
        format!(
            "{}/.ltm/http-token",
            std::env::var("HOME").unwrap_or_default()
        )
    });
    std::fs::read_to_string(path)
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

/// MCP tool list -> OpenAI `tools` array (function definitions).
fn to_openai_tools(tools: &[Tool]) -> Value {
    Value::Array(
        tools
            .iter()
            .map(|t| {
                json!({
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description.as_deref().unwrap_or(""),
                        "parameters": Value::Object((*t.input_schema).clone()),
                    }
                })
            })
            .collect(),
    )
}

/// Flatten a tool result into the string an OpenAI `role:"tool"` message
/// carries: prefer the structured JSON, fall back to joined text blocks.
fn result_text(r: &CallToolResult) -> String {
    let body = match &r.structured_content {
        Some(v) => v.to_string(),
        None => {
            let texts: Vec<&str> = r
                .content
                .iter()
                .filter_map(|c| c.as_text().map(|t| t.text.as_str()))
                .collect();
            if texts.is_empty() {
                "(empty result)".to_string()
            } else {
                texts.join("\n")
            }
        }
    };
    if r.is_error == Some(true) {
        format!("TOOL ERROR: {body}")
    } else {
        body
    }
}

async fn run_turn(
    api: &LlmApi,
    client: &rmcp::service::RunningService<rmcp::RoleClient, ()>,
    tools: &Value,
    messages: &mut Vec<Value>,
) -> Result<String, Box<dyn std::error::Error>> {
    for _ in 0..MAX_TOOL_ROUNDS {
        let msg = api.chat(messages, tools).await?;
        messages.push(msg.clone());
        let calls = msg
            .get("tool_calls")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        if calls.is_empty() {
            return Ok(msg
                .get("content")
                .and_then(Value::as_str)
                .unwrap_or("(no content)")
                .to_string());
        }
        for call in &calls {
            let name = call["function"]["name"].as_str().unwrap_or_default();
            let raw_args = call["function"]["arguments"].as_str().unwrap_or("{}");
            let args: Value = serde_json::from_str(raw_args).unwrap_or_else(|_| json!({}));
            eprintln!(
                "  [tool] {name}({})",
                &raw_args[..raw_args.len().min(200)]
            );
            let mut params = CallToolRequestParams::new(name.to_string());
            if let Some(obj) = args.as_object() {
                params = params.with_arguments(obj.clone());
            }
            let result = client.call_tool(params).await;
            let content = match result {
                Ok(r) => result_text(&r),
                Err(e) => format!("TOOL ERROR: {e}"),
            };
            messages.push(json!({
                "role": "tool",
                "tool_call_id": call["id"],
                "content": content,
            }));
        }
    }
    Ok(format!("(stopped after {MAX_TOOL_ROUNDS} tool rounds)"))
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let base = std::env::var("OPENAI_BASE_URL")
        .unwrap_or_else(|_| "https://openrouter.ai/api/v1".to_string())
        .trim_end_matches('/')
        .to_string();
    let key = std::env::var("OPENAI_API_KEY").unwrap_or_default();
    let model = std::env::var("OPENAI_MODEL").unwrap_or_default();
    if key.is_empty() || model.is_empty() {
        eprintln!(
            "set OPENAI_API_KEY and OPENAI_MODEL (and OPENAI_BASE_URL unless using \
             OpenRouter), e.g.\n  OPENAI_MODEL=openai/gpt-4o-mini          # OpenRouter\n  \
             OPENAI_MODEL=qwen2.5-7b-instruct         # LM Studio local"
        );
        std::process::exit(2);
    }
    let mcp_url =
        std::env::var("LTM_MCP_URL").unwrap_or_else(|_| "http://127.0.0.1:8322/mcp".to_string());

    let mut config = StreamableHttpClientTransportConfig::with_uri(mcp_url.clone());
    if let Some(tok) = bearer_token() {
        config = config.auth_header(tok); // sent as `Authorization: Bearer <tok>`
    }
    let transport = StreamableHttpClientTransport::from_config(config);
    let client = ().serve(transport).await?;

    let tool_list = client.list_all_tools().await?;
    let tools = to_openai_tools(&tool_list);
    eprintln!(
        "connected to {mcp_url} — tools: {}",
        tool_list
            .iter()
            .map(|t| t.name.as_ref())
            .collect::<Vec<_>>()
            .join(", ")
    );

    let api = LlmApi {
        base,
        key,
        model,
        http: reqwest::Client::new(),
    };
    let mut messages = vec![json!({ "role": "system", "content": SYSTEM })];

    let question = std::env::args().skip(1).collect::<Vec<_>>().join(" ");
    if !question.trim().is_empty() {
        messages.push(json!({ "role": "user", "content": question.trim() }));
        println!("{}", run_turn(&api, &client, &tools, &mut messages).await?);
        let _ = client.cancel().await;
        return Ok(());
    }

    let stdin = std::io::stdin();
    loop {
        eprint!("you> ");
        std::io::stderr().flush().ok();
        let mut line = String::new();
        if stdin.lock().read_line(&mut line)? == 0 {
            break; // EOF
        }
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        messages.push(json!({ "role": "user", "content": line }));
        println!("{}", run_turn(&api, &client, &tools, &mut messages).await?);
    }
    let _ = client.cancel().await;
    Ok(())
}
