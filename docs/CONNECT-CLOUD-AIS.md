# Connecting other AIs to the LTM

How to let claude.ai, ChatGPT (incl. Deep Research), Codex, Gemini, Cursor,
LM Studio, and OpenAI-compatible APIs (OpenRouter, ...) query your local
corpus from anywhere, without opening a port. The pattern: the LTM server in
HTTP mode (`ltm-mcp --http`, the Rust binary; `server.py --http` behaves the
same minus the `search`/`fetch` aliases) bound to 127.0.0.1, an outbound-only
tunnel (Cloudflare Tunnel or Tailscale Funnel) in front of it, bearer token
required on every request.

| Client | Transport | Auth form | Section |
|---|---|---|---|
| claude.ai / Claude Desktop / mobile | streamable HTTP | token-in-URL | [claude.ai](#claudeai-web--desktop--mobile) |
| Claude Code | stdio (or HTTP) | — / header | README quickstart |
| ChatGPT + Deep Research | streamable HTTP | token-in-URL | [ChatGPT](#chatgpt-apps--connectors) |
| Codex (CLI + IDE extension) | stdio or HTTP | — / bearer env var | [Codex](#openai-codex-cli--ide) |
| Gemini CLI / Gemini API | HTTP | header | [Gemini](#gemini) |
| Cursor | stdio or HTTP | — / header | [Cursor](#cursor) |
| LM Studio | stdio or HTTP | — / header | [LM Studio](#lm-studio) |
| OpenRouter, Ollama, Groq, any OpenAI-compatible API | via `ltm-bridge` | — | [OpenAI-compatible APIs](#openrouter-and-other-openai-compatible-apis) |

Throughout, `ltm.<domain>` is your tunnel hostname (set `LTM_PUBLIC_HOST` to
it) and `<token>` is the contents of your bearer-token file
(`~/.ltm/http-token` by default).

## Two ways a client can authenticate

1. **Header** (preferred when the UI has a header/token field):
   `URL = https://ltm.<domain>/mcp` with `Authorization: Bearer <token>`
2. **Token-in-URL** (for connector UIs with *no* header field — claude.ai,
   ChatGPT): `URL = https://ltm.<domain>/t/<token>/mcp`
   The server strips the prefix and treats it exactly like a bearer header;
   comparisons are constant-time and access logging is disabled so the token
   never lands in a log. Treat the whole URL as a secret.

## claude.ai (web + desktop + mobile)

1. claude.ai → **Settings → Connectors** (Pro/Max/Team/Enterprise;
   org admins: Admin settings → Connectors).
2. **Add custom connector**.
3. Name: `LTM`. URL: `https://ltm.<domain>/t/<token>/mcp`
   (claude.ai's custom-connector form has OAuth fields but no plain
   bearer-header field — the token-in-URL form is the way).
4. Add → enable it in a chat via the search-and-tools menu.
5. Test: ask "use ltm_stats" — expect your corpus counts.

## ChatGPT (Apps & Connectors)

1. ChatGPT → **Settings → Apps & Connectors → Advanced settings** → enable
   **Developer mode** (needed for custom MCP connectors).
2. Back in Apps & Connectors → **Create** (custom connector).
3. Name `LTM`, MCP server URL: `https://ltm.<domain>/t/<token>/mcp`,
   Authentication: **No authentication** (the token rides in the URL).
4. Trust prompt → create. Use it in chat with Developer mode on.
5. **Deep Research** works too: OpenAI's connector contract requires tools
   named exactly `search`/`fetch`, and the Rust server registers those
   aliases automatically in HTTP mode (thin wrappers over
   `ltm_search`/`ltm_get_document`; `LTM_COMPAT_TOOLS=on|off|auto`
   overrides). Pick the LTM connector as a source when starting a Deep
   Research run.

## OpenAI Codex (CLI + IDE)

Codex CLI and the Codex IDE extension share `~/.codex/config.toml`
(project-scoped: `.codex/config.toml`).

Same machine as the LTM (stdio — simplest):

```toml
[mcp_servers.ltm]
command = "/path/to/ltm-mcp"        # cargo build --release → target/release/ltm-mcp
```

Remote over the tunnel (streamable HTTP; keeps the token out of the config
file via an env var):

```bash
export LTM_TOKEN=<token>            # put in your shell profile
codex mcp add ltm --url "https://ltm.<domain>/mcp" \
  --bearer-token-env-var LTM_TOKEN
```

or equivalently in `config.toml`:

```toml
[mcp_servers.ltm]
url = "https://ltm.<domain>/mcp"
bearer_token_env_var = "LTM_TOKEN"
```

Verify with `codex mcp list`, then ask Codex to "use ltm_stats".

## Cursor

`~/.cursor/mcp.json` (all projects) or `.cursor/mcp.json` (per project) —
Cursor supports both stdio and remote HTTP with headers:

```json
{
  "mcpServers": {
    "ltm": {
      "url": "https://ltm.<domain>/mcp",
      "headers": { "Authorization": "Bearer <token>" }
    }
  }
}
```

(Same machine instead: `{ "command": "/path/to/ltm-mcp" }`.) Then Cursor
Settings → MCP → confirm the `ltm_*` tools show up; the agent uses them
automatically.

## LM Studio

LM Studio's `mcp.json` (chat sidebar → **Program** tab → Install → **Edit
mcp.json**) follows Cursor's notation exactly — both forms above work
verbatim: `command` when LM Studio runs on the same machine as the LTM, or
`url` + `headers` to reach it over the tunnel from another machine.

## Gemini

As of mid-2026 the **Gemini consumer app has no custom MCP connector
support**. MCP works in Google's developer surfaces:

- **Gemini CLI**: `~/.gemini/settings.json` →
  ```json
  {
    "mcpServers": {
      "ltm": {
        "httpUrl": "https://ltm.<domain>/mcp",
        "headers": { "Authorization": "Bearer <token>" }
      }
    }
  }
  ```
  (the CLI supports headers, so use the cleaner header form.)

- **Gemini API** (`google-genai` Python SDK): the SDK accepts an MCP
  `ClientSession` directly as a tool and drives the calls itself:

  ```python
  import asyncio
  from google import genai
  from mcp import ClientSession
  from mcp.client.streamable_http import streamablehttp_client

  async def main():
      client = genai.Client()  # needs GEMINI_API_KEY
      async with streamablehttp_client(
              "https://ltm.<domain>/mcp",
              headers={"Authorization": "Bearer <token>"}) as (r, w, _):
          async with ClientSession(r, w) as session:
              await session.initialize()
              resp = await client.aio.models.generate_content(
                  model="gemini-2.5-pro",
                  contents="Which lease covers unit 4B? Search the LTM.",
                  config=genai.types.GenerateContentConfig(tools=[session]))
              print(resp.text)

  asyncio.run(main())
  ```

## OpenRouter and other OpenAI-compatible APIs

OpenRouter, Ollama, Groq, vLLM, and the OpenAI API itself don't execute MCP
tools server-side — the model returns `tool_calls` and *your* code must run
them. The workspace ships that code as a Rust binary:
[`crates/ltm-bridge`](../crates/ltm-bridge). It exposes the LTM tools to any
`/v1/chat/completions` endpoint and loops model → tool → model.

```bash
export OPENAI_BASE_URL=https://openrouter.ai/api/v1   # or LM Studio/Ollama/...
export OPENAI_API_KEY=sk-or-...
export OPENAI_MODEL=openai/gpt-4o-mini                # any tool-calling model
cargo run --release -p ltm-bridge -- "which lease covers unit 4B?"
# or with no question for an interactive chat loop
```

Alternatively, apps that sit on top of OpenRouter and have their own MCP
support (Cline, Open WebUI, LibreChat, ...) can connect directly with the
same `url` + `Authorization` header shape as the Cursor config above.

## Rotate the bearer token

```bash
python3 -c "import secrets; print(secrets.token_urlsafe(32))" > ~/.ltm/http-token
chmod 600 ~/.ltm/http-token
# restart the HTTP service, then update the URL/token in each connected client
```

## Kill switch

Stop the tunnel daemon and the public URL dies instantly; the server itself
was never reachable from outside. Belt-and-braces: stop the HTTP service too.
