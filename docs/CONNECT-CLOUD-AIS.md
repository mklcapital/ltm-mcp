# Connecting cloud AIs to the LTM

How to let claude.ai, ChatGPT, and Gemini CLI query your local corpus from
anywhere, without opening a port. The pattern: `server.py --http` bound to
127.0.0.1, an outbound-only tunnel (Cloudflare Tunnel or Tailscale Funnel) in
front of it, bearer token required on every request.

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
5. Honest limitation: ChatGPT's **Deep Research** connector slot expects
   tools named exactly `search`/`fetch`; this server exposes
   `ltm_search`/`ltm_get_document`/`ltm_stats`, so it works in normal
   dev-mode chat, not as a Deep Research source (alias tools could be added).

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

## Rotate the bearer token

```bash
python3 -c "import secrets; print(secrets.token_urlsafe(32))" > ~/.ltm/http-token
chmod 600 ~/.ltm/http-token
# restart the HTTP service, then update the URL/token in each connected client
```

## Kill switch

Stop the tunnel daemon and the public URL dies instantly; the server itself
was never reachable from outside. Belt-and-braces: stop the HTTP service too.
