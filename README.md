# Jouzetsu

Self-hosted web interface for [LM Studio](https://lmstudio.ai/), built with
FastHTML. It streams chats with local models and stores conversations and
character profiles as JSON files you control.

## Requirements

- Python 3.14 or newer
- [uv](https://docs.astral.sh/uv/)
- LM Studio with its local server running (default: `http://localhost:1234`)
- Linux only

## Run

```sh
uv sync
uv run jouzetsu
```

Open <http://127.0.0.1:8080/>. Jouzetsu listens on port 8080 by default, so a
LAN device can use `http://<host-lan-ip>:8080/` after approval.

On first run, it writes `config.json` and `data/` in the source checkout. For
an installed or portable copy, choose an explicit application home:

```sh
jouzetsu --home /path/to/Jouzetsu
# or: JOUZETSU_HOME=/path/to/Jouzetsu jouzetsu
```

## Included

- Streaming chats with per-chat model, sampling, prompt, reasoning, and
  British-spelling settings; messages can be edited, regenerated, continued,
  forked, merged, or undone.
- Downloaded-model discovery and loading through LM Studio, including an
  optional idle unload time-to-live.
- Character profiles and JSON preset packs; start immutable single-character or
  multi-character chats in Roleplay, Assistant, Story, or Custom prompt modes.
- Optional post-reply continuity review, proposed rewrites requiring user
  approval, host statistics, local-only assets, and SSE updates.
- Default-private browser access with localhost device approval controls.

## Configuration and data

Use the in-app settings for normal changes. `config.json` is also editable;
restart Jouzetsu after editing it directly. Its sections are `server`,
`generation`, `ui`, `theme`, `logging`, `host_stats`, and `access`.
It is a complete, strict, versioned document (`"version": 1`): missing,
unknown, or invalid settings prevent startup rather than being silently changed.
Built-in British-spelling replacements and the theme palette are packaged
separately. `generation.british_spelling_replacements` and `theme` are written
only when you customise them.

Global Settings separates chat-wide Behaviour from Appearance. Model selection,
aliases, and auto-unload live in the Models dialog's Defaults tab. The
Appearance editor has four independent colourways: App for neutral surfaces,
text, controls, and key interface visuals; User for your messages and composer;
Assistant for replies and generation highlights; and System for messages and
notifications. The built-in palette uses black-and-white App surfaces with a
purple App accent, red User, blue Assistant, and green System accents.
Each has one main colour with automatic border, background, and hover shades.
App also selects light/dark appearance. Advanced controls use compact native
colour pickers with an Auto toggle to restore generated shading. Key Visual
sits beside App's main colour; App's Advanced section exposes the uniform
message-action colour, icon artwork colours, and the activity-meter gradient.
Message actions default to the uniform App colour, but can instead use
independently configurable semantic colours for delete, regenerate, merge,
edit, and continue/resend.

Existing flat palettes load automatically into the new colourways. User and
Assistant accents and custom surface/shade overrides are retained; App and System
receive independent default accents. Earlier message-action icon preferences
and muted action colours migrate to the semantic action setting and palette on
the next save. The new theme shape is then written normally.

Useful environment overrides:

- `JOUZETSU_HOME` — application home
- `JOUZETSU_HOST` / `JOUZETSU_PORT` — listen address and port
- `JOUZETSU_LOG_LEVEL` — `DEBUG`, `INFO` (default), or `WARNING`

The application home contains `config.json`, `data/chats/`,
`data/characters/`, `data/character-presets/`, and `data/logs/`. Deleted
chats move to `data/chats/trash/`; legacy `data/chats.json` is migrated on
startup and retained as a timestamped recovery copy. Add custom empty-chat
messages to `data/empty-chat-messages.txt`, one line per message.

Add name suggestions to `data/character-names.json`. The file supplements the
built-in pools by default, accepts an empty pool when you only want to add one
name part, and uses this shape:

```json
{
  "schema_version": 1,
  "given_names": ["Ayla"],
  "family_names": ["Khan"]
}
```

Set `"disable_vanilla": true` to use only the supplied names instead. In this
mode, both pools must be non-empty so each randomise button remains usable:

```json
{
  "schema_version": 1,
  "disable_vanilla": true,
  "given_names": ["Ayla"],
  "family_names": ["Khan"]
}
```

Malformed custom name files are ignored with a visible editor warning and a log
entry, so the built-in randomiser remains available.

Custom character preset packs are JSON files in `data/character-presets/`.
They are data only and are never executed; malformed packs are skipped and
reported in the character editor and logs.

## Network safety

Jouzetsu is a single-user shared workspace, not a multi-user system. Its device approval is a convenience gate for trusted browsers. The
built-in server is HTTP-only: keep it on localhost, a trusted LAN, or a
private network such as Tailscale. Use an authenticated TLS reverse proxy if
you need wider access; never expose the port directly to the public internet.

## Development

```sh
uv sync --group dev
uv run --group dev playwright install chromium
uv run --group dev pytest -q
uv run --group dev ruff check .
uv run --group dev basedpyright
```

The focused Playwright tests run with the normal pytest suite and cover the
browser-only state transitions that the fast HTTP and unit tests cannot see.

`uv run python -m scripts.smoke_server` starts a temporary smoke-test server.
`uv run python -m scripts.e2e_stream` exercises streaming against the LM
Studio configuration in `config.json`.
