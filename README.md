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
- Character profiles and JSON preset packs; starting a character chat records
  that profile's compiled prompt in the new conversation.
- Optional post-reply continuity review, proposed rewrites requiring user
  approval, host statistics, local-only assets, and SSE updates.
- Default-private browser access with localhost device approval controls.

## Configuration and data

Use the in-app settings for normal changes. `config.json` is also editable;
restart Jouzetsu after editing it directly. Its sections are `server`,
`generation`, `ui`, `theme`, `logging`, `host_stats`, and `access`.

Useful environment overrides:

- `JOUZETSU_HOME` — application home
- `JOUZETSU_HOST` / `JOUZETSU_PORT` — listen address and port
- `JOUZETSU_LOG_LEVEL` — `DEBUG`, `INFO` (default), or `WARNING`

The application home contains `config.json`, `data/chats/`,
`data/characters/`, `data/character-presets/`, and `data/logs/`. Deleted
chats move to `data/chats/trash/`; legacy `data/chats.json` is migrated on
startup and retained as a timestamped recovery copy. Add custom empty-chat
messages to `data/empty-chat-messages.txt`, one line per message.

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
uv run --group dev pytest -q
uv run --group dev ruff check .
uv run --group dev basedpyright
```

`uv run python -m scripts.smoke_server` starts a temporary smoke-test server.
`uv run python -m scripts.e2e_stream` exercises streaming against the LM
Studio configuration in `config.json`.
