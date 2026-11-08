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
- Vision chats with multiple JPEG, PNG, or WebP images: upload, paste, or drop
  images into the composer, preview them, and send them with or without text.
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

Vision models are marked in model selectors. Images stay with their original
user message in later requests, regeneration, continuations, and continuity
reviews. A conversation containing images requires a vision-capable model;
switching to a text-only model blocks generation until a compatible model is
selected. Editing a caption preserves its images.

Original image files and small previews live in `data/images/`; the original
resolution is preserved. Draft attachments survive reloads, and forks and undo
retain their references. Image files are kept when references are removed so
other chats and recoverable trash remain intact. Initial limits allow 32 images
per message, 64 MiB per image, 512 MiB per upload, and 100 million pixels per
image. The upload policy is centralised in `ImageLimits` for future settings.
Chat documents use schema version 2; historical chats load without attachments
and upgrade when saved.
WebP originals are sent to LM Studio as lossless PNG images to work around the
0.4.26 compatibility endpoint rejecting WebP data URLs.

Global Settings separates chat-wide Behaviour from Appearance. The Models dialog
has Loaded instances, per-model Aliases, and global Defaults for model selection
and auto-unload. The Appearance editor has four independent colourways:
App for neutral surfaces,
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

### Persistence boundaries and recovery

Committed chat edits, new/forked/character-based chats, chat selection, deletion,
settings, and character changes save before publishing authoritative state or
notifying browsers. Existing chat/message and shared configuration references
are retained when saved candidates are applied. Generation acceptance saves its
initial transcript before starting an inference stream. Generation output is
transient while streaming and is saved on completion or cancellation.

Draft text and draft attachments deliberately use different semantics: they
change the live draft immediately and use the existing coalesced persistence
path. Attachment operations await that path; typing is debounced. A failed
draft save retains the accepted draft for a later flush or orderly shutdown.
It cannot replay a rejected committed edit or recreate a successfully deleted
chat. Recent drafts and in-progress generation can still be lost in a crash.

Runtime JSON serialisation, writes, moves, and synchronisation use the existing
worker pool. State-changing commands are serialized from preparation through
publication because the current application still has a global active chat and
shared configuration. Nested commands share that boundary. A separate writer
lock orders deferred drafts and committed/configuration writes. Readers, SSE,
and unrelated asynchronous services can run while storage is busy. Request
cancellation waits for the accepted command to finish before releasing these
boundaries; it cannot stop a filesystem operation already running in a thread.
Image preflight runs outside the command lock and revalidates its conversation
and model context after acquiring the lock, so a backend check cannot block edits.
Startup loading and initial empty-store creation remain synchronous, before
the state is exposed to requests. Snapshots are copied on the event loop so a
worker never serialises a mutable live transcript.

Separate chat and configuration files are **not a transaction**:

- Creation writes the chat, then its active selection in `config.json`, then
  publishes both in memory. If the second write fails, the call fails without
  publication. The extra valid chat file remains recoverable and is loaded on
  restart; the existing saved selection is retained when valid. Later commands
  do not automatically register or retry that failed creation.
- Active-chat deletion writes any necessary empty replacement, then its saved
  selection, then moves the old document to `data/chats/trash/`, then publishes.
  A failed intermediate step can leave an extra replacement or a saved
  selection ahead of the live selection. The old chat remains live in memory
  on failure; its file is either still active or recoverable in trash. Deleting
  a non-active chat needs no configuration write. Restart loads active files
  and honours a valid saved selection, falling back to the most recent chat if
  that selection is absent. There is no automatic multi-file rollback.

Each replacement syncs file contents before rename and its destination directory
afterward. Newly created directories sync their parent entries; existing
directories need no extra parent sync. Moves confirm both affected directories,
and removals confirm their parent. A post-mutation sync failure raises
`CommitUncertainError` and blocks subsequent writes to the affected document in
that store instance. Do not treat it as a safe rejection or blindly retry it:
stop the application, inspect the active JSON/configuration and recoverable
trash, reconcile the intended state, then reopen the stores. These guarantees
assume one Jouzetsu process owns the application home; external edits require a
restart. Actual power-loss durability still depends on the filesystem and
storage device honouring synchronisation.

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
