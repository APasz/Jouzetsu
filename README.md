# Jouzetsu

Self-hosted web interface for [LM Studio](https://lmstudio.ai/), built with
FastHTML. It streams chats with local models and stores conversations and
character profiles as JSON files you control.

## Requirements

- Python 3.14 or newer
- [uv](https://docs.astral.sh/uv/)
- An optional local LM Studio or OpenAI-compatible Chat Completions server
  (the application also works while inference is unavailable)
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
  forked, merged, truncated, or deleted.
- Multiple independently configured backends, FIFO queues, concurrency limits,
  manual model registration and aliases. LM Studio also supports model loading
  and optional application-managed idle unloading.
- Vision chats with multiple JPEG, PNG, or WebP images: upload, paste, or drop
  images into the composer, preview them, and send them with or without text.
- Character profiles and JSON preset packs; start immutable single-character or
  multi-character chats in Roleplay, Assistant, Story, or Custom prompt modes.
- Shared drafts and live SSE updates, explicit chat URLs and browser-local
  navigation. The composer's **+** menu groups Add images, Send only and quick templates;
  Shift-click on Send also saves without generating.
- Phrase-based browser enrollment; actual localhost connections always have
  administrative access. Approved remote browsers share ordinary settings.

## Configuration and data

Use the in-app settings for normal changes. `config.json` is also editable;
restart Jouzetsu after editing it directly. Its sections are `generation`, `ui`, `theme`, `logging`, `access`, and `backends`.
It is a complete, strict, versioned document (`"version": 2`): missing,
unknown, or invalid settings prevent startup rather than being silently changed.
Built-in British-spelling replacements and the theme palette are packaged
separately. `generation.british_spelling_replacements` and `theme` are written
only when you customise them.

Vision models are marked in model selectors. Images stay with their original
user message in later requests, regeneration, continuations. A conversation containing images requires a model with confirmed vision support;
switching to a text-only model blocks generation until a compatible model is
selected. Editing a caption preserves its images.

Original image files and small previews live in `data/images/`; the original
resolution is preserved. Draft attachments survive reloads, and forks retain their references. Image files are kept when references are removed so
other chats and recoverable trash remain intact. Initial limits allow 32 images
per message, 64 MiB per image, 512 MiB per upload, and 100 million pixels per
image. The upload policy is centralised in `ImageLimits` for future settings.
Chat documents use schema version 4, including backend profiles, generation
attribution, independent committed/draft revisions, and submission receipts.
WebP originals are sent to LM Studio as lossless PNG images to work around the
0.4.26 compatibility endpoint rejecting WebP data URLs.

### Rewrite checkpoint: Gate 3 browser integration

`InferenceRuntime` is now the sole owner of chat/configuration commands and
inference jobs. The FastHTML browser application uses explicit chat IDs and
committed revisions for every transcript/settings edit, and independent revisions
for the shared draft. A stale edit is rejected; locally typed text remains in the
browser for manual resolution. There is no automatic merge.

Each chat has a stable `/chats/<id>` URL. Selection, history, scroll position and
open panels belong to the browser, and browsing never writes a global selection.
Returning to `/chats` restores that browser's last-opened chat using a persistent
navigation cookie. Missing/deleted selections fall back to an available chat;
explicit bookmarked chat URLs take precedence. This cookie grants no access.
The shared transcript, draft, generation status and ordinary settings synchronise
through SSE. Reconnecting refreshes state without submitting a generation.
Draft notifications refresh only the composer. A committed-revision mismatch
falls back to a full refresh; streamed output and status continue to update.
The Thinking disclosure keeps its open/closed state while reasoning updates.
Right-click a message on desktop, or press and hold it on a touchscreen, to open
Copy, Message details, and Fork chat. Moving a touch cancels the long press so
scrolling remains available.

Backend forms carry a fingerprint of the configuration they displayed. Conflicting
saves return HTTP 409 and keep locally edited values for manual resolution.

The Gate 3 corrective profile used 160 messages (163,890 transcript characters),
two Chromium browsers and eight shared draft updates on desktop loopback:

| Draft refresh measurement | Before | After |
| --- | ---: | ---: |
| Full transcript renders | 16 | 0 |
| Combined fragment response bytes | 12,621,696 | 92,080 |
| Mean request duration, browser 1 / 2 | 176 / 125 ms | 3.4 / 2.7 ms |

The new path used 16 composer renders. Resource Timing response bytes fell by
99.3%; these are desktop measurements, not Raspberry Pi performance guarantees.
Browser regressions check unchanged transcript nodes and draft conflict/attachment
preservation without timing thresholds.

Fresh installs have no backend. Add one in **Global settings → Backends**, register
or discover models, choose its default profile, and optionally select it as the
new-chat default. Each chat keeps an independent profile for each backend it has
used. Changing defaults affects new chats and explicit profile resets. Removing a
backend retains historical IDs, attribution and chat profiles. Generic backends
have no native load/unload controls. Unknown vision support blocks image inference;
unsupported/unknown sampling parameters remain stored and are marked inactive.
No credentials or arbitrary generation parameters are exposed.

**Access & Logs** is available from localhost. Set an approval phrase there and
enter it in each new remote browser. The phrase is the sole enrollment path;
localhost administrators can revoke devices but cannot directly approve them.
Failed phrase attempts have a bounded, in-memory per-peer cooldown. Revocation
blocks subsequent requests and ends existing SSE connections. Forwarded headers
are not trusted by the bundled server (`proxy_headers=False`).

Undo, continuity review, host monitoring and the in-app process Exit control have
been removed. This checkpoint uses fresh versioned data; aggregate chat-store
migration is not supported. Gate 4 deployment work remains separate.

The service locks committed chat data as soon as a job is accepted, including
queued/loading work. Drafts remain editable. Jobs and errors are memory-only;
caller cancellation does not cancel server-owned generation. Completed,
interrupted, and deliberately cancelled text is saved before publication.
Reasoning alone never creates or replaces an assistant response. Attribution
retains backend/model identities and label snapshots after renames or deletion.
Repeated submission IDs return their persisted message IDs without regenerating,
even after a message is edited or deleted.
Once a submission is accepted, its save and generation acceptance finish despite
caller cancellation. Intentional shutdown still suppresses a new generation.
Continue creates a separate assistant message, preserving the preceding response.
Regenerating a continuation keeps the preceding history and its synthetic
Continue prompt; cancelled or interrupted output is retained in the new message.

Manual models on generic backends can generate despite unsupported, malformed,
or unavailable discovery endpoints. Discovery errors remain visible through
`discovery_error()` and status notifications; they do not erase cached metadata
or establish that inference is offline. Connection failures still mark a backend
offline, and image requests still require known vision support. Native LM Studio
management continues to require its native inventory.

Stopping waits up to two seconds for completion, then reports a timeout if work
remains. The chat lock and backend slot stay held until the adapter terminates and
finalisation completes; late output after cancellation is ignored. An adapter
that never terminates can require process recovery. A timeout never forcibly
releases its resources or starts another job in its occupied slot.

Streaming previews never enter persisted transcripts. A failed final save keeps
the preview and chat lock; `retry_save()` retries persistence without inference.
An uncertain filesystem commit instead requires storage recovery and restart,
as enforced by Gate 1's write guard. Deferred draft errors are exposed through
`PersistenceController.chat_write_error` and status notifications. Draft writes
wait 1.5 seconds after activity pauses in each chat, with a ten-second continuous-typing bound,
and flush during shutdown. JSON encoding and durable filesystem operations reuse
the existing worker executor and serialised write boundary.

These service operations write one chat document or the configuration, without
updating a global active chat. They introduce no multi-file transaction. A crash
can leave newer cached backend metadata beside an older chat; historical IDs and
profiles remain recoverable. There is no persisted chat selection and therefore no chat/configuration pair
to commit during chat creation or deletion. `shutdown()` stops accepting commands, drains accepted
commands and jobs, flushes drafts, and then closes adapters. It returns false if
work cannot finish within its shutdown window, output remains unsaved, or cleanup
fails. In-flight resources remain available after a timeout; callers must report
the outcome and can retry shutdown after pending work finishes. Adapter cleanup
failures are logged and retried without rejecting an already committed edit.

Global Settings separates Behaviour, Appearance and Backends. Appearance retains
the existing App, User, Assistant and System colourways, automatic shades,
light/dark mode, icon colours and semantic message-action colours (the default).
Interface icons use locally bundled Phosphor SVGs; their licence and source
details are in `jouzetsu/web/static/icons/`. Generation errors use the System
colourway with a top accent rail.

Useful environment overrides:

- `JOUZETSU_HOME` — application home
- `JOUZETSU_HOST` / `JOUZETSU_PORT` — listen address and port
- `JOUZETSU_LOG_LEVEL` — `DEBUG`, `INFO` (default), or `WARNING`

The application home contains `config.json`, `data/chats/`,
`data/characters/`, `data/character-presets/`, and `data/logs/`. Deleted
chats move to `data/chats/trash/`. Invalid chat/character files are isolated in
place and logged; other valid documents remain usable. They are not moved,
overwritten, repaired or deleted automatically. Add custom empty-chat
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

Committed chat edits, new/forked/character-based chats, deletion,
settings, and character changes save before publishing authoritative state or
notifying browsers. Existing chat/message and shared configuration references
are retained when saved candidates are applied. Generation acceptance saves its
initial transcript before starting an inference stream. Generation output is
transient while streaming and is saved on completion or cancellation.

Draft text and attachments publish to the shared in-memory draft immediately and
are written through ordered, coalesced persistence. The browser serialises its
text, attachment and submission requests against that draft revision. Rejected
image uploads stay in the browser for retry/download. A failed deferred disk
write is visible with a retry action; recent drafts can still be lost in a crash.

Runtime JSON serialisation, writes, moves and synchronisation use the existing
worker pool. Per-chat locks cover candidate preparation through durable
publication, and one configuration lock covers all settings/backend writes.
The writer lock orders drafts and committed/configuration writes. Readers and SSE
can run while storage is busy. Accepted submissions finish saving and accepting
inference independently of caller cancellation; shutdown still rejects new work.
Startup loading remains synchronous before requests are accepted.

Separate JSON files are **not a transaction**. Chat lifecycle operations now write
only the affected chat document, leaving no active-selection/configuration write
to fail afterward. Deletion of the last chat leaves an empty workspace; a new
chat is created only on request. Backend metadata/configuration and independent
chat documents may still reflect different moments after a crash. Stable IDs,
retained profiles and saved attribution preserve recoverability without rollback
or a transaction engine. A missing preview can be rebuilt from its verified
original; missing originals keep their references, show an unavailable placeholder
and block inference.

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
private network such as Tailscale. Do not expose the application port directly to the public internet. Proxy and
Pi deployment guidance belongs to Gate 4.

## Development

Run the checkout manually, using its root for local configuration and data:

```sh
uv sync --frozen --group dev
uv run jouzetsu --home .
```

The server runs in the foreground; Ctrl+C shuts it down and flushes pending
drafts. The listen port comes from the local `config.json` (8080 on a fresh
configuration). Run one instance per application home; the data-directory lock
prevents duplicate instances. Local archives under `archives/` are ignored by Git.

Development checks:

```sh
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
