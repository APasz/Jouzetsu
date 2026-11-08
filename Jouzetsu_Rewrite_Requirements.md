# Jouzetsu — Partial Rewrite Requirements

**Planning baseline:** [APasz/Jouzetsu](https://github.com/APasz/Jouzetsu), commit [`71ab489eee01895378bfd6824c46235b2891a3f6`](https://github.com/APasz/Jouzetsu/commit/71ab489eee01895378bfd6824c46235b2891a3f6).

**Status:** Requirements captured through an extended design interview on 7–8 November 2026. This is a **requirements baseline**, not authority to implement the entire project without checkpoints. Where a stated requirement conflicts with older behaviour or earlier decisions, the **latest decision wins**. Major architectural changes require the owner's approval.

## 1. Purpose and governing principles

Jouzetsu is a **single-user**, self-hosted chat/storytelling frontend offering substantial control over conversations. It is being refactored so that the application and its authoritative data run continuously on a Raspberry Pi, while inference is provided by independently available backends (initially LM Studio on a desktop and generic OpenAI-compatible servers). Browser clients connect from the home LAN directly, or remotely through Tailscale. The desktop is **not** required to be powered on for normal chat/character browsing and editing.

Principles:

1. Preserve the existing visual identity, interaction quality, and supported features unless changes are explicitly approved. Material architecture improvements are welcome; redesign for its own sake is not.
2. Prefer correctness, reliable failure handling, simple implementation, and explicit user control to speculative features or clever automation.
3. No legacy data migration: the Pi deployment starts with fresh configuration and data. Old desktop data is left untouched and may be archived independently.
4. Work in **approved, testable phases**. Audit and propose a plan first, with no production code changes until approved. At each checkpoint: explain changes, tests, risks, and remaining limitations.
5. Preserve comprehensive automated tests, Ruff, and strict basedpyright. Simulated backends should exercise the ordinary test suite without requiring live LM Studio.

## 2. Hosting, process, network, and files

- **Development:** Linux desktop. **Production:** headless Raspberry Pi, using an externally attached SSD/NVMe drive, preferably USB boot. The application must be agnostic about boot media and disk device.
- **Deployment:** All Jouzetsu-owned application code, `.venv`, `config.json`, chats, characters, images, and local log files reside inside one self-contained project/application root. OS-managed exceptions, such as systemd unit configuration and journald, are acceptable.
- **Execution:** native `systemd` service under an unprivileged user, automatic boot startup and crash restart, graceful shutdown. Retain command-line execution for development. No Docker or containers. Remove the in-app **Exit** control and its associated functionality.
- **Updates:** manually initiated SSH/`rsync` deployment from development desktop to Pi. Transfer code but **protect** production config, data, and virtual environment; install ARM-native dependencies on Pi using `uv sync --frozen`, restart service, and report result. No staging, automatic rollback, or release-management system initially.
- **Network:** direct HTTP over trusted home LAN; remote access via HTTP over Tailscale. No public port forwarding or direct internet exposure. Binding to `0.0.0.0` is not sufficient to restrict access; document firewall/Tailscale enforcement. HTTPS/reverse-proxy configuration is deferred.
- **Storage:** Pi filesystem is authoritative. No bidirectional sync with desktop or another server, and no external data storage dependency. No database migration. Preserve JSON chats, characters, and configuration; image binaries remain standalone files. A cache/database such as DiskCache is not a requirement. Built-in backups and comprehensive trash management are deferred.

## 3. Device security

- The application is **strictly single-user**, not a multi-account product. All approved browsers can access and edit the same chats, characters, normal global settings, and backend configurations. No per-user roles.
- New browser approval uses the existing `approval_phrase`. **An already approved remote browser must NOT be allowed to approve or revoke other browsers.** The user's latest intent is that the phrase is the sole way a new browser becomes approved.
- Requests actually originating from localhost are always authorised with full administrative access; access administration and device revocation are localhost-restricted. Confirm during the audit whether any current localhost *direct approval* UI should be removed to uphold phrase-only approval. Do not mistake spoofable HTTP headers for a loopback socket peer.
- Add modest, in-memory, per-client failed-approval rate limiting and temporary cooldowns. No permanent global lockout. Do not log phrases, cookies, other secrets, or request bodies.
- Revoking a browser immediately denies subsequent requests and disconnects any active SSE stream. Keep CSRF protections and secure, unpredictable device identifiers.
- The browser approval mechanism is not a substitute for a public-internet authentication gateway. A future security/privacy hardening pass is explicitly deferred.

## 4. Inference backends

- Implement a **backend-agnostic application interface**, not a grandiose plugin system. Initial concrete adapters: **LM Studio** (native model discovery/load/unload where supported, plus OpenAI-compatible Chat Completions) and **generic OpenAI-compatible Chat Completions**.
- Multiple independently configured backend instances can exist and operate simultaneously. Each backend has a stable opaque identifier distinct from display name/address; each model is identified in the context of its backend. A chat can switch backend/model at any time.
- Initial generic-protocol support is **streaming `/v1/chat/completions` only**. No `/v1/responses`, non-streaming Chat Completions, special provider protocols, cloud-specific authentication, or API key management initially. The first eventual cloud integration of interest is OpenAI, but defer it.
- Backends need capability-aware interfaces. Model loading/unloading, idle TTL, vision, structured reasoning, and parameter support are **optional**, not assumed universal. Do not force every adapter to emulate LM Studio.
- **Model discovery:** automatic wherever possible; manual model registration as fallback. Preserve cached model metadata and user-defined aliases while offline; don't remove registered models just because a discovery response omits them. Unknown capabilities remain distinguishable from known unsupported capabilities.
- **Vision:** block image-containing requests if selected model is known not to support vision **or if vision capability is unknown**. Never silently strip attachments. Manual vision-capability overrides and intentional text-only omission are deferred.
- **Reasoning:** support separately structured reasoning provided by the backend; display it separately and retain existing user choice to save/discard it. Do not parse `<think>` tags out of ordinary text initially.
- **Parameters:** generic backends use a conservative standard Chat Completions subset with known support (e.g. temperature, top-p, token limit). Keep unsupported settings preserved but clearly marked in UI; do not silently claim they are active. No speculative arbitrary-parameter forwarding, auto-fallback, or retry-with-different-parameters.
- **Model management:** where supported, automatic loading may be enabled/disabled per backend; default automatic for LM Studio. Retain configurable idle unloading, protecting active/queued jobs from Jouzetsu-initiated unloads. Backend is responsible for GPU/RAM allocation; do not unload unrelated idle models merely to make room. No automatic desktop power-on.
- **Backend management:** in Global Settings and manually in `config.json` (manual edits take effect upon restart). Add, edit, test, enable/disable, and remove. Offline/disabled/deleted backends must not silently alter chat model selections. Preserve historical references/profiles after deletion; reusing a backend name/URL must not silently reassociate old identity. Block disruptive config mutations while a backend has queued/active work; allow nondisruptive rename.
- **Monitoring:** initial status check and activity-aware periodic checks; reduce/suspend idle polling when no clients/jobs. Distinguish online, offline, and unknown/stale. Never wake machines as a side effect. Keep inference metrics but **remove detailed hardware host-monitoring UI and implementation**; machine monitoring belongs to a separate future project.
- **Timeouts:** sensible fixed connection, loading, and inactivity timeouts, generously allowing lengthy initial prompt processing. No overall generation time limit. Expose config only if real experience warrants it. Never automatically retry generation. Discovery/connectivity checks can retry safely.

## 5. Chat-level settings and defaults

- A chat has a **current backend selection** plus an independent **per-backend generation settings profile** stored in its JSON. A backend profile includes its last-selected model and applicable generation settings; profiles are **not** per model at this stage.
- Switching to a backend never previously used in that chat initialises a profile by copying **compatible settings** from the previously selected profile. Thereafter the two profiles evolve independently, even when switching between them. A copy-from-profile control can be added later.
- Separately, every backend has **global defaults**. Creating a new chat snapshots its initially selected backend's global defaults into its profile. Changing global defaults never automatically modifies existing profiles. A reset-to-backend-defaults action is suitable.
- New chats use an explicitly configured global default backend/model, even if it is offline. Fresh Jouzetsu installs start with **no configured backend** and **no setup wizard**. Chats created before a default is configured remain unassigned after one is subsequently established; the user selects a backend manually.
- **Chat-wide** rather than backend-specific: system prompt, character identities, story/roleplay/assistant/custom mode, British-spelling and other text-processing preferences, and other behaviour. The entire conversation history is sent as configured; no automatic context trimming or custom context boundaries yet.
- Every generated assistant response records the **actual backend/model** used. Existing attribution persists across model switches and backend deletion. Regenerating uses the **currently selected backend/model**, not the originally attributed one.
- No silent fallback between models or backends, even if the selected backend is offline. Manual switching is always possible.

## 6. Generation, queuing, and editing

- **Server-owned** generation: a task continues if the initiating browser closes/disconnects. All approved browsers can observe it live and can stop it. Generation is not restarted on reconnection; no automatic resumption after Jouzetsu process restart.
- Different chats may generate simultaneously, including on the same backend, constrained by an individually configurable **per-backend concurrency limit**, default **1**. Coordinate actual inference usage including any internal inference tasks. One chat must never have two competing generations.
- When a backend is busy, **prefer** an in-memory FIFO wait queue per backend, cancellable before it starts; queued chat locks immediately. If the queue proves disproportionately complex, a clear busy rejection is an explicitly acceptable alternative **only after discussing the trade-off**. Queues are never persistent, never wait for offline backends, and fail visibly if their backend goes offline.
- A chat's **committed messages and generation settings** are locked from the instant a generation request is accepted (including queued, model loading, and streaming), until completion/failure/cancellation. Enforce locks server-side. Other chats remain usable.
- **Drafts are exempt:** a browser can continue composing text and attaching images during generation/queueing, but cannot submit another user message until the existing operation ends. Draft content is not appended to an already accepted inference request.
- Sending a message and generating a response are separate backend operations. Normal **Send** persists the message and starts generation when possible; **Send only** persists without inference. When offline, Send still saves the message and reports that generation did not start. Allow consecutive user messages; never silently merge/reorder messages to satisfy a backend's chat template.
- UI: preserve normal Send action; add a compact Send-only option in a split button. Shift-click as Send-only shortcut; mobile press-and-hold is a potential convenience if robust and accessible, not essential to correctness.
- Inference never automatically wakes the desktop; no automatic offline generation queue; no automatic generation retry. The separate host-management project and its eventual explicit Start Desktop integration are **deferred completely**, including speculative API definitions.
- Persist complete/cancelled/interrupted assistant responses when generation ends. **Do not persist streaming tokens periodically**. Accept losing in-progress response if Jouzetsu crashes. Failed connection to inference while Jouzetsu remains alive still preserves partial output.
- If generation fails **without producing assistant text**, create no assistant message. If some content was produced, retain it; failures appear separately. Deliberate cancellation preserves produced text as an ordinary message.
- **Regeneration:** only directly regenerate the final assistant response. Keep the original response intact until the replacement produces actual response text; structured reasoning alone does not count. If replacement fails before text, retain the original. After first real text, replace it; interruption then preserves partial replacement. Old-response in-place regeneration and automatic fork-on-regenerate are deferred.
- Retain existing message editing, continue, resend, fork, merge, and truncation functionality unless specifically changed. **Remove Undo** and its underlying snapshots/routes/UI/tests. Deleting one message is immediate; deletions affecting multiple messages, truncation, and whole-chat deletion require confirmation.
- Remove the optional **continuity-review/rewrite subsystem** entirely for this release.

## 7. Error presentation and diagnostics

- Generation errors are **not text in transcript messages** and are never included in prompt context or stored as conversation content.
- Keep the latest generation error per chat in server **memory**. Render it visually as the last chat entry, using the configurable **System (green by default) colourway** with an **accent rail across the top** (not the user/assistant left/right rails). An error following partial output appears after that partial assistant response.
- Any approved browser opening/reconnecting to that chat can see the current error. Errors can be dismissed; accepting a new generation clears the previous error. Errors are forgotten on process restart.
- Preserve error categorisation (offline, load failure, timeout, incompatibility, other). Log identifiers and diagnostic metadata, timing and token usage, but **no actual message text, prompts, reasoning, image bytes, approval phrase, credentials, or unredacted echoed backend request bodies**.
- Retain useful application logging; no verbose prompt/response logging mode and no new retention policy required now.

## 8. Browser state, live synchronisation, drafts, and data integrity

- Browser-local: navigation selection, chat URLs/history, scroll position, open panels/dialogs and other presentation state. Give each chat a stable URL such as `/chats/<id>`. Browser Back, Forward, reload, and bookmarking work. **Remove the application-wide `active_chat_id` assumption.** All mutations explicitly identify the target chat; no cross-browser selection changes.
- Server-shared: chat contents, characters, generation state, backend configuration, and **a single draft per chat** (not per device).
- Changes synchronise live via SSE or another justified mechanism. Reconnection refreshes stale views without triggering another generation. Preserve focused unsaved text instead of overwriting it on incoming updates.
- **Committed chat data:** optimistic concurrency with a per-chat revision; reject stale modifications rather than merging. Browser retains rejected edits for manual resolution. Do not treat high-frequency streaming UI events as committed edits.
- **Drafts:** independent draft revision, separate from committed-chat revision. Keep a live authoritative in-memory draft with fast, ordered/coalesced updates across browsers. Reject stale draft mutations without losing local text/attachments. No complex collaborative editor or automatic merge.
- **Draft disk durability:** save approximately 1–2 seconds after typing pauses and at least every 10 seconds of continuous typing; flush on orderly shutdown. Recent draft changes may be lost in a Pi crash. Avoid writing a large chat file on each keystroke; optionally separate internal draft persistence if justified. Show errors if draft persistence fails.
- **Committed writes:** successful disk persistence is mandatory before acknowledging a committed change. On failure, no success event, no authoritative in-memory change, no damaged prior file. Atomic writes alone are insufficient if memory is mutated first without rollback.
- **Message idempotency:** browser generates a stable identifier per submission; persist it with the resulting user message. Repeated identical submissions return the same message and **never** launch a duplicate generation. Reuse of an identifier with different content is rejected. Identifier remains valid after a service restart. If crash occurred after saving but before generation, manual generation is acceptable; retries must not start duplicate generation.
- **No offline browser data editing.** A connection to the Pi is needed for reading/updating saved data. Already displayed content and typed text should remain present where practical during disconnect. No offline-first database, service worker, or sync engine.
- **Invalid JSON chats/characters:** isolate damaged files, load other valid files, never delete/overwrite/auto-repair damaged data; display warning and log diagnostics. Invalid critical security/configuration should fail startup, not revert to insecure defaults.
- **Missing image originals:** keep chat usable and references intact, display placeholders; block generation if required images are missing/corrupt. Rebuild missing thumbnails from valid originals when possible; never silently strip attachments.
- **Deletion:** deleted chat JSON goes to recoverable `data/chats/trash/` folder; retain originals for images; no automatic cleanup or garbage collection. Full trash management/backups deferred.

## 9. Frontend and browser quality

- Preserve existing flat/compact visual identity, existing customised colours, layout feel and control affordances. Maintain **mobile feature parity**, while permitting focused mobile/touch adaptations where useful, simple and consistent with shared code.
- **Keep FastHTML provisionally.** During Phase 0, audit whether its current architecture creates demonstrable maintainability, UI state, rendering, or synchronisation problems. It may be substantially refactored/replaced only after a documented comparison and explicit approval; do not switch frameworks merely for novelty.
- Investigate targeted performance improvements (large transcripts, repeated fragment rendering, SSE update fan-out, DOM work). Profile before large changes. No automatic virtualisation or pagination project without evidence.
- Use standards-based browser APIs, with automated and practical testing focused on Chromium on Linux/Android. Cross-browser release certification is not required.
- Comprehensive unit/integration/browser tests are mandatory, plus failure-path coverage; preserve Ruff and strict basedpyright. Browser/device-specific UI tests should check draft preservation on refresh and status changes.

## 10. Defer / out of scope

- Multi-user accounts/roles or publicly exposed service.
- Cloud-specific integrations, API credentials, OpenAI Responses API, arbitrary non-streaming responses, protocol-specific adapters beyond LM Studio and generic Chat Completions.
- Host power controls/Wake-on-LAN/management-service integration and machine resource dashboard.
- Backup/restore features, trash UI and retention, automatic image garbage collection, migration of old desktop JSON.
- Automatic fallback or inference retries, offline generation queues, cross-restart generation recovery.
- Automatic context trimming and user-defined context boundaries, extracting `<think>` tagged reasoning, manual vision-capability overrides, and intentionally omitting images from context.
- Per-model generation profiles, live inheritance of global defaults, advanced backend-specific parameters, elaborate connection retry configuration.
- Persistent undo/history, early-message regeneration shortcuts, and continuity review.
- TLS certificate management/reverse proxy integration (deployment option for the future).

## 11. Phase 0 audit: required investigation and report

**DO NOT MAKE PRODUCTION CODE CHANGES IN PHASE 0.** Read the actual repo at the specified baseline (or report divergence), run existing tests if feasible, and deliver a written technical report for approval.

The report should include:

1. A factual map of modules and responsibilities, including the existing LM Studio client/protocol, state registry and generation controller, config/JSON schemas, FastHTML routes, JS/SSE refresh, access control, and asset/character storage.
2. A gap/impact table mapping every major requirement to affected files, recommended design and tests. Distinguish confirmed limitations from hypotheses; cite local files/functions/line ranges.
3. Evidence-based evaluation of retaining FastHTML versus alternatives. Preserve UI by default. State whether a frontend rewrite is actually justified, and whether it should be independent of the inference refactor.
4. Proposed backend interface and capability representation, model identity, per-backend configuration/global-default/profile schemas, missing-backend historical references, and strict unknown-vision handling.
5. Generation lifecycle/state-machine proposal: queue/locks, concurrent backends, streaming fan-out, cancellation, model loading/unloading, retry/timeout handling, regeneration transaction, and response/error persistence semantics.
6. JSON transactional-write approach, revisions, draft write coalescing, idempotency, data corruption isolation, image integrity and invalidation behaviour. Be candid where requirements need a two-phase commit or revised storage boundary.
7. Access control threat review: phrase-only approval, local admin, actual socket peer trust, HTTP LAN/Tailscale limitations, brute-force limits, CSRF, cookie safety, SSE revocation, and server-side route checks. Identify any incompatibilities between 'localhost full admin' and 'phrase-only new approval' rather than guessing.
8. A proposed **sequence of small, reviewable implementation phases**, dependencies, tests and acceptance criteria, and identified risks or unverified assumptions. Require owner approval at every major checkpoint.
9. Deployment proposal for Pi systemd and safe direct `rsync` without touching production data/configuration. Note native ARM dependencies and necessary operational documentation.
10. A concise list of **only those decisions genuinely blocking implementation**, with a recommended option and trade-offs for each. Do not reopen decisions already settled above merely because a different architecture is familiar.

## 12. Suggested high-level implementation gates (not immutable)

- **Gate 0:** Approve audit, frontend decision, and detailed sequence.
- **Gate 1:** Backend architecture, configuration/data schema, and pure unit tests.
- **Gate 2:** Generation lifecycle, multiple backends, queues, state and persistence guarantees; simulated backend tests.
- **Gate 3:** Browser/API/UI, live synchronisation, per-chat URLs, drafts, permissions and regression tests.
- **Gate 4:** Pi deployment and systemd documentation, comprehensive failure/security/integration testing, final review.

For each gate, Codex must deliver: code changed, tests executed and exact outcomes, manual tests still needed, deviations from this specification, risks, and unresolved questions. **Do not proceed into the next major phase without explicit approval.**
