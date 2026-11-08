# Codex instructions — Jouzetsu Phase 0: architecture audit ONLY

Repository: https://github.com/APasz/Jouzetsu  
Planning baseline: `71ab489eee01895378bfd6824c46235b2891a3f6`

I am planning a substantial but targeted refactor of Jouzetsu, my single-user, Python/FastHTML, self-hosted frontend for local LLMs. The production application and its authoritative JSON data will move to a Raspberry Pi, while inference runs on separate backends such as LM Studio on my desktop. We want simultaneous independently configured inference backends, initially LM Studio and streaming generic OpenAI-compatible `/v1/chat/completions` servers. The app must remain operational for reading and editing chats/characters while inference is offline. Development will still occur on Linux desktop; production is a native systemd service on a headless Pi, deployed directly by rsync/SSH, with all application-owned data/config inside its project root.

**Read the attached `Jouzetsu_Rewrite_Requirements.md` in full. It is the agreed scope and takes precedence over old application behaviour.** The decisions were made through an extended interview; do not ask me to decide the same issues again without a new, concrete technical reason.

## Your job in this phase

**Perform an architectural assessment and propose a staged implementation plan. Do not change application code, tests, schemas, routes, UI, or production files, and do not begin implementation.** You may inspect files, run existing tests/checks, and write a review report if appropriate. Identify repo changes since the baseline before relying on stale assumptions.

Your assessment must:

1. Map the existing architecture with concrete references to code. Trace application startup, JSON storage/transactions, chat/session registry and `active_chat_id`, generation lifecycle, model discovery/loading, request construction, image handling, SSE/browser synchronisation, access approval and global settings, and frontend rendering.
2. Identify specific coupling preventing multiple backend instances or independent browser navigation. Propose narrow interfaces that permit future adapters without prematurely building a broad plugin system. State which concepts belong to backends versus the core app.
3. Propose backend IDs, model identities, capabilities, per-backend global defaults, and per-chat independent backend profiles. Show example JSON for the new schema. There is **no requirement to migrate legacy chat data**.
4. Design the server-owned generation lifecycle, with chat-level locks, per-backend concurrency limits (default one), preferred in-memory FIFO queue, streamed updates to multiple browsers, cancellation, no automatic retry, model loading and unload protection, and interruption handling.
5. Design robust data consistency: committed writes confirmed only after durable success, independent draft revisions, 1–2-second debounced draft persistence with a ten-second maximum interval, SSE propagation and draft conflict handling, and per-message persisted idempotency keys preventing duplicate sends or generation.
6. Investigate the frontend. **Do not assume replacing FastHTML is desirable.** Assess concrete code complexity, performance, regression risk and maintainability; compare realistic alternatives only if evidence supports a change. Preserving the current visual design and mobile feature parity is mandatory.
7. Audit security in the new headless deployment: new device approval by `approval_phrase` only; remote approved browsers have normal application/settings rights but cannot approve/revoke devices; localhost is always trusted with full administration. Clarify any tension between those two statements rather than silently relaxing the phrase-only rule. Cover rate limiting, SSE revocation, CSRF and trusted client IP detection.
8. Identify modules/features to remove (continuity review, hardware host monitoring, Exit control, Undo) and modules/features to preserve. Identify how to preserve chat appearance while rendering transient green/System error entries with a top accent rail, not as real chat messages.
9. Propose safe systemd deployment and manual rsync deployment without touching data/config/virtual environment. Account for native ARM64 dependencies and avoid containers.
10. Provide test plan with unit/integration/browser and failure-path tests; keep Ruff and strict basedpyright. No actual LM Studio needed for ordinary CI tests.

## Format of your response

Present:

- **Executive assessment:** which parts are sound, which need redesign, and the largest risks.
- **Architecture map:** modules, responsibilities, and precise references.
- **Proposed architecture:** interfaces, data schemas, state transitions, and concurrency model.
- **Frontend recommendation:** keep/refactor/replace, with evidence and estimated conceptual scope (not a time estimate).
- **Staged implementation plan:** small testable phases, acceptance criteria, and explicit approval gates.
- **Risks and blockers:** separating facts from assumptions, recommended resolutions.
- **Questions for me:** only new decisions truly needed to proceed, not previously settled preferences.

Treat this as a design review; be critical and technically specific. Do not proceed to implementation without my approval.
