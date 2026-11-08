import { CSRF_FORM_FIELD, csrfToken } from './forms.js';
import { RETRY_DRAFT_NOTICE_ACTION } from './notices.js';

const DRAFT_SAVE_DELAY_MS = 250;

export class DraftController {
    #timer = 0;
    #queue = Promise.resolve();
    #dirty = false;
    #savingDraft = false;
    #suppressPageHide = false;
    #currentDraft;
    #announce;
    #revisions = new Map();
    #conflicted = false;

    constructor(currentDraft, announce) {
        this.#currentDraft = currentDraft;
        this.#announce = announce;
    }

    get isDirty() {
        return this.#dirty;
    }

    markClean() { this.#dirty = false; this.#conflicted = false; }

    acceptRevision(chatId, revision) {
        if (!this.#dirty) this.#revisions.set(chatId, Number(revision));
    }

    async preserve() {
        return !this.#dirty || await this.flush();
    }

    handleInput() {
        const draft = this.#currentDraft();
        if (draft && !this.#dirty) this.#revisions.set(draft.chatId, draft.revision);
        this.#dirty = true;
        // Coalesce input without postponing shared updates during continuous typing.
        if (this.#timer || this.#savingDraft) return;
        this.#timer = window.setTimeout(() => {
            this.#timer = 0;
            const draft = this.#currentDraft();
            if (!draft) return;
            this.#savingDraft = true;
            void this.#queueDraft(draft.chatId, draft.value)
                .then(() => {
                    if (this.#isCurrent(draft)) this.#dirty = false;
                })
                .catch(() => this.#reportFailure())
                .finally(() => {
                    this.#savingDraft = false;
                    if (this.#dirty && !this.#isCurrent(draft)) this.handleInput();
                });
        }, DRAFT_SAVE_DELAY_MS);
    }

    async flush() {
        if (this.#timer) {
            window.clearTimeout(this.#timer);
            this.#timer = 0;
        }
        const draft = this.#currentDraft();
        if (!draft || !this.#dirty) return true;
        try {
            await this.#queueDraft(draft.chatId, draft.value);
            if (this.#isCurrent(draft)) this.#dirty = false;
            return true;
        } catch {
            this.#reportFailure();
            return false;
        }
    }

    deferPageHide() {
        this.#suppressPageHide = true;
    }

    persistOnPageHide() {
        if (this.#suppressPageHide) return;
        if (this.#timer) {
            window.clearTimeout(this.#timer);
            this.#timer = 0;
        }
        const draft = this.#currentDraft();
        if (!draft || !this.#dirty || !csrfToken()) return;
        const body = new URLSearchParams({ chat_id: draft.chatId, draft: draft.value, draft_revision: String(this.#revisions.get(draft.chatId) ?? draft.revision), [CSRF_FORM_FIELD]: csrfToken() });
        if (navigator.sendBeacon) {
            navigator.sendBeacon('/chat/draft', body);
            return;
        }
        void this.#persist(draft.chatId, draft.value, true);
    }

    #reportFailure() {
        this.#announce(this.#conflicted ? 'Draft changed elsewhere. Your text is preserved; copy it before reloading to resolve.' : 'Could not save the current draft', true, this.#conflicted ? '' : RETRY_DRAFT_NOTICE_ACTION);
    }

    #isCurrent(draft) {
        const current = this.#currentDraft();
        return Boolean(current && current.chatId === draft.chatId && current.value === draft.value);
    }

    serialize(operation) {
        const write = this.#queue.catch(() => undefined).then(operation);
        this.#queue = write.catch(() => undefined);
        return write;
    }

    savedRevision(chatId, revision) {
        this.#revisions.set(chatId, revision);
        const shell = document.getElementById('chat-fragment');
        if (shell?.dataset.chatId === chatId) shell.dataset.draftRevision = String(revision);
        document.querySelectorAll(`form[data-chat-id="${chatId}"] input[name="draft_revision"]`).forEach((input) => { input.value = String(revision); });
    }

    #queueDraft(chatId, value) {
        return this.serialize(() => this.#persist(chatId, value));
    }

    async #persist(chatId, value, keepalive = false) {
        if (this.#conflicted) throw new Error('Draft changed elsewhere; copy your local text before reloading');
        const current = this.#currentDraft();
        const revision = this.#revisions.get(chatId) ?? current?.revision;
        const body = new URLSearchParams({ chat_id: chatId, draft: value, draft_revision: String(revision), [CSRF_FORM_FIELD]: csrfToken() });
        const response = await window.fetch('/chat/draft', {
            method: 'POST',
            headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
            body,
            keepalive,
        });
        if (response.status === 409) {
            this.#conflicted = true;
        }
        if (!response.ok) throw new Error('Draft save failed');
        const result = await response.json();
        this.savedRevision(chatId, result.draft_revision);
    }
}
