import { CSRF_FORM_FIELD, csrfToken } from './forms.js';
import { RETRY_DRAFT_NOTICE_ACTION } from './notices.js';

const DRAFT_SAVE_DELAY_MS = 250;

export class DraftController {
    #timer = 0;
    #queue = Promise.resolve();
    #dirty = false;
    #suppressPageHide = false;
    #currentDraft;
    #announce;

    constructor(currentDraft, announce) {
        this.#currentDraft = currentDraft;
        this.#announce = announce;
    }

    get isDirty() {
        return this.#dirty;
    }

    markClean() {
        this.#dirty = false;
    }

    async preserve() {
        return !this.#dirty || await this.flush();
    }

    handleInput() {
        this.#dirty = true;
        if (this.#timer) window.clearTimeout(this.#timer);
        this.#timer = window.setTimeout(() => {
            this.#timer = 0;
            const draft = this.#currentDraft();
            if (!draft) return;
            void this.#queueDraft(draft.chatId, draft.value)
                .then(() => {
                    if (this.#isCurrent(draft)) this.#dirty = false;
                })
                .catch(() => this.#announce('Could not save the current draft', true, RETRY_DRAFT_NOTICE_ACTION));
        }, DRAFT_SAVE_DELAY_MS);
    }

    async flush() {
        if (this.#timer) {
            window.clearTimeout(this.#timer);
            this.#timer = 0;
        }
        const draft = this.#currentDraft();
        if (!draft) return true;
        try {
            await this.#queueDraft(draft.chatId, draft.value);
            if (this.#isCurrent(draft)) this.#dirty = false;
            return true;
        } catch {
            this.#announce('Could not save the current draft', true, RETRY_DRAFT_NOTICE_ACTION);
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
        const body = new URLSearchParams({ chat_id: draft.chatId, draft: draft.value, [CSRF_FORM_FIELD]: csrfToken() });
        if (navigator.sendBeacon) {
            navigator.sendBeacon('/chat/draft', body);
            return;
        }
        void this.#persist(draft.chatId, draft.value, true);
    }

    #isCurrent(draft) {
        const current = this.#currentDraft();
        return Boolean(current && current.chatId === draft.chatId && current.value === draft.value);
    }

    #queueDraft(chatId, value) {
        const write = this.#queue.catch(() => undefined).then(() => this.#persist(chatId, value));
        this.#queue = write.catch(() => undefined);
        return write;
    }

    async #persist(chatId, value, keepalive = false) {
        const body = new URLSearchParams({ chat_id: chatId, draft: value, [CSRF_FORM_FIELD]: csrfToken() });
        const response = await window.fetch('/chat/draft', {
            method: 'POST',
            headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
            body,
            keepalive,
        });
        if (!response.ok) throw new Error('Draft save failed');
    }
}
