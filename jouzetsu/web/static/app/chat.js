import {
    byId,
    chatFragmentUrl,
    localizeMessageUpdatedTimes,
    parseFragment,
    syncElementAttributes,
} from './dom.js';
import { addCsrfToken, liveFormBody } from './forms.js';
import { currentComposerDraft } from './composer.js';

const FULL_REFRESH = 'full';
const LIVE_REFRESH = 'live';
const STREAM_REFRESH_DELAY_MS = 75;
const LATEST_MESSAGE_ACTION_PATTERN = /^\/messages\/[^/]+\/(?:continue|regenerate|resend)$/;

const messageChildren = (messages) => Array.from(messages.children).filter(
    (child) => child instanceof HTMLElement && child.dataset.messageId,
);

const stableMarkup = (element) => {
    const clone = element.cloneNode(true);
    if (!(clone instanceof HTMLElement)) return '';
    [clone, ...clone.querySelectorAll('[style]')].forEach((item) => item.removeAttribute('style'));
    clone.classList.remove('is-resized');
    clone.removeAttribute('data-composer-requested-height');
    clone.removeAttribute('data-composer-minimum-height');
    return clone.outerHTML;
};

const matchesServerMessageMarkup = (current, next) => {
    const clone = current.cloneNode(true);
    if (!(clone instanceof HTMLElement)) return false;
    clone.querySelectorAll('[data-message-updated-at]').forEach((timestamp) => timestamp.replaceChildren());
    return clone.outerHTML === next.outerHTML;
};

export class ChatController {
    #composer;
    #messages;
    #drafts;
    #notices;
    #panels;
    #refreshTimer = 0;
    #refreshInFlight = false;
    #scheduledRefreshKind = '';
    #queuedRefreshKind = '';
    #latestFullRefreshRequest = 0;
    #panelRefreshesInFlight = new Set();

    constructor({ composer, messages, drafts, notices, panels }) {
        this.#composer = composer;
        this.#messages = messages;
        this.#drafts = drafts;
        this.#notices = notices;
        this.#panels = panels;
    }

    async replaceFragment({ followLatest = false } = {}) {
        const current = byId('chat-fragment');
        if (!(current instanceof HTMLElement)) return;
        this.#messages.closeContextMenu();
        const refreshRequest = this.#latestFullRefreshRequest + 1;
        this.#latestFullRefreshRequest = refreshRequest;
        try {
            const response = await window.fetch(chatFragmentUrl('/fragments/chat'), {
                headers: { 'HX-Request': 'true' },
                cache: 'no-store',
            });
            if (!response.ok) {
                if (response.status === 403) window.location.reload();
                return;
            }
            const markup = await response.text();
            if (refreshRequest !== this.#latestFullRefreshRequest) return;
            const next = parseFragment(markup, 'chat-fragment');
            if (!next) return;
            if (document.documentElement.classList.contains('jouzetsu-panel-open')) {
                next.setAttribute('inert', '');
                next.setAttribute('aria-hidden', 'true');
            }
            this.#reconcileFragment(current, next, followLatest);
            this.#messages.watchScrollContainer();
            localizeMessageUpdatedTimes();
            this.#messages.refreshOpenDetails();
        } catch {
            // Keep the working page in place through a temporary reconnect.
        }
    }

    async replacePanelFragment(name) {
        if (this.#panelRefreshesInFlight.has(name)) return;
        const current = byId(`${name}-panel`);
        if (!(current instanceof HTMLElement) || current.classList.contains('is-open')) return;
        this.#panelRefreshesInFlight.add(name);
        try {
            const response = await window.fetch(`/fragments/${encodeURIComponent(name)}`, {
                headers: { 'HX-Request': 'true' },
                cache: 'no-store',
            });
            if (!response.ok) return;
            const next = parseFragment(await response.text(), `${name}-panel`);
            if (!next || !current.isConnected || current.classList.contains('is-open')) return;
            if (current.outerHTML !== next.outerHTML) current.replaceWith(next);
        } catch {
            // Auxiliary panels should never take down chat interaction.
        } finally {
            this.#panelRefreshesInFlight.delete(name);
        }
    }

    refreshPanels() {
        return Promise.all([this.replacePanelFragment('navigation'), this.replacePanelFragment('settings')]);
    }

    async submitMutation(form) {
        if (form.dataset.chatMutationSaving === 'true') return;
        const action = new URL(form.action, window.location.origin).pathname;
        const followLatest = action === '/chat/send' || LATEST_MESSAGE_ACTION_PATTERN.test(action);
        const openedPanel = form.closest('.jouzetsu-panel.is-open');
        form.dataset.chatMutationSaving = 'true';
        form.setAttribute('aria-busy', 'true');
        try {
            if (action !== '/chat/stop' && this.#drafts.isDirty) await this.#drafts.flush();
            addCsrfToken(form);
            const response = await window.fetch(form.action, {
                method: form.method || 'POST',
                headers: { 'Content-Type': 'application/x-www-form-urlencoded', 'HX-Request': 'true' },
                body: liveFormBody(form),
                cache: 'no-store',
            });
            if (response.status === 403) {
                window.location.reload();
                return;
            }
            if (!response.ok) throw new Error(`Chat mutation request failed: ${response.status}`);
            const resultUrl = new URL(response.url, window.location.origin);
            const error = resultUrl.searchParams.get('error');
            if (error) {
                this.#notices.announce(error, true);
                return;
            }
            if (action === '/messages/edit') this.#exitEditMode();
            if (action === '/chat/send' || action === '/messages/edit') this.#drafts.markClean();
            if (openedPanel instanceof HTMLElement) this.#panels.close(false);
            const detailsDialog = form.closest('#message-details-dialog');
            if (detailsDialog instanceof HTMLDialogElement) detailsDialog.close();
            await Promise.all([this.replaceFragment({ followLatest }), this.refreshPanels()]);
            const notice = resultUrl.searchParams.get('notice');
            if (notice) this.#notices.announce(notice);
        } catch {
            this.#notices.announce('Could not save this change', true);
        } finally {
            delete form.dataset.chatMutationSaving;
            form.removeAttribute('aria-busy');
        }
    }

    async submitLiveForm(form, submitter = null) {
        addCsrfToken(form);
        form.dataset.liveSaving = 'true';
        form.setAttribute('aria-busy', 'true');
        try {
            const action = submitter?.hasAttribute('formaction')
                ? submitter.formAction
                : form.action;
            const method = submitter?.hasAttribute('formmethod')
                ? submitter.formMethod
                : form.method || 'POST';
            const response = await window.fetch(action, {
                method,
                headers: { 'Content-Type': 'application/x-www-form-urlencoded', 'HX-Request': 'true' },
                body: liveFormBody(form),
                cache: 'no-store',
            });
            if (response.status === 403) {
                window.location.reload();
                return;
            }
            if (!response.ok) throw new Error(`Live form request failed: ${response.status}`);
            const resultUrl = new URL(response.url, window.location.origin);
            const error = resultUrl.searchParams.get('error');
            if (error) {
                this.#notices.announce(error, true);
                return;
            }
            await Promise.all([this.replaceFragment(), this.refreshPanels()]);
            this.#notices.announce(submitter?.dataset.liveNotice || form.dataset.liveNotice || 'Saved');
        } catch {
            this.#notices.announce('Could not save this change', true);
        } finally {
            delete form.dataset.liveSaving;
            form.removeAttribute('aria-busy');
        }
    }

    async navigate(href) {
        const url = new URL(href, window.location.origin);
        if (url.origin !== window.location.origin || !['/', '/chats'].includes(url.pathname)) return;
        if (!await this.#drafts.preserve()) return;
        const currentUrl = new URL(window.location.href);
        this.#setHistoryUrl(url, currentUrl.searchParams.has('edit') && !url.searchParams.has('edit'));
        await this.replaceFragment();
    }

    async handlePopstate() {
        if (!['/', '/chats'].includes(window.location.pathname) || !await this.#drafts.preserve()) return;
        await this.replaceFragment();
    }

    scheduleRefresh(kind) {
        if (this.#refreshInFlight) {
            this.#queuedRefreshKind = this.#preferredRefresh(this.#queuedRefreshKind, kind);
            return;
        }
        this.#scheduledRefreshKind = this.#preferredRefresh(this.#scheduledRefreshKind, kind);
        if (this.#refreshTimer) return;
        this.#refreshTimer = window.setTimeout(() => {
            this.#refreshTimer = 0;
            const scheduledKind = this.#scheduledRefreshKind;
            this.#scheduledRefreshKind = '';
            void this.#refresh(scheduledKind);
        }, STREAM_REFRESH_DELAY_MS);
    }

    scheduleFullRefresh() {
        this.scheduleRefresh(FULL_REFRESH);
    }

    scheduleLiveRefresh() {
        this.scheduleRefresh(LIVE_REFRESH);
    }

    #reconcileFragment(current, next, followLatest) {
        const currentTopbar = current.querySelector(':scope > .jouzetsu-topbar');
        const nextTopbar = next.querySelector(':scope > .jouzetsu-topbar');
        const currentMessages = current.querySelector('#message-list');
        const nextMessages = next.querySelector('#message-list');
        const currentComposer = current.querySelector(':scope > .jouzetsu-composer');
        const nextComposer = next.querySelector(':scope > .jouzetsu-composer');
        if (
            !(currentTopbar instanceof HTMLElement)
            || !(nextTopbar instanceof HTMLElement)
            || !(currentMessages instanceof HTMLElement)
            || !(nextMessages instanceof HTMLElement)
            || !(currentComposer instanceof HTMLElement)
            || !(nextComposer instanceof HTMLElement)
        ) {
            current.replaceWith(next);
            return;
        }
        const currentForm = currentComposer.querySelector('.jouzetsu-composer-form');
        const nextForm = nextComposer.querySelector('.jouzetsu-composer-form');
        const currentChatId = currentForm instanceof HTMLFormElement ? currentForm.dataset.chatId || '' : '';
        const nextChatId = nextForm instanceof HTMLFormElement ? nextForm.dataset.chatId || '' : '';
        const sameChat = Boolean(currentChatId && currentChatId === nextChatId);
        const sameComposerMode = (
            currentForm instanceof HTMLFormElement
            && nextForm instanceof HTMLFormElement
            && currentForm.dataset.composerMode === nextForm.dataset.composerMode
        );
        const enteringEditMode = (
            currentForm instanceof HTMLFormElement
            && nextForm instanceof HTMLFormElement
            && currentForm.dataset.composerMode === 'compose'
            && nextForm.dataset.composerMode === 'edit'
        );
        const generationStarted = (
            currentForm instanceof HTMLFormElement
            && nextForm instanceof HTMLFormElement
            && currentForm.dataset.composerGenerating === 'false'
            && nextForm.dataset.composerGenerating === 'true'
        );
        const messageScrollTop = sameChat ? currentMessages.scrollTop : null;
        const stickToBottom = sameChat && this.#messages.isNearBottom(currentMessages);
        const preserveComposerDraft = sameChat && sameComposerMode && !generationStarted;
        const liveDraft = preserveComposerDraft ? currentComposerDraft() : null;
        const selection = preserveComposerDraft ? this.#composer.focusedSelection() : null;
        syncElementAttributes(current, next);
        if (currentTopbar.outerHTML !== nextTopbar.outerHTML) currentTopbar.replaceWith(nextTopbar);
        this.#reconcileMessageList(currentMessages, nextMessages);
        let renderedComposer = currentComposer;
        if (stableMarkup(currentComposer) !== stableMarkup(nextComposer)) {
            currentComposer.replaceWith(nextComposer);
            renderedComposer = nextComposer;
        }
        const renderedForm = renderedComposer.querySelector('.jouzetsu-composer-form');
        const renderedInput = renderedForm?.querySelector('.jouzetsu-message-input');
        if (
            liveDraft
            && renderedForm instanceof HTMLFormElement
            && renderedInput instanceof HTMLTextAreaElement
            && renderedForm.dataset.chatId === liveDraft.chatId
        ) renderedInput.value = liveDraft.value;
        if (enteringEditMode) this.#composer.focusInput(renderedInput);
        else if (
            renderedForm instanceof HTMLFormElement
            && renderedInput instanceof HTMLTextAreaElement
            && renderedForm.dataset.chatId === currentChatId
        ) this.#composer.restoreSelection(renderedInput, selection);
        this.#composer.reconcileViewport();
        this.#composer.restoreHeight();
        this.#composer.autoSizeInput(renderedInput);
        if (followLatest || !sameChat || stickToBottom) this.#messages.scrollListToBottom(currentMessages);
        else this.#messages.restoreScroll(currentMessages, messageScrollTop);
        this.#composer.syncPrimary();
        this.#composer.syncStatusElapsedTimer();
    }

    #reconcileMessageList(current, next) {
        syncElementAttributes(current, next);
        const nextMessages = messageChildren(next);
        if (!nextMessages.length) {
            if (current.innerHTML !== next.innerHTML) {
                current.replaceChildren(...Array.from(next.childNodes, (child) => child.cloneNode(true)));
            }
            return;
        }
        const currentById = new Map(messageChildren(current).map((message) => [message.dataset.messageId, message]));
        const nextIds = new Set(nextMessages.map((message) => message.dataset.messageId));
        for (const message of messageChildren(current)) {
            if (!nextIds.has(message.dataset.messageId)) message.remove();
        }
        for (const nextMessage of nextMessages) {
            const messageId = nextMessage.dataset.messageId;
            const existing = currentById.get(messageId);
            if (
                existing instanceof HTMLElement
                && existing.isConnected
                && !matchesServerMessageMarkup(existing, nextMessage)
            ) {
                const replacement = nextMessage.cloneNode(true);
                existing.replaceWith(replacement);
                currentById.set(messageId, replacement);
            }
        }
        let insertionPoint = current.firstElementChild;
        for (const nextMessage of nextMessages) {
            const messageId = nextMessage.dataset.messageId;
            const existing = currentById.get(messageId);
            const message = existing instanceof HTMLElement && existing.isConnected ? existing : nextMessage.cloneNode(true);
            currentById.set(messageId, message);
            if (message !== insertionPoint) current.insertBefore(message, insertionPoint);
            insertionPoint = message.nextElementSibling;
        }
        while (insertionPoint) {
            const nextSibling = insertionPoint.nextElementSibling;
            insertionPoint.remove();
            insertionPoint = nextSibling;
        }
    }

    async #replaceLiveFragment() {
        const current = byId('chat-fragment');
        if (!(current instanceof HTMLElement)) return;
        const fullRefreshAtRequestStart = this.#latestFullRefreshRequest;
        try {
            const response = await window.fetch(chatFragmentUrl('/fragments/chat/live'), {
                headers: { 'HX-Request': 'true' },
                cache: 'no-store',
            });
            if (!response.ok) {
                if (response.status === 403) window.location.reload();
                return;
            }
            const markup = await response.text();
            if (fullRefreshAtRequestStart !== this.#latestFullRefreshRequest) return;
            const next = parseFragment(markup, 'chat-live-fragment');
            if (!next) return;
            const currentForm = current.querySelector('.jouzetsu-composer-form');
            if (!(currentForm instanceof HTMLFormElement) || currentForm.dataset.chatId !== next.dataset.chatId) {
                this.scheduleFullRefresh();
                return;
            }
            if (!this.#composer.syncLiveStatus(
                current.querySelector('[data-composer-status]'),
                next.querySelector('[data-live-composer-status]'),
            )) {
                this.scheduleFullRefresh();
                return;
            }
            if (!this.#composer.syncGenerationReasoning(
                current.querySelector('[data-generation-reasoning]'),
                next.querySelector('[data-live-generation-reasoning]'),
            )) {
                this.scheduleFullRefresh();
                return;
            }
            const nextMessage = next.querySelector('[data-live-streaming-message-id]');
            if (!(nextMessage instanceof HTMLElement)) return;
            const messageId = nextMessage.dataset.liveStreamingMessageId;
            const currentContent = Array.from(current.querySelectorAll('[data-streaming-message-id]')).find(
                (item) => item instanceof HTMLElement && item.dataset.streamingMessageId === messageId,
            );
            if (!(currentContent instanceof HTMLElement)) {
                this.scheduleFullRefresh();
                return;
            }
            const currentMessage = currentContent.closest('[data-message-id]');
            if (!(currentMessage instanceof HTMLElement)) {
                this.scheduleFullRefresh();
                return;
            }
            const messageList = byId('message-list');
            const stickToBottom = this.#messages.isNearBottom(messageList);
            this.#messages.syncStreamingContent(currentContent, nextMessage);
            if (stickToBottom) this.#messages.scrollStreamingMessageUntilTop(messageList, currentMessage);
        } catch {
            // A full refresh on the next event safely heals transient failures.
        }
    }

    async #refresh(kind) {
        if (this.#refreshInFlight) {
            this.#queuedRefreshKind = this.#preferredRefresh(this.#queuedRefreshKind, kind);
            return;
        }
        this.#refreshInFlight = true;
        try {
            if (kind === FULL_REFRESH) await this.replaceFragment();
            else await this.#replaceLiveFragment();
        } finally {
            this.#refreshInFlight = false;
            if (!this.#queuedRefreshKind) return;
            const queuedKind = this.#queuedRefreshKind;
            this.#queuedRefreshKind = '';
            this.scheduleRefresh(queuedKind);
        }
    }

    #preferredRefresh(currentKind, nextKind) {
        return currentKind === FULL_REFRESH || nextKind === FULL_REFRESH ? FULL_REFRESH : LIVE_REFRESH;
    }

    #setHistoryUrl(url, replace) {
        const destination = `${url.pathname}${url.search}${url.hash}`;
        if (replace) window.history.replaceState({}, '', destination);
        else window.history.pushState({}, '', destination);
    }

    #exitEditMode() {
        const url = new URL(window.location.href);
        if (!url.searchParams.has('edit')) return;
        url.searchParams.delete('edit');
        this.#setHistoryUrl(url, true);
    }

}
