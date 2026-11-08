import {
    byId,
    chatFragmentUrl,
    localizeMessageUpdatedTimes,
    parseFragment,
    syncElementAttributes,
} from './dom.js';
import { addCsrfToken, liveFormBody, randomIdentifier } from './forms.js';

const FULL_REFRESH = 'full';
const LIVE_REFRESH = 'live';
const STREAM_REFRESH_DELAY_MS = 75;
const LATEST_MESSAGE_ACTION_PATTERN = /^\/messages\/[^/]+\/(?:continue|regenerate|resend)$/;
const submissionKey = (chatId) => `jouzetsu-submission-${chatId}`;

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
    const stable = (element) => {
        const clone = element.cloneNode(true);
        if (!(clone instanceof HTMLElement)) return '';
        clone.querySelectorAll('[data-message-updated-at]').forEach((timestamp) => timestamp.replaceChildren());
        clone.querySelectorAll('input[name="revision"], input[name="draft_revision"]').forEach((input) => input.setAttribute('value', ''));
        return clone.outerHTML;
    };
    if (stable(current) !== stable(next)) return false;
    for (const name of ['revision', 'draft_revision']) {
        const value = next.querySelector(`input[name="${name}"]`)?.value;
        if (value !== undefined) current.querySelectorAll(`input[name="${name}"]`).forEach((input) => { input.value = value; });
    }
    return true;
};

const formKey = (form) => `${form.action}|${form.elements.namedItem('chat_id')?.value || ''}|${form.elements.namedItem('model_id')?.value || ''}`;

const reconcileSurface = (current, next) => {
    const focus = current.contains(document.activeElement) ? document.activeElement : null;
    const focusName = focus?.name;
    const focusForm = focus?.form instanceof HTMLFormElement ? formKey(focus.form) : '';
    const selection = typeof focus?.selectionStart === 'number' ? [focus.selectionStart, focus.selectionEnd] : null;
    const nextForms = Array.from(next.querySelectorAll('form'));
    const preserved = [];
    for (const form of current.querySelectorAll('form[data-local-dirty]')) {
        const target = nextForms.find((candidate) => formKey(candidate) === formKey(form));
        if (!target) return false;
        preserved.push([target, form]);
    }
    preserved.forEach(([target, form]) => target.replaceWith(form));
    current.replaceWith(next);
    const restored = focus?.isConnected ? focus : Array.from(next.querySelectorAll('form')).find((form) => formKey(form) === focusForm)?.elements.namedItem(focusName);
    if (restored instanceof HTMLElement) {
        restored.focus({preventScroll: true});
        if (selection && typeof restored.setSelectionRange === 'function') restored.setSelectionRange(...selection);
    }
    return restored instanceof HTMLElement ? restored : true;
};

export class ChatController {
    #composer;
    #messages;
    #drafts;
    #notices;
    #panels;
    #images;
    #refreshTimer = 0;
    #refreshInFlight = false;
    #scheduledRefreshKind = '';
    #queuedRefreshKind = '';
    #latestFullRefreshRequest = 0;
    #panelRefreshesInFlight = new Set();
    #queuedPanelRefreshes = new Set();
    #submissions = new Map();
    #dialogRequests = new Map();

    constructor({ composer, messages, drafts, notices, panels, images }) {
        this.#composer = composer;
        this.#messages = messages;
        this.#drafts = drafts;
        this.#notices = notices;
        this.#panels = panels;
        this.#images = images;
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
                if (response.status === 404) {
                    if (this.#drafts.isDirty || current.querySelector('[data-composer-mode=edit]')) this.#notices.announce('This chat was deleted. Your local text is preserved; copy it before leaving.', true);
                    else window.location.assign('/chats');
                }
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
            this.#images.synchronize();
            this.#messages.watchScrollContainer();
            localizeMessageUpdatedTimes();
            this.#messages.refreshOpenDetails();
        } catch {
            // Keep the working page in place through a temporary reconnect.
        }
    }

    async replacePanelFragment(name) {
        if (this.#panelRefreshesInFlight.has(name)) {
            this.#queuedPanelRefreshes.add(name);
            return;
        }
        const current = byId(`${name}-panel`);
        if (!(current instanceof HTMLElement)) return;
        this.#panelRefreshesInFlight.add(name);
        const path = window.location.pathname;
        try {
            const response = await window.fetch(chatFragmentUrl(`/fragments/${encodeURIComponent(name)}`), {
                headers: { 'HX-Request': 'true' },
                cache: 'no-store',
            });
            if (!response.ok) return;
            const next = parseFragment(await response.text(), `${name}-panel`);
            if (!next || !current.isConnected || path !== window.location.pathname) return;
            if (current.classList.contains('is-open')) { next.classList.add('is-open'); next.removeAttribute('inert'); next.setAttribute('aria-hidden', 'false'); }
            if (current.dataset.chatId !== next.dataset.chatId) current.replaceWith(next);
            else if (current.outerHTML !== next.outerHTML) reconcileSurface(current, next);
        } catch {
            // Auxiliary panels should never take down chat interaction.
        } finally {
            this.#panelRefreshesInFlight.delete(name);
            const queued = this.#queuedPanelRefreshes.delete(name);
            if (queued || path !== window.location.pathname) void this.replacePanelFragment(name);
        }
    }

    refreshPanels() {
        return Promise.all([this.replacePanelFragment('navigation'), this.replacePanelFragment('settings')]);
    }

    refreshDialogs() {
        return Promise.all([this.#refreshDialog('global-settings-dialog', 'global'), this.#refreshDialog('access-dialog', 'access')]);
    }

    async #refreshDialog(id, fragment) {
        let current = byId(id);
        if (!(current instanceof HTMLDialogElement)) return;
        const request = (this.#dialogRequests.get(id) || 0) + 1;
        this.#dialogRequests.set(id, request);
        const tab = current.querySelector('[role="tab"][aria-selected="true"]')?.dataset.tabName || '';
        try {
            const response = await fetch(`/fragments/${fragment}?tab=${encodeURIComponent(tab)}`, {
                headers: {'HX-Request': 'true'}, cache: 'no-store',
            });
            const markup = await response.text();
            if (!response.ok || this.#dialogRequests.get(id) !== request) return;
            current = byId(id);
            if (!(current instanceof HTMLDialogElement)) return;
            const next = parseFragment(markup, id);
            if (!next) return;
            const active = current.querySelector('[role="tab"][aria-selected="true"]')?.dataset.tabName || '';
            const open = current.open;
            const detailKey = (details, index) => details.id || details.dataset.testid || String(index);
            const expanded = new Map(Array.from(current.querySelectorAll('details')).map((details, index) => [detailKey(details, index), details.open]));
            next.removeAttribute('open');
            next.querySelectorAll('details').forEach((details, index) => { details.open = expanded.get(detailKey(details, index)) || false; });
            const feedback = current.querySelector('.jouzetsu-dialog-feedback');
            if (feedback && !next.querySelector('.jouzetsu-dialog-feedback')) next.firstElementChild?.prepend(feedback.cloneNode(true));
            const reconciled = reconcileSurface(current, next);
            if (!reconciled) return;
            const activeTab = next.querySelector(`[role="tab"][data-tab-name="${CSS.escape(active)}"]`);
            if (activeTab) this.#panels.activateTab(activeTab);
            if (open) next.showModal();
            if (reconciled instanceof HTMLElement) reconciled.focus({preventScroll: true});
        } catch { /* Reconnection will refresh settings from the shared configuration. */ }
    }

    async submitMutation(form, submitter = null) {
        const sendOnly = form.dataset.sendOnly === 'true';
        delete form.dataset.sendOnly;
        if (form.dataset.chatMutationSaving === 'true') return;
        const action = new URL(form.action, window.location.origin).pathname;
        const followLatest = action === '/chat/send' || LATEST_MESSAGE_ACTION_PATTERN.test(action);
        const openedPanel = form.closest('.jouzetsu-panel.is-open');
        const sourcePath = window.location.pathname;
        const submitted = liveFormBody(form, submitter);
        form.dataset.chatMutationSaving = 'true';
        form.setAttribute('aria-busy', 'true');
        try {
            if (action !== '/chat/stop' && this.#images.isUploading && !await this.#images.flush()) return;
            const previousSubmission = action === '/chat/send'
                ? this.#submission(submissionKey(submitted.get('chat_id'))) : null;
            const retry = previousSubmission
                && previousSubmission.content === submitted.get('content')
                && previousSubmission.images === submitted.get('images');
            // An unacknowledged send may already have cleared the shared draft.
            // Retry its receipt before attempting another mutation of that draft.
            if (action !== '/chat/stop' && !retry && this.#drafts.isDirty && !await this.#drafts.flush()) return;
            addCsrfToken(form);
            const body = liveFormBody(form, submitter);
            if (action === '/chat/send' || action === '/messages/edit') body.set('content', submitted.get('content') || '');
            body.set('revision', submitted.get('revision') || '');
            if (action === '/chat/send') {
                const key = submissionKey(body.get('chat_id'));
                const content = body.get('content');
                const images = body.get('images');
                let attempt = this.#submission(key);
                if (!attempt || attempt.content !== content || attempt.images !== images) attempt = {id: randomIdentifier(), content, images};
                this.#submissions.set(key, attempt);
                try { sessionStorage.setItem(key, JSON.stringify(attempt)); } catch { /* The in-memory attempt still protects retries. */ }
                body.set('submission_id', attempt.id);
                if (sendOnly) body.set('generate', 'false');
            }
            const persist = async () => {
                const response = await window.fetch(form.action, {
                    method: form.method || 'POST',
                    headers: { 'Content-Type': 'application/x-www-form-urlencoded', 'HX-Request': 'true' },
                    body,
                    cache: 'no-store',
                });
                if (response.ok && response.headers.get('X-Jouzetsu-Draft-Cleared') === 'true') {
                    this.#drafts.savedRevision(body.get('chat_id'), Number(response.headers.get('X-Jouzetsu-Draft-Revision')));
                }
                return response;
            };
            const response = action === '/chat/send' ? await this.#drafts.serialize(persist) : await persist();
            if (response.status === 403) {
                window.location.reload();
                return;
            }
            if (!response.ok) {
                this.#notices.announce(await response.text(), true);
                return;
            }
            const resultUrl = new URL(response.headers.get("X-Jouzetsu-Location") || response.url, window.location.origin);
            const error = resultUrl.searchParams.get('error');
            if (error) {
                this.#notices.announce(error, true);
                return;
            }
            if (action === '/messages/edit' && sourcePath === window.location.pathname) this.#exitEditMode();
            if (action === '/chat/send') {
                const key = submissionKey(body.get('chat_id'));
                if (this.#submissions.get(key)?.id === body.get('submission_id')) {
                    this.#submissions.delete(key);
                    try { sessionStorage.removeItem(key); } catch { /* Browser storage is optional. */ }
                }
                const input = document.querySelector(`.jouzetsu-composer-form[data-chat-id="${body.get('chat_id')}"][data-composer-mode="compose"] .jouzetsu-message-input`);
                if (input instanceof HTMLTextAreaElement && input.value === body.get('content')) { input.value = ''; this.#drafts.markClean(); }
            }
            if (action === '/messages/edit' && sourcePath === window.location.pathname) this.#drafts.markClean();
            if (sourcePath === window.location.pathname && resultUrl.pathname !== sourcePath && (resultUrl.pathname === '/chats' || /^\/chats\/[A-Za-z0-9]+$/.test(resultUrl.pathname))) {
                window.location.assign(resultUrl.href);
                return;
            }
            if (openedPanel instanceof HTMLElement && sourcePath === window.location.pathname) this.#panels.close(false);
            const detailsDialog = form.closest('#message-details-dialog');
            if (detailsDialog instanceof HTMLDialogElement) detailsDialog.close();
            await Promise.all([this.replaceFragment({ followLatest: followLatest && sourcePath === window.location.pathname }), this.refreshPanels()]);
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
        const submitted = liveFormBody(form, submitter).toString();
        let resubmit = false;
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
                body: liveFormBody(form, submitter),
                cache: 'no-store',
            });
            if (response.status === 403) {
                window.location.reload();
                return;
            }
            if (!response.ok) { this.#notices.announce(await response.text(), true); return; }
            const resultUrl = new URL(response.headers.get("X-Jouzetsu-Location") || response.url, window.location.origin);
            const error = resultUrl.searchParams.get('error');
            if (error) {
                this.#notices.announce(error, true);
                return;
            }
            const revision = response.headers.get('X-Jouzetsu-Chat-Revision');
            resubmit = liveFormBody(form, submitter).toString() !== submitted;
            if (revision && form.elements.namedItem('revision')) form.elements.namedItem('revision').value = revision;
            if (!resubmit) delete form.dataset.localDirty;
            await Promise.all([this.replaceFragment(), this.refreshPanels()]);
            this.#notices.announce(submitter?.dataset.liveNotice || form.dataset.liveNotice || 'Saved');
        } catch {
            this.#notices.announce('Could not save this change', true);
        } finally {
            delete form.dataset.liveSaving;
            form.removeAttribute('aria-busy');
            if (resubmit && form.isConnected && form.querySelector('[data-live-autosave]')) form.requestSubmit();
        }
    }

    #submission(key) {
        const current = this.#submissions.get(key);
        if (current) return current;
        try {
            const attempt = JSON.parse(sessionStorage.getItem(key));
            return attempt && typeof attempt.id === 'string'
                && typeof attempt.content === 'string' && typeof attempt.images === 'string'
                ? attempt : null;
        }
        catch { return null; }
    }

    async navigate(href) {
        const url = new URL(href, window.location.origin);
        if (url.origin !== window.location.origin || !/^\/chats\/[A-Za-z0-9]+$/.test(url.pathname)) return;
        if (this.#images.isUploading && !await this.#images.flush()) return;
        if (!await this.#drafts.preserve()) return;
        const currentUrl = new URL(window.location.href);
        this.#setHistoryUrl(url, currentUrl.searchParams.has('edit') && !url.searchParams.has('edit'));
        await Promise.all([this.replaceFragment(), this.refreshPanels()]);
    }

    async handlePopstate() {
        if (!/^\/chats\/[A-Za-z0-9]+$/.test(window.location.pathname)) return;
        if (this.#images.isUploading && !await this.#images.flush() || !await this.#drafts.preserve()) {
            const id = byId('chat-fragment')?.dataset.chatId;
            if (id) history.replaceState({}, '', `/chats/${id}`);
            return;
        }
        await Promise.all([this.replaceFragment(), this.refreshPanels()]);
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
        if (!sameChat) this.#messages.saveScrollForNavigation();
        const sameComposerMode = (
            currentForm instanceof HTMLFormElement
            && nextForm instanceof HTMLFormElement
            && currentForm.dataset.composerMode === nextForm.dataset.composerMode
        );
        const missingEditedMessage = sameChat
            && currentForm?.dataset.composerMode === 'edit'
            && nextForm?.dataset.composerMode !== 'edit'
            && new URL(location.href).searchParams.get('edit') === currentForm.elements.namedItem('message_id')?.value;
        const enteringEditMode = (
            currentForm instanceof HTMLFormElement
            && nextForm instanceof HTMLFormElement
            && currentForm.dataset.composerMode === 'compose'
            && nextForm.dataset.composerMode === 'edit'
        );
        const messageScrollTop = sameChat ? currentMessages.scrollTop : null;
        const nextStreaming = nextMessages.querySelector('article.is-streaming[data-message-id]');
        const firstPreview = sameChat && nextStreaming instanceof HTMLElement && !Array.from(currentMessages.querySelectorAll('article.is-streaming[data-message-id]')).some((message) => message.dataset.messageId === nextStreaming.dataset.messageId);
        const streamingMessageAnchor = sameChat
            ? this.#messages.captureStreamingMessageAnchor(currentMessages)
            : null;
        const stickToBottom = (
            sameChat
            && !streamingMessageAnchor
            && this.#messages.isNearBottom(currentMessages)
        );
        const preserveComposerDraft = sameChat && (sameComposerMode || missingEditedMessage) && (
            this.#drafts.isDirty || this.#submissions.has(submissionKey(currentChatId))
            || currentForm?.dataset.composerMode === 'edit'
        );
        const currentInput = currentForm?.querySelector('.jouzetsu-message-input');
        const liveDraft = preserveComposerDraft && currentInput instanceof HTMLTextAreaElement ? {
            chatId: currentChatId, value: currentInput.value,
            revision: Number(currentForm.elements.namedItem('draft_revision')?.value || '0'),
            images: currentForm.elements.namedItem('images')?.value,
            hasImages: currentForm.dataset.hasImages,
        } : null;
        const preserveImages = liveDraft && Number(next.dataset.draftRevision) > liveDraft.revision;
        const imageUpload = preserveImages ? currentComposer.querySelector('[data-image-upload-form]') : null;
        const editRevision = currentForm?.dataset.composerMode === 'edit' ? currentForm.elements.namedItem('revision')?.value : null;
        const selection = preserveComposerDraft ? this.#composer.focusedSelection() : null;
        syncElementAttributes(current, next);
        if (currentTopbar.outerHTML !== nextTopbar.outerHTML) currentTopbar.replaceWith(nextTopbar);
        this.#reconcileMessageList(currentMessages, nextMessages);
        let renderedComposer = currentComposer;
        if (!missingEditedMessage && stableMarkup(currentComposer) !== stableMarkup(nextComposer)) {
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
        if (editRevision !== null && renderedForm instanceof HTMLFormElement && sameComposerMode) renderedForm.elements.namedItem('revision').value = editRevision;
        if (enteringEditMode) this.#composer.focusInput(renderedInput);
        else if (
            renderedForm instanceof HTMLFormElement
            && renderedInput instanceof HTMLTextAreaElement
            && renderedForm.dataset.chatId === currentChatId
        ) this.#composer.restoreSelection(renderedInput, selection);
        if (liveDraft && renderedForm instanceof HTMLFormElement) {
            const revision = renderedForm.elements.namedItem('draft_revision');
            if (revision instanceof HTMLInputElement) revision.value = String(liveDraft.revision);
            if (preserveImages) {
                const images = renderedForm.elements.namedItem('images');
                if (images instanceof HTMLInputElement) images.value = liveDraft.images;
                renderedForm.dataset.hasImages = liveDraft.hasImages;
                if (renderedInput instanceof HTMLTextAreaElement) renderedInput.required = liveDraft.hasImages !== 'true';
                const nextUpload = renderedComposer.querySelector('[data-image-upload-form]');
                if (imageUpload && nextUpload) nextUpload.replaceWith(imageUpload);
            }
        }
        this.#drafts.acceptRevision(nextChatId, next.dataset.draftRevision);
        if (missingEditedMessage) this.#notices.announce('This message was deleted. Your local edit is preserved; copy it before cancelling.', true);
        this.#composer.reconcileViewport();
        this.#composer.restoreHeight();
        this.#composer.autoSizeInput(renderedInput);
        if (firstPreview && (followLatest || stickToBottom)) {
            this.#messages.restoreScroll(currentMessages, messageScrollTop);
            window.requestAnimationFrame(() => this.#messages.scrollStreamingMessageUntilTop(currentMessages, currentMessages.querySelector('article.is-streaming')));
        } else if (!sameChat) {
            if (!this.#messages.restoreScrollAfterNavigation()) this.#messages.snapListToBottom(currentMessages);
        } else if (followLatest || stickToBottom) this.#messages.snapListToBottom(currentMessages);
        else {
            this.#messages.restoreScroll(
                currentMessages,
                messageScrollTop,
                streamingMessageAnchor,
            );
        }
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
                if (response.status === 404) {
                    if (this.#drafts.isDirty || current.querySelector('[data-composer-mode=edit]')) this.#notices.announce('This chat was deleted. Your local text is preserved; copy it before leaving.', true);
                    else window.location.assign('/chats');
                }
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
