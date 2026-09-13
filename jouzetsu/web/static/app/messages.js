import { byId, localizeMessageDetailsTimes, parseFragment } from './dom.js';

const DOUBLE_TAP_DELAY_MS = 350;
const DOUBLE_SWIPE_DELAY_MS = 1000;
const SWIPE_MIN_DISTANCE_PX = 40;
const SWIPE_MAX_HORIZONTAL_DISTANCE_PX = 72;
const CONTINUE_SWIPE_BOTTOM_THRESHOLD_PX = 160;
const BOTTOM_THRESHOLD_PX = 24;
const MESSAGE_SCROLL_STATE_KEY = 'jouzetsu-message-scroll-v1';

export class MessageController {
    #lastTouchedId = '';
    #lastTouchAt = 0;
    #touchCount = 0;
    #pendingScrollTimer = 0;
    #pendingScrollTarget = null;
    #swipeStartId = '';
    #swipeStartX = 0;
    #swipeStartY = 0;
    #lastSwipedId = '';
    #lastSwipeAt = 0;
    #continueArmTimer = 0;
    #armedContinueControl = null;
    #streamingScrollId = '';
    #streamingScrollReachedMessageTop = false;
    #scrollContainer = null;
    #scrollContainerPinned = false;
    #scrollContainerObserver = null;
    #bottomSnapFrame = 0;
    #scrollRestoreFrame = 0;
    #detailsRequest = 0;

    #handleScrollContainerScroll = () => {
        this.#recordScrollContainerPosition(this.#scrollContainer);
    };

    watchScrollContainer() {
        const messages = byId('message-list');
        if (messages === this.#scrollContainer) return;
        if (this.#scrollContainer instanceof HTMLElement) {
            this.#scrollContainer.removeEventListener('scroll', this.#handleScrollContainerScroll);
        }
        this.#scrollContainerObserver?.disconnect();
        this.#scrollContainerObserver = null;
        this.#scrollContainer = messages instanceof HTMLElement ? messages : null;
        this.#recordScrollContainerPosition(this.#scrollContainer);
        if (!(messages instanceof HTMLElement)) return;
        messages.addEventListener('scroll', this.#handleScrollContainerScroll, { passive: true });
        const ResizeObserverConstructor = window.ResizeObserver;
        if (typeof ResizeObserverConstructor !== 'function') return;
        this.#scrollContainerObserver = new ResizeObserverConstructor(() => {
            if (this.#scrollContainerPinned && this.#scrollContainer === messages) {
                this.scrollListToBottom(messages);
            }
        });
        this.#scrollContainerObserver.observe(messages);
    }

    isNearBottom(messages, threshold = BOTTOM_THRESHOLD_PX) {
        return messages instanceof HTMLElement
            && messages.scrollHeight - messages.clientHeight - messages.scrollTop <= threshold;
    }

    captureStreamingMessageAnchor(messages) {
        const message = this.#streamingMessage(messages);
        if (!message) return null;
        return {
            messageId: message.dataset.messageId,
            relativeTop: this.#relativeMessageTop(messages, message),
        };
    }

    scrollListToBottom(messages) {
        this.#cancelScrollRestore();
        if (!(messages instanceof HTMLElement)) return;
        messages.scrollTop = messages.scrollHeight;
        this.#recordScrollContainerPosition(messages);
    }

    snapListToBottom(messages) {
        this.#cancelBottomSnap();
        this.scrollListToBottom(messages);
        if (!(messages instanceof HTMLElement)) return;
        this.#bottomSnapFrame = window.requestAnimationFrame(() => {
            this.#bottomSnapFrame = 0;
            if (!messages.isConnected) return;
            this.scrollListToBottom(messages);
        });
    }

    restoreScroll(messages, scrollTop, messageAnchor = null) {
        this.#cancelBottomSnap();
        this.#cancelScrollRestore();
        if (!(messages instanceof HTMLElement) || scrollTop === null) return;
        const suppressBottomPinning = messageAnchor !== null;
        const restored = this.#restoreScrollPosition(messages, scrollTop, messageAnchor);
        this.#recordScrollContainerPosition(messages, suppressBottomPinning);
        if (!restored && !messageAnchor) return;
        this.#scrollRestoreFrame = window.requestAnimationFrame(() => {
            this.#scrollRestoreFrame = 0;
            if (!messages.isConnected) return;
            this.#restoreScrollPosition(messages, scrollTop, messageAnchor);
            this.#recordScrollContainerPosition(messages, suppressBottomPinning);
        });
    }

    saveScrollForNavigation() {
        const messages = byId('message-list');
        const form = document.querySelector('.jouzetsu-composer-form');
        if (!(messages instanceof HTMLElement) || !(form instanceof HTMLFormElement) || !form.dataset.chatId) return;
        try {
            window.sessionStorage.setItem(
                MESSAGE_SCROLL_STATE_KEY,
                JSON.stringify({ chatId: form.dataset.chatId, scrollTop: messages.scrollTop }),
            );
        } catch {
            // Navigation is still safe if browser storage is unavailable.
        }
    }

    restoreScrollAfterNavigation() {
        let savedState = null;
        try {
            const serialized = window.sessionStorage.getItem(MESSAGE_SCROLL_STATE_KEY);
            window.sessionStorage.removeItem(MESSAGE_SCROLL_STATE_KEY);
            if (serialized) savedState = JSON.parse(serialized);
        } catch {
            return false;
        }
        if (
            !savedState
            || typeof savedState !== 'object'
            || typeof savedState.chatId !== 'string'
            || !Number.isFinite(savedState.scrollTop)
        ) return false;
        const form = document.querySelector('.jouzetsu-composer-form');
        const messages = byId('message-list');
        if (!(form instanceof HTMLFormElement) || form.dataset.chatId !== savedState.chatId) return false;
        this.restoreScroll(messages, savedState.scrollTop);
        return true;
    }

    syncStreamingContent(message, content) {
        const text = content.dataset.liveStreamingText ?? '';
        if (!message.dataset.streamingText && message.querySelector('[data-streaming-pending]')) message.replaceChildren();
        if (message.innerHTML !== content.innerHTML) {
            message.replaceChildren(...Array.from(content.childNodes, (child) => child.cloneNode(true)));
        }
        message.dataset.streamingText = text;
    }

    scrollStreamingMessageUntilTop(messages, message) {
        if (!(messages instanceof HTMLElement) || !(message instanceof HTMLElement)) return;
        const messageId = message.dataset.messageId || '';
        if (!messageId) return;
        if (this.#streamingScrollId !== messageId) {
            this.#streamingScrollId = messageId;
            this.#streamingScrollReachedMessageTop = false;
        }
        if (this.#streamingScrollReachedMessageTop) return;
        const desiredScrollTop = messages.scrollHeight - messages.clientHeight;
        if (desiredScrollTop <= messages.scrollTop) return;
        const messageTop = message.getBoundingClientRect().top;
        const messageListTop = messages.getBoundingClientRect().top;
        const scrollRoomBeforeMessageTop = Math.max(0, messageTop - messageListTop);
        const nextScrollTop = Math.min(desiredScrollTop, messages.scrollTop + scrollRoomBeforeMessageTop);
        this.#cancelBottomSnap();
        this.#cancelScrollRestore();
        messages.scrollTop = nextScrollTop;
        if (nextScrollTop < desiredScrollTop) this.#streamingScrollReachedMessageTop = true;
        this.#recordScrollContainerPosition(messages);
    }

    closeContextMenu() {
        const menu = byId('message-context-menu');
        if (!(menu instanceof HTMLElement) || menu.hidden) return false;
        menu.hidden = true;
        return true;
    }

    openContextMenu(message, clientX, clientY) {
        const menu = byId('message-context-menu');
        if (!(menu instanceof HTMLElement)) return;
        const messageId = message.dataset.messageId || '';
        menu.dataset.messageId = messageId;
        this.#syncContextMenuForkAction(menu, messageId);
        const bounds = message.getBoundingClientRect();
        const defaultX = bounds.left + Math.min(bounds.width / 2, 24);
        const defaultY = bounds.top + Math.min(bounds.height / 2, 24);
        menu.hidden = false;
        const menuBounds = menu.getBoundingClientRect();
        const margin = 8;
        menu.style.left = `${Math.max(margin, Math.min(clientX || defaultX, window.innerWidth - menuBounds.width - margin))}px`;
        menu.style.top = `${Math.max(margin, Math.min(clientY || defaultY, window.innerHeight - menuBounds.height - margin))}px`;
        const action = menu.querySelector('[data-message-copy]');
        if (action instanceof HTMLButtonElement) action.focus({ preventScroll: true });
    }

    openDetails() {
        const menu = byId('message-context-menu');
        const dialog = byId('message-details-dialog');
        if (!(menu instanceof HTMLElement) || !(dialog instanceof HTMLDialogElement) || dialog.open) return;
        const messageId = menu.dataset.messageId || '';
        if (!messageId) return;
        this.closeContextMenu();
        dialog.showModal();
        void this.#loadDetails(dialog, messageId);
    }

    async copySelectedMessage() {
        const menu = byId('message-context-menu');
        if (!(menu instanceof HTMLElement)) return false;
        const message = this.#messageById(menu.dataset.messageId || '');
        const content = message?.querySelector('.jouzetsu-message-text');
        this.closeContextMenu();
        if (!(content instanceof HTMLElement)) return false;
        return this.#copyText(content.innerText);
    }

    refreshOpenDetails() {
        const dialog = byId('message-details-dialog');
        if (!(dialog instanceof HTMLDialogElement) || !dialog.open) return;
        const messageId = dialog.dataset.messageDetailsMessageId || '';
        if (!messageId) {
            dialog.close();
            return;
        }
        void this.#loadDetails(dialog, messageId);
    }

    clearDetailsRequest() {
        this.#detailsRequest += 1;
        const dialog = byId('message-details-dialog');
        if (dialog instanceof HTMLDialogElement) delete dialog.dataset.messageDetailsMessageId;
    }

    handleContextMenu(event) {
        const message = this.#messageForTarget(event.target);
        if (!message) return;
        event.preventDefault();
        this.openContextMenu(message, event.clientX, event.clientY);
    }

    handlePointerDown(event) {
        const menu = byId('message-context-menu');
        if (menu instanceof HTMLElement && !menu.contains(event.target)) this.closeContextMenu();
    }

    handleMessageClick(event) {
        const message = this.#messageForTarget(event.target);
        if (message && event.detail === 2) {
            event.preventDefault();
            this.#queueTopScroll(message);
        } else if (message && event.detail === 3) {
            event.preventDefault();
            this.#cancelPendingScroll();
            this.#scrollMessageTo(message, 'end');
        }
    }

    handleDoubleClick(event) {
        if (this.#messageForTarget(event.target)) event.preventDefault();
    }

    handleTouchStart(event) {
        if (event.touches.length !== 1) {
            this.#clearSwipeStart();
            this.#clearSwipeSequence();
            return;
        }
        const message = this.#messageForTarget(event.target);
        const messages = byId('message-list');
        const touch = event.touches.item(0);
        if (!(message instanceof HTMLElement) || !(messages instanceof HTMLElement) || !touch) {
            this.#clearSwipeStart();
            this.#clearSwipeSequence();
            return;
        }
        const messageId = message.dataset.messageId || '';
        const now = performance.now();
        const continuingSwipe = messageId === this.#lastSwipedId && now - this.#lastSwipeAt <= DOUBLE_SWIPE_DELAY_MS;
        if (
            !messageId
            || !this.#isLastMessage(message, messages)
            || !this.#continueControl(message)
            || (!this.isNearBottom(messages, CONTINUE_SWIPE_BOTTOM_THRESHOLD_PX) && !continuingSwipe)
        ) {
            this.#clearSwipeStart();
            this.#clearSwipeSequence();
            return;
        }
        this.#swipeStartId = messageId;
        this.#swipeStartX = touch.clientX;
        this.#swipeStartY = touch.clientY;
    }

    handleTouchCancel() {
        this.#clearSwipeStart();
    }

    handleTouchEnd(event) {
        if (event.touches.length) return;
        const message = this.#messageForTarget(event.target);
        const touch = event.changedTouches.item(0);
        const messageId = message?.dataset.messageId || '';
        const verticalDistance = touch ? this.#swipeStartY - touch.clientY : 0;
        const horizontalDistance = touch ? Math.abs(this.#swipeStartX - touch.clientX) : 0;
        const isUpwardSwipe = (
            message instanceof HTMLElement
            && messageId === this.#swipeStartId
            && verticalDistance >= SWIPE_MIN_DISTANCE_PX
            && horizontalDistance <= SWIPE_MAX_HORIZONTAL_DISTANCE_PX
        );
        this.#clearSwipeStart();
        if (isUpwardSwipe) {
            this.#lastTouchedId = '';
            this.#lastTouchAt = 0;
            this.#touchCount = 0;
            this.#cancelPendingScroll();
            const now = performance.now();
            const completesDoubleSwipe = messageId === this.#lastSwipedId && now - this.#lastSwipeAt <= DOUBLE_SWIPE_DELAY_MS;
            if (completesDoubleSwipe) {
                this.#clearSwipeSequence();
                if (this.#invokeContinue(message)) event.preventDefault();
                return;
            }
            this.#lastSwipedId = messageId;
            this.#lastSwipeAt = now;
            this.#armContinueSwipe(this.#continueControl(message));
            return;
        }
        this.#clearSwipeSequence();
        if (!message) return;
        const now = performance.now();
        const isContinuation = messageId === this.#lastTouchedId && now - this.#lastTouchAt <= DOUBLE_TAP_DELAY_MS;
        this.#touchCount = isContinuation ? this.#touchCount + 1 : 1;
        this.#lastTouchedId = messageId;
        this.#lastTouchAt = now;
        if (this.#touchCount === 1) return;
        event.preventDefault();
        if (this.#touchCount === 2) {
            this.#queueTopScroll(message);
            return;
        }
        this.#lastTouchedId = '';
        this.#lastTouchAt = 0;
        this.#touchCount = 0;
        this.#cancelPendingScroll();
        this.#scrollMessageTo(message, 'end');
    }

    #messageForTarget(target) {
        if (!(target instanceof Element) || target.closest('button, a, input, textarea, select, label')) return null;
        const message = target.closest('[data-message-id]');
        return message instanceof HTMLElement ? message : null;
    }

    #messageById(messageId) {
        if (!messageId) return null;
        for (const message of document.querySelectorAll('[data-message-id]')) {
            if (message instanceof HTMLElement && message.dataset.messageId === messageId) return message;
        }
        return null;
    }

    #scrollMessageTo(message, block) {
        this.#cancelBottomSnap();
        this.#cancelScrollRestore();
        if (this.#scrollContainer?.contains(message)) this.#scrollContainerPinned = false;
        const reducedMotion = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;
        message.scrollIntoView({ behavior: reducedMotion ? 'auto' : 'smooth', block, inline: 'nearest' });
    }

    #detailsContent(dialog) {
        const content = dialog.querySelector('#message-details-content');
        return content instanceof HTMLElement ? content : null;
    }

    #setDetailsStatus(dialog, message) {
        const content = this.#detailsContent(dialog);
        if (!content) return;
        const status = document.createElement('p');
        status.className = 'jouzetsu-message-details-status';
        status.textContent = message;
        content.replaceChildren(status);
    }

    #syncContextMenuForkAction(menu, messageId) {
        const form = menu.querySelector('[data-message-context-fork-form]');
        const button = form?.querySelector('button');
        if (!(form instanceof HTMLFormElement) || !(button instanceof HTMLButtonElement)) return;
        const composer = document.querySelector('.jouzetsu-composer-form');
        const isGenerating = composer instanceof HTMLFormElement && composer.dataset.composerGenerating === 'true';
        const canFork = Boolean(messageId) && !isGenerating;
        form.action = canFork ? `/messages/${encodeURIComponent(messageId)}/fork` : '';
        form.hidden = !canFork;
        button.disabled = !canFork;
    }

    async #copyText(text) {
        if (navigator.clipboard?.writeText) {
            try {
                await navigator.clipboard.writeText(text);
                return true;
            } catch {
                // Local HTTP instances may not expose the modern Clipboard API.
            }
        }
        if (!document.body) return false;
        const textarea = document.createElement('textarea');
        textarea.value = text;
        textarea.setAttribute('readonly', '');
        textarea.setAttribute('aria-hidden', 'true');
        textarea.style.position = 'fixed';
        textarea.style.opacity = '0';
        document.body.append(textarea);
        try {
            textarea.focus({ preventScroll: true });
            textarea.select();
            return document.execCommand('copy');
        } catch {
            return false;
        } finally {
            textarea.remove();
        }
    }

    async #loadDetails(dialog, messageId) {
        const request = this.#detailsRequest + 1;
        this.#detailsRequest = request;
        dialog.dataset.messageDetailsMessageId = messageId;
        this.#setDetailsStatus(dialog, 'Loading message details…');
        try {
            const response = await window.fetch(`/fragments/messages/${encodeURIComponent(messageId)}/details`, {
                headers: { 'HX-Request': 'true' },
                cache: 'no-store',
            });
            if (response.status === 403) {
                window.location.reload();
                return;
            }
            if (!response.ok) throw new Error(`Message details request failed: ${response.status}`);
            const markup = await response.text();
            if (request !== this.#detailsRequest || !dialog.open || dialog.dataset.messageDetailsMessageId !== messageId) return;
            const next = parseFragment(markup, 'message-details-content');
            if (!next) throw new Error('Message details fragment was malformed');
            const current = this.#detailsContent(dialog);
            if (!current) return;
            current.replaceWith(next);
            localizeMessageDetailsTimes(next);
        } catch {
            if (request === this.#detailsRequest && dialog.open && dialog.dataset.messageDetailsMessageId === messageId) {
                this.#setDetailsStatus(dialog, 'Message details are no longer available.');
            }
        }
    }

    #continueControl(message) {
        const control = message.querySelector('[data-message-action="continue"]');
        return control instanceof HTMLButtonElement && !control.disabled ? control : null;
    }

    #isLastMessage(message, messages) {
        const elements = messages.querySelectorAll('[data-message-id]');
        return elements.item(elements.length - 1) === message;
    }

    #clearSwipeStart() {
        this.#swipeStartId = '';
        this.#swipeStartX = 0;
        this.#swipeStartY = 0;
    }

    #clearContinueArm() {
        if (this.#continueArmTimer) window.clearTimeout(this.#continueArmTimer);
        this.#continueArmTimer = 0;
        if (this.#armedContinueControl instanceof HTMLElement) this.#armedContinueControl.classList.remove('is-swipe-armed');
        this.#armedContinueControl = null;
    }

    #armContinueSwipe(control) {
        this.#clearContinueArm();
        if (!(control instanceof HTMLButtonElement)) return;
        this.#armedContinueControl = control;
        control.classList.add('is-swipe-armed');
        this.#continueArmTimer = window.setTimeout(() => this.#clearContinueArm(), DOUBLE_SWIPE_DELAY_MS);
    }

    #clearSwipeSequence() {
        this.#lastSwipedId = '';
        this.#lastSwipeAt = 0;
        this.#clearContinueArm();
    }

    #invokeContinue(message) {
        const control = this.#continueControl(message);
        const form = control?.closest('form');
        if (!(control instanceof HTMLButtonElement) || !(form instanceof HTMLFormElement)) return false;
        form.requestSubmit(control);
        return true;
    }

    #cancelPendingScroll() {
        if (this.#pendingScrollTimer) window.clearTimeout(this.#pendingScrollTimer);
        this.#pendingScrollTimer = 0;
        this.#pendingScrollTarget = null;
    }

    #cancelScrollRestore() {
        if (this.#scrollRestoreFrame) window.cancelAnimationFrame(this.#scrollRestoreFrame);
        this.#scrollRestoreFrame = 0;
    }

    #cancelBottomSnap() {
        if (this.#bottomSnapFrame) window.cancelAnimationFrame(this.#bottomSnapFrame);
        this.#bottomSnapFrame = 0;
    }

    #recordScrollContainerPosition(messages, suppressBottomPinning = false) {
        if (messages === this.#scrollContainer) {
            this.#scrollContainerPinned = (
                !suppressBottomPinning
                && !this.#isStreamingMessageTopLocked(messages)
                && this.isNearBottom(messages)
            );
        }
    }

    #isStreamingMessageTopLocked(messages) {
        return this.#streamingMessage(messages) !== null;
    }

    #streamingMessage(messages) {
        if (
            !this.#streamingScrollReachedMessageTop
            || !(messages instanceof HTMLElement)
        ) return null;
        const streamingMessage = messages.querySelector(
            ':scope > .is-streaming[data-message-id]',
        );
        if (
            !(streamingMessage instanceof HTMLElement)
            || streamingMessage.dataset.messageId !== this.#streamingScrollId
        ) return null;
        return streamingMessage;
    }

    #restoreScrollPosition(messages, scrollTop, messageAnchor) {
        if (messageAnchor) {
            const message = this.#messageWithId(messages, messageAnchor.messageId);
            if (message) {
                const relativeTop = this.#relativeMessageTop(messages, message);
                const anchoredScrollTop = messages.scrollTop + relativeTop - messageAnchor.relativeTop;
                if (Math.abs(messages.scrollTop - anchoredScrollTop) < 1) return false;
                messages.scrollTop = anchoredScrollTop;
                return true;
            }
        }
        if (Math.abs(messages.scrollTop - scrollTop) < 1) return false;
        messages.scrollTop = scrollTop;
        return true;
    }

    #messageWithId(messages, messageId) {
        for (const child of messages.children) {
            if (child instanceof HTMLElement && child.dataset.messageId === messageId) return child;
        }
        return null;
    }

    #relativeMessageTop(messages, message) {
        return message.getBoundingClientRect().top - messages.getBoundingClientRect().top;
    }

    #queueTopScroll(message) {
        this.#cancelPendingScroll();
        this.#pendingScrollTarget = message;
        this.#pendingScrollTimer = window.setTimeout(() => {
            if (this.#pendingScrollTarget instanceof HTMLElement) this.#scrollMessageTo(this.#pendingScrollTarget, 'start');
            this.#cancelPendingScroll();
        }, DOUBLE_TAP_DELAY_MS);
    }
}
