import { CharacterEditorController } from './character-editor.js';
import { ChatController } from './chat.js';
import { ComposerController, currentComposerDraft } from './composer.js';
import { DeviceActivityController } from './device-activity.js';
import { DraftController } from './drafts.js';
import { byId, localizeMessageUpdatedTimes } from './dom.js';
import { addCsrfToken } from './forms.js';
import { HostStatsController } from './host-stats.js';
import { MessageController } from './messages.js';
import { NoticeController, RETRY_DRAFT_NOTICE_ACTION } from './notices.js';
import { PanelController } from './panels.js';

const DRAFT_FLUSH_TIMEOUT_MS = 650;
const DEVICE_COOKIE_NAME = 'jouzetsu_device_id';
const DEVICE_ID_PATTERN = /^dvc_[A-Za-z0-9_-]{8,160}$/;

const bootstrapDeviceCookie = () => {
    const bootstrap = document.querySelector('[data-device-cookie-bootstrap="true"]');
    if (!(bootstrap instanceof HTMLElement)) return;
    const maxAgeSeconds = Number(bootstrap.dataset.deviceCookieMaxAge || '0');
    if (!Number.isInteger(maxAgeSeconds) || maxAgeSeconds < 1) return;
    const readCookie = () => {
        const prefix = `${DEVICE_COOKIE_NAME}=`;
        const match = document.cookie.split(';').map((part) => part.trim()).find((part) => part.startsWith(prefix));
        return match ? decodeURIComponent(match.slice(prefix.length)) : '';
    };
    const createDeviceId = () => {
        if (window.crypto?.randomUUID) return `dvc_${window.crypto.randomUUID()}`;
        const random = new Uint32Array(4);
        window.crypto?.getRandomValues(random);
        return `dvc_${Array.from(random, (value) => value.toString(16)).join('_')}`;
    };
    let deviceId = '';
    try {
        deviceId = window.localStorage.getItem(DEVICE_COOKIE_NAME) || readCookie();
    } catch {
        deviceId = readCookie();
    }
    if (!DEVICE_ID_PATTERN.test(deviceId)) {
        deviceId = createDeviceId();
        try {
            window.localStorage.setItem(DEVICE_COOKIE_NAME, deviceId);
        } catch {
            // Cookie persistence remains available when local storage is blocked.
        }
    }
    if (readCookie() !== deviceId) {
        document.cookie = `${DEVICE_COOKIE_NAME}=${encodeURIComponent(deviceId)}; Path=/; SameSite=Lax; Max-Age=${maxAgeSeconds}`;
        window.setTimeout(() => window.location.reload(), 50);
    }
};

const installComposerResize = (composer, event, controller) => {
    if (!(composer instanceof HTMLElement)) return;
    const handle = event.target instanceof Element ? event.target.closest('.jouzetsu-composer-resize-handle') : null;
    if (!(handle instanceof HTMLElement)) return;
    const pointerId = event.pointerId;
    const startY = event.clientY;
    const startHeight = composer.getBoundingClientRect().height;
    const move = (moveEvent) => {
        if (moveEvent.pointerId === pointerId) controller.setHeight(composer, startHeight + startY - moveEvent.clientY);
    };
    const finish = (finishEvent) => {
        if (finishEvent.pointerId !== pointerId) return;
        handle.removeEventListener('pointermove', move);
        handle.removeEventListener('pointerup', finish);
        handle.removeEventListener('pointercancel', finish);
        if (handle.hasPointerCapture(pointerId)) handle.releasePointerCapture(pointerId);
    };
    event.preventDefault();
    handle.setPointerCapture(pointerId);
    handle.addEventListener('pointermove', move);
    handle.addEventListener('pointerup', finish);
    handle.addEventListener('pointercancel', finish);
};

const openDeleteDialog = (target) => {
    const messageId = target.dataset.deleteChoiceOpen || '';
    const dialog = byId('message-delete-dialog');
    if (!messageId || !(dialog instanceof HTMLDialogElement) || dialog.open) return;
    const single = dialog.querySelector('[data-delete-choice-form="single"]');
    const following = dialog.querySelector('[data-delete-choice-form="following"]');
    if (!(single instanceof HTMLFormElement) || !(following instanceof HTMLFormElement)) return;
    const encodedId = encodeURIComponent(messageId);
    single.action = `/messages/${encodedId}/delete`;
    following.action = `/messages/${encodedId}/delete-following`;
    dialog.showModal();
};

export const startApplication = () => {
    const notices = new NoticeController();
    const composer = new ComposerController();
    const characters = new CharacterEditorController();
    const messages = new MessageController();
    const drafts = new DraftController(currentComposerDraft, notices.announce.bind(notices));
    const deviceActivity = new DeviceActivityController();
    const panels = new PanelController();
    const hostStats = new HostStatsController();
    const chat = new ChatController({ composer, messages, drafts, notices, panels });
    panels.setRefreshPanel(chat.replacePanelFragment.bind(chat));

    document.addEventListener('click', (event) => {
        const target = event.target instanceof Element ? event.target : null;
        if (!target) return;
        const tab = target.closest('[data-tab-target]');
        if (tab instanceof HTMLButtonElement) {
            event.preventDefault();
            panels.activateTab(tab);
            return;
        }
        if (characters.handleClick(target)) {
            event.preventDefault();
            return;
        }
        const noticeAction = target.closest('[data-notice-action]');
        if (noticeAction?.dataset.noticeAction === RETRY_DRAFT_NOTICE_ACTION) {
            event.preventDefault();
            notices.dismiss();
            void drafts.flush();
            return;
        }
        if (noticeAction) return;
        if (target.closest('[data-notice-dismiss]')) {
            event.preventDefault();
            notices.dismiss();
            panels.removeTransientParameters();
            return;
        }
        const panelTarget = target.closest('[data-panel-open]');
        if (panelTarget instanceof HTMLElement) {
            panels.open(panelTarget.dataset.panelOpen || '', panelTarget);
            return;
        }
        if (target.closest('[data-panel-close]') || target.matches('[data-panel-backdrop]')) {
            panels.close();
            return;
        }
        const dialogTarget = target.closest('[data-dialog-open]');
        if (dialogTarget instanceof HTMLElement) {
            const dialog = byId(dialogTarget.dataset.dialogOpen || '');
            if (dialog instanceof HTMLDialogElement && !dialog.open) {
                panels.close(false);
                dialog.showModal();
                hostStats.start();
            }
            return;
        }
        const dialogClose = target.closest('[data-dialog-close]');
        if (dialogClose instanceof HTMLElement) {
            const dialog = byId(dialogClose.dataset.dialogClose || '');
            if (dialog instanceof HTMLDialogElement) panels.closeDialog(dialog);
            return;
        }
        if (target.closest('[data-message-copy]')) {
            event.preventDefault();
            void messages.copySelectedMessage().then((copied) => {
                notices.announce(copied ? 'Message copied' : 'Could not copy message', !copied);
            });
            return;
        }
        if (target.closest('[data-message-details-open]')) {
            event.preventDefault();
            messages.openDetails();
            return;
        }
        const contextMenu = byId('message-context-menu');
        if (contextMenu instanceof HTMLElement && !contextMenu.contains(target)) messages.closeContextMenu();
        if (target.closest('[data-host-stats-refresh]')) {
            event.preventDefault();
            void hostStats.refresh(true);
            return;
        }
        const suggestion = target.closest('[data-suggestion]');
        if (suggestion instanceof HTMLElement) {
            const input = document.querySelector('[data-draft-sync]');
            if (!(input instanceof HTMLTextAreaElement)) return;
            const separator = !input.value || /[\s]$/.test(input.value) ? '' : '\n';
            input.value = `${input.value}${separator}${suggestion.dataset.suggestion || ''}`;
            input.dispatchEvent(new Event('input', { bubbles: true }));
            input.focus();
            return;
        }
        const deleteChoice = target.closest('[data-delete-choice-open]');
        if (deleteChoice instanceof HTMLElement) {
            openDeleteDialog(deleteChoice);
            return;
        }
        const link = target.closest('a[href]');
        if (link instanceof HTMLAnchorElement && link.dataset.chatNavigation === 'true') {
            event.preventDefault();
            void chat.navigate(link.href);
            return;
        }
        if (!event.defaultPrevented && link instanceof HTMLAnchorElement && link.closest('#chat-fragment')) {
            messages.saveScrollForNavigation();
        }
    });

    document.addEventListener('submit', (event) => {
        const form = event.target;
        if (!(form instanceof HTMLFormElement)) return;
        if (!characters.validateForm(form)) {
            event.preventDefault();
            return;
        }
        const submitter = event.submitter;
        const confirmation = submitter instanceof HTMLElement ? submitter.dataset.confirm : '';
        if (confirmation && !window.confirm(confirmation)) {
            event.preventDefault();
            return;
        }
        addCsrfToken(form);
        characters.beginSubmit(form);
        const composerInput = form.querySelector('.jouzetsu-message-input');
        if (
            composerInput instanceof HTMLTextAreaElement
            && (form.action.endsWith('/chat/send') || form.action.endsWith('/messages/edit'))
        ) composer.resetHeight(composerInput.closest('.jouzetsu-composer'));
        if (form.dataset.liveSubmit === 'true') {
            event.preventDefault();
            if (form.dataset.liveSaving !== 'true') void chat.submitLiveForm(form);
            return;
        }
        if (form.dataset.chatMutation === 'true') {
            event.preventDefault();
            void chat.submitMutation(form);
            return;
        }
        messages.saveScrollForNavigation();
        if (form.dataset.draftFlushed === 'true') return;
        const draft = currentComposerDraft();
        if (!draft || !drafts.isDirty || form.action.endsWith('/chat/draft') || form.action.endsWith('/chat/stop')) return;
        event.preventDefault();
        if (form.dataset.draftFlushing === 'true') return;
        form.dataset.draftFlushing = 'true';
        drafts.deferPageHide();
        const fallback = new Promise((resolve) => window.setTimeout(resolve, DRAFT_FLUSH_TIMEOUT_MS));
        void Promise.race([drafts.flush(), fallback]).finally(() => {
            form.dataset.draftFlushed = 'true';
            form.submit();
        });
    });

    document.addEventListener('input', (event) => {
        const input = event.target;
        if (!(input instanceof Element)) return;
        characters.handleInput(input);
        if (!(input instanceof HTMLTextAreaElement)) return;
        composer.autoSizeInput(input);
        if (input.dataset.composerInput === 'true') composer.syncPrimary();
        if (input.dataset.draftSync === 'true') drafts.handleInput();
    });

    document.addEventListener('change', (event) => {
        const control = event.target;
        if (!(control instanceof Element)) return;
        characters.handleChange(control);
        const autosaveControl = control.closest('[data-live-autosave]');
        if (!(autosaveControl instanceof HTMLElement) || autosaveControl.matches(':disabled')) return;
        const form = autosaveControl.closest('form');
        if (form instanceof HTMLFormElement && form.dataset.liveSubmit === 'true') form.requestSubmit();
    });

    document.addEventListener('pointerdown', (event) => {
        messages.handlePointerDown(event);
        const composerElement = event.target instanceof Element ? event.target.closest('.jouzetsu-composer') : null;
        installComposerResize(composerElement, event, composer);
    });
    document.addEventListener('click', (event) => messages.handleMessageClick(event));
    document.addEventListener('dblclick', (event) => {
        messages.handleDoubleClick(event);
        if (event.defaultPrevented) return;
        const handle = event.target instanceof Element ? event.target.closest('.jouzetsu-composer-resize-handle') : null;
        composer.resetHeight(handle?.closest('.jouzetsu-composer'));
    });
    document.addEventListener('touchstart', (event) => messages.handleTouchStart(event));
    document.addEventListener('touchcancel', () => messages.handleTouchCancel());
    document.addEventListener('touchend', (event) => messages.handleTouchEnd(event));
    document.addEventListener('contextmenu', (event) => messages.handleContextMenu(event));
    window.addEventListener('resize', () => messages.closeContextMenu());
    document.addEventListener('scroll', () => messages.closeContextMenu(), true);

    document.addEventListener('keydown', (event) => {
        if (characters.handleKeyDown(event)) return;
        if (event.key === 'Escape' && messages.closeContextMenu()) {
            event.preventDefault();
            return;
        }
        if (event.key === 'Escape' && document.documentElement.classList.contains('jouzetsu-panel-open')) {
            event.preventDefault();
            panels.close();
            return;
        }
        if (panels.handleTabKeyDown(event)) return;
        const messageInput = event.target;
        if (
            messageInput instanceof HTMLTextAreaElement
            && messageInput.matches('.jouzetsu-message-input')
            && event.key === 'Enter'
            && (event.shiftKey || event.metaKey)
            && !event.altKey
            && !event.isComposing
        ) {
            const form = messageInput.closest('form');
            const primary = form?.querySelector('[data-composer-primary]');
            if (
                form instanceof HTMLFormElement
                && form.dataset.composerGenerating !== 'true'
                && primary instanceof HTMLButtonElement
                && !primary.disabled
            ) {
                event.preventDefault();
                form.requestSubmit(primary);
            }
            return;
        }
        const handle = event.target instanceof Element ? event.target.closest('.jouzetsu-composer-resize-handle') : null;
        const composerElement = handle?.closest('.jouzetsu-composer');
        if (!(composerElement instanceof HTMLElement)) return;
        const currentHeight = composerElement.getBoundingClientRect().height;
        const step = event.shiftKey ? 64 : 24;
        if (event.key === 'ArrowUp') {
            event.preventDefault();
            composer.setHeight(composerElement, currentHeight + step);
        } else if (event.key === 'ArrowDown') {
            event.preventDefault();
            composer.setHeight(composerElement, currentHeight - step);
        } else if (event.key === 'Home') {
            event.preventDefault();
            composer.setHeight(composerElement, composer.minimumHeight(composerElement));
        } else if (event.key === 'End') {
            event.preventDefault();
            composer.setHeight(composerElement, composer.maximumHeight());
        } else if (event.key === 'Escape') {
            event.preventDefault();
            composer.resetHeight(composerElement);
        }
    });

    document.querySelectorAll('dialog').forEach((dialog) => {
        if (!(dialog instanceof HTMLDialogElement)) return;
        dialog.addEventListener('click', (event) => {
            const bounds = dialog.getBoundingClientRect();
            if (
                event.clientX < bounds.left
                || event.clientX > bounds.right
                || event.clientY < bounds.top
                || event.clientY > bounds.bottom
            ) dialog.close();
        });
        dialog.addEventListener('cancel', () => panels.removeTransientParameters());
        dialog.addEventListener('close', () => {
            if (dialog.id === 'message-details-dialog') messages.clearDetailsRequest();
            panels.removeTransientParameters();
            const feedback = dialog.querySelector('.jouzetsu-dialog-feedback');
            if (feedback instanceof HTMLElement && feedback.textContent) {
                notices.announce(feedback.textContent, feedback.classList.contains('is-error'));
            }
        });
    });

    composer.reconcileViewport();
    characters.initialize();
    composer.restoreHeight();
    localizeMessageUpdatedTimes();
    composer.autoSizeInput(document.querySelector('.jouzetsu-message-input'));
    composer.syncPrimary();
    composer.syncStatusElapsedTimer();
    const initialForm = document.querySelector('.jouzetsu-composer-form');
    const initialInput = initialForm?.querySelector('.jouzetsu-message-input');
    if (initialForm instanceof HTMLFormElement && initialForm.dataset.composerMode === 'edit') composer.focusInput(initialInput);
    if (!messages.restoreScrollAfterNavigation()) messages.scrollListToBottom(byId('message-list'));

    window.addEventListener('resize', () => composer.scheduleViewportReconciliation());
    window.addEventListener('resize', () => characters.handleViewportChange());
    window.visualViewport?.addEventListener('resize', () => composer.scheduleViewportReconciliation());
    window.visualViewport?.addEventListener('scroll', () => composer.scheduleViewportReconciliation());
    window.addEventListener('popstate', () => void chat.handlePopstate());
    window.addEventListener('pagehide', () => drafts.persistOnPageHide());

    bootstrapDeviceCookie();
    deviceActivity.start();
    const requestedDialogId = document.querySelector('[data-open-dialog]')?.getAttribute('data-open-dialog');
    if (requestedDialogId) {
        const dialog = byId(requestedDialogId);
        if (dialog instanceof HTMLDialogElement && !dialog.open) dialog.showModal();
    }
    notices.scheduleInitialDismissal();
    panels.removeTransientParameters();
    hostStats.start();

    if ('EventSource' in window && !document.querySelector('[data-testid="access-locked-page"]')) {
        const source = new EventSource('/events');
        source.addEventListener('state', (event) => {
            if (!(event instanceof MessageEvent)) {
                chat.scheduleFullRefresh();
                return;
            }
            try {
                const payload = JSON.parse(event.data);
                if (payload?.characters_changed === true && characters.refreshIfNeeded(notices.announce.bind(notices))) return;
                if (payload?.kind === 'full') {
                    chat.scheduleFullRefresh();
                    void chat.refreshPanels();
                } else if (payload?.kind === 'characters') {
                    return;
                } else if (payload?.kind === 'stream' || payload?.kind === 'status') {
                    chat.scheduleLiveRefresh();
                } else {
                    chat.scheduleFullRefresh();
                }
            } catch {
                chat.scheduleFullRefresh();
            }
        });
        window.addEventListener('beforeunload', () => source.close(), { once: true });
    }
};
