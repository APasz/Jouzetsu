import { byId } from './dom.js';

const COMPOSER_STATUS_ELAPSED_INTERVAL_MS = 100;
const COMPOSER_MAX_VIEWPORT_SHARE = 0.78;
const KEYBOARD_DETECTION_THRESHOLD_PX = 80;
const KEYBOARD_BOTTOM_CLEARANCE_PX = 12;

const formatElapsed = (seconds) => `${Math.max(0, seconds).toFixed(1)}s`;

export const currentComposerDraft = () => {
    const form = document.querySelector('.jouzetsu-composer-form');
    const input = form?.querySelector('[data-draft-sync="true"]');
    if (!(form instanceof HTMLFormElement) || !(input instanceof HTMLTextAreaElement)) return null;
    const chatId = form.dataset.chatId || '';
    return chatId ? { chatId, value: input.value } : null;
};

export class ComposerController {
    #elapsedTimer = 0;
    #sessionHeight = null;
    #viewportFrame = 0;

    autoSizeInput(input) {
        if (!(input instanceof HTMLTextAreaElement) || !input.matches('.jouzetsu-message-input')) return;
        const composer = input.closest('.jouzetsu-composer');
        if (!(composer instanceof HTMLElement)) return;
        if (composer.classList.contains('is-resized')) {
            input.style.removeProperty('height');
            return;
        }
        if (!input.value) {
            input.style.removeProperty('height');
            input.style.removeProperty('overflow-y');
            return;
        }
        input.style.height = '0px';
        const maximumHeight = this.#inputMaximumHeight(composer, input);
        const nextHeight = Math.min(input.scrollHeight, maximumHeight);
        input.style.height = `${nextHeight}px`;
        input.style.overflowY = input.scrollHeight > nextHeight ? 'auto' : 'hidden';
    }

    reconcileViewport() {
        this.#syncAppViewportHeight();
        const composer = document.querySelector('.jouzetsu-composer');
        if (!(composer instanceof HTMLElement)) return;
        composer.style.setProperty('--jouzetsu-composer-max-height', `${this.#maximumHeight()}px`);
        const input = composer.querySelector('.jouzetsu-message-input');
        if (input instanceof HTMLTextAreaElement) {
            composer.style.setProperty('--jouzetsu-composer-input-max-height', `${this.#inputMaximumHeight(composer, input)}px`);
        }
        if (!composer.classList.contains('is-resized')) {
            this.autoSizeInput(input);
            return;
        }
        delete composer.dataset.composerMinimumHeight;
        const requestedHeight = Number.parseFloat(composer.dataset.composerRequestedHeight || '');
        if (Number.isFinite(requestedHeight)) this.#applyHeight(composer, requestedHeight);
    }

    scheduleViewportReconciliation() {
        if (this.#viewportFrame) return;
        this.#viewportFrame = window.requestAnimationFrame(() => {
            this.#viewportFrame = 0;
            this.reconcileViewport();
        });
    }

    restoreHeight() {
        const composer = document.querySelector('.jouzetsu-composer');
        if (!(composer instanceof HTMLElement) || !Number.isFinite(this.#sessionHeight)) return;
        composer.dataset.composerRequestedHeight = String(this.#sessionHeight);
        this.#applyHeight(composer, this.#sessionHeight);
    }

    setHeight(composer, height) {
        if (!(composer instanceof HTMLElement)) return;
        const clamped = this.#clampHeight(composer, height);
        composer.dataset.composerRequestedHeight = String(clamped);
        this.#sessionHeight = clamped;
        this.#applyHeight(composer, clamped);
    }

    resetHeight(composer) {
        if (!(composer instanceof HTMLElement)) return;
        composer.style.removeProperty('--jouzetsu-composer-height');
        composer.classList.remove('is-resized');
        delete composer.dataset.composerRequestedHeight;
        delete composer.dataset.composerMinimumHeight;
        this.#sessionHeight = null;
        this.autoSizeInput(composer.querySelector('.jouzetsu-message-input'));
    }

    maximumHeight() {
        return this.#maximumHeight();
    }

    minimumHeight(composer) {
        return this.#minimumHeight(composer);
    }

    syncPrimary() {
        const form = document.querySelector('.jouzetsu-composer-form');
        const input = form?.querySelector('[data-composer-input="true"]');
        const primary = form?.querySelector('[data-composer-primary]');
        if (!(form instanceof HTMLFormElement) || !(input instanceof HTMLTextAreaElement)) return;
        if (primary instanceof HTMLButtonElement) {
            const isGenerating = form.dataset.composerGenerating === 'true';
            primary.disabled = !isGenerating && !input.value.trim();
        }
        if (form.dataset.composerMode === 'edit') this.#syncEditValidation(form, input);
    }

    focusedSelection() {
        const input = document.activeElement;
        if (!(input instanceof HTMLTextAreaElement) || !input.matches('.jouzetsu-message-input')) return null;
        return { start: input.selectionStart, end: input.selectionEnd, direction: input.selectionDirection };
    }

    restoreSelection(input, selection) {
        if (!(input instanceof HTMLTextAreaElement) || !selection) return;
        input.focus({ preventScroll: true });
        input.setSelectionRange(selection.start, selection.end, selection.direction);
    }

    focusInput(input) {
        if (!(input instanceof HTMLTextAreaElement)) return;
        window.requestAnimationFrame(() => {
            if (!input.isConnected) return;
            input.focus({ preventScroll: true });
            input.setSelectionRange(input.value.length, input.value.length);
        });
    }

    syncLiveStatus(currentStatus, nextStatus) {
        if (!(currentStatus instanceof HTMLElement) || !(nextStatus instanceof HTMLElement)) return false;
        if (!this.#syncStatusText(currentStatus, nextStatus, '[data-composer-status-state]')) return false;
        if (!this.#syncStatusText(currentStatus, nextStatus, '[data-composer-status-detail]')) return false;
        const stageSynced = this.#syncStatusDuration(
            currentStatus,
            nextStatus,
            'composerStatusStageElapsedSeconds',
            'composerStatusStageElapsedUpdatedAt',
            '[data-composer-status-stage-elapsed]',
            '[data-composer-status-stage-timer]',
        );
        const overallSynced = this.#syncStatusDuration(
            currentStatus,
            nextStatus,
            'composerStatusOverallElapsedSeconds',
            'composerStatusOverallElapsedUpdatedAt',
            '[data-composer-status-overall-elapsed]',
            '[data-composer-status-overall-timer]',
        );
        if (!stageSynced || !overallSynced) return false;
        this.syncStatusElapsedTimer();
        return true;
    }

    syncGenerationReasoning(currentReasoning, nextReasoning) {
        if (!(currentReasoning instanceof HTMLDetailsElement) || !(nextReasoning instanceof HTMLElement)) return false;
        const content = currentReasoning.querySelector('.jouzetsu-generation-reasoning-content');
        if (!(content instanceof HTMLElement)) return false;
        const reasoning = nextReasoning.dataset.liveGenerationReasoningText || '';
        const visibilityChanged = currentReasoning.hidden === Boolean(reasoning);
        if (content.textContent !== reasoning) content.textContent = reasoning;
        currentReasoning.hidden = !reasoning;
        if (visibilityChanged) this.scheduleViewportReconciliation();
        return true;
    }

    syncStatusElapsedTimer() {
        const trackingElapsed = this.#updateStatusElapsed(byId('composer-status'));
        if (trackingElapsed && !this.#elapsedTimer) {
            this.#elapsedTimer = window.setInterval(() => {
                if (this.#updateStatusElapsed(byId('composer-status'))) return;
                window.clearInterval(this.#elapsedTimer);
                this.#elapsedTimer = 0;
            }, COMPOSER_STATUS_ELAPSED_INTERVAL_MS);
        } else if (!trackingElapsed && this.#elapsedTimer) {
            window.clearInterval(this.#elapsedTimer);
            this.#elapsedTimer = 0;
        }
    }

    #viewportMetrics() {
        const visualViewport = window.visualViewport;
        const layoutViewportHeight = window.innerHeight;
        const visualViewportHeight = visualViewport?.height || layoutViewportHeight;
        const visualViewportBottom = Math.min(
            layoutViewportHeight,
            Math.max(0, (visualViewport?.offsetTop || 0) + visualViewportHeight),
        );
        const keyboardInset = Math.max(0, layoutViewportHeight - visualViewportBottom);
        const keyboardOpen = keyboardInset >= KEYBOARD_DETECTION_THRESHOLD_PX;
        return {
            height: Math.max(0, visualViewportBottom - (keyboardOpen ? KEYBOARD_BOTTOM_CLEARANCE_PX : 0)),
            keyboardOpen,
        };
    }

    #syncAppViewportHeight() {
        const app = document.querySelector('.jouzetsu-app');
        if (!(app instanceof HTMLElement)) return;
        const { height, keyboardOpen } = this.#viewportMetrics();
        if (!Number.isFinite(height) || height <= 0) return;
        app.style.setProperty('--jouzetsu-app-height', `${Math.floor(height)}px`);
        app.classList.toggle('is-keyboard-open', keyboardOpen);
    }

    #maximumHeight() {
        return Math.floor(this.#viewportMetrics().height * COMPOSER_MAX_VIEWPORT_SHARE);
    }

    #inputMaximumHeight(composer, input) {
        const footer = composer.querySelector('.jouzetsu-composer-footer');
        const styles = window.getComputedStyle(composer);
        const verticalPadding = Number.parseFloat(styles.paddingTop) + Number.parseFloat(styles.paddingBottom);
        const footerHeight = footer instanceof HTMLElement ? footer.getBoundingClientRect().height : 0;
        const minimumHeight = Number.parseFloat(window.getComputedStyle(input).minHeight);
        return Math.max(minimumHeight, Math.floor(this.#maximumHeight() - verticalPadding - footerHeight));
    }

    #minimumHeight(composer) {
        const cached = Number.parseFloat(composer.dataset.composerMinimumHeight || '');
        if (Number.isFinite(cached)) return cached;
        const input = composer.querySelector('.jouzetsu-message-input');
        if (!(input instanceof HTMLTextAreaElement)) return 0;
        const hadManualHeight = composer.classList.contains('is-resized');
        const savedComposerHeight = composer.style.getPropertyValue('--jouzetsu-composer-height');
        const savedInputHeight = input.style.height;
        const savedInputOverflow = input.style.overflowY;
        const savedValue = input.value;
        composer.classList.remove('is-resized');
        composer.style.removeProperty('--jouzetsu-composer-height');
        input.value = '';
        input.style.removeProperty('height');
        input.style.removeProperty('overflow-y');
        const minimumHeight = Math.ceil(composer.getBoundingClientRect().height);
        input.value = savedValue;
        input.style.height = savedInputHeight;
        input.style.overflowY = savedInputOverflow;
        if (savedComposerHeight) composer.style.setProperty('--jouzetsu-composer-height', savedComposerHeight);
        if (hadManualHeight) composer.classList.add('is-resized');
        composer.dataset.composerMinimumHeight = String(minimumHeight);
        return minimumHeight;
    }

    #clampHeight(composer, height) {
        return Math.max(this.#minimumHeight(composer), Math.min(height, this.#maximumHeight()));
    }

    #applyHeight(composer, requestedHeight) {
        const clamped = this.#clampHeight(composer, requestedHeight);
        composer.style.setProperty('--jouzetsu-composer-height', `${clamped}px`);
        composer.classList.add('is-resized');
        const input = composer.querySelector('.jouzetsu-message-input');
        if (input instanceof HTMLTextAreaElement) input.style.removeProperty('height');
    }

    #syncEditValidation(form, input) {
        const validation = form.querySelector('[data-composer-validation="true"]');
        if (!(validation instanceof HTMLElement)) return;
        const isEmpty = !input.value.trim();
        input.setAttribute('aria-invalid', String(isEmpty));
        validation.hidden = !isEmpty;
    }

    #updateStatusElapsed(status) {
        if (!(status instanceof HTMLElement)) return false;
        const now = performance.now();
        const trackingStage = this.#updateStatusDuration(
            status,
            'composerStatusStageElapsedSeconds',
            'composerStatusStageElapsedUpdatedAt',
            '[data-composer-status-stage-elapsed]',
            now,
        );
        const trackingOverall = this.#updateStatusDuration(
            status,
            'composerStatusOverallElapsedSeconds',
            'composerStatusOverallElapsedUpdatedAt',
            '[data-composer-status-overall-elapsed]',
            now,
        );
        return trackingStage || trackingOverall;
    }

    #updateStatusDuration(status, secondsKey, updatedAtKey, selector, now) {
        const baseSeconds = Number.parseFloat(status.dataset[secondsKey] || '');
        const elapsed = status.querySelector(selector);
        if (!Number.isFinite(baseSeconds) || !(elapsed instanceof HTMLElement)) return false;
        const updatedAt = Number.parseFloat(status.dataset[updatedAtKey] || '');
        const baseline = Number.isFinite(updatedAt) ? updatedAt : now;
        if (!Number.isFinite(updatedAt)) status.dataset[updatedAtKey] = String(baseline);
        elapsed.textContent = formatElapsed(baseSeconds + (now - baseline) / 1000);
        return true;
    }

    #syncStatusText(currentStatus, nextStatus, selector) {
        const currentText = currentStatus.querySelector(selector);
        const nextText = nextStatus.querySelector(selector);
        if (!(currentText instanceof HTMLElement) || !(nextText instanceof HTMLElement)) return false;
        currentText.textContent = nextText.textContent;
        return true;
    }

    #syncStatusDuration(currentStatus, nextStatus, secondsKey, updatedAtKey, selector, timerSelector) {
        const nextElapsedSeconds = Number.parseFloat(nextStatus.dataset[secondsKey] || '');
        const currentElapsed = currentStatus.querySelector(selector);
        const currentTimer = currentStatus.querySelector(timerSelector);
        if (!(currentElapsed instanceof HTMLElement) || !(currentTimer instanceof HTMLElement)) return false;
        if (!Number.isFinite(nextElapsedSeconds)) {
            delete currentStatus.dataset[secondsKey];
            delete currentStatus.dataset[updatedAtKey];
            currentElapsed.textContent = '';
            currentTimer.hidden = true;
            return true;
        }
        currentTimer.hidden = false;
        currentStatus.dataset[secondsKey] = String(nextElapsedSeconds);
        currentStatus.dataset[updatedAtKey] = String(performance.now());
        currentElapsed.textContent = formatElapsed(nextElapsedSeconds);
        return true;
    }
}
