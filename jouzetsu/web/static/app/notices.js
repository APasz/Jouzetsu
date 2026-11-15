import { byId } from './dom.js';
import { createIcon } from './icons.js';

export const RETRY_DRAFT_NOTICE_ACTION = 'retry-draft';

const NOTICE_DISMISS_DELAY_MS = 3600;
const ERROR_NOTICE_DISMISS_DELAY_MS = 8000;

export class NoticeController {
    #dismissTimer = 0;

    dismiss() {
        const area = byId('notice-area');
        if (this.#dismissTimer) window.clearTimeout(this.#dismissTimer);
        this.#dismissTimer = 0;
        area?.querySelectorAll('[data-notice-kind]').forEach((notice) => notice.remove());
    }

    announce(message, isError = false, action = '') {
        const area = byId('notice-area');
        if (!area || !message) return;
        const notice = this.#build(message, isError);
        if (action === RETRY_DRAFT_NOTICE_ACTION) {
            const retry = document.createElement('button');
            retry.type = 'button';
            retry.className = 'jouzetsu-notice-action';
            retry.dataset.noticeAction = action;
            retry.textContent = 'Retry';
            notice.append(retry);
        }
        this.dismiss();
        area.prepend(notice);
        this.scheduleDismissal(notice, isError);
    }

    announceForForm(form, message, isError = false) {
        const dialog = this.#dialogForForm(form);
        if (!dialog) { this.announce(message, isError); return; }
        let feedback = this.#dialogFeedback(dialog);
        if (!feedback) {
            feedback = document.createElement('div');
            feedback.dataset.dialogFeedback = 'true';
            dialog.firstElementChild?.prepend(feedback);
        }
        const notice = this.#build(message, isError);
        feedback.replaceChildren(...notice.childNodes);
        feedback.className = `jouzetsu-dialog-feedback${isError ? ' is-error' : ''}`;
        feedback.setAttribute('role', isError ? 'alert' : 'status');
        feedback.hidden = false;
    }

    clearForForm(form) {
        const dialog = this.#dialogForForm(form);
        const feedback = dialog && this.#dialogFeedback(dialog);
        if (feedback) { feedback.replaceChildren(); feedback.hidden = true; }
    }

    preserveDialogFeedback(current, next) {
        for (const feedback of current.querySelectorAll('[data-dialog-feedback]')) {
            if (feedback.hidden || !feedback.textContent) continue;
            const dialogId = feedback.closest('dialog')?.id;
            if (!dialogId) throw new Error('Dialog feedback must belong to a named dialog.');
            const dialog = next.id === dialogId ? next : next.querySelector(`#${CSS.escape(dialogId)}`);
            if (!dialog) continue;
            const replacement = this.#dialogFeedback(dialog);
            if (replacement) replacement.replaceWith(feedback.cloneNode(true));
            else dialog.firstElementChild?.prepend(feedback.cloneNode(true));
        }
    }

    #dialogForForm(form) {
        for (let source = form.closest('dialog'); source; source = source.parentElement?.closest('dialog')) {
            const dialog = byId(source.id);
            if (dialog instanceof HTMLDialogElement && dialog.open) return dialog;
        }
        return null;
    }

    #dialogFeedback(dialog) {
        return Array.from(dialog.querySelectorAll('[data-dialog-feedback]')).find(
            (feedback) => feedback.closest('dialog') === dialog,
        );
    }

    scheduleDismissal(notice, isError) {
        if (this.#dismissTimer) window.clearTimeout(this.#dismissTimer);
        this.#dismissTimer = window.setTimeout(() => {
            if (notice.isConnected) this.dismiss();
        }, isError ? ERROR_NOTICE_DISMISS_DELAY_MS : NOTICE_DISMISS_DELAY_MS);
    }

    scheduleInitialDismissal() {
        const notice = byId('notice-area')?.querySelector('[data-notice-kind]');
        if (notice instanceof HTMLElement) {
            this.scheduleDismissal(notice, notice.dataset.noticeKind === 'error');
        }
    }

    #build(message, isError) {
        const notice = document.createElement('div');
        notice.className = isError ? 'jouzetsu-notice jouzetsu-notice-error' : 'jouzetsu-notice';
        notice.dataset.noticeKind = isError ? 'error' : 'success';
        notice.dataset.noticeDismiss = 'true';
        notice.setAttribute('role', isError ? 'alert' : 'status');
        if (isError) notice.append(createIcon('warning', 'jouzetsu-svg-icon jouzetsu-error-icon'));
        const text = document.createElement('span');
        text.className = 'jouzetsu-notice-message';
        text.textContent = message;
        notice.append(text);
        return notice;
    }
}
