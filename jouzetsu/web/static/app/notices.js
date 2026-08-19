import { byId } from './dom.js';

export const RETRY_DRAFT_NOTICE_ACTION = 'retry-draft';

const NOTICE_DISMISS_DELAY_MS = 3600;
const ERROR_NOTICE_DISMISS_DELAY_MS = 8000;

export class NoticeController {
    #dismissTimer = 0;

    dismiss() {
        const area = byId('notice-area');
        if (this.#dismissTimer) window.clearTimeout(this.#dismissTimer);
        this.#dismissTimer = 0;
        area?.replaceChildren();
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
        area.replaceChildren(notice);
        this.scheduleDismissal(notice, isError);
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
        const text = document.createElement('span');
        text.className = 'jouzetsu-notice-message';
        text.textContent = message;
        notice.append(text);
        return notice;
    }
}
