import { csrfHeaders } from './forms.js';

export class ImageController {
    #announce;
    #syncPrimary;
    #drafts;
    #refresh = async () => {};
    #queue = Promise.resolve(true);
    #pending = 0;
    #failed = [];

    constructor({ announce, syncPrimary, drafts }) {
        this.#announce = announce;
        this.#syncPrimary = syncPrimary;
        this.#drafts = drafts;
        document.addEventListener('paste', (event) => {
            if (!(event.target instanceof Element) || !event.target.closest('.jouzetsu-composer')) return;
            const files = Array.from(event.clipboardData?.files || []);
            if (!files.length || !this.#uploadForm()) return;
            event.preventDefault();
            this.#attachFiles(files);
        });
        document.addEventListener('dragover', (event) => {
            if (!event.dataTransfer?.types.includes('Files') || !this.#dropTarget(event.target)) return;
            event.preventDefault();
            event.dataTransfer.dropEffect = 'copy';
            document.querySelector('.jouzetsu-composer')?.classList.add('is-image-dragover');
        });
        document.addEventListener('dragleave', (event) => {
            if (event.relatedTarget instanceof Element && event.relatedTarget.closest('#chat-fragment')) return;
            document.querySelector('.jouzetsu-composer')?.classList.remove('is-image-dragover');
        });
        document.addEventListener('drop', (event) => {
            document.querySelector('.jouzetsu-composer')?.classList.remove('is-image-dragover');
            if (!this.#dropTarget(event.target) || !event.dataTransfer?.files.length) return;
            event.preventDefault();
            this.#attachFiles(Array.from(event.dataTransfer.files));
        });
    }

    setRefresh(refresh) {
        this.#refresh = refresh;
    }

    get isUploading() {
        return this.#pending > 0;
    }

    synchronize() {
        document.querySelectorAll('[data-image-upload-pending]').forEach((hint) => { hint.hidden = !this.isUploading; });
        this.#syncPrimary();
        const form = this.#uploadForm();
        for (const failed of this.#failed) {
            if (form?.dataset.chatId === failed.chatId) document.querySelector('.jouzetsu-composer')?.append(failed.node);
            else failed.node.remove();
        }
    }

    async flush() {
        let succeeded = true;
        let queue;
        do {
            queue = this.#queue;
            const batchSucceeded = await queue;
            succeeded = batchSucceeded && succeeded;
        } while (queue !== this.#queue);
        return succeeded;
    }

    handleChange(control) {
        if (!(control instanceof HTMLInputElement) || control.type !== 'file') return;
        const form = control.form;
        if (!(form instanceof HTMLFormElement) || form.dataset.imageUploadForm !== 'true' || !control.files?.length) return;
        form.requestSubmit();
    }

    handleSubmit(form, submitter) {
        if (form.dataset.imageUploadForm !== 'true') return false;
        const isRemoval = submitter instanceof HTMLButtonElement && submitter.hasAttribute('formaction');
        const action = isRemoval ? submitter.formAction : form.action;
        this.#enqueue(action, isRemoval ? new URLSearchParams() : new FormData(form));
        const input = form.elements.namedItem('images');
        if (input instanceof HTMLInputElement) input.value = '';
        return true;
    }

    #uploadForm() {
        return document.querySelector('[data-image-upload-form]');
    }

    #dropTarget(target) {
        return target instanceof Element && target.closest('#chat-fragment') && this.#uploadForm();
    }

    #attachFiles(files) {
        const form = this.#uploadForm();
        if (!(form instanceof HTMLFormElement)) return;
        const body = new FormData();
        files.forEach((file) => body.append('images', file, file.name));
        this.#enqueue(form.action, body);
    }

    #retainFailed(action, body) {
        if (!(body instanceof FormData)) return;
        const files = body.getAll('images').filter((file) => file instanceof File);
        if (!files.length) return;
        const chatId = new URL(action, location.href).pathname.split('/')[2];
        const node = document.createElement('div');
        node.className = 'jouzetsu-dialog-feedback is-error';
        node.dataset.failedAttachments = 'true';
        const label = document.createElement('p');
        label.textContent = 'These attachments were not accepted. They remain in this browser for retry or download.';
        node.append(label);
        const urls = files.map((file) => {
            const url = URL.createObjectURL(file);
            const link = document.createElement('a');
            link.href = url; link.download = file.name; link.textContent = file.name;
            node.append(link, document.createTextNode(' '));
            return url;
        });
        const failed = {chatId, node};
        this.#failed.push(failed);
        const remove = () => {
            this.#failed = this.#failed.filter((entry) => entry !== failed);
            urls.forEach((url) => URL.revokeObjectURL(url)); node.remove();
            if (!this.isUploading) this.#queue = Promise.resolve(true);
        };
        for (const [text, actionCallback] of [
            ['Retry attachments', () => { remove(); this.#enqueue(action, body); }],
            ['Discard pending attachments', remove],
        ]) {
            const button = document.createElement('button');
            button.type = 'button'; button.className = 'jouzetsu-button'; button.textContent = text;
            button.addEventListener('click', actionCallback); node.append(button);
        }
    }

    #enqueue(action, body) {
        if (!this.isUploading) this.#queue = Promise.resolve(true);
        this.#pending += 1;
        this.synchronize();
        this.#queue = this.#queue.then(async (succeeded) => {
            try {
                if (!await this.#drafts.flush()) { this.#retainFailed(action, body); return false; }
                return await this.#drafts.serialize(async () => {
                const form = this.#uploadForm();
                if (form?.dataset.chatId !== new URL(action, location.href).pathname.split('/')[2]) throw new Error('Return to the original chat to retry these attachments');
                for (const name of ['chat_id', 'revision', 'draft_revision']) {
                    const value = form?.elements.namedItem(name)?.value;
                    if (value !== undefined) body.set(name, value);
                }
                const response = await window.fetch(action, {
                    method: 'POST', headers: csrfHeaders(), body, cache: 'no-store',
                });
                if (!response.ok) throw new Error(response.status === 403 ? 'Image upload is not allowed' : 'Could not save images');
                const error = new URL(response.url, window.location.origin).searchParams.get('error');
                if (error) throw new Error(error);
                const result = await response.json();
                this.#drafts.savedRevision(body.get('chat_id'), result.draft_revision);
                await this.#refresh();
                return succeeded;
                });
            } catch (error) {
                this.#retainFailed(action, body);
                this.#announce(error instanceof Error ? error.message : 'Could not save images', true);
                return false;
            } finally {
                this.#pending -= 1;
                this.synchronize();
            }
        });
    }
}
