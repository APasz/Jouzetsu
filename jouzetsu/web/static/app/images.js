import { csrfHeaders } from './forms.js';

export class ImageController {
    #announce;
    #syncPrimary;
    #refresh = async () => {};
    #queue = Promise.resolve(true);
    #pending = 0;

    constructor({ announce, syncPrimary }) {
        this.#announce = announce;
        this.#syncPrimary = syncPrimary;
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

    #enqueue(action, body) {
        if (!this.isUploading) this.#queue = Promise.resolve(true);
        this.#pending += 1;
        this.synchronize();
        this.#queue = this.#queue.then(async (succeeded) => {
            try {
                const response = await window.fetch(action, {
                    method: 'POST', headers: csrfHeaders(), body, cache: 'no-store',
                });
                if (!response.ok) throw new Error(response.status === 403 ? 'Image upload is not allowed' : 'Could not save images');
                const error = new URL(response.url, window.location.origin).searchParams.get('error');
                if (error) throw new Error(error);
                await this.#refresh();
                return succeeded;
            } catch (error) {
                this.#announce(error instanceof Error ? error.message : 'Could not save images', true);
                return false;
            } finally {
                this.#pending -= 1;
                this.synchronize();
            }
        });
    }
}
