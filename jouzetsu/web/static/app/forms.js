const CSRF_FORM_FIELD = '_jouzetsu_csrf';
const CSRF_HEADER_NAME = 'X-Jouzetsu-CSRF';

export const csrfToken = () => document.querySelector('[data-csrf-token]')?.getAttribute('data-csrf-token') || '';

export const csrfHeaders = () => ({ 'HX-Request': 'true', [CSRF_HEADER_NAME]: csrfToken() });

export const addCsrfToken = (form) => {
    if (!(form instanceof HTMLFormElement) || form.method.toLowerCase() !== 'post') return;
    const token = csrfToken();
    if (!token) return;
    let input = form.querySelector(`input[name="${CSRF_FORM_FIELD}"]`);
    if (!(input instanceof HTMLInputElement)) {
        input = document.createElement('input');
        input.type = 'hidden';
        input.name = CSRF_FORM_FIELD;
        form.append(input);
    }
    input.value = token;
    const match = window.location.pathname.match(/^\/chats\/([A-Za-z0-9]+)$/);
    if (match && !form.elements.namedItem('return_chat_id')) {
        const destination = document.createElement('input'); destination.type = 'hidden'; destination.name = 'return_chat_id'; destination.value = match[1]; form.append(destination);
    }
};

export const liveFormBody = (form, submitter = null) => {
    const body = new URLSearchParams();
    new FormData(form).forEach((value, name) => {
        if (typeof value === 'string') body.append(name, value);
    });
    if (submitter?.name) body.set(submitter.name, submitter.value);
    return body;
};

export { CSRF_FORM_FIELD };


export const scopeChatForm = (form) => {
    const shell = document.getElementById('chat-fragment');
    if (!(form instanceof HTMLFormElement) || !shell?.dataset.chatId) return;
    for (const [name, value] of Object.entries({chat_id: shell.dataset.chatId, revision: shell.dataset.revision, draft_revision: shell.dataset.draftRevision})) {
        let input = form.elements.namedItem(name);
        if (!(input instanceof HTMLInputElement)) {
            input = document.createElement('input'); input.type = 'hidden'; input.name = name; form.append(input);
        }
        input.value = value;
    }
};


export const randomIdentifier = () => {
    if (typeof window.crypto?.randomUUID === 'function') return window.crypto.randomUUID();
    if (typeof window.crypto?.getRandomValues !== 'function') throw new Error('Secure browser randomness is unavailable');
    const bytes = new Uint8Array(16);
    window.crypto.getRandomValues(bytes);
    return Array.from(bytes, (value) => value.toString(16).padStart(2, '0')).join('');
};
