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
};

export const liveFormBody = (form) => {
    const body = new URLSearchParams();
    new FormData(form).forEach((value, name) => {
        if (typeof value === 'string') body.append(name, value);
    });
    return body;
};

export { CSRF_FORM_FIELD };
