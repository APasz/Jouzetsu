/** Backend-specific form hints, using examples and transport values from the server. */

const synchronizeBackendControl = (control) => {
    const form = control.closest('form');
    if (!(form instanceof HTMLFormElement)) throw new Error('Backend control has no form.');
    if (control.matches('[data-backend-kind]')) {
        const example = control.selectedOptions[0]?.dataset.baseUrlExample;
        const input = form.querySelector('[data-backend-base-url]');
        const hint = form.querySelector('[data-backend-url-example]');
        if (!example || !(input instanceof HTMLInputElement) || !(hint instanceof HTMLElement)) {
            throw new Error('Backend type is missing its URL example or controls.');
        }
        input.placeholder = example;
        hint.textContent = example;
    } else if (control.matches('[data-backend-transport]')) {
        const help = form.querySelector('[data-backend-responses-help]');
        if (!(help instanceof HTMLElement)) throw new Error('Backend transport has no help text.');
        help.hidden = control.value !== control.dataset.responsesValue;
    }
};

export const synchronizeBackendControls = (root = document) => {
    root.querySelectorAll('[data-backend-kind], [data-backend-transport]').forEach(synchronizeBackendControl);
};

export const handleBackendControlChange = (control) => {
    if (control instanceof HTMLSelectElement && control.matches('[data-backend-kind], [data-backend-transport]')) {
        synchronizeBackendControl(control);
    }
};
