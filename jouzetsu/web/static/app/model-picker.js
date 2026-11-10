/** Icon-bearing model comboboxes backed by the existing model form values. */
/**
 * @typedef {Object} ModelPickerControl
 * @property {HTMLElement} picker
 * @property {HTMLButtonElement} toggle
 * @property {HTMLInputElement} input
 * @property {HTMLElement} list
 * @property {HTMLElement[]} options
 */

const TYPEAHEAD_TIMEOUT_MS = 700;
const OPEN_PICKER_SELECTOR = '[data-model-picker-toggle][aria-expanded="true"]';
/** @type {WeakMap<HTMLElement, ModelPickerControl>} */
const controls = new WeakMap();

/** @param {Element | null} picker @returns {ModelPickerControl} */
const parts = (picker) => {
    if (!(picker instanceof HTMLElement)) throw new Error('The model picker container is missing.');
    const cached = controls.get(picker);
    if (cached) return cached;
    const toggle = picker.querySelector('[data-model-picker-toggle]');
    const input = picker.querySelector('[data-model-picker-value]');
    const list = picker.querySelector('[data-model-picker-options]');
    if (!(toggle instanceof HTMLButtonElement) || !(input instanceof HTMLInputElement) || !(list instanceof HTMLElement)) {
        throw new Error('The model picker is missing its control, value, or options.');
    }
    const options = Array.from(list.querySelectorAll('[data-model-picker-option]'), option => {
        if (!(option instanceof HTMLElement)) throw new Error('The model picker contains an invalid option.');
        return option;
    });
    const control = { picker, toggle, input, list, options };
    controls.set(picker, control);
    return control;
};

/** @param {ModelPickerControl} control */
const enabledOptions = (control) => control.options.filter(option => option.getAttribute('aria-disabled') !== 'true');
/** @param {ModelPickerControl} control */
const isOpen = (control) => control.toggle.getAttribute('aria-expanded') === 'true';

/** @param {ModelPickerControl} control */
const positionList = (control) => {
    if (!control.picker.isConnected) return;
    const container = control.picker.closest('.jouzetsu-panel-content, dialog');
    if (!(container instanceof HTMLElement)) throw new Error('The model picker has no scroll container.');
    control.list.classList.remove('opens-upward');
    control.list.style.removeProperty('max-height');
    const clip = container.getBoundingClientRect();
    const anchor = control.picker.getBoundingClientRect();
    const bounds = control.list.getBoundingClientRect();
    const viewport = window.visualViewport;
    const viewportTop = viewport?.offsetTop ?? 0;
    const top = Math.max(clip.top + container.clientTop, viewportTop);
    const bottom = Math.min(clip.top + container.clientTop + container.clientHeight,
        viewportTop + (viewport?.height ?? window.innerHeight));
    const gap = bounds.top - anchor.bottom;
    const below = Math.max(0, bottom - anchor.bottom - gap);
    const above = Math.max(0, anchor.top - top - gap);
    const upward = below < bounds.height && above > below;
    control.list.classList.toggle('opens-upward', upward);
    control.list.style.maxHeight = `${Math.min(bounds.height, upward ? above : below)}px`;
};

/** @param {ModelPickerControl} control @param {HTMLElement} option */
const revealOption = (control, option) => {
    const listBounds = control.list.getBoundingClientRect();
    const bounds = option.getBoundingClientRect();
    const top = listBounds.top + control.list.clientTop;
    const bottom = top + control.list.clientHeight;
    if (bounds.top < top) control.list.scrollTop += bounds.top - top;
    else if (bounds.bottom > bottom) control.list.scrollTop += bounds.bottom - bottom;
};

/** @param {ModelPickerControl} control @param {HTMLElement} option */
const setActive = (control, option, scroll = true) => {
    control.options.forEach(candidate => candidate.classList.toggle('is-active', candidate === option));
    control.toggle.setAttribute('aria-activedescendant', option.id);
    if (scroll) revealOption(control, option);
};

/** @param {ModelPickerControl} control */
const close = (control) => {
    control.list.hidden = true;
    control.toggle.setAttribute('aria-expanded', 'false');
    control.toggle.removeAttribute('aria-activedescendant');
    control.options.forEach(option => option.classList.remove('is-active'));
};

/** @param {ModelPickerControl} control */
const open = (control, activeValue = control.input.value, scroll = true) => {
    if (control.toggle.disabled || control.input.disabled) return;
    const options = enabledOptions(control);
    const active = options.find(option => option.dataset.modelValue === activeValue) || options[0];
    if (!active) return;
    control.list.hidden = false;
    control.toggle.setAttribute('aria-expanded', 'true');
    positionList(control);
    setActive(control, active, scroll);
};

/** @param {Element} current @param {Element} next */
export const synchronizeModelPickers = (current, next) => {
    current.querySelectorAll(OPEN_PICKER_SELECTOR).forEach(toggle => {
        const picker = toggle.closest('[data-model-picker]');
        const nextPicker = next.querySelector(`#${CSS.escape(toggle.id)}`)?.closest('[data-model-picker]');
        if (!picker || !nextPicker) return;
        const control = parts(picker);
        const nextControl = parts(nextPicker);
        const active = control.options.find(option => option.classList.contains('is-active'));
        const scrollTop = control.list.scrollTop;
        open(nextControl, active?.dataset.modelValue, false);
        window.requestAnimationFrame(() => {
            if (nextControl.toggle.isConnected && isOpen(nextControl)) {
                positionList(nextControl);
                nextControl.list.scrollTop = scrollTop;
            }
        });
    });
};

export class ModelPickerController {
    #search = '';
    #searchedAt = 0;
    #viewportFrame = 0;

    /** @param {Element | null} except */
    #closeOthers(except = null) {
        document.querySelectorAll(OPEN_PICKER_SELECTOR).forEach(toggle => {
            const picker = toggle.closest('[data-model-picker]');
            if (picker && picker !== except) close(parts(picker));
        });
    }

    /** @param {Element} target */
    handleClick(target) {
        const picker = target.closest('[data-model-picker]');
        this.#closeOthers(picker);
        if (!picker) return false;
        const control = parts(picker);
        const option = target.closest('[data-model-picker-option]');
        if (option instanceof HTMLElement && control.options.includes(option)) {
            this.#choose(control, option);
            return true;
        }
        if (target.closest('[data-model-picker-toggle]')) {
            if (isOpen(control)) close(control);
            else open(control);
            this.#search = '';
            return true;
        }
        return false;
    }

    /** @param {EventTarget | null} target */
    handleFocusIn(target) {
        this.#closeOthers(target instanceof Element ? target.closest('[data-model-picker]') : null);
    }

    /** @param {Event} event */
    handleViewportChange(event) {
        if (event.target instanceof Element && event.target.closest('[data-model-picker-options]')) return;
        if (this.#viewportFrame || !document.querySelector(OPEN_PICKER_SELECTOR)) return;
        this.#viewportFrame = window.requestAnimationFrame(() => {
            this.#viewportFrame = 0;
            document.querySelectorAll(OPEN_PICKER_SELECTOR).forEach(toggle => {
                positionList(parts(toggle.closest('[data-model-picker]')));
            });
        });
    }

    /** @param {PointerEvent} event */
    handlePointerDown(event) {
        if (event.target instanceof Element && event.target.closest('[data-model-picker-option]')) {
            event.preventDefault();
        }
    }

    /** @param {KeyboardEvent} event */
    handleKeyDown(event) {
        const toggle = event.target instanceof Element ? event.target.closest('[data-model-picker-toggle]') : null;
        if (!(toggle instanceof HTMLButtonElement) || toggle.disabled) return false;
        const control = parts(toggle.closest('[data-model-picker]'));
        const opened = isOpen(control);
        if (event.key === 'Tab') {
            close(control);
            return false;
        }
        if (event.key === 'Escape') {
            if (!opened) return false;
            event.preventDefault();
            close(control);
            return true;
        }
        if (event.ctrlKey || event.metaKey) return false;
        const options = enabledOptions(control);
        const activeIndex = Math.max(0, options.findIndex(option => opened
            ? option.classList.contains('is-active') : option.dataset.modelValue === control.input.value));
        if (['ArrowDown', 'ArrowUp', 'Home', 'End'].includes(event.key)) {
            event.preventDefault();
            this.#closeOthers(control.picker);
            if (!opened) open(control);
            let index = options.findIndex(option => option.classList.contains('is-active'));
            if (event.key === 'Home') index = 0;
            else if (event.key === 'End') index = options.length - 1;
            else if (opened) index = Math.max(0, Math.min(options.length - 1, activeIndex + (event.key === 'ArrowDown' ? 1 : -1)));
            if (options[index]) setActive(control, options[index]);
            this.#search = '';
            return true;
        }
        if (event.key === 'Enter' || event.key === ' ') {
            event.preventDefault();
            if (opened) this.#choose(control, options[activeIndex]);
            else {
                this.#closeOthers(control.picker);
                open(control);
            }
            this.#search = '';
            return true;
        }
        if (event.key.length !== 1 || event.altKey) return false;
        event.preventDefault();
        this.#closeOthers(control.picker);
        if (!opened) open(control);
        const now = performance.now();
        const letter = event.key.toLocaleLowerCase();
        this.#search = !opened || now - this.#searchedAt > TYPEAHEAD_TIMEOUT_MS ? letter
            : this.#search === letter ? letter : this.#search + letter;
        this.#searchedAt = now;
        const start = this.#search.length === 1 ? activeIndex + 1 : activeIndex;
        for (let offset = 0; offset < options.length; offset += 1) {
            const option = options[(start + offset) % options.length];
            if (option.dataset.modelLabel?.toLocaleLowerCase().startsWith(this.#search)) {
                setActive(control, option);
                break;
            }
        }
        return true;
    }

    /** @param {ModelPickerControl} control @param {HTMLElement | undefined} option */
    #choose(control, option) {
        if (!option || control.toggle.disabled || control.input.disabled || option.getAttribute('aria-disabled') === 'true') return;
        const content = option.querySelector('[data-model-choice-content]');
        const current = control.toggle.querySelector('[data-model-choice-content]');
        const value = option.dataset.modelValue;
        if (!content || !current || typeof value !== 'string') throw new Error('The model option is missing its label or value.');
        const changed = control.input.value !== value;
        control.input.value = value;
        current.replaceWith(content.cloneNode(true));
        control.options.forEach(candidate => candidate.setAttribute('aria-selected', String(candidate === option)));
        close(control);
        control.toggle.focus({ preventScroll: true });
        if (changed) control.input.dispatchEvent(new Event('change', { bubbles: true }));
    }
}
