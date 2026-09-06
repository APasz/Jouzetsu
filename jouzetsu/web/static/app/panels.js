import { byId } from './dom.js';

export class PanelController {
    #opener = null;
    #refreshPanel = () => {};

    setRefreshPanel(refreshPanel) {
        this.#refreshPanel = refreshPanel;
    }

    open(name, opener) {
        const panel = byId(`${name}-panel`);
        if (!(panel instanceof HTMLElement)) return;
        this.close(false);
        this.#opener = opener instanceof HTMLElement ? opener : null;
        panel.classList.add('is-open');
        this.#setPanelAccessibility(panel, true);
        this.#setChatAccessibility(true);
        document.documentElement.classList.add('jouzetsu-panel-open');
        const focusTarget = panel.querySelector('button, a, input, select, textarea, [tabindex]:not([tabindex="-1"])');
        if (focusTarget instanceof HTMLElement) {
            window.requestAnimationFrame(() => focusTarget.focus({ preventScroll: true }));
        }
    }

    close(restoreFocus = true) {
        const closedNames = [];
        document.querySelectorAll('.jouzetsu-panel').forEach((panel) => {
            if (!(panel instanceof HTMLElement)) return;
            if (panel.classList.contains('is-open') && panel.dataset.panelName) closedNames.push(panel.dataset.panelName);
            panel.classList.remove('is-open');
            this.#setPanelAccessibility(panel, false);
        });
        this.#setChatAccessibility(false);
        document.documentElement.classList.remove('jouzetsu-panel-open');
        if (restoreFocus && this.#opener instanceof HTMLElement) this.#opener.focus();
        this.#opener = null;
        if (restoreFocus) closedNames.forEach((name) => void this.#refreshPanel(name));
    }

    removeTransientParameters() {
        const url = new URL(window.location.href);
        const names = ['dialog', 'tab', 'notice', 'error', 'undo'];
        if (!names.some((name) => url.searchParams.has(name))) return;
        names.forEach((name) => url.searchParams.delete(name));
        window.history.replaceState({}, '', `${url.pathname}${url.search}${url.hash}`);
    }

    closeDialog(dialog) {
        if (dialog?.open) dialog.close();
        else this.removeTransientParameters();
    }

    activateTab(tab, focus = false) {
        if (!(tab instanceof HTMLButtonElement)) return;
        const tabList = tab.closest('[data-tab-list]');
        if (!(tabList instanceof HTMLElement)) return;
        const tabs = this.#tabButtons(tabList);
        if (!tabs.includes(tab)) return;
        tabs.forEach((candidate) => {
            const selected = candidate === tab;
            candidate.setAttribute('aria-selected', String(selected));
            candidate.tabIndex = selected ? 0 : -1;
            const panel = byId(candidate.dataset.tabTarget || '');
            if (panel instanceof HTMLElement) panel.hidden = !selected;
        });
        if (focus) tab.focus();
    }

    handleTabKeyDown(event) {
        const tab = event.target instanceof Element ? event.target.closest('[data-tab-target]') : null;
        if (!(tab instanceof HTMLButtonElement)) return false;
        const tabList = tab.closest('[data-tab-list]');
        if (!(tabList instanceof HTMLElement)) return false;
        const tabs = this.#tabButtons(tabList);
        const currentIndex = tabs.indexOf(tab);
        if (currentIndex < 0) return false;
        let nextIndex = currentIndex;
        if (event.key === 'ArrowLeft' || event.key === 'ArrowUp') nextIndex = (currentIndex - 1 + tabs.length) % tabs.length;
        else if (event.key === 'ArrowRight' || event.key === 'ArrowDown') nextIndex = (currentIndex + 1) % tabs.length;
        else if (event.key === 'Home') nextIndex = 0;
        else if (event.key === 'End') nextIndex = tabs.length - 1;
        else return false;
        event.preventDefault();
        this.activateTab(tabs[nextIndex], true);
        return true;
    }

    #setPanelAccessibility(panel, isOpen) {
        panel.toggleAttribute('inert', !isOpen);
        panel.setAttribute('aria-hidden', String(!isOpen));
    }

    #setChatAccessibility(isHidden) {
        const chat = byId('chat-fragment');
        if (!(chat instanceof HTMLElement)) return;
        chat.toggleAttribute('inert', isHidden);
        chat.setAttribute('aria-hidden', String(isHidden));
    }

    #tabButtons(tabList) {
        return Array.from(tabList.children).filter(
            (child) => child instanceof HTMLButtonElement && child.dataset.tabTarget,
        );
    }
}
