import { byId, cloneChildren, parseFragment, syncElementAttributes } from './dom.js';
import { csrfHeaders } from './forms.js';

const REFRESH_INTERVAL_MS = 1000;

export class HostStatsController {
    #timer = 0;
    #refreshInFlight = false;
    #interactionUntil = 0;

    async refresh(force = false) {
        if (this.#refreshInFlight) return;
        const current = byId('host-stats-content');
        if (!(current instanceof HTMLElement)) return;
        this.#refreshInFlight = true;
        try {
            const response = await window.fetch(
                force ? '/fragments/host-stats/refresh' : '/fragments/host-stats',
                {
                    method: force ? 'POST' : 'GET',
                    headers: force ? csrfHeaders() : { 'HX-Request': 'true' },
                    cache: 'no-store',
                },
            );
            if (response.status === 403) {
                window.location.reload();
                return;
            }
            if (!response.ok) return;
            const next = parseFragment(await response.text(), 'host-stats-content');
            if (!next || performance.now() < this.#interactionUntil) return;
            if (force || next.dataset.sampledAt !== current.dataset.sampledAt) {
                if (!this.#patch(current, next)) this.#replace(current, next);
            }
        } catch {
            // Diagnostics are deliberately best-effort.
        } finally {
            this.#refreshInFlight = false;
        }
    }

    start() {
        const dialog = byId('host-stats-dialog');
        if (!(dialog instanceof HTMLDialogElement) || !dialog.open || this.#timer) return;
        const listeners = new AbortController();
        const stop = () => {
            if (this.#timer) {
                window.clearInterval(this.#timer);
                this.#timer = 0;
            }
            listeners.abort();
        };
        dialog.addEventListener('close', stop, { once: true });
        const noteInteraction = (event) => {
            if (event.target instanceof Element && event.target.closest('.jouzetsu-host-stat-summary')) {
                this.#interactionUntil = performance.now() + 350;
            }
        };
        dialog.addEventListener('pointerdown', noteInteraction, { capture: true, signal: listeners.signal });
        dialog.addEventListener('keydown', (event) => {
            if (event.key === 'Enter' || event.key === ' ') noteInteraction(event);
        }, { capture: true, signal: listeners.signal });
        this.#timer = window.setInterval(() => void this.refresh(), REFRESH_INTERVAL_MS);
    }

    #detailsKey(details, index) {
        return `${index}:${details.querySelector('.jouzetsu-host-stat-label')?.textContent || ''}`;
    }

    #replace(current, next) {
        const openDetails = new Set(Array.from(current.querySelectorAll('details[open]'), (details, index) => this.#detailsKey(details, index)));
        current.replaceWith(next);
        Array.from(next.querySelectorAll('details')).forEach((details, index) => {
            details.open = openDetails.has(this.#detailsKey(details, index));
        });
    }

    #patch(current, next) {
        const currentRows = [...current.querySelectorAll('[data-host-stat-row]')];
        const nextRows = [...next.querySelectorAll('[data-host-stat-row]')];
        if (currentRows.length !== nextRows.length) return false;
        for (let index = 0; index < currentRows.length; index += 1) {
            const row = currentRows[index];
            const nextRow = nextRows[index];
            if (row.dataset.hostStatRow !== nextRow.dataset.hostStatRow) return false;
            const copy = row.querySelector('.jouzetsu-host-stat-copy');
            const nextCopy = nextRow.querySelector('.jouzetsu-host-stat-copy');
            const meter = row.querySelector('.jouzetsu-host-stat-meter');
            const nextMeter = nextRow.querySelector('.jouzetsu-host-stat-meter');
            if (!(copy instanceof HTMLElement) || !(nextCopy instanceof HTMLElement) || (meter === null) !== (nextMeter === null)) return false;
            syncElementAttributes(row, nextRow);
            if (meter instanceof HTMLElement && nextMeter instanceof HTMLElement) syncElementAttributes(meter, nextMeter);
            const label = copy.querySelector('.jouzetsu-host-stat-label');
            const nextLabel = nextCopy.querySelector('.jouzetsu-host-stat-label');
            const value = copy.querySelector('.jouzetsu-host-stat-value');
            const nextValue = nextCopy.querySelector('.jouzetsu-host-stat-value');
            if (
                !(label instanceof HTMLElement)
                || !(nextLabel instanceof HTMLElement)
                || !(value instanceof HTMLElement)
                || !(nextValue instanceof HTMLElement)
            ) return false;
            if (label.textContent !== nextLabel.textContent) label.textContent = nextLabel.textContent;
            if (value.innerHTML !== nextValue.innerHTML) value.replaceChildren(...cloneChildren(nextValue));
        }
        syncElementAttributes(current, next);
        return true;
    }
}
