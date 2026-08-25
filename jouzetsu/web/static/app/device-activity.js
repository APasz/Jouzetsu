import { csrfHeaders } from './forms.js';

const HEARTBEAT_INTERVAL_MS = 5 * 60 * 1000;
const ACTIVITY_ENDPOINT = '/access/current-device/activity';

export class DeviceActivityController {
    #started = false;
    #requestInFlight = false;

    start() {
        if (this.#started) return;
        this.#started = true;
        const heartbeatIfVisible = () => this.#heartbeatIfVisible();
        document.addEventListener('visibilitychange', heartbeatIfVisible);
        window.addEventListener('focus', heartbeatIfVisible);
        window.setInterval(heartbeatIfVisible, HEARTBEAT_INTERVAL_MS);
    }

    #heartbeatIfVisible() {
        if (document.visibilityState !== 'visible') return;
        void this.#heartbeat();
    }

    async #heartbeat() {
        if (this.#requestInFlight) return;
        this.#requestInFlight = true;
        try {
            await window.fetch(ACTIVITY_ENDPOINT, {
                method: 'POST',
                headers: csrfHeaders(),
                credentials: 'same-origin',
                cache: 'no-store',
            });
        } catch {
            // Activity tracking is best-effort and must not affect chat use.
        } finally {
            this.#requestInFlight = false;
        }
    }
}
