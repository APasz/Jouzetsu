export const byId = (id) => document.getElementById(id);

export const parseFragment = (markup, expectedId) => {
    const template = document.createElement('template');
    template.innerHTML = markup.trim();
    const fragment = template.content.firstElementChild;
    return fragment instanceof HTMLElement && fragment.id === expectedId ? fragment : null;
};

export const syncElementAttributes = (current, next) => {
    const nextAttributeNames = new Set(Array.from(next.attributes, (attribute) => attribute.name));
    for (const attribute of Array.from(current.attributes)) {
        if (attribute.name !== 'style' && !nextAttributeNames.has(attribute.name)) {
            current.removeAttribute(attribute.name);
        }
    }
    for (const attribute of Array.from(next.attributes)) {
        if (attribute.name !== 'style' && current.getAttribute(attribute.name) !== attribute.value) {
            current.setAttribute(attribute.name, attribute.value);
        }
    }
};

export const cloneChildren = (element) => Array.from(element.childNodes, (child) => child.cloneNode(true));

const formatTimestamp = (value, fallback, includeDate) => {
    const timestamp = Number(value);
    if (!Number.isFinite(timestamp)) return fallback;
    const date = new Date(timestamp * 1000);
    if (Number.isNaN(date.getTime())) return fallback;
    const pad = (part) => String(part).padStart(2, '0');
    const time = `${pad(date.getHours())}:${pad(date.getMinutes())}`;
    return includeDate
        ? `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())} ${time}:${pad(date.getSeconds())}`
        : `${pad(date.getMonth() + 1)}-${pad(date.getDate())} ${time}`;
};

export const localizeMessageUpdatedTimes = () => {
    document.querySelectorAll('[data-message-updated-at]').forEach((timestamp) => {
        if (timestamp instanceof HTMLElement) {
            timestamp.textContent = formatTimestamp(timestamp.dataset.messageUpdatedAt, '', false);
        }
    });
};

export const localizeMessageDetailsTimes = (content) => {
    if (!(content instanceof HTMLElement)) return;
    content.querySelectorAll('[data-message-details-timestamp]').forEach((timestamp) => {
        if (timestamp instanceof HTMLElement) {
            timestamp.textContent = formatTimestamp(timestamp.dataset.messageDetailsTimestamp, 'Unknown', true);
        }
    });
};

export const chatFragmentUrl = (path) => {
    const edit = new URLSearchParams(window.location.search).get('edit');
    return edit ? `${path}?edit=${encodeURIComponent(edit)}` : path;
};
