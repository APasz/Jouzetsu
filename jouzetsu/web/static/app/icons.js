/** Create client-side icons from the same packaged sprite as server-rendered controls. */
export const createIcon = (name, classes = 'jouzetsu-svg-icon') => {
    const metadata = document.querySelector('meta[name="jouzetsu-icons"]');
    if (!(metadata instanceof HTMLMetaElement) || !metadata.content || !metadata.dataset.viewBox) {
        throw new Error('The page is missing its Phosphor icon sprite configuration.');
    }
    const namespace = 'http://www.w3.org/2000/svg';
    const icon = document.createElementNS(namespace, 'svg');
    icon.setAttribute('class', classes);
    icon.setAttribute('viewBox', metadata.dataset.viewBox);
    icon.setAttribute('fill', 'currentColor');
    icon.setAttribute('stroke', 'none');
    icon.setAttribute('aria-hidden', 'true');
    icon.setAttribute('focusable', 'false');
    const use = document.createElementNS(namespace, 'use');
    use.setAttribute('href', `${metadata.content}#${encodeURIComponent(name)}`);
    icon.append(use);
    return icon;
};
