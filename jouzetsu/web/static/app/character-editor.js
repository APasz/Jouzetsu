const CHARACTER_FIELD_KINDS = new Map([
    ['short_text', 'Short text'],
    ['long_text', 'Long text'],
]);

const createSpellingRow = (source = '', replacement = '') => {
    const row = document.createElement('div');
    row.className = 'jouzetsu-spelling-row';
    row.dataset.spellingRow = 'true';
    const sourceInput = document.createElement('input');
    sourceInput.placeholder = 'color';
    sourceInput.value = source;
    sourceInput.setAttribute('aria-label', 'Source spelling');
    sourceInput.dataset.spellingSource = 'true';
    const arrow = document.createElement('span');
    arrow.className = 'jouzetsu-spelling-arrow';
    arrow.setAttribute('aria-hidden', 'true');
    arrow.textContent = '→';
    const replacementInput = document.createElement('input');
    replacementInput.placeholder = 'colour';
    replacementInput.value = replacement;
    replacementInput.setAttribute('aria-label', 'Replacement spelling');
    replacementInput.dataset.spellingReplacement = 'true';
    const remove = document.createElement('button');
    remove.type = 'button';
    remove.className = 'jouzetsu-spelling-remove';
    remove.dataset.spellingRemove = 'true';
    remove.textContent = 'Remove';
    row.append(sourceInput, arrow, replacementInput, remove);
    return row;
};

const characterFieldId = () => {
    if (window.crypto?.randomUUID) return `field_${window.crypto.randomUUID()}`;
    return `field_${Date.now()}_${Math.random().toString(16).slice(2)}`;
};

const field = (label, control, className = '') => {
    const wrapper = document.createElement('label');
    wrapper.className = `jouzetsu-field ${className}`.trim();
    const title = document.createElement('span');
    title.className = 'jouzetsu-field-label';
    title.textContent = label;
    wrapper.append(title, control);
    return wrapper;
};

const createValueControl = (kind, value = '') => {
    const control = kind === 'long_text' ? document.createElement('textarea') : document.createElement('input');
    control.name = 'field_value';
    control.value = value;
    control.dataset.characterFieldValue = 'true';
    control.setAttribute('aria-label', 'Character field value');
    if (control instanceof HTMLInputElement) control.autocomplete = 'off';
    else control.rows = 5;
    return control;
};

const characterFieldOptionsId = (fieldId) => `character-field-options-${fieldId}`;

const fieldMenuForToggle = (toggle) => {
    const row = toggle.closest('[data-character-field]');
    const menu = row?.querySelector('[data-character-field-menu]');
    return menu instanceof HTMLElement ? menu : null;
};

const fieldToggleForMenu = (menu) => {
    const row = menu.closest('[data-character-field]');
    const toggle = row?.querySelector('[data-character-field-menu-toggle]');
    return toggle instanceof HTMLButtonElement ? toggle : null;
};

const createFieldKindControl = (kind = 'short_text', label = '') => {
    const control = document.createElement('select');
    control.name = 'field_kind';
    control.setAttribute('aria-label', label ? `${label} input style` : 'Character field input style');
    for (const [value, text] of CHARACTER_FIELD_KINDS) {
        const option = document.createElement('option');
        option.value = value;
        option.textContent = text;
        option.selected = value === kind;
        control.append(option);
    }
    return control;
};

const createFieldActions = (fieldId, kind, label) => {
    const actions = document.createElement('div');
    actions.className = 'jouzetsu-character-field-actions';
    const optionsId = characterFieldOptionsId(fieldId);
    const toggle = document.createElement('button');
    toggle.type = 'button';
    toggle.className = 'jouzetsu-button jouzetsu-character-field-menu-toggle';
    toggle.title = 'Field options';
    toggle.setAttribute('aria-label', label ? `Options for ${label}` : 'Field options');
    toggle.setAttribute('aria-controls', optionsId);
    toggle.setAttribute('aria-expanded', 'false');
    toggle.dataset.characterFieldMenuToggle = 'true';
    const icon = document.createElement('span');
    icon.className = 'jouzetsu-character-field-menu-icon';
    icon.setAttribute('aria-hidden', 'true');
    icon.textContent = '•••';
    toggle.append(icon);
    const menu = document.createElement('div');
    menu.id = optionsId;
    menu.className = 'jouzetsu-character-field-menu';
    menu.dataset.characterFieldMenu = 'true';
    menu.hidden = true;
    const remove = document.createElement('button');
    remove.type = 'button';
    remove.className = 'jouzetsu-button jouzetsu-button-muted';
    remove.dataset.characterRemoveField = 'true';
    remove.textContent = 'Remove field';
    menu.append(field('Input style', createFieldKindControl(kind, label)), remove);
    actions.append(toggle, menu);
    return actions;
};

const createFieldRow = (definition = {}) => {
    const row = document.createElement('div');
    row.className = 'jouzetsu-character-field-row';
    row.dataset.characterField = 'true';
    const fieldId = characterFieldId();
    const labelText = definition.label || '';
    const kind = definition.kind || 'short_text';
    const id = document.createElement('input');
    id.type = 'hidden';
    id.name = 'field_id';
    id.value = fieldId;
    const label = document.createElement('input');
    label.name = 'field_label';
    label.autocomplete = 'off';
    label.required = true;
    label.value = labelText;
    label.setAttribute('aria-label', 'Character field label');
    label.dataset.characterFieldLabel = 'true';
    const value = createValueControl(kind, definition.value || '');
    value.setAttribute('aria-label', labelText ? `${labelText} value` : 'Character field value');
    row.append(
        id,
        field('Field', label, 'jouzetsu-character-field-name'),
        field('Value', value, 'jouzetsu-character-field-value'),
        createFieldActions(fieldId, kind, labelText),
    );
    return row;
};

const presetFields = (control) => {
    if (!(control instanceof HTMLElement)) return [];
    try {
        const parsed = JSON.parse(control.dataset.characterPresetFields || '[]');
        return Array.isArray(parsed)
            ? parsed.filter((candidate) => (
                candidate
                && typeof candidate.label === 'string'
                && CHARACTER_FIELD_KINDS.has(candidate.kind)
                && typeof candidate.value === 'string'
            ))
            : [];
    } catch {
        return [];
    }
};

const normalizedFieldLabel = (value) => value.trim().toLowerCase();

const addPresetRows = (fields, definitions) => {
    const labels = new Set(
        Array.from(fields.querySelectorAll('input[name="field_label"]')).flatMap((control) => (
            control instanceof HTMLInputElement && normalizedFieldLabel(control.value)
                ? [normalizedFieldLabel(control.value)]
                : []
        )),
    );
    const addedRows = [];
    definitions.forEach((definition) => {
        const label = normalizedFieldLabel(definition.label);
        if (!label || labels.has(label)) return;
        const row = createFieldRow(definition);
        fields.append(row);
        labels.add(label);
        addedRows.push(row);
    });
    return addedRows;
};

export class CharacterEditorController {
    #dirty = false;
    #submitting = false;
    #refreshPending = false;

    initialize() {
        document.querySelectorAll('[data-character-editor-form]').forEach((form) => {
            if (form instanceof HTMLFormElement) this.#syncPromptPreview(form);
        });
        this.#syncResponsiveState();
    }

    handleViewportChange() {
        this.#syncResponsiveState();
    }

    handleClick(target) {
        if (!(target instanceof Element)) return false;
        if (!target.closest('[data-character-field-menu], [data-character-field-menu-toggle]')) {
            this.#closeFieldMenus();
        }
        const button = target.closest('button');
        if (button instanceof HTMLButtonElement && button.matches('[data-character-field-menu-toggle]')) {
            this.#toggleFieldMenu(button);
            return true;
        }
        if (button instanceof HTMLButtonElement && button.matches('[data-character-library-toggle]')) {
            this.#toggleCharacterLibrary(button);
            return true;
        }
        if (button instanceof HTMLButtonElement && button.matches('[data-character-preset-toggle]')) {
            this.#togglePresetEditor(button);
            return true;
        }
        if (button instanceof HTMLButtonElement && button.matches('[data-character-add-field]')) {
            const fields = button.closest('[data-character-fields-editor]')?.querySelector('[data-character-fields]');
            if (!(fields instanceof HTMLElement)) return true;
            const row = createFieldRow();
            fields.append(row);
            this.markDirty();
            this.#syncPromptPreview(button.closest('[data-character-editor-form]'));
            row.querySelector('input[name="field_label"]')?.focus();
            return true;
        }
        if (button instanceof HTMLButtonElement && button.matches('[data-character-apply-presets]')) {
            this.#applyPresets(button);
            return true;
        }
        const removeField = target.closest('[data-character-remove-field]');
        if (removeField) {
            const row = removeField.closest('[data-character-field]');
            const form = row?.closest('[data-character-editor-form]');
            row?.remove();
            this.markDirty();
            this.#syncPromptPreview(form);
            return true;
        }
        const spellingAdd = target.closest('[data-spelling-add]');
        if (spellingAdd) {
            const rows = spellingAdd.closest('[data-spelling-editor]')?.querySelector('[data-spelling-rows]');
            if (!(rows instanceof HTMLElement)) return true;
            const row = createSpellingRow();
            rows.append(row);
            row.querySelector('[data-spelling-source]')?.focus();
            return true;
        }
        const spellingRemove = target.closest('[data-spelling-remove]');
        if (spellingRemove) {
            const editor = spellingRemove.closest('[data-spelling-editor]');
            spellingRemove.closest('[data-spelling-row]')?.remove();
            if (editor instanceof HTMLElement) this.syncSpellingEditor(editor);
            return true;
        }
        return false;
    }

    handleInput(target) {
        if (!(target instanceof Element)) return;
        const form = target.closest('[data-character-editor-form]');
        if (form instanceof HTMLFormElement) {
            this.markDirty();
            this.#syncPromptPreview(form);
        }
        const editor = target.closest('[data-spelling-editor]');
        if (editor instanceof HTMLElement) this.syncSpellingEditor(editor);
    }

    handleChange(target) {
        if (!(target instanceof Element)) return;
        const form = target.closest('[data-character-editor-form]');
        if (form instanceof HTMLFormElement) this.markDirty();
        const style = target.closest('select[name="field_kind"]');
        if (style instanceof HTMLSelectElement) this.#syncFieldValueControl(style.closest('[data-character-field]'));
        if (form instanceof HTMLFormElement) this.#syncPromptPreview(form);
    }

    handleKeyDown(event) {
        if (event.key !== 'Escape') return false;
        const menu = document.querySelector('[data-character-field-menu]:not([hidden])');
        if (!(menu instanceof HTMLElement)) return false;
        const toggle = fieldToggleForMenu(menu);
        menu.hidden = true;
        if (toggle instanceof HTMLButtonElement) {
            toggle.setAttribute('aria-expanded', 'false');
            toggle.focus();
        }
        event.preventDefault();
        return true;
    }

    validateForm(form) {
        const editor = form.querySelector('[data-spelling-editor]');
        if (editor instanceof HTMLElement && !this.syncSpellingEditor(editor)) return false;
        return true;
    }

    beginSubmit(form) {
        if (form.matches('[data-character-editor-form]')) {
            this.#dirty = false;
            this.#submitting = true;
        }
    }

    markDirty() {
        this.#dirty = true;
    }

    refreshIfNeeded(announce) {
        if (!document.querySelector('[data-character-workspace="true"]')) return false;
        if (this.#submitting) return true;
        if (this.#dirty) {
            announce('A character changed elsewhere. Reload before saving your edits.', true);
            return true;
        }
        if (!this.#refreshPending) {
            this.#refreshPending = true;
            window.location.reload();
        }
        return true;
    }

    syncSpellingEditor(editor) {
        const serialized = editor.querySelector('[data-spelling-serialized]');
        const validation = editor.querySelector('[data-spelling-validation]');
        if (!(serialized instanceof HTMLTextAreaElement) || !(validation instanceof HTMLElement)) return true;
        const sources = new Set();
        const lines = [];
        let error = '';
        editor.querySelectorAll('[data-spelling-row]').forEach((row) => {
            const source = row.querySelector('[data-spelling-source]');
            const replacement = row.querySelector('[data-spelling-replacement]');
            if (!(source instanceof HTMLInputElement) || !(replacement instanceof HTMLInputElement)) return;
            source.classList.remove('is-invalid');
            replacement.classList.remove('is-invalid');
            const sourceText = source.value.trim();
            const replacementText = replacement.value.trim();
            if (!sourceText && !replacementText) return;
            if (!sourceText || !replacementText || /\s/.test(sourceText) || /\s/.test(replacementText)) {
                error = 'Each replacement needs two single-word values.';
                source.classList.add('is-invalid');
                replacement.classList.add('is-invalid');
                return;
            }
            const sourceKey = sourceText.toLocaleLowerCase();
            if (sources.has(sourceKey)) {
                error = `Duplicate source: ${sourceText}`;
                source.classList.add('is-invalid');
                return;
            }
            sources.add(sourceKey);
            lines.push(`${sourceText}\t${replacementText}`);
        });
        validation.textContent = error;
        validation.classList.toggle('is-error', Boolean(error));
        if (error) return false;
        serialized.value = lines.join('\n');
        return true;
    }

    #syncPromptPreview(form) {
        if (!(form instanceof HTMLFormElement)) return;
        const editor = form.closest('.jouzetsu-character-editor');
        const preview = editor?.querySelector('[data-character-prompt-preview]');
        const name = form.querySelector('input[name="name"]');
        if (!(preview instanceof HTMLElement) || !(name instanceof HTMLInputElement)) return;
        const nameTemplate = preview.dataset.characterPromptNameTemplate;
        const profileHeading = preview.dataset.characterPromptProfileHeading;
        const emptyProfile = preview.dataset.characterPromptEmptyProfile;
        const closing = preview.dataset.characterPromptClosing;
        if (!nameTemplate || !profileHeading || !emptyProfile || !closing) return;
        const lines = [];
        form.querySelectorAll('[data-character-field]').forEach((row) => {
            const label = row.querySelector('[data-character-field-label]');
            const value = row.querySelector('[data-character-field-value]');
            if (!(label instanceof HTMLInputElement)) return;
            if (!(value instanceof HTMLInputElement || value instanceof HTMLTextAreaElement)) return;
            const labelText = label.value.trim();
            const valueText = value.value.trim();
            if (labelText && valueText) lines.push(`${labelText}: ${valueText}`);
            const menuToggle = row.querySelector('[data-character-field-menu-toggle]');
            if (menuToggle instanceof HTMLButtonElement) {
                menuToggle.setAttribute('aria-label', labelText ? `Options for ${labelText}` : 'Field options');
            }
        });
        const profile = lines.length ? lines.join('\n') : emptyProfile;
        preview.textContent = `${nameTemplate.replace('{name}', () => name.value.trim())}\n\n${profileHeading}\n${profile}\n\n${closing}`;
    }

    #toggleFieldMenu(toggle) {
        const menu = fieldMenuForToggle(toggle);
        if (!menu) return;
        const shouldOpen = menu.hidden;
        this.#closeFieldMenus(menu);
        menu.hidden = !shouldOpen;
        toggle.setAttribute('aria-expanded', String(shouldOpen));
        if (shouldOpen) {
            window.requestAnimationFrame(() => {
                const style = menu.querySelector('select');
                if (!menu.hidden && style instanceof HTMLSelectElement) style.focus();
            });
        }
    }

    #closeFieldMenus(exceptMenu = null) {
        document.querySelectorAll('[data-character-field-menu]').forEach((menu) => {
            if (!(menu instanceof HTMLElement) || menu === exceptMenu) return;
            menu.hidden = true;
            fieldToggleForMenu(menu)?.setAttribute('aria-expanded', 'false');
        });
    }

    #toggleCharacterLibrary(toggle) {
        const library = toggle.closest('[data-character-library]');
        if (!(library instanceof HTMLElement)) return;
        this.#setCharacterLibraryCollapsed(
            library,
            !library.classList.contains('is-collapsed'),
        );
        library.dataset.characterLibraryMobileReady = 'true';
    }

    #setCharacterLibraryCollapsed(library, collapsed) {
        library.classList.toggle('is-collapsed', collapsed);
        const toggle = library.querySelector('[data-character-library-toggle]');
        if (toggle instanceof HTMLButtonElement) toggle.setAttribute('aria-expanded', String(!collapsed));
    }

    #togglePresetEditor(toggle) {
        const editor = toggle.closest('[data-character-preset-editor]');
        const controls = editor?.querySelector('[data-character-preset-controls]');
        if (!(editor instanceof HTMLElement) || !(controls instanceof HTMLElement)) return;
        this.#setPresetEditorCollapsed(editor, !controls.hidden);
        editor.dataset.characterPresetMobileReady = 'true';
    }

    #setPresetEditorCollapsed(editor, collapsed) {
        const controls = editor.querySelector('[data-character-preset-controls]');
        const toggle = editor.querySelector('[data-character-preset-toggle]');
        if (!(controls instanceof HTMLElement)) return;
        controls.hidden = collapsed;
        editor.classList.toggle('is-collapsed', collapsed);
        if (toggle instanceof HTMLButtonElement) {
            toggle.textContent = collapsed ? 'Show templates' : 'Hide templates';
            toggle.setAttribute('aria-expanded', String(!collapsed));
        }
    }

    #syncResponsiveState() {
        const isNarrow = window.matchMedia('(max-width: 640px)').matches;
        document.querySelectorAll('[data-character-library]').forEach((library) => {
            if (!(library instanceof HTMLElement)) return;
            if (!isNarrow) {
                this.#setCharacterLibraryCollapsed(library, false);
                delete library.dataset.characterLibraryMobileReady;
            } else if (!library.dataset.characterLibraryMobileReady) {
                this.#setCharacterLibraryCollapsed(library, true);
                library.dataset.characterLibraryMobileReady = 'true';
            }
        });
        document.querySelectorAll('[data-character-preset-editor]').forEach((editor) => {
            if (!(editor instanceof HTMLElement)) return;
            if (!isNarrow) {
                this.#setPresetEditorCollapsed(editor, false);
                delete editor.dataset.characterPresetMobileReady;
                return;
            }
            if (editor.dataset.characterPresetMobileReady) return;
            const fields = editor.closest('[data-character-editor-form]')?.querySelector('[data-character-fields]');
            this.#setPresetEditorCollapsed(
                editor,
                fields instanceof HTMLElement && fields.childElementCount > 0,
            );
            editor.dataset.characterPresetMobileReady = 'true';
        });
    }

    #applyPresets(button) {
        const editor = button.closest('[data-character-preset-editor]');
        const form = button.closest('[data-character-editor-form]');
        const fields = form?.querySelector('[data-character-fields]');
        const base = editor?.querySelector('[data-character-base-preset]');
        if (!(editor instanceof HTMLElement) || !(form instanceof HTMLFormElement) || !(fields instanceof HTMLElement) || !(base instanceof HTMLSelectElement)) return;
        const addedRows = [];
        const selectedBase = base.selectedOptions.item(0);
        if (selectedBase instanceof HTMLOptionElement && base.value) {
            addedRows.push(...addPresetRows(fields, presetFields(selectedBase)));
        }
        const extras = Array.from(editor.querySelectorAll('[data-character-extra-preset]')).filter(
            (control) => control instanceof HTMLInputElement,
        );
        extras.filter((control) => control.checked).forEach((control) => {
            addedRows.push(...addPresetRows(fields, presetFields(control)));
        });
        this.markDirty();
        this.#syncPromptPreview(form);
        if (!addedRows.length) return;
        if (window.matchMedia('(max-width: 640px)').matches) {
            this.#setPresetEditorCollapsed(editor, true);
        }
        const firstLabel = addedRows[0]?.querySelector('input[name="field_label"]');
        if (firstLabel instanceof HTMLInputElement) {
            window.requestAnimationFrame(() => {
                firstLabel.scrollIntoView({ block: 'nearest' });
                firstLabel.focus();
            });
        }
    }

    #syncFieldValueControl(row) {
        if (!(row instanceof HTMLElement)) return;
        const style = row.querySelector('select[name="field_kind"]');
        const current = row.querySelector('[data-character-field-value]');
        if (!(style instanceof HTMLSelectElement) || !(current instanceof HTMLInputElement || current instanceof HTMLTextAreaElement)) return;
        if ((style.value === 'long_text') === (current instanceof HTMLTextAreaElement)) return;
        const label = row.querySelector('[data-character-field-label]');
        const value = createValueControl(style.value, current.value);
        if (label instanceof HTMLInputElement && label.value.trim()) {
            value.setAttribute('aria-label', `${label.value.trim()} value`);
        }
        current.replaceWith(value);
    }
}
