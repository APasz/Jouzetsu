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

const field = (label, control) => {
    const wrapper = document.createElement('label');
    wrapper.className = 'jouzetsu-field';
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

const createFieldRow = (definition = {}) => {
    const row = document.createElement('div');
    row.className = 'jouzetsu-character-field-row';
    row.dataset.characterField = 'true';
    if (definition.presetLayer) row.dataset.characterPresetLayer = definition.presetLayer;
    if (definition.presetId) row.dataset.characterPresetId = definition.presetId;
    const id = document.createElement('input');
    id.type = 'hidden';
    id.name = 'field_id';
    id.value = characterFieldId();
    const label = document.createElement('input');
    label.name = 'field_label';
    label.autocomplete = 'off';
    label.required = true;
    label.value = definition.label || '';
    label.setAttribute('aria-label', 'Character field label');
    const kind = document.createElement('select');
    kind.name = 'field_kind';
    kind.setAttribute('aria-label', 'Character field input style');
    for (const [value, text] of CHARACTER_FIELD_KINDS) {
        const option = document.createElement('option');
        option.value = value;
        option.textContent = text;
        option.selected = value === (definition.kind || 'short_text');
        kind.append(option);
    }
    const remove = document.createElement('button');
    remove.type = 'button';
    remove.className = 'jouzetsu-button jouzetsu-button-muted';
    remove.dataset.characterRemoveField = 'true';
    remove.textContent = 'Remove field';
    row.append(
        id,
        field('Label', label),
        field('Input style', kind),
        field('Value', createValueControl(definition.kind || 'short_text', definition.value || '')),
        remove,
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

const removePresetRows = (fields, layer, excludedPresetIds = new Set()) => {
    fields.querySelectorAll('[data-character-preset-layer]').forEach((row) => {
        if (!(row instanceof HTMLElement) || row.dataset.characterPresetLayer !== layer) return;
        if (!excludedPresetIds.has(row.dataset.characterPresetId || '')) row.remove();
    });
};

const addPresetRows = (fields, presetId, layer, definitions) => {
    if (!presetId) return;
    const alreadyApplied = Array.from(fields.querySelectorAll('[data-character-preset-layer]')).some((row) => (
        row instanceof HTMLElement
        && row.dataset.characterPresetLayer === layer
        && row.dataset.characterPresetId === presetId
    ));
    if (!alreadyApplied) {
        definitions.forEach((definition) => fields.append(createFieldRow({ ...definition, presetId, presetLayer: layer })));
    }
};

export class CharacterEditorController {
    #dirty = false;
    #submitting = false;
    #refreshPending = false;

    handleClick(target) {
        if (!(target instanceof Element)) return false;
        const button = target.closest('button');
        if (button instanceof HTMLButtonElement && button.matches('[data-character-add-field]')) {
            const fields = button.closest('[data-character-fields-editor]')?.querySelector('[data-character-fields]');
            if (!(fields instanceof HTMLElement)) return true;
            const row = createFieldRow();
            fields.append(row);
            this.markDirty();
            row.querySelector('input[name="field_label"]')?.focus();
            return true;
        }
        if (button instanceof HTMLButtonElement && button.matches('[data-character-apply-presets]')) {
            this.#applyPresets(button);
            return true;
        }
        const removeField = target.closest('[data-character-remove-field]');
        if (removeField) {
            removeField.closest('[data-character-field]')?.remove();
            this.markDirty();
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
        if (target.closest('[data-character-editor-form]')) this.markDirty();
        const editor = target.closest('[data-spelling-editor]');
        if (editor instanceof HTMLElement) this.syncSpellingEditor(editor);
    }

    handleChange(target) {
        if (!(target instanceof Element)) return;
        if (target.closest('[data-character-editor-form]')) this.markDirty();
        const style = target.closest('select[name="field_kind"]');
        if (style instanceof HTMLSelectElement) this.#syncFieldValueControl(style.closest('[data-character-field]'));
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

    #applyPresets(button) {
        const editor = button.closest('[data-character-preset-editor]');
        const form = button.closest('[data-character-editor-form]');
        const fields = form?.querySelector('[data-character-fields]');
        const base = editor?.querySelector('[data-character-base-preset]');
        if (!(editor instanceof HTMLElement) || !(fields instanceof HTMLElement) || !(base instanceof HTMLSelectElement)) return;
        const selectedBase = base.selectedOptions.item(0);
        const appliedBaseIds = new Set(
            Array.from(fields.querySelectorAll('[data-character-preset-layer="base"]')).flatMap((row) => (
                row instanceof HTMLElement && row.dataset.characterPresetId ? [row.dataset.characterPresetId] : []
            )),
        );
        if (!base.value || appliedBaseIds.size !== 1 || !appliedBaseIds.has(base.value)) {
            removePresetRows(fields, 'base');
            if (selectedBase instanceof HTMLOptionElement && base.value) {
                addPresetRows(fields, base.value, 'base', presetFields(selectedBase));
            }
        }
        const extras = Array.from(editor.querySelectorAll('[data-character-extra-preset]')).filter(
            (control) => control instanceof HTMLInputElement,
        );
        const selectedIds = new Set(extras.filter((control) => control.checked).map((control) => control.value));
        removePresetRows(fields, 'extra', selectedIds);
        extras.filter((control) => control.checked).forEach((control) => {
            addPresetRows(fields, control.value, 'extra', presetFields(control));
        });
        this.markDirty();
    }

    #syncFieldValueControl(row) {
        if (!(row instanceof HTMLElement)) return;
        const style = row.querySelector('select[name="field_kind"]');
        const current = row.querySelector('[data-character-field-value]');
        if (!(style instanceof HTMLSelectElement) || !(current instanceof HTMLInputElement || current instanceof HTMLTextAreaElement)) return;
        if ((style.value === 'long_text') === (current instanceof HTMLTextAreaElement)) return;
        current.replaceWith(createValueControl(style.value, current.value));
    }
}
