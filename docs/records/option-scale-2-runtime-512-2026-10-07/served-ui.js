
(() => {
  'use strict';
  const maxBodyBytes = 16 * 1024 * 1024;
  const maxOptions = 512;
  const tokenInput = document.querySelector('#token');
  const connectButton = document.querySelector('#connect');
  const modelStatus = document.querySelector('#model-status');
  const connectionStatus = document.querySelector('#connection-status');
  const optionsElement = document.querySelector('#options');
  const addOptionButton = document.querySelector('#add-option');
  const bulkOptions = document.querySelector('#bulk-options');
  const importOptionsButton = document.querySelector('#import-options');
  const submitButton = document.querySelector('#submit');
  const decisionForm = document.querySelector('#decision-form');
  const imageInput = document.querySelector('#image');
  const imageStatus = document.querySelector('#image-status');
  const formStatus = document.querySelector('#form-status');
  const results = document.querySelector('#results');
  const selected = document.querySelector('#selected');
  const probabilities = document.querySelector('#probabilities');
  const timings = document.querySelector('#timings');
  const rawResponse = document.querySelector('#raw-response');
  let authToken = '';
  let modelId = '';
  let imageValue = null;

  const setStatus = (element, message, kind = '') => {
    element.textContent = message;
    element.className = 'status' + (kind ? ' ' + kind : '');
  };

  const addOption = (key = '', description = '') => {
    const row = document.createElement('div');
    row.className = 'row';
    const keyLabel = document.createElement('label');
    keyLabel.textContent = 'Key';
    const keyInput = document.createElement('input');
    keyInput.className = 'option-key';
    keyInput.required = true;
    keyInput.value = key;
    keyInput.placeholder = 'keep';
    keyLabel.append(keyInput);
    const descriptionLabel = document.createElement('label');
    descriptionLabel.textContent = 'Description (optional)';
    const descriptionInput = document.createElement('input');
    descriptionInput.className = 'option-description';
    descriptionInput.value = description;
    descriptionInput.placeholder = 'Keep the service running';
    descriptionLabel.append(descriptionInput);
    const removeButton = document.createElement('button');
    removeButton.className = 'danger';
    removeButton.type = 'button';
    removeButton.textContent = 'Remove';
    removeButton.addEventListener('click', () => row.remove());
    row.append(keyLabel, descriptionLabel, removeButton);
    optionsElement.append(row);
  };
  addOption('keep', 'Keep it running');
  addOption('stop', 'Stop it');
  addOptionButton.addEventListener('click', () => {
    if (optionsElement.children.length < maxOptions) addOption();
  });
  importOptionsButton.addEventListener('click', () => {
    const lines = bulkOptions.value.split(/\r?\n/).filter(line => line.trim());
    const imported = lines.map(line => {
      const tab = line.indexOf('\t');
      return tab < 0 ? [line.trim(), ''] : [line.slice(0, tab).trim(), line.slice(tab + 1).trim()];
    });
    if (imported.length < 2 || imported.length > maxOptions ||
        imported.some(row => !row[0]) || new Set(imported.map(row => row[0])).size !== imported.length) {
      setStatus(formStatus, 'Paste 2–' + maxOptions + ' unique, nonempty option keys.', 'error');
      return;
    }
    optionsElement.replaceChildren();
    imported.forEach(row => addOption(...row));
    setStatus(formStatus, 'Loaded ' + imported.length + ' options.');
  });
  tokenInput.addEventListener('input', () => {
    authToken = '';
    modelId = '';
    modelStatus.textContent = 'Not connected';
  });

  const request = async (url, init = {}) => {
    const headers = new Headers(init.headers || {});
    if (authToken) headers.set('Authorization', 'Bearer ' + authToken);
    return fetch(url, { ...init, headers });
  };
  const responseError = async (response) => {
    let payload = null;
    try { payload = await response.json(); } catch (_) {}
    const detail = payload && payload.error ? payload.error.message : response.statusText;
    const code = payload && payload.error ? ' (' + payload.error.code + ')' : '';
    throw new Error(response.status + code + ': ' + (detail || 'request failed'));
  };
  connectButton.addEventListener('click', async () => {
    connectButton.disabled = true;
    setStatus(connectionStatus, 'Connecting...');
    authToken = tokenInput.value.trim();
    try {
      const response = await request('/healthz');
      if (!response.ok) await responseError(response);
      const payload = await response.json();
      modelId = typeof payload.model === 'string' ? payload.model : '';
      if (!modelId) throw new Error('health response did not include a model');
      modelStatus.textContent = 'Ready: ' + modelId;
      setStatus(connectionStatus, 'Connected', 'ok');
    } catch (error) {
      authToken = '';
      modelId = '';
      modelStatus.textContent = 'Not connected';
      setStatus(connectionStatus, error.message, 'error');
    } finally { connectButton.disabled = false; }
  });

  imageInput.addEventListener('change', () => {
    imageValue = null;
    const file = imageInput.files && imageInput.files[0];
    if (!file) { imageStatus.textContent = ''; return; }
    if (file.type !== 'image/png' && file.type !== 'image/jpeg') {
      imageInput.value = '';
      imageStatus.textContent = 'Choose a PNG or JPEG file.';
      return;
    }
    imageStatus.textContent = file.name + ' selected';
  });
  const fileAsImage = (file) => new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () => reject(new Error('could not read image file'));
    reader.onload = () => {
      const value = String(reader.result || '');
      const comma = value.indexOf(',');
      if (comma < 0) reject(new Error('could not encode image file'));
      else resolve({ media_type: file.type, data_base64: value.slice(comma + 1) });
    };
    reader.readAsDataURL(file);
  });

  const appendTiming = (label, value) => {
    const dt = document.createElement('dt');
    dt.textContent = label;
    const dd = document.createElement('dd');
    dd.textContent = String(value);
    timings.append(dt, dd);
  };
  const renderResponse = (payload) => {
    results.hidden = false;
    probabilities.replaceChildren();
    timings.replaceChildren();
    const answer = payload.answers && payload.answers.decision;
    selected.textContent = answer && typeof answer.choice === 'string' ? answer.choice : 'Unavailable';
    const raw = payload.branchscore && payload.branchscore.questions && payload.branchscore.questions.decision;
    const logits = raw && raw.raw_logits ? raw.raw_logits : {};
    const values = answer && answer.probabilities ? answer.probabilities : {};
    Object.keys(values).forEach((key) => {
      const probability = Number(values[key]);
      const row = document.createElement('div');
      row.className = 'probability';
      const name = document.createElement('span');
      name.textContent = key;
      const bar = document.createElement('div');
      bar.className = 'bar';
      const fill = document.createElement('span');
      fill.style.width = Math.max(0, Math.min(100, probability * 100)) + '%';
      bar.append(fill);
      const value = document.createElement('span');
      value.textContent = probability.toFixed(4) + ' | logit ' + (Number(logits[key]).toFixed(4));
      row.append(name, bar, value);
      probabilities.append(row);
    });
    appendTiming('Input tokens', payload.usage && payload.usage.input_tokens);
    appendTiming('HTTP handling (ms)', payload.branchscore && payload.branchscore.http_handling_ms);
    if (raw && raw.timings_ms) Object.entries(raw.timings_ms).forEach(([key, value]) => appendTiming(key, value));
    rawResponse.textContent = JSON.stringify(payload, null, 2);
  };

  decisionForm.addEventListener('submit', async (event) => {
    event.preventDefault();
    setStatus(formStatus, '');
    results.hidden = true;
    if (!modelId || !authToken && tokenInput.value.trim() !== '') {
      setStatus(formStatus, 'Connect before evaluating.', 'error');
      return;
    }
    const state = document.querySelector('#state').value.trim();
    const instructions = document.querySelector('#instructions').value.trim();
    const rows = [...optionsElement.querySelectorAll('.row')];
    const criteria = Object.create(null);
    if (!state || !instructions || rows.length < 2 || rows.length > maxOptions) {
      setStatus(formStatus, 'Enter state, question, and 2–' + maxOptions + ' options.', 'error');
      return;
    }
    for (const row of rows) {
      const key = row.querySelector('.option-key').value.trim();
      const description = row.querySelector('.option-description').value.trim();
      if (!key || Object.prototype.hasOwnProperty.call(criteria, key)) {
        setStatus(formStatus, 'Option keys must be nonempty and unique.', 'error');
        return;
      }
      criteria[key] = description ? description : null;
    }
    submitButton.disabled = true;
    addOptionButton.disabled = true;
    importOptionsButton.disabled = true;
    try {
      if (imageInput.files && imageInput.files[0]) imageValue = await fileAsImage(imageInput.files[0]);
      const payload = { state, model: modelId, questions: { decision: { type: 'choice', instructions, criteria } } };
      if (imageValue) payload.image = imageValue;
      const body = JSON.stringify(payload);
      if (new TextEncoder().encode(body).byteLength > maxBodyBytes) throw new Error('request exceeds the 16 MiB body limit');
      const response = await request('/v1/systemone', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body });
      if (!response.ok) await responseError(response);
      renderResponse(await response.json());
      setStatus(formStatus, 'Evaluation complete.', 'ok');
    } catch (error) { setStatus(formStatus, error.message, 'error'); }
    finally { submitButton.disabled = false; addOptionButton.disabled = false; importOptionsButton.disabled = false; }
  });
})();
