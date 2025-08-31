// Game State Annotation Tool - Stateless Frontend
// Uses: /api/sessions, /api/frame?session_id&idx, /api/image?session_id&idx, /api/save_annotation

(function () {
  'use strict';

  // Determine app base prefix (supports mounting under /annotation or other prefix)
  const APP_BASE = (() => {
    // If a global is provided, honor it
    if (window.APP_BASE && typeof window.APP_BASE === 'string') return window.APP_BASE.replace(/\/$/, '');
    try {
      const path = window.location.pathname || '/';
      // If served from /something/static/..., take the part before /static/
      const idx = path.indexOf('/static/');
      if (idx > 0) return path.substring(0, idx);
      // Fallback: if path starts with /annotation, use it
      if (path.startsWith('/annotation')) return '/annotation';
      // Extra heuristic: infer from script src (e.g., /annotation/static/annotation.js)
      const script = document.currentScript || Array.from(document.getElementsByTagName('script')).find(s => (s.src||'').includes('/static/annotation.js'));
      if (script && script.src) {
        const u = new URL(script.src, window.location.origin);
        const sIdx = u.pathname.indexOf('/static/');
        if (sIdx > 0) return u.pathname.substring(0, sIdx);
        if (u.pathname.startsWith('/annotation')) return '/annotation';
      }
    } catch {}
    return '';
  })();

  // ---------- Multi-label UI ----------
  function renderMultilabelPanel() {
    const ml = document.getElementById('multilabel-panel');
    if (!ml) return;
    // Build checklist from state.datasetClasses
    const rows = state.datasetClasses || [];
    const body = document.createElement('div');
    body.className = 'panel';
    body.innerHTML = '<h3>Multi-label</h3>';
    const wrap = document.createElement('div');
    wrap.className = 'panel__body';
    wrap.style.display = 'grid';
    wrap.style.gridTemplateColumns = 'repeat(auto-fill,minmax(160px,1fr))';
    wrap.style.gap = '8px';
    rows.forEach(r => {
      const id = `ml-${r.id}`;
      const item = document.createElement('label');
      item.setAttribute('for', id);
      item.style.display = 'flex';
      item.style.alignItems = 'center';
      item.style.gap = '8px';
      item.innerHTML = `<input type="checkbox" id="${id}" data-class-id="${r.id}"><span>${r.name}</span>`;
      wrap.appendChild(item);
    });
    body.appendChild(wrap);
    ml.innerHTML = '';
    ml.appendChild(body);
  }

  function getSelectedMultilabelClassIds() {
    const checks = document.querySelectorAll('#multilabel-panel input[type="checkbox"][data-class-id]');
    const ids = [];
    checks.forEach(ch => { if (ch.checked) ids.push(parseInt(ch.getAttribute('data-class-id'), 10)); });
    return ids;
  }

  async function saveMultilabel() {
    if (!state.dataset_id) { toast('Select a dataset first'); return false; }
    const classIds = getSelectedMultilabelClassIds();
    if (!classIds.length) { toast('Select at least one label'); return false; }
    try {
      const payload = {
        session_id: state.session_id,
        dataset_id: state.dataset_id,
        frame_idx: state.currentIdx,
        class_ids: classIds,
      };
      const res = await apiPost('annotations/multilabel', payload);
      if (res && (res.ok || res.saved || res.status === 'ok')) {
        toast('Saved');
        refreshProgress();
        return true;
      }
      toast('Save failed');
      return false;
    } catch (e) {
      console.error(e);
      toast('Error while saving');
      return false;
    }
  }

  // ---------- Regression UI ----------
  function renderRegressionPanel() {
    const rg = document.getElementById('regression-panel');
    if (!rg) return;
    const body = document.createElement('div');
    body.className = 'panel';
    body.innerHTML = '<h3>Regression</h3>';
    const wrap = document.createElement('div');
    wrap.className = 'panel__body';
    wrap.style.display = 'flex';
    wrap.style.alignItems = 'center';
    wrap.style.gap = '12px';
    wrap.innerHTML = `
      <label style="min-width:80px">Value</label>
      <input id="regression-input" type="range" min="0" max="100" step="1" value="0" style="flex:1"> 
      <input id="regression-number" type="number" min="0" max="100" step="1" value="0" style="width:96px">
    `;
    body.appendChild(wrap);
    rg.innerHTML = '';
    rg.appendChild(body);
    // Sync number and range
    const range = rg.querySelector('#regression-input');
    const number = rg.querySelector('#regression-number');
    if (range && number) {
      range.addEventListener('input', () => { number.value = range.value; });
      number.addEventListener('input', () => { range.value = number.value; });
    }
  }

  function getRegressionValue() {
    const rg = document.getElementById('regression-panel');
    if (!rg) return null;
    const number = rg.querySelector('#regression-number');
    const v = number ? parseFloat(number.value) : NaN;
    return Number.isFinite(v) ? v : null;
  }

  async function saveRegression() {
    if (!state.dataset_id) { toast('Select a dataset first'); return false; }
    const v = getRegressionValue();
    if (v == null) { toast('Enter a value'); return false; }
    try {
      const payload = {
        session_id: state.session_id,
        dataset_id: state.dataset_id,
        frame_idx: state.currentIdx,
        value: v,
      };
      const res = await apiPost('annotations/regression', payload);
      if (res && (res.ok || res.saved || res.status === 'ok')) {
        toast('Saved');
        refreshProgress();
        return true;
      }
      toast('Save failed');
      return false;
    } catch (e) {
      console.error(e);
      toast('Error while saving');
      return false;
    }
  }

  // Simple cookie helpers for persistence across reloads (complementing localStorage)
  function setCookie(name, value, days = 365) {
    try {
      const d = new Date();
      d.setTime(d.getTime() + (days*24*60*60*1000));
      const expires = 'expires=' + d.toUTCString();
      document.cookie = `${encodeURIComponent(name)}=${encodeURIComponent(value)};${expires};path=/`;
    } catch {}
  }
  function getCookie(name) {
    try {
      const key = encodeURIComponent(name) + '=';
      const ca = document.cookie.split(';');
      for (let c of ca) {
        while (c.charAt(0) === ' ') c = c.substring(1);
        if (c.indexOf(key) === 0) return decodeURIComponent(c.substring(key.length, c.length));
      }
    } catch {}
    return null;
  }

  // Modal helpers
  function getModal(id) { return document.getElementById(id); }
  function openModal(id) {
    const m = getModal(id);
    if (!m) return;
    try {
      // Ensure modal is under <body> to avoid stacking context issues
      if (m.parentElement !== document.body) {
        document.body.appendChild(m);
      }
    } catch {}
    m.classList.remove('hidden');
    m.querySelectorAll('[data-modal-close]').forEach(btn => btn.onclick = () => closeModal(id));
    const backdrop = m.querySelector('.modal__backdrop');
    if (backdrop) backdrop.onclick = () => closeModal(id);
    // Simple debug marker
    try { m.setAttribute('data-opened', String(Date.now())); } catch {}
  }
  function closeModal(id) {
    const m = getModal(id);
    if (m) m.classList.add('hidden');
  }

  // Helper: full dataset modal to enter name/desc and pick target type
  // Returns Promise<{ name: string, description: string, target_type_id: number }|null>
  async function showDatasetModal(opts = { title: 'Create Dataset', name: '', description: '', target_type_id: 1 }) {
    const types = await listTargetTypes().catch(() => []);
    if (!Array.isArray(types) || types.length === 0) return null;
    const title = opts.title || 'Create Dataset';
    const wrap = document.createElement('div');
    wrap.className = 'modal';
    wrap.setAttribute('style', 'position:fixed;inset:0;z-index:10000;display:flex;align-items:center;justify-content:center;');
    wrap.innerHTML = `
      <div class="modal__backdrop" style="position:absolute;inset:0;"></div>
      <div class="modal__content" style="position:relative;max-width:720px;width:90vw;max-height:85vh;overflow:auto">
        <div class="modal__header"><h3>${title}</h3></div>
        <div class="modal__body" style="display:flex;flex-direction:column;gap:12px">
          <input class="input" id="ds-name" placeholder="Dataset name" value="${opts.name || ''}">
          <input class="input" id="ds-desc" placeholder="Description (optional)" value="${opts.description || ''}">
          <select class="input" id="ds-type"></select>
        </div>
        <div class="modal__footer" style="display:flex;gap:8px;justify-content:flex-end">
          <button class="btn" id="ds-cancel">Cancel</button>
          <button class="btn btn--primary" id="ds-save">Save</button>
        </div>
      </div>`;
    document.body.appendChild(wrap);
    const typeSel = wrap.querySelector('#ds-type');
    typeSel.innerHTML = '';
    types.forEach(t => {
      const opt = document.createElement('option');
      opt.value = String(t.id);
      opt.textContent = t.name || `Type ${t.id}`;
      typeSel.appendChild(opt);
    });
    const preferred = (opts.target_type_id != null) ? String(opts.target_type_id) : (types.find(t => String(t.id) === '1') ? '1' : String(types[0].id));
    typeSel.value = preferred;
    wrap.classList.remove('hidden');
    const contentEl = wrap.querySelector('.modal__content');
    if (contentEl) contentEl.addEventListener('click', (e) => e.stopPropagation());
    return await new Promise((resolve) => {
      const cleanup = () => { try { document.body.removeChild(wrap); } catch {} };
      const cancel = () => { cleanup(); resolve(null); };
      wrap.querySelector('#ds-cancel').onclick = cancel;
      const backdrop = wrap.querySelector('.modal__backdrop');
      if (backdrop) backdrop.onclick = cancel;
      wrap.querySelector('#ds-save').onclick = () => {
        const name = (wrap.querySelector('#ds-name').value || '').trim();
        const description = (wrap.querySelector('#ds-desc').value || '').trim();
        const ttid = parseInt(typeSel.value, 10);
        cleanup();
        resolve({ name, description, target_type_id: ttid });
      };
    });
  }

  // Helper: modal picker for target types (returns Promise<number|null>)
  async function chooseTargetType(defaultId = 1) {
    try {
      const types = await listTargetTypes();
      if (!Array.isArray(types) || types.length === 0) return null;
      // Build transient modal
      const wrap = document.createElement('div');
      wrap.className = 'modal';
      wrap.setAttribute('style', 'position:fixed;inset:0;z-index:10000;display:flex;align-items:center;justify-content:center;');
      wrap.innerHTML = `
        <div class="modal__backdrop" style="position:absolute;inset:0;"></div>
        <div class="modal__content" style="position:relative;max-width:420px">
          <div class="modal__header"><h3>Select Target Type</h3></div>
          <div class="modal__body">
            <select id="tt-select" class="input" style="width:100%"></select>
          </div>
          <div class="modal__footer" style="display:flex;gap:8px;justify-content:flex-end">
            <button class="btn" id="tt-cancel">Cancel</button>
            <button class="btn btn--primary" id="tt-ok">OK</button>
          </div>
        </div>`;
      document.body.appendChild(wrap);
      const sel = wrap.querySelector('#tt-select');
      types.forEach(t => {
        const opt = document.createElement('option');
        opt.value = String(t.id);
        opt.textContent = t.name || `Type ${t.id}`;
        sel.appendChild(opt);
      });
      sel.value = String(defaultId);
      wrap.classList.remove('hidden');
      const contentEl = wrap.querySelector('.modal__content');
      if (contentEl) contentEl.addEventListener('click', (e) => e.stopPropagation());
      // Return promise for selection
      return await new Promise((resolve) => {
        const cleanup = () => { try { document.body.removeChild(wrap); } catch {} };
        wrap.querySelector('#tt-cancel').onclick = () => { cleanup(); resolve(null); };
        wrap.querySelector('#tt-ok').onclick = () => {
          const v = parseInt(sel.value, 10);
          cleanup();
          resolve(Number.isFinite(v) ? v : null);
        };
        const backdrop = wrap.querySelector('.modal__backdrop');
        if (backdrop) backdrop.onclick = () => { cleanup(); resolve(null); };
      });
    } catch {
      return null;
    }
  }

  // Project Management UI
  async function renderProjectManager() {
    const body = document.getElementById('project-modal-body');
    if (!body) return;
    body.innerHTML = '';
    // Create form
    const form = document.createElement('div');
    form.className = 'm-row';
    form.innerHTML = `
      <input class="input m-input" id="pm-name" placeholder="Project name">
      <input class="input m-input" id="pm-desc" placeholder="Description (optional)">
      <button class="btn" id="pm-create">Create</button>
    `;
    body.appendChild(form);
    const listWrap = document.createElement('div');
    listWrap.className = 'm-list';
    body.appendChild(listWrap);
    const projects = await listProjects().catch(() => []);
    projects.forEach(p => {
      const row = document.createElement('div');
      row.className = 'm-item';
      row.innerHTML = `
        <div>
          <div class="m-item__title">${p.name}</div>
          <div class="m-item__meta">${p.description || ''}</div>
        </div>
        <div class="m-row">
          <button class="btn" data-edit>Rename</button>
          <button class="btn" data-delete>Delete</button>
        </div>
      `;
      // actions
      row.querySelector('[data-edit]').onclick = async () => {
        const newName = prompt('New project name', p.name) || p.name;
        const newDesc = prompt('Description (optional)', p.description || '') || '';
        try {
          await apiPut(`projects/${p.id}`, { name: newName, description: newDesc });
          await populateProjectSelect(newName);
          await populateDatasetSelect();
          await renderProjectManager();
          toast('Project updated');
        } catch { toast('Update failed'); }
      };
      row.querySelector('[data-delete]').onclick = async () => {
        if (!confirm(`Delete project '${p.name}'?`)) return;
        let url = `projects/${p.id}`;
        const force = confirm('Force delete and cascade all datasets and their annotations?');
        if (force) url += '?force=1';
        try {
          await apiDelete(url);
          await populateProjectSelect();
          await populateDatasetSelect();
          await renderProjectManager();
          toast('Project deleted');
        } catch (e) { toast('Delete failed'); }
      };
      listWrap.appendChild(row);
    });
    // Create handler
    form.querySelector('#pm-create').onclick = async () => {
      const name = (form.querySelector('#pm-name').value || '').trim();
      const desc = (form.querySelector('#pm-desc').value || '').trim();
      if (!name) { toast('Name required'); return; }
      try {
        await createProject(name, desc);
        await populateProjectSelect(name);
        await populateDatasetSelect();
        await renderProjectManager();
        await loadSessions();
        toast('Project created');
      } catch { toast('Create failed'); }
    };
  }
  ;

  // Dataset Management UI
  async function renderDatasetManager() {
    const body = document.getElementById('dataset-modal-body');
    if (!body) return;
    body.innerHTML = '';
    const projectId = getSelectedProjectId();
    if (!projectId) { body.innerHTML = '<div class="m-item__meta">Select a project first.</div>'; return; }
    // Create form
    const form = document.createElement('div');
    form.className = 'm-row';
    form.innerHTML = `
      <input class="input m-input" id="dm-name" placeholder="Dataset name">
      <input class="input m-input" id="dm-desc" placeholder="Description (optional)">
      <button class="btn" id="dm-create">Create</button>
    `;
    body.appendChild(form);
    const listWrap = document.createElement('div');
    listWrap.className = 'm-list';
    body.appendChild(listWrap);
    const datasets = await listDatasets(projectId).catch(() => []);
    datasets.forEach(d => {
      const row = document.createElement('div');
      row.className = 'm-item';
      const typeLabel = d.target_type_name || `Type ${d.target_type_id}`;
      row.innerHTML = `
        <div>
          <div class="m-item__title">${d.name}</div>
          <div class="m-item__meta">${d.description || ''} · ${typeLabel}</div>
        </div>
        <div class="m-row">
          <button class="btn" data-edit>Edit</button>
          <button class="btn" data-delete>Delete</button>
          <button class="btn" data-select>Select</button>
        </div>
      `;
      row.querySelector('[data-edit]').onclick = async () => {
        const result = await showDatasetModal({ title: 'Edit Dataset', name: d.name, description: d.description || '', target_type_id: d.target_type_id });
        if (!result) return;
        const { name: newName, description: newDesc, target_type_id } = result;
        try {
          await apiPut(`datasets/${d.id}`, { name: newName || d.name, description: newDesc || '', target_type_id });
          await populateDatasetSelect(String(d.id));
          // Refresh target-type UI and sessions if this dataset is selected
          const details = await fetchDatasetDetails(d.id);
          if (String(els.datasetSelect().value) === String(d.id)) {
            state.dataset_id = d.id;
            state.dataset_name = newName || d.name;
            state.target_type_name = details ? details.target_type_name : null;
            await loadDatasetClasses(state.dataset_id);
            applyModeVisibility();
            refreshProgress();
            await loadSessions();
          }
          await renderDatasetManager();
          toast('Dataset updated');
        } catch { toast('Update failed'); }
      };
      row.querySelector('[data-delete]').onclick = async () => {
        if (!confirm(`Delete dataset '${d.name}'?`)) return;
        let url = `datasets/${d.id}`;
        const force = confirm('Force delete and cascade all annotations in this dataset?');
        if (force) url += '?force=1';
        try {
          await apiDelete(url);
          await populateDatasetSelect();
          await renderDatasetManager();
          await loadSessions();
          toast('Dataset deleted');
        } catch { toast('Delete failed'); }
      };
      row.querySelector('[data-select]').onclick = async () => {
        // Select this dataset in toolbar
        const dsSel = els.datasetSelect();
        if (dsSel) {
          dsSel.value = String(d.id);
          const opt = Array.from(dsSel.options).find(o => o.value === String(d.id));
          state.dataset_id = d.id;
          state.dataset_name = d.name;
          if (opt) {
            state.dataset_name = opt.dataset.datasetName || d.name;
          }
          if (state.session_id && state.dataset_id) {
            try { localStorage.setItem(`dataset:${state.session_id}`, JSON.stringify({ id: state.dataset_id, name: state.dataset_name || '' })); } catch {}
          }
          // Update dataset-driven UI and sessions
          const details = await fetchDatasetDetails(state.dataset_id);
          state.target_type_name = details ? details.target_type_name : null;
          await loadDatasetClasses(state.dataset_id);
          applyModeVisibility();
          refreshProgress();
          await loadSessions();
          closeModal('dataset-modal');
          toast('Dataset selected');
        }
      };
      listWrap.appendChild(row);
    });
    // Create handler
    form.querySelector('#dm-create').onclick = async () => {
      const defaults = {
        title: 'Create Dataset',
        name: (form.querySelector('#dm-name').value || '').trim(),
        description: (form.querySelector('#dm-desc').value || '').trim(),
        target_type_id: 1,
      };
      const result = await showDatasetModal(defaults);
      if (!result) return;
      const { name, description, target_type_id } = result;
      if (!name) { toast('Name required'); return; }
      try {
        const created = await createDataset(projectId, name, description, target_type_id);
        await populateDatasetSelect(String(created.id));
        // Apply target-type UI and classes for the newly selected dataset
        const details = await fetchDatasetDetails(created.id);
        state.dataset_id = created.id;
        state.dataset_name = name;
        state.target_type_name = details ? details.target_type_name : null;
        await loadDatasetClasses(state.dataset_id);
        applyModeVisibility();
        refreshProgress();
        await loadSessions();
        await renderDatasetManager();
        toast('Dataset created');
      } catch { toast('Create failed'); }
    };
  }

  function withBase(p) { return `${APP_BASE}${p}`; }

  // App state
  let state = {
    session_id: null,
    project_id: null,
    project_name: 'default',
    dataset_id: null,
    dataset_name: null,
    currentIdx: 0,
    totalFrames: 0,
    categories: [],
    hotkeys: {},
    // dataset class info from backend
    datasetClasses: [], // [{id,name,idx}]
    classIdByName: {}, // { name: id }
    savedCategoryForFrame: null,
    frameSaved: false,
    settingsLoaded: false,
  };

  const els = {
    sessionList: () => document.getElementById('session-list'),
    sessionSelector: () => document.getElementById('session-selector'),
    projectSelect: () => document.getElementById('project-select'),
    datasetSelect: () => document.getElementById('dataset-select'),
    annotationInterface: () => document.getElementById('annotation-interface'),
    sessionInfo: () => document.getElementById('session-info'),
    sessionName: () => document.getElementById('session-name'),
    progressFill: () => document.getElementById('progress-fill'),
    progressText: () => document.getElementById('progress-text'),

    img: () => document.getElementById('frame-image'),
    frameId: () => document.getElementById('frame-id'),
    frameFilename: () => document.getElementById('frame-filename'),
    frameTimestamp: () => document.getElementById('frame-timestamp'),

    firstBtn: () => document.getElementById('first-btn'),
    prevBtn: () => document.getElementById('prev-btn'),
    nextBtn: () => document.getElementById('next-btn'),
    lastBtn: () => document.getElementById('last-btn'),
    frameInput: () => document.getElementById('frame-input'),

    categoryList: () => document.getElementById('category-list'),
    newCategoryInput: () => document.getElementById('new-category-input'),
    addCategoryBtn: () => document.getElementById('add-category-btn'),
    notes: () => document.getElementById('notes-input'),

    saveBtn: () => document.getElementById('save-btn'),
    saveNextBtn: () => document.getElementById('save-next-btn'),
    skipBtn: () => document.getElementById('skip-btn'),
    statsGrid: () => document.getElementById('stats-grid'),
    dynamicShortcuts: () => document.getElementById('dynamic-shortcuts'),
  };

  // API client
  async function apiGet(url, params = {}) {
    const usp = new URLSearchParams(params);
    const full = url.startsWith('/api') ? url : `/api/${String(url).replace(/^\/?/, '')}`;
    const qs = usp.toString();
    const target = qs ? `${withBase(full)}?${qs}` : withBase(full);
    console.log(target);
    const res = await fetch(target);
    if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
    return res.json();
  }

  async function apiPut(url, body) {
    const full = url.startsWith('/api') ? url : `/api/${String(url).replace(/^\/?/, '')}`;
    const res = await fetch(withBase(full), {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body || {}),
    });
    if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
    return res.json();
  }

  async function apiDelete(url) {
    const full = url.startsWith('/api') ? url : `/api/${String(url).replace(/^\/?/, '')}`;
    const res = await fetch(withBase(full), { method: 'DELETE' });
    if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
    return res.json();
  }

  async function ensureDatasetSelected(sessionId) {
    // If dataset already chosen via toolbar, accept it
    if (Number.isFinite(Number(state.dataset_id)) && Number(state.dataset_id) > 0) {
      return { id: Number(state.dataset_id), name: state.dataset_name || null };
    }
    // Try restore
    try {
      const saved = localStorage.getItem(`dataset:${sessionId}`);
      if (saved) {
        const obj = JSON.parse(saved);
        if (obj && obj.id != null) {
          const n = typeof obj.id === 'string' ? parseInt(obj.id, 10) : obj.id;
          if (Number.isFinite(n) && n > 0) {
            state.dataset_id = n; state.dataset_name = obj.name || null; return { id: n, name: state.dataset_name };
          }
        }
      }
    } catch {}

    // Prompt choose or create project/dataset, and enroll idempotently
    await startEnrollmentFlow(sessionId);
    // After enrollment, datasetId will be last created/selected. We need to re-read from storage
    try {
      const saved2 = localStorage.getItem(`dataset:${sessionId}`);
      if (saved2) {
        const obj2 = JSON.parse(saved2);
        if (obj2 && obj2.id != null) {
          const n2 = typeof obj2.id === 'string' ? parseInt(obj2.id, 10) : obj2.id;
          if (Number.isFinite(n2) && n2 > 0) {
            state.dataset_id = n2; state.dataset_name = obj2.name || null; return { id: n2, name: state.dataset_name };
          }
        }
      }
    } catch {}
    return null;
  }

  async function fetchDatasetSessionSettings(datasetId, sessionId) {
    try {
      const invalid = (v) => v == null || v === 'null' || v === 'undefined' || Number.isNaN(Number(v)) || Number(v) <= 0;
      if (invalid(datasetId)) return {};
      const dsid = Number(datasetId);
      const url = `/api/datasets/${datasetId}/sessions/${encodeURIComponent(sessionId)}/settings`;
      const res = await fetch(withBase(url));
      if (!res.ok) throw new Error('settings fetch failed');
      const data = await res.json();
      return data.settings || {};
    } catch (e) {
      console.warn('No settings for dataset/session, using defaults');
      return {};
    }
  }

  async function apiPost(url, body) {
    const full = url.startsWith('/api') ? url : `/api/${String(url).replace(/^\/?/, '')}`;
    const res = await fetch(withBase(full), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
    return res.json();
  }

  async function apiPostNoBody(url) {
    const full = url.startsWith('/api') ? url : `/api/${String(url).replace(/^\/?/, '')}`;
    const res = await fetch(withBase(full), { method: 'POST' });
    if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
    return res.json();
  }

  // Projects & Datasets API helpers (DB-backed)
  async function listProjects() {
    return apiGet('projects');
  }
  async function createProject(name, description = '') {
    return apiPost('projects', { name, description });
  }
  async function listDatasets(projectId) {
    return apiGet(`projects/${projectId}/datasets`);
  }
  async function createDataset(projectId, name, description = '', target_type_id = 1) {
    // DB schema target_types: 0=Regression, 1=SingleLabelClassification, 2=MultiLabelClassification
    return apiPost(`projects/${projectId}/datasets`, { name, description, target_type_id });
  }
  async function listTargetTypes() {
    return apiGet('target_types');
  }
  async function enrollSession(datasetId, sessionId, settings = undefined) {
    // New paradigm: backend enrollment no longer accepts/apply baseline settings
    return apiPost(`datasets/${datasetId}/enroll_session`, { session_id: sessionId });
  }
  async function datasetProgress(datasetId) {
    return apiGet(`datasets/${datasetId}/progress`);
  }

  async function fetchDatasetDetails(datasetId) {
    const invalid = (v) => v == null || v === 'null' || v === 'undefined' || Number.isNaN(Number(v)) || Number(v) <= 0;
    if (invalid(datasetId)) return null;
    return apiGet(`datasets/${datasetId}`);
  }

  // Enrollment helpers
  async function listDatasetEnrollments(datasetId) {
    const invalid = (v) => v == null || v === 'null' || v === 'undefined' || Number.isNaN(Number(v)) || Number(v) <= 0;
    if (invalid(datasetId)) return [];
    // Returns an array of session_ids enrolled in this dataset
    return apiGet(`datasets/${datasetId}/enrollments`).catch(() => []);
  }

  // --- Mode toggle scaffolding ---
  function ensureModePanels() {
    const root = els.annotationInterface && els.annotationInterface();
    if (!root) return;
    // Create placeholders once
    if (!document.getElementById('multilabel-panel')) {
      const ml = document.createElement('div');
      ml.id = 'multilabel-panel';
      ml.className = 'hidden';
      ml.innerHTML = '<div class="panel"><h3>Multi-label</h3><div class="panel__body">Multiple selection UI will appear here.</div></div>';
      root.appendChild(ml);
    }
    if (!document.getElementById('regression-panel')) {
      const rg = document.createElement('div');
      rg.id = 'regression-panel';
      rg.className = 'hidden';
      rg.innerHTML = '<div class="panel"><h3>Regression</h3><div class="panel__body">Slider and keypoints will appear here.</div></div>';
      root.appendChild(rg);
    }
  }

  function applyModeVisibility() {
    ensureModePanels();
    const t = state.target_type_name;
    const isSingle = (t === 'SingleLabelClassification') || (!t); // default to single-label if unknown
    const isMulti = (t === 'MultiLabelClassification');
    const isRegr = (t === 'Regression');

    const catList = els.categoryList && els.categoryList();
    const addBtn = els.addCategoryBtn && els.addCategoryBtn();
    const newCat = els.newCategoryInput && els.newCategoryInput();
    const shortcuts = els.dynamicShortcuts && els.dynamicShortcuts();
    const mlPanel = document.getElementById('multilabel-panel');
    const rgPanel = document.getElementById('regression-panel');

    // Show/hide SingleLabel controls
    [catList, addBtn, newCat, shortcuts].forEach(el => {
      if (!el) return;
      if (isSingle) el.classList.remove('hidden'); else el.classList.add('hidden');
    });
    // Show/hide placeholders for other modes
    if (mlPanel) { if (isMulti) mlPanel.classList.remove('hidden'); else mlPanel.classList.add('hidden'); }
    if (rgPanel) { if (isRegr) rgPanel.classList.remove('hidden'); else rgPanel.classList.add('hidden'); }
    // Render dynamic contents for panels when visible
    if (isMulti) renderMultilabelPanel();
    if (isRegr) renderRegressionPanel();
  }

  async function listDatasetClasses(datasetId) {
    if (!datasetId) return [];
    return apiGet(`datasets/${datasetId}/classes`);
  }

  async function loadDatasetClasses(datasetId) {
    try {
      const rows = await listDatasetClasses(datasetId);
      state.datasetClasses = Array.isArray(rows) ? rows : [];
      const map = {};
      state.datasetClasses.forEach(r => { if (r && typeof r.name === 'string') map[r.name] = r.id; });
      state.classIdByName = map;
    } catch (e) {
      console.warn('Failed to load dataset classes');
      state.datasetClasses = [];
      state.classIdByName = {};
    }
  }

  // Persistence helpers (localStorage per session/project)
  function lsKey(prefix) {
    if (!state.session_id) return `${prefix}:none`;
    return `${prefix}:${state.session_id}:${state.project_name}`;
  }

  function loadCategoriesFromStorage() {
    try {
      const cats = localStorage.getItem(lsKey('categories'));
      const hotkeys = localStorage.getItem(lsKey('hotkeys'));
      state.categories = cats ? JSON.parse(cats) : ['battle', 'menu', 'loading', 'other'];
      state.hotkeys = hotkeys ? JSON.parse(hotkeys) : {};
    } catch {
      state.categories = ['battle', 'menu', 'loading', 'other'];
      state.hotkeys = {};
    }
  }

  function saveCategoriesToStorage() {
    try {
      localStorage.setItem(lsKey('categories'), JSON.stringify(state.categories));
      localStorage.setItem(lsKey('hotkeys'), JSON.stringify(state.hotkeys));
    } catch {}
  }

  // Persist dataset-session settings to backend
  async function saveDatasetSessionSettings(datasetId, sessionId, settingsObj) {
    try {
      await apiPut(`datasets/${datasetId}/sessions/${encodeURIComponent(sessionId)}/settings`, { settings: settingsObj || {} });
    } catch (e) {
      // Non-fatal
    }
  }

  function saveSessionSelection() {
    try {
      localStorage.setItem('currentSession', state.session_id || '');
      localStorage.setItem('currentProjectId', state.project_id || '');
      localStorage.setItem('currentProject', state.project_name || 'default');
      if (state.dataset_id) {
        localStorage.setItem(`dataset:${state.session_id}`, JSON.stringify({ id: state.dataset_id, name: state.dataset_name || '' }));
      }
      localStorage.setItem('appState', 'annotation');
    } catch {}
  }

// Projects toolbar
async function populateProjectSelect(pref = null) {
const select = els.projectSelect();
if (!select) return;
select.innerHTML = '';
let projects = [];
try { projects = await listProjects(); } catch { projects = []; }
// No projects: prompt user to create the first one via modal
if (!projects || projects.length === 0) {
  state.project_id = null;
  state.project_name = null;
  // Keep selector empty and open project creation modal
  try {
    await renderProjectManager();
    openModal('project-modal');
    toast('Create your first project');
  } catch {}
  return;
}
projects.forEach(p => {
  const opt = document.createElement('option');
  opt.value = String(p.id);
  opt.textContent = p.name;
  opt.dataset.projectId = String(p.id);
  opt.dataset.projectName = p.name;
  select.appendChild(opt);
});
// Choose preferred or saved project (prefer ID)
const savedId = (pref && /^\d+$/.test(String(pref))) ? String(pref)
               : (localStorage.getItem('currentProjectId') || getCookie('currentProjectId') || null);
let chosen = null;
if (savedId) {
  chosen = Array.from(select.options).find(o => String(o.value) === String(savedId));
}
if (!chosen) {
  const savedName = (pref && !/^\d+$/.test(String(pref))) ? String(pref)
                  : (localStorage.getItem('currentProject') || getCookie('currentProject') || null);
  if (savedName) chosen = Array.from(select.options).find(o => (o.dataset.projectName === savedName));
}
if (!chosen) chosen = select.options[0];
if (chosen) {
  select.value = chosen.value;
  const opt = chosen;
  state.project_id = parseInt(opt.dataset.projectId, 10);
  state.project_name = opt.dataset.projectName || opt.textContent || 'default';
}
// Persist selection
try { setCookie('currentProjectId', String(state.project_id)); setCookie('currentProject', state.project_name); } catch {}
}

  function getSelectedProjectId() {
    const ps = els.projectSelect();
    if (!ps) return null;
    const opt = ps.selectedOptions && ps.selectedOptions[0];
    if (!opt) return null;
    const pid = parseInt(opt.dataset.projectId || ps.value, 10);
    return Number.isFinite(pid) ? pid : null;
  }

  async function populateDatasetSelect(prefId = null) {
    const select = els.datasetSelect();
    if (!select) return;
    select.innerHTML = '';
    const projectId = getSelectedProjectId();
    if (!projectId) {
      state.dataset_id = null;
      state.dataset_name = null;
      return;
    }
    let datasets = [];
    try { datasets = await listDatasets(projectId); } catch { datasets = []; }
    datasets.forEach(d => {
      const opt = document.createElement('option');
      opt.value = String(d.id);
      const tname = d.target_type_name || `Type ${d.target_type_id}`;
      opt.textContent = `${d.name} [${tname}]`;
      opt.dataset.datasetName = d.name;
      opt.dataset.targetTypeId = String(d.target_type_id);
      select.appendChild(opt);
    });
    if (datasets.length > 0) {
      let savedId = null;
      try {
        if (state.session_id) {
          const saved = localStorage.getItem(`dataset:${state.session_id}`);
          if (saved) {
            const obj = JSON.parse(saved);
            if (obj && obj.id != null) savedId = String(obj.id);
          }
        }
      } catch {}
      // Fallback to cookie if no per-session saved dataset
      let cookieId = getCookie && getCookie('currentDatasetId');
      if (cookieId && !datasets.some(d => String(d.id) === String(cookieId))) cookieId = null;
      const targetVal = prefId || savedId || cookieId || String(datasets[0].id);
      select.value = targetVal;
      const selOpt = select.selectedOptions[0];
      state.dataset_id = Number(select.value);
      state.dataset_name = selOpt ? (selOpt.dataset.datasetName || null) : null;
      // When auto-selected, also refresh dataset details and UI mode and sessions
      try { setCookie('currentDatasetId', String(state.dataset_id)); } catch {}
      const details = await fetchDatasetDetails(state.dataset_id);
      state.target_type_name = details ? details.target_type_name : null;
      await loadDatasetClasses(state.dataset_id);
      applyModeVisibility();
      refreshProgress();
      await loadSessions();
    } else {
      state.dataset_id = null;
      state.dataset_name = null;
    }
    // Persist to cookies when selected
    if (state.dataset_id) {
      try { setCookie('currentDatasetId', String(state.dataset_id)); } catch {}
    }
  }

  // UI rendering
  function renderSessionList(sessions, enrolledSet = null) {
    const list = els.sessionList();
    list.innerHTML = '';
    sessions.forEach((s) => {
      const div = document.createElement('div');
      const isEnrolled = !!(enrolledSet && enrolledSet.has(s.session_id));
      div.className = 'selector__item' + (isEnrolled ? ' selector__item--enrolled' : '');
      if (isEnrolled) {
        // Visual highlight for enrolled
        div.style.border = '2px solid #22c55e';
        div.style.boxShadow = '0 0 0 3px rgba(34,197,94,0.15)';
      }
      // Make the entire card actionable
      div.style.cursor = 'pointer';
      const hint = isEnrolled ? 'Click to open' : 'Not enrolled — click to enroll';
      div.innerHTML = `
        <h4>${s.session_id}</h4>
        <div class="selector__meta">Game: ${s.game_name || '-'} · Frames: ${s.frames_count || '-'}</div>
        <div class="selector__meta">Started: ${s.start_time ? new Date(s.start_time).toLocaleString() : '-'}</div>
        <div class="selector__meta" style="margin-top:6px;color:${isEnrolled ? '#15803d' : '#6b7280'}">${hint}</div>
      `;
      // Card click behavior
      div.addEventListener('click', async () => {
        const sel = els.projectSelect();
        let projName = state.project_name || 'default';
        let projId = state.project_id || null;
        if (sel && sel.selectedOptions && sel.selectedOptions[0]) {
          const opt = sel.selectedOptions[0];
          projName = opt.dataset.projectName || opt.textContent || projName;
          projId = parseInt(opt.dataset.projectId || sel.value, 10);
        }
        const dsSel = els.datasetSelect();
        if (dsSel && dsSel.value) {
          state.dataset_id = Number(dsSel.value);
          state.dataset_name = (dsSel.selectedOptions[0] && dsSel.selectedOptions[0].dataset.datasetName) || null;
        }
        if (isEnrolled) {
          selectSession(s.session_id, projName, { pushHistory: true, preload: s });
          return;
        }
        // Auto-enroll with currently selected dataset (no popups)
        if (!state.dataset_id) {
          toast('Select a dataset first');
          return;
        }
        try {
          await enrollSession(state.dataset_id, s.session_id);
          // Persist selection
          try {
            localStorage.setItem(`dataset:${s.session_id}`, JSON.stringify({ id: state.dataset_id, name: state.dataset_name || '' }));
            setCookie('currentProject', projName);
            if (projId != null) setCookie('currentProjectId', String(projId));
            setCookie('currentDatasetId', String(state.dataset_id));
          } catch {}
          await loadSessions();
          // Auto-open after enrollment
          selectSession(s.session_id, projName, { pushHistory: true, preload: s });
        } catch (e) {
          console.error(e);
          toast('Enrollment failed');
        }
      });
      list.appendChild(div);
    });
    if (sessions.length === 0) {
      list.innerHTML = '<div class="selector__meta">No annotation sessions found. Run data collection first.</div>';
    }
  }

  async function startEnrollmentFlow(sessionId) {
    try {
      // 1) Choose or create Project (prefill with toolbar selection if available)
      const projects = await listProjects();
      const choices = projects.map((p, i) => `${i + 1}) ${p.name}`).join('\n');
      const toolbarSelected = (els.projectSelect() && els.projectSelect().value) ? els.projectSelect().value : state.project_name || 'default';
      let sel = prompt(`Select a project by number or type 'new' to create:\n${choices}\n(Current: ${toolbarSelected})`);
      let projectId;
      if (sel && sel.toLowerCase() === 'new') {
        const name = prompt('New project name:');
        if (!name) return;
        const desc = prompt('Description (optional):') || '';
        const created = await createProject(name, desc);
        projectId = created.id;
        await populateProjectSelect(name);
      } else if (!sel || sel.trim() === '') {
        const found = projects.find(p => p.name === toolbarSelected) || projects[0];
        projectId = found ? found.id : projects[0].id;
      } else {
        const idx = parseInt(sel, 10) - 1;
        if (Number.isNaN(idx) || idx < 0 || idx >= projects.length) return;
        projectId = projects[idx].id;
      }

      // 2) Choose or create Dataset in selected Project
      const datasets = await listDatasets(projectId);
      const dsChoices = datasets.map((d, i) => {
        const tname = d.target_type_name || `Type ${d.target_type_id}`;
        return `${i + 1}) ${d.name} [${tname}]`;
      }).join('\n');
      sel = prompt(`Select a dataset by number or type 'new' to create:\n${dsChoices || '(none yet)'}`);
      let datasetId;
      if (sel && sel.toLowerCase() === 'new') {
        const modalRes = await showDatasetModal({ title: 'Create Dataset', name: '', description: '', target_type_id: 1 });
        if (!modalRes) return;
        const createdDs = await createDataset(projectId, modalRes.name, modalRes.description, modalRes.target_type_id);
        datasetId = createdDs.id;
        // Persist selection
        state.dataset_id = datasetId;
        state.dataset_name = createdDs.name;
        try { localStorage.setItem(`dataset:${sessionId}`, JSON.stringify({ id: datasetId, name: createdDs.name, target_type_id: createdDs.target_type_id })); } catch {}
        // update toolbar
        await populateDatasetSelect(String(datasetId));
      } else {
        const idx = parseInt(sel, 10) - 1;
        if (Number.isNaN(idx) || idx < 0 || idx >= datasets.length) return;
        const picked = datasets[idx];
        datasetId = picked.id;
        state.dataset_id = datasetId;
        state.dataset_name = picked.name;
        try { localStorage.setItem(`dataset:${sessionId}`, JSON.stringify({ id: datasetId, name: picked.name, target_type_id: picked.target_type_id })); } catch {}
        await populateDatasetSelect(String(datasetId));
      }

      // 4) Enroll (no baseline settings sent)
      await enrollSession(datasetId, sessionId);
      toast('Session enrolled');
    } catch (e) {
      console.error(e);
      toast('Enrollment failed');
    }
  }

  async function refreshProgress() {
    try {
      if (!state.dataset_id) return;
      const p = await datasetProgress(state.dataset_id);
      const el = els.statsGrid();
      if (!el) return;
      el.innerHTML = '';
      const mk = (label, value) => {
        const d = document.createElement('div');
        d.className = 'stat';
        d.innerHTML = `<div class="stat__label">${label}</div><div class="stat__value">${value}</div>`;
        return d;
      };
      const total = Number.isFinite(p.total) ? Number(p.total) : null;
      const labeled = Number.isFinite(p.labeled) ? Number(p.labeled) : null;
      const unlabeled = Number.isFinite(p.unlabeled) ? Number(p.unlabeled) : (total != null && labeled != null ? Math.max(0, total - labeled) : null);
      const pct = total && total > 0 && labeled != null ? ((labeled / total) * 100) : null;

      el.appendChild(mk('Total', total ?? '-'));
      el.appendChild(mk('Labeled', labeled ?? '-'));
      el.appendChild(mk('Unlabeled', unlabeled ?? '-'));
      el.appendChild(mk('Percent', pct != null ? `${pct.toFixed(1)}%` : '-'));
      if (state.dataset_name) el.appendChild(mk('Dataset', state.dataset_name));
      if (state.session_id) el.appendChild(mk('Session', state.session_id));
    } catch (e) {
      console.warn('Failed to load progress');
    }
  }

  function renderCategories() {
    const wrap = els.categoryList();
    wrap.innerHTML = '';
    state.categories.forEach((cat) => {
      const row = document.createElement('div');
      row.className = 'category';
      row.innerHTML = `
        <label style="display:flex;align-items:center;gap:8px;cursor:pointer">
          <input type="radio" name="category" value="${cat}">
          <span>${cat}</span>
        </label>
        <div class="category__controls">
          <input type="text" class="input input--small" value="${state.hotkeys[cat] || ''}" placeholder="Key" maxlength="1" data-category="${cat}">
          <button class="btn btn--small" data-remove="${cat}">Remove</button>
        </div>
      `;
      wrap.appendChild(row);

      const input = row.querySelector('input[type="text"]');
      input.addEventListener('input', () => {
        const v = (input.value || '').trim().toLowerCase();
        state.hotkeys[cat] = v;
        saveCategoriesToStorage();
        // Persist to backend session settings
        if (state.dataset_id && state.session_id) {
          saveDatasetSessionSettings(state.dataset_id, state.session_id, { categories: state.categories, hotkeys: state.hotkeys }).catch(() => {});
        }
      });

      const radio = row.querySelector('input[type="radio"]');
      radio.addEventListener('change', () => { state.frameSaved = false; highlightCategoryStates(); });

      const removeBtn = row.querySelector('button[data-remove]');
      removeBtn.addEventListener('click', () => {
        const name = removeBtn.getAttribute('data-remove');
        state.categories = state.categories.filter((c) => c !== name);
        delete state.hotkeys[name];
        saveCategoriesToStorage();
        renderCategories();
        renderDynamicShortcuts();
        // Persist to backend session settings
        if (state.dataset_id && state.session_id) {
          saveDatasetSessionSettings(state.dataset_id, state.session_id, { categories: state.categories, hotkeys: state.hotkeys }).catch(() => {});
        }
      });
    });
    highlightCategoryStates();
  }

  function renderDynamicShortcuts() {
    const el = els.dynamicShortcuts();
    el.innerHTML = '';
    Object.entries(state.hotkeys).forEach(([cat, key]) => {
      const row = document.createElement('div');
      row.className = 'shortcut';
      row.innerHTML = `<span>${cat}</span><span><span class="key">${key.toUpperCase()}</span></span>`;
      el.appendChild(row);
    });
  }

  function highlightCategoryStates() {
    const options = document.querySelectorAll('.category');
    options.forEach((opt) => {
      opt.classList.remove('category--selected');
      const radio = opt.querySelector('input[type="radio"]');
      if (radio && radio.checked) opt.classList.add('category--selected');
    });
  }

  function updateProgress(annotated, total) {
    const percent = total > 0 ? (annotated / total) * 100 : 0;
    els.progressFill().style.width = percent + '%';
    els.progressText().textContent = `${percent.toFixed(1)}% complete (${annotated}/${total})`;
  }

  function setSelectedCategory(category) {
    if (!category) return;
    const radio = document.querySelector(`input[name="category"][value="${category}"]`);
    if (radio) radio.checked = true;
    highlightCategoryStates();
  }

  function getSelectedCategory() {
    const selected = document.querySelector('input[name="category"]:checked');
    return selected ? selected.value : null;
  }

  // Core flows
  async function loadSessions() {
    try {
      const sessions = await apiGet('sessions');
      let enrolledSet = null;
      if (state.dataset_id) {
        const enrolledIds = await listDatasetEnrollments(state.dataset_id);
        if (Array.isArray(enrolledIds)) enrolledSet = new Set(enrolledIds);
      }
      renderSessionList(sessions, enrolledSet);
    } catch (e) {
      console.error(e);
      toast('Failed to load sessions');
    }
  }

  async function selectSession(session_id, project_name = 'default', opts = { pushHistory: true }) {
    state.session_id = session_id;
    state.project_name = project_name;
    state.currentIdx = 0;
    // Initialize totalFrames from preload if available
    if (opts && opts.preload && typeof opts.preload.frames_count === 'number') {
      state.totalFrames = opts.preload.frames_count;
      try { localStorage.setItem(lsKey('totalFrames'), String(state.totalFrames)); } catch {}
    }

    // Ensure dataset is selected/enrolled and fetch settings
    await ensureDatasetSelected(session_id);
    // Load dataset details and classes for mapping, update UI mode
    if (state.dataset_id) {
      const details = await fetchDatasetDetails(state.dataset_id);
      state.target_type_name = details ? details.target_type_name : null;
      await loadDatasetClasses(state.dataset_id);
      applyModeVisibility();
      refreshProgress();
    }
    // Load session settings from backend and apply
    try {
      const settings = await fetchDatasetSessionSettings(state.dataset_id, session_id);
      if (settings && (Array.isArray(settings.categories) || settings.hotkeys)) {
        if (Array.isArray(settings.categories)) state.categories = settings.categories;
        if (settings.hotkeys && typeof settings.hotkeys === 'object') state.hotkeys = settings.hotkeys;
        state.settingsLoaded = true;
        saveCategoriesToStorage();
      } else {
        // Fallback to local defaults if backend has nothing yet
        loadCategoriesFromStorage();
      }
    } catch {
      loadCategoriesFromStorage();
    }
    renderCategories();
    renderDynamicShortcuts();

    // Show annotation interface
    els.sessionSelector().classList.add('hidden');
    els.annotationInterface().classList.remove('hidden');
    els.sessionInfo().classList.remove('hidden');
    const dsText = state.dataset_name ? ` · Dataset: ${state.dataset_name} (#${state.dataset_id})` : (state.dataset_id ? ` · Dataset: #${state.dataset_id}` : '');
    els.sessionName().textContent = `Session: ${session_id} · Project: ${project_name}${dsText}`;

    // Update header back button to go back to sessions
    const backBtn = document.getElementById('back-btn');
    if (backBtn) {
      backBtn.textContent = 'Back to Sessions';
      backBtn.onclick = () => showSessionSelector();
    }

    saveSessionSelection();
    if (opts && opts.pushHistory) {
      history.pushState({ state: 'annotation', session: session_id, project: project_name }, 'Annotation', '#annotation');
    }
    // Update dataset progress panel
    applyModeVisibility();
    refreshProgress();
    await loadFrame(0);
  }

  async function loadFrame(idx) {
    if (idx < 0) idx = 0;
    try {
      const data = await apiGet('frame', {
        session_id: state.session_id,
        project_name: state.project_name,
        idx: idx,
      });

      state.currentIdx = idx;
      const frame = data.frame || {};
      const ann = data.annotation || {};

      // Compute total frames only once from first frame request (we don't have a stats endpoint)
      // We can infer totalFrames from last session discovery payload later; for now keep it if already set
      if (!state.totalFrames || state.totalFrames < 1) {
        // Try to read value from localStorage hint if we saved earlier; else leave unknown
        const tf = parseInt(localStorage.getItem(lsKey('totalFrames') || '0'), 10);
        if (!Number.isNaN(tf) && tf > 0) state.totalFrames = tf;
      }

      // Update image and frame info
      els.img().src = withBase(`/api/image?${new URLSearchParams({ session_id: state.session_id, idx: String(idx) }).toString()}`);
      els.frameId().textContent = frame.frame_id ?? '-';
      els.frameFilename().textContent = frame.filename ?? '-';
      const ts = frame.timestamp;
      els.frameTimestamp().textContent = typeof ts === 'number' ? `${ts.toFixed(3)}s` : '-';

      // Annotation fields
      const savedCat = ann?.category || ann?.annotations?.category || null;
      setSelectedCategory(savedCat);
      state.savedCategoryForFrame = savedCat;
      state.frameSaved = !!savedCat;
      const notesVal = ann?.notes || ann?.annotations?.notes || '';
      els.notes().value = notesVal;

      // Navigation UI text and buttons
      els.frameInput().value = state.totalFrames > 0 ? `${idx + 1}/${state.totalFrames}` : `${idx + 1}`;
      els.prevBtn().disabled = idx <= 0;
      els.firstBtn().disabled = idx <= 0;
      els.nextBtn().disabled = state.totalFrames ? idx >= state.totalFrames - 1 : false;
      els.lastBtn().disabled = state.totalFrames ? idx >= state.totalFrames - 1 : false;

      // We cannot compute progress without totals; leave as-is
    } catch (e) {
      console.error(e);
      toast('Failed to load frame');
    }
  }

  async function saveAnnotation() {
    const t = state.target_type_name;
    const isSingle = (t === 'SingleLabelClassification') || (!t);
    if (isSingle) {
      // Save single-label annotation to DB-backed endpoint
      if (!state.dataset_id) { toast('Select a dataset first (use Enroll)'); return false; }
      const category = getSelectedCategory();
      if (!category) { toast('Pick a category'); return false; }
      const classId = state.classIdByName ? state.classIdByName[category] : undefined;
      try {
        const payload = {
          session_id: state.session_id,
          dataset_id: state.dataset_id,
          frame_idx: state.currentIdx,
          ...(Number.isInteger(classId) || (typeof classId === 'number' && !Number.isNaN(classId))
            ? { class_id: classId }
            : { category_name: category }),
        };
        const res = await apiPost('annotations/single_label', payload);
        if (res && (res.ok || res.saved || res.status === 'ok')) {
          state.savedCategoryForFrame = category;
          state.frameSaved = true;
          toast('Saved');
          if (!(Number.isInteger(classId) || (typeof classId === 'number' && !Number.isNaN(classId)))) {
            await loadDatasetClasses(state.dataset_id);
          }
          refreshProgress();
          return true;
        }
        toast('Save failed');
        return false;
      } catch (e) {
        console.error(e);
        toast('Error while saving');
        return false;
      }
    } else if (t === 'MultiLabelClassification') {
      return await saveMultilabel();
    } else if (t === 'Regression') {
      return await saveRegression();
    }
    return false;
  }

  async function saveAndNext() {
    const ok = await saveAnnotation();
    if (ok) goToFrame(state.currentIdx + 1);
  }

  function goToFrame(idx) {
    if (idx < 0) return;
    if (state.totalFrames && idx >= state.totalFrames) return;
    loadFrame(idx);
  }

  function showSessionSelector(opts = { pushHistory: true }) {
    els.sessionSelector().classList.remove('hidden');
    els.annotationInterface().classList.add('hidden');
    els.sessionInfo().classList.add('hidden');
    try {
      localStorage.setItem('appState', 'sessions');
      localStorage.removeItem('currentSession');
      localStorage.removeItem('currentProject');
      localStorage.removeItem('currentProjectId');
    } catch {}
    // Update header back button to go back to dashboard
    const backBtn = document.getElementById('back-btn');
    if (backBtn) {
      backBtn.textContent = 'Back to Dashboard';
      backBtn.onclick = () => { window.location.href = '/'; };
    }
    if (opts && opts.pushHistory) {
      history.pushState({ state: 'sessions' }, 'Select Session', '#sessions');
    }
    loadSessions();
  }

  function toast(text) {
    const el = document.createElement('div');
    el.className = 'toast';
    el.textContent = text;
    document.body.appendChild(el);
    setTimeout(() => el.remove(), 1200);
  }

  // Event wiring
  function bindEvents() {
    // Project selector
    const ps = els.projectSelect();
    if (ps) {
      ps.addEventListener('change', async () => {
        const hasOptions = ps.options && ps.options.length > 0;
        const opt = ps.selectedOptions && ps.selectedOptions[0];
        if (!hasOptions || !opt) {
          // No projects yet: prompt creation and do not override state
          try { await renderProjectManager(); } catch {}
          openModal('project-modal');
          return;
        }
        state.project_id = parseInt(opt.dataset.projectId || ps.value, 10);
        state.project_name = (opt.dataset.projectName || opt.textContent || '').trim() || null;
        try {
          localStorage.setItem('currentProjectId', state.project_id || '');
          localStorage.setItem('currentProject', state.project_name || '');
          setCookie('currentProjectId', String(state.project_id || ''));
          if (state.project_name) setCookie('currentProject', state.project_name);
        } catch {}
        await populateDatasetSelect();
        // Refresh sessions to reflect enrollments for datasets under new project
        await loadSessions();
      });
    }
    // Project Manage button
    const manageProjectsBtn = document.getElementById('manage-projects');
    if (manageProjectsBtn) {
      manageProjectsBtn.addEventListener('click', async () => {
        try { await renderProjectManager(); } catch {}
        openModal('project-modal');
      });
    }
    // Dataset Manage button
    const manageDatasetsBtn = document.getElementById('manage-datasets');
    if (manageDatasetsBtn) {
      manageDatasetsBtn.addEventListener('click', async () => {
        try { await renderDatasetManager(); } catch {}
        openModal('dataset-modal');
        await loadSessions();
      });
    }
    // Dataset selector
    const ds = els.datasetSelect();
    if (ds) {
      ds.addEventListener('change', async () => {
        const sel = ds.selectedOptions[0];
        state.dataset_id = Number(ds.value);
        state.dataset_name = sel ? (sel.dataset.datasetName || null) : null;
        if (state.session_id && state.dataset_id) {
          try { localStorage.setItem(`dataset:${state.session_id}`, JSON.stringify({ id: state.dataset_id, name: state.dataset_name || '' })); } catch {}
        }
        try { setCookie('currentDatasetId', String(state.dataset_id)); } catch {}
        // Load dataset details and classes and update UI mode
        const details = await fetchDatasetDetails(state.dataset_id);
        state.target_type_name = details ? details.target_type_name : null;
        await loadDatasetClasses(state.dataset_id);
        applyModeVisibility();
        refreshProgress();
        // Refresh sessions to highlight enrollments for this dataset
        await loadSessions();
      });
    }
    const dsManageBtn = document.getElementById('manage-datasets');
    if (dsManageBtn) {
      dsManageBtn.addEventListener('click', async () => {
        await renderDatasetManager();
        openModal('dataset-modal');
      });
    }
    els.addCategoryBtn().addEventListener('click', () => {
      const input = els.newCategoryInput();
      const name = (input.value || '').trim();
      if (!name) return;
      if (!state.categories.includes(name)) {
        state.categories.push(name);
        saveCategoriesToStorage();
        renderCategories();
        renderDynamicShortcuts();
        // Persist to backend session settings
        if (state.dataset_id && state.session_id) {
          saveDatasetSessionSettings(state.dataset_id, state.session_id, { categories: state.categories, hotkeys: state.hotkeys }).catch(() => {});
        }
      }
      input.value = '';
    });
    els.newCategoryInput().addEventListener('keypress', (e) => { if (e.key === 'Enter') els.addCategoryBtn().click(); });

    els.firstBtn().addEventListener('click', () => goToFrame(0));
    els.prevBtn().addEventListener('click', () => goToFrame(state.currentIdx - 1));
    els.nextBtn().addEventListener('click', () => goToFrame(state.currentIdx + 1));
    els.lastBtn().addEventListener('click', () => { if (state.totalFrames) goToFrame(state.totalFrames - 1); });
    els.frameInput().addEventListener('keypress', (e) => {
      if (e.key === 'Enter') {
        const val = parseInt(e.target.value, 10) - 1;
        if (!Number.isNaN(val)) goToFrame(val);
      }
    });

    els.saveBtn().addEventListener('click', saveAnnotation);
    els.saveNextBtn().addEventListener('click', saveAndNext);
    els.skipBtn().addEventListener('click', () => goToFrame(state.currentIdx + 1));

    // Header back button (dynamic)
    const header = document.querySelector('.header');
    let backBtn = document.getElementById('back-btn');
    if (!backBtn) {
      backBtn = document.createElement('button');
      backBtn.id = 'back-btn';
      backBtn.className = 'btn';
      backBtn.style.marginTop = '8px';
      header.appendChild(backBtn);
    }
    // Initialize default state based on current view (on first bind we're in sessions)
    backBtn.textContent = 'Back to Dashboard';
    backBtn.onclick = () => { window.location.href = '/'; };

    // Keyboard shortcuts
    document.addEventListener('keydown', (e) => {
      if (e.target && (e.target.tagName === 'TEXTAREA' || e.target.tagName === 'INPUT')) return;
      switch (e.key) {
        case 'ArrowLeft': e.preventDefault(); goToFrame(state.currentIdx - 1); break;
        case 'ArrowRight': e.preventDefault(); goToFrame(state.currentIdx + 1); break;
        case ' ': e.preventDefault(); saveAndNext(); break;
        default: {
          const key = e.key.toLowerCase();
          for (const [cat, hk] of Object.entries(state.hotkeys)) {
            if (key === hk) {
              const radio = document.querySelector(`input[name="category"][value="${cat}"]`);
              if (radio) { radio.checked = true; state.frameSaved = false; highlightCategoryStates(); }
            }
          }
        }
      }
    });

    // Browser navigation basics
    window.addEventListener('popstate', (event) => {
      const st = event.state;
      if (st && st.state === 'annotation' && st.session && st.project) {
        // navigate without pushing a new history entry
        selectSession(st.session, st.project, { pushHistory: false });
      } else {
        showSessionSelector({ pushHistory: false });
      }
    });

    // Session selector controls
    const btnRefresh = document.getElementById('refresh-sessions');
    const btnReindex = document.getElementById('reindex-sessions');
    if (btnRefresh) btnRefresh.addEventListener('click', loadSessions);
    if (btnReindex) btnReindex.addEventListener('click', async () => {
      const prev = btnReindex.textContent;
      btnReindex.disabled = true;
      btnReindex.textContent = 'Reindexing...';
      try {
        await apiPostNoBody('reindex');
      } catch (e) {
        toast('Reindex failed');
      } finally {
        btnReindex.disabled = false;
        btnReindex.textContent = prev;
        loadSessions();
      }
    });
  }

  async function init() {
    // Populate projects first so toolbar has a value
    await populateProjectSelect();
    // Secondary safeguard: if selector still empty, prompt creation
    const ps = els.projectSelect();
    if (ps && (!ps.options || ps.options.length === 0)) {
      try { await renderProjectManager(); } catch {}
      openModal('project-modal');
    }
    await populateDatasetSelect();
    bindEvents();

    // Always load sessions first so we can validate any saved selection
    await loadSessions();

    // Initial state from URL or localStorage
    const hash = (location.hash || '').toLowerCase();
    const savedSession = localStorage.getItem('currentSession');
    const savedProject = localStorage.getItem('currentProject');

    // Validate saved session exists in discovered list
    let canAutoSelect = false;
    if (savedSession && savedProject && hash === '#annotation') {
      try {
        const sessions = await apiGet('sessions');
        canAutoSelect = Array.isArray(sessions) && sessions.some(s => s.session_id === savedSession);
      } catch { canAutoSelect = false; }
    }

    if (canAutoSelect) {
      await selectSession(savedSession, savedProject, { pushHistory: false });
    } else {
      // Clear stale saved values
      try { localStorage.removeItem('currentSession'); localStorage.removeItem('currentProject'); } catch {}
      history.replaceState({ state: 'sessions' }, 'Select Session', '#sessions');
    }
  }

  document.addEventListener('DOMContentLoaded', init);
})();
