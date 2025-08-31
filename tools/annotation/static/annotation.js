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

  // ---------- Core helpers (API, selectors, state) ----------
  function withBase(path) {
    try {
      if (!path) return APP_BASE;
      if (path.startsWith('http://') || path.startsWith('https://')) return path;
      if (path.startsWith('/')) return APP_BASE + path;
      return APP_BASE + '/' + path;
    } catch { return path; }
  }

  async function apiRequest(method, path, body) {
    const url = path.startsWith('/api') ? withBase(path) : withBase('/api/' + path.replace(/^\/?/, ''));
    const init = { method, headers: { 'Accept': 'application/json' } };
    if (body !== undefined) {
      init.headers['Content-Type'] = 'application/json';
      init.body = JSON.stringify(body);
    }
    const res = await fetch(url, init);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const ct = res.headers.get('content-type') || '';
    if (ct.includes('application/json')) return await res.json();
    return { ok: true };
  }
  const apiGet = (path) => apiRequest('GET', path);
  const apiPost = (path, body) => apiRequest('POST', path, body);
  const apiPut = (path, body) => apiRequest('PUT', path, body);
  const apiDelete = (path) => apiRequest('DELETE', path);
  const apiPostNoBody = (path) => apiRequest('POST', path);

  // GET with AbortSignal and query params support (used by frame loader)
  async function apiGetWithSignal(path, queryParams = null, signal = undefined) {
    const base = path.startsWith('/api') ? withBase(path) : withBase('/api/' + path.replace(/^\/?/, ''));
    const url = new URL(base, window.location.origin);
    if (queryParams && typeof queryParams === 'object') {
      const usp = new URLSearchParams();
      Object.entries(queryParams).forEach(([k, v]) => {
        if (v === undefined || v === null) return;
        usp.append(k, String(v));
      });
      // If base already had search params, preserve them
      const existing = url.search;
      url.search = existing ? (existing + '&' + usp.toString()) : ('?' + usp.toString());
    }
    const res = await fetch(url.toString(), { method: 'GET', headers: { 'Accept': 'application/json' }, signal });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const ct = res.headers.get('content-type') || '';
    if (ct.includes('application/json')) return await res.json();
    return { ok: true };
  }

  // Element selectors used throughout UI
  const els = {
    projectSelect: () => document.getElementById('project-select'),
    datasetSelect: () => document.getElementById('dataset-select'),
    sessionSelector: () => document.getElementById('session-selector'),
    annotationInterface: () => document.getElementById('annotation-interface'),
    sessionInfo: () => document.getElementById('session-info'),
    img: () => document.getElementById('frame-image'),
    frameId: () => document.getElementById('frame-id'),
    frameFilename: () => document.getElementById('frame-filename'),
    frameTimestamp: () => document.getElementById('frame-timestamp'),
    notes: () => document.getElementById('notes-input'),
    categoryList: () => document.getElementById('category-list'),
    dynamicShortcuts: () => document.getElementById('dynamic-shortcuts'),
    progressFill: () => document.getElementById('progress-fill'),
    progressText: () => document.getElementById('progress-text'),
    sessionName: () => document.getElementById('session-name'),
    // Category add controls
    addCategoryBtn: () => document.getElementById('add-category-btn'),
    newCategoryInput: () => document.getElementById('new-category-input'),
    // Navigation & actions
    firstBtn: () => document.getElementById('first-btn'),
    prevBtn: () => document.getElementById('prev-btn'),
    nextBtn: () => document.getElementById('next-btn'),
    lastBtn: () => document.getElementById('last-btn'),
    frameInput: () => document.getElementById('frame-input'),
    saveNextBtn: () => document.getElementById('save-next-btn'),
    saveBtn: () => document.getElementById('save-btn'),
    skipBtn: () => document.getElementById('skip-btn'),
    sessionList: () => document.getElementById('session-list')
  };

  // App state
  const state = {
    project_id: null,
    project_name: 'default',
    dataset_id: null,
    dataset_name: null,
    target_type_id: null,
    target_type_name: null,
    datasetClasses: [],
    session_id: null,
    currentIdx: 0,
    totalFrames: null,
    categories: [],
    hotkeys: {},
    regressionMin: null,
    regressionMax: null,
    frameSaved: true,
  };
  // Controller to cancel in-flight frame fetches
  let frameRequestController = null;
  function lsKey(k) {
    const sid = state.session_id ? String(state.session_id) : 'global';
    return `annot:${sid}:${k}`;
  }

  function toast(msg) {
    try { console.log('[toast]', msg); } catch {}
  }

  function saveCategoriesToStorage() {
    try {
      localStorage.setItem(lsKey('categories'), JSON.stringify(state.categories || []));
      localStorage.setItem(lsKey('hotkeys'), JSON.stringify(state.hotkeys || {}));
    } catch {}
  }

  // Load categories and hotkeys from localStorage (scoped by current session via lsKey)
  function loadCategoriesFromStorage() {
    try {
      const catsRaw = localStorage.getItem(lsKey('categories'));
      const hksRaw = localStorage.getItem(lsKey('hotkeys'));
      const cats = catsRaw ? JSON.parse(catsRaw) : [];
      const hotkeys = hksRaw ? JSON.parse(hksRaw) : {};
      state.categories = Array.isArray(cats) ? cats : [];
      state.hotkeys = (hotkeys && typeof hotkeys === 'object') ? hotkeys : {};
    } catch {
      // On any parse or access error, fall back to safe defaults
      state.categories = Array.isArray(state.categories) ? state.categories : [];
      state.hotkeys = (state.hotkeys && typeof state.hotkeys === 'object') ? state.hotkeys : {};
    }
  }

  async function saveSettings() {
    if (!state.dataset_id || !state.session_id) return;
    const settings = {
      categories: state.categories || [],
      hotkeys: state.hotkeys || {},
      regressionMin: state.regressionMin,
      regressionMax: state.regressionMax,
    };
    try {
      await apiPut(`datasets/${encodeURIComponent(state.dataset_id)}/sessions/${encodeURIComponent(state.session_id)}/settings`, { settings });
    } catch {}
  }
  let saveSettingsTimer = null;
  function debouncedSaveSettings() {
    if (saveSettingsTimer) clearTimeout(saveSettingsTimer);
    saveSettingsTimer = setTimeout(saveSettings, 400);
  }

  function getSelectedProjectId() {
    const sel = els.projectSelect();
    if (!sel) return null;
    const opt = sel.selectedOptions && sel.selectedOptions[0];
    if (opt && opt.dataset && opt.dataset.projectId) return parseInt(opt.dataset.projectId, 10);
    const val = sel.value != null ? parseInt(sel.value, 10) : NaN;
    return Number.isFinite(val) ? val : null;
  }

  async function populateProjectSelect(selectName = null) {
    const sel = els.projectSelect();
    if (!sel) return;
    sel.innerHTML = '';
    const projects = await apiGet('projects').catch(() => []);
    projects.forEach(p => {
      const o = document.createElement('option');
      o.value = String(p.id);
      o.textContent = String(p.name || '');
      o.dataset.projectId = String(p.id);
      o.dataset.projectName = String(p.name || '');
      sel.appendChild(o);
    });
    if (selectName) {
      const opt = Array.from(sel.options).find(o => (o.textContent || '') === String(selectName));
      if (opt) sel.value = opt.value;
    }
    const opt = sel.selectedOptions && sel.selectedOptions[0];
    if (opt) { state.project_id = parseInt(opt.dataset.projectId, 10); state.project_name = opt.dataset.projectName; }
  }

  async function populateDatasetSelect(selectId = null) {
    const sel = els.datasetSelect();
    if (!sel) return;
    sel.innerHTML = '';
    const pid = getSelectedProjectId();
    if (!pid) return;
    const datasets = await listDatasets(pid).catch(() => []);
    datasets.forEach(d => {
      const o = document.createElement('option');
      o.value = String(d.id);
      o.textContent = String(d.name || '');
      o.dataset.datasetId = String(d.id);
      o.dataset.datasetName = String(d.name || '');
      sel.appendChild(o);
    });
    if (selectId) sel.value = String(selectId);
    const opt = sel.selectedOptions && sel.selectedOptions[0];
    if (opt) { state.dataset_id = parseInt(opt.datasetId || sel.value, 10); state.dataset_name = opt.datasetName; }
  }

  async function fetchDatasetDetails(datasetId) {
    return await apiGet(`datasets/${datasetId}`).catch(() => null);
  }
  async function loadDatasetClasses(datasetId) {
    const details = await fetchDatasetDetails(datasetId);
    const classes = (details && Array.isArray(details.classes)) ? details.classes : [];
    state.datasetClasses = classes;
    // Update single vs multi-label UI based on target type
    state.target_type_name = details ? details.target_type_name : state.target_type_name;
    state.target_type_id = (details && typeof details.target_type_id === 'number') ? details.target_type_id : state.target_type_id;
    return classes;
  }
  async function listTargetTypes() {
    return await apiGet('target_types').catch(() => []);
  }

  async function listDatasets(projectId) {
    if (!projectId) return [];
    return await apiGet(`projects/${encodeURIComponent(projectId)}/datasets`).catch(() => []);
  }

  async function createProject(name, description = '') {
    return await apiPost('projects', { name, description }).catch(() => null);
  }
  async function createDataset(project_id, name, description = '', target_type_id = 1) {
    if (!project_id) return null;
    return await apiPost(`projects/${encodeURIComponent(project_id)}/datasets`, { name, description, target_type_id }).catch(() => null);
  }

  // ---------- Dataset/session helpers ----------
  async function listDatasetEnrollments(datasetId) {
    try {
      const res = await apiGet(`datasets/${encodeURIComponent(datasetId)}/enrollments`);
      // Expect array of session_ids
      return Array.isArray(res) ? res : [];
    } catch { return []; }
  }

  async function enrollSession(datasetId, sessionId) {
    try {
      const payload = { session_id: sessionId };
      const res = await apiPost(`datasets/${encodeURIComponent(datasetId)}/enroll_session`, payload);
      return !!res;
    } catch { return false; }
  }

  async function ensureDatasetSelected(sessionId) {
    // If dataset already chosen, ensure it's enrolled; otherwise try to restore from localStorage
    if (!state.dataset_id) {
      try {
        const saved = localStorage.getItem(`dataset:${sessionId}`);
        if (saved) {
          const obj = JSON.parse(saved);
          if (obj && obj.id) {
            state.dataset_id = obj.id;
            state.dataset_name = obj.name || null;
          }
        }
      } catch {}
    }
    if (!state.dataset_id) return; // caller may prompt user
    try {
      const enrolled = await listDatasetEnrollments(state.dataset_id);
      const isEnrolled = Array.isArray(enrolled) && enrolled.includes(sessionId);
      if (!isEnrolled) {
        await enrollSession(state.dataset_id, sessionId);
      }
    } catch {}
  }

  async function fetchDatasetSessionSettings(datasetId, sessionId) {
    try {
      return await apiGet(`datasets/${encodeURIComponent(datasetId)}/sessions/${encodeURIComponent(sessionId)}/settings`);
    } catch { return null; }
  }

  function applyModeVisibility() {
    const t = state.target_type_name;
    // Panels
    const catList = els.categoryList && els.categoryList();
    const mlPanel = document.getElementById('multilabel-panel') || null; // optional
    const regPanel = document.getElementById('regression-panel') || null;
    // Determine mode
    const isSingle = (t === 'SingleLabelClassification') || (!t);
    const isMulti = (t === 'MultiLabelClassification');
    const isReg = (t === 'Regression');
    if (catList) catList.parentElement && (catList.parentElement.style.display = isSingle ? '' : (isMulti ? '' : ''));
    if (regPanel) regPanel.style.display = isReg ? '' : 'none';
    // Render appropriate content
    if (isMulti) {
      renderMultilabelPanel();
    } else if (isSingle) {
      renderCategories();
    }
  }

  async function refreshProgress() {
    try {
      if (!state.dataset_id || !state.session_id) return;
      const res = await apiGet(`datasets/${encodeURIComponent(state.dataset_id)}/progress`);
      if (res && typeof res.annotated === 'number' && typeof res.total === 'number') {
        updateProgress(res.annotated, res.total);
        state.totalFrames = res.total;
      }
    } catch {}
  }

  // ---------- Multi-label UI (reuse #category-list with checkboxes) ----------
  function renderMultilabelPanel() {
    const list = els.categoryList && els.categoryList();
    if (!list) return;
    list.innerHTML = '';
    list.classList.add('categories');
    const rows = state.datasetClasses || [];
    const grid = document.createElement('div');
    grid.style.display = 'grid';
    grid.style.gridTemplateColumns = 'repeat(auto-fill,minmax(160px,1fr))';
    grid.style.gap = '8px';
    rows.forEach(r => {
      const id = `ml-${r.id}`;
      const item = document.createElement('label');
      item.setAttribute('for', id);
      item.style.display = 'flex';
      item.style.alignItems = 'center';
      item.style.gap = '8px';
      const input = document.createElement('input');
      input.type = 'checkbox';
      input.id = id;
      input.setAttribute('data-class-id', String(r.id));
      const span = document.createElement('span');
      span.textContent = String(r.name || '');
      item.appendChild(input);
      item.appendChild(span);
      grid.appendChild(item);
    });
    list.appendChild(grid);
  }

  function getSelectedMultilabelClassIds() {
    const checks = document.querySelectorAll('#category-list input[type="checkbox"][data-class-id]');
    const ids = [];
    checks.forEach(ch => { if (ch.checked) ids.push(parseInt(ch.getAttribute('data-class-id'), 10)); });
    return ids;
  }

  async function saveMultilabel() {
    if (!state.dataset_id) { toast('Select a dataset first'); return false; }
    const classIds = getSelectedMultilabelClassIds();
    if (!classIds.length) { toast('Select at least one label'); return false; }
    try {
      // Map class_ids to names if we have them (backend accepts either)
      const idToName = new Map((state.datasetClasses || []).map(r => [r.id, r.name]));
      const categoryNames = classIds.map(id => idToName.get(id)).filter(n => typeof n === 'string' && n.length > 0);
      const payload = {
        session_id: state.session_id,
        dataset_id: state.dataset_id,
        frame_idx: state.currentIdx,
        class_ids: classIds,
        ...(categoryNames.length > 0 ? { category_names: categoryNames } : {}),
        override_settings: {
          notes: els.notes().value,
        },
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
    const header = '<h3 class="panel__title">Regression</h3>';
    const wrap = document.createElement('div');
    wrap.className = 'panel__body';
    wrap.style.display = 'grid';
    wrap.style.gridTemplateColumns = 'auto 1fr auto';
    wrap.style.gap = '12px';
    wrap.innerHTML = `
      <label style="align-self:center">Value</label>
      <input id="regression-input" type="range" min="0" max="100" step="1" value="0" style="width:100%"> 
      <input id="regression-number" class="input" type="number" min="0" max="100" step="1" value="0" style="width:96px">
      <div style="grid-column: 1 / -1; display:flex; gap:12px; align-items:center;">
        <label>Min</label>
        <input id="regression-min" class="input" type="number" value="0" style="width:100%">
        <label>Max</label>
        <input id="regression-max" class="input" type="number" value="100" style="width:100%">
      </div>
    `;
    rg.innerHTML = header;
    rg.appendChild(wrap);
    // Wire up interactions
    const range = rg.querySelector('#regression-input');
    const number = rg.querySelector('#regression-number');
    const minEl = rg.querySelector('#regression-min');
    const maxEl = rg.querySelector('#regression-max');
    if (range && number && minEl && maxEl) {
      const clamp = (v, lo, hi) => Math.min(Math.max(v, lo), hi);
      const clearShortcutSelection = () => {
        document.querySelectorAll('input[name="reg-shortcut"]').forEach(r => { r.checked = false; });
      };
      const setStepForBounds = (minV, maxV) => {
        const span = Math.abs(maxV - minV);
        let step = 1;
        if (span <= 1) step = 0.01;
        if (span <= 0.1) step = 0.001;
        range.step = String(step);
        number.step = String(step);
      };
      const syncNumber = () => { number.value = range.value; clearShortcutSelection(); };
      const syncRange = () => { range.value = number.value; clearShortcutSelection(); };
      const syncBounds = () => {
        let minV = parseFloat(minEl.value); if (!Number.isFinite(minV)) minV = 0;
        let maxV = parseFloat(maxEl.value); if (!Number.isFinite(maxV)) maxV = 100;
        if (maxV < minV) { const t = minV; minV = maxV; maxV = t; minEl.value = String(minV); maxEl.value = String(maxV); }
        setStepForBounds(minV, maxV);
        range.min = String(minV); range.max = String(maxV);
        number.min = String(minV); number.max = String(maxV);
        const cur = parseFloat(number.value);
        const clamped = clamp(Number.isFinite(cur) ? cur : minV, minV, maxV);
        number.value = String(clamped);
        range.value = String(clamped);
        // Persist to state + localStorage + backend
        state.regressionMin = minV; state.regressionMax = maxV;
        try { localStorage.setItem(lsKey('regression_min'), String(minV)); localStorage.setItem(lsKey('regression_max'), String(maxV)); } catch {}
        if (state.dataset_id && state.session_id) {
          debouncedSaveSettings();
        }
      };
      range.addEventListener('input', syncNumber);
      number.addEventListener('input', syncRange);
      minEl.addEventListener('input', syncBounds);
      maxEl.addEventListener('input', syncBounds);
      // Initialize bounds from state, then localStorage fallback
      let initMin = (state.regressionMin != null) ? state.regressionMin : null;
      let initMax = (state.regressionMax != null) ? state.regressionMax : null;
      if (initMin == null) { const s = localStorage.getItem(lsKey('regression_min')); if (s != null) initMin = parseFloat(s); }
      if (initMax == null) { const s = localStorage.getItem(lsKey('regression_max')); if (s != null) initMax = parseFloat(s); }
      if (!Number.isFinite(initMin)) initMin = 0;
      if (!Number.isFinite(initMax)) initMax = 100;
      minEl.value = String(initMin);
      maxEl.value = String(initMax);
      state.regressionMin = initMin; state.regressionMax = initMax;
      syncBounds();
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
        override_settings: {
          notes: els.notes().value,
        },
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

  function deleteCookie(name) {
    try {
      document.cookie = `${encodeURIComponent(name)}=;expires=Thu, 01 Jan 1970 00:00:00 GMT;path=/`;
    } catch {}
  }

  // Persist current selection to localStorage and cookies
  function saveSessionSelection() {
    try {
      if (state.session_id != null) localStorage.setItem('currentSession', String(state.session_id));
      if (state.project_name != null) localStorage.setItem('currentProject', String(state.project_name));
      if (state.project_id != null) localStorage.setItem('currentProjectId', String(state.project_id));
      if (state.dataset_id != null && state.session_id != null) {
        localStorage.setItem(`dataset:${state.session_id}`, JSON.stringify({ id: state.dataset_id, name: state.dataset_name || '' }));
        setCookie('currentDatasetId', String(state.dataset_id));
      }
      setCookie('currentProject', state.project_name || '');
      if (state.project_id != null) setCookie('currentProjectId', String(state.project_id));
    } catch {}
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
    // Close on Escape and prevent body scroll while modal is open
    const onKey = (ev) => { if (ev.key === 'Escape') closeModal(id); };
    try { m._onKey = onKey; } catch {}
    document.addEventListener('keydown', onKey);
    try { document.body.classList.add('no-scroll'); } catch {}
    // Simple debug marker
    try { m.setAttribute('data-opened', String(Date.now())); } catch {}
  }
  function closeModal(id) {
    const m = getModal(id);
    if (m) {
      m.classList.add('hidden');
      // Remove key listener if we added one
      const onKey = m._onKey;
      if (onKey) {
        document.removeEventListener('keydown', onKey);
        try { delete m._onKey; } catch {}
      }
    }
    // If no other visible modals, restore body scroll
    try {
      const anyOpen = Array.from(document.querySelectorAll('.modal')).some(el => !el.classList.contains('hidden'));
      if (!anyOpen) document.body.classList.remove('no-scroll');
    } catch {}
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
    {
      const name = document.createElement('input');
      name.className = 'input m-input';
      name.id = 'pm-name';
      name.placeholder = 'Project name';
      const desc = document.createElement('input');
      desc.className = 'input m-input';
      desc.id = 'pm-desc';
      desc.placeholder = 'Description (optional)';
      const btn = document.createElement('button');
      btn.className = 'btn';
      btn.id = 'pm-create';
      btn.textContent = 'Create';
      form.appendChild(name);
      form.appendChild(desc);
      form.appendChild(btn);
    }
    body.appendChild(form);
    const listWrap = document.createElement('div');
    listWrap.className = 'm-list';
    body.appendChild(listWrap);
    const projects = await listProjects().catch(() => []);
    projects.forEach(p => {
      const row = document.createElement('div');
      row.className = 'm-item';
      const left = document.createElement('div');
      const title = document.createElement('div');
      title.className = 'm-item__title';
      title.textContent = String(p.name || '');
      const meta = document.createElement('div');
      meta.className = 'm-item__meta';
      meta.textContent = String(p.description || '');
      left.appendChild(title);
      left.appendChild(meta);
      const right = document.createElement('div');
      const btn = document.createElement('button');
      btn.className = 'btn';
      btn.setAttribute('data-select', '');
      btn.textContent = 'Use';
      right.appendChild(btn);
      row.appendChild(left);
      row.appendChild(right);
      row.querySelector('[data-select]').onclick = async () => {
        try {
          const sel = els.projectSelect();
          if (sel) {
            sel.value = String(p.id);
            const opt = Array.from(sel.options).find(o => o.value === String(p.id));
            state.project_id = p.id;
            state.project_name = p.name;
            if (opt) {
              state.project_name = opt.dataset.projectName || p.name;
            }
            if (state.session_id && state.project_id) {
              try { localStorage.setItem(`dataset:${state.session_id}`, JSON.stringify({ id: state.dataset_id, name: state.dataset_name || '' })); } catch {}
            }
            // Update project-driven UI and sessions
            await populateDatasetSelect();
            await loadSessions();
            closeModal('project-modal');
            toast('Project selected');
          }
        } catch {}
      };
      row.querySelector('[data-edit]').onclick = async () => {
        const newName = prompt('New project name', p.name) || p.name;
        const newDesc = prompt('Description (optional)', p.description || '') || '';
        try {
          await apiPut(`projects/${p.id}`, { name: newName, description: newDesc });
          await populateProjectSelect(newName);
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
        const created = await createProject(name, desc);
        await populateProjectSelect(name);
        await populateDatasetSelect();
        await renderProjectManager();
        await loadSessions();
        toast('Project created');
      } catch { toast('Create failed'); }
    };
  }

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
    {
      const name = document.createElement('input');
      name.className = 'input m-input';
      name.id = 'dm-name';
      name.placeholder = 'Dataset name';
      const desc = document.createElement('input');
      desc.className = 'input m-input';
      desc.id = 'dm-desc';
      desc.placeholder = 'Description (optional)';
      const btn = document.createElement('button');
      btn.className = 'btn';
      btn.id = 'dm-create';
      btn.textContent = 'Create';
      form.appendChild(name);
      form.appendChild(desc);
      form.appendChild(btn);
    }
    body.appendChild(form);
    const listWrap = document.createElement('div');
    listWrap.className = 'm-list';
    body.appendChild(listWrap);
    const datasets = await listDatasets(projectId).catch(() => []);
    datasets.forEach(d => {
      const row = document.createElement('div');
      row.className = 'm-item';
      const typeLabel = d.target_type_name || `Type ${d.target_type_id}`;
      const left = document.createElement('div');
      const title = document.createElement('div');
      title.className = 'm-item__title';
      title.textContent = String(d.name || '');
      const meta = document.createElement('div');
      meta.className = 'm-item__meta';
      meta.textContent = `${String(d.description || '')} · ${String(typeLabel)}`;
      left.appendChild(title);
      left.appendChild(meta);
      const right = document.createElement('div');
      const btn = document.createElement('button');
      btn.className = 'btn';
      btn.setAttribute('data-select', '');
      btn.textContent = 'Select';
      right.appendChild(btn);
      row.appendChild(left);
      row.appendChild(right);
      row.querySelector('[data-select]').onclick = async () => {
        const dsId = d.id;
        const dsName = d.name;
        const sel = els.datasetSelect();
        if (sel) {
          sel.value = String(dsId);
          const opt = Array.from(sel.options).find(o => o.value === String(dsId));
          state.dataset_id = dsId;
          state.dataset_name = dsName;
          if (opt) {
            state.dataset_name = opt.dataset.datasetName || dsName;
          }
          if (state.session_id && state.dataset_id) {
            try { localStorage.setItem(`dataset:${state.session_id}`, JSON.stringify({ id: state.dataset_id, name: state.dataset_name || '' })); } catch {}
          }
          // Update dataset-driven UI and sessions
          const details = await fetchDatasetDetails(state.dataset_id);
          state.target_type_name = details ? details.target_type_name : null;
          state.target_type_id = (details && typeof details.target_type_id === 'number') ? details.target_type_id : state.target_type_id;
          await loadDatasetClasses(state.dataset_id);
          applyModeVisibility();
          refreshProgress();
          await loadSessions();
          closeModal('dataset-modal');
          toast('Dataset selected');
        }
      };
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
            state.target_type_id = (details && typeof details.target_type_id === 'number') ? details.target_type_id : state.target_type_id;
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
        state.target_type_id = (details && typeof details.target_type_id === 'number') ? details.target_type_id : (typeof target_type_id === 'number' ? target_type_id : null);
        await loadDatasetClasses(state.dataset_id);
        applyModeVisibility();
        refreshProgress();
        await loadSessions();
        await renderDatasetManager();
        toast('Dataset created');
      } catch { toast('Create failed'); }
    };
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
      const h4 = document.createElement('h4');
      h4.textContent = String(s.session_id || '');
      const meta1 = document.createElement('div');
      meta1.className = 'selector__meta';
      meta1.textContent = `Game: ${String(s.game_name || '-') } · Frames: ${String(s.frames_count || '-')}`;
      const meta2 = document.createElement('div');
      meta2.className = 'selector__meta';
      meta2.textContent = `Started: ${s.start_time ? new Date(s.start_time).toLocaleString() : '-'}`;
      const meta3 = document.createElement('div');
      meta3.className = 'selector__meta';
      meta3.textContent = hint;
      div.appendChild(h4);
      div.appendChild(meta1);
      div.appendChild(meta2);
      div.appendChild(meta3);
      // Click behavior
      div.onclick = async () => {
        if (isEnrolled) {
          selectSession(s.session_id, state.project_name, { pushHistory: true, preload: s });
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
            setCookie('currentProject', state.project_name);
            if (state.project_id != null) setCookie('currentProjectId', String(state.project_id));
            setCookie('currentDatasetId', String(state.dataset_id));
          } catch {}
          await loadSessions();
          // Auto-open after enrollment
          selectSession(s.session_id, state.project_name, { pushHistory: true, preload: s });
        } catch (e) {
          console.error(e);
          toast('Enrollment failed');
        }
      };
      list.appendChild(div);
    });
    if (sessions.length === 0) {
      list.innerHTML = '<div class="selector__meta">No annotation sessions found. Run data collection first.</div>';
    }
  }

  function renderCategories() {
    const list = els.categoryList && els.categoryList();
    if (!list) return;
    list.innerHTML = '';
    const cats = Array.isArray(state.categories) ? state.categories : [];
    cats.forEach((cat) => {
      const row = document.createElement('div');
      row.className = 'category';

      const radio = document.createElement('input');
      radio.type = 'radio';
      radio.name = 'category';
      radio.value = String(cat);

      const title = document.createElement('span');
      title.textContent = String(cat);

      const wrapDiv = document.createElement('div');
      wrapDiv.style.display = 'flex';
      wrapDiv.style.alignItems = 'center';
      wrapDiv.style.gap = 'var(--space-2)';
      const input = document.createElement('input');
      input.type = 'text';
      input.placeholder = 'key';
      
      // Use existing small input style from stylesheet
      input.className = 'input input--small';
      input.value = (state.hotkeys && state.hotkeys[cat]) ? String(state.hotkeys[cat]) : '';

      const removeBtn = document.createElement('button');
      removeBtn.className = 'btn btn--sm';
      removeBtn.setAttribute('data-remove', String(cat));
      removeBtn.textContent = 'Remove';

      row.appendChild(radio);
      row.appendChild(title);
      wrapDiv.appendChild(input);
      wrapDiv.appendChild(removeBtn);
      row.appendChild(wrapDiv);
      list.appendChild(row);

      input.addEventListener('input', () => {
        const v = (input.value || '').trim().toLowerCase();
        if (!state.hotkeys) state.hotkeys = {};
        state.hotkeys[cat] = v;
        saveCategoriesToStorage();
        renderDynamicShortcuts();
        if (state.dataset_id && state.session_id) {
          debouncedSaveSettings();
        }
      });

      radio.addEventListener('change', () => { state.frameSaved = false; highlightCategoryStates(); });

      removeBtn.addEventListener('click', () => {
        const name = removeBtn.getAttribute('data-remove');
        state.categories = (state.categories || []).filter((c) => c !== name);
        if (state.hotkeys) delete state.hotkeys[name];
        saveCategoriesToStorage();
        renderCategories();
        renderDynamicShortcuts();
        if (state.dataset_id && state.session_id) {
          debouncedSaveSettings();
        }
      });
    });
    highlightCategoryStates();
  }

  function renderDynamicShortcuts() {
    const el = els.dynamicShortcuts();
    el.innerHTML = '';
    // Build rows from categories to ensure visibility even when no hotkeys are set
    const cats = Array.isArray(state.categories) ? state.categories : [];
    cats.forEach((cat) => {
      const key = state.hotkeys && state.hotkeys[cat] ? state.hotkeys[cat] : '';
      const row = document.createElement('div');
      row.className = 'shortcut';
      const left = document.createElement('span');
      left.textContent = String(cat);
      const right = document.createElement('span');
      const keySpan = document.createElement('span');
      keySpan.className = 'key';
      const keyText = String(key || '').toUpperCase();
      keySpan.textContent = keyText || '—';
      right.appendChild(keySpan);
      row.appendChild(left);
      row.appendChild(right);
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
      // Normalize API shape: accept either {categories, hotkeys, regression} or {settings: {...}}
      const s = (settings && settings.settings && typeof settings.settings === 'object') ? settings.settings : settings;
      if (s && (Array.isArray(s.categories) || s.hotkeys)) {
        if (Array.isArray(s.categories)) state.categories = s.categories;
        if (s.hotkeys && typeof s.hotkeys === 'object') state.hotkeys = s.hotkeys;
        state.settingsLoaded = true;
        saveCategoriesToStorage();
      } else {
        // Fallback to local defaults if backend has nothing yet
        loadCategoriesFromStorage();
      }
      // Apply regression settings if present
      const reg = s && s.regression ? s.regression : null;
      const regMin = (reg && Number.isFinite(reg.min)) ? reg.min : (Number.isFinite(s?.regressionMin) ? s.regressionMin : null);
      const regMax = (reg && Number.isFinite(reg.max)) ? reg.max : (Number.isFinite(s?.regressionMax) ? s.regressionMax : null);
      const regShort = (reg && Array.isArray(reg.shortcuts)) ? reg.shortcuts : (Array.isArray(s?.regressionShortcuts) ? s.regressionShortcuts : null);
      if (regMin != null) state.regressionMin = regMin;
      if (regMax != null) state.regressionMax = regMax;
      if (regShort) state.regressionShortcuts = regShort;
      try {
        if (state.regressionMin != null) localStorage.setItem(lsKey('regression_min'), String(state.regressionMin));
        if (state.regressionMax != null) localStorage.setItem(lsKey('regression_max'), String(state.regressionMax));
        if (state.regressionShortcuts) localStorage.setItem(lsKey('regression_shortcuts'), JSON.stringify(state.regressionShortcuts || []));
      } catch {}
      // If current dataset is Regression, refresh UI for panel + shortcuts
      if (state.target_type_name === 'Regression') {
        try { renderRegressionPanel(); } catch {}
        try { renderRegressionShortcuts(); } catch {}
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
      // Use unified /api/frame; include dataset_id when available to retrieve annotation payload
      const params = {
        session_id: state.session_id,
        project_name: state.project_name,
        idx: idx,
      };
      if (state.dataset_id) params.dataset_id = state.dataset_id;
      // Cancel any in-flight frame request before starting a new one
      if (frameRequestController) {
        try { frameRequestController.abort(); } catch {}
      }
      frameRequestController = new AbortController();
      const data = await apiGetWithSignal('frame', params, frameRequestController.signal);

      state.currentIdx = idx;
      const frame = data.frame || {};
      // Annotation/effective settings are embedded when dataset_id is provided
      let ann = (data && data.annotation) ? data.annotation : {};

      // Compute total frames only once from first frame request (we don't have a stats endpoint)
      // We can infer totalFrames from last session discovery payload later; for now keep it if already set
      if (!state.totalFrames || state.totalFrames < 1) {
        // Try to read value from localStorage hint if we saved earlier; else leave unknown
        let tf = 0;
        try {
          const s = localStorage.getItem(lsKey('totalFrames'));
          if (s != null) tf = parseInt(s, 10);
        } catch {}
        if (Number.isFinite(tf) && tf > 0) state.totalFrames = tf;
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
      // Restore MultiLabel selections if present
      if (state.target_type_name === 'MultiLabelClassification') {
        const ids = (Array.isArray(ann?.class_ids) ? ann.class_ids : (Array.isArray(ann?.annotations?.class_ids) ? ann.annotations.class_ids : []));
        if (Array.isArray(ids)) {
          const checks = document.querySelectorAll('#category-list input[type="checkbox"][data-class-id]');
          checks.forEach(ch => {
            const id = parseInt(ch.getAttribute('data-class-id'), 10);
            ch.checked = ids.includes(id);
          });
        }
      }
      let notesVal = '';
      // Prefer effective_settings.notes; fallback to legacy override_settings.notes if present
      if (ann && ann.effective_settings && typeof ann.effective_settings.notes === 'string') {
        notesVal = ann.effective_settings.notes;
      } else if (ann && ann.override_settings && typeof ann.override_settings.notes === 'string') {
        notesVal = ann.override_settings.notes;
      }
      els.notes().value = notesVal;

      // Navigation UI text and buttons
      els.frameInput().value = state.totalFrames > 0 ? `${idx + 1}/${state.totalFrames}` : `${idx + 1}`;
      els.prevBtn().disabled = idx <= 0;
      els.firstBtn().disabled = idx <= 0;
      els.nextBtn().disabled = state.totalFrames ? idx >= state.totalFrames - 1 : false;
      els.lastBtn().disabled = state.totalFrames ? idx >= state.totalFrames - 1 : false;

      // Prefetch next frame image to speed up forward navigation
      try {
        const nextIdx = (typeof state.totalFrames === 'number' && state.totalFrames > 0)
          ? Math.min(idx + 1, state.totalFrames - 1)
          : idx + 1;
        if (nextIdx > idx) {
          const paramsNext = {
            session_id: state.session_id,
            idx: nextIdx,
          };
          const prefetchUrl = withBase(`/api/image?${new URLSearchParams(paramsNext).toString()}`);
          const img = new Image();
          img.src = prefetchUrl;
        }
      } catch {}

      // We cannot compute progress without totals; leave as-is
    } catch (e) {
      // Swallow aborts (user navigated quickly)
      if (e && (e.name === 'AbortError' || String(e.message || '').includes('AbortError'))) return;
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
          override_settings: {
            notes: els.notes().value,
          },
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
      // Clear session-scoped keys
      const sid = state.session_id ? String(state.session_id) : null;
      if (sid) {
        try { localStorage.removeItem(`annot:${sid}:categories`); } catch {}
        try { localStorage.removeItem(`annot:${sid}:hotkeys`); } catch {}
        try { localStorage.removeItem(`annot:${sid}:totalFrames`); } catch {}
        try { localStorage.removeItem(`dataset:${sid}`); } catch {}
      }
      // Clear cookies
      deleteCookie('currentProject');
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
          debouncedSaveSettings();
        }
      }
      input.value = '';
    });
    els.newCategoryInput().addEventListener('keypress', (e) => { if (e.key === 'Enter') els.addCategoryBtn().click(); });
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
    const btnReindex = document.getElementById('reindex-sessions');
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
