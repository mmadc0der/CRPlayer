// Game State Annotation Tool - Stateless Frontend
// Uses: /api/sessions, /api/frame?session_path&project_name&idx, /api/image?session_path&idx, /api/save_annotation

(function () {
  'use strict';

  // App state
  let state = {
    session_id: null,
    project_name: 'default',
    dataset_id: null,
    dataset_name: null,
    currentIdx: 0,
    totalFrames: 0,
    categories: [],
    hotkeys: {},
    savedCategoryForFrame: null,
    frameSaved: false,
    settingsLoaded: false,
  };

  const els = {
    sessionList: () => document.getElementById('session-list'),
    sessionSelector: () => document.getElementById('session-selector'),
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
    const res = await fetch(`${url}?${usp.toString()}`);
    if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
    return res.json();
  }

  async function ensureDatasetSelected(sessionId) {
    // Try restore
    try {
      const saved = localStorage.getItem(`dataset:${sessionId}`);
      if (saved) {
        const obj = JSON.parse(saved);
        if (obj && obj.id) {
          state.dataset_id = obj.id; state.dataset_name = obj.name || null; return obj;
        }
      }
    } catch {}

    // Prompt choose or create project/dataset, and enroll idempotently
    await startEnrollmentFlow(sessionId);
    // After enrollment, datasetId will be last created/selected. We need to re-read from storage
    try {
      const saved2 = localStorage.getItem(`dataset:${sessionId}`);
      if (saved2) return JSON.parse(saved2);
    } catch {}
    return null;
  }

  async function fetchDatasetSessionSettings(datasetId, sessionId) {
    try {
      const res = await fetch(`/api/datasets/${datasetId}/sessions/${encodeURIComponent(sessionId)}/settings`);
      if (!res.ok) throw new Error('settings fetch failed');
      const data = await res.json();
      return data.settings || {};
    } catch (e) {
      console.warn('No settings for dataset/session, using defaults');
      return {};
    }
  }

  async function apiPost(url, body) {
    const res = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
    return res.json();
  }

  async function apiPostNoBody(url) {
    const res = await fetch(url, { method: 'POST' });
    if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
    return res.json();
  }

  // Projects & Datasets API helpers (DB-backed)
  async function listProjects() {
    return apiGet('/api/projects');
  }
  async function createProject(name, description = '') {
    return apiPost('/api/projects', { name, description });
  }
  async function listDatasets(projectId) {
    return apiGet(`/api/projects/${projectId}/datasets`);
  }
  async function createDataset(projectId, name, description = '', target_type_id = 2) {
    return apiPost(`/api/projects/${projectId}/datasets`, { name, description, target_type_id });
  }
  async function enrollSession(datasetId, sessionId, settings = undefined) {
    return apiPost(`/api/datasets/${datasetId}/enroll_session`, { session_id: sessionId, settings });
  }
  async function datasetProgress(datasetId) {
    return apiGet(`/api/datasets/${datasetId}/progress`);
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

  function saveSessionSelection() {
    try {
      localStorage.setItem('currentSession', state.session_id || '');
      localStorage.setItem('currentProject', state.project_name || 'default');
      if (state.dataset_id) localStorage.setItem(`dataset:${state.session_id}`, JSON.stringify({ id: state.dataset_id, name: state.dataset_name || '' }));
      localStorage.setItem('appState', 'annotation');
    } catch {}
  }

  // UI rendering
  function renderSessionList(sessions) {
    const list = els.sessionList();
    list.innerHTML = '';
    sessions.forEach((s) => {
      const div = document.createElement('div');
      div.className = 'selector__item';
      div.innerHTML = `
        <h4>${s.session_id}</h4>
        <div class="selector__meta">Game: ${s.game_name || '-'} · Frames: ${s.frames_count || '-'}</div>
        <div class="selector__meta">Started: ${s.start_time ? new Date(s.start_time).toLocaleString() : '-'}</div>
        <div class="selector__actions" style="margin-top:6px; display:flex; gap:8px;">
          <button class="btn btn--primary" data-action="open">Open</button>
          <button class="btn" data-action="enroll">Enroll</button>
        </div>
      `;
      const openBtn = div.querySelector('[data-action="open"]');
      const enrollBtn = div.querySelector('[data-action="enroll"]');
      openBtn.addEventListener('click', () => selectSession(s.session_id, 'default', { pushHistory: true, preload: s }));
      enrollBtn.addEventListener('click', () => startEnrollmentFlow(s.session_id));
      list.appendChild(div);
    });
    if (sessions.length === 0) {
      list.innerHTML = '<div class="selector__meta">No annotation sessions found. Run data collection first.</div>';
    }
  }

  async function startEnrollmentFlow(sessionId) {
    try {
      // 1) Choose or create Project
      const projects = await listProjects();
      const choices = projects.map((p, i) => `${i + 1}) ${p.name}`).join('\n');
      let sel = prompt(`Select a project by number or type 'new' to create:\n${choices}`);
      let projectId;
      if (sel && sel.toLowerCase() === 'new') {
        const name = prompt('New project name:');
        if (!name) return;
        const desc = prompt('Description (optional):') || '';
        const created = await createProject(name, desc);
        projectId = created.id;
      } else {
        const idx = parseInt(sel, 10) - 1;
        if (Number.isNaN(idx) || idx < 0 || idx >= projects.length) return;
        projectId = projects[idx].id;
      }

      // 2) Choose or create Dataset in selected Project
      const datasets = await listDatasets(projectId);
      const dsChoices = datasets.map((d, i) => `${i + 1}) ${d.name} [type ${d.target_type_id}]`).join('\n');
      sel = prompt(`Select a dataset by number or type 'new' to create:\n${dsChoices || '(none yet)'}`);
      let datasetId;
      if (sel && sel.toLowerCase() === 'new') {
        const dsName = prompt('New dataset name:');
        if (!dsName) return;
        const dsDesc = prompt('Description (optional):') || '';
        const typeStr = prompt('Target type id (1=Regression, 2=SingleLabelClassification, 3=MultiLabelClassification):', '2') || '2';
        const type = Math.max(1, Math.min(3, parseInt(typeStr, 10) || 2));
        const createdDs = await createDataset(projectId, dsName, dsDesc, type);
        datasetId = createdDs.id;
        // Persist selection
        state.dataset_id = datasetId;
        state.dataset_name = createdDs.name;
        try { localStorage.setItem(`dataset:${sessionId}`, JSON.stringify({ id: datasetId, name: createdDs.name, target_type_id: createdDs.target_type_id })); } catch {}
      } else {
        const idx = parseInt(sel, 10) - 1;
        if (Number.isNaN(idx) || idx < 0 || idx >= datasets.length) return;
        const picked = datasets[idx];
        datasetId = picked.id;
        state.dataset_id = datasetId;
        state.dataset_name = picked.name;
        try { localStorage.setItem(`dataset:${sessionId}`, JSON.stringify({ id: datasetId, name: picked.name, target_type_id: picked.target_type_id })); } catch {}
      }

      // 3) Optional baseline settings JSON
      let settings = undefined;
      const wantSettings = confirm('Provide baseline settings JSON for this dataset+session pair?');
      if (wantSettings) {
        const raw = prompt('Enter JSON (e.g., {"categories":["battle","menu"],"hotkeys":{}}):', '{}') || '{}';
        try { settings = JSON.parse(raw); } catch { settings = undefined; }
      }

      // 4) Enroll
      await enrollSession(datasetId, sessionId, settings);
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
      el.appendChild(mk('Total', p.total ?? '-'));
      el.appendChild(mk('Labeled', p.labeled ?? '-'));
      el.appendChild(mk('Unlabeled', p.unlabeled ?? '-'));
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
          <button class="btn" data-remove="${cat}">Remove</button>
        </div>
      `;
      wrap.appendChild(row);

      const input = row.querySelector('input.input');
      input.addEventListener('input', (e) => {
        const key = e.target.value.toLowerCase();
        const category = e.target.dataset.category;
        if (key) state.hotkeys[category] = key; else delete state.hotkeys[category];
        renderDynamicShortcuts();
        saveCategoriesToStorage();
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
      const sessions = await apiGet('/api/sessions');
      renderSessionList(sessions);
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
    const settings = await fetchDatasetSessionSettings(state.dataset_id, state.session_id);
    // Apply settings to categories/hotkeys if present
    if (settings && Array.isArray(settings.categories)) {
      state.categories = settings.categories.slice();
    } else {
      loadCategoriesFromStorage();
    }
    if (settings && settings.hotkeys && typeof settings.hotkeys === 'object') {
      state.hotkeys = settings.hotkeys;
    }
    state.settingsLoaded = true;
    renderCategories();
    renderDynamicShortcuts();

    // Show annotation interface
    els.sessionSelector().classList.add('hidden');
    els.annotationInterface().classList.remove('hidden');
    els.sessionInfo().classList.remove('hidden');
    const dsText = state.dataset_name ? ` · Dataset: ${state.dataset_name} (#${state.dataset_id})` : (state.dataset_id ? ` · Dataset: #${state.dataset_id}` : '');
    els.sessionName().textContent = `Session: ${session_id} · Project: ${project_name}${dsText}`;

    saveSessionSelection();
    if (opts && opts.pushHistory) {
      history.pushState({ state: 'annotation', session: session_id, project: project_name }, 'Annotation', '#annotation');
    }
    // Update dataset progress panel
    refreshProgress();
    await loadFrame(0);
  }

  async function loadFrame(idx) {
    if (idx < 0) idx = 0;
    try {
      const data = await apiGet('/api/frame', {
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
      els.img().src = `/api/image?${new URLSearchParams({ session_id: state.session_id, idx: String(idx) }).toString()}`;
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
    // Save single-label annotation to DB-backed endpoint
    if (!state.dataset_id) {
      toast('Select a dataset first (use Enroll)');
      return false;
    }
    const category = getSelectedCategory();
    if (!category) {
      toast('Pick a category');
      return false;
    }
    // Map category -> class_id using settings.categories order
    const idx = Array.isArray(state.categories) ? state.categories.indexOf(category) : -1;
    if (idx < 0) {
      toast('Category not in dataset settings');
      return false;
    }
    try {
      const payload = {
        session_id: state.session_id,
        dataset_id: state.dataset_id,
        frame_idx: state.currentIdx,
        class_id: idx,
        // override_settings can be added in future
      };
      const res = await apiPost('/api/annotations/single_label', payload);
      if (res && (res.ok || res.saved || res.status === 'ok')) {
        state.savedCategoryForFrame = category;
        state.frameSaved = true;
        toast('Saved');
        // Update progress after successful save
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
    } catch {}
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
    els.addCategoryBtn().addEventListener('click', () => {
      const input = els.newCategoryInput();
      const name = (input.value || '').trim();
      if (!name) return;
      if (!state.categories.includes(name)) {
        state.categories.push(name);
        saveCategoriesToStorage();
        renderCategories();
        renderDynamicShortcuts();
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

    // Header back button
    const header = document.querySelector('.header');
    const backBtn = document.createElement('button');
    backBtn.textContent = 'Back to Sessions';
    backBtn.className = 'btn';
    backBtn.style.marginTop = '8px';
    backBtn.addEventListener('click', showSessionSelector);
    header.appendChild(backBtn);

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
      if (st && st.state === 'annotation' && st.session_id && st.project) {
        // navigate without pushing a new history entry
        selectSession(st.session_id, st.project, { pushHistory: false });
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
        await apiPostNoBody('/api/reindex');
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
    bindEvents();

    // Initial state from URL or localStorage
    const hash = (location.hash || '').toLowerCase();
    // Backward compatibility: if 'currentSession' stored a path earlier, we still pass it as session_id string
    const savedSession = localStorage.getItem('currentSession');
    const savedProject = localStorage.getItem('currentProject');
    if (hash === '#annotation' && savedSession && savedProject) {
      await selectSession(savedSession, savedProject, { pushHistory: false });
    } else {
      await loadSessions();
      history.replaceState({ state: 'sessions' }, 'Select Session', '#sessions');
    }
  }

  document.addEventListener('DOMContentLoaded', init);
})();
