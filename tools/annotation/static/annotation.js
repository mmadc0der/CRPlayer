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

  // Helper: modal picker for target types (returns Promise<number|null>)
  async function chooseTargetType(defaultId = 1) {
    try {
      const types = await listTargetTypes();
      if (!Array.isArray(types) || types.length === 0) return null;
      // Build transient modal
      const wrap = document.createElement('div');
      wrap.className = 'modal';
      wrap.innerHTML = `
        <div class="modal__backdrop"></div>
        <div class="modal__content" style="max-width:420px">
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
          await renderProjectManager();
          toast('Project updated');
        } catch { toast('Update failed'); }
      };
      row.querySelector('[data-delete]').onclick = async () => {
        if (!confirm(`Delete project '${p.name}'? (must have no datasets)`)) return;
        try {
          await apiDelete(`projects/${p.id}`);
          await populateProjectSelect();
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
        await renderProjectManager();
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
    form.innerHTML = `
      <input class="input m-input" id="dm-name" placeholder="Dataset name">
      <input class="input m-input" id="dm-desc" placeholder="Description (optional)">
      <select class="input" id="dm-type"></select>
      <button class="btn" id="dm-create">Create</button>
    `;
    body.appendChild(form);
    // Populate target types dynamically
    try {
      const types = await listTargetTypes();
      const typeSel = form.querySelector('#dm-type');
      typeSel.innerHTML = '';
      types.forEach(t => {
        const opt = document.createElement('option');
        opt.value = String(t.id);
        opt.textContent = t.name || `Type ${t.id}`;
        typeSel.appendChild(opt);
      });
      // prefer SingleLabel (1) if present, else first
      const preferred = types.find(t => String(t.id) === '1');
      typeSel.value = preferred ? '1' : (types[0] ? String(types[0].id) : '1');
    } catch {}
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
        // Build inline modal with inputs and dropdown for types
        const types = await listTargetTypes().catch(() => []);
        const wrap = document.createElement('div');
        wrap.className = 'modal';
        wrap.innerHTML = `
          <div class="modal__backdrop"></div>
          <div class="modal__content" style="max-width:520px">
            <div class="modal__header"><h3>Edit Dataset</h3></div>
            <div class="modal__body" style="display:flex;flex-direction:column;gap:8px">
              <input class="input" id="ed-name" placeholder="Dataset name" value="${d.name}">
              <input class="input" id="ed-desc" placeholder="Description (optional)" value="${d.description || ''}">
              <select class="input" id="ed-type"></select>
            </div>
            <div class="modal__footer" style="display:flex;gap:8px;justify-content:flex-end">
              <button class="btn" id="ed-cancel">Cancel</button>
              <button class="btn btn--primary" id="ed-save">Save</button>
            </div>
          </div>`;
        document.body.appendChild(wrap);
        const typeSel = wrap.querySelector('#ed-type');
        typeSel.innerHTML = '';
        types.forEach(t => {
          const opt = document.createElement('option');
          opt.value = String(t.id);
          opt.textContent = t.name || `Type ${t.id}`;
          typeSel.appendChild(opt);
        });
        typeSel.value = String(d.target_type_id);
        wrap.classList.remove('hidden');
        const cleanup = () => { try { document.body.removeChild(wrap); } catch {} };
        wrap.querySelector('#ed-cancel').onclick = cleanup;
        wrap.querySelector('.modal__backdrop').onclick = cleanup;
        wrap.querySelector('#ed-save').onclick = async () => {
          const newName = wrap.querySelector('#ed-name').value || d.name;
          const newDesc = wrap.querySelector('#ed-desc').value || '';
          const target_type_id = parseInt(typeSel.value, 10);
          cleanup();
          try {
            await apiPut(`datasets/${d.id}`, { name: newName, description: newDesc, target_type_id });
            await populateDatasetSelect(String(d.id));
            await renderDatasetManager();
            toast('Dataset updated');
          } catch { toast('Update failed'); }
        };
      };
      row.querySelector('[data-delete]').onclick = async () => {
        if (!confirm(`Delete dataset '${d.name}'?`)) return;
        try {
          await apiDelete(`datasets/${d.id}`);
          await populateDatasetSelect();
          await renderDatasetManager();
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
          refreshProgress();
          closeModal('dataset-modal');
          toast('Dataset selected');
        }
      };
      listWrap.appendChild(row);
    });
    // Create handler
    form.querySelector('#dm-create').onclick = async () => {
      const name = (form.querySelector('#dm-name').value || '').trim();
      const desc = (form.querySelector('#dm-desc').value || '').trim();
      const type = parseInt(form.querySelector('#dm-type').value, 10);
      if (!name) { toast('Name required'); return; }
      try {
        const created = await createDataset(projectId, name, desc, type);
        await populateDatasetSelect(String(created.id));
        await renderDatasetManager();
        toast('Dataset created');
      } catch { toast('Create failed'); }
    };
  }

  function withBase(p) { return `${APP_BASE}${p}`; }

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
    return apiPost(`datasets/${datasetId}/enroll_session`, { session_id: sessionId, settings });
  }
  async function datasetProgress(datasetId) {
    return apiGet(`datasets/${datasetId}/progress`);
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

  // Projects toolbar
  async function populateProjectSelect(prefName = null) {
    const select = els.projectSelect();
    if (!select) return;
    select.innerHTML = '';
    let projects = [];
    try { projects = await listProjects(); } catch { projects = []; }
    // Fallback default option if none
    if (!projects || projects.length === 0) {
      const opt = document.createElement('option');
      opt.value = 'default'; opt.textContent = 'default';
      select.appendChild(opt);
      state.project_name = 'default';
      return;
    }
    projects.forEach(p => {
      const opt = document.createElement('option');
      opt.value = p.name; opt.textContent = p.name; opt.dataset.projectId = p.id;
      select.appendChild(opt);
    });
    // Choose preferred or saved project
    const saved = prefName || (localStorage.getItem('currentProject') || 'default');
    const match = Array.from(select.options).find(o => o.value === saved) || select.options[0];
    if (match) { select.value = match.value; state.project_name = match.value; }
  }

  // Use backend-provided target_type_name; no frontend mapping to avoid drift

  function getSelectedProjectId() {
    const ps = els.projectSelect();
    if (!ps) return null;
    const opt = ps.selectedOptions && ps.selectedOptions[0];
    if (!opt) return null;
    const pid = parseInt(opt.dataset.projectId, 10);
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
      const targetVal = prefId || savedId || String(datasets[0].id);
      select.value = targetVal;
      const selOpt = select.selectedOptions[0];
      state.dataset_id = Number(select.value);
      state.dataset_name = selOpt ? (selOpt.dataset.datasetName || null) : null;
    } else {
      state.dataset_id = null;
      state.dataset_name = null;
    }
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
      openBtn.addEventListener('click', () => {
        const sel = els.projectSelect();
        const proj = sel && sel.value ? sel.value : state.project_name || 'default';
        // If user picked a dataset in toolbar, set it as active
        const dsSel = els.datasetSelect();
        if (dsSel && dsSel.value) {
          state.dataset_id = Number(dsSel.value);
          state.dataset_name = (dsSel.selectedOptions[0] && dsSel.selectedOptions[0].dataset.datasetName) || null;
        }
        selectSession(s.session_id, proj, { pushHistory: true, preload: s });
      });
      enrollBtn.addEventListener('click', () => startEnrollmentFlow(s.session_id));
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
        const dsName = prompt('New dataset name:');
        if (!dsName) return;
        const dsDesc = prompt('Description (optional):') || '';
        const pickedType = await chooseTargetType(1);
        if (pickedType == null) return;
        const targetTypeId = pickedType;
        const createdDs = await createDataset(projectId, dsName, dsDesc, targetTypeId);
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
      const sessions = await apiGet('sessions');
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
    // If we still don't have a valid dataset, stop here and prompt user to enroll
    if (!(Number.isFinite(Number(state.dataset_id)) && Number(state.dataset_id) > 0)) {
      toast('Please enroll this session into a dataset first');
      // keep session selector visible
      showSessionSelector({ pushHistory: false });
      return;
    }
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
      const res = await apiPost('annotations/single_label', payload);
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
    // Project selector
    const ps = els.projectSelect();
    if (ps) {
      ps.addEventListener('change', () => {
        state.project_name = ps.value || 'default';
        try { localStorage.setItem('currentProject', state.project_name); } catch {}
        populateDatasetSelect();
      });
    }
    const manageBtn = document.getElementById('manage-projects');
    if (manageBtn) {
      manageBtn.addEventListener('click', async () => {
        await renderProjectManager();
        openModal('project-modal');
      });
    }
    // Dataset selector
    const dsSel = els.datasetSelect();
    if (dsSel) {
      dsSel.addEventListener('change', () => {
        const opt = dsSel.selectedOptions && dsSel.selectedOptions[0];
        state.dataset_id = dsSel.value ? Number(dsSel.value) : null;
        state.dataset_name = opt ? (opt.dataset.datasetName || null) : null;
        if (state.session_id && state.dataset_id) {
          try { localStorage.setItem(`dataset:${state.session_id}`, JSON.stringify({ id: state.dataset_id, name: state.dataset_name || '' })); } catch {}
        }
        refreshProgress();
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
