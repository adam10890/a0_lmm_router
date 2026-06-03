/**
 * dashboard-store.js — Alpine.js store for the LMM Router dashboard.
 *
 * Polls /plugins/a0_lmm_router/lmm_compute_stats every N seconds and
 * exposes reactive data for GPU, CPU, slots, and model recommendations.
 */

// ── A0 API compatibility shim ────────────────────────────────────────────
// A0 v1.15 dropped `window.api`. Most plugin pages import /js/api.js as an
// ES module, but this dashboard is loaded as a classic script from a modal.
// Use any already-injected global first, then lazily import the official API
// module so the dashboard is not stuck in "Connection failed". The modal can
// inject this script more than once into the same page, so keep the cache on
// globalThis instead of declaring a top-level `let`.
globalThis.__a0LmmDashboardApiModulePromise =
  globalThis.__a0LmmDashboardApiModulePromise || null;

function _a0ApiCall(endpoint, data) {
  if (typeof globalThis.sendJsonData === 'function') {
    return globalThis.sendJsonData(endpoint, data);
  }
  if (globalThis.api && typeof globalThis.api.callJsonApi === 'function') {
    return globalThis.api.callJsonApi(endpoint, data);
  }
  if (!globalThis.__a0LmmDashboardApiModulePromise) {
    globalThis.__a0LmmDashboardApiModulePromise = import('/js/api.js');
  }
  return globalThis.__a0LmmDashboardApiModulePromise
    .then(({ callJsonApi }) => callJsonApi(endpoint, data));
}

function createDashboardStore() {
  const POLL_INTERVAL_MS = 5000;
  // A0 dispatches all plugin APIs under /api/plugins/<plugin_name>/<handler>.
  // (See /a0/helpers/api.py register_api_route.)
  const API_BASE = '/api/plugins/a0_lmm_router';

  const ENDPOINTS = {
    computeStats: `${API_BASE}/lmm_compute_stats`,
    statsSummary: `${API_BASE}/lmm_stats_summary`,
    recommend:    `${API_BASE}/lmm_model_recommend`,
    install:      `${API_BASE}/lmm_model_install`,
    listModels:   `${API_BASE}/llamacpp_list_models`,
    assignModel:  `${API_BASE}/assign_model`,
    jobStatus:    `${API_BASE}/job_status`,
    control:      `${API_BASE}/llamacpp_control`,
    status:       `${API_BASE}/llamacpp_status`,
    ignite:       `${API_BASE}/lmm_fleet_ignite`,
    hostIgnite:   `${API_BASE}/lmm_host_ignite`,
    hardwareScan:     `${API_BASE}/lmm_hardware_recommend`,
    slotRecs:         `${API_BASE}/lmm_slot_recommendations`,
    routerModels:     `${API_BASE}/router_models`,
    setRouterDefault: `${API_BASE}/set_router_default`,
    routerAliases:    `${API_BASE}/router_aliases`,
    setRouterAliasModel: `${API_BASE}/set_router_alias_model`,
    fleetReconnect:   `${API_BASE}/fleet_reconnect`,
  };

  return {
    // ── state ──────────────────────────────────────────────────
    gpus: [],
    cpu: { load_pct: 0, ram_total_mb: 0, ram_used_mb: 0, ram_free_mb: 0 },
    slots: [],
    recommendations: [],
    installedModels: {},     // keyed by model_id from manifest
    installStatus: {},        // keyed by filename or job_id
    jobPollTimers: {},        // keyed by job_id
    pollTimer: null,
    loading: true,
    error: '',
    lastUpdated: null,
    swapInProgress: {},       // keyed by slot_id — true when assigning/restarting
    slotErrors: {},            // keyed by slot_id — error message from last assign/load
    fleetMode: { mode: 'unknown', router_running: false, three_slot_running: false, containers: {} },
    reconnecting: false,       // true while reconnectFleet() is probing
    roleBindings: [],
    roleBindingsSource: '',
    roleBindingsError: '',
    roleBindingSelections: {},
    roleBindingShowAll: {},
    roleBindingUpdating: {},
    roleBindingErrors: {},
    roleBindingJustApplied: {},
    toasts: [],
    _toastSeq: 0,

    // Fleet ignition state
    igniteState: 'idle',      // idle | pending | ok | needs_host | error
    igniteMessage: '',
    igniteHostHint: '',

    // HF install form state
    installForm: {
      repo: '',
      file: '',
      role: '',
      installing: false,
      progress: 0,
      status: '',
      error: '',
      success: '',
      jobId: null,
    },

    // Live tier — derived from compute_monitor stats on every poll
    // (single source of truth shared with all role-aware UI)
    liveTier: {
      tier: '',                  // T0..T9, '' until first response
      eim_gb: 0,
      eim_basis: '',             // 'vram' | 'unified_memory' | 'system_ram'
      gpu_summary: '',
      ram_gb: 0,
    },

    // Per-slot suggestions, keyed by slot_id. Each entry:
    //   { slot_id, role, current_model_id, current_size_gb,
    //     current_status, current_status_reason, suggestions: [...] }
    slotSuggestions: {},
    slotRecsError: '',

    // Installed-models diagnostics (when the host returned empty)
    installedDiag: null,

    // Hardware-aware picks (collapsible Advanced section — on-demand scan)
    hw: {
      loading: false,
      error: '',
      snapshot: '',
      source: '',          // 'host_helper' | 'local' | ''
      os_name: '',
      os_version: '',
      cpu_name: '',
      gpu_label: '',       // pre-formatted GPU summary for the header pill
      ram_gb: 0,
      disk_free_gb: 0,
      eim_gb: 0,
      eim_basis: '',       // 'vram' | 'unified_memory' | 'system_ram'
      tier: '',            // T0..T9, '' before first scan
      picks: { comfortable: null, balanced: null, stretch: null },
      notes: [],
    },

    // Stats tracking (failover + usage)
    statsWindow: '24h',      // 24h | 7d | 30d
    stats: {
      total_requests: 0,
      total_requests_window: 0,
      total_tokens_input: 0,
      total_tokens_output: 0,
      total_savings_usd: 0.0,
      total_savings_window: 0.0,
      slots: [],
      failovers: { total: 0, by_reason: {}, by_slot: {} },
      hourly_distribution: [],
    },

    // ── lifecycle ──────────────────────────────────────────────
    init() {
      this.refresh();
      this.pollTimer = setInterval(() => this.refresh(), POLL_INTERVAL_MS);
    },

    cleanup() {
      if (this.pollTimer) clearInterval(this.pollTimer);
      // Clean up any job polling timers
      Object.values(this.jobPollTimers).forEach(t => clearInterval(t));
      this.jobPollTimers = {};
    },

    // ── data fetching ──────────────────────────────────────────
    async refresh() {
      // Run stats + installed models first; slot recs depends on both.
      await Promise.all([
        this._fetchStats(),
        this._fetchStatsSummary(),
        this._fetchInstalledModels(),
        this._fetchRoleBindings(),
      ]);
      // Slot recommendations needs slots + installed models loaded.
      await this._fetchSlotRecommendations();
      this.loading = false;
    },

    async _fetchInstalledModels() {
      try {
        const d = await _a0ApiCall(ENDPOINTS.listModels, {});
        if (d.ok && d.models) {
          this.installedModels = d.models;
        }
      } catch (_) { /* silent */ }
    },

    async _fetchStatsSummary() {
      try {
        const d = await _a0ApiCall(ENDPOINTS.statsSummary, { window: this.statsWindow });
        if (d.ok && d.stats) {
          this.stats = d.stats;
        }
      } catch (_) { /* silent */ }
    },

    setStatsWindow(window) {
      this.statsWindow = window;
      this._fetchStatsSummary();
    },

    async _fetchStats() {
      try {
        const d = await _a0ApiCall(ENDPOINTS.computeStats, {});
        if (d.ok) {
          this.gpus = d.gpus || [];
          this.cpu = d.cpu || this.cpu;
          this.fleetMode = d.fleet_mode || this.fleetMode;
          // Merge fresh slot data into existing slots so UI-only
          // properties (selectedModelId, _ctxSlider, _ctxOverride)
          // survive poll refreshes.
          const fresh = d.slots || [];
          const oldMap = {};
          for (const s of (this.slots || [])) {
            oldMap[s.id] = s;
          }
          this.slots = fresh.map(s => {
            const old = oldMap[s.id];
            if (old) {
              if (old.selectedModelId !== undefined) s.selectedModelId = old.selectedModelId;
              if (old._ctxSlider !== undefined) s._ctxSlider = old._ctxSlider;
              if (old._ctxOverride !== undefined) s._ctxOverride = old._ctxOverride;
            }
            if (s._ctxSlider === undefined) {
              const binding = this.getRoleBinding(s.role);
              if (binding && binding.ctx_size) s._ctxSlider = binding.ctx_size;
            }
            return s;
          });
          this.lastUpdated = new Date();
          this.error = '';
        } else {
          this.error = d.error || 'Stats unavailable';
        }
      } catch (e) {
        this.error = 'Connection failed';
      }
    },

    async _fetchRecommendations() {
      try {
        const d = await _a0ApiCall(ENDPOINTS.recommend, {});
        if (d.ok) this.recommendations = d.recommendations || [];
      } catch (_) { /* silent */ }
    },

    // ── Unified slot recommendations (role-aware, tier-aware) ──
    // Pulls: live tier + installed models + per-slot suggestions in one call.
    // Driven by the backend's lmm_slot_recommendations endpoint which
    // internally reads compute_monitor + tier_catalog + fleet_models.
    async _fetchSlotRecommendations() {
      try {
        const d = await _a0ApiCall(ENDPOINTS.slotRecs, {});
        if (!d.ok) {
          this.slotRecsError = d.error || 'Slot recommendations failed';
          return;
        }
        this.slotRecsError = '';

        // Live tier from the backend (derived from live compute snapshot)
        this.liveTier = {
          tier: d.tier || '',
          eim_gb: d.eim_gb || 0,
          eim_basis: d.eim_basis || '',
          gpu_summary: d.gpu_summary || '',
          ram_gb: d.ram_gb || 0,
        };

        // Index slot suggestions by slot_id for O(1) UI lookup
        const map = {};
        for (const slot of (d.slots || [])) {
          map[slot.slot_id] = slot;
        }
        this.slotSuggestions = map;

        // Installed-models diagnostics (when count == 0)
        this.installedDiag = d.installed_diagnostics || null;
      } catch (e) {
        this.slotRecsError = 'Connection failed: ' + (e.message || e);
      }
    },

    // ── Hardware-aware recommender (local-llm-recommender skill) ──
    async scanHardware() {
      this.hw.loading = true;
      this.hw.error = '';
      try {
        const d = await _a0ApiCall(ENDPOINTS.hardwareScan, {});
        if (!d.ok) {
          this.hw.error = d.error || 'Scan failed';
          this.hw.loading = false;
          return;
        }
        const h = d.hardware || {};
        const gpuLabel = (h.gpus || []).map(g => {
          const vram = g.unified_memory ? 'unified' :
                       g.total_vram_mb ? `${(g.total_vram_mb/1024).toFixed(1)} GB` : '?';
          return `${g.name} (${vram})`;
        }).join(', ');

        this.hw.snapshot = d.snapshot || '';
        this.hw.source = h.source || '';
        this.hw.os_name = h.os_name || '';
        this.hw.os_version = h.os_version || '';
        this.hw.cpu_name = h.cpu_name || '';
        this.hw.gpu_label = gpuLabel;
        this.hw.ram_gb = h.ram_gb || 0;
        this.hw.disk_free_gb = h.disk_free_gb || 0;
        this.hw.eim_gb = h.eim_gb || 0;
        this.hw.eim_basis = h.eim_basis || '';
        this.hw.tier = h.tier || '';
        this.hw.picks = d.picks || { comfortable: null, balanced: null, stretch: null };
        this.hw.notes = d.notes || [];
      } catch (e) {
        this.hw.error = 'Connection failed: ' + (e.message || e);
      } finally {
        this.hw.loading = false;
      }
    },

    copyInstall(cmd) {
      if (!cmd) return;
      try {
        navigator.clipboard.writeText(cmd);
      } catch (_) {
        const ta = document.createElement('textarea');
        ta.value = cmd;
        document.body.appendChild(ta);
        ta.select();
        try { document.execCommand('copy'); } catch (_) { /* silent */ }
        document.body.removeChild(ta);
      }
    },

    // ── actions ────────────────────────────────────────────────
    async hostAction(action) {
      this.igniteState = 'pending';
      this.igniteMessage = 'Communicating with host...';
      this.igniteHostHint = '';
      try {
        const d = await _a0ApiCall(ENDPOINTS.hostIgnite, { action });
        if (!d.ok) {
          this.igniteState = 'error';
          this.igniteMessage = d.error || 'host helper call failed';
          this.igniteHostHint = d.hint || '';
          return;
        }
        this.igniteState = 'ok';
        this.igniteMessage = d.message || `host ${action} OK`;
        // give containers a sec to come up, then refresh
        setTimeout(() => this._fetchStats(), 3000);
      } catch (e) {
        this.igniteState = 'error';
        this.igniteMessage = 'Connection failed';
      }
    },

    async igniteFleet() {
      this.igniteState = 'pending';
      this.igniteMessage = 'Checking fleet…';
      this.igniteHostHint = '';
      try {
        const d = await _a0ApiCall(ENDPOINTS.ignite, {});
        if (!d.ok) {
          this.igniteState = 'error';
          this.igniteMessage = d.error || 'ignite failed';
          return;
        }
        this.igniteMessage = d.message || '';
        this.igniteHostHint = d.host_command || d.docker_compose_hint || '';
        if (d.state === 'fleet_healthy') this.igniteState = 'ok';
        else if (d.state === 'needs_host_ignition') this.igniteState = 'needs_host';
        else if (d.state === 'no_slots') this.igniteState = 'error';
        else this.igniteState = 'needs_host';
        // refresh slot state so dots update immediately
        await this._fetchStats();
      } catch (e) {
        this.igniteState = 'error';
        this.igniteMessage = 'Connection failed';
      }
    },

    async controlSlot(op, serverId) {
      // Map slot id → role for host helper calls
      const roleMap = {
        slot_chat: 'chat', slot_utility: 'utility', slot_embedding: 'embedding',
        slot_embed: 'embedding', slot_vision: 'vision', slot_reasoning: 'reasoning',
      };
      const role = roleMap[serverId] || serverId;

      try {
        if (op === 'start') {
          // Use host helper to actually start the container via docker compose
          const d = await _a0ApiCall(ENDPOINTS.hostIgnite, { action: 'start_slot', slot: role });
          if (!d.ok) {
            this.slotErrors[serverId] = d.error || 'Start failed';
          } else {
            this.slotErrors[serverId] = '';
          }
        } else if (op === 'stop') {
          // Use host helper to actually stop the container via docker compose
          const d = await _a0ApiCall(ENDPOINTS.hostIgnite, { action: 'stop_slot', slot: role });
          if (!d.ok) {
            this.slotErrors[serverId] = d.error || 'Stop failed';
          } else {
            this.slotErrors[serverId] = '';
          }
        } else if (op === 'start_all' || op === 'stop_all') {
          // Fall back to BackendManager for bulk operations
          await _a0ApiCall(ENDPOINTS.control, { data: { operation: op, server: serverId } });
        }
        await this._fetchStats();
      } catch (e) {
        this.slotErrors[serverId] = e.message || 'Connection failed';
      }
    },

    async installModel(rec) {
      const key = rec.filename;
      this.installStatus[key] = { status: 'downloading', percent: 0, job_id: null };
      try {
        const d = await _a0ApiCall(ENDPOINTS.install, { repo_id: rec.repo_id, filename: rec.filename, role: rec.role });
        if (d.ok && d.job_id) {
          this.installStatus[key].job_id = d.job_id;
          // Start polling job status
          this._pollJobStatus(d.job_id, key);
        } else {
          this.installStatus[key].status = 'error';
          this.installStatus[key].error = d.error || 'Install failed';
        }
      } catch (_) {
        this.installStatus[key].status = 'error';
      }
    },

    async installFromForm() {
      const { repo, file, role } = this.installForm;
      if (!repo || !file) return;

      this.installForm.installing = true;
      this.installForm.progress = 0;
      this.installForm.status = 'queued';
      this.installForm.error = '';
      this.installForm.success = '';

      try {
        const d = await _a0ApiCall(ENDPOINTS.install, { repo_id: repo, filename: file, role });
        if (d.ok && d.job_id) {
          this.installForm.jobId = d.job_id;
          // Poll the job
          const timer = setInterval(async () => {
            try {
              const jd = await _a0ApiCall(ENDPOINTS.jobStatus, { job_id: d.job_id });
              if (jd.ok) {
                this.installForm.progress = jd.percent || 0;
                this.installForm.status = jd.status;
                if (jd.status === 'done') {
                  clearInterval(timer);
                  this.installForm.installing = false;
                  this.installForm.success = `Model installed: ${jd.model_id || file}`;
                  this.installForm.jobId = null;
                  this._fetchInstalledModels();
                } else if (jd.status === 'error' || jd.status === 'cancelled') {
                  clearInterval(timer);
                  this.installForm.installing = false;
                  this.installForm.error = jd.error || 'Download failed';
                  this.installForm.jobId = null;
                }
              }
            } catch (_) {
              // Silent, keep polling
            }
          }, 2000);
          this.jobPollTimers[d.job_id] = timer;
        } else {
          this.installForm.installing = false;
          this.installForm.error = d.error || 'Failed to start download';
        }
      } catch (e) {
        this.installForm.installing = false;
        this.installForm.error = e.message || 'Connection failed';
      }
    },

    _pollJobStatus(jobId, statusKey) {
      // Poll every 2 seconds
      const timer = setInterval(async () => {
        try {
          const d = await _a0ApiCall(ENDPOINTS.jobStatus, { job_id: jobId });
          if (d.ok) {
            this.installStatus[statusKey].percent = d.percent || 0;
            if (d.status === 'done') {
              this.installStatus[statusKey].status = 'done';
              clearInterval(timer);
              delete this.jobPollTimers[jobId];
              // Refresh installed models list
              this._fetchInstalledModels();
            } else if (d.status === 'error' || d.status === 'cancelled') {
              this.installStatus[statusKey].status = d.status;
              this.installStatus[statusKey].error = d.error || 'Download failed';
              clearInterval(timer);
              delete this.jobPollTimers[jobId];
            }
            // Keep polling if queued or downloading
          }
        } catch (_) {
          // Silent fail, keep polling
        }
      }, 2000);
      this.jobPollTimers[jobId] = timer;
    },

    async assignModelToSlot(slotId, modelId) {
      this.swapInProgress[slotId] = true;
      this.slotErrors[slotId] = '';
      try {
        const d = await _a0ApiCall(ENDPOINTS.assignModel, { slot: slotId, model_id: modelId, apply_now: true });
        if (d.ok) {
          this.slotErrors[slotId] = '';
          // Wait a bit for container restart, then refresh
          setTimeout(() => this._fetchStats(), 5000);
          return { ok: true, restarted: d.restarted };
        } else {
          this.slotErrors[slotId] = d.error || 'Assignment failed';
          return { ok: false, error: d.error || 'Assignment failed' };
        }
      } catch (e) {
        const msg = e.message || 'Connection failed';
        this.slotErrors[slotId] = msg;
        return { ok: false, error: msg };
      } finally {
        // Keep spinner until stats refresh completes
        setTimeout(() => { this.swapInProgress[slotId] = false; }, 5000);
      }
    },

    async _fetchRoleBindings() {
      try {
        const d = await _a0ApiCall(ENDPOINTS.routerAliases, { slot_id: 'slot_router' });
        if (!d.ok) {
          this.roleBindingsError = d.error || 'Role bindings unavailable';
          return;
        }
        this.roleBindingsError = '';
        this.roleBindings = d.roles || [];
        this.roleBindingsSource = d.source || '';
        if (d.models && Object.keys(d.models).length) {
          this.installedModels = d.models;
        }
        this._syncRoleBindingSelections();
        for (const binding of this.roleBindings) {
          for (const slot of this.slots || []) {
            if (slot.role === binding.alias && slot._ctxSlider === undefined && binding.ctx_size) {
              slot._ctxSlider = binding.ctx_size;
            }
          }
        }
      } catch (e) {
        this.roleBindingsError = e.message || 'Connection failed';
      }
    },

    _syncRoleBindingSelections() {
      for (const binding of this.roleBindings || []) {
        const current = this.normalizeModelPath(binding.model_path || '');
        const options = this.getRoleModelOptions(binding.alias);
        const match = options.find(o => this.pathsMatch(o.value, current));
        this.roleBindingSelections[binding.alias] = match ? match.value : current;
      }
    },

    normalizeModelPath(path) {
      if (!path) return '';
      let p = String(path).trim().replaceAll('\\', '/');
      while (p.startsWith('//')) p = p.slice(1);
      if (p && !p.startsWith('/')) p = '/models/' + p.replace(/^\/+/, '');
      return p.replace(/\/+/g, '/');
    },

    pathsMatch(a, b) {
      const na = this.normalizeModelPath(a);
      const nb = this.normalizeModelPath(b);
      if (!na || !nb) return false;
      if (na === nb) return true;
      return na.split('/').pop() === nb.split('/').pop();
    },

    async setRoleAliasModel(alias, modelPath) {
      if (!alias || !modelPath) return;
      this.roleBindingUpdating[alias] = true;
      this.roleBindingErrors[alias] = '';
      this.roleBindingJustApplied[alias] = false;
      this.showToast(`Updating ${alias} model — restarting router…`, 'info', 12000);
      try {
        const d = await _a0ApiCall(ENDPOINTS.setRouterAliasModel, {
          slot_id: 'slot_router',
          alias,
          model_path: modelPath,
        });
        if (!d.ok) {
          const err = d.error || d.warning || 'Failed to update alias';
          this.roleBindingErrors[alias] = err;
          this.roleBindingUpdating[alias] = false;
          this.showToast(err, 'err', 10000);
          return;
        }
        if (d.partial || d.warning) {
          this.roleBindingErrors[alias] = d.warning || 'Preset saved — restart router to apply';
          this.showToast(d.warning || 'Preset saved — restart router to apply', 'warn', 12000);
        }
        setTimeout(async () => {
          await this._fetchRoleBindings();
          await this._fetchStats();
          if (!d.partial) {
            this.roleBindingJustApplied[alias] = true;
            this.showToast(`${alias} binding updated`, 'ok');
            setTimeout(() => { this.roleBindingJustApplied[alias] = false; }, 6000);
          }
        }, 3000);
        setTimeout(() => { this.roleBindingUpdating[alias] = false; }, 8000);
      } catch (e) {
        this.roleBindingErrors[alias] = e.message || 'Connection failed';
        this.roleBindingUpdating[alias] = false;
        this.showToast(e.message || 'Connection failed', 'err');
      }
    },

    showToast(message, type = 'info', durationMs = 5000) {
      const id = ++this._toastSeq;
      // Single active toast avoids overlapping status lines in the sticky stack.
      this.toasts = [{ id, message, type }];
      setTimeout(() => {
        this.toasts = this.toasts.filter(t => t.id !== id);
      }, durationMs);
    },

    async copyModelPath(path) {
      const text = path || '';
      if (!text) return;
      try {
        await navigator.clipboard.writeText(text);
        this.showToast('Path copied to clipboard', 'ok', 2500);
      } catch (_) {
        this.showToast(text, 'info', 8000);
      }
    },

    async refreshFleetAndBindings() {
      this.showToast('Refreshing fleet status…', 'info', 2000);
      await Promise.all([this._fetchStats(), this._fetchRoleBindings()]);
    },

    // Stronger than refresh: probes the fleet over HTTP (no Docker socket
    // needed, works inside the A0 container), resets the BackendManager so
    // its cached slot view is rebuilt, then re-fetches everything. This is
    // what recovers the dashboard when a router was started out-of-band and
    // the config still describes the old 3-slot fleet.
    async reconnectFleet() {
      this.reconnecting = true;
      this.showToast('Reconnecting to LMM fleet…', 'info', 3000);
      try {
        const d = await _a0ApiCall(ENDPOINTS.fleetReconnect, { reset: true });
        if (d.ok) {
          const mode = d.mode || 'unknown';
          if (mode === 'router' && d.router) {
            const loaded = (d.router.loaded || []).join(', ') || 'none loaded yet';
            this.showToast(
              `Router detected on :${d.router.port} — ${d.router.model_count} models registered (${loaded})`,
              'ok', 6000);
          } else if (mode === 'three_slot') {
            this.showToast('3-slot fleet detected and reconnected.', 'ok', 4000);
          } else {
            this.showToast('No llama.cpp fleet is answering. Is a container running?', 'warn', 6000);
          }
        } else {
          this.showToast(d.error || 'Reconnect failed', 'error', 6000);
        }
      } catch (e) {
        this.showToast(e.message || 'Reconnect connection error', 'error', 6000);
      } finally {
        // Always re-pull state so the UI reflects whatever the probe found.
        await Promise.all([this._fetchStats(), this._fetchRoleBindings()]);
        this.reconnecting = false;
      }
    },

    async igniteRouterFleet() {
      await this.hostAction('ignite');
    },

    getSlotModelOptions(slotRole) {
      // Return installed models suitable for this slot role
      const models = Object.entries(this.installedModels || {});
      return models.map(([id, m]) => ({
        id,
        label: `${id} (${m.size_gb} GB)`,
        size_gb: m.size_gb,
        role_hint: m.role_hint,
      }));
    },

    getSlotCurrentModelId(slot) {
      // Match slot's model_id to installed models
      const slotModelId = slot.model_id;
      if (!slotModelId) return null;
      // Try exact match first
      if (this.installedModels[slotModelId]) return slotModelId;
      // Try matching by filename stem
      for (const [id, m] of Object.entries(this.installedModels)) {
        if (m.file && m.file.replace('.gguf', '') === slotModelId) return id;
        if (id === slotModelId) return id;
      }
      return null;
    },

    modelContainerPath(model) {
      if (!model) return '';
      if (model.model_path && String(model.model_path).startsWith('/models/')) return model.model_path;
      const file = model.file || model.filename || '';
      const relPath = String(model.path || '').replaceAll('\\', '/').replace(/^\/+|\/+$/g, '');
      if (!file) return '';
      return relPath ? `/models/${relPath}/${file}` : `/models/${file}`;
    },

    getRoleBinding(alias) {
      return (this.roleBindings || []).find(b => b.alias === alias) || null;
    },

    getRoleModelOptions(alias) {
      const showAll = !!this.roleBindingShowAll[alias];
      const rows = Object.entries(this.installedModels || {});
      return rows
        .filter(([_, m]) => showAll || !m.role_hint || m.role_hint === alias || (alias === 'embedding' && m.role_hint === 'embed'))
        .map(([id, m]) => ({
          id,
          label: `${m.file || id}${m.size_gb ? ' (' + m.size_gb + ' GB)' : ''}`,
          value: this.modelContainerPath(m),
          role_hint: m.role_hint || '',
        }))
        .filter(opt => !!opt.value);
    },

    _findModelByContainerPath(modelPath) {
      if (!modelPath) return null;
      const filename = modelPath.split('/').pop();
      for (const [id, m] of Object.entries(this.installedModels || {})) {
        if (this.modelContainerPath(m) === modelPath || m.file === filename || id === filename || id === modelPath) {
          return { id, ...m };
        }
      }
      return null;
    },

    // ── computed helpers ───────────────────────────────────────
    get runningSlots() { return this.slots.filter(s => s.running).length; },
    get totalSlots()   { return this.slots.length; },
    get hasGPU()       { return this.gpus.length > 0; },

    get hasRouterSlot() {
      return (this.slots || []).some(s => s.router_mode);
    },

    get effectiveFleetMode() {
      const raw = this.fleetMode?.mode || 'unknown';
      if (raw !== 'unknown' && raw !== 'idle') return raw;
      if ((this.roleBindings || []).length > 0) {
        return this.roleBindingsSource === 'live' ? 'router' : 'router_config';
      }
      return raw;
    },

    get isRouterPrimaryUI() {
      const mode = this.effectiveFleetMode;
      return mode === 'router' || mode === 'router_config';
    },

    get fleetModeDisplay() {
      const mode = this.effectiveFleetMode;
      if (mode === 'router_config') return 'ROUTER CONFIG';
      if (mode === 'three_slot') return '3-SLOT';
      return String(mode || 'unknown').replace('_', '-').toUpperCase();
    },

    fleetModeBannerClass() {
      const mode = this.effectiveFleetMode;
      if (mode === 'conflict') return 'fleet-mode-banner--conflict';
      if (mode === 'router' || mode === 'router_config') return 'fleet-mode-banner--router';
      if (mode === 'three_slot') return 'fleet-mode-banner--three_slot';
      return 'fleet-mode-banner--idle';
    },

    fleetBannerNote() {
      const mode = this.effectiveFleetMode;
      if (mode === 'conflict') {
        return 'Router and 3-slot containers are both running. Stop one stack before ignite.';
      }
      if (mode === 'router') {
        return 'Native llama.cpp Router Mode is active. Role bindings reflect live /v1/models.';
      }
      if (mode === 'router_config') {
        return 'Router aliases loaded from preset (fleet not detected). Refresh or ignite to verify containers.';
      }
      if (mode === 'three_slot') {
        return 'Legacy 3-slot fleet is active. Use Role Bindings only when Router Mode is enabled.';
      }
      return 'No llama.cpp fleet detected. Ignite Router Mode to load chat / utility / embedding aliases.';
    },

    roleBindingsSourceLabel() {
      if (this.roleBindingsSource === 'live') return 'live';
      if (this.roleBindingsSource === 'preset') return 'preset';
      return '';
    },

    roleBindingLoadedLabel(binding) {
      if (!binding) return '';
      if (!binding.loaded) return 'Autoload';
      if (binding.port) return `Loaded :${binding.port}`;
      return 'Loaded';
    },

    roleBindingDisplayTitle(binding) {
      if (!binding) return '(not set)';
      const model = this._findModelByContainerPath(binding.model_path);
      if (model && model.id && model.id !== binding.model_filename) {
        return model.id;
      }
      const name = binding.model_filename || binding.model_path || '';
      if (!name) return '(not set)';
      return name.replace(/\.gguf$/i, '');
    },

    roleBindingSizeLabel(binding) {
      const model = this._findModelByContainerPath(binding?.model_path);
      if (model && model.size_gb) return `${model.size_gb} GB`;
      return '';
    },

    roleBindingCtxLabel(binding) {
      if (!binding?.ctx_size) return '';
      const ctx = Number(binding.ctx_size);
      if (ctx >= 1024) return `ctx ${(ctx / 1024).toFixed(0)}K`;
      return `ctx ${ctx}`;
    },

    roleBindingApplyDisabled(alias, binding) {
      const selected = this.roleBindingSelections[alias];
      if (!selected || this.roleBindingUpdating[alias]) return true;
      return this.pathsMatch(selected, binding?.model_path);
    },

    roleBindingApplyClass(alias, binding) {
      if (this.roleBindingUpdating[alias]) return 'btn btn-primary slot-assign-btn btn-pending';
      if (this.roleBindingApplyDisabled(alias, binding)) return 'btn btn-ghost slot-assign-btn btn-no-change';
      return 'btn btn-primary slot-assign-btn btn-ready';
    },

    roleBindingCardClass(alias) {
      if (this.roleBindingJustApplied[alias]) return 'role-binding-card role-binding-card--applied';
      if (this.roleBindingUpdating[alias]) return 'role-binding-card role-binding-card--updating';
      return 'role-binding-card';
    },

    vramPct(gpu) {
      if (!gpu.total_vram_mb) return 0;
      return Math.round((gpu.used_vram_mb / gpu.total_vram_mb) * 100);
    },

    ramPct() {
      if (!this.cpu.ram_total_mb) return 0;
      return Math.round((this.cpu.ram_used_mb / this.cpu.ram_total_mb) * 100);
    },

    fmtMB(mb) {
      if (mb >= 1024) return (mb / 1024).toFixed(1) + ' GB';
      return mb + ' MB';
    },

    timeAgo() {
      if (!this.lastUpdated) return '—';
      const s = Math.round((Date.now() - this.lastUpdated.getTime()) / 1000);
      return s < 5 ? 'just now' : s + 's ago';
    },

    timeAgoShort(isoString) {
      if (!isoString) return '—';
      const date = new Date(isoString);
      const s = Math.round((Date.now() - date.getTime()) / 1000);
      if (s < 60) return s + 's';
      if (s < 3600) return Math.round(s / 60) + 'm';
      if (s < 86400) return Math.round(s / 3600) + 'h';
      return Math.round(s / 86400) + 'd';
    },

    // ── VRAM Budget Visualizer (computed) ─────────────────────
    // Slot-to-color mapping for the stacked bar
    _slotColors: {
      chat:      '#cba6f7', // accent purple
      utility:   '#89b4fa', // info blue
      embed:     '#94e2d5', // teal
      vision:    '#f9e2af', // warn yellow
      reasoning: '#f38ba8', // err pink
      _other:    '#6c7086', // muted
    },

    get vramBudget() {
      const gpu = this.gpus[0];
      if (!gpu) return { segments: [], freeGB: '0.0', freePct: 100, usedPct: 0 };

      const totalGB = gpu.total_vram_mb / 1024;
      const usedGB = gpu.used_vram_mb / 1024;
      const freeGB = gpu.free_vram_mb / 1024;

      // Build segments from router role bindings when available; in Router Mode
      // this reflects the aliases actually resident in VRAM.
      const segments = [];
      const loadedBindings = (this.roleBindings || []).filter(b => b.loaded);

      if (loadedBindings.length) {
        for (const b of loadedBindings) {
          const model = this._findModelByContainerPath(b.model_path);
          const estGB = model ? model.size_gb * 1.15 : 3.0;
          const role = b.alias || 'unknown';
          const labelName = b.model_filename || (model ? model.file || model.id : '');
          const color = this._slotColors[role] || this._slotColors._other;
          segments.push({
            label: role.charAt(0).toUpperCase() + role.slice(1) + (labelName ? ' (' + labelName + ')' : ''),
            short: role.charAt(0).toUpperCase() + role.slice(1),
            gb: estGB,
            pct: Math.min((estGB / totalGB) * 100, 100),
            color,
          });
        }
      }

      const runningSlots = loadedBindings.length ? [] : this.slots.filter(s => s.running);
      for (const s of runningSlots) {
        const mid = this.getSlotCurrentModelId(s);
        const model = mid ? this.installedModels[mid] : null;
        // Estimate: model file size * 1.15 for runtime overhead
        const estGB = model ? model.size_gb * 1.15 : 3.0;
        const role = s.role || s.id || 'unknown';
        const color = this._slotColors[role] || this._slotColors._other;

        segments.push({
          label: role.charAt(0).toUpperCase() + role.slice(1) + (model ? ' (' + (mid || '').split('/').pop() + ')' : ''),
          short: role.charAt(0).toUpperCase() + role.slice(1),
          gb: estGB,
          pct: Math.min((estGB / totalGB) * 100, 100),
          color,
        });
      }

      // Calculate totals
      const segTotal = segments.reduce((a, s) => a + s.gb, 0);
      const segPct = segments.reduce((a, s) => a + s.pct, 0);
      const freePct = Math.max(100 - segPct, 0);

      return {
        segments,
        freeGB: freeGB.toFixed(1),
        freePct: Math.max(freePct, 0),
        usedPct: Math.min(segPct, 100),
        totalGB: totalGB.toFixed(1),
      };
    },

    // ── Context slider helpers ────────────────────────────────
    /**
     * Get minimum allowed context window for a slot role.
     * These are hard limits based on llama_cpp_servers.yaml configuration.
     */
    getMinContextForRole(role) {
      const MIN_CTX = {
        chat: 65536,
        utility: 16384,
        embedding: 4096,
        vision: 8192,
        reasoning: 32768,
      };
      return MIN_CTX[role] || 2048;
    },

    getSlotCtxDefault(slot) {
      const binding = this.getRoleBinding(slot?.role);
      return Number(binding?.ctx_size || slot?._ctxSlider || this.getMinContextForRole(slot?.role));
    },

    /**
     * Estimate KV cache VRAM in GB for a given context size.
     * Formula matches context_calculator.py: ctx × 2 × n_layer × n_embd × bytes_per_token
     */
    estimateKVCacheGB(modelId, ctxSize) {
      const m = this.installedModels[modelId];
      if (!m || !m.n_layer || !m.n_embd) {
        // Rough fallback: ~0.5 GB per 8K context per 10GB model
        const sizeGB = m ? m.size_gb : 5;
        return (ctxSize / 8192) * (sizeGB / 10) * 0.5;
      }
      const bytesPerToken = 2; // FP16 KV cache
      const totalBytes = ctxSize * 2 * m.n_layer * m.n_embd * bytesPerToken;
      return totalBytes / (1024 ** 3);
    },

    /**
     * Color indicator for context VRAM impact.
     * green = comfortable, yellow = tight, red = won't fit
     */
    ctxVramColor(modelId, ctxSize) {
      const gpu = this.gpus[0];
      if (!gpu) return 'var(--muted)';
      const m = this.installedModels[modelId];
      const weightsGB = m ? m.size_gb * 1.15 : 5;
      const kvGB = this.estimateKVCacheGB(modelId, ctxSize);
      const totalNeeded = weightsGB + kvGB;
      const totalAvail = gpu.total_vram_mb / 1024;
      const pct = (totalNeeded / totalAvail) * 100;
      if (pct > 95) return 'var(--err)';   // won't fit
      if (pct > 80) return 'var(--warn)';  // tight
      return 'var(--ok)';                   // comfortable
    },

    // ── Router Mode helpers ────────────────────────────────────────
    // routerModels: { [slot_id]: { models: [], current_default: '', loading, error, setMsg } }
    routerModels: {},

    async loadRouterModels(slotId) {
      if (!this.routerModels[slotId]) {
        this.routerModels = { ...this.routerModels, [slotId]: { models: [], current_default: '', loading: true, error: '', setMsg: '' } };
      }
      this.routerModels[slotId].loading = true;
      this.routerModels[slotId].error   = '';
      try {
        const d = await _a0ApiCall(ENDPOINTS.routerModels, { slot_id: slotId });
        if (d.ok) {
          this.routerModels[slotId].models          = d.models || [];
          this.routerModels[slotId].current_default = d.current_default || '';
        } else {
          this.routerModels[slotId].error = d.error || 'Failed to load models';
        }
      } catch (e) {
        this.routerModels[slotId].error = 'Connection error';
      } finally {
        this.routerModels[slotId].loading = false;
      }
    },

    async setRouterDefault(slotId, alias) {
      if (!alias) return;
      const state = this.routerModels[slotId];
      if (state) state.setMsg = '';
      try {
        const d = await _a0ApiCall(ENDPOINTS.setRouterDefault, { slot_id: slotId, model_alias: alias });
        if (d.ok) {
          if (state) {
            state.current_default = alias;
            state.setMsg = `✓ Default set to '${alias}' — restart slot to apply`;
            state.models.forEach(m => { m.is_default = (m.alias === alias); });
          }
          const slot = this.slots.find(s => s.id === slotId);
          if (slot) slot.router_default_model = alias;
        } else {
          if (state) state.setMsg = `✗ ${d.error || 'Failed'}`;
        }
      } catch (e) {
        if (state) state.setMsg = '✗ Connection error';
      }
    },
  };
}

// Export for use in dashboard.html
window.createDashboardStore = createDashboardStore;
