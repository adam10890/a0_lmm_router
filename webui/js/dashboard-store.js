/**
 * dashboard-store.js — Alpine.js store for the LMM Router dashboard.
 *
 * Polls /plugins/a0_lmm_router/lmm_compute_stats every N seconds and
 * exposes reactive data for GPU, CPU, slots, and model recommendations.
 */

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
  loadModel:    `${API_BASE}/load_model`,
  jobStatus:    `${API_BASE}/job_status`,
  control:      `${API_BASE}/llamacpp_control`,
  status:       `${API_BASE}/llamacpp_status`,
  ignite:       `${API_BASE}/lmm_fleet_ignite`,
  hostIgnite:   `${API_BASE}/lmm_host_ignite`,
  hardwareScan: `${API_BASE}/lmm_hardware_recommend`,
  slotRecs:     `${API_BASE}/lmm_slot_recommendations`,
};

function createDashboardStore() {
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
      ]);
      // Slot recommendations needs slots + installed models loaded.
      await this._fetchSlotRecommendations();
      this.loading = false;
    },

    async _fetchInstalledModels() {
      try {
        const r = await fetch(ENDPOINTS.listModels, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          credentials: 'same-origin',
          body: JSON.stringify({}),
        });
        const d = await r.json();
        if (d.ok && d.models) {
          this.installedModels = d.models;
        }
      } catch (_) { /* silent */ }
    },

    async _fetchStatsSummary() {
      try {
        const r = await fetch(ENDPOINTS.statsSummary, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          credentials: 'same-origin',
          body: JSON.stringify({ window: this.statsWindow }),
        });
        const d = await r.json();
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
        const r = await fetch(ENDPOINTS.computeStats, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          credentials: 'same-origin',
          body: JSON.stringify({}),
        });
        const d = await r.json();
        if (d.ok) {
          this.gpus = d.gpus || [];
          this.cpu = d.cpu || this.cpu;
          this.slots = d.slots || [];
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
        const r = await fetch(ENDPOINTS.recommend, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          credentials: 'same-origin',
          body: JSON.stringify({}),
        });
        const d = await r.json();
        if (d.ok) this.recommendations = d.recommendations || [];
      } catch (_) { /* silent */ }
    },

    // ── Unified slot recommendations (role-aware, tier-aware) ──
    // Pulls: live tier + installed models + per-slot suggestions in one call.
    // Driven by the backend's lmm_slot_recommendations endpoint which
    // internally reads compute_monitor + tier_catalog + fleet_models.
    async _fetchSlotRecommendations() {
      try {
        const r = await fetch(ENDPOINTS.slotRecs, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          credentials: 'same-origin',
          body: JSON.stringify({}),
        });
        const d = await r.json();
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
        const r = await fetch(ENDPOINTS.hardwareScan, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          credentials: 'same-origin',
          body: JSON.stringify({}),
        });
        const d = await r.json();
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
      this.igniteMessage = `Calling host helper: ${action}…`;
      this.igniteHostHint = '';
      try {
        const r = await fetch(ENDPOINTS.hostIgnite, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          credentials: 'same-origin',
          body: JSON.stringify({ action }),
        });
        const d = await r.json();
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
        const r = await fetch(ENDPOINTS.ignite, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          credentials: 'same-origin',
          body: JSON.stringify({}),
        });
        const d = await r.json();
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
      try {
        const r = await fetch(ENDPOINTS.control, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          credentials: 'same-origin',
          body: JSON.stringify({ data: { operation: op, server: serverId } }),
        });
        await r.json();
        await this._fetchStats();
      } catch (_) { /* silent */ }
    },

    async installModel(rec) {
      const key = rec.filename;
      this.installStatus[key] = { status: 'downloading', percent: 0, job_id: null };
      try {
        const r = await fetch(ENDPOINTS.install, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          credentials: 'same-origin',
          body: JSON.stringify({ repo_id: rec.repo_id, filename: rec.filename, role: rec.role }),
        });
        const d = await r.json();
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
        const r = await fetch(ENDPOINTS.install, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          credentials: 'same-origin',
          body: JSON.stringify({ repo_id: repo, filename: file, role }),
        });
        const d = await r.json();
        if (d.ok && d.job_id) {
          this.installForm.jobId = d.job_id;
          // Poll the job
          const timer = setInterval(async () => {
            try {
              const jr = await fetch(ENDPOINTS.jobStatus, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                credentials: 'same-origin',
                body: JSON.stringify({ job_id: d.job_id }),
              });
              const jd = await jr.json();
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
          const r = await fetch(ENDPOINTS.jobStatus, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            credentials: 'same-origin',
            body: JSON.stringify({ job_id: jobId }),
          });
          const d = await r.json();
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
      try {
        const r = await fetch(ENDPOINTS.assignModel, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          credentials: 'same-origin',
          body: JSON.stringify({ slot: slotId, model_id: modelId, apply_now: true }),
        });
        const d = await r.json();
        if (d.ok) {
          // Wait a bit for container restart, then refresh
          setTimeout(() => this._fetchStats(), 5000);
          return { ok: true, restarted: d.restarted };
        } else {
          return { ok: false, error: d.error || 'Assignment failed' };
        }
      } catch (e) {
        return { ok: false, error: e.message || 'Connection failed' };
      } finally {
        // Keep spinner for a few seconds to show it's restarting
        setTimeout(() => { this.swapInProgress[slotId] = false; }, 3000);
      }
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

    canLoadModel(modelId) {
      // Check VRAM constraints - don't load if this model + existing > free VRAM
      const model = this.installedModels[modelId];
      if (!model) return { ok: false, reason: 'Model not found' };

      // Get currently loaded (running) model sizes
      const runningSlots = this.slots.filter(s => s.running);
      let usedVramGB = 0;
      for (const s of runningSlots) {
        const mid = this.getSlotCurrentModelId(s);
        if (mid && this.installedModels[mid]) {
          usedVramGB += this.installedModels[mid].size_gb;
        }
      }

      // Check against first GPU (assuming single GPU for now)
      const gpu = this.gpus[0];
      if (!gpu) return { ok: true }; // No GPU, can't check

      const freeVramGB = gpu.free_vram_mb / 1024;
      const safetyMargin = 2; // GB
      const required = model.size_gb + safetyMargin;

      if (usedVramGB + required > (gpu.total_vram_mb / 1024)) {
        return {
          ok: false,
          reason: `Not enough VRAM. Model needs ~${model.size_gb} GB, but only ~${freeVramGB.toFixed(1)} GB free with safety margin.`,
          usedVramGB,
          freeVramGB,
          required,
        };
      }
      return { ok: true };
    },

    // ── computed helpers ───────────────────────────────────────
    get runningSlots() { return this.slots.filter(s => s.running).length; },
    get totalSlots()   { return this.slots.length; },
    get hasGPU()       { return this.gpus.length > 0; },

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

      // Build segments from running slots
      const segments = [];
      const runningSlots = this.slots.filter(s => s.running);

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

    // ── Load model (combined flow) ────────────────────────────
    async loadModelToSlot(slotId, modelId, ctxSize) {
      this.swapInProgress[slotId] = true;
      try {
        const body = { slot: slotId, model_id: modelId };
        if (ctxSize) body.ctx_size = parseInt(ctxSize);
        const r = await fetch(ENDPOINTS.loadModel, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          credentials: 'same-origin',
          body: JSON.stringify(body),
        });
        const d = await r.json();
        if (d.ok) {
          setTimeout(() => this._fetchStats(), 3000);
          return { ok: true, context: d.context };
        } else {
          return { ok: false, error: d.error || 'Load failed' };
        }
      } catch (e) {
        return { ok: false, error: e.message || 'Connection failed' };
      } finally {
        setTimeout(() => { this.swapInProgress[slotId] = false; }, 3000);
      }


    // ── Context slider helpers ────────────────────────────────
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
  };
}

// Export for use in dashboard.html
window.createDashboardStore = createDashboardStore;
