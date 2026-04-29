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
  recommend:    `${API_BASE}/lmm_model_recommend`,
  install:      `${API_BASE}/lmm_model_install`,
  control:      `${API_BASE}/llamacpp_control`,
  status:       `${API_BASE}/llamacpp_status`,
  ignite:       `${API_BASE}/lmm_fleet_ignite`,
  hostIgnite:   `${API_BASE}/lmm_host_ignite`,
};

function createDashboardStore() {
  return {
    // ── state ──────────────────────────────────────────────────
    gpus: [],
    cpu: { load_pct: 0, ram_total_mb: 0, ram_used_mb: 0, ram_free_mb: 0 },
    slots: [],
    recommendations: [],
    installStatus: {},        // keyed by filename
    pollTimer: null,
    loading: true,
    error: '',
    lastUpdated: null,

    // Fleet ignition state
    igniteState: 'idle',      // idle | pending | ok | needs_host | error
    igniteMessage: '',
    igniteHostHint: '',

    // ── lifecycle ──────────────────────────────────────────────
    init() {
      this.refresh();
      this.pollTimer = setInterval(() => this.refresh(), POLL_INTERVAL_MS);
    },

    cleanup() {
      if (this.pollTimer) clearInterval(this.pollTimer);
    },

    // ── data fetching ──────────────────────────────────────────
    async refresh() {
      await Promise.all([
        this._fetchStats(),
        this._fetchRecommendations(),
      ]);
      this.loading = false;
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
      this.installStatus[key] = 'downloading';
      try {
        const r = await fetch(ENDPOINTS.install, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          credentials: 'same-origin',
          body: JSON.stringify({ repo_id: rec.repo_id, filename: rec.filename }),
        });
        const d = await r.json();
        this.installStatus[key] = d.ok ? 'done' : 'error';
      } catch (_) {
        this.installStatus[key] = 'error';
      }
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
  };
}

// Export for use in dashboard.html
window.createDashboardStore = createDashboardStore;
