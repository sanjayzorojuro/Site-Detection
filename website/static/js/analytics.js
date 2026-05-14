/**
 * Analytics Page — JavaScript Controller
 * Fetches analytics data and renders charts + processed videos table.
 */
(function () {
    'use strict';

    // Violation type → color mapping
    const TYPE_COLORS = {
        'NO HELMET': 'red',
        'NO VEST': 'amber',
        'FALL DETECTED': 'yellow',
        'NEAR EDGE / HEIGHT DANGER': 'purple',
        'FALLING OBJECT!': 'red',
        'MOTIONLESS - CHECK PERSON': 'amber',
        'PERSON IN DANGER': 'orange',
    };

    function getBarColor(type) {
        for (const [key, color] of Object.entries(TYPE_COLORS)) {
            if (type.toUpperCase().includes(key.split(' ')[0])) return color;
        }
        return 'yellow';
    }

    // ── Load Analytics Summary ──
    async function loadAnalytics() {
        try {
            const resp = await fetch('/api/analytics');
            const data = await resp.json();

            // Summary cards
            document.getElementById('a-total').textContent = data.total_violations || 0;
            document.getElementById('a-videos').textContent = data.processed_videos || 0;

            const byRisk = data.by_risk || {};
            document.getElementById('a-danger').textContent = byRisk.danger || 0;
            document.getElementById('a-warning').textContent = byRisk.warning || 0;

            // Bar chart — violations by type
            renderBarChart(data.by_type || {});

            // Donut chart — risk distribution
            renderDonutChart(byRisk);

        } catch (e) {
            console.error('Analytics load error:', e);
        }
    }

    // ── Bar Chart ──
    function renderBarChart(byType) {
        const container = document.getElementById('type-chart');
        const entries = Object.entries(byType).sort((a, b) => b[1] - a[1]);

        if (entries.length === 0) {
            container.innerHTML = '<div style="text-align:center;padding:40px;color:var(--text-muted);font-size:13px">No violation data yet. Process a video to see analytics.</div>';
            return;
        }

        const maxVal = entries[0][1] || 1;
        container.innerHTML = entries.map(([type, count]) => {
            const pct = (count / maxVal) * 100;
            const color = getBarColor(type);
            return `<div class="bar-row">
                <span class="bar-label">${escapeHtml(type)}</span>
                <div class="bar-track"><div class="bar-fill ${color}" style="width:${pct}%"></div></div>
                <span class="bar-count">${count}</span>
            </div>`;
        }).join('');
    }

    // ── Donut Chart ──
    function renderDonutChart(byRisk) {
        const danger = byRisk.danger || 0;
        const warning = byRisk.warning || 0;
        const safe = byRisk.safe || 0;
        const total = danger + warning + safe;

        const legend = document.getElementById('donut-legend');

        if (total === 0) {
            document.getElementById('seg-danger').setAttribute('stroke-dasharray', '0 100');
            document.getElementById('seg-warning').setAttribute('stroke-dasharray', '0 100');
            document.getElementById('seg-safe').setAttribute('stroke-dasharray', '0 100');
            legend.innerHTML = '<div style="color:var(--text-muted);font-size:13px">No data</div>';
            return;
        }

        const dangerPct = (danger / total) * 100;
        const warningPct = (warning / total) * 100;
        const safePct = (safe / total) * 100;

        // SVG donut segments
        const segDanger = document.getElementById('seg-danger');
        const segWarning = document.getElementById('seg-warning');
        const segSafe = document.getElementById('seg-safe');

        segDanger.setAttribute('stroke-dasharray', `${dangerPct} ${100 - dangerPct}`);
        segDanger.setAttribute('stroke-dashoffset', '25');

        segWarning.setAttribute('stroke-dasharray', `${warningPct} ${100 - warningPct}`);
        segWarning.setAttribute('stroke-dashoffset', String(25 - dangerPct));

        segSafe.setAttribute('stroke-dasharray', `${safePct} ${100 - safePct}`);
        segSafe.setAttribute('stroke-dashoffset', String(25 - dangerPct - warningPct));

        legend.innerHTML = `
            <div class="legend-item"><span class="legend-dot danger"></span>Danger<span class="legend-val">${danger} (${dangerPct.toFixed(0)}%)</span></div>
            <div class="legend-item"><span class="legend-dot warning"></span>Warning<span class="legend-val">${warning} (${warningPct.toFixed(0)}%)</span></div>
            <div class="legend-item"><span class="legend-dot safe"></span>Safe<span class="legend-val">${safe} (${safePct.toFixed(0)}%)</span></div>
        `;
    }

    // ── Processed Videos Table ──
    async function loadVideos() {
        try {
            const resp = await fetch('/api/processed-videos?limit=50');
            const data = await resp.json();
            const videos = data.videos || [];
            const body = document.getElementById('videos-body');

            if (videos.length === 0) {
                body.innerHTML = '<tr><td colspan="6" style="text-align:center;padding:30px;color:var(--text-muted)">No videos processed yet.</td></tr>';
                return;
            }

            body.innerHTML = videos.map(v => {
                const date = v.date_processed ? new Date(v.date_processed).toLocaleString() : '--';
                const riskClass = v.risk_level || 'low';
                const dur = v.duration_seconds ? `${Math.round(v.duration_seconds)}s` : '--';
                return `<tr>
                    <td style="color:var(--text-primary);font-weight:600">${escapeHtml(v.video_name || '')}</td>
                    <td style="font-family:'JetBrains Mono',monospace;font-size:12px">${v.total_frames || 0}</td>
                    <td>${v.total_violations || 0}</td>
                    <td><span class="risk-tag ${riskClass}">${riskClass}</span></td>
                    <td>${dur}</td>
                    <td style="font-size:12px;color:var(--text-muted)">${date}</td>
                </tr>`;
            }).join('');

        } catch (e) {
            console.error('Videos load error:', e);
        }
    }

    function escapeHtml(str) {
        const div = document.createElement('div');
        div.textContent = str;
        return div.innerHTML;
    }

    // ── Init ──
    loadAnalytics();
    loadVideos();

    // Auto-refresh every 15 seconds
    setInterval(() => { loadAnalytics(); loadVideos(); }, 15000);
})();
