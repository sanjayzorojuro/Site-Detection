/**
 * Alerts Page — JavaScript Controller
 * Fetches violation records from the database, renders table, handles filtering/pagination.
 */
(function () {
    'use strict';

    const violationsBody = document.getElementById('violations-body');
    const filterType = document.getElementById('filter-type');
    const filterVideo = document.getElementById('filter-video');
    const refreshBtn = document.getElementById('refresh-btn');
    const clearBtn = document.getElementById('clear-btn');
    const prevBtn = document.getElementById('prev-btn');
    const nextBtn = document.getElementById('next-btn');
    const pageInfo = document.getElementById('page-info');
    const totalCount = document.getElementById('total-count');
    const dangerCount = document.getElementById('danger-count');
    const warningCount = document.getElementById('warning-count');

    const PAGE_SIZE = 30;
    let currentPage = 0;
    let totalViolations = 0;

    // ── Load Processed Videos for Filter ──
    async function loadVideoFilter() {
        try {
            const resp = await fetch('/api/processed-videos?limit=50');
            const data = await resp.json();
            const videos = data.videos || [];
            filterVideo.innerHTML = '<option value="">All Videos</option>';
            const seen = new Set();
            videos.forEach(v => {
                if (!seen.has(v.video_name)) {
                    seen.add(v.video_name);
                    const opt = document.createElement('option');
                    opt.value = v.video_name;
                    opt.textContent = v.video_name;
                    filterVideo.appendChild(opt);
                }
            });
        } catch (e) {
            console.error('Failed to load video list:', e);
        }
    }

    // ── Fetch & Render Violations ──
    async function loadViolations() {
        const type = filterType.value;
        const video = filterVideo.value;
        const skip = currentPage * PAGE_SIZE;

        let url = `/api/violations?limit=${PAGE_SIZE}&skip=${skip}`;
        if (type) url += `&violation_type=${encodeURIComponent(type)}`;
        if (video) url += `&video_name=${encodeURIComponent(video)}`;

        try {
            const resp = await fetch(url);
            const data = await resp.json();
            const violations = data.violations || [];
            totalViolations = data.total || 0;

            // Update summary
            totalCount.textContent = totalViolations;
            let dc = 0, wc = 0;
            violations.forEach(v => {
                if (v.risk_level === 'danger') dc++;
                else if (v.risk_level === 'warning') wc++;
            });
            dangerCount.textContent = dc;
            warningCount.textContent = wc;

            // Render table
            if (violations.length === 0) {
                violationsBody.innerHTML = '<tr class="empty-row"><td colspan="7">No violations found.</td></tr>';
            } else {
                violationsBody.innerHTML = violations.map(v => {
                    const ts = v.timestamp ? new Date(v.timestamp).toLocaleString() : '--';
                    const riskClass = v.risk_level || 'warning';
                    return `<tr>
                        <td><span class="risk-badge ${riskClass}">${riskClass}</span></td>
                        <td style="color:var(--text-primary);font-weight:600">${escapeHtml(v.violation_type || '')}</td>
                        <td>${escapeHtml(v.video_name || '')}</td>
                        <td>#${v.person_id != null ? v.person_id : '--'}</td>
                        <td style="font-family:'JetBrains Mono',monospace;font-size:12px">${v.frame_number || '--'}</td>
                        <td style="font-size:12px;color:var(--text-muted)">${ts}</td>
                        <td style="font-size:12px">${escapeHtml(v.details || '')}</td>
                    </tr>`;
                }).join('');
            }

            // Pagination
            const totalPages = Math.max(1, Math.ceil(totalViolations / PAGE_SIZE));
            pageInfo.textContent = `Page ${currentPage + 1} of ${totalPages}`;
            prevBtn.disabled = currentPage === 0;
            nextBtn.disabled = currentPage >= totalPages - 1;

        } catch (e) {
            violationsBody.innerHTML = '<tr class="empty-row"><td colspan="7">Failed to load violations.</td></tr>';
            console.error('Load error:', e);
        }
    }

    function escapeHtml(str) {
        const div = document.createElement('div');
        div.textContent = str;
        return div.innerHTML;
    }

    // ── Event Handlers ──
    filterType.addEventListener('change', () => { currentPage = 0; loadViolations(); });
    filterVideo.addEventListener('change', () => { currentPage = 0; loadViolations(); });
    refreshBtn.addEventListener('click', () => loadViolations());
    prevBtn.addEventListener('click', () => { if (currentPage > 0) { currentPage--; loadViolations(); } });
    nextBtn.addEventListener('click', () => { currentPage++; loadViolations(); });

    clearBtn.addEventListener('click', async () => {
        if (!confirm('Clear ALL violation records? This cannot be undone.')) return;
        try {
            await fetch('/api/violations', { method: 'DELETE' });
            currentPage = 0;
            loadViolations();
        } catch (e) {
            alert('Failed to clear: ' + e.message);
        }
    });

    // ── Init ──
    loadVideoFilter();
    loadViolations();

    // Auto-refresh every 10 seconds
    setInterval(loadViolations, 10000);
})();
