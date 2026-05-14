/**
 * Live Monitor Page — JavaScript Controller
 * Handles video upload, webcam, WebSocket connections, real-time UI updates,
 * and the real-time Risk Graph (Canvas-based line chart).
 */
(function () {
    'use strict';

    // ── DOM Elements ──
    const fileInput = document.getElementById('video-file-input');
    const uploadBtn = document.getElementById('upload-btn');
    const webcamBtn = document.getElementById('webcam-btn');
    const demoBtn = document.getElementById('demo-btn');
    const stopBtn = document.getElementById('stop-btn');
    const videoFeed = document.getElementById('video-feed');
    const videoPlaceholder = document.getElementById('video-placeholder');
    const progressFill = document.getElementById('progress-fill');
    const alertsFeed = document.getElementById('alerts-feed');
    const alertEmpty = document.getElementById('alert-empty');
    const alertCount = document.getElementById('alert-count');
    const statusDot = document.getElementById('status-dot');
    const statusText = document.getElementById('status-text');
    const fpsBadge = document.getElementById('fps-badge');
    const sourceBadge = document.getElementById('source-badge');

    // Stat elements
    const statWorkers = document.getElementById('stat-workers');
    const statHelmet = document.getElementById('stat-helmet');
    const statVest = document.getElementById('stat-vest');
    const statRisk = document.getElementById('stat-risk');
    const riskBar = document.getElementById('risk-bar');
    const statDanger = document.getElementById('stat-danger');
    const statWarning = document.getElementById('stat-warning');
    const statSafe = document.getElementById('stat-safe');
    const statMachine = document.getElementById('stat-machine');
    const statFrame = document.getElementById('stat-frame');
    const statElapsed = document.getElementById('stat-elapsed');

    let ws = null;
    let statsInterval = null;
    let alertItems = [];

    // ══════════════════════════════════════════════════
    // ── REAL-TIME RISK GRAPH ──
    // ══════════════════════════════════════════════════
    const riskCanvas = document.getElementById('risk-graph-canvas');
    const riskCtx = riskCanvas ? riskCanvas.getContext('2d') : null;
    const GRAPH_HISTORY = 120; // data points (120 = ~2 mins at 1/s)
    let riskHistory = [];

    function initRiskGraph() {
        if (!riskCanvas) return;
        // Set canvas resolution
        const rect = riskCanvas.getBoundingClientRect();
        riskCanvas.width = rect.width * 2;
        riskCanvas.height = rect.height * 2;
        riskCtx.scale(2, 2);
        drawRiskGraph();
    }

    function pushRiskData(score) {
        riskHistory.push({ value: Math.min(100, Math.max(0, score)), time: Date.now() });
        if (riskHistory.length > GRAPH_HISTORY) riskHistory.shift();
        drawRiskGraph();
    }

    function drawRiskGraph() {
        if (!riskCtx) return;
        const W = riskCanvas.width / 2;
        const H = riskCanvas.height / 2;
        const pad = { top: 8, right: 12, bottom: 24, left: 36 };
        const gW = W - pad.left - pad.right;
        const gH = H - pad.top - pad.bottom;

        riskCtx.clearRect(0, 0, W, H);

        // Background zones
        const zones = [
            { from: 0, to: 30, color: 'rgba(34,197,94,0.06)' },
            { from: 30, to: 60, color: 'rgba(245,158,11,0.06)' },
            { from: 60, to: 100, color: 'rgba(239,68,68,0.06)' },
        ];
        zones.forEach(z => {
            const y1 = pad.top + gH * (1 - z.to / 100);
            const y2 = pad.top + gH * (1 - z.from / 100);
            riskCtx.fillStyle = z.color;
            riskCtx.fillRect(pad.left, y1, gW, y2 - y1);
        });

        // Zone threshold lines
        [30, 60].forEach(thresh => {
            const y = pad.top + gH * (1 - thresh / 100);
            riskCtx.strokeStyle = thresh === 60 ? 'rgba(239,68,68,0.2)' : 'rgba(245,158,11,0.2)';
            riskCtx.lineWidth = 1;
            riskCtx.setLineDash([4, 4]);
            riskCtx.beginPath();
            riskCtx.moveTo(pad.left, y);
            riskCtx.lineTo(pad.left + gW, y);
            riskCtx.stroke();
            riskCtx.setLineDash([]);
        });

        // Y-axis labels
        riskCtx.fillStyle = 'rgba(255,255,255,0.25)';
        riskCtx.font = '9px Rajdhani, sans-serif';
        riskCtx.textAlign = 'right';
        [0, 30, 60, 100].forEach(v => {
            const y = pad.top + gH * (1 - v / 100);
            riskCtx.fillText(v, pad.left - 6, y + 3);
        });

        // X-axis label
        riskCtx.fillStyle = 'rgba(255,255,255,0.2)';
        riskCtx.textAlign = 'center';
        riskCtx.font = '9px Rajdhani, sans-serif';
        riskCtx.fillText('Time →', pad.left + gW / 2, H - 4);

        if (riskHistory.length < 2) {
            // Empty state
            riskCtx.fillStyle = 'rgba(255,255,255,0.15)';
            riskCtx.textAlign = 'center';
            riskCtx.font = '11px Rajdhani, sans-serif';
            riskCtx.fillText('Waiting for data...', W / 2, H / 2);
            return;
        }

        // Draw line
        const dataLen = riskHistory.length;
        const stepX = gW / (GRAPH_HISTORY - 1);

        // Create gradient for the line
        const grad = riskCtx.createLinearGradient(0, pad.top, 0, pad.top + gH);
        grad.addColorStop(0, '#ef4444');
        grad.addColorStop(0.4, '#f59e0b');
        grad.addColorStop(1, '#22c55e');

        // Filled area under curve
        riskCtx.beginPath();
        const startIdx = GRAPH_HISTORY - dataLen;
        for (let i = 0; i < dataLen; i++) {
            const x = pad.left + (startIdx + i) * stepX;
            const y = pad.top + gH * (1 - riskHistory[i].value / 100);
            if (i === 0) riskCtx.moveTo(x, y);
            else riskCtx.lineTo(x, y);
        }
        // Close to bottom for fill
        const lastX = pad.left + (startIdx + dataLen - 1) * stepX;
        const firstX = pad.left + startIdx * stepX;
        riskCtx.lineTo(lastX, pad.top + gH);
        riskCtx.lineTo(firstX, pad.top + gH);
        riskCtx.closePath();

        const fillGrad = riskCtx.createLinearGradient(0, pad.top, 0, pad.top + gH);
        fillGrad.addColorStop(0, 'rgba(239,68,68,0.12)');
        fillGrad.addColorStop(0.4, 'rgba(245,158,11,0.08)');
        fillGrad.addColorStop(1, 'rgba(34,197,94,0.04)');
        riskCtx.fillStyle = fillGrad;
        riskCtx.fill();

        // Stroke line
        riskCtx.beginPath();
        for (let i = 0; i < dataLen; i++) {
            const x = pad.left + (startIdx + i) * stepX;
            const y = pad.top + gH * (1 - riskHistory[i].value / 100);
            if (i === 0) riskCtx.moveTo(x, y);
            else riskCtx.lineTo(x, y);
        }
        riskCtx.strokeStyle = grad;
        riskCtx.lineWidth = 2;
        riskCtx.lineJoin = 'round';
        riskCtx.lineCap = 'round';
        riskCtx.stroke();

        // Current value dot (glow)
        const lastVal = riskHistory[dataLen - 1].value;
        const dotX = lastX;
        const dotY = pad.top + gH * (1 - lastVal / 100);
        const dotColor = lastVal > 60 ? '#ef4444' : lastVal > 30 ? '#f59e0b' : '#22c55e';

        // Glow
        riskCtx.beginPath();
        riskCtx.arc(dotX, dotY, 6, 0, Math.PI * 2);
        riskCtx.fillStyle = dotColor + '33';
        riskCtx.fill();

        // Solid dot
        riskCtx.beginPath();
        riskCtx.arc(dotX, dotY, 3, 0, Math.PI * 2);
        riskCtx.fillStyle = dotColor;
        riskCtx.fill();

        // Value label
        riskCtx.fillStyle = '#fff';
        riskCtx.font = 'bold 10px Orbitron, sans-serif';
        riskCtx.textAlign = 'left';
        riskCtx.fillText(Math.round(lastVal), dotX + 8, dotY + 4);
    }

    // Init graph on load
    initRiskGraph();
    window.addEventListener('resize', initRiskGraph);

    // ══════════════════════════════════════════════════
    // ── WEBSOCKET ──
    // ══════════════════════════════════════════════════
    function connectWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/detections`;

        ws = new WebSocket(wsUrl);

        ws.onopen = () => {
            console.log('WebSocket connected');
        };

        ws.onmessage = (event) => {
            try {
                const msg = JSON.parse(event.data);
                if (msg.type === 'stats') {
                    updateStats(msg.data);
                } else if (msg.type === 'alert') {
                    addAlert(msg.data);
                }
            } catch (e) {
                console.error('WS parse error:', e);
            }
        };

        ws.onclose = () => {
            console.log('WebSocket disconnected, reconnecting...');
            setTimeout(connectWebSocket, 3000);
        };

        ws.onerror = (e) => {
            console.error('WebSocket error:', e);
        };
    }

    // ── Update Stats UI ──
    function updateStats(data) {
        if (!data) return;

        // Status indicator
        if (data.is_processing) {
            statusDot.className = 'status-dot online';
            statusText.textContent = 'Processing';
            stopBtn.disabled = false;
        } else {
            statusDot.className = 'status-dot offline';
            statusText.textContent = 'Idle';
            stopBtn.disabled = true;
        }

        // Stats
        const stats = data.stats || {};
        statWorkers.textContent = stats.worker_count || 0;

        const hc = stats.helmet_compliance || 100;
        statHelmet.textContent = hc + '%';
        statHelmet.className = 'stat-value ' + (hc >= 80 ? 'green' : hc >= 50 ? 'amber' : 'red');

        const vc = stats.vest_compliance || 100;
        statVest.textContent = vc + '%';
        statVest.className = 'stat-value ' + (vc >= 80 ? 'green' : vc >= 50 ? 'amber' : 'red');

        const risk = data.risk_score || 0;
        statRisk.textContent = Math.round(risk);
        riskBar.style.width = risk + '%';
        if (risk > 60) {
            riskBar.style.background = '#ef4444';
            statRisk.className = 'stat-value red';
        } else if (risk > 30) {
            riskBar.style.background = '#f59e0b';
            statRisk.className = 'stat-value amber';
        } else {
            riskBar.style.background = '#22c55e';
            statRisk.className = 'stat-value green';
        }

        // Push risk score to graph (only when actively processing)
        if (data.is_processing) {
            pushRiskData(risk);
        }

        statDanger.textContent = stats.danger_count || 0;
        statWarning.textContent = stats.warning_count || 0;
        statSafe.textContent = stats.safe_count || 0;
        if (statMachine) statMachine.textContent = stats.machine_danger_count || 0;

        // Frame progress
        const fc = data.frame_count || 0;
        const tf = data.total_frames || 0;
        statFrame.textContent = tf > 0 ? `${fc} / ${tf}` : fc;
        if (tf > 0) {
            progressFill.style.width = ((fc / tf) * 100) + '%';
        }

        // Elapsed time
        const elapsed = data.elapsed || 0;
        const mins = Math.floor(elapsed / 60);
        const secs = Math.floor(elapsed % 60);
        statElapsed.textContent = `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;

        // FPS badge
        fpsBadge.textContent = (data.fps || 0) + ' FPS';
        sourceBadge.textContent = data.video_name || data.source_type || '--';

        // Alert count
        alertCount.textContent = (data.alert_count || 0) + ' alerts';
    }

    // ── Add Alert to Feed ──
    function addAlert(alert) {
        if (alertEmpty) alertEmpty.style.display = 'none';

        const item = document.createElement('div');
        item.className = 'alert-item';

        const dotClass = alert.risk_level === 'danger' ? 'danger' : 'warning';
        const timestamp = alert.timestamp ? new Date(alert.timestamp).toLocaleTimeString() : '--';

        item.innerHTML = `
            <div class="alert-dot ${dotClass}"></div>
            <span class="alert-type">${escapeHtml(alert.type)}</span>
            <span class="alert-person">Person #${alert.person_id || 0}</span>
            <span class="alert-time">${timestamp}</span>
        `;

        // Insert at top
        alertsFeed.insertBefore(item, alertsFeed.firstChild);

        // Limit displayed alerts
        alertItems.push(item);
        if (alertItems.length > 100) {
            const old = alertItems.shift();
            if (old.parentNode) old.parentNode.removeChild(old);
        }
    }

    function escapeHtml(str) {
        const div = document.createElement('div');
        div.textContent = str;
        return div.innerHTML;
    }

    // ── Start Video Feed Display ──
    function startFeed() {
        videoPlaceholder.style.display = 'none';
        videoFeed.style.display = 'block';
        videoFeed.src = '/api/video-feed?' + Date.now();
        stopBtn.disabled = false;
        // Reset risk graph for new session
        riskHistory = [];
        if (riskCtx) drawRiskGraph();
        // Clear old alerts
        alertItems = [];
        if (alertsFeed) alertsFeed.innerHTML = '';
        if (alertEmpty) alertEmpty.style.display = '';
        if (alertCount) alertCount.textContent = '0 alerts';
    }

    // ── Stop Feed ──
    function stopFeed() {
        // Stop the MJPEG stream by clearing img src
        videoFeed.src = '';
        videoFeed.style.display = 'none';
        videoPlaceholder.style.display = '';
        stopBtn.disabled = true;
        progressFill.style.width = '0%';

        // Reset all stats to idle state
        statusDot.className = 'status-dot offline';
        statusText.textContent = 'Idle';
        fpsBadge.textContent = '-- FPS';
        sourceBadge.textContent = '--';
        statWorkers.textContent = '0';
        statHelmet.textContent = '100%';
        statHelmet.className = 'stat-value green';
        statVest.textContent = '100%';
        statVest.className = 'stat-value green';
        statRisk.textContent = '0';
        statRisk.className = 'stat-value green';
        riskBar.style.width = '0%';
        riskBar.style.background = '#22c55e';
        statDanger.textContent = '0';
        statWarning.textContent = '0';
        statSafe.textContent = '0';
        if (statMachine) statMachine.textContent = '0';
        statFrame.textContent = '0 / 0';
        statElapsed.textContent = '00:00';

        // Stop the risk graph
        riskHistory = [];
        if (riskCtx) drawRiskGraph();
    }

    // ── Upload Video ──
    fileInput.addEventListener('change', async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        const formData = new FormData();
        formData.append('file', file);

        try {
            const resp = await fetch('/api/upload-video', { method: 'POST', body: formData });
            const data = await resp.json();
            if (resp.ok) {
                startFeed();
            } else {
                alert(data.error || 'Upload failed');
            }
        } catch (err) {
            alert('Upload error: ' + err.message);
        }
        fileInput.value = '';
    });

    // ── Webcam ──
    webcamBtn.addEventListener('click', async () => {
        try {
            const resp = await fetch('/api/start-webcam', { method: 'POST' });
            const data = await resp.json();
            if (resp.ok) {
                startFeed();
            } else {
                alert(data.error || 'Webcam start failed');
            }
        } catch (err) {
            alert('Webcam error: ' + err.message);
        }
    });

    // ── Demo Video ──
    demoBtn.addEventListener('click', async () => {
        try {
            const resp = await fetch('/api/start-demo', { method: 'POST' });
            const data = await resp.json();
            if (resp.ok) {
                startFeed();
            } else {
                alert(data.error || 'Demo start failed');
            }
        } catch (err) {
            alert('Demo error: ' + err.message);
        }
    });

    // ── Stop ──
    stopBtn.addEventListener('click', async () => {
        try {
            stopBtn.disabled = true;
            await fetch('/api/stop', { method: 'POST' });
        } catch (err) {
            console.error('Stop error:', err);
        }
        stopFeed();
    });

    // ── Init ──
    connectWebSocket();

    // Fallback stats polling (if WebSocket is slow)
    statsInterval = setInterval(async () => {
        try {
            const resp = await fetch('/api/stats');
            const data = await resp.json();
            updateStats(data);
        } catch (e) { /* ignore */ }
    }, 2000);
})();
