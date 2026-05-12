/**
 * Live Monitor Page — JavaScript Controller
 * Handles video upload, webcam, WebSocket connections, and real-time UI updates
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
    const statFrame = document.getElementById('stat-frame');
    const statElapsed = document.getElementById('stat-elapsed');

    let ws = null;
    let statsInterval = null;
    let alertItems = [];

    // ── WebSocket Connection ──
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
            riskBar.style.background = '#ff3e5e';
            statRisk.className = 'stat-value red';
        } else if (risk > 30) {
            riskBar.style.background = '#ffb020';
            statRisk.className = 'stat-value amber';
        } else {
            riskBar.style.background = '#00e676';
            statRisk.className = 'stat-value green';
        }

        statDanger.textContent = stats.danger_count || 0;
        statWarning.textContent = stats.warning_count || 0;
        statSafe.textContent = stats.safe_count || 0;

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
    }

    // ── Stop Feed ──
    function stopFeed() {
        videoFeed.style.display = 'none';
        videoFeed.src = '';
        videoPlaceholder.style.display = '';
        stopBtn.disabled = true;
        progressFill.style.width = '0%';
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
            await fetch('/api/stop', { method: 'POST' });
            stopFeed();
        } catch (err) {
            console.error('Stop error:', err);
        }
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
