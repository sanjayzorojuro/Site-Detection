/* ══════════════════════════════════════════════
   SCSS – live.js
   Live Monitor page interactivity
══════════════════════════════════════════════ */

document.addEventListener('DOMContentLoaded', () => {

  /* ── 1. Hamburger / mobile nav ── */
  const hamburger = document.getElementById('hamburger');
  const navLinks  = document.getElementById('navLinks');

  if (hamburger && navLinks) {
    hamburger.addEventListener('click', () => {
      navLinks.classList.toggle('open');
      hamburger.classList.toggle('active');
    });

    // Close on link click
    navLinks.querySelectorAll('a').forEach(a => {
      a.addEventListener('click', () => {
        navLinks.classList.remove('open');
        hamburger.classList.remove('active');
      });
    });
  }

  /* ── 2. Sticky navbar scroll effect ── */
  const navbar = document.getElementById('navbar');
  window.addEventListener('scroll', () => {
    if (navbar) {
      navbar.style.boxShadow = window.scrollY > 10
        ? '0 4px 32px rgba(0,0,0,0.5)'
        : 'none';
    }
  }, { passive: true });

  /* ── 3. Live timestamps ── */
  const timeEls = [
    document.getElementById('time1'),
    document.getElementById('time2'),
    document.getElementById('time3'),
    document.getElementById('time4'),
  ];

  function updateTimes() {
    const now = new Date();
    const h   = now.getHours();
    const m   = String(now.getMinutes()).padStart(2,'0');
    const s   = String(now.getSeconds()).padStart(2,'0');
    const ampm = h >= 12 ? 'PM' : 'AM';
    const h12  = (h % 12 || 12);
    const str  = `${h12}:${m}:${s} ${ampm}`;
    timeEls.forEach(el => { if (el) el.textContent = str; });
  }

  updateTimes();
  setInterval(updateTimes, 1000);

  /* ── 4. Filter pills ── */
  const pills   = document.querySelectorAll('.pill');
  const cards   = document.querySelectorAll('.camera-card');

  pills.forEach(pill => {
    pill.addEventListener('click', () => {
      pills.forEach(p => p.classList.remove('active'));
      pill.classList.add('active');

      const filter = pill.dataset.filter;

      cards.forEach(card => {
        if (filter === 'all' || card.dataset.site === filter) {
          card.style.display = '';
          card.style.opacity = '0';
          card.style.transform = 'translateY(16px)';
          requestAnimationFrame(() => {
            card.style.transition = 'opacity 0.4s ease, transform 0.4s ease';
            card.style.opacity = '1';
            card.style.transform = 'translateY(0)';
          });
        } else {
          card.style.display = 'none';
        }
      });
    });
  });

  /* ── 5. View mode buttons ── */
  const viewBtns = document.querySelectorAll('.view-btn');
  const grid     = document.getElementById('cameraGrid');

  viewBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      viewBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');

      const view = btn.dataset.view;
      if (grid) {
        grid.className = 'camera-grid'; // reset
        if (view === 'list')   grid.classList.add('view-list');
        if (view === 'expand') grid.classList.add('view-expand');
      }
    });
  });

  /* ── 6. Donut chart (pure Canvas) ── */
  const canvas = document.getElementById('donutChart');
  if (canvas) {
    drawDonut(canvas, [
      { value: 14, color: '#22c55e', glow: 'rgba(34,197,94,0.5)' },
      { value: 1,  color: '#ef4444', glow: 'rgba(239,68,68,0.5)'  },
      { value: 1,  color: '#f5b400', glow: 'rgba(245,180,0,0.5)'  },
    ]);
  }

  function drawDonut(canvas, segments) {
    const ctx   = canvas.getContext('2d');
    const cx    = canvas.width  / 2;
    const cy    = canvas.height / 2;
    const outer = Math.min(cx, cy) - 6;
    const inner = outer * 0.62;
    const total = segments.reduce((s, d) => s + d.value, 0);
    let   start = -Math.PI / 2;
    const gap   = 0.04; // radians gap between segments

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    segments.forEach(seg => {
      const angle = (seg.value / total) * (Math.PI * 2) - gap;

      // Glow shadow
      ctx.save();
      ctx.shadowColor = seg.glow;
      ctx.shadowBlur  = 14;

      ctx.beginPath();
      ctx.arc(cx, cy, outer, start, start + angle);
      ctx.arc(cx, cy, inner, start + angle, start, true);
      ctx.closePath();
      ctx.fillStyle = seg.color;
      ctx.fill();

      ctx.restore();

      start += angle + gap;
    });

    // Inner dark circle
    ctx.beginPath();
    ctx.arc(cx, cy, inner - 2, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(5,10,18,0.95)';
    ctx.fill();
  }

  /* ── 7. Hover tooltip on camera cards ── */
  cards.forEach(card => {
    const feed = card.querySelector('.cam-feed');
    if (!feed) return;

    feed.addEventListener('mouseenter', () => {
      const corners = card.querySelectorAll('.frame-corner');
      corners.forEach(c => { c.style.opacity = '1'; c.style.transition = 'opacity 0.3s'; });
    });

    feed.addEventListener('mouseleave', () => {
      const corners = card.querySelectorAll('.frame-corner');
      corners.forEach(c => { c.style.opacity = '0.7'; });
    });
  });

  /* ── 8. Add Camera button placeholder ── */
  const addBtn = document.querySelector('.btn-add-camera');
  if (addBtn) {
    addBtn.addEventListener('click', () => {
      showToast('<i class="fas fa-camera"></i> Camera configuration coming soon!');
    });
  }

  /* ── 9. Toast helper ── */
  function showToast(html) {
    let toast = document.querySelector('.scss-toast');
    if (!toast) {
      toast = document.createElement('div');
      toast.className = 'scss-toast';
      toast.style.cssText = `
        position:fixed; bottom:30px; right:30px; z-index:9999;
        background:rgba(9,15,28,0.95); border:1px solid rgba(245,180,0,0.4);
        color:#e8ecf4; padding:12px 20px; border-radius:10px;
        font-size:0.85rem; display:flex; align-items:center; gap:10px;
        backdrop-filter:blur(12px); box-shadow:0 8px 30px rgba(0,0,0,0.5),0 0 20px rgba(245,180,0,0.15);
        transform:translateY(20px); opacity:0;
        transition:opacity 0.3s ease, transform 0.3s ease;
      `;
      document.body.appendChild(toast);
    }

    toast.innerHTML = html;
    requestAnimationFrame(() => {
      toast.style.opacity = '1';
      toast.style.transform = 'translateY(0)';
    });

    clearTimeout(toast._timer);
    toast._timer = setTimeout(() => {
      toast.style.opacity  = '0';
      toast.style.transform = 'translateY(20px)';
    }, 3000);
  }

  /* ── 10. Detection item hover ── */
  document.querySelectorAll('.detection-item').forEach(item => {
    item.addEventListener('mouseenter', () => {
      item.style.transform = 'translateX(4px)';
      item.style.transition = 'transform 0.25s ease';
    });
    item.addEventListener('mouseleave', () => {
      item.style.transform = 'translateX(0)';
    });
  });

  /* ── 11. Simulated live alert ticker ── */
  const alerts = [
    { text: '⚠️ No Helmet – Camera 2', color: '#ef4444' },
    { text: '🚧 Restricted Area – Camera 3', color: '#f97316' },
    { text: '✅ All Clear – Camera 4', color: '#22c55e' },
    { text: '⚠️ No Vest – Camera 1', color: '#f97316' },
  ];

  let alertIdx = 0;
  setInterval(() => {
    const a = alerts[alertIdx % alerts.length];
    alertIdx++;

    // Only show occasional simulated alerts
    if (Math.random() > 0.4) return;

    const el = document.createElement('div');
    el.innerHTML = a.text;
    el.style.cssText = `
      position:fixed; top:90px; right:20px; z-index:9998;
      background:rgba(9,15,28,0.95); border-left:3px solid ${a.color};
      color:#e8ecf4; padding:10px 16px; border-radius:6px;
      font-size:0.8rem; font-weight:600;
      backdrop-filter:blur(10px);
      box-shadow:0 4px 20px rgba(0,0,0,0.4);
      transform:translateX(120%); opacity:0;
      transition:transform 0.4s ease, opacity 0.4s ease;
      max-width:260px;
    `;
    document.body.appendChild(el);

    requestAnimationFrame(() => {
      el.style.transform = 'translateX(0)';
      el.style.opacity   = '1';
    });

    setTimeout(() => {
      el.style.transform = 'translateX(120%)';
      el.style.opacity   = '0';
      setTimeout(() => el.remove(), 400);
    }, 4000);

  }, 8000);

});

/* ── View mode grid overrides ── */
const style = document.createElement('style');
style.textContent = `
  .camera-grid.view-list {
    grid-template-columns: 1fr !important;
  }
  .camera-grid.view-list .cam-feed {
    height: 200px;
  }
  .camera-grid.view-expand {
    grid-template-columns: 1fr !important;
  }
  .camera-grid.view-expand .cam-feed {
    height: 400px;
  }
`;
document.head.appendChild(style);