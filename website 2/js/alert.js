/* ══════════════════════════════════════════════
   SCSS – alert.js
   Alerts Dashboard page interactivity
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
    navLinks.querySelectorAll('a').forEach(a =>
      a.addEventListener('click', () => {
        navLinks.classList.remove('open');
        hamburger.classList.remove('active');
      })
    );
  }

  /* ── 2. Navbar shadow on scroll ── */
  const navbar = document.getElementById('navbar');
  window.addEventListener('scroll', () => {
    if (navbar)
      navbar.style.boxShadow = window.scrollY > 10
        ? '0 4px 32px rgba(0,0,0,0.5)'
        : 'none';
  }, { passive: true });

  /* ── 3. Animated counter for summary cards ── */
  function animateCounter(el, target, duration = 1200) {
    let start     = 0;
    const step    = target / (duration / 16);
    const ticker  = setInterval(() => {
      start += step;
      if (start >= target) { el.textContent = target; clearInterval(ticker); return; }
      el.textContent = Math.floor(start);
    }, 16);
  }

  // Trigger counters when cards enter viewport
  const counterEls = document.querySelectorAll('.sc-number[data-target]');
  if ('IntersectionObserver' in window) {
    const io = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          const el     = entry.target;
          const target = parseInt(el.dataset.target, 10);
          animateCounter(el, target);
          io.unobserve(el);
        }
      });
    }, { threshold: 0.3 });
    counterEls.forEach(el => io.observe(el));
  } else {
    counterEls.forEach(el => { el.textContent = el.dataset.target; });
  }

  /* ── 4. Live search / filter on table ── */
  const searchInput = document.getElementById('alertSearch');
  const tableRows   = document.querySelectorAll('.alert-row');

  if (searchInput) {
    searchInput.addEventListener('input', () => {
      const q = searchInput.value.toLowerCase().trim();
      tableRows.forEach(row => {
        const text = row.textContent.toLowerCase();
        row.style.display = text.includes(q) ? '' : 'none';
      });
    });
  }

  /* ── 5. Filter button – severity toggle ── */
  const filterBtn = document.querySelector('.btn-filter');
  let filterState = 'all'; // 'all' | 'high' | 'medium'

  if (filterBtn) {
    filterBtn.addEventListener('click', () => {
      if (filterState === 'all')    filterState = 'high';
      else if (filterState === 'high')   filterState = 'medium';
      else filterState = 'all';

      filterBtn.textContent =
        filterState === 'all'    ? '⚙ Filter' :
        filterState === 'high'   ? '🔴 High Only' :
                                   '🟠 Medium Only';

      tableRows.forEach(row => {
        if (filterState === 'all') { row.style.display = ''; return; }
        row.style.display = row.dataset.severity === filterState ? '' : 'none';
      });
    });
  }

  /* ── 6. Row click → highlight ── */
  tableRows.forEach(row => {
    row.addEventListener('click', () => {
      tableRows.forEach(r => r.classList.remove('selected'));
      row.classList.add('selected');
    });
  });

  // Add selected row style dynamically
  const selStyle = document.createElement('style');
  selStyle.textContent = `
    .alert-row.selected {
      background: rgba(245,180,0,0.07) !important;
      border-left: 3px solid var(--accent);
    }
  `;
  document.head.appendChild(selStyle);

  /* ── 7. Export Report button ── */
  const exportBtn = document.querySelector('.btn-export');
  if (exportBtn) {
    exportBtn.addEventListener('click', () => {
      showToast('<i class="fas fa-download"></i> Report export started…');
    });
  }

  /* ── 8. View All Alerts button ── */
  const viewAllBtn = document.querySelector('.btn-view-all');
  if (viewAllBtn) {
    viewAllBtn.addEventListener('click', () => {
      showToast('<i class="fas fa-list"></i> Loading full alerts list…');
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
        background:rgba(9,15,28,0.96); border:1px solid rgba(245,180,0,0.4);
        color:#e8ecf4; padding:12px 20px; border-radius:10px;
        font-size:0.85rem; display:flex; align-items:center; gap:10px;
        backdrop-filter:blur(12px);
        box-shadow:0 8px 30px rgba(0,0,0,0.5),0 0 20px rgba(245,180,0,0.12);
        transform:translateY(20px); opacity:0;
        transition:opacity 0.3s ease, transform 0.3s ease;
        max-width:280px;
      `;
      document.body.appendChild(toast);
    }
    toast.innerHTML = html;
    requestAnimationFrame(() => {
      toast.style.opacity   = '1';
      toast.style.transform = 'translateY(0)';
    });
    clearTimeout(toast._timer);
    toast._timer = setTimeout(() => {
      toast.style.opacity   = '0';
      toast.style.transform = 'translateY(20px)';
    }, 3000);
  }

  /* ── 10. Simulated real-time new alert ── */
  setTimeout(() => {
    showToast('<i class="fas fa-triangle-exclamation" style="color:#ef4444"></i> New alert: PPE violation – Site C');
  }, 6000);

  /* ── 11. Summary card click → scroll to table ── */
  document.querySelectorAll('.sc-link').forEach(link => {
    link.addEventListener('click', e => {
      e.preventDefault();
      const target = document.getElementById('alertsTable');
      if (target) target.scrollIntoView({ behavior: 'smooth', block: 'start' });
    });
  });

});