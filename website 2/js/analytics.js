/* ═══════════════════════════════════════════════════════
   SCSS – Analytics Page JavaScript
   ═══════════════════════════════════════════════════════ */

/* ─── Hamburger Toggle ──────────────────────────────── */
const hamburger = document.getElementById('hamburger');
const navLinks  = document.getElementById('navLinks');
if (hamburger && navLinks) {
  hamburger.addEventListener('click', () => {
    navLinks.classList.toggle('open');
  });
  document.addEventListener('click', (e) => {
    if (!hamburger.contains(e.target) && !navLinks.contains(e.target)) {
      navLinks.classList.remove('open');
    }
  });
}

/* ─── Animated Counters ─────────────────────────────── */
function animateCounter(el) {
  const target = parseInt(el.dataset.target, 10);
  const suffix = el.dataset.suffix || '';
  const duration = 1400;
  const start = performance.now();

  function step(now) {
    const elapsed = now - start;
    const progress = Math.min(elapsed / duration, 1);
    // ease out expo
    const eased = progress === 1 ? 1 : 1 - Math.pow(2, -10 * progress);
    const current = Math.round(eased * target);
    el.textContent = current + suffix;
    if (progress < 1) requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
}

const counterEls = document.querySelectorAll('.stat-value[data-target]');
const counterObs = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      animateCounter(entry.target);
      counterObs.unobserve(entry.target);
    }
  });
}, { threshold: 0.5 });
counterEls.forEach(el => counterObs.observe(el));

/* ─── Chart.js defaults ─────────────────────────────── */
Chart.defaults.color = '#9aa3b8';
Chart.defaults.font.family = "'Exo 2', sans-serif";

/* ─── Alerts Over Time – Line Chart ────────────────── */
(function buildLineChart() {
  const ctx = document.getElementById('alertsLineChart');
  if (!ctx) return;

  const labels = ['May 2', 'May 3', 'May 4', 'May 5', 'May 6', 'May 7', 'May 8'];
  const data   = [12, 18, 15, 24, 20, 16, 23];

  // Gradient fill
  const gradient = ctx.getContext('2d').createLinearGradient(0, 0, 0, 260);
  gradient.addColorStop(0, 'rgba(239,68,68,0.35)');
  gradient.addColorStop(1, 'rgba(239,68,68,0.00)');

  new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [{
        label: 'Alerts',
        data,
        borderColor: '#ef4444',
        borderWidth: 2.5,
        pointBackgroundColor: '#ef4444',
        pointBorderColor: '#ef4444',
        pointBorderWidth: 2,
        pointRadius: 5,
        pointHoverRadius: 8,
        pointShadowBlur: 12,
        fill: true,
        backgroundColor: gradient,
        tension: 0.45,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 1400, easing: 'easeOutQuart' },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: 'rgba(8,9,15,0.9)',
          borderColor: 'rgba(239,68,68,0.4)',
          borderWidth: 1,
          titleColor: '#f0f4ff',
          bodyColor: '#ef4444',
          padding: 10,
          callbacks: {
            label: ctx => ` ${ctx.parsed.y} alerts`
          }
        }
      },
      scales: {
        x: {
          grid: { color: 'rgba(255,255,255,0.04)', drawBorder: false },
          ticks: { font: { size: 11 } }
        },
        y: {
          beginAtZero: true,
          max: 30,
          grid: { color: 'rgba(255,255,255,0.04)', drawBorder: false },
          ticks: { font: { size: 11 }, stepSize: 10 }
        }
      }
    }
  });
})();

/* ─── Alerts by Type – Donut Chart ─────────────────── */
(function buildDonutChart() {
  const ctx = document.getElementById('alertsDonutChart');
  if (!ctx) return;

  const types = [
    { label: 'No Helmet',      value: 9,  pct: '39%', color: '#ef4444' },
    { label: 'No Vest',        value: 6,  pct: '26%', color: '#f5b400' },
    { label: 'Restricted Area',value: 4,  pct: '17%', color: '#3b8eff' },
    { label: 'Fire Risk',      value: 2,  pct: '9%',  color: '#f97316' },
    { label: 'Other',          value: 2,  pct: '9%',  color: '#22c55e' },
  ];

  new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: types.map(t => t.label),
      datasets: [{
        data: types.map(t => t.value),
        backgroundColor: types.map(t => t.color + 'cc'),
        borderColor: types.map(t => t.color),
        borderWidth: 2,
        hoverBackgroundColor: types.map(t => t.color),
        hoverOffset: 8,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      cutout: '68%',
      animation: { duration: 1400, easing: 'easeOutQuart' },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: 'rgba(8,9,15,0.9)',
          borderColor: 'rgba(255,255,255,0.1)',
          borderWidth: 1,
          titleColor: '#f0f4ff',
          bodyColor: '#9aa3b8',
          padding: 10,
          callbacks: {
            label: ctx => ` ${ctx.label}: ${ctx.parsed} (${types[ctx.dataIndex].pct})`
          }
        }
      }
    }
  });

  // Build legend
  const legend = document.getElementById('donutLegend');
  if (legend) {
    types.forEach(t => {
      const li = document.createElement('li');
      li.innerHTML = `
        <span class="legend-dot" style="background:${t.color};box-shadow:0 0 8px ${t.color}88;"></span>
        <span>${t.label}</span>
        <span class="legend-pct">${t.value} <span style="color:#5a6480;">(${t.pct})</span></span>
      `;
      legend.appendChild(li);
    });
  }
})();

/* ─── Compliance Overview – Bar Chart ───────────────── */
(function buildBarChart() {
  const ctx = document.getElementById('complianceBarChart');
  if (!ctx) return;

  const sites  = ['Site A', 'Site B', 'Site C'];
  const values = [98, 94, 97];
  const colors = ['#22c55e', '#f5b400', '#3b8eff'];

  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: sites,
      datasets: [{
        label: 'Compliance %',
        data: values,
        backgroundColor: colors.map(c => c + 'bb'),
        borderColor: colors,
        borderWidth: 2,
        borderRadius: 8,
        borderSkipped: false,
        hoverBackgroundColor: colors,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: {
        duration: 1400,
        easing: 'easeOutBounce',
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: 'rgba(8,9,15,0.9)',
          borderColor: 'rgba(255,255,255,0.1)',
          borderWidth: 1,
          titleColor: '#f0f4ff',
          bodyColor: '#9aa3b8',
          padding: 10,
          callbacks: {
            label: ctx => ` ${ctx.parsed.y}% compliance`
          }
        },
        datalabels: { display: false }
      },
      scales: {
        x: {
          grid: { display: false },
          ticks: { font: { size: 13, weight: '600' } }
        },
        y: {
          beginAtZero: false,
          min: 85,
          max: 100,
          grid: { color: 'rgba(255,255,255,0.04)', drawBorder: false },
          ticks: {
            font: { size: 11 },
            callback: v => v + '%'
          }
        }
      }
    },
    plugins: [{
      // Draw percentage labels above bars
      id: 'barLabels',
      afterDraw(chart) {
        const { ctx, scales: { x, y } } = chart;
        chart.data.datasets.forEach((dataset, i) => {
          const meta = chart.getDatasetMeta(i);
          meta.data.forEach((bar, j) => {
            const value = dataset.data[j];
            ctx.save();
            ctx.fillStyle = '#f0f4ff';
            ctx.font = 'bold 13px "Exo 2", sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText(value + '%', bar.x, bar.y - 8);
            ctx.restore();
          });
        });
      }
    }]
  });
})();

/* ─── Smooth reveal for chart cards ────────────────── */
const cardEls = document.querySelectorAll('.chart-card, .insight-banner');
const cardObs = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      entry.target.style.opacity = '1';
      entry.target.style.transform = 'translateY(0)';
      cardObs.unobserve(entry.target);
    }
  });
}, { threshold: 0.1 });

cardEls.forEach(el => {
  el.style.opacity = '0';
  el.style.transform = 'translateY(28px)';
  el.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
  cardObs.observe(el);
});

/* ─── Date picker button (simple toggle) ───────────── */
const datePicker = document.querySelector('.date-picker-btn');
if (datePicker) {
  datePicker.addEventListener('click', () => {
    datePicker.style.borderColor = 'var(--gold)';
    setTimeout(() => { datePicker.style.borderColor = ''; }, 600);
  });
}