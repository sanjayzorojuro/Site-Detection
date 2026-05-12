/* ═══════════════════════════════════════════════════
   SCSS – Contact Page JavaScript
   js/contact.js
═══════════════════════════════════════════════════ */

/* ── Navbar scroll shadow ── */
const navbar = document.getElementById('navbar');
window.addEventListener('scroll', () => {
  navbar.classList.toggle('scrolled', window.scrollY > 20);
});

/* ── Hamburger toggle ── */
const hamburger = document.getElementById('hamburger');
const navLinks  = document.getElementById('navLinks');

hamburger.addEventListener('click', () => {
  hamburger.classList.toggle('open');
  navLinks.classList.toggle('open');
});

// Close menu on nav link click
document.querySelectorAll('.nav-link').forEach(link => {
  link.addEventListener('click', () => {
    hamburger.classList.remove('open');
    navLinks.classList.remove('open');
  });
});

/* ── Scroll reveal for fade-up elements ── */
const revealObserver = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      entry.target.style.animationPlayState = 'running';
      revealObserver.unobserve(entry.target);
    }
  });
}, { threshold: 0.12 });

// Pause animations initially then play on scroll
document.querySelectorAll('.fade-up').forEach(el => {
  el.style.animationPlayState = 'paused';
  revealObserver.observe(el);
});

/* ── Form Validation & Submit ── */
const form       = document.getElementById('contactForm');
const sendBtn    = document.getElementById('sendBtn');
const btnLoader  = document.getElementById('btnLoader');
const formSuccess = document.getElementById('formSuccess');

// Field refs
const nameField    = document.getElementById('name');
const emailField   = document.getElementById('email');
const subjectField = document.getElementById('subject');
const messageField = document.getElementById('message');

// Error refs
const nameErr    = document.getElementById('nameErr');
const emailErr   = document.getElementById('emailErr');
const subjectErr = document.getElementById('subjectErr');
const messageErr = document.getElementById('messageErr');

/* Validate a single field */
function validateField(field, errorEl, condition, message) {
  if (!condition) {
    field.classList.add('error');
    field.classList.remove('valid');
    errorEl.textContent = message;
    errorEl.classList.add('show');
    return false;
  } else {
    field.classList.remove('error');
    field.classList.add('valid');
    errorEl.classList.remove('show');
    return true;
  }
}

/* Real-time validation on blur */
nameField.addEventListener('blur', () => {
  validateField(nameField, nameErr, nameField.value.trim().length >= 2, 'Please enter your name (min 2 chars).');
});

emailField.addEventListener('blur', () => {
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  validateField(emailField, emailErr, emailRegex.test(emailField.value.trim()), 'Please enter a valid email address.');
});

subjectField.addEventListener('blur', () => {
  validateField(subjectField, subjectErr, subjectField.value.trim().length >= 3, 'Please enter a subject (min 3 chars).');
});

messageField.addEventListener('blur', () => {
  validateField(messageField, messageErr, messageField.value.trim().length >= 10, 'Message must be at least 10 characters.');
});

/* Clear error on input */
[nameField, emailField, subjectField, messageField].forEach(field => {
  field.addEventListener('input', () => {
    if (field.classList.contains('error')) {
      field.classList.remove('error');
    }
  });
});

/* Form submit handler */
form.addEventListener('submit', function(e) {
  e.preventDefault();

  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

  const isNameValid    = validateField(nameField,    nameErr,    nameField.value.trim().length >= 2,                'Please enter your name (min 2 chars).');
  const isEmailValid   = validateField(emailField,   emailErr,   emailRegex.test(emailField.value.trim()),          'Please enter a valid email address.');
  const isSubjectValid = validateField(subjectField, subjectErr, subjectField.value.trim().length >= 3,             'Please enter a subject (min 3 chars).');
  const isMessageValid = validateField(messageField, messageErr, messageField.value.trim().length >= 10,            'Message must be at least 10 characters.');

  if (!isNameValid || !isEmailValid || !isSubjectValid || !isMessageValid) return;

  /* Show loading state */
  sendBtn.querySelector('.btn-text').style.display = 'none';
  btnLoader.style.display = 'flex';
  sendBtn.disabled = true;
  sendBtn.style.opacity = '0.8';

  /* Simulate API call with 1.8s delay */
  setTimeout(() => {
    // Hide loader, show success
    btnLoader.style.display = 'none';
    sendBtn.querySelector('.btn-text').style.display = 'flex';
    sendBtn.disabled = false;
    sendBtn.style.opacity = '1';

    formSuccess.style.display = 'flex';
    form.reset();

    // Remove validation classes
    [nameField, emailField, subjectField, messageField].forEach(f => {
      f.classList.remove('valid', 'error');
    });

    // Auto-hide success message after 5s
    setTimeout(() => {
      formSuccess.style.display = 'none';
    }, 5000);
  }, 1800);
});

/* ── Input focus glow effect ── */
document.querySelectorAll('.form-input').forEach(input => {
  input.addEventListener('focus', () => {
    input.closest('.input-wrap').style.filter = 'drop-shadow(0 0 6px rgba(250,204,21,0.12))';
  });
  input.addEventListener('blur', () => {
    input.closest('.input-wrap').style.filter = 'none';
  });
});