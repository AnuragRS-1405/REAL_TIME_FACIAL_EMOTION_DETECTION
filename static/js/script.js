/* ─────────────────────────────────────────────
   Real-Time Facial Emotion Recognition
   script.js  –  Frontend Logic
──────────────────────────────────────────────── */

// Automatically connect to FastAPI backend on port 8000 when opened via VS Code Live Server (port 5500)
window.API = (window.location.port && window.location.port !== '8000') ? 'http://localhost:8000' : '';
const API = window.API;

// Emotion display mappings
const EMOTION_EMOJIS = {
  'Angry': '😠',
  'Disgust': '🤢',
  'Fear': '😨',
  'Happy': '😊',
  'Neutral': '😐',
  'Sad': '😢',
  'Surprise': '😲',
};
const EMOTION_COLORS = {
  'Angry': '#ff6b6b',
  'Disgust': '#a29bfe',
  'Fear': '#74b9ff',
  'Happy': '#55efc4',
  'Neutral': '#b2bec3',
  'Sad': '#6c5ce7',
  'Surprise': '#ffeaa7',
};

/* ──────────────────────────────────────────────
   TOAST NOTIFICATION
──────────────────────────────────────────────── */
let toastTimeout;
function showToast(msg, type = 'info') {
  const t = document.getElementById('toast');
  if (!t) return;
  t.textContent = msg;
  t.className = `toast show ${type}`;
  clearTimeout(toastTimeout);
  toastTimeout = setTimeout(() => t.classList.remove('show'), 3500);
}

/* ──────────────────────────────────────────────
   LOADING OVERLAY
──────────────────────────────────────────────── */
function showLoading(text = 'Analysing...') {
  const el = document.getElementById('loadingOverlay');
  if (!el) return;
  el.querySelector('.loading-text').textContent = text;
  el.classList.add('show');
}
function hideLoading() {
  const el = document.getElementById('loadingOverlay');
  if (el) el.classList.remove('show');
}

/* ──────────────────────────────────────────────
   TAB SYSTEM
──────────────────────────────────────────────── */
function initTabs() {
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const group = btn.closest('[data-tabs]') || btn.parentElement.parentElement;
      const target = btn.dataset.tab;
      group.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      group.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
      btn.classList.add('active');
      const content = group.querySelector(`[data-tab-content="${target}"]`);
      if (content) content.classList.add('active');
    });
  });
}

/* ──────────────────────────────────────────────
   ACTIVE NAV LINK
──────────────────────────────────────────────── */
function setActiveNav() {
  const path = window.location.pathname;
  // If running via Live Server (port 5500), make relative links work between template files
  if (window.location.port && window.location.port !== '8000') {
    document.querySelectorAll('.nav-links a').forEach(a => {
      const href = a.getAttribute('href');
      if (href === '/') a.href = 'index.html';
      else if (href === '/analysis') a.href = 'analysis.html';
      else if (href === '/evaluation') a.href = 'evaluation.html';
    });
  }

  document.querySelectorAll('.nav-links a').forEach(a => {
    a.classList.remove('active');
    const href = a.getAttribute('href');
    if (href === path || (path === '/' && href === '/') ||
      (path.includes('analysis') && href.includes('analysis')) ||
      (path.includes('evaluation') && href.includes('evaluation'))) {
      a.classList.add('active');
    }
  });
}

/* ──────────────────────────────────────────────
   API STATUS CHECK
──────────────────────────────────────────────── */
async function checkStatus() {
  try {
    const resp = await fetch(`${API}/status`, { signal: AbortSignal.timeout(2500) });
    const data = await resp.json();
    const bar = document.getElementById('statusBar');
    if (!bar) return data;

    bar.innerHTML = '';
    const items = [
      { label: 'ANN Model', ok: data.ann_loaded },
      { label: 'CNN Model', ok: data.cnn_loaded },
      { label: 'Metrics', ok: data.metrics_available },
    ];
    items.forEach(item => {
      const div = document.createElement('div');
      div.className = 'status-item';
      div.innerHTML = `
        <span class="status-dot ${item.ok ? 'ok' : 'warn'}"></span>
        <span>${item.label}: ${item.ok ? 'Ready' : 'Not loaded'}</span>`;
      bar.appendChild(div);
    });
    return data;
  } catch (err) {
    const bar = document.getElementById('statusBar');
    if (bar) {
      bar.innerHTML = `
        <div class="status-item">
          <span class="status-dot ok"></span>
          <span>Ready (Waiting for Backend on Port 8000)</span>
        </div>`;
    }
    return null;
  }
}

/* ──────────────────────────────────────────────
   HOME PAGE – PREDICTION
──────────────────────────────────────────────── */
function initHomePage() {
  // ── Model selector ──
  let selectedModel = 'CNN';
  const modelBtns = document.querySelectorAll('.model-btn');
  modelBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      modelBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      selectedModel = btn.dataset.model;
    });
  });
  // Set CNN active by default
  document.querySelector('[data-model="CNN"]')?.classList.add('active');

  // ── Upload / Drag-drop ──
  const uploadZone = document.getElementById('uploadZone');
  const fileInput = document.getElementById('fileInput');
  const previewCont = document.getElementById('previewContainer');
  const previewImg = document.getElementById('previewImg');
  const removeBtn = document.getElementById('removeBtn');
  const predictBtn = document.getElementById('predictBtn');

  if (!uploadZone) return;

  uploadZone.addEventListener('dragover', e => {
    e.preventDefault();
    uploadZone.classList.add('drag-over');
  });
  uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('drag-over'));
  uploadZone.addEventListener('drop', e => {
    e.preventDefault();
    uploadZone.classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  });
  fileInput.addEventListener('change', () => {
    if (fileInput.files[0]) handleFile(fileInput.files[0]);
  });
  removeBtn?.addEventListener('click', () => {
    previewCont.style.display = 'none';
    uploadZone.style.display = '';
    fileInput.value = '';
    predictBtn.disabled = true;
    clearResult();
  });

  let currentFile = null;
  function handleFile(file) {
    if (!file.type.startsWith('image/')) {
      showToast('Please upload a valid image file (JPG, PNG, WebP).', 'error');
      return;
    }
    currentFile = file;
    const url = URL.createObjectURL(file);
    previewImg.src = url;
    previewImg.style.display = 'block';
    previewCont.style.display = 'block';
    uploadZone.style.display = 'none';
    predictBtn.disabled = false;
  }

  // ── Predict button ──
  predictBtn?.addEventListener('click', async () => {
    if (!currentFile) { showToast('Please select an image first.', 'error'); return; }

    showLoading('Running inference...');
    predictBtn.disabled = true;

    const fd = new FormData();
    fd.append('file', currentFile);

    try {
      const resp = await fetch(`${API}/predict/image?model=${selectedModel}`, {
        method: 'POST', body: fd
      });
      if (!resp.ok) {
        const err = await resp.json();
        throw new Error(err.detail || 'Prediction failed');
      }
      const data = await resp.json();
      displayResult(data);
      showToast(`Emotion detected: ${data.predicted_emotion}`, 'success');
    } catch (err) {
      showToast(err.message, 'error');
      console.error(err);
    } finally {
      hideLoading();
      predictBtn.disabled = false;
    }
  });

  function displayResult(data) {
    const emotion = data.predicted_emotion;
    const confidence = data.confidence;
    const probs = data.probabilities;

    // Main result
    const emoEl = document.getElementById('resultEmotion');
    const emoEmoji = document.getElementById('resultEmoji');
    const confEl = document.getElementById('resultConfidence');
    const confBar = document.getElementById('confidenceBar');

    if (emoEl) {
      emoEl.textContent = emotion;
      emoEl.style.color = EMOTION_COLORS[emotion] || 'white';
    }
    if (emoEmoji) emoEmoji.textContent = EMOTION_EMOJIS[emotion] || '🤔';
    if (confEl) confEl.textContent = `Confidence: ${(confidence * 100).toFixed(1)}%`;
    if (confBar) setTimeout(() => confBar.style.width = `${confidence * 100}%`, 50);

    // Probability bars
    if (probs) {
      const probList = document.getElementById('probList');
      if (probList) {
        probList.innerHTML = '';
        const sorted = Object.entries(probs).sort((a, b) => b[1] - a[1]);
        sorted.forEach(([label, prob]) => {
          const row = document.createElement('div');
          row.className = 'prob-row';
          row.innerHTML = `
            <span class="prob-label">${label}</span>
            <div class="prob-bar-wrap">
              <div class="prob-bar" style="background:${EMOTION_COLORS[label] || 'var(--accent-1)'}"></div>
            </div>
            <span class="prob-val">${(prob * 100).toFixed(1)}%</span>`;
          probList.appendChild(row);
          // Animate bar
          setTimeout(() => {
            row.querySelector('.prob-bar').style.width = `${prob * 100}%`;
          }, 100);
        });
      }
    }

    // Show result area
    const resultArea = document.getElementById('resultArea');
    if (resultArea) {
      resultArea.style.display = 'block';
      resultArea.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
  }

  function clearResult() {
    const resultArea = document.getElementById('resultArea');
    if (resultArea) resultArea.style.display = 'none';
  }
}

/* ──────────────────────────────────────────────
   ANALYSIS PAGE
──────────────────────────────────────────────── */
async function initAnalysisPage() {
  try {
    const resp = await fetch(`${API}/analysis/data`, { signal: AbortSignal.timeout(3000) });
    const data = await resp.json();

    const plots = data.plots || {};

    // Load each plot image
    const plotMap = {
      'plotDistribution': plots.class_distribution,
      'plotAnnHistory': plots.ann_history,
      'plotCnnHistory': plots.cnn_history,
      'plotAnnCm': plots.ann_confusion_matrix,
      'plotCnnCm': plots.cnn_confusion_matrix,
      'plotComparison': plots.ann_vs_cnn,
      'plotSamples': plots.sample_predictions,
      'plotSampleImgs': plots.sample_images,
      'plotTrainTime': plots.training_time,
    };
    Object.entries(plotMap).forEach(([id, src]) => {
      const el = document.getElementById(id);
      if (el && src) {
        const targetSrc = API ? `${API}${src}` : src;
        el.src = targetSrc;
      }
    });

    // If history data exists, render live charts
    if (data.history) {
      renderHistoryChart('annAccChart', data.history.ann,
        'accuracy', 'val_accuracy', 'ANN Accuracy', '#ff6b6b');
      renderHistoryChart('cnnAccChart', data.history.cnn,
        'accuracy', 'val_accuracy', 'CNN Accuracy', '#4ecdc4');
      renderHistoryChart('annLossChart', data.history.ann,
        'loss', 'val_loss', 'ANN Loss', '#ff6b6b');
      renderHistoryChart('cnnLossChart', data.history.cnn,
        'loss', 'val_loss', 'CNN Loss', '#4ecdc4');
    }
  } catch (err) {
    console.warn('Analysis data load failed:', err);
    showToast('Could not load analysis data. Train models first.', 'error');
  }
}

function renderHistoryChart(canvasId, history, trainKey, valKey, label, color) {
  const canvas = document.getElementById(canvasId);
  if (!canvas || !history) return;
  const ctx = canvas.getContext('2d');
  const n = history[trainKey]?.length || 0;
  if (n === 0) return;

  const W = canvas.width, H = canvas.height;
  const pad = { top: 35, right: 25, bottom: 45, left: 55 };
  const chartW = W - pad.left - pad.right;
  const chartH = H - pad.top - pad.bottom;

  const trainData = history[trainKey] || [];
  const valData = history[valKey] || [];
  const allVals = [...trainData, ...valData].filter(v => isFinite(v));
  const minV = Math.max(0, Math.min(...allVals) * 0.9);
  const maxV = Math.min(2.5, Math.max(...allVals) * 1.05);
  const range = maxV - minV || 1;

  const toX = i => pad.left + (i / Math.max(1, n - 1)) * chartW;
  const toY = v => pad.top + chartH - ((v - minV) / range) * chartH;

  // Clear & background
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = '#0f172a';
  ctx.fillRect(0, 0, W, H);

  // Subtle border
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
  ctx.lineWidth = 1;
  ctx.strokeRect(0, 0, W, H);

  // Grid lines
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
  for (let i = 0; i <= 4; i++) {
    const y = pad.top + (i / 4) * chartH;
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(W - pad.right, y);
    ctx.stroke();

    ctx.fillStyle = '#94a3b8';
    ctx.font = '10px Inter, sans-serif';
    ctx.textAlign = 'right';
    const val = maxV - (i / 4) * range;
    ctx.fillText(val.toFixed(2), pad.left - 8, y + 4);
  }

  // Draw gradient area under training curve
  if (trainData.length > 1) {
    const grad = ctx.createLinearGradient(0, pad.top, 0, pad.top + chartH);
    grad.addColorStop(0, color === '#4ecdc4' ? 'rgba(0, 242, 254, 0.25)' : 'rgba(255, 75, 114, 0.25)');
    grad.addColorStop(1, 'rgba(15, 23, 42, 0.0)');
    ctx.beginPath();
    ctx.moveTo(toX(0), toY(trainData[0]));
    for (let i = 1; i < trainData.length; i++) {
      ctx.lineTo(toX(i), toY(trainData[i]));
    }
    ctx.lineTo(toX(trainData.length - 1), pad.top + chartH);
    ctx.lineTo(toX(0), pad.top + chartH);
    ctx.closePath();
    ctx.fillStyle = grad;
    ctx.fill();
  }

  // Draw curves
  function drawLine(data, strokeColor, isDash = false) {
    if (!data.length) return;
    ctx.save();
    ctx.beginPath();
    ctx.strokeStyle = strokeColor;
    ctx.lineWidth = 2.4;
    ctx.shadowColor = strokeColor;
    ctx.shadowBlur = isDash ? 0 : 8;
    if (isDash) ctx.setLineDash([5, 3]);
    data.forEach((v, i) => {
      i === 0 ? ctx.moveTo(toX(i), toY(v)) : ctx.lineTo(toX(i), toY(v));
    });
    ctx.stroke();
    ctx.restore();
  }

  const primaryColor = color === '#4ecdc4' ? '#00f2fe' : '#ff4b72';
  const valColor = color === '#4ecdc4' ? '#10b981' : '#a855f7';

  drawLine(trainData, primaryColor, false);
  drawLine(valData, valColor, true);

  // Modern Legend
  ctx.font = '600 11px Inter, sans-serif';
  ctx.textAlign = 'left';
  ctx.fillStyle = primaryColor;
  ctx.fillText('● Train', pad.left + 5, 20);

  ctx.fillStyle = valColor;
  ctx.fillText('╌╌ Val', pad.left + 75, 20);
}

/* ──────────────────────────────────────────────
   EVALUATION PAGE
──────────────────────────────────────────────── */
async function initEvaluationPage() {
  try {
    const resp = await fetch(`${API}/metrics`, { signal: AbortSignal.timeout(3000) });
    const data = await resp.json();

    // ANN metrics
    fillMetrics('ann', data.ann, data.cnn);
    fillMetrics('cnn', data.cnn, data.ann);

    // Comparison table
    buildComparisonTable(data);

    // Animate bars
    setTimeout(animateBars, 200);
  } catch (err) {
    console.warn('Metrics load failed:', err);
    showToast('Metrics not available. Run train_models.py first.', 'error');
  }
}

function fillMetrics(model, my, their) {
  const prefix = model;
  const acc = my?.accuracy ?? 0;
  const prec = my?.precision ?? 0;
  const rec = my?.recall ?? 0;
  const f1 = my?.f1 ?? 0;

  setEl(`${prefix}Accuracy`, `${(acc * 100).toFixed(2)}%`);
  setEl(`${prefix}Precision`, `${(prec * 100).toFixed(2)}%`);
  setEl(`${prefix}Recall`, `${(rec * 100).toFixed(2)}%`);
  setEl(`${prefix}F1`, `${(f1 * 100).toFixed(2)}%`);
  setEl(`${prefix}TrainTime`, my?.train_time ? `${my.train_time.toFixed(1)}s` : 'N/A');
  setEl(`${prefix}InfTime`, my?.inf_time ? `${(my.inf_time * 1000).toFixed(3)}ms` : 'N/A');

  // Bar widths
  setBar(`${prefix}AccBar`, acc * 100);
  setBar(`${prefix}PrecBar`, prec * 100);
  setBar(`${prefix}RecBar`, rec * 100);
  setBar(`${prefix}F1Bar`, f1 * 100);
}

function setEl(id, val) {
  const el = document.getElementById(id);
  if (el) el.textContent = val;
}
function setBar(id, pct) {
  const el = document.getElementById(id);
  if (el) el.setAttribute('data-width', pct);
}
function animateBars() {
  document.querySelectorAll('.metric-bar[data-width]').forEach(bar => {
    setTimeout(() => bar.style.width = bar.getAttribute('data-width') + '%', 50);
  });
}

function buildComparisonTable(data) {
  const tbody = document.getElementById('comparisonTbody');
  if (!tbody) return;
  const ann = data.ann || {};
  const cnn = data.cnn || {};

  const rows = [
    ['Accuracy', ann.accuracy, cnn.accuracy, 'higher'],
    ['Precision', ann.precision, cnn.precision, 'higher'],
    ['Recall', ann.recall, cnn.recall, 'higher'],
    ['F1-Score', ann.f1, cnn.f1, 'higher'],
    ['Train Time', ann.train_time, cnn.train_time, 'lower'],
    ['Inf Time (ms)', ann.inf_time ? ann.inf_time * 1000 : null,
      cnn.inf_time ? cnn.inf_time * 1000 : null, 'lower'],
  ];

  tbody.innerHTML = '';
  rows.forEach(([label, av, cv, prefer]) => {
    const annBetter = prefer === 'higher' ? av >= cv : av <= cv;
    const annClass = annBetter ? 'winner' : 'loser';
    const cnnClass = annBetter ? 'loser' : 'winner';
    const fmt = v => v != null ? (label.includes('Time') ? `${v.toFixed(2)}` : `${(v * 100).toFixed(2)}%`) : 'N/A';
    const winner = annBetter ? 'ANN' : 'CNN';
    tbody.innerHTML += `
      <tr>
        <td>${label}</td>
        <td class="${annClass}">${fmt(av)}</td>
        <td class="${cnnClass}">${fmt(cv)}</td>
        <td><span style="color:var(--accent-4);font-weight:600">${winner}</span></td>
      </tr>`;
  });
}

/* ──────────────────────────────────────────────
   INIT
──────────────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', () => {
  setActiveNav();
  initTabs();
  checkStatus();

  const path = window.location.pathname;
  if (path === '/' || path.endsWith('index.html')) {
    initHomePage();
  } else if (path.includes('analysis')) {
    initAnalysisPage();
  } else if (path.includes('evaluation')) {
    initEvaluationPage();
  }
});
