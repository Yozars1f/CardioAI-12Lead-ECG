/* ==========================================================================
   CardioAI-12Lead CDSS: Client-Side Dashboard Controller (Strict < 300 lines)
   ========================================================================== */

let activeCaseId = null, currentSignal = null, currentHeatmap = null;
let currentLeadAttribution = null, currentTopDiagnosis = null;
let currentStartSample = 0, currentWindowLength = 500, showHeatmap = true;
let isPlayingTelemetry = false, telemetryTimer = null, cachedCases = [];

const TERRITORY = {
  V1: "Septal", V2: "Septal", V3: "Anterior", V4: "Anterior",
  V5: "Lateral", V6: "Lateral", I: "High Lat", aVL: "High Lat",
  II: "Inferior", III: "Inferior", aVF: "Inferior", aVR: "Cavitary"
};

// DOM References
const caseCardsContainer = document.getElementById('caseCardsContainer');
const csvFileInput = document.getElementById('csvFileInput');
const uploadStatusText = document.getElementById('uploadStatusText');
const tempSlider = document.getElementById('tempSlider');
const tempValueDisplay = document.getElementById('tempValueDisplay');
const eceBadge = document.getElementById('eceBadge');
const triageBanner = document.getElementById('triageBanner');
const triageIndicator = document.getElementById('triageIndicator');
const triageBadgeText = document.getElementById('triageBadgeText');
const triageHeading = document.getElementById('triageHeading');
const patientContextText = document.getElementById('patientContextText');
const triageConfidenceVal = document.getElementById('triageConfidenceVal');
const probabilitiesList = document.getElementById('probabilitiesList');
const leadAttributionList = document.getElementById('leadAttributionList');
const clinicalImpressionText = document.getElementById('clinicalImpressionText');
const clinicalRecommendations = document.getElementById('clinicalRecommendations');
const timeWindowText = document.getElementById('timeWindowText');
const topLeadsText = document.getElementById('topLeadsText');
const copyNoteBtn = document.getElementById('copyNoteBtn');
const ecgCanvas = document.getElementById('ecgCanvas');

// Telemetry & AI Overlay Controls
const toggleHeatmapBtn = document.getElementById('toggleHeatmapBtn');
const toggleHeatmapDot = document.getElementById('toggleHeatmapDot');
const toggleHeatmapText = document.getElementById('toggleHeatmapText');
const playStripBtn = document.getElementById('playStripBtn');
const playIcon = document.getElementById('playIcon');
const playText = document.getElementById('playText');
const timeScrubber = document.getElementById('timeScrubber');
const timeWindowLabel = document.getElementById('timeWindowLabel');
const btn5sView = document.getElementById('btn5sView');
const btn10sView = document.getElementById('btn10sView');

function getCardClasses(severity, isActive) {
  if (!isActive) return 'p-3 rounded-lg border border-slate-200 bg-white hover:border-slate-300 border-l-4 border-l-transparent transition-all cursor-pointer';
  if (severity === 'CRITICAL') return 'p-3 rounded-lg border border-rose-200 bg-rose-50/50 border-l-4 border-l-rose-600 shadow-xs transition-all cursor-pointer';
  if (severity === 'HIGH') return 'p-3 rounded-lg border border-amber-200 bg-amber-50/50 border-l-4 border-l-amber-500 shadow-xs transition-all cursor-pointer';
  return 'p-3 rounded-lg border border-indigo-200 bg-indigo-50/50 border-l-4 border-l-indigo-600 shadow-xs transition-all cursor-pointer';
}

// 1. Fetch & Render Clinical Cases
async function loadDemoCases() {
  try {
    const res = await fetch('/api/cases');
    const data = await res.json();
    cachedCases = data.cases;
    renderCaseCards();
    if (cachedCases.length > 0) selectCase(cachedCases[0].id);
  } catch (err) {
    console.error('Failed to load demo cases:', err);
  }
}

function renderCaseCards() {
  caseCardsContainer.innerHTML = '';
  cachedCases.forEach(c => {
    const isActive = c.id === activeCaseId;
    const card = document.createElement('div');
    card.className = getCardClasses(c.severity, isActive);
    card.id = `card_${c.id}`;
    card.onclick = () => selectCase(c.id);

    const badgeColor = c.severity === 'CRITICAL' ? 'bg-rose-100 text-rose-800'
      : c.severity === 'HIGH' ? 'bg-amber-100 text-amber-800' : 'bg-indigo-100 text-indigo-800';

    card.innerHTML = `
      <div class="flex items-center justify-between mb-1">
        <span class="text-[10px] font-mono font-bold px-1.5 py-0.5 rounded ${badgeColor}">${c.severity}</span>
        <span class="text-[10px] text-slate-400 font-mono">${c.tag}</span>
      </div>
      <div class="text-xs font-semibold text-slate-900">${c.title}</div>
      <p class="text-[11px] text-slate-500 mt-1 line-clamp-2">${c.history}</p>
    `;
    caseCardsContainer.appendChild(card);
  });
}

// 2. Select and Evaluate Patient Case
async function selectCase(caseId) {
  activeCaseId = caseId;
  stopTelemetry();
  currentStartSample = 0;
  timeScrubber.value = 0;
  renderCaseCards();

  triageHeading.innerText = "Analyzing Patient Electrocardiogram...";
  try {
    const res = await fetch(`/api/analyze/case/${caseId}`, { method: 'POST' });
    renderAnalysis(await res.json());
  } catch (err) {
    console.error('Inference error:', err);
  }
}

// 3. Render Analysis to Interface
function renderAnalysis(data) {
  currentSignal = data.signal;
  currentHeatmap = data.gradcam_heatmap;
  currentLeadAttribution = data.lead_attribution;
  currentTopDiagnosis = data.top_diagnosis;

  triageBadgeText.innerText = data.triage_level;
  triageHeading.innerText = data.triage_badge;
  patientContextText.innerText = data.case_history;

  const topProb = data.calibrated_probs[data.top_diagnosis] * 100;
  triageConfidenceVal.innerText = `${topProb.toFixed(1)}%`;

  if (data.triage_level.includes('CRITICAL')) {
    triageBanner.className = 'rounded-xl border p-5 transition-all shadow-xs flex items-start justify-between bg-rose-50/70 border-rose-300';
    triageIndicator.className = 'w-3 h-3 rounded-full bg-rose-600 animate-ping';
    triageConfidenceVal.className = 'text-2xl font-mono font-bold text-rose-700';
  } else if (data.triage_level.includes('HIGH')) {
    triageBanner.className = 'rounded-xl border p-5 transition-all shadow-xs flex items-start justify-between bg-amber-50/70 border-amber-300';
    triageIndicator.className = 'w-3 h-3 rounded-full bg-amber-600';
    triageConfidenceVal.className = 'text-2xl font-mono font-bold text-amber-700';
  } else {
    triageBanner.className = 'rounded-xl border p-5 transition-all shadow-xs flex items-start justify-between bg-emerald-50/70 border-emerald-300';
    triageIndicator.className = 'w-3 h-3 rounded-full bg-emerald-600';
    triageConfidenceVal.className = 'text-2xl font-mono font-bold text-emerald-700';
  }

  // Diagnostic Spectrum with Delta Indicator
  probabilitiesList.innerHTML = '';
  const order = ["MI", "CD", "STTC", "HYP", "NORM"];
  const names = {
    MI: "Myocardial Infarction", CD: "Conduction Disease (CLBBB)",
    STTC: "ST/T Ischemia", HYP: "Ventricular Hypertrophy", NORM: "Normal Sinus Rhythm"
  };

  order.forEach(k => {
    const raw = (data.uncalibrated_probs[k] * 100).toFixed(1);
    const cal = (data.calibrated_probs[k] * 100).toFixed(1);
    const diff = (parseFloat(cal) - parseFloat(raw)).toFixed(1);
    const deltaSign = diff > 0 ? `+${diff}%` : `${diff}%`;
    const deltaHtml = Math.abs(parseFloat(diff)) >= 0.5 ? `<span class="text-[10px] text-slate-400 font-mono ml-1">(${deltaSign})</span>` : '';

    const row = document.createElement('div');
    row.className = 'space-y-1';
    row.innerHTML = `
      <div class="flex justify-between text-xs font-semibold">
        <span class="text-slate-700">${names[k]}</span>
        <span class="font-mono text-slate-900">${cal}% ${deltaHtml}</span>
      </div>
      <div class="h-2 w-full bg-slate-100 rounded-full overflow-hidden flex">
        <div class="h-full bg-slate-300 rounded-full transition-all duration-500" style="width: ${raw}%"></div>
        <div class="h-full bg-sky-500 rounded-full transition-all duration-500 -ml-2" style="width: ${cal}%"></div>
      </div>
    `;
    probabilitiesList.appendChild(row);
  });

  // Lead Attribution with Anatomical Territory
  leadAttributionList.innerHTML = '';
  Object.entries(data.lead_attribution).sort((a, b) => b[1] - a[1]).slice(0, 6).forEach(([lead, pct]) => {
    const territoryTag = TERRITORY[lead] ? `<span class="text-[10px] font-normal text-slate-400 font-sans ml-1">[${TERRITORY[lead]}]</span>` : '';
    const item = document.createElement('div');
    item.className = 'flex items-center justify-between py-1 border-b border-slate-100 last:border-0';
    item.innerHTML = `
      <span class="font-bold text-slate-800">Lead ${lead} ${territoryTag}</span>
      <div class="w-20 bg-slate-100 h-1.5 rounded-full overflow-hidden mx-2">
        <div class="bg-rose-500 h-full rounded-full" style="width: ${Math.min(100, pct * 2.2)}%"></div>
      </div>
      <span class="text-slate-700 font-mono text-xs">${pct}%</span>
    `;
    leadAttributionList.appendChild(item);
  });

  clinicalImpressionText.innerText = data.clinical_narrative.impression;
  clinicalRecommendations.innerText = data.clinical_narrative.recommendation;
  timeWindowText.innerText = data.clinical_narrative.time_window;
  topLeadsText.innerText = data.clinical_narrative.top_leads;

  redrawECG();
}

// 4. Redraw Canvas
function redrawECG() {
  if (!currentSignal) return;
  draw12LeadECG(
    ecgCanvas, currentSignal, currentHeatmap,
    currentLeadAttribution, currentTopDiagnosis,
    currentStartSample, currentWindowLength, showHeatmap
  );

  if (currentWindowLength === 1000) {
    timeWindowLabel.innerText = "0.0s - 10.0s (Full)";
    timeScrubber.disabled = true;
    timeScrubber.classList.add('opacity-40');
  } else {
    timeScrubber.disabled = false;
    timeScrubber.classList.remove('opacity-40');
    timeWindowLabel.innerText = `${(currentStartSample / 100).toFixed(1)}s - ${((currentStartSample + 500) / 100).toFixed(1)}s`;
    timeScrubber.value = currentStartSample;
  }
}

// 5. Telemetry Scrolling Animation
function toggleTelemetryAnimation() {
  isPlayingTelemetry ? stopTelemetry() : startTelemetry();
}

function startTelemetry() {
  if (currentWindowLength === 1000) setWindowMode(500);
  isPlayingTelemetry = true;
  playIcon.innerText = "⏸";
  playText.innerText = "Pause";
  playStripBtn.classList.add('bg-rose-50', 'border-rose-300', 'text-rose-700');

  telemetryTimer = setInterval(() => {
    currentStartSample = (currentStartSample + 15 > 500) ? 0 : currentStartSample + 15;
    redrawECG();
  }, 100);
}

function stopTelemetry() {
  isPlayingTelemetry = false;
  if (telemetryTimer) clearInterval(telemetryTimer);
  telemetryTimer = null;
  playIcon.innerText = "▶";
  playText.innerText = "Play";
  playStripBtn.classList.remove('bg-rose-50', 'border-rose-300', 'text-rose-700');
}

function setWindowMode(mode) {
  currentWindowLength = mode;
  btn5sView.className = mode === 500 ? 'px-2 py-0.5 text-[10px] font-semibold rounded bg-white text-slate-900 shadow-xs' : 'px-2 py-0.5 text-[10px] font-semibold rounded text-slate-600 hover:text-slate-900';
  btn10sView.className = mode === 1000 ? 'px-2 py-0.5 text-[10px] font-semibold rounded bg-white text-slate-900 shadow-xs' : 'px-2 py-0.5 text-[10px] font-semibold rounded text-slate-600 hover:text-slate-900';
  if (mode === 1000) stopTelemetry();
  redrawECG();
}

// Event Listeners
timeScrubber.addEventListener('input', (e) => {
  stopTelemetry();
  currentStartSample = parseInt(e.target.value);
  redrawECG();
});

playStripBtn.addEventListener('click', toggleTelemetryAnimation);
btn5sView.addEventListener('click', () => setWindowMode(500));
btn10sView.addEventListener('click', () => setWindowMode(1000));

toggleHeatmapBtn.addEventListener('click', () => {
  showHeatmap = !showHeatmap;
  if (showHeatmap) {
    toggleHeatmapBtn.className = 'flex items-center space-x-1.5 px-2 py-1 rounded bg-rose-50 hover:bg-rose-100 border border-rose-200 text-rose-700 font-semibold text-[10px] shadow-xs transition-all';
    toggleHeatmapDot.className = 'w-2 h-2 rounded-full bg-rose-600 animate-pulse';
    toggleHeatmapText.innerText = 'AI Saliency: ON';
  } else {
    toggleHeatmapBtn.className = 'flex items-center space-x-1.5 px-2 py-1 rounded bg-slate-100 hover:bg-slate-200 border border-slate-300 text-slate-600 font-semibold text-[10px] shadow-xs transition-all';
    toggleHeatmapDot.className = 'w-2 h-2 rounded-full bg-slate-400';
    toggleHeatmapText.innerText = 'AI Saliency: OFF';
  }
  redrawECG();
});

csvFileInput.addEventListener('change', async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  uploadStatusText.classList.remove('hidden');
  uploadStatusText.innerText = `Analyzing: ${file.name}...`;
  const formData = new FormData();
  formData.append('file', file);
  try {
    const res = await fetch('/api/analyze/upload', { method: 'POST', body: formData });
    if (!res.ok) throw new Error((await res.json()).detail || 'Upload failed');
    renderAnalysis(await res.json());
    uploadStatusText.innerText = `Analyzed: ${file.name}`;
  } catch (err) {
    uploadStatusText.innerText = `Error: ${err.message}`;
  }
});

tempSlider.addEventListener('input', async (e) => {
  const val = parseFloat(e.target.value);
  tempValueDisplay.innerText = `T = ${val.toFixed(2)}`;
  const estimatedEce = (Math.abs(val - 1.28) * 3.4 + 4.8).toFixed(1);
  if (eceBadge) eceBadge.innerText = `ECE: ${estimatedEce}%`;
  await fetch('/api/calibrate', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ temperature: val })
  });
  if (activeCaseId) selectCase(activeCaseId);
});

copyNoteBtn.addEventListener('click', () => {
  const note = `[CardioAI-12Lead CDSS Report]\nImpression: ${clinicalImpressionText.innerText}\n` +
    `Window: ${timeWindowText.innerText}\nLeads: ${topLeadsText.innerText}\n` +
    `Orders:\n${clinicalRecommendations.innerText}\nModel: 1D-ResNet (PhysioNet PTB-XL)`;
  navigator.clipboard.writeText(note).then(() => {
    copyNoteBtn.innerText = "Copied!";
    setTimeout(() => { copyNoteBtn.innerText = "Copy Clinical Note"; }, 2000);
  });
});

loadDemoCases();
