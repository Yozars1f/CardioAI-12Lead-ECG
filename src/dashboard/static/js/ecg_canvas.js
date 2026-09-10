/* ==========================================================================
   CardioAI-12Lead CDSS: 12-Lead Canvas & 1D Grad-CAM Saliency Renderer
   Strict Rule: File under 300 lines.
   ========================================================================== */

const LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"];

/**
 * Renders 12 standard leads across a 6-row by 2-column clinical layout.
 * Features row-isolated clipping (prevents cross-lead bleeding), true isoelectric 
 * 0.0mV baseline alignment, and calibrated clinical voltage scaling.
 */
function draw12LeadECG(canvas, signal, heatmap, leadAttribution, topDiagnosis, startSample = 0, windowLength = 500, showHeatmap = true) {
  if (!canvas || !signal || signal.length < 12) return;
  const ctx = canvas.getContext('2d');
  const width = canvas.width;
  const height = canvas.height;
  
  ctx.clearRect(0, 0, width, height);

  const rows = 6;
  const cols = 2;
  const rowHeight = height / rows;
  const colWidth = width / cols;

  const totalPoints = signal[0].length;
  const winLen = windowLength === 1000 ? totalPoints : Math.min(500, totalPoints);
  const startPt = windowLength === 1000 ? 0 : Math.max(0, Math.min(startSample, totalPoints - winLen));

  // Determine peak voltage across visible leads for clinical adaptive gain
  let maxPTP = 1.0;
  for (let i = 0; i < 12; i++) {
    let minV = Infinity, maxV = -Infinity;
    for (let p = 0; p < winLen; p++) {
      const v = signal[i][startPt + p] || 0;
      if (v < minV) minV = v;
      if (v > maxV) maxV = v;
    }
    const ptp = maxV - minV;
    if (ptp > maxPTP) maxPTP = ptp;
  }

  // Clinical standard: Half-gain (5mm/mV) for massive hypertrophy/block voltages > 2.8mV
  const maxAllowablePx = (rowHeight * 0.44);
  const scaleY = maxPTP > 2.8 ? (maxAllowablePx / (maxPTP * 0.55)) : (maxAllowablePx / 1.5);

  // Identify Top-3 pathology-driving leads (strictly >= 10.0% attribution)
  let topLeadNames = new Set();
  if (showHeatmap && leadAttribution && topDiagnosis !== 'NORM') {
    const sorted = Object.entries(leadAttribution)
      .filter(([_, pct]) => pct >= 10.0)
      .sort((a, b) => b[1] - a[1]);
    topLeadNames = new Set(sorted.slice(0, 3).map(x => x[0]));
  }

  for (let i = 0; i < 12; i++) {
    const leadName = LEAD_NAMES[i];
    const r = i % 6;
    const c = i < 6 ? 0 : 1;

    const xStart = c * colWidth;
    const yStart = r * rowHeight;
    const yCenter = yStart + (rowHeight / 2);

    const leadAttr = leadAttribution && leadAttribution[leadName] !== undefined
      ? leadAttribution[leadName]
      : 0;
    const isLeadAffected = showHeatmap && topLeadNames.has(leadName);

    // 1. Draw Isoelectric Reference Baseline
    ctx.strokeStyle = "rgba(226, 232, 240, 0.9)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(xStart, yCenter);
    ctx.lineTo(xStart + colWidth, yCenter);
    ctx.stroke();

    // 2. Draw Lead Label
    ctx.fillStyle = isLeadAffected ? "#B91C1C" : "#0F172A";
    ctx.font = isLeadAffected ? "bold 13px 'JetBrains Mono', monospace" : "600 13px 'JetBrains Mono', monospace";
    const labelText = isLeadAffected ? `${leadName} (${leadAttr}%)` : leadName;
    ctx.fillText(labelText, xStart + 12, yStart + 22);

    // 3. Isolated Clipped Waveform Render (Prevents lead overlap)
    ctx.save();
    ctx.beginPath();
    ctx.rect(xStart + 70, yStart + 2, colWidth - 75, rowHeight - 4);
    ctx.clip();

    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';

    const leadSig = signal[i];

    for (let pt = 0; pt < winLen - 1; pt++) {
      const idx1 = startPt + pt;
      const idx2 = startPt + pt + 1;
      if (idx2 >= totalPoints) break;

      const x1 = xStart + 80 + (pt / (winLen - 1)) * (colWidth - 95);
      const y1 = yCenter - (leadSig[idx1] * scaleY);

      const x2 = xStart + 80 + ((pt + 1) / (winLen - 1)) * (colWidth - 95);
      const y2 = yCenter - (leadSig[idx2] * scaleY);

      const saliency = (showHeatmap && heatmap && topDiagnosis !== 'NORM') ? (heatmap[idx1] || 0) : 0;

      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);

      // Highlight focal pathological segments (peak voltage, ST shifts) rather than entire strip
      if (isLeadAffected && saliency > 0.52) {
        ctx.strokeStyle = "#DC2626";
        ctx.lineWidth = 2.4;
      } else {
        ctx.strokeStyle = "#0F172A";
        ctx.lineWidth = 1.35;
      }
      ctx.stroke();
    }

    ctx.restore();
  }
}
