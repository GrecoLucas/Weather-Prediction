// ─── DOM refs ─────────────────────────────────────────────────────────────
const statusPill       = document.getElementById("status-pill");
const level5Form       = document.getElementById("level5-form");
const datasetPathInput = document.getElementById("dataset-path");
const locationSelect   = document.getElementById("location-select");
const targetDateInput  = document.getElementById("target-date");
const refreshOptionsBtn= document.getElementById("refresh-options-btn");
const datasetRange     = document.getElementById("dataset-range");
const kpiGrid          = document.getElementById("kpi-grid");
const predictionsBody  = document.getElementById("predictions-body");
const targetTabs       = document.getElementById("target-tabs");
const bars             = document.getElementById("bars");
const splitGrid        = document.getElementById("split-grid");
const scoreFormula     = document.getElementById("score-formula");
const metricsReport    = document.getElementById("metrics-report");
const predictionChip   = document.getElementById("prediction-chip");
const runLog           = document.getElementById("run-log");
const kpiTemplate      = document.getElementById("kpi-template");

// State
let cachedReport   = null;   // from /api/level5-report
let stepMAEData    = {};     // { target: [{step, mae}] }
let activeTarget   = null;   // for tab-based chart switching

// ─── Logging ──────────────────────────────────────────────────────────────
function log(message, type = "info") {
  const time   = new Date().toLocaleTimeString();
  const entry  = document.createElement("p");
  entry.className = `log-entry log-${type}`;
  entry.innerHTML = `<span class="log-time">${time}</span>${message}`;
  runLog.prepend(entry);
}

function setStatus(message) {
  statusPill.textContent = message;
}

// ─── Formatters ───────────────────────────────────────────────────────────
function fmt(v, decimals = 4) {
  if (v == null || Number.isNaN(Number(v))) return "–";
  return Number(v).toFixed(decimals);
}

// ─── KPI rendering ────────────────────────────────────────────────────────
function renderKPIs(data) {
  kpiGrid.innerHTML = "";

  const score     = data.score        ?? cachedReport?.score;
  const globalMAE = data.globalMAE    ?? cachedReport?.globalMAE;
  const nTargets  = data.nTargets     ?? cachedReport?.nTargets  ?? 2;
  const total     = data.totalLabels  ?? 17;

  const perTarget = data.perTargetMAE ?? {};
  const tempMAE   = perTarget["target_temperature_2m"] ?? cachedReport?.perTarget?.["target_temperature_2m"]?.mae;
  const rainMAE   = perTarget["target_rain"]           ?? cachedReport?.perTarget?.["target_rain"]?.mae;

  const items = [
    { label: "Competition Score",  value: score     != null ? fmt(score, 2)     : "–", extra: `Formula: 2.5/(1+MAE) × (${nTargets}/${total}) × 100` },
    { label: "Global MAE",        value: globalMAE != null ? fmt(globalMAE, 4) : "–", extra: "Mean across all targets" },
    { label: "Temperature MAE",   value: tempMAE   != null ? fmt(tempMAE, 4)   : "–", extra: "Temperature 2m (°C)" },
    { label: "Rain MAE",          value: rainMAE   != null ? fmt(rainMAE, 4)   : "–", extra: "Rain (mm), Tweedie XGBoost" },
  ];

  for (const item of items) {
    const node = kpiTemplate.content.cloneNode(true);
    node.querySelector(".kpi-label").textContent = item.label;
    node.querySelector(".kpi-value").textContent = item.value;
    node.querySelector(".kpi-extra").textContent = item.extra;
    kpiGrid.appendChild(node);
  }
}

// ─── Predictions table ────────────────────────────────────────────────────
function renderPredictionsTable(prediction) {
  predictionsBody.innerHTML = "";

  const perTargetMAE = prediction.perTargetMAE ?? {};

  for (const p of prediction.predictions) {
    const tr = document.createElement("tr");
    const mae = perTargetMAE[p.target];

    // Colour the actual cell based on how close the prediction is
    let actualCell = "–";
    if (p.actual != null) {
      const err = Math.abs(p.predicted - p.actual);
      const withinMAE = mae != null && err <= mae;
      const color = withinMAE ? "color:#1a7f37; font-weight:600" : "color:#a0522d; font-weight:600";
      actualCell = `<span style="${color}">${fmt(p.actual, 3)}</span>`;
    }

    tr.innerHTML = `
      <td><strong>${p.label}</strong></td>
      <td>${fmt(p.predicted, 3)}</td>
      <td>${actualCell}</td>
      <td>${p.unit}</td>
      <td>${mae != null ? fmt(mae, 4) : "–"}</td>
    `;
    predictionsBody.appendChild(tr);
  }

  if (!prediction.predictions.length) {
    predictionsBody.innerHTML = `<tr><td colspan="5">No predictions available.</td></tr>`;
  }
}

// ─── Step-MAE bar chart ───────────────────────────────────────────────────
function renderTargetTabs(keys) {
  targetTabs.innerHTML = "";
  for (const key of keys) {
    const btn = document.createElement("button");
    btn.className = "tab-btn" + (key === activeTarget ? " active" : "");
    btn.type = "button";
    btn.dataset.target = key;
    // Pretty label
    const label = key.replace("target_", "").replaceAll("_", " ");
    btn.textContent = label.charAt(0).toUpperCase() + label.slice(1);
    btn.addEventListener("click", () => {
      activeTarget = key;
      renderTargetTabs(keys);
      renderStepBars(stepMAEData[key] || []);
    });
    targetTabs.appendChild(btn);
  }
}

function renderStepBars(stepRows) {
  bars.innerHTML = "";

  if (!Array.isArray(stepRows) || !stepRows.length) {
    bars.innerHTML = '<p class="hint">No step-level MAE data available. Run the pipeline first.</p>';
    return;
  }

  const maxMAE = Math.max(...stepRows.map((r) => Number(r.mae || 0)), 0.001);

  for (const row of stepRows) {
    const width = Math.max(2, (Number(row.mae) / maxMAE) * 100);
    const dateLabel = row.date ? `<span class="bar-date">· ${row.date}</span>` : "";
    const bar   = document.createElement("div");
    bar.className = "bar-row";
    bar.innerHTML = `
      <span class="bar-label">Step ${row.step} ${dateLabel}</span>
      <div class="bar-track"><div class="bar-fill" style="width:${width}%; background:var(--primary)"></div></div>
      <strong>${fmt(row.mae, 4)}</strong>
    `;
    bars.appendChild(bar);
  }
}

function renderStepChart(stepMAE) {
  stepMAEData = stepMAE || {};
  const keys = Object.keys(stepMAEData);
  if (!keys.length) {
    bars.innerHTML = '<p class="hint">No step-level MAE data available.</p>';
    return;
  }
  if (!activeTarget || !stepMAEData[activeTarget]) {
    activeTarget = keys[0];
  }
  renderTargetTabs(keys);
  renderStepBars(stepMAEData[activeTarget]);
}

// ─── Details card ─────────────────────────────────────────────────────────
function renderDetails(prediction) {
  splitGrid.innerHTML = "";

  const fromCache     = prediction.fromCache;
  const cacheStatus   = prediction.cacheStatus ?? {};
  const cacheEntries  = Object.entries(cacheStatus)
    .map(([t, c]) => `${t.replace("target_", "")}: ${c ? "✓ cached" : "newly trained"}`)
    .join("<br>");

  const cards = [
    {
      title: "Prediction",
      stats: {
        location:          prediction.location,
        reference_date:    prediction.targetDate,
        input_time_used:   prediction.inputTime || "–",
        training_samples:  (prediction.trainingSamples || 0).toLocaleString(),
      },
    },
    {
      title: "Models",
      stats: {
        temperature_model: "LightGBM (LGBM)",
        rain_model:        "XGBoost Tweedie",
        cache_status:      fromCache ? "All loaded from cache" : "Newly trained",
      },
    },
  ];

  for (const cardData of cards) {
    const card = document.createElement("article");
    card.className = "split-card";
    card.innerHTML = `
      <h4>${cardData.title}</h4>
      ${Object.entries(cardData.stats)
        .map(([key, value]) => `<p class="split-stat"><strong>${key.replaceAll("_", " ")}:</strong> ${value}</p>`)
        .join("")}
    `;
    splitGrid.appendChild(card);
  }

  // Cache detail card
  if (Object.keys(cacheStatus).length) {
    const card = document.createElement("article");
    card.className = "split-card";
    card.innerHTML = `<h4>Per-Model Cache</h4><p class="split-stat" style="line-height:1.8">${cacheEntries}</p>`;
    splitGrid.appendChild(card);
  }

  const score     = prediction.score;
  const globalMAE = prediction.globalMAE;
  const nTargets  = prediction.nTargets  ?? 2;
  const total     = prediction.totalLabels ?? 17;

  scoreFormula.innerHTML = score != null
    ? `Score = 2.5 / (1 + ${fmt(globalMAE, 4)}) × (${nTargets} / ${total}) × 100 = <strong>${fmt(score, 2)}</strong>`
    : "Score formula: 2.5 / (1 + Mean_MAE) × (N_predicted / 17) × 100\nRun the pipeline to compute MAE.";

  predictionChip.textContent = `Predicted ${prediction.predictions.length} variable${prediction.predictions.length !== 1 ? "s" : ""}`;
  predictionChip.className = "result-chip chip-rain";
}

// ─── Load options ─────────────────────────────────────────────────────────
async function loadOptions() {
  setStatus("Loading Level 5 options…");
  const response = await fetch("/api/level5-options");
  const payload  = await response.json();
  if (!response.ok || !payload.ok) throw new Error(payload.error || "Could not load Level 5 options.");

  datasetPathInput.value = payload.datasetPath;

  locationSelect.innerHTML = payload.locations
    .map((loc) => `<option value="${loc}">${loc}</option>`)
    .join("");

  datasetRange.textContent =
    `Available data from ${payload.minDate} to ${payload.maxDate}. ` +
    `${payload.allCached ? "✓ Models cached." : "Models not yet cached – first prediction will train them."}`;

  // Pre-set date near the end of training period
  if (!targetDateInput.value && payload.maxDate) {
    // Subtract 30 days to give margin
    const d = new Date(payload.maxDate);
    d.setDate(d.getDate() - 30);
    targetDateInput.value = d.toISOString().slice(0, 10);
  }

  setStatus("Level 5 options loaded");
  log("Loaded locations for Level 5 meteorology forecasting.");
}

// ─── Load cached metrics report ───────────────────────────────────────────
async function loadReport() {
  try {
    const response = await fetch("/api/level5-report");
    const payload  = await response.json();
    if (!response.ok || !payload.ok) return;

    cachedReport = payload;

    if (payload.reportText) {
      metricsReport.textContent = payload.reportText;
    }

    // Render KPIs from cached metrics (before any prediction)
    renderKPIs({});

    // Step chart
    if (payload.stepMAE) {
      renderStepChart(payload.stepMAE);
    }

    log("Loaded pre-computed walk-forward metrics.");
  } catch {
    // Non-fatal: report just won't show until pipeline is run
  }
}

// ─── Predict handler ──────────────────────────────────────────────────────
async function handlePredict(event) {
  event.preventDefault();

  const payload = {
    datasetPath: datasetPathInput.value.trim(),
    location:    locationSelect.value,
    targetDate:  targetDateInput.value,
  };

  if (!payload.targetDate) {
    log("Please select a reference date.", "error");
    return;
  }

  setStatus("Running Level 5 forecast…");
  log(`Predicting next-day values for ${payload.location} after ${payload.targetDate}…`);

  const response = await fetch("/api/predict-meteorology", {
    method:  "POST",
    headers: { "Content-Type": "application/json" },
    body:    JSON.stringify(payload),
  });
  const result = await response.json();
  if (!response.ok || !result.ok) throw new Error(result.error || "Prediction failed.");

  const p = result.prediction;

  renderKPIs(p);
  renderPredictionsTable(p);
  renderDetails(p);

  const cacheMsg = p.fromCache ? " (models loaded from cache)" : " (models trained fresh)";
  setStatus(`Forecast ready in ${p.durationMs} ms${cacheMsg}`);
  log(`Forecast complete for ${p.location}. Score: ${p.score != null ? fmt(p.score, 2) : "N/A"}. Duration: ${p.durationMs} ms.`);
}

// ─── Event listeners ──────────────────────────────────────────────────────
level5Form.addEventListener("submit", (event) => {
  handlePredict(event).catch((error) => {
    setStatus("Prediction failed");
    log(error.message, "error");
  });
});

refreshOptionsBtn.addEventListener("click", () => {
  loadOptions().catch((error) => {
    setStatus("Options failed");
    log(error.message, "error");
  });
});

// ─── Startup ─────────────────────────────────────────────────────────────
Promise.all([
  loadOptions().catch((error) => {
    setStatus("Startup failed");
    log(error.message, "error");
  }),
  loadReport(),
]);
