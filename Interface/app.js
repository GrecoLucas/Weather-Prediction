const APP_STATE = {
  report: null,
  selectedModel: null,
  selectedMetric: "f1",
};

const METRIC_LABELS = {
  f1: "F1",
  precision: "Precision",
  recall: "Recall",
};

const METRIC_COLORS = ["#216869", "#d66a2d", "#2b8a3e", "#6f8f72", "#9d5b34"];

const DEMO_REPORT = `RAIN PREDICTION - MODEL COMPARISON REPORT
============================================================

Dataset Split Sizes:
	Train      : 126489
	Validation : 15811
	Test       : 15812


============================================================
MODEL: LightGBM
============================================================
Chosen Hyperparameters : {'n_estimators': 200, 'max_depth': 8, 'learning_rate': 0.2, 'num_leaves': 63}

	--- Train Set ---
	F1-Score  : 0.8918
	Precision : 0.8097
	Recall    : 0.9923
	Confusion Matrix:
[[95506  5858]
 [  194 24931]]

	--- Validation Set ---
	F1-Score  : 0.8014
	Precision : 0.7229
	Recall    : 0.8990
	Confusion Matrix:
[[11589  1082]
 [  317  2823]]

	--- Test Set ---
	F1-Score  : 0.7985
	Precision : 0.7198
	Recall    : 0.8965
	Confusion Matrix:
[[11575  1096]
 [  325  2816]]


============================================================
MODEL: XGBoost
============================================================
Chosen Hyperparameters : {'n_estimators': 400, 'learning_rate': 0.05, 'max_depth': 5, 'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_lambda': 1, 'reg_alpha': 0.1, 'early_stopping_rounds': 30}

	--- Train Set ---
	F1-Score  : 0.8025
	Precision : 0.6928
	Recall    : 0.9534
	Confusion Matrix:
[[90744 10620]
 [ 1170 23955]]

	--- Validation Set ---
	F1-Score  : 0.7782
	Precision : 0.6712
	Recall    : 0.9258
	Confusion Matrix:
[[11247  1424]
 [  233  2907]]

	--- Test Set ---
	F1-Score  : 0.7755
	Precision : 0.6671
	Recall    : 0.9258
	Confusion Matrix:
[[11220  1451]
 [  233  2908]]


============================================================
MODEL: RandomForest
============================================================
Chosen Hyperparameters : {'n_estimators': 200, 'max_depth': 10, 'min_samples_leaf': 10}

	--- Train Set ---
	F1-Score  : 0.7727
	Precision : 0.6490
	Recall    : 0.9548
	Confusion Matrix:
[[88388 12976]
 [ 1135 23990]]

	--- Validation Set ---
	F1-Score  : 0.7486
	Precision : 0.6265
	Recall    : 0.9299
	Confusion Matrix:
[[10930  1741]
 [  220  2920]]

	--- Test Set ---
	F1-Score  : 0.7435
	Precision : 0.6221
	Recall    : 0.9239
	Confusion Matrix:
[[10908  1763]
 [  239  2902]]


============================================================
MODEL COMPARISON SUMMARY (ranked by Validation F1)
============================================================
Rank  Model          Val F1    Test F1   Val Prec    Val Rec
------------------------------------------------------------
1     LightGBM       0.8014    0.7985    0.7229      0.8990
2     XGBoost        0.7782    0.7755    0.6712      0.9258
3     RandomForest   0.7486    0.7435    0.6265      0.9299

============================================================
BEST MODEL: LightGBM
	Validation F1          : 0.8014
	Test F1                : 0.7985
	Chosen Hyperparameters : {'n_estimators': 200, 'max_depth': 8, 'learning_rate': 0.2, 'num_leaves': 63}
============================================================`;

const els = {
  runForm: document.getElementById("run-form"),
  runBtn: document.getElementById("run-btn"),
  loadDemoBtn: document.getElementById("load-demo-btn"),
  metricsUpload: document.getElementById("metrics-upload"),
  datasetPath: document.getElementById("dataset-path"),
  modelFamily: document.getElementById("model-family"),
  profile: document.getElementById("profile"),
  statusPill: document.getElementById("status-pill"),
  kpiGrid: document.getElementById("kpi-grid"),
  kpiTemplate: document.getElementById("kpi-template"),
  rankingBody: document.getElementById("ranking-body"),
  metricSelect: document.getElementById("metric-select"),
  bars: document.getElementById("bars"),
  modelSelect: document.getElementById("model-select"),
  splitGrid: document.getElementById("split-grid"),
  confusionMatrix: document.getElementById("confusion-matrix"),
  hyperparams: document.getElementById("hyperparams"),
  runLog: document.getElementById("run-log"),
};
const statusPill = document.getElementById("status-pill");
const predictionForm = document.getElementById("prediction-form");
const datasetPathInput = document.getElementById("dataset-path");
const selectedDateInput = document.getElementById("selected-date");
const locationSelect = document.getElementById("location-select");
const modelFamilySelect = document.getElementById("model-family");
const refreshOptionsBtn = document.getElementById("refresh-options-btn");
const datasetRange = document.getElementById("dataset-range");
const kpiGrid = document.getElementById("kpi-grid");
const hourlyBody = document.getElementById("hourly-body");
const bars = document.getElementById("bars");
const splitGrid = document.getElementById("split-grid");
const predictionNotes = document.getElementById("prediction-notes");
const hyperparams = document.getElementById("hyperparams");
const predictionChip = document.getElementById("prediction-chip");
const runLog = document.getElementById("run-log");
const kpiTemplate = document.getElementById("kpi-template");

let currentPrediction = null;

function log(message, type = "info") {
  const time = new Date().toLocaleTimeString();
  const entry = document.createElement("p");
  entry.className = `log-entry log-${type}`;
  entry.innerHTML = `<span class="log-time">${time}</span>${message}`;
  runLog.prepend(entry);
}

function setStatus(message) {
  statusPill.textContent = message;
}

function formatPercent(value) {
  return `${(value * 100).toFixed(1)}%`;
}

function formatMetric(value) {
  return typeof value === "number" && Number.isFinite(value)
    ? value.toFixed(4)
    : "-";
}

function renderKPIs(prediction) {
  kpiGrid.innerHTML = "";

  const items = [
    {
      label: "Outcome",
      value: prediction.willRain ? "Rain" : "No rain",
      extra: `${prediction.location} on ${prediction.selectedDate}`,
    },
    {
      label: "Confidence",
      value: formatPercent(prediction.confidence),
      extra: "Average predicted rain probability",
    },
    {
      label: "Rainy Hours",
      value: `${prediction.rainyHours}/${prediction.totalHours}`,
      extra: "Hours predicted as rainy",
    },
    {
      label: "Observed",
      value: prediction.observedRain ? "Rain seen" : "No rain seen",
      extra: "Based on the rows in the dataset for that day",
    },
  ];

  for (const item of items) {
    const node = kpiTemplate.content.cloneNode(true);
    node.querySelector(".kpi-label").textContent = item.label;
    node.querySelector(".kpi-value").textContent = item.value;
    node.querySelector(".kpi-extra").textContent = item.extra;
    kpiGrid.appendChild(node);
  }
}

function renderHourlyTable(prediction) {
  hourlyBody.innerHTML = "";
  for (const row of prediction.hourly) {
    const tr = document.createElement("tr");
    if (row.predictedRain) tr.classList.add("highlight");
    tr.innerHTML = `
			<td>${new Date(row.time).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}</td>
			<td><span class="badge ${row.predictedRain ? "badge-rain" : "badge-clear"}">${row.predictedRain ? "Rain" : "No rain"}</span></td>
			<td>${formatPercent(row.confidence)}</td>
			<td>${row.observedRain ? "Rain" : "No rain"}</td>
		`;
    hourlyBody.appendChild(tr);
  }
  if (!prediction.hourly.length) {
    hourlyBody.innerHTML =
      '<tr><td colspan="4">No hourly rows found.</td></tr>';
  }
}

function renderBars(prediction) {
  bars.innerHTML = "";
  for (const row of prediction.hourly) {
    const value = Math.max(0.02, row.confidence);
    const bar = document.createElement("div");
    bar.className = "bar-row";
    bar.innerHTML = `
			<span>${new Date(row.time).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}</span>
			<div class="bar-track"><div class="bar-fill" style="width:${value * 100}%; background:${row.predictedRain ? "var(--secondary)" : "var(--accent)"}"></div></div>
			<strong>${formatPercent(row.confidence)}</strong>
		`;
    bars.appendChild(bar);
  }
}

function renderDetails(prediction) {
  splitGrid.innerHTML = "";
  const detailCards = [
    {
      title: "Model",
      stats: {
        name: prediction.modelName,
        training_samples: prediction.trainingSamples,
        global_train_f1: formatMetric(prediction.globalTrainF1),
        global_val_f1: formatMetric(prediction.globalValF1),
        global_test_f1: formatMetric(prediction.globalTestF1),
      },
    },
    {
      title: "Day Summary",
      stats: {
        location: prediction.location,
        selected_date: prediction.selectedDate,
        rainy_hours: prediction.rainyHours,
        total_hours: prediction.totalHours,
      },
    },
  ];

  for (const item of detailCards) {
    const card = document.createElement("article");
    card.className = "split-card";
    card.innerHTML = `
			<h4>${item.title}</h4>
			${Object.entries(item.stats)
        .map(
          ([key, value]) =>
            `<p class="split-stat"><strong>${key.replaceAll("_", " ")}:</strong> ${typeof value === "number" && value <= 1 ? value.toFixed(4) : value}</p>`,
        )
        .join("")}
		`;
    splitGrid.appendChild(card);
  }

  predictionChip.textContent = prediction.willRain
    ? "Rain expected"
    : "No rain expected";
  predictionChip.className = `result-chip ${prediction.willRain ? "chip-rain" : "chip-clear"}`;
  predictionNotes.innerHTML = `
    <p><strong>Method:</strong> the selected model family is trained once on all locations and cached. The prediction request then reuses that cached model for ${prediction.location} on ${prediction.selectedDate}.</p>
    <p><strong>Global quality (fixed per selected model):</strong> train F1 <strong>${formatMetric(prediction.globalTrainF1)}</strong>, validation F1 <strong>${formatMetric(prediction.globalValF1)}</strong>, test F1 <strong>${formatMetric(prediction.globalTestF1)}</strong>.</p>
		<p><strong>Decision rule:</strong> the daily label is "Rain" if at least one hour is predicted as rainy.</p>
		<p><strong>Reality check:</strong> the dataset shows ${prediction.observedRain ? "rain on that day" : "no rain on that day"} for ${prediction.location} on ${prediction.selectedDate}.</p>
	`;
  hyperparams.textContent = JSON.stringify(prediction.chosenParams, null, 2);
}

function renderPrediction(prediction, durationMs) {
  currentPrediction = prediction;
  renderKPIs(prediction);
  renderHourlyTable(prediction);
  renderBars(prediction);
  renderDetails(prediction);
  setStatus(`Prediction ready in ${durationMs} ms`);
  log(
    `Predicted ${prediction.selectedDate} for ${prediction.location} with ${prediction.modelName} (cached).`,
  );
}

async function loadOptions() {
  setStatus("Loading dataset options...");
  const response = await fetch("/api/level1-options");
  const payload = await response.json();
  if (!response.ok || !payload.ok) {
    throw new Error(payload.error || "Could not load dataset options.");
  }

  datasetPathInput.value = payload.datasetPath;
  selectedDateInput.min = payload.minDate || "";
  selectedDateInput.max = payload.maxDate || "";
  selectedDateInput.value = selectedDateInput.value || payload.maxDate || "";
  datasetRange.textContent = `Available data from ${payload.minDate} to ${payload.maxDate}.`;
  locationSelect.innerHTML = payload.locations
    .map((location) => `<option value="${location}">${location}</option>`)
    .join("");
  setStatus("Options loaded");
  log("Loaded available dates and locations.");
}

async function handlePrediction(event) {
  event.preventDefault();
  const payload = {
    datasetPath: datasetPathInput.value.trim(),
    selectedDate: selectedDateInput.value,
    location: locationSelect.value,
    modelFamily: modelFamilySelect.value,
  };

  setStatus("Running rain prediction...");
  log(
    `Predicting ${payload.selectedDate} for ${payload.location} using ${payload.modelFamily}.`,
  );

  const response = await fetch("/api/predict-rain-day", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const result = await response.json();
  if (!response.ok || !result.ok) {
    throw new Error(result.error || "Prediction failed.");
  }

  renderPrediction(result.prediction, result.durationMs);
}

predictionForm.addEventListener("submit", (event) => {
  handlePrediction(event).catch((error) => {
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

loadOptions().catch((error) => {
  setStatus("Startup failed");
  log(error.message, "error");
});
