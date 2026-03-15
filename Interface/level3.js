const statusPill = document.getElementById("status-pill");
const level3Form = document.getElementById("level3-form");
const datasetPathInput = document.getElementById("dataset-path");
const locationSelect = document.getElementById("location-select");
const refreshOptionsBtn = document.getElementById("refresh-options-btn");
const datasetRange = document.getElementById("dataset-range");
const kpiGrid = document.getElementById("kpi-grid");
const hourlyBody = document.getElementById("hourly-body");
const observedSnowHeader = document.getElementById("observed-snow-header");
const bars = document.getElementById("bars");
const splitGrid = document.getElementById("split-grid");
const predictionNotes = document.getElementById("prediction-notes");
const predictionChip = document.getElementById("prediction-chip");
const hyperparams = document.getElementById("hyperparams");
const runLog = document.getElementById("run-log");
const kpiTemplate = document.getElementById("kpi-template");

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
  return `${(Number(value || 0) * 100).toFixed(2)}%`;
}

function formatTemp(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "-";
  }
  return Number(value).toFixed(2);
}

function renderKPIs(prediction) {
  kpiGrid.innerHTML = "";
  const items = [
    {
      label: "District",
      value: prediction.location,
      extra: prediction.modelName,
    },
    {
      label: "Snowfall Hours",
      value: `${prediction.snowyHours}/${prediction.totalHours}`,
      extra: "Predicted hours with snow_fall = 1",
    },
    {
      label: "Snowfall Rate",
      value: formatPercent(prediction.snowyRate),
      extra: "Share of district hours predicted as snowfall",
    },
    {
      label: "Avg Anomaly Score",
      value: `${(Number(prediction.confidence || 0) * 100).toFixed(1)}%`,
      extra: "Relative anomaly intensity (not calibrated probability)",
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
  const showObservedSnow = Boolean(prediction.hasObservedSnow);
  if (observedSnowHeader) {
    observedSnowHeader.style.display = showObservedSnow ? "table-cell" : "none";
  }

  for (const row of prediction.hourly) {
    const tr = document.createElement("tr");
    tr.innerHTML = `
			<td>${new Date(row.time).toLocaleString()}</td>
			<td>${(Number(row.confidence || 0) * 100).toFixed(1)}%</td>
			<td>${formatTemp(row.temperature)}</td>
			<td>${row.humidity == null ? "-" : Number(row.humidity).toFixed(1)}</td>
			<td>${row.rain == null ? "-" : Number(row.rain).toFixed(2)}</td>${showObservedSnow ? `<td>${row.observedSnow === null ? "N/A" : row.observedSnow ? "Yes" : "No"}</td>` : ""}
		`;
    hourlyBody.appendChild(tr);
  }

  if (!prediction.hourly.length) {
    const colSpan = showObservedSnow ? 6 : 5;
    hourlyBody.innerHTML = `<tr><td colspan="${colSpan}">No snowfall hours predicted for this district.</td></tr>`;
  }
}

function renderBars(prediction) {
  bars.innerHTML = "";
  if (
    !Array.isArray(prediction.monthlyCounts) ||
    !prediction.monthlyCounts.length
  ) {
    bars.innerHTML =
      '<p class="hint">No monthly snowfall counts to display.</p>';
    return;
  }

  const maxCount = Math.max(
    ...prediction.monthlyCounts.map((row) => Number(row.count || 0)),
    1,
  );
  for (const row of prediction.monthlyCounts) {
    const width = Math.max(4, (Number(row.count || 0) / maxCount) * 100);
    const bar = document.createElement("div");
    bar.className = "bar-row";
    bar.innerHTML = `
			<span>${row.month}</span>
			<div class="bar-track"><div class="bar-fill" style="width:${width}%; background:var(--secondary)"></div></div>
			<strong>${row.count}</strong>
		`;
    bars.appendChild(bar);
  }
}

function renderDetails(prediction) {
  splitGrid.innerHTML = "";
  const cards = [
    {
      title: "Coverage",
      stats: {
        district: prediction.location,
        total_hours: prediction.totalHours,
        snowfall_hours: prediction.snowyHours,
        first_snow_hour: prediction.firstSnowHour || "N/A",
        last_snow_hour: prediction.lastSnowHour || "N/A",
      },
    },
    {
      title: "Model",
      stats: {
        name: prediction.modelName,
        training_samples: prediction.trainingSamples,
      },
    },
  ];

  for (const cardData of cards) {
    const card = document.createElement("article");
    card.className = "split-card";
    card.innerHTML = `
			<h4>${cardData.title}</h4>
			${Object.entries(cardData.stats)
        .map(
          ([key, value]) =>
            `<p class="split-stat"><strong>${key.replaceAll("_", " ")}:</strong> ${value}</p>`,
        )
        .join("")}
		`;
    splitGrid.appendChild(card);
  }

  predictionChip.textContent =
    prediction.snowyHours > 0 ? "Snowfall detected" : "No snowfall detected";
  predictionChip.className = `result-chip ${prediction.snowyHours > 0 ? "chip-rain" : "chip-clear"}`;

  predictionNotes.innerHTML = `
		<p><strong>Method:</strong> best Level 3 unsupervised anomaly model + snow-like filter from the notebook workflow.</p>
		<p><strong>Scope:</strong> all dataset rows for district <strong>${prediction.location}</strong>.</p>
		<p><strong>Output:</strong> predicted snowfall hours and monthly distribution.</p>
    <p><strong>Anomaly Score:</strong> relative anomaly intensity within the selected district (not a calibrated snowfall probability).</p>
    <p><strong>Score Quality:</strong> there is no universal “good” score threshold. Use district seasonality and monthly pattern consistency to judge quality.</p>
    <p><strong>Observed Snow:</strong> ${prediction.hasObservedSnow ? "derived from dataset 'snowfall' values" : "N/A (dataset has no 'snowfall' ground-truth column)"}.</p>
	`;

  hyperparams.textContent = JSON.stringify(prediction.chosenParams, null, 2);
}

function renderPrediction(prediction, durationMs) {
  renderKPIs(prediction);
  renderHourlyTable(prediction);
  renderBars(prediction);
  renderDetails(prediction);
  setStatus(`Prediction ready in ${durationMs} ms`);
  log(`Predicted yearly snowfall hours for district ${prediction.location}.`);
}

async function loadOptions() {
  setStatus("Loading Level 3 options...");
  const response = await fetch("/api/level3-options");
  const payload = await response.json();
  if (!response.ok || !payload.ok) {
    throw new Error(payload.error || "Could not load Level 3 options.");
  }

  datasetPathInput.value = payload.datasetPath;
  locationSelect.innerHTML = payload.locations
    .map((location) => `<option value="${location}">${location}</option>`)
    .join("");
  datasetRange.textContent = `Available data from ${payload.minDate} to ${payload.maxDate}. Select a district to predict snowfall hours.`;

  setStatus("Level 3 options loaded");
  log("Loaded districts for Level 3 snowfall prediction.");
}

async function handlePrediction(event) {
  event.preventDefault();
  const payload = {
    datasetPath: datasetPathInput.value.trim(),
    location: locationSelect.value,
  };

  setStatus("Running Level 3 snowfall prediction...");
  log(`Predicting snowfall hours for district ${payload.location}.`);

  const response = await fetch("/api/predict-snowfall-district", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const result = await response.json();
  if (!response.ok || !result.ok) {
    throw new Error(result.error || "Snowfall prediction failed.");
  }

  renderPrediction(result.prediction, result.durationMs);
}

level3Form.addEventListener("submit", (event) => {
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
