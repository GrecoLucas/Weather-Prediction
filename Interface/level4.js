const statusPill = document.getElementById("status-pill");
const level4Form = document.getElementById("level4-form");
const datasetPathInput = document.getElementById("dataset-path");
const selectedDateInput = document.getElementById("selected-date");
const locationSelect = document.getElementById("location-select");
const underPenaltyInput = document.getElementById("under-penalty");
const refreshOptionsBtn = document.getElementById("refresh-options-btn");
const datasetRange = document.getElementById("dataset-range");
const kpiGrid = document.getElementById("kpi-grid");
const rowsBody = document.getElementById("rows-body");
const bars = document.getElementById("bars");
const splitGrid = document.getElementById("split-grid");
const impactSummary = document.getElementById("impact-summary");
const predictionChip = document.getElementById("prediction-chip");
const runLog = document.getElementById("run-log");
const kpiTemplate = document.getElementById("kpi-template");

let staticMeanVehicleErrorAllDays = null;
let notebookEquivalent = null;

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

function fmt(value, decimals = 3) {
  if (value === null || value === undefined || Number.isNaN(Number(value)))
    return "-";
  return Number(value).toFixed(decimals);
}

function getSummaryRow(summary, model, mode) {
  return summary.find((r) => r.Model === model && r.Mode === mode) || null;
}

function renderKPIs(prediction) {
  const summary = prediction.summary || [];
  const rows = prediction.rows || [];
  const notebookRows = (notebookEquivalent && notebookEquivalent.rows) || [];
  const notebookGlobalNo = getSummaryRow(notebookRows, "Global", "No uplift");
  const best = summary.reduce((acc, row) => {
    if (
      !acc ||
      Number(row["Total Vehicle Error"]) < Number(acc["Total Vehicle Error"])
    )
      return row;
    return acc;
  }, null);

  const locationCount = new Set(rows.map((r) => r.location)).size;
  const isAllLocations = locationCount > 1;

  const bestTotalError = best ? Number(best["Total Vehicle Error"]) : null;
  const bestMeanError = best ? Number(best["Mean Vehicle Error"]) : null;
  const bestActualVeh = best ? Number(best["Total Actual Vehicles"]) : null;
  const bestPredVeh = best ? Number(best["Total Pred Vehicles"]) : null;
  const bestUnderRate = best ? Number(best["Underestimation Rate"]) : null;
  const vehicleGap =
    bestActualVeh != null && bestPredVeh != null
      ? bestPredVeh - bestActualVeh
      : null;

  const items = [
    {
      label: "Vehicle Error (Objective)",
      value: bestTotalError != null ? fmt(bestTotalError, 2) : "-",
      extra: isAllLocations
        ? `Total for ${locationCount} locations on ${prediction.selectedDate}`
        : `Selected location on ${prediction.selectedDate}`,
    },
    {
      label: isAllLocations ? "Mean Vehicle Error" : "Vehicle Error (Mean)",
      value: bestMeanError != null ? fmt(bestMeanError, 2) : "-",
      extra: isAllLocations
        ? "Average across selected day rows"
        : "Average per row for selected location/day",
    },
    {
      label: "Vehicles Needed vs Predicted",
      value:
        bestActualVeh != null && bestPredVeh != null
          ? `${fmt(bestActualVeh, 0)} vs ${fmt(bestPredVeh, 0)}`
          : "-",
      extra: "Needed = 3 x actual accidents",
    },
    {
      label: "Allocation Gap + Underestimation",
      value:
        vehicleGap != null && bestUnderRate != null
          ? `${fmt(vehicleGap, 0)} | ${fmt(bestUnderRate, 3)}`
          : "-",
      extra: "Pred-Actual vehicles | underestimation rate",
    },
    {
      label: "Best Setup",
      value: prediction.bestSetup || "-",
      extra: "Lowest Vehicle Error setup",
    },
    {
      label: "Mean Vehicle Error (Global, No Uplift)",
      value: notebookGlobalNo
        ? fmt(notebookGlobalNo["Mean Vehicle Error"], 2)
        : "-",
      extra: notebookEquivalent
        ? `Notebook test window: ${notebookEquivalent.testStartDate} to ${notebookEquivalent.testEndDate}`
        : "Notebook-equivalent summary not available",
      danger: true,
    },
    {
      label: "Median Vehicle Error (Global, No Uplift)",
      value:
        notebookGlobalNo &&
        notebookGlobalNo["Median Vehicle Error"] !== undefined
          ? fmt(notebookGlobalNo["Median Vehicle Error"], 2)
          : "-",
      extra: "Same metric definition as notebook final compact table",
      danger: true,
    },
  ];

  kpiGrid.innerHTML = "";
  for (const item of items) {
    const node = kpiTemplate.content.cloneNode(true);
    const labelEl = node.querySelector(".kpi-label");
    const valueEl = node.querySelector(".kpi-value");
    labelEl.textContent = item.label;
    valueEl.textContent = item.value;
    node.querySelector(".kpi-extra").textContent = item.extra;
    if (item.danger) {
      valueEl.style.color = "#15803d";
      valueEl.style.fontWeight = "700";
      labelEl.style.color = "#14532d";
    }
    kpiGrid.appendChild(node);
  }
}

function renderRows(prediction) {
  rowsBody.innerHTML = "";
  const rows = prediction.rows || [];

  for (const row of rows) {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${row.location}</td>
      <td>${fmt(row.actualAccidents, 2)}</td>
      <td>${fmt(row.globalNoUplift, 2)}</td>
      <td>${fmt(row.globalWithUplift, 2)}</td>
      <td>${fmt(row.regionalNoUplift, 2)}</td>
      <td>${fmt(row.regionalWithUplift, 2)}</td>
    `;
    rowsBody.appendChild(tr);
  }

  if (!rows.length) {
    rowsBody.innerHTML =
      '<tr><td colspan="6">No rows returned for this date/location.</td></tr>';
  }
}

function renderBars(prediction) {
  bars.innerHTML = "";
  const summary = prediction.summary || [];
  if (!summary.length) {
    bars.innerHTML = '<p class="hint">No summary rows returned.</p>';
    return;
  }

  const maxErr = Math.max(
    ...summary.map((r) => Number(r["Total Vehicle Error"]) || 0),
    0.001,
  );

  for (const row of summary) {
    const totalErr = Number(row["Total Vehicle Error"]) || 0;
    const width = Math.max(2, (totalErr / maxErr) * 100);
    const label = `${row.Model} - ${row.Mode}`;
    const bar = document.createElement("div");
    bar.className = "bar-row";
    bar.innerHTML = `
      <span>${label}</span>
      <div class="bar-track"><div class="bar-fill" style="width:${width}%; background:var(--secondary)"></div></div>
      <strong>${fmt(totalErr, 2)}</strong>
    `;
    bars.appendChild(bar);
  }
}

function renderDetails(prediction) {
  splitGrid.innerHTML = "";
  const cards = [
    {
      title: "Request",
      stats: {
        date: prediction.selectedDate,
        location: prediction.location,
        under_penalty: prediction.underPenalty,
        duration_ms: prediction.durationMs,
      },
    },
    {
      title: "Uplift",
      stats: {
        global_uplift: fmt(prediction.globalUplift, 3),
        interpretation: "Additive accidents/day",
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

  const summary = prediction.summary || [];
  const impactLines = [];
  for (const model of ["Global", "Regional"]) {
    const noRow = getSummaryRow(summary, model, "No uplift");
    const yesRow = getSummaryRow(summary, model, "With uplift");
    if (!noRow || !yesRow) continue;

    const errDelta =
      Number(yesRow["Total Vehicle Error"]) -
      Number(noRow["Total Vehicle Error"]);
    const underDelta =
      Number(yesRow["Underestimation Rate"]) -
      Number(noRow["Underestimation Rate"]);
    impactLines.push(
      `${model}: Vehicle Error shift = ${fmt(errDelta, 2)} | Underestimation shift = ${fmt(underDelta, 3)}`,
    );
  }
  const objectiveLine =
    "Objective formula: Vehicle Error = | 3 x actual accidents - predicted vehicles |";
  const notebookLine = notebookEquivalent
    ? `Notebook-equivalent test window: ${notebookEquivalent.testStartDate} to ${notebookEquivalent.testEndDate} (${notebookEquivalent.testDays} days)`
    : "Notebook-equivalent test window not available.";
  const impactText =
    impactLines.join("\n") || "No uplift impact summary available.";
  impactSummary.textContent = `${objectiveLine}\n${notebookLine}\n${impactText}`;

  predictionChip.textContent = prediction.bestSetup || "Prediction ready";
  predictionChip.className = "result-chip chip-rain";
}

function renderPrediction(prediction, durationMs) {
  prediction.durationMs = durationMs;
  renderKPIs(prediction);
  renderRows(prediction);
  renderBars(prediction);
  renderDetails(prediction);
  setStatus(`Prediction ready in ${durationMs} ms`);
  log(
    `Level 4 prediction done for ${prediction.selectedDate} (${prediction.location}).`,
  );
}

async function loadOptions() {
  setStatus("Loading Level 4 options...");
  const response = await fetch("/api/level4-options");
  const payload = await response.json();

  if (!response.ok || !payload.ok) {
    throw new Error(payload.error || "Could not load Level 4 options.");
  }

  datasetPathInput.value = payload.datasetPath;
  staticMeanVehicleErrorAllDays = payload.staticMeanVehicleErrorAllDays;
  notebookEquivalent = payload.notebookEquivalent || null;
  selectedDateInput.min = payload.minDate || "";
  selectedDateInput.max = payload.maxDate || "";
  if (!selectedDateInput.value) {
    selectedDateInput.value = payload.maxDate || "";
  }

  locationSelect.innerHTML = '<option value="">All locations</option>';
  for (const location of payload.locations || []) {
    const option = document.createElement("option");
    option.value = location;
    option.textContent = location;
    locationSelect.appendChild(option);
  }

  const notebookText = notebookEquivalent
    ? ` Notebook-equivalent test window: ${notebookEquivalent.testStartDate} to ${notebookEquivalent.testEndDate}.`
    : "";
  datasetRange.textContent = `Available dates: ${payload.minDate} to ${payload.maxDate}. Choose a date to compare uplift impact.${notebookText}`;
  setStatus("Level 4 options loaded");
  log("Loaded Level 4 options.");
}

async function handlePrediction(event) {
  event.preventDefault();

  const payload = {
    datasetPath: datasetPathInput.value.trim(),
    selectedDate: selectedDateInput.value,
    location: locationSelect.value,
    underPenalty: Number(underPenaltyInput.value),
  };

  if (!payload.selectedDate) {
    throw new Error("selectedDate is required.");
  }

  setStatus("Running Level 4 prediction...");
  log(
    `Predicting accidents for ${payload.selectedDate} (location: ${payload.location || "ALL"}).`,
  );

  const response = await fetch("/api/predict-accidents-day", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  const result = await response.json();
  if (!response.ok || !result.ok) {
    throw new Error(result.error || "Level 4 prediction failed.");
  }

  renderPrediction(result.prediction, result.durationMs);
}

level4Form.addEventListener("submit", (event) => {
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
