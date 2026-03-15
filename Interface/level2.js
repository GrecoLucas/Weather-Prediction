const statusPill = document.getElementById("status-pill");
const level2Form = document.getElementById("level2-form");
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
const predictionChip = document.getElementById("prediction-chip");
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

function formatTemp(value) {
	if (value === null || value === undefined || Number.isNaN(Number(value))) {
		return "N/A";
	}
	return `${Number(value).toFixed(2)} C`;
}

function renderKPIs(prediction) {
	kpiGrid.innerHTML = "";
	const isFuture = prediction.forecastMode === "future";
	const items = [
		{
			label: "Predicted Avg",
			value: formatTemp(prediction.predictedAverage),
			extra: `${prediction.modelName}`,
		},
		{
			label: "Actual Avg",
			value: formatTemp(prediction.actualAverage),
			extra: isFuture ? `Future date (actual unknown)` : `Selected day in dataset`,
		},
		{
			label: "Day MAE",
			value: formatTemp(prediction.dayMetrics?.mae),
			extra: isFuture ? `Not available for future forecasts` : `Hourly absolute error average`,
		},
		{
			label: "Prediction Range",
			value: `${formatTemp(prediction.predictedMin)} to ${formatTemp(prediction.predictedMax)}`,
			extra: `${prediction.totalHours} hourly points`,
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
		const actualValue = row.actualTemperature === null ? "-" : formatTemp(row.actualTemperature);
		const absErrValue = row.absoluteError === null ? "-" : formatTemp(row.absoluteError);
		tr.innerHTML = `
			<td>${new Date(row.time).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}</td>
			<td>${formatTemp(row.predictedTemperature)}</td>
			<td>${actualValue}</td>
			<td>${absErrValue}</td>
		`;
		hourlyBody.appendChild(tr);
	}
	if (!prediction.hourly.length) {
		hourlyBody.innerHTML = '<tr><td colspan="4">No hourly rows found.</td></tr>';
	}
}

function renderBars(prediction) {
	bars.innerHTML = "";
	const tempValues = prediction.hourly.map((row) => Number(row.predictedTemperature));
	const maxTemp = Math.max(...tempValues);
	const minTemp = Math.min(...tempValues);
	const span = Math.max(1, maxTemp - minTemp);

	for (const row of prediction.hourly) {
		const normalized = ((Number(row.predictedTemperature) - minTemp) / span) * 100;
		const width = Math.max(8, Math.min(100, normalized));
		const bar = document.createElement("div");
		bar.className = "bar-row";
		bar.innerHTML = `
			<span>${new Date(row.time).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}</span>
			<div class="bar-track"><div class="bar-fill" style="width:${width}%; background:var(--secondary)"></div></div>
			<strong>${formatTemp(row.predictedTemperature)}</strong>
		`;
		bars.appendChild(bar);
	}
}

function renderDetails(prediction) {
	splitGrid.innerHTML = "";
	const isFuture = prediction.forecastMode === "future";
	const cards = [
		{
			title: "Validation",
			stats: prediction.validation,
		},
		{
			title: "Selected Day",
			stats: prediction.dayMetrics || { note: "No actual values available yet" },
		},
		{
			title: "Context",
			stats: {
				date: prediction.selectedDate,
				location: prediction.location || "All",
				forecast_mode: prediction.forecastMode,
				last_historical_date: prediction.lastHistoricalDate,
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
				.map(([key, value]) => `<p class="split-stat"><strong>${key.replaceAll("_", " ")}:</strong> ${typeof value === "number" ? value.toFixed(4) : value}</p>`)
				.join("")}
		`;
		splitGrid.appendChild(card);
	}

	predictionChip.textContent = `Predicted avg ${formatTemp(prediction.predictedAverage)}`;
	predictionChip.className = "result-chip chip-clear";

	predictionNotes.innerHTML = `
		<p><strong>Pipeline source:</strong> Level 2 feature engineering and model flow adapted from level2_models.ipynb into Python module level2/temperature_prediction.py.</p>
		<p><strong>Prediction target:</strong> next-hour temperature values across all rows for ${prediction.selectedDate}.</p>
		<p><strong>Mode:</strong> ${isFuture ? "Future forecast (actual values are not in the dataset yet)." : "Historical backtest (actual values are available)."}</p>
		<p><strong>Tip:</strong> if MAE is unstable, compare model families (LGBM/XGB/RF/LR/Vote) and adjust excluded features in the Python module.</p>
	`;
}

function renderPrediction(prediction, durationMs) {
	renderKPIs(prediction);
	renderHourlyTable(prediction);
	renderBars(prediction);
	renderDetails(prediction);
	setStatus(`Prediction ready in ${durationMs} ms`);
	log(`Predicted ${prediction.selectedDate} with ${prediction.modelName}.`);
}

async function loadOptions() {
	setStatus("Loading Level 2 options...");
	const response = await fetch("/api/level2-options");
	const payload = await response.json();
	if (!response.ok || !payload.ok) {
		throw new Error(payload.error || "Could not load Level 2 options.");
	}

	datasetPathInput.value = payload.datasetPath;
	selectedDateInput.min = payload.minDate || "";
	selectedDateInput.max = "";
	if (!selectedDateInput.value) {
		const maxDay = payload.maxDate ? new Date(payload.maxDate) : new Date();
		maxDay.setDate(maxDay.getDate() + 1);
		selectedDateInput.value = maxDay.toISOString().slice(0, 10);
	}
	datasetRange.textContent = `Historical data available from ${payload.minDate} to ${payload.maxDate}. You can also choose future dates.`;

	locationSelect.innerHTML = '<option value="">All available locations</option>';
	if (Array.isArray(payload.locations)) {
		for (const location of payload.locations) {
			const option = document.createElement("option");
			option.value = location;
			option.textContent = location;
			locationSelect.appendChild(option);
		}
	}

	setStatus("Level 2 options loaded");
	log("Loaded available dates and locations for temperature prediction.");
}

async function handlePrediction(event) {
	event.preventDefault();
	const payload = {
		datasetPath: datasetPathInput.value.trim(),
		selectedDate: selectedDateInput.value,
		location: locationSelect.value,
		modelFamily: modelFamilySelect.value,
	};

	setStatus("Running temperature prediction...");
	log(`Predicting temperature for ${payload.selectedDate} using ${payload.modelFamily}.`);

	const response = await fetch("/api/predict-temperature-day", {
		method: "POST",
		headers: { "Content-Type": "application/json" },
		body: JSON.stringify(payload),
	});
	const result = await response.json();
	if (!response.ok || !result.ok) {
		throw new Error(result.error || "Temperature prediction failed.");
	}

	renderPrediction(result.prediction, result.durationMs);
}

level2Form.addEventListener("submit", (event) => {
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
