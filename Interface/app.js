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

function parseReport(text) {
	const normalized = text.replace(/\r/g, "");

	const splitMatch = normalized.match(/Train\s*:\s*(\d+)[\s\S]*?Validation\s*:\s*(\d+)[\s\S]*?Test\s*:\s*(\d+)/i);
	const splitSizes = splitMatch
		? {
				train: Number(splitMatch[1]),
				validation: Number(splitMatch[2]),
				test: Number(splitMatch[3]),
			}
		: { train: 0, validation: 0, test: 0 };

	const models = {};
	const sections = normalized.split(/\n={20,}\nMODEL:\s*/g);

	for (let i = 1; i < sections.length; i += 1) {
		const section = sections[i];
		const firstLineBreak = section.indexOf("\n");
		const modelName = section.slice(0, firstLineBreak).trim();
		const body = section.slice(firstLineBreak);

		const paramsMatch = body.match(/Chosen Hyperparameters\s*:\s*(\{[^\n]*\})/i);
		const splitData = extractSplitData(body);

		models[modelName] = {
			name: modelName,
			params: paramsMatch ? paramsMatch[1].trim() : "{}",
			train: splitData.train,
			validation: splitData.validation,
			test: splitData.test,
		};
	}

	const ranking = extractRanking(normalized);
	applyRankingFallback(models, ranking);
	const bestModelMatch = normalized.match(/BEST MODEL:\s*([^\n]+)/i);
	const bestModel = bestModelMatch ? bestModelMatch[1].trim() : ranking[0]?.model;

	return {
		splitSizes,
		models,
		ranking,
		bestModel,
	};
}

function extractSplitData(modelBody) {
	const getSplit = (splitName) => {
		const splitRegex = new RegExp(
			`---\\s*${splitName}\\s*Set\\s*---[\\s\\S]*?F1-Score\\s*:\\s*([\\d.]+)[\\s\\S]*?Precision\\s*:\\s*([\\d.]+)[\\s\\S]*?Recall\\s*:\\s*([\\d.]+)[\\s\\S]*?Confusion Matrix:\\s*\\n\\[\\[\\s*(\\d+)\\s+(\\d+)\\]\\s*\\n\\s*\\[\\s*(\\d+)\\s+(\\d+)\\]\\]`,
			"i"
		);
		const match = modelBody.match(splitRegex);
		if (!match) {
			return {
				f1: 0,
				precision: 0,
				recall: 0,
				confusionMatrix: [[0, 0], [0, 0]],
			};
		}

		return {
			f1: Number(match[1]),
			precision: Number(match[2]),
			recall: Number(match[3]),
			confusionMatrix: [
				[Number(match[4]), Number(match[5])],
				[Number(match[6]), Number(match[7])],
			],
		};
	};

	return {
		train: getSplit("Train"),
		validation: getSplit("Validation"),
		test: getSplit("Test"),
	};
}

function extractRanking(reportText) {
	const lines = reportText.split("\n");
	const ranking = [];
	const rowRegex = /^\s*(\d+)\s+([A-Za-z0-9_\-]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)/;

	for (const line of lines) {
		const row = line.match(rowRegex);
		if (row) {
			ranking.push({
				rank: Number(row[1]),
				model: row[2],
				valF1: Number(row[3]),
				testF1: Number(row[4]),
				valPrecision: Number(row[5]),
				valRecall: Number(row[6]),
			});
		}
	}

	if (ranking.length > 0) {
		return ranking;
	}

	const inferred = [];
	return inferred;
}

function applyRankingFallback(models, ranking) {
	ranking.forEach((row) => {
		const model = models[row.model];
		if (!model) {
			return;
		}

		if (model.validation.f1 === 0) {
			model.validation.f1 = row.valF1;
		}
		if (model.validation.precision === 0) {
			model.validation.precision = row.valPrecision;
		}
		if (model.validation.recall === 0) {
			model.validation.recall = row.valRecall;
		}
		if (model.test.f1 === 0) {
			model.test.f1 = row.testF1;
		}
	});
}

function renderDashboard(parsed) {
	APP_STATE.report = parsed;

	if (!APP_STATE.selectedModel || !parsed.models[APP_STATE.selectedModel]) {
		APP_STATE.selectedModel = parsed.bestModel || Object.keys(parsed.models)[0] || null;
	}

	renderKPIs(parsed);
	renderRanking(parsed);
	renderMetricBars(parsed);
	renderModelOptions(parsed);
	renderModelDetails(parsed, APP_STATE.selectedModel);
}

function renderKPIs(parsed) {
	const bestName = parsed.bestModel;
	const best = bestName ? parsed.models[bestName] : null;
	const modelCount = Object.keys(parsed.models).length;

	const snapshot = [
		{
			label: "Best Model",
			value: bestName || "N/A",
			extra: best ? `Validation F1: ${best.validation.f1.toFixed(4)}` : "No model parsed",
		},
		{
			label: "Total Samples",
			value: `${(parsed.splitSizes.train + parsed.splitSizes.validation + parsed.splitSizes.test).toLocaleString()}`,
			extra: `Train ${parsed.splitSizes.train.toLocaleString()} | Val ${parsed.splitSizes.validation.toLocaleString()} | Test ${parsed.splitSizes.test.toLocaleString()}`,
		},
		{
			label: "Candidates",
			value: String(modelCount),
			extra: "Models benchmarked in this run",
		}
	];

	els.kpiGrid.innerHTML = "";
	snapshot.forEach((item) => {
		const node = els.kpiTemplate.content.firstElementChild.cloneNode(true);
		node.querySelector(".kpi-label").textContent = item.label;
		node.querySelector(".kpi-value").textContent = item.value;
		node.querySelector(".kpi-extra").textContent = item.extra;
		els.kpiGrid.appendChild(node);
	});
}

function renderRanking(parsed) {
	els.rankingBody.innerHTML = "";

	const rows = parsed.ranking.length > 0
		? parsed.ranking
		: Object.values(parsed.models)
				.map((m) => ({
					model: m.name,
					valF1: m.validation.f1,
					testF1: m.test.f1,
					valPrecision: m.validation.precision,
					valRecall: m.validation.recall,
				}))
				.sort((a, b) => b.valF1 - a.valF1)
				.map((item, idx) => ({ rank: idx + 1, ...item }));

	rows.forEach((row) => {
		const tr = document.createElement("tr");
		if (row.model === parsed.bestModel) {
			tr.classList.add("highlight");
		}

		tr.innerHTML = `
			<td>${row.rank}</td>
			<td>${row.model}</td>
			<td>${row.valF1.toFixed(4)}</td>
			<td>${row.testF1.toFixed(4)}</td>
			<td>${row.valPrecision.toFixed(4)}</td>
			<td>${row.valRecall.toFixed(4)}</td>
		`;

		els.rankingBody.appendChild(tr);
	});
}

function renderMetricBars(parsed) {
	const metric = APP_STATE.selectedMetric;
	const rows = Object.values(parsed.models)
		.map((m) => ({
			name: m.name,
			value: m.validation[metric],
		}))
		.sort((a, b) => b.value - a.value);

	els.bars.innerHTML = "";
	rows.forEach((row, idx) => {
		const line = document.createElement("div");
		line.className = "bar-row";
		const width = Math.max(2, Math.round(row.value * 100));
		line.innerHTML = `
			<strong>${row.name}</strong>
			<div class="bar-track">
				<div class="bar-fill" style="width: ${width}%; background: ${METRIC_COLORS[idx % METRIC_COLORS.length]};"></div>
			</div>
			<span>${row.value.toFixed(4)}</span>
		`;
		els.bars.appendChild(line);
	});
}

function renderModelOptions(parsed) {
	const previousValue = APP_STATE.selectedModel;
	const names = Object.keys(parsed.models);
	els.modelSelect.innerHTML = "";

	names.forEach((name) => {
		const option = document.createElement("option");
		option.value = name;
		option.textContent = name;
		els.modelSelect.appendChild(option);
	});

	if (previousValue && names.includes(previousValue)) {
		els.modelSelect.value = previousValue;
	} else {
		els.modelSelect.value = parsed.bestModel || names[0] || "";
	}
}

function renderModelDetails(parsed, modelName) {
	const model = parsed.models[modelName];
	if (!model) {
		return;
	}

	APP_STATE.selectedModel = modelName;

	els.splitGrid.innerHTML = "";
	[
		["Train", model.train],
		["Validation", model.validation],
		["Test", model.test],
	].forEach(([label, data]) => {
		const card = document.createElement("article");
		card.className = "split-card";
		card.innerHTML = `
			<h4>${label}</h4>
			<p class="split-stat">F1: <strong>${data.f1.toFixed(4)}</strong></p>
			<p class="split-stat">Precision: <strong>${data.precision.toFixed(4)}</strong></p>
			<p class="split-stat">Recall: <strong>${data.recall.toFixed(4)}</strong></p>
		`;
		els.splitGrid.appendChild(card);
	});

	const m = model.validation.confusionMatrix;
	const cells = [
		{ label: "TN", value: m[0][0] },
		{ label: "FP", value: m[0][1] },
		{ label: "FN", value: m[1][0] },
		{ label: "TP", value: m[1][1] },
	];

	els.confusionMatrix.innerHTML = "";
	cells.forEach((cell) => {
		const div = document.createElement("div");
		div.className = "cell";
		div.innerHTML = `<span class="cell-label">${cell.label}</span><span class="cell-value">${cell.value.toLocaleString()}</span>`;
		els.confusionMatrix.appendChild(div);
	});

	els.hyperparams.textContent = model.params;
}

function setStatus(message) {
	els.statusPill.textContent = message;
}

function addLog(message) {
	const entry = document.createElement("p");
	entry.className = "log-entry";
	const now = new Date().toLocaleTimeString();
	entry.innerHTML = `<span class="log-time">${now}</span>${message}`;
	els.runLog.prepend(entry);
}

function safeParseAndRender(text, sourceLabel) {
	try {
		const parsed = parseReport(text);
		const hasModels = Object.keys(parsed.models).length > 0;
		if (!hasModels) {
			throw new Error("No models found in report.");
		}

		renderDashboard(parsed);
		setStatus(`Loaded from ${sourceLabel}`);
		addLog(`Report ingested successfully from ${sourceLabel}.`);
	} catch (err) {
		setStatus("Error while parsing report");
		addLog(`Parsing error: ${err.message}`);
	}
}

async function handleRunExperiment() {
	const payload = {
		datasetPath: els.datasetPath.value.trim(),
		modelFamily: els.modelFamily.value,
		profile: els.profile.value,
	};

	setStatus("Running model pipeline...");
	addLog(`Run requested with dataset=${payload.datasetPath}, family=${payload.modelFamily}, profile=${payload.profile}.`);

	els.runBtn.disabled = true;
	try {
		const response = await fetch("/api/run-experiment", {
			method: "POST",
			headers: {
				"Content-Type": "application/json",
			},
			body: JSON.stringify(payload),
		});

		const data = await response.json();
		if (!response.ok) {
			throw new Error(data.error || "Run failed.");
		}

		safeParseAndRender(data.report, "backend run");
		const durationSeconds = ((data.durationMs || 0) / 1000).toFixed(1);
		addLog(`Run completed in ${durationSeconds}s. Report loaded from backend.`);
	} catch (err) {
		setStatus("Run failed");
		addLog(`Run failed: ${err.message}`);
	} finally {
		els.runBtn.disabled = false;
	}
}

async function loadLatestBackendReport() {
	try {
		const response = await fetch("/api/latest-report");
		if (!response.ok) {
			return false;
		}

		const data = await response.json();
		safeParseAndRender(data.report, "latest backend report");
		addLog("Loaded latest saved report from backend.");
		return true;
	} catch (err) {
		addLog(`Backend report check skipped: ${err.message}`);
		return false;
	}
}

function setupEventListeners() {
	els.runForm.addEventListener("submit", async (event) => {
		event.preventDefault();
		await handleRunExperiment();
	});

	els.loadDemoBtn.addEventListener("click", () => {
		safeParseAndRender(DEMO_REPORT, "demo bundle");
	});

	els.metricsUpload.addEventListener("change", (event) => {
		const file = event.target.files?.[0];
		if (!file) {
			return;
		}

		const reader = new FileReader();
		reader.onload = () => {
			safeParseAndRender(String(reader.result || ""), file.name);
		};
		reader.onerror = () => {
			setStatus("Failed to read uploaded file");
			addLog("File reading failed.");
		};
		reader.readAsText(file);
	});

	els.metricSelect.addEventListener("change", () => {
		APP_STATE.selectedMetric = els.metricSelect.value;
		if (APP_STATE.report) {
			renderMetricBars(APP_STATE.report);
			addLog(`Metric switched to ${METRIC_LABELS[APP_STATE.selectedMetric]}.`);
		}
	});

	els.modelSelect.addEventListener("change", () => {
		const modelName = els.modelSelect.value;
		if (APP_STATE.report) {
			renderModelDetails(APP_STATE.report, modelName);
			addLog(`Deep dive changed to ${modelName}.`);
		}
	});
}

function bootstrap() {
	setupEventListeners();
	loadLatestBackendReport().then((loaded) => {
		if (!loaded) {
			safeParseAndRender(DEMO_REPORT, "initial demo");
			addLog("Dashboard initialized with demo data. Start backend for live runs.");
		}
	});
}

bootstrap();
