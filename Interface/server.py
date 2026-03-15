import json
import importlib.util
import os
import subprocess
import sys
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INTERFACE_DIR = os.path.join(ROOT_DIR, "Interface")
MODEL_DIR = os.path.join(ROOT_DIR, "Level_1_Rain_Classification")
MODEL_SCRIPT = os.path.join(MODEL_DIR, "rain_prediction.py")
LATEST_REPORT_PATH = os.path.join(MODEL_DIR, "model_metrics.txt")
LEVEL2_DIR = os.path.join(ROOT_DIR, "level2")
LEVEL2_SCRIPT = os.path.join(LEVEL2_DIR, "temperature_prediction.py")
LEVEL3_DIR = os.path.join(ROOT_DIR, "Level_3_Unsupervised Snow Detection")
LEVEL3_SCRIPT = os.path.join(LEVEL3_DIR, "snow_prediction.py")

model_spec = importlib.util.spec_from_file_location("rain_prediction_module", MODEL_SCRIPT)
rain_prediction_module = importlib.util.module_from_spec(model_spec)
assert model_spec and model_spec.loader
model_spec.loader.exec_module(rain_prediction_module)

get_prediction_options = rain_prediction_module.get_prediction_options
predict_rain_for_day = rain_prediction_module.predict_rain_for_day

level2_spec = importlib.util.spec_from_file_location("temperature_prediction_module", LEVEL2_SCRIPT)
temperature_prediction_module = importlib.util.module_from_spec(level2_spec)
assert level2_spec and level2_spec.loader
level2_spec.loader.exec_module(temperature_prediction_module)

get_temperature_prediction_options = temperature_prediction_module.get_temperature_prediction_options
predict_temperature_for_day = temperature_prediction_module.predict_temperature_for_day
list_saved_temperature_models = temperature_prediction_module.list_saved_temperature_models


def _reload_level2_module():
    global get_temperature_prediction_options
    global predict_temperature_for_day
    global list_saved_temperature_models

    module_spec = importlib.util.spec_from_file_location("temperature_prediction_module", LEVEL2_SCRIPT)
    module = importlib.util.module_from_spec(module_spec)
    assert module_spec and module_spec.loader
    module_spec.loader.exec_module(module)

    get_temperature_prediction_options = module.get_temperature_prediction_options
    predict_temperature_for_day = module.predict_temperature_for_day
    list_saved_temperature_models = module.list_saved_temperature_models

level3_spec = importlib.util.spec_from_file_location("snow_prediction_module", LEVEL3_SCRIPT)
snow_prediction_module = importlib.util.module_from_spec(level3_spec)
assert level3_spec and level3_spec.loader
level3_spec.loader.exec_module(snow_prediction_module)

get_snowfall_prediction_options = snow_prediction_module.get_snowfall_prediction_options
predict_snowfall_for_district = snow_prediction_module.predict_snowfall_for_district


class DashboardHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=INTERFACE_DIR, **kwargs)

    def _send_json(self, status_code, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/latest-report":
            self.handle_latest_report()
            return

        if self.path == "/api/level1-options":
            self.handle_level1_options()
            return

        if self.path == "/api/level2-options":
            self.handle_level2_options()
            return

        if self.path == "/api/level3-options":
            self.handle_level3_options()
            return

        if self.path in ["/", "/index.html"]:
            self.path = "/index.html"

        return super().do_GET()

    def do_POST(self):
        if self.path == "/api/run-experiment":
            self.handle_run_experiment()
            return

        if self.path == "/api/predict-rain-day":
            self.handle_predict_rain_day()
            return

        if self.path == "/api/predict-temperature-day":
            self.handle_predict_temperature_day()
            return

        if self.path == "/api/predict-snowfall-district":
            self.handle_predict_snowfall_district()
            return

        self._send_json(404, {"error": "Not found"})

    def _resolve_dataset_path(self, dataset_path):
        dataset_path = str(dataset_path or os.path.join(ROOT_DIR, "data/meteorology_dataset.csv")).strip()

        if not os.path.isabs(dataset_path):
            candidate_from_root = os.path.normpath(os.path.join(ROOT_DIR, dataset_path))
            if os.path.exists(candidate_from_root):
                dataset_path = candidate_from_root
            else:
                dataset_path = os.path.join(ROOT_DIR, os.path.basename(dataset_path))

        return dataset_path

    def handle_latest_report(self):
        if not os.path.exists(LATEST_REPORT_PATH):
            self._send_json(404, {"error": "No report found yet. Run an experiment first."})
            return

        with open(LATEST_REPORT_PATH, "r", encoding="utf-8") as report_file:
            report_text = report_file.read()

        self._send_json(
            200,
            {
                "ok": True,
                "source": "latest-file",
                "report": report_text,
                "reportPath": LATEST_REPORT_PATH,
            },
        )

    def handle_level1_options(self):
        dataset_path = self._resolve_dataset_path(os.path.join(ROOT_DIR, "data/meteorology_dataset.csv"))
        if not os.path.exists(dataset_path):
            self._send_json(400, {"error": f"Dataset not found: {dataset_path}"})
            return

        try:
            options = get_prediction_options(dataset_path)
        except Exception as exc:
            self._send_json(500, {"error": str(exc)})
            return

        self._send_json(200, {"ok": True, **options, "datasetPath": dataset_path})

    def handle_level2_options(self):
        _reload_level2_module()
        dataset_path = self._resolve_dataset_path(os.path.join(ROOT_DIR, "data/meteorology_dataset.csv"))
        if not os.path.exists(dataset_path):
            self._send_json(400, {"error": f"Dataset not found: {dataset_path}"})
            return

        try:
            options = get_temperature_prediction_options(dataset_path)
        except Exception as exc:
            self._send_json(500, {"error": str(exc)})
            return

        self._send_json(200, {"ok": True, **options, "datasetPath": dataset_path})

    def handle_level3_options(self):
        dataset_path = self._resolve_dataset_path(os.path.join(ROOT_DIR, "data/meteorology_dataset.csv"))
        if not os.path.exists(dataset_path):
            self._send_json(400, {"error": f"Dataset not found: {dataset_path}"})
            return

        try:
            options = get_snowfall_prediction_options(dataset_path)
        except Exception as exc:
            self._send_json(500, {"error": str(exc)})
            return

        self._send_json(200, {"ok": True, **options, "datasetPath": dataset_path})

    def handle_run_experiment(self):
        content_length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(content_length) if content_length > 0 else b"{}"

        try:
            payload = json.loads(body.decode("utf-8"))
        except json.JSONDecodeError:
            self._send_json(400, {"error": "Invalid JSON body."})
            return

        dataset_path = self._resolve_dataset_path(payload.get("datasetPath"))
        model_family = str(payload.get("modelFamily") or "all").strip().lower()
        profile = str(payload.get("profile") or "balanced").strip().lower()

        valid_families = {"all", "rf", "xgb", "lgbm"}
        valid_profiles = {"balanced", "recall", "precision"}

        if model_family not in valid_families:
            self._send_json(400, {"error": f"Invalid modelFamily '{model_family}'."})
            return

        if profile not in valid_profiles:
            self._send_json(400, {"error": f"Invalid profile '{profile}'."})
            return

        if not os.path.exists(dataset_path):
            self._send_json(400, {"error": f"Dataset not found: {dataset_path}"})
            return

        start_time = time.time()

        cmd = [
            sys.executable,
            MODEL_SCRIPT,
            "--dataset-path",
            dataset_path,
            "--model-family",
            model_family,
            "--profile",
            profile,
            "--output-path",
            LATEST_REPORT_PATH,
        ]

        completed = subprocess.run(
            cmd,
            cwd=MODEL_DIR,
            capture_output=True,
            text=True,
            check=False,
        )

        duration_ms = int((time.time() - start_time) * 1000)

        if completed.returncode != 0:
            self._send_json(
                500,
                {
                    "error": "Model run failed.",
                    "stdout": completed.stdout,
                    "stderr": completed.stderr,
                    "durationMs": duration_ms,
                },
            )
            return

        if not os.path.exists(LATEST_REPORT_PATH):
            self._send_json(
                500,
                {
                    "error": "Model run finished but report file was not produced.",
                    "stdout": completed.stdout,
                    "stderr": completed.stderr,
                    "durationMs": duration_ms,
                },
            )
            return

        with open(LATEST_REPORT_PATH, "r", encoding="utf-8") as report_file:
            report_text = report_file.read()

        self._send_json(
            200,
            {
                "ok": True,
                "report": report_text,
                "reportPath": LATEST_REPORT_PATH,
                "durationMs": duration_ms,
                "stdout": completed.stdout,
            },
        )

    def handle_predict_rain_day(self):
        content_length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(content_length) if content_length > 0 else b"{}"

        try:
            payload = json.loads(body.decode("utf-8"))
        except json.JSONDecodeError:
            self._send_json(400, {"error": "Invalid JSON body."})
            return

        dataset_path = self._resolve_dataset_path(payload.get("datasetPath"))
        selected_date = str(payload.get("selectedDate") or "").strip()
        location = str(payload.get("location") or "").strip()
        model_family = str(payload.get("modelFamily") or "all").strip().lower()
        profile = str(payload.get("profile") or "balanced").strip().lower()

        if not selected_date:
            self._send_json(400, {"error": "selectedDate is required."})
            return

        if not location:
            self._send_json(400, {"error": "location is required."})
            return

        if not os.path.exists(dataset_path):
            self._send_json(400, {"error": f"Dataset not found: {dataset_path}"})
            return

        start_time = time.time()
        try:
            prediction = predict_rain_for_day(
                filepath=dataset_path,
                selected_date=selected_date,
                location=location,
                model_family=model_family,
                profile=profile,
            )
        except Exception as exc:
            self._send_json(500, {"error": str(exc)})
            return

        duration_ms = int((time.time() - start_time) * 1000)
        self._send_json(200, {"ok": True, "prediction": prediction, "durationMs": duration_ms})

    def handle_predict_temperature_day(self):
        _reload_level2_module()
        content_length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(content_length) if content_length > 0 else b"{}"

        try:
            payload = json.loads(body.decode("utf-8"))
        except json.JSONDecodeError:
            self._send_json(400, {"error": "Invalid JSON body."})
            return

        dataset_path = self._resolve_dataset_path(payload.get("datasetPath"))
        selected_date = str(payload.get("selectedDate") or "").strip()
        location = str(payload.get("location") or "").strip()
        saved_model = str(payload.get("savedModel") or "").strip()

        if not selected_date:
            self._send_json(400, {"error": "selectedDate is required."})
            return

        if not saved_model:
            self._send_json(400, {"error": "savedModel is required."})
            return

        if not os.path.exists(dataset_path):
            self._send_json(400, {"error": f"Dataset not found: {dataset_path}"})
            return

        available_models = {item["value"] for item in list_saved_temperature_models()}
        if saved_model not in available_models:
            self._send_json(400, {"error": f"Saved model not found: {saved_model}"})
            return

        start_time = time.time()
        try:
            prediction = predict_temperature_for_day(
                filepath=dataset_path,
                selected_date=selected_date,
                location=location,
                saved_model_name=saved_model,
            )
        except Exception as exc:
            self._send_json(500, {"error": str(exc)})
            return

        duration_ms = int((time.time() - start_time) * 1000)
        self._send_json(200, {"ok": True, "prediction": prediction, "durationMs": duration_ms})

    def handle_predict_snowfall_district(self):
        content_length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(content_length) if content_length > 0 else b"{}"

        try:
            payload = json.loads(body.decode("utf-8"))
        except json.JSONDecodeError:
            self._send_json(400, {"error": "Invalid JSON body."})
            return

        dataset_path = self._resolve_dataset_path(payload.get("datasetPath"))
        location = str(payload.get("location") or "").strip()

        if not location:
            self._send_json(400, {"error": "location is required."})
            return

        if not os.path.exists(dataset_path):
            self._send_json(400, {"error": f"Dataset not found: {dataset_path}"})
            return

        start_time = time.time()
        try:
            prediction = predict_snowfall_for_district(
                filepath=dataset_path,
                location=location,
            )
        except Exception as exc:
            self._send_json(500, {"error": str(exc)})
            return

        duration_ms = int((time.time() - start_time) * 1000)
        self._send_json(200, {"ok": True, "prediction": prediction, "durationMs": duration_ms})


def run_server(port=8000):
    server = ThreadingHTTPServer(("127.0.0.1", port), DashboardHandler)
    print(f"Weather Intelligence Studio running at http://127.0.0.1:{port}")
    print("Press Ctrl+C to stop the server.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    run_server()
