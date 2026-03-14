import json
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

        if self.path in ["/", "/index.html"]:
            self.path = "/index.html"

        return super().do_GET()

    def do_POST(self):
        if self.path == "/api/run-experiment":
            self.handle_run_experiment()
            return

        self._send_json(404, {"error": "Not found"})

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

    def handle_run_experiment(self):
        content_length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(content_length) if content_length > 0 else b"{}"

        try:
            payload = json.loads(body.decode("utf-8"))
        except json.JSONDecodeError:
            self._send_json(400, {"error": "Invalid JSON body."})
            return

        dataset_path = str(payload.get("datasetPath") or os.path.join(ROOT_DIR, "metherology_dataset.csv")).strip()
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

        if not os.path.isabs(dataset_path):
            candidate_from_root = os.path.normpath(os.path.join(ROOT_DIR, dataset_path))
            if os.path.exists(candidate_from_root):
                dataset_path = candidate_from_root
            else:
                # Fallback: if a UI-provided relative path escapes root (e.g. ../file.csv),
                # still try locating the file by name inside the workspace root.
                dataset_path = os.path.join(ROOT_DIR, os.path.basename(dataset_path))

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
