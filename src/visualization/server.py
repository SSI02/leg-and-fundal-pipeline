"""
Simple Flask server to serve the 3D viewer and pipeline outputs.

Usage:
    conda activate leg_pipeline
    python src/visualization/server.py --output_dir data/output/patient_001 [--port 8080]

Opens a web browser with the 3D viewer, automatically loading
the point cloud and measurements from the specified output directory.
"""

import os
import json
import argparse
import webbrowser
from flask import Flask, send_from_directory, jsonify, send_file
from flask_cors import CORS

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
VIEWER_DIR = os.path.join(PROJECT_DIR, "viewer")

app = Flask(__name__)
CORS(app)

# Will be set via command-line args
OUTPUT_DIR = None


@app.route("/")
def index():
    return send_from_directory(VIEWER_DIR, "index.html")


@app.route("/viewer/<path:filename>")
def serve_viewer(filename):
    return send_from_directory(VIEWER_DIR, filename)


@app.route("/api/files")
def list_files():
    """List available output files."""
    files = {}
    if OUTPUT_DIR and os.path.isdir(OUTPUT_DIR):
        for root, dirs, filenames in os.walk(OUTPUT_DIR):
            for f in filenames:
                rel_path = os.path.relpath(os.path.join(root, f), OUTPUT_DIR)
                files[rel_path] = os.path.join(root, f)
    return jsonify(list(files.keys()))


@app.route("/api/file/<path:filepath>")
def serve_file(filepath):
    """Serve a file from the output directory."""
    full_path = os.path.join(OUTPUT_DIR, filepath)
    if not os.path.exists(full_path):
        return jsonify({"error": "File not found"}), 404
    return send_file(full_path)


@app.route("/api/measurements")
def get_measurements():
    """Return clinical measurements JSON."""
    meas_path = os.path.join(OUTPUT_DIR, "clinical_measurements.json")
    if os.path.exists(meas_path):
        with open(meas_path, "r") as f:
            return jsonify(json.load(f))
    return jsonify({"error": "No measurements found"}), 404


@app.route("/api/pipeline_results")
def get_pipeline_results():
    """Return pipeline results JSON."""
    results_path = os.path.join(OUTPUT_DIR, "pipeline_results.json")
    if os.path.exists(results_path):
        with open(results_path, "r") as f:
            return jsonify(json.load(f))
    return jsonify({"error": "No pipeline results found"}), 404


def main():
    global OUTPUT_DIR

    parser = argparse.ArgumentParser(description="Serve 3D viewer with pipeline outputs")
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Pipeline output directory to serve",
    )
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--no_browser", action="store_true")
    args = parser.parse_args()

    OUTPUT_DIR = os.path.abspath(args.output_dir)
    print(f"Serving output from: {OUTPUT_DIR}")
    print(f"Viewer: http://localhost:{args.port}")

    if not args.no_browser:
        webbrowser.open(f"http://localhost:{args.port}")

    app.run(host="0.0.0.0", port=args.port, debug=False)


if __name__ == "__main__":
    main()
