import argparse
import json
import os
import socket
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

from flask import Flask, abort, jsonify, redirect, render_template_string, request, send_file, url_for


APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent
SRC_DIR = PROJECT_DIR / "src"
VIEWER_HTML_PATH = PROJECT_DIR / "viewer" / "index.html"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

app = Flask(__name__)

JOBS = {}
JOBS_LOCK = threading.Lock()
PICKERS = {}
PICKER_PORTS = {
    "scale": 8090,
    "anterior": 8091,
    "seed": 8092,
}


INDEX_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Leg + Fundal Pipeline Front End</title>
  <style>
    :root {
      --bg: #f2eee7;
      --ink: #172126;
      --muted: #59666d;
      --panel: #fffaf3;
      --line: #d4cabd;
      --accent: #0f766e;
      --accent-2: #b45309;
      --danger: #b42318;
      --ok: #13795b;
      --shadow: 0 18px 50px rgba(20, 24, 28, 0.12);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(15,118,110,0.09), transparent 28%),
        radial-gradient(circle at top right, rgba(180,83,9,0.12), transparent 24%),
        linear-gradient(180deg, #f8f4ee, var(--bg));
    }
    .shell {
      max-width: 1500px;
      margin: 0 auto;
      padding: 28px;
    }
    .hero {
      display: grid;
      grid-template-columns: 1.2fr 0.8fr;
      gap: 24px;
      margin-bottom: 24px;
    }
    .card {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 20px;
      box-shadow: var(--shadow);
    }
    .hero-copy {
      padding: 28px;
      min-height: 220px;
      background:
        linear-gradient(135deg, rgba(15,118,110,0.09), rgba(180,83,9,0.12)),
        var(--panel);
    }
    .hero-copy h1 {
      font-size: clamp(2rem, 4vw, 3.4rem);
      margin: 0 0 10px;
      line-height: 0.95;
      letter-spacing: -0.04em;
    }
    .hero-copy p {
      margin: 0;
      max-width: 50rem;
      color: var(--muted);
      font-size: 1rem;
      line-height: 1.6;
    }
    .hero-note {
      padding: 28px;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
    }
    .hero-note h2 {
      margin: 0 0 10px;
      font-size: 1rem;
      text-transform: uppercase;
      letter-spacing: 0.12em;
      color: var(--accent);
    }
    .hero-note ul {
      margin: 0;
      padding-left: 18px;
      color: var(--muted);
      line-height: 1.8;
    }
    .layout {
      display: grid;
      grid-template-columns: 460px 1fr;
      gap: 24px;
      align-items: start;
    }
    .panel {
      padding: 22px;
    }
    h3 {
      margin: 0 0 16px;
      font-size: 1.15rem;
      letter-spacing: -0.02em;
    }
    .section {
      margin-bottom: 18px;
      padding-bottom: 18px;
      border-bottom: 1px solid var(--line);
    }
    .section:last-child {
      margin-bottom: 0;
      padding-bottom: 0;
      border-bottom: 0;
    }
    .row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
      margin-bottom: 12px;
    }
    .row.full {
      grid-template-columns: 1fr;
    }
    label {
      display: block;
      font-size: 0.83rem;
      margin-bottom: 6px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }
    input, select, textarea {
      width: 100%;
      border: 1px solid var(--line);
      background: white;
      color: var(--ink);
      border-radius: 12px;
      padding: 12px 13px;
      font: inherit;
    }
    textarea {
      min-height: 88px;
      resize: vertical;
    }
    .button-row {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
    }
    button, .link-button {
      appearance: none;
      border: 0;
      border-radius: 999px;
      padding: 11px 16px;
      background: var(--accent);
      color: white;
      cursor: pointer;
      font: inherit;
      text-decoration: none;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      transition: transform 0.12s ease, opacity 0.12s ease;
    }
    button.alt, .link-button.alt { background: var(--accent-2); }
    button.ghost, .link-button.ghost {
      background: transparent;
      color: var(--ink);
      border: 1px solid var(--line);
    }
    button:hover, .link-button:hover { transform: translateY(-1px); }
    .muted {
      color: var(--muted);
      font-size: 0.94rem;
      line-height: 1.55;
    }
    .status-chip {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 8px 12px;
      border-radius: 999px;
      font-size: 0.92rem;
      background: #efe7d7;
      color: var(--ink);
    }
    .status-chip.running { background: #dff3ef; color: #0a5d57; }
    .status-chip.completed { background: #dbefe8; color: var(--ok); }
    .status-chip.failed { background: #f8dfdf; color: var(--danger); }
    .summary {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
      margin-top: 14px;
    }
    .metric {
      background: white;
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 14px;
    }
    .metric .k {
      color: var(--muted);
      font-size: 0.78rem;
      text-transform: uppercase;
      letter-spacing: 0.1em;
      margin-bottom: 6px;
    }
    .metric .v {
      font-size: 1rem;
      line-height: 1.4;
    }
    .log-box {
      background: #101616;
      color: #d8f5ef;
      border-radius: 16px;
      padding: 16px;
      min-height: 300px;
      max-height: 500px;
      overflow: auto;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
      line-height: 1.5;
      white-space: pre-wrap;
      border: 1px solid #223030;
    }
    .iframe-wrap {
      border: 1px solid var(--line);
      border-radius: 18px;
      overflow: hidden;
      background: white;
      margin-top: 16px;
      min-height: 560px;
    }
    iframe {
      width: 100%;
      height: 820px;
      border: 0;
      background: white;
      display: block;
    }
    .outputs {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 10px;
      margin-top: 14px;
      align-items: center;
    }
    .hint {
      margin-top: 10px;
      padding: 12px 14px;
      border-radius: 12px;
      background: #fbf4e8;
      border: 1px solid #ebdcc5;
      color: #755530;
      font-size: 0.92rem;
      line-height: 1.5;
    }
    .hidden { display: none; }
    @media (max-width: 1100px) {
      .hero, .layout { grid-template-columns: 1fr; }
      iframe { height: 640px; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <div class="hero">
      <section class="card hero-copy">
        <h1>Leg and Fundal Pipeline Front End</h1>
        <p>
          This wrapper keeps the full workflow in one place: choose the pipeline,
          launch the manual scaler, launch the leg anterior-frame picker or fundal seed picker,
          run the pipeline, then open the result you want from the finished output folder.
        </p>
      </section>
      <aside class="card hero-note">
        <div>
          <h2>Expected flow</h2>
          <ul>
            <li>Pick `leg` or `fundal`.</li>
            <li>Point the app at a video or extracted frames directory.</li>
            <li>Use the embedded picker tools before the main run.</li>
            <li>Start the job and watch the orchestrator log stream.</li>
            <li>Open summaries, galleries, JSON outputs, or the 3D viewer.</li>
          </ul>
        </div>
      </aside>
    </div>

    <div class="layout">
      <section class="card panel">
        <h3>Run Setup</h3>

        <div class="section">
          <div class="row">
            <div>
              <label for="pipeline_kind">Pipeline</label>
              <select id="pipeline_kind">
                <option value="leg">Leg pipeline</option>
                <option value="fundal">Fundal pipeline</option>
              </select>
            </div>
            <div>
              <label for="subject">Subject preset</label>
              <select id="subject"></select>
            </div>
          </div>

          <div class="row full">
            <div>
              <label for="input_path">Input path</label>
              <input id="input_path" type="text" placeholder="e.g. data/input/patient_005.mp4">
            </div>
          </div>

          <div class="row full">
            <div>
              <label for="output_dir">Output directory</label>
              <input id="output_dir" type="text" placeholder="e.g. data/output/patient_005">
            </div>
          </div>

          <div class="button-row">
            <button class="ghost" id="btn_defaults">Derive default paths</button>
          </div>
        </div>

        <div class="section">
          <div class="row">
            <div>
              <label for="scale_path">Scale calibration JSON</label>
              <input id="scale_path" type="text">
            </div>
            <div id="seed_wrap">
              <label for="use_seed_picker">Fundal seed input</label>
              <select id="use_seed_picker">
                <option value="no_seed">Do not use seed</option>
                <option value="use_seed">Use seed picker / seed JSON</option>
              </select>
            </div>
          </div>

          <div class="row">
            <div id="seed_path_wrap">
              <label for="seed_path">Fundal seed JSON</label>
              <input id="seed_path" type="text">
            </div>
            <div id="anterior_wrap">
              <label for="anterior_json">Anterior picker JSON</label>
              <input id="anterior_json" type="text">
            </div>
          </div>

          <div class="row">
            <div>
              <label for="anterior_frame">Anterior frame override</label>
              <input id="anterior_frame" type="text" placeholder="Optional direct frame filename">
            </div>
            <div></div>
          </div>

          <div class="button-row">
            <button id="btn_scale">Open Scale Picker</button>
            <button class="alt" id="btn_anterior">Open Anterior Picker</button>
            <button class="alt" id="btn_seed">Open Fundal Seed Picker</button>
          </div>
          <div class="hint">
            The pickers run as their existing Flask tools and open in a new browser tab.
          </div>
        </div>

        <div class="section">
          <div class="row">
            <div>
              <label for="n_frames">Extracted frames</label>
              <input id="n_frames" type="number" min="1" value="30">
            </div>
            <div>
              <label for="recon_max_frames">Reconstruction frame cap</label>
              <input id="recon_max_frames" type="number" min="1" value="20">
            </div>
          </div>
          <div class="button-row">
            <button class="ghost" id="btn_run">Run Pipeline</button>
          </div>
        </div>
      </section>

      <section class="card panel">
        <h3>Live Run</h3>
        <div id="job_status" class="status-chip">No run started</div>
        <div id="job_summary" class="summary hidden"></div>
        <div id="job_hint" class="hint">
          After you start a run, this panel streams the orchestrator output and lets you view one selected output at a time.
        </div>
        <div id="picker_status" class="hint hidden"></div>
        <div class="outputs">
          <select id="job_output_select" disabled>
            <option value="">Select an output to view</option>
          </select>
          <button id="btn_open_output" class="ghost" type="button" disabled>Open Output</button>
        </div>
        <div id="output_viewer_wrap" class="iframe-wrap hidden">
          <iframe id="output_viewer" src="about:blank"></iframe>
        </div>
        <div class="section" style="margin-top:18px;">
          <div class="log-box" id="job_log">Waiting for a job...</div>
        </div>
      </section>
    </div>
  </div>

  <script>
    const subjectOptions = {
      leg: [{value: "standing", label: "standing"}],
      fundal: [
        {value: "pregnant", label: "pregnant"},
        {value: "balloon", label: "balloon"},
        {value: "balloon_held", label: "balloon held"}
      ]
    };

    let currentJobId = null;
    let pollHandle = null;
    let currentOutputs = [];

    const el = (id) => document.getElementById(id);

    function setSubjects() {
      const kind = el("pipeline_kind").value;
      const subject = el("subject");
      subject.innerHTML = "";
      for (const opt of subjectOptions[kind]) {
        const option = document.createElement("option");
        option.value = opt.value;
        option.textContent = opt.label;
        subject.appendChild(option);
      }
      syncVisibility();
    }

    function syncVisibility() {
      const kind = el("pipeline_kind").value;
      el("seed_wrap").classList.toggle("hidden", kind !== "fundal");
      el("seed_path_wrap").classList.toggle("hidden", kind !== "fundal" || el("use_seed_picker").value !== "use_seed");
      el("btn_seed").classList.toggle("hidden", kind !== "fundal");
      el("btn_seed").disabled = kind !== "fundal" || el("use_seed_picker").value !== "use_seed";
      el("anterior_wrap").classList.toggle("hidden", kind !== "leg");
      el("btn_anterior").classList.toggle("hidden", kind !== "leg");
      if (kind !== "fundal") {
        el("use_seed_picker").value = "no_seed";
      }
    }

    async function deriveDefaults() {
      const res = await fetch("/api/derive_paths", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({
          pipeline_kind: el("pipeline_kind").value,
          input_path: el("input_path").value
        })
      });
      const data = await res.json();
      if (data.error) {
        alert(data.error);
        return;
      }
      el("output_dir").value = data.output_dir || "";
      el("scale_path").value = data.scale_path || "";
      el("seed_path").value = data.seed_path || "";
      el("anterior_json").value = data.anterior_json || "";
    }

    function payloadBase() {
      return {
        pipeline_kind: el("pipeline_kind").value,
        subject: el("subject").value,
        input_path: el("input_path").value.trim(),
        output_dir: el("output_dir").value.trim(),
        scale_path: el("scale_path").value.trim(),
        seed_path: el("use_seed_picker").value === "use_seed" ? el("seed_path").value.trim() : "",
        use_seed_picker: el("use_seed_picker").value,
        anterior_json: el("anterior_json").value.trim(),
        anterior_frame: el("anterior_frame").value.trim(),
        n_frames: Number(el("n_frames").value),
        recon_max_frames: Number(el("recon_max_frames").value),
        recon_model: "vggt",
        pose_model: "human",
        sam_confidence: 0.25,
        seed_usage: "default",
        poisson_depth: 8,
        conf_pct_keep: 75,
        sam_fallback_prompts: ""
      };
    }

    async function launchPicker(kind) {
      const labels = {
        scale: "scale picker",
        anterior: "anterior picker",
        seed: "fundal seed picker"
      };
      const pickerTab = window.open("", "_blank");
      if (pickerTab) {
        pickerTab.document.write(`
          <!doctype html>
          <html>
          <head>
            <meta charset="utf-8">
            <title>Preparing ${labels[kind] || "picker"}</title>
            <style>
              body {
                margin: 0;
                min-height: 100vh;
                display: grid;
                place-items: center;
                background: #f5f0e8;
                color: #172126;
                font-family: Georgia, "Times New Roman", serif;
              }
              .card {
                width: min(640px, calc(100vw - 48px));
                background: #fffaf3;
                border: 1px solid #d6cabd;
                border-radius: 20px;
                padding: 28px;
                box-shadow: 0 18px 50px rgba(20, 24, 28, 0.08);
              }
              h1 { margin: 0 0 12px; font-size: 1.8rem; }
              p { margin: 0; color: #5e6870; line-height: 1.7; }
            </style>
          </head>
          <body>
            <div class="card">
              <h1>Preparing ${labels[kind] || "picker"}</h1>
              <p>Frame extraction and picker startup are running. This tab will redirect automatically when the picker is ready.</p>
            </div>
          </body>
          </html>
        `);
        pickerTab.document.close();
      }

      const pickerStatus = el("picker_status");
      pickerStatus.classList.remove("hidden");
      pickerStatus.textContent = `Preparing ${labels[kind] || "picker"}: extracting frames if needed, then starting the picker server.`;

      const res = await fetch("/api/picker/start", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({...payloadBase(), picker_kind: kind})
      });
      const data = await res.json();
      if (data.error) {
        pickerStatus.textContent = `Failed to open ${labels[kind] || "picker"}.`;
        if (pickerTab) {
          pickerTab.document.body.innerHTML = `
            <div style="margin:0;min-height:100vh;display:grid;place-items:center;background:#fdf1f1;color:#7a271a;font-family:Georgia, 'Times New Roman', serif;">
              <div style="width:min(760px,calc(100vw - 48px));background:white;border:1px solid #efc6c6;border-radius:20px;padding:28px;box-shadow:0 18px 50px rgba(20,24,28,0.08);">
                <h1 style="margin:0 0 12px;font-size:1.8rem;">Picker failed to start</h1>
                <pre style="white-space:pre-wrap;word-break:break-word;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:13px;line-height:1.6;">${String(data.error).replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;")}</pre>
              </div>
            </div>`;
        }
        alert(data.error);
        return;
      }
      if (kind === "anterior" && data.output_path) {
        el("anterior_json").value = data.output_path;
      }
      if (kind === "scale" && data.output_path) {
        el("scale_path").value = data.output_path;
      }
      if (kind === "seed" && data.output_path) {
        el("seed_path").value = data.output_path;
        el("use_seed_picker").value = "use_seed";
        syncVisibility();
      }
      pickerStatus.textContent = `${labels[kind] || "Picker"} is being prepared in a new tab.`;
      if (pickerTab) {
        pickerTab.location.replace(data.loading_url);
      } else {
        window.open(data.loading_url, "_blank", "noopener,noreferrer");
      }
    }

    async function startJob() {
      const res = await fetch("/api/jobs/start", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(payloadBase())
      });
      const data = await res.json();
      if (data.error) {
        alert(data.error);
        return;
      }
      currentJobId = data.job_id;
      pollJob(true);
      if (pollHandle) window.clearInterval(pollHandle);
      pollHandle = window.setInterval(() => pollJob(false), 2000);
    }

    function renderSummary(items) {
      const box = el("job_summary");
      if (!items || !items.length) {
        box.classList.add("hidden");
        box.innerHTML = "";
        return;
      }
      box.classList.remove("hidden");
      box.innerHTML = items.map(item => `
        <div class="metric">
          <div class="k">${item.label}</div>
          <div class="v">${item.value}</div>
        </div>
      `).join("");
    }

    function renderOutputs(outputs) {
      currentOutputs = outputs || [];
      const select = el("job_output_select");
      const openBtn = el("btn_open_output");
      select.innerHTML = '<option value="">Select an output to view</option>';
      currentOutputs.forEach((item, idx) => {
        const option = document.createElement("option");
        option.value = String(idx);
        option.textContent = item.label;
        select.appendChild(option);
      });
      const enabled = currentOutputs.length > 0;
      select.disabled = !enabled;
      openBtn.disabled = !enabled;
      if (enabled && select.value === "") {
        select.value = "0";
      }
    }

    function openSelectedOutput() {
      const idx = el("job_output_select").value;
      if (idx === "") return;
      const item = currentOutputs[Number(idx)];
      if (!item) return;
      openOutput(item.url);
    }

    function openOutput(url) {
      const wrap = el("output_viewer_wrap");
      const frame = el("output_viewer");
      wrap.classList.remove("hidden");
      frame.src = url;
      wrap.scrollIntoView({behavior: "smooth", block: "nearest"});
    }

    function resizeViewerFrame() {
      const frame = el("output_viewer");
      try {
        const doc = frame.contentWindow.document;
        if (!doc || !doc.body) return;
        const bodyH = Math.max(
          doc.body.scrollHeight || 0,
          doc.documentElement ? doc.documentElement.scrollHeight || 0 : 0
        );
        frame.style.height = Math.max(620, Math.min(bodyH + 24, 1400)) + "px";
      } catch (err) {
      }
    }

    async function pollJob(scrollToBottom) {
      if (!currentJobId) return;
      const res = await fetch(`/api/jobs/${currentJobId}`);
      if (!res.ok) return;
      const data = await res.json();

      const chip = el("job_status");
      chip.textContent = `${data.status.toUpperCase()} · ${data.pipeline_kind} · ${data.output_dir}`;
      chip.className = `status-chip ${data.status}`;

      const logBox = el("job_log");
      logBox.textContent = (data.logs || []).join("");
      if (scrollToBottom) {
        logBox.scrollTop = logBox.scrollHeight;
      }

      renderSummary(data.summary_cards || []);
      renderOutputs(data.outputs || []);

      if (data.status === "completed" || data.status === "failed") {
        if (pollHandle) {
          window.clearInterval(pollHandle);
          pollHandle = null;
        }
      }
    }

    el("pipeline_kind").addEventListener("change", setSubjects);
    el("use_seed_picker").addEventListener("change", syncVisibility);
    el("btn_defaults").addEventListener("click", deriveDefaults);
    el("btn_scale").addEventListener("click", () => launchPicker("scale"));
    el("btn_anterior").addEventListener("click", () => launchPicker("anterior"));
    el("btn_seed").addEventListener("click", () => launchPicker("seed"));
    el("btn_run").addEventListener("click", startJob);
    el("btn_open_output").addEventListener("click", openSelectedOutput);
    el("job_output_select").addEventListener("change", openSelectedOutput);
    el("output_viewer").addEventListener("load", resizeViewerFrame);

    setSubjects();
  </script>
</body>
</html>
"""


SUMMARY_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{{ title }}</title>
  <style>
    body {
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      background: #f5f0e8;
      color: #1a2328;
    }
    .wrap {
      max-width: 1100px;
      margin: 0 auto;
      padding: 28px;
    }
    .card {
      background: #fffaf3;
      border: 1px solid #d6cabd;
      border-radius: 20px;
      padding: 24px;
      box-shadow: 0 18px 50px rgba(20, 24, 28, 0.08);
      margin-bottom: 18px;
    }
    h1, h2 { margin-top: 0; }
    .headline {
      font-size: 2rem;
      letter-spacing: -0.03em;
      margin-bottom: 8px;
    }
    .muted { color: #5e6870; line-height: 1.6; }
    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 12px;
    }
    .metric {
      border: 1px solid #d6cabd;
      border-radius: 14px;
      padding: 14px;
      background: white;
    }
    .k {
      color: #5e6870;
      text-transform: uppercase;
      letter-spacing: 0.1em;
      font-size: 0.78rem;
      margin-bottom: 6px;
    }
    .v {
      font-size: 1rem;
      line-height: 1.4;
    }
    ul {
      margin: 0;
      padding-left: 18px;
      line-height: 1.7;
    }
    a {
      color: #0f766e;
      text-decoration: none;
    }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <div class="headline">{{ headline }}</div>
      <div class="muted">{{ subtitle }}</div>
    </div>
    <div class="card">
      <h2>Key Metrics</h2>
      <div class="grid">
        {% for item in cards %}
          <div class="metric">
            <div class="k">{{ item.label }}</div>
            <div class="v">{{ item.value }}</div>
          </div>
        {% endfor %}
      </div>
    </div>
    {% if notes %}
      <div class="card">
        <h2>Notes</h2>
        <ul>
          {% for note in notes %}
            <li>{{ note }}</li>
          {% endfor %}
        </ul>
      </div>
    {% endif %}
    <div class="card">
      <h2>Files</h2>
      <ul>
        {% for item in file_links %}
          <li><a target="_blank" rel="noopener noreferrer" href="{{ item.url }}">{{ item.label }}</a></li>
        {% endfor %}
      </ul>
    </div>
  </div>
</body>
</html>
"""


GALLERY_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{{ title }}</title>
  <style>
    body {
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      background: #f2eee7;
      color: #172126;
    }
    .wrap {
      max-width: 1400px;
      margin: 0 auto;
      padding: 24px;
    }
    h1 {
      margin: 0 0 8px;
      font-size: 2rem;
      letter-spacing: -0.03em;
    }
    p {
      margin: 0 0 18px;
      color: #5e6870;
      line-height: 1.6;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
      gap: 14px;
    }
    .card {
      background: #fffaf3;
      border: 1px solid #d6cabd;
      border-radius: 18px;
      overflow: hidden;
      box-shadow: 0 16px 42px rgba(20, 24, 28, 0.08);
    }
    img {
      display: block;
      width: 100%;
      background: white;
    }
    .meta {
      padding: 12px 14px;
      font-size: 0.94rem;
      word-break: break-word;
    }
    a {
      color: #0f766e;
      text-decoration: none;
    }
  </style>
</head>
<body>
  <div class="wrap">
    <h1>{{ title }}</h1>
    <p>{{ description }}</p>
    <div class="grid">
      {% for image in images %}
        <div class="card">
          <a href="{{ image.url }}" target="_blank" rel="noopener noreferrer">
            <img src="{{ image.url }}" alt="{{ image.name }}">
          </a>
          <div class="meta">{{ image.name }}</div>
        </div>
      {% endfor %}
    </div>
  </div>
</body>
</html>
"""


JSON_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{{ title }}</title>
  <style>
    body {
      margin: 0;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      background: #111718;
      color: #d8f5ef;
    }
    pre {
      margin: 0;
      padding: 24px;
      white-space: pre-wrap;
      word-break: break-word;
      line-height: 1.6;
      font-size: 13px;
    }
  </style>
</head>
<body>
  <pre>{{ payload }}</pre>
</body>
</html>
"""


FILE_PREVIEW_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{{ title }}</title>
  <style>
    body {
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      background: #f2eee7;
      color: #172126;
    }
    .wrap {
      max-width: 1400px;
      margin: 0 auto;
      padding: 24px;
    }
    .card {
      background: #fffaf3;
      border: 1px solid #d6cabd;
      border-radius: 20px;
      padding: 20px;
      box-shadow: 0 18px 50px rgba(20, 24, 28, 0.08);
    }
    h1 {
      margin: 0 0 8px;
      font-size: 1.8rem;
      letter-spacing: -0.03em;
    }
    p {
      margin: 0 0 14px;
      color: #5e6870;
      line-height: 1.6;
    }
    .stage {
      display: grid;
      place-items: center;
      min-height: 70vh;
      background: #f7f2ea;
      border: 1px solid #eadfce;
      border-radius: 16px;
      overflow: hidden;
    }
    img {
      display: block;
      max-width: 100%;
      max-height: 78vh;
      width: auto;
      height: auto;
      object-fit: contain;
      background: white;
    }
    a {
      color: #0f766e;
      text-decoration: none;
    }
    pre {
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 13px;
      line-height: 1.6;
      color: #15201f;
    }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <h1>{{ title }}</h1>
      <p>{{ subtitle }}</p>
      {% if image_url %}
        <div class="stage">
          <img src="{{ image_url }}" alt="{{ title }}">
        </div>
      {% elif text_payload %}
        <div class="stage" style="place-items: stretch; min-height: auto;">
          <div style="padding:20px;"><pre>{{ text_payload }}</pre></div>
        </div>
      {% else %}
        <div class="stage">
          <div>
            <p>No inline preview available for this file.</p>
            <p><a href="{{ raw_url }}" target="_blank" rel="noopener noreferrer">Open raw file</a></p>
          </div>
        </div>
      {% endif %}
    </div>
  </div>
</body>
</html>
"""


def _project_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_DIR.resolve()))
    except ValueError:
        return str(path.resolve())


def _safe_input_flag(input_path: Path):
    if input_path.is_dir():
        return ["--image_dir", str(input_path)]
    if input_path.is_file():
        return ["--video", str(input_path)]
    raise FileNotFoundError(f"Input path not found: {input_path}")


def _derived_paths(pipeline_kind: str, input_path_raw: str):
    if not input_path_raw:
        return {}
    raw_path = Path(input_path_raw)
    input_path = raw_path if raw_path.is_absolute() else (PROJECT_DIR / raw_path)
    if input_path.is_dir():
        stem = input_path.name.rstrip("/\\")
        scale_path = input_path / "scale_calibration.json"
        seed_path = input_path / "belly_seed.json"
        anterior_path = input_path / "anterior_frame.json"
    else:
        stem = input_path.stem
        scale_path = input_path.with_name(f"{stem}_scale.json")
        seed_path = input_path.with_name(f"{stem}_seed.json")
        anterior_path = input_path.with_name(f"{stem}_anterior.json")
    output_dir = PROJECT_DIR / "data" / "output" / stem
    return {
        "output_dir": _project_relative(output_dir),
        "scale_path": _project_relative(scale_path),
        "seed_path": _project_relative(seed_path),
        "anterior_json": _project_relative(anterior_path),
    }


def _read_json(path: Path):
    if not path.exists():
        return None
    with open(path, "r") as f:
        return json.load(f)


def _candidate_paths(path_raw: str):
    if not path_raw:
        return []
    raw = Path(path_raw)
    candidates = []
    if raw.is_absolute():
        candidates.append(raw.resolve())
    else:
        candidates.append((PROJECT_DIR / raw).resolve())
        candidates.append(raw.resolve())
    deduped = []
    seen = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    return deduped


def _start_background_process(cmd, cwd: Path):
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    return subprocess.Popen(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )


def _wait_for_http_ready(url: str, process: subprocess.Popen, timeout_seconds: float = 90.0):
    deadline = time.time() + timeout_seconds
    last_error = None
    while time.time() < deadline:
        if process.poll() not in (None, 0):
            raise RuntimeError("Picker process exited before the server became ready.")
        try:
            with urlopen(url, timeout=2) as resp:
                if 200 <= getattr(resp, "status", 200) < 500:
                    return
        except URLError as exc:
            last_error = exc
        except Exception as exc:
            last_error = exc
        time.sleep(0.5)
    raise TimeoutError(f"Timed out waiting for picker server at {url}. Last error: {last_error}")


def _track_picker(key: str, process: subprocess.Popen, url: str):
    existing = PICKERS.get(key, {}).get("logs")
    log_buffer = deque(existing or [], maxlen=300)

    def _reader():
        if process.stdout is None:
            return
        for line in process.stdout:
            log_buffer.append(line)

    thread = threading.Thread(target=_reader, daemon=True)
    thread.start()
    info = PICKERS.setdefault(key, {})
    info["process"] = process
    info["url"] = url
    info["logs"] = log_buffer
    info["started_at"] = time.time()


def _picker_log(token: str, message: str):
    info = PICKERS.get(token)
    if info is not None:
        info["logs"].append(message.rstrip("\n"))


def _allocate_port(preferred: int) -> int:
    for port in range(preferred, preferred + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"Could not find a free port starting at {preferred}")


def _prepare_frames_if_needed(input_path: Path, n_frames: int, token: str):
    if input_path.is_dir():
        _picker_log(token, f"Using existing frames directory: {input_path}")
        return input_path, "--image_dir"

    if not input_path.is_file():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    from calibration.extract_frames import extract_frames, is_video_file

    if not is_video_file(str(input_path)):
        raise RuntimeError(f"Unsupported video file: {input_path}")

    frames_dir = Path(os.path.splitext(str(input_path))[0] + "_frames")
    manifest_path = frames_dir / "_video_frame_manifest.json"

    if manifest_path.exists():
        manifest = _read_json(manifest_path) or {}
        saved = manifest.get("saved", [])
        if len(saved) == n_frames:
            _picker_log(token, f"Reusing previously extracted frames in {frames_dir}")
            return frames_dir, "--image_dir"

    _picker_log(token, f"Extracting {n_frames} frames from video: {input_path.name}")
    extract_frames(str(input_path), str(frames_dir), n_frames=n_frames)
    _picker_log(token, f"Frame extraction complete: {frames_dir}")
    return frames_dir, "--image_dir"


def _picker_worker(token: str, data: dict):
    try:
        picker_kind = data["picker_kind"]
        input_path_raw = (data.get("input_path") or "").strip()
        if not input_path_raw:
            raise RuntimeError("Input path is required")

        input_path = (PROJECT_DIR / input_path_raw).resolve() if not Path(input_path_raw).is_absolute() else Path(input_path_raw).resolve()
        n_frames = int(data.get("n_frames", 30))
        prepared_input, input_flag = _prepare_frames_if_needed(input_path, n_frames, token)
        port = _allocate_port(PICKER_PORTS[picker_kind])
        PICKERS[token]["port"] = port
        PICKERS[token]["status_message"] = "Starting picker server"

        if picker_kind == "anterior":
            output_path_raw = (data.get("anterior_json") or "").strip()
            if not output_path_raw:
                output_path_raw = _derived_paths(data.get("pipeline_kind", "leg"), input_path_raw).get("anterior_json", "")
            output_path = (PROJECT_DIR / output_path_raw).resolve() if not Path(output_path_raw).is_absolute() else Path(output_path_raw).resolve()
            cmd = [
                sys.executable,
                str(SRC_DIR / "calibration" / "anterior_picker.py"),
                input_flag, str(prepared_input),
                "--output", str(output_path),
                "--port", str(port),
                "--n_frames", str(n_frames),
            ]
        else:
            if picker_kind == "seed":
                output_path_raw = (data.get("seed_path") or "").strip()
                if not output_path_raw:
                    output_path_raw = _derived_paths(data.get("pipeline_kind", "fundal"), input_path_raw).get("seed_path", "")
            else:
                output_path_raw = (data.get("scale_path") or "").strip()
                if not output_path_raw:
                    output_path_raw = _derived_paths(data.get("pipeline_kind", "leg"), input_path_raw).get("scale_path", "")
            output_path = (PROJECT_DIR / output_path_raw).resolve() if not Path(output_path_raw).is_absolute() else Path(output_path_raw).resolve()
            cmd = [
                sys.executable,
                str(SRC_DIR / "calibration" / "scale_picker.py"),
                input_flag, str(prepared_input),
                "--output", str(output_path),
                "--port", str(port),
            ]
            if picker_kind == "seed":
                cmd.extend(["--mode", "seed"])

        _picker_log(token, "Launching picker process")
        process = _start_background_process(cmd, PROJECT_DIR)
        url = f"http://127.0.0.1:{port}"
        PICKERS[token]["process"] = process
        PICKERS[token]["url"] = url
        PICKERS[token]["output_path"] = _project_relative(output_path)
        _track_picker(token, process, url)
        _wait_for_http_ready(url, process)
        PICKERS[token]["status"] = "ready"
        PICKERS[token]["status_message"] = "Picker is ready"
        _picker_log(token, f"Picker ready at {url}")
    except Exception as exc:
        info = PICKERS.get(token)
        if info is not None:
            info["status"] = "failed"
            info["status_message"] = str(exc)
            _picker_log(token, f"ERROR: {exc}")


def _ensure_within(base: Path, candidate: Path):
    base_resolved = base.resolve()
    candidate_resolved = candidate.resolve()
    try:
        candidate_resolved.relative_to(base_resolved)
    except ValueError as exc:
        raise PermissionError("Path escapes output directory") from exc
    return candidate_resolved


def _resolve_job(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if not job:
        abort(404)
    return job


def _summary_cards(job):
    output_dir = Path(job["output_dir"])
    pipeline = job["pipeline_kind"]
    cards = [
        {"label": "Pipeline", "value": pipeline},
        {"label": "Output Dir", "value": _project_relative(output_dir)},
        {"label": "Status", "value": job["status"]},
    ]
    if pipeline == "leg":
        data = _read_json(output_dir / "leg_assessment.json") or {}
        if data:
            left_vol = ((data.get("lower_leg_volume_left") or {}).get("volume_cm3"))
            right_vol = ((data.get("lower_leg_volume_right") or {}).get("volume_cm3"))
            cards.extend([
                {"label": "Overall Assessment", "value": data.get("overall_assessment") or "n/a"},
                {"label": "Method", "value": data.get("primary_method") or "n/a"},
                {"label": "Left Leg", "value": (data.get("left") or {}).get("classification") or "n/a"},
                {"label": "Right Leg", "value": (data.get("right") or {}).get("classification") or "n/a"},
                {"label": "Left Volume", "value": f"{left_vol:.0f} cm³" if isinstance(left_vol, (int, float)) else "n/a"},
                {"label": "Right Volume", "value": f"{right_vol:.0f} cm³" if isinstance(right_vol, (int, float)) else "n/a"},
            ])
    else:
        data = _read_json(output_dir / "belly" / "belly_results.json") or {}
        volume = ((data.get("volume") or {}).get("bulge_volume_cm3"))
        dist = ((data.get("distances") or {}).get("distance_belly_to_midfeet_cm"))
        cards.extend([
            {"label": "Bulge Volume", "value": f"{volume:.1f} cm³" if isinstance(volume, (int, float)) else "n/a"},
            {"label": "Belly To Midfeet", "value": f"{dist:.1f} cm" if isinstance(dist, (int, float)) else "n/a"},
        ])
    return cards


def _output_buttons(job_id: str, job):
    output_dir = Path(job["output_dir"])
    buttons = []
    debug_roots = []
    if (output_dir / "debug" / "leg").exists():
        debug_roots.append(output_dir / "debug" / "leg")
    if (output_dir / "debug" / "belly").exists():
        debug_roots.append(output_dir / "debug" / "belly")
    if (output_dir / "debug" / "reconstruction").exists():
        debug_roots.append(output_dir / "debug" / "reconstruction")

    for root in debug_roots:
        for path in sorted(root.glob("*")):
            if not path.is_file():
                continue
            rel_path = path.relative_to(output_dir)
            buttons.append({
                "label": f"Debug: {path.name}",
                "url": url_for("job_file_preview", job_id=job_id, rel_path=str(rel_path)),
                "variant": "ghost",
            })
    return buttons


def _resolve_existing_path(path_raw: str, fallback_raw: str = ""):
    searched = []
    for candidate in _candidate_paths(path_raw) + _candidate_paths(fallback_raw):
        sc = str(candidate)
        if sc not in searched:
            searched.append(sc)
        if candidate.exists():
            return candidate, searched
    return None, searched


def _build_pipeline_command(payload):
    pipeline_kind = payload["pipeline_kind"]
    input_path = (PROJECT_DIR / payload["input_path"]).resolve() if not Path(payload["input_path"]).is_absolute() else Path(payload["input_path"]).resolve()
    output_dir = (PROJECT_DIR / payload["output_dir"]).resolve() if not Path(payload["output_dir"]).is_absolute() else Path(payload["output_dir"]).resolve()
    cmd = [sys.executable]
    if pipeline_kind == "leg":
        cmd.append(str(SRC_DIR / "pipeline" / "leg_orchestrator.py"))
    else:
        cmd.append(str(SRC_DIR / "pipeline" / "belly_orchestrator.py"))
    cmd.extend(_safe_input_flag(input_path))
    cmd.extend([
        "--output_dir", str(output_dir),
        "--subject", payload["subject"],
        "--n_frames", str(payload["n_frames"]),
        "--recon_max_frames", str(payload["recon_max_frames"]),
        "--recon_model", payload["recon_model"],
        "--sam_confidence", str(payload["sam_confidence"]),
    ])

    scale_path = payload.get("scale_path", "").strip()
    derived_scale = _derived_paths(pipeline_kind, payload["input_path"]).get("scale_path", "")
    scale_abs, scale_searched = _resolve_existing_path(scale_path, derived_scale)
    if pipeline_kind in {"leg", "fundal"} and scale_abs is None:
        raise RuntimeError(
            "Volume-producing workflows require a saved scale calibration JSON. "
            f"Searched scale JSON paths: {scale_searched or ['<none>']}"
        )
    if scale_abs is not None:
        cmd.extend(["--scale_calibration", str(scale_abs)])

    if payload.get("sam_fallback_prompts"):
        cmd.extend(["--sam_fallback_prompts", payload["sam_fallback_prompts"]])

    if payload.get("seed_usage") == "force_on":
        cmd.append("--use_seed_points")
    elif payload.get("seed_usage") == "force_off":
        cmd.append("--no_seed_points")

    if pipeline_kind == "leg":
        cmd.extend(["--pose_model", payload["pose_model"]])
        anterior_frame = payload.get("anterior_frame", "").strip()
        anterior_json = payload.get("anterior_json", "").strip()
        searched_paths = []
        if not anterior_frame:
            derived = _derived_paths("leg", payload["input_path"]).get("anterior_json", "")
            for candidate in _candidate_paths(anterior_json) + _candidate_paths(derived):
                searched_paths.append(str(candidate))
                if not candidate.exists():
                    continue
                data = _read_json(candidate) or {}
                frame_name = str(data.get("anterior_frame") or "").strip()
                frame_idx = data.get("anterior_frame_idx")
                if frame_name:
                    anterior_frame = frame_name
                    break
                if frame_idx is not None and str(frame_idx).strip() != "":
                    anterior_frame = str(frame_idx).strip()
                    break
        if not anterior_frame:
            raise RuntimeError(
                "Leg pipeline requires an anterior-frame selection. "
                "Run the anterior picker first, or provide an anterior frame override. "
                f"Searched anterior JSON paths: {searched_paths or ['<none>']}"
            )
        if anterior_frame:
            cmd.extend(["--anterior_frame", anterior_frame])
    else:
        seed_path = payload.get("seed_path", "").strip()
        if seed_path:
            seed_abs = (PROJECT_DIR / seed_path).resolve() if not Path(seed_path).is_absolute() else Path(seed_path).resolve()
            cmd.extend(["--seed_points", str(seed_abs)])
        cmd.extend([
            "--poisson_depth", str(payload["poisson_depth"]),
            "--conf_pct_keep", str(payload["conf_pct_keep"]),
        ])

    return cmd, output_dir


def _job_worker(job_id: str, payload):
    try:
        cmd, output_dir = _build_pipeline_command(payload)
    except Exception as exc:
        with JOBS_LOCK:
            JOBS[job_id]["status"] = "failed"
            JOBS[job_id]["logs"].append(f"Failed to build command: {exc}\n")
        return

    process = _start_background_process(cmd, PROJECT_DIR)
    with JOBS_LOCK:
        JOBS[job_id]["process"] = process
        JOBS[job_id]["command"] = cmd
        JOBS[job_id]["output_dir"] = str(output_dir)

    if process.stdout is not None:
        for line in process.stdout:
            with JOBS_LOCK:
                JOBS[job_id]["logs"].append(line)

    return_code = process.wait()
    with JOBS_LOCK:
        JOBS[job_id]["finished_at"] = time.time()
        JOBS[job_id]["status"] = "completed" if return_code == 0 else "failed"
        JOBS[job_id]["return_code"] = return_code
        if return_code == 0:
            validation_error = _validate_volume_outputs(JOBS[job_id])
            if validation_error:
                JOBS[job_id]["status"] = "failed"
                JOBS[job_id]["logs"].append(validation_error + "\n")


def _validate_volume_outputs(job):
    output_dir = Path(job["output_dir"])
    if job["pipeline_kind"] == "leg":
        assessment = _read_json(output_dir / "leg_assessment.json") or {}
        left_vol = ((assessment.get("lower_leg_volume_left") or {}).get("volume_cm3"))
        right_vol = ((assessment.get("lower_leg_volume_right") or {}).get("volume_cm3"))
        slab_viz = output_dir / "debug" / "leg" / "lower_leg_volume_slabs.jpg"
        if not isinstance(left_vol, (int, float)) and not isinstance(right_vol, (int, float)):
            return (
                "Run finished without estimated lower-leg volume. "
                "Check that the scale calibration JSON contains saved clicks and that the capture produced a usable person cloud."
            )
        if not slab_viz.exists():
            return (
                "Run finished without lower-leg volume visualisation "
                f"({slab_viz})."
            )
    elif job["pipeline_kind"] == "fundal":
        results = _read_json(output_dir / "belly" / "belly_results.json") or {}
        vol = ((results.get("volume") or {}).get("bulge_volume_cm3"))
        if not isinstance(vol, (int, float)):
            return (
                "Run finished without a fundal volume estimate. "
                "Check the scale calibration and segmentation quality."
            )
    return None


@app.route("/")
def index():
    return render_template_string(INDEX_HTML)


@app.post("/api/derive_paths")
def derive_paths():
    data = request.get_json(force=True) or {}
    input_path = (data.get("input_path") or "").strip()
    if not input_path:
      return jsonify({"error": "input_path is required"}), 400
    return jsonify(_derived_paths(data.get("pipeline_kind", "leg"), input_path))


@app.post("/api/picker/start")
def start_picker():
    data = request.get_json(force=True) or {}
    picker_kind = (data.get("picker_kind") or "").strip()
    if picker_kind not in PICKER_PORTS:
        return jsonify({"error": "Unknown picker kind"}), 400
    token = uuid.uuid4().hex[:12]
    PICKERS[token] = {
        "token": token,
        "picker_kind": picker_kind,
        "status": "starting",
        "status_message": "Preparing picker",
        "logs": deque(maxlen=500),
        "process": None,
        "url": None,
        "port": None,
        "output_path": None,
        "started_at": time.time(),
    }
    thread = threading.Thread(target=_picker_worker, args=(token, data), daemon=True)
    thread.start()
    return jsonify({
        "ok": True,
        "token": token,
        "loading_url": url_for("picker_loading", token=token),
    })


@app.get("/api/picker/status/<token>")
def picker_status(token: str):
    info = PICKERS.get(token)
    if not info:
        return jsonify({"error": "Picker session not found"}), 404
    return jsonify({
        "token": token,
        "picker_kind": info["picker_kind"],
        "status": info["status"],
        "status_message": info["status_message"],
        "url": info["url"],
        "logs": list(info["logs"]),
        "output_path": info["output_path"],
    })


@app.get("/picker/loading/<token>")
def picker_loading(token: str):
    info = PICKERS.get(token)
    if not info:
        abort(404)
    return render_template_string(
        """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Preparing Picker</title>
  <style>
    body {
      margin: 0;
      min-height: 100vh;
      display: grid;
      place-items: center;
      background: #f5f0e8;
      color: #172126;
      font-family: Georgia, "Times New Roman", serif;
    }
    .card {
      width: min(760px, calc(100vw - 48px));
      background: #fffaf3;
      border: 1px solid #d6cabd;
      border-radius: 20px;
      padding: 28px;
      box-shadow: 0 18px 50px rgba(20, 24, 28, 0.08);
    }
    h1 { margin: 0 0 12px; font-size: 1.9rem; }
    p { margin: 0 0 12px; color: #5e6870; line-height: 1.7; }
    pre {
      margin: 16px 0 0;
      padding: 16px;
      border-radius: 14px;
      background: #15201f;
      color: #d8f5ef;
      min-height: 160px;
      white-space: pre-wrap;
      word-break: break-word;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
      line-height: 1.55;
    }
    .error {
      background: #fff1f1;
      color: #8d2c1d;
      border: 1px solid #f1c4c4;
      border-radius: 12px;
      padding: 12px 14px;
      margin-top: 14px;
      display: none;
    }
  </style>
</head>
<body>
  <div class="card">
    <h1>Preparing {{ picker_kind }}</h1>
    <p id="status">Starting…</p>
    <div id="error" class="error"></div>
    <pre id="logs">Waiting for updates…</pre>
  </div>
  <script>
    async function poll() {
      const resp = await fetch("{{ url_for('picker_status', token=token) }}");
      const data = await resp.json();
      document.getElementById("status").textContent = data.status_message || data.status;
      document.getElementById("logs").textContent = (data.logs || []).join("\\n");
      if (data.status === "ready" && data.url) {
        window.location.replace(data.url);
        return;
      }
      if (data.status === "failed") {
        const err = document.getElementById("error");
        err.style.display = "block";
        err.textContent = data.status_message || "Picker failed to start.";
        return;
      }
      setTimeout(poll, 1000);
    }
    poll();
  </script>
</body>
</html>
        """,
        token=token,
        picker_kind=info["picker_kind"],
    )


@app.post("/api/jobs/start")
def start_job():
    data = request.get_json(force=True) or {}
    required = ["pipeline_kind", "input_path", "output_dir", "subject"]
    missing = [key for key in required if not (data.get(key) or "").strip()]
    if missing:
        return jsonify({"error": f"Missing required fields: {', '.join(missing)}"}), 400

    job_id = uuid.uuid4().hex[:10]
    with JOBS_LOCK:
        JOBS[job_id] = {
            "job_id": job_id,
            "pipeline_kind": data["pipeline_kind"],
            "status": "running",
            "input_path": data["input_path"],
            "output_dir": data["output_dir"],
            "logs": deque([f"Starting {data['pipeline_kind']} pipeline...\n"], maxlen=4000),
            "started_at": time.time(),
            "finished_at": None,
            "return_code": None,
            "process": None,
            "command": [],
        }

    thread = threading.Thread(target=_job_worker, args=(job_id, data), daemon=True)
    thread.start()
    return jsonify({"ok": True, "job_id": job_id})


@app.get("/api/jobs/<job_id>")
def job_status(job_id: str):
    job = _resolve_job(job_id)
    payload = {
        "job_id": job["job_id"],
        "pipeline_kind": job["pipeline_kind"],
        "status": job["status"],
        "input_path": job["input_path"],
        "output_dir": job["output_dir"],
        "return_code": job["return_code"],
        "logs": list(job["logs"]),
        "summary_cards": _summary_cards(job),
        "outputs": _output_buttons(job_id, job) if job["status"] == "completed" else [],
    }
    return jsonify(payload)


@app.get("/jobs/<job_id>/summary")
def job_summary(job_id: str):
    job = _resolve_job(job_id)
    output_dir = Path(job["output_dir"])
    file_links = _output_buttons(job_id, job)
    notes = []

    if job["pipeline_kind"] == "leg":
        data = _read_json(output_dir / "leg_assessment.json") or {}
        headline = data.get("overall_assessment") or "Leg run summary"
        subtitle = f"Primary method: {data.get('primary_method') or 'n/a'}"
        notes = data.get("notes") or []
    else:
        data = _read_json(output_dir / "belly" / "belly_results.json") or {}
        volume = (data.get("volume") or {}).get("bulge_volume_cm3")
        headline = "Fundal run summary"
        subtitle = f"Bulge volume: {volume:.1f} cm³" if isinstance(volume, (int, float)) else "Volume not available"

    return render_template_string(
        SUMMARY_HTML,
        title=f"Run Summary · {job_id}",
        headline=headline,
        subtitle=subtitle,
        cards=_summary_cards(job),
        notes=notes,
        file_links=file_links,
    )


@app.get("/jobs/<job_id>/json/<path:rel_path>")
def job_json_view(job_id: str, rel_path: str):
    job = _resolve_job(job_id)
    output_dir = Path(job["output_dir"])
    full_path = _ensure_within(output_dir, output_dir / rel_path)
    if not full_path.exists():
        abort(404)
    payload = json.dumps(_read_json(full_path), indent=2)
    return render_template_string(JSON_HTML, title=rel_path, payload=payload)


@app.get("/jobs/<job_id>/gallery")
def job_gallery(job_id: str):
    job = _resolve_job(job_id)
    output_dir = Path(job["output_dir"])
    rel_root = request.args.get("rel_root", "")
    base = _ensure_within(output_dir, output_dir / rel_root)
    if not base.exists():
        abort(404)

    images = []
    for path in sorted(base.rglob("*")):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue
        rel_path = path.relative_to(output_dir)
        images.append({
            "name": str(rel_path),
            "url": url_for("job_file", job_id=job_id, rel_path=str(rel_path)),
        })

    return render_template_string(
        GALLERY_HTML,
        title=f"{rel_root or 'Output'} Gallery",
        description=f"Images from {rel_root or '.'}",
        images=images,
    )


@app.get("/jobs/<job_id>/file/<path:rel_path>")
def job_file(job_id: str, rel_path: str):
    job = _resolve_job(job_id)
    output_dir = Path(job["output_dir"])
    full_path = _ensure_within(output_dir, output_dir / rel_path)
    if not full_path.exists():
        abort(404)
    return send_file(full_path)


@app.get("/jobs/<job_id>/preview/<path:rel_path>")
def job_file_preview(job_id: str, rel_path: str):
    job = _resolve_job(job_id)
    output_dir = Path(job["output_dir"])
    full_path = _ensure_within(output_dir, output_dir / rel_path)
    if not full_path.exists():
        abort(404)

    suffix = full_path.suffix.lower()
    image_suffixes = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
    text_suffixes = {".json", ".txt", ".log", ".csv", ".md"}

    image_url = None
    text_payload = None
    if suffix in image_suffixes:
        image_url = url_for("job_file", job_id=job_id, rel_path=rel_path)
    elif suffix in text_suffixes:
        try:
            text_payload = full_path.read_text()
        except Exception:
            text_payload = None

    return render_template_string(
        FILE_PREVIEW_HTML,
        title=rel_path,
        subtitle=f"Preview of {rel_path}",
        image_url=image_url,
        text_payload=text_payload,
        raw_url=url_for("job_file", job_id=job_id, rel_path=rel_path),
    )


@app.get("/jobs/<job_id>/viewer/")
def job_viewer(job_id: str):
    _resolve_job(job_id)
    html = VIEWER_HTML_PATH.read_text()
    html = html.replace("/api/file/", "./api/file/")
    html = html.replace("fetch('/api/", "fetch('./api/")
    html = html.replace('fetch("/api/', 'fetch("./api/')
    return html


@app.get("/jobs/<job_id>/viewer/api/files")
def viewer_files(job_id: str):
    job = _resolve_job(job_id)
    output_dir = Path(job["output_dir"])
    files = []
    if output_dir.exists():
        for path in sorted(output_dir.rglob("*")):
            if path.is_file():
                files.append(str(path.relative_to(output_dir)))
    return jsonify(files)


@app.get("/jobs/<job_id>/viewer/api/file/<path:rel_path>")
def viewer_file(job_id: str, rel_path: str):
    return redirect(url_for("job_file", job_id=job_id, rel_path=rel_path))


@app.get("/jobs/<job_id>/viewer/api/measurements")
def viewer_measurements(job_id: str):
    _resolve_job(job_id)
    # The bundled viewer expects an older per-frame clinical JSON schema.
    # The wrapper uses dedicated summary pages for current leg/fundal metrics,
    # and keeps the 3D viewer focused on point clouds plus image artifacts.
    return jsonify({})


@app.get("/jobs/<job_id>/viewer/api/pipeline_results")
def viewer_pipeline_results(job_id: str):
    job = _resolve_job(job_id)
    output_dir = Path(job["output_dir"])
    data = _read_json(output_dir / "pipeline_results.json")
    return jsonify(data or {"error": "No pipeline results found"})


def main():
    parser = argparse.ArgumentParser(description="Frontend wrapper for the leg and fundal pipelines")
    parser.add_argument("--port", type=int, default=8070)
    args = parser.parse_args()
    app.run(host="0.0.0.0", port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
