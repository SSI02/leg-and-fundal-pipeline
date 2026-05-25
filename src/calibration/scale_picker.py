"""
Interactive Scale Picker — manual per-image pixel-to-metric calibration.

The user uploads images and clicks two points on a known-size object in each
image (e.g., a 30cm ruler, an ArUco marker side, a known body part).
The tool records:
    - Pixel coordinates of the two points
    - Real-world distance entered by user (in cm)
    - Computes scale_cm_per_pixel = real_distance / pixel_distance

Each image gets its OWN scale factor because images may be captured from
different camera distances. This per-image scale is far more accurate than
a single global scale averaged across all images.

Output: scale_calibration.json with structure:
{
    "front.jpg": {
        "p1": [120, 450],
        "p2": [180, 450],
        "real_distance_cm": 10.0,
        "pixel_distance": 60.0,
        "scale_cm_per_pixel": 0.1667,
        "object_description": "ArUco marker side"
    },
    ...
}

Usage:
    conda activate leg_pipeline
    python src/calibration/scale_picker.py \
        --image_dir data/input/patient_001 \
        --output data/input/patient_001/scale_calibration.json \
        [--port 8090]

Then open http://localhost:8090 in your browser.
"""

import os
import sys
import json
import glob
import argparse
import math

import numpy as np
from flask import Flask, request, jsonify, send_file, render_template_string

# Make sibling modules importable when running this script directly
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(_THIS_DIR)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

# Optional imports (only needed for tracking / video)
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Scale Picker</title>
<style>
  body { margin: 0; padding: 0; font-family: -apple-system, system-ui, sans-serif;
         background: #1a1a1a; color: #e0e0e0; display: flex; height: 100vh; }
  #sidebar { width: 280px; background: #252525; padding: 16px; overflow-y: auto;
             border-right: 1px solid #333; }
  #main { flex: 1; display: flex; flex-direction: column; padding: 16px;
          overflow: hidden; }
  h1 { font-size: 16px; margin: 0 0 12px 0; color: #4cafef; }
  h2 { font-size: 13px; margin: 16px 0 8px 0; color: #888;
       text-transform: uppercase; letter-spacing: 0.5px; }
  .image-list { list-style: none; padding: 0; margin: 0; }
  .image-list li { padding: 8px 12px; margin: 4px 0; cursor: pointer;
                   border-radius: 4px; font-size: 13px; transition: background 0.1s; }
  .image-list li:hover { background: #333; }
  .image-list li.active { background: #4cafef; color: #fff; }
  .image-list li.calibrated::before { content: "\2713 "; color: #4caf50; font-weight: bold; }
  .image-list li.uncalibrated::before { content: "\2022 "; color: #888; }
  #canvas-container { flex: 1; overflow: auto; background: #0d0d0d;
                      border: 1px solid #333; border-radius: 4px;
                      position: relative; }
  #canvas-wrap { display: inline-block; transform-origin: 0 0; }
  #canvas { cursor: crosshair; display: block; }
  #zoom-bar { display: flex; gap: 4px; padding: 6px 8px; background: #2a2a2a;
              border-bottom: 1px solid #333; align-items: center;
              position: sticky; top: 0; z-index: 5; }
  #zoom-bar button { padding: 4px 10px; font-size: 12px; background: #444; color: #fff;
                     border: none; border-radius: 3px; cursor: pointer; }
  #zoom-bar button:hover { background: #555; }
  #zoom-bar span { font-size: 12px; color: #aaa; margin-left: 6px; }
  .magnifier { position: absolute; pointer-events: none;
                width: 180px; height: 180px; border: 2px solid #4cafef;
                border-radius: 50%; overflow: hidden; display: none;
                z-index: 10; background: #000; }
  .magnifier-canvas { display: block; }
  .crosshair-h, .crosshair-v { position: absolute; background: #ff5252; }
  .crosshair-h { left: 0; right: 0; height: 1px; top: 50%; }
  .crosshair-v { top: 0; bottom: 0; width: 1px; left: 50%; }
  #controls { padding: 12px 0; display: flex; gap: 8px; align-items: center;
              flex-wrap: wrap; }
  button { padding: 8px 14px; background: #4cafef; color: #fff; border: none;
           border-radius: 4px; cursor: pointer; font-size: 13px; }
  button:hover { background: #5fb8f0; }
  button.secondary { background: #444; }
  button.secondary:hover { background: #555; }
  button:disabled { background: #333; color: #666; cursor: not-allowed; }
  input[type="number"], input[type="text"] {
    padding: 7px 10px; background: #1a1a1a; color: #e0e0e0;
    border: 1px solid #444; border-radius: 4px; font-size: 13px; width: 90px;
  }
  input[type="text"] { width: 200px; }
  label { font-size: 13px; color: #aaa; }
  #info { padding: 10px 12px; background: #2a2a2a; border-radius: 4px;
          font-size: 12px; color: #aaa; line-height: 1.5; }
  #info code { color: #4cafef; background: #1a1a1a; padding: 1px 5px;
                border-radius: 3px; }
  #status { font-size: 12px; color: #4caf50; margin-left: auto; }
  .pt-info { display: inline-block; padding: 4px 10px; background: #1a1a1a;
             border-radius: 3px; font-size: 12px; margin-right: 6px; }
  .scale-display { font-weight: bold; color: #4cafef; }
</style>
</head>
<body>
<div id="sidebar">
  <h1>Scale Picker</h1>
  <div id="info">
    <b>How to use:</b><br>
    1. Click an image<br>
    2. Zoom in (wheel / 1:1 / 2x / 4x) for pixel accuracy<br>
    3. Click 2 points on a known object<br>
    4. Enter real distance in cm<br>
    5. <code>Save</code> single, or <code>Track to all</code> for batch<br><br>
    <b>Controls:</b> Wheel = zoom, Shift+drag = pan, M = magnifier toggle.<br><br>
    <b>Track method:</b> <span id="track-method-info">checking...</span>
  </div>
  <h2>Images</h2>
  <ul class="image-list" id="image-list"></ul>
  <h2>Actions</h2>
  <button onclick="trackToAll()" style="width:100%">Track to all frames</button>
  <div style="height: 6px;"></div>
  <button class="secondary" onclick="downloadCalib()" style="width:100%">Download JSON</button>
  <div style="height: 6px;"></div>
  <button class="secondary" onclick="exportSummary()" style="width:100%">Print Summary</button>
</div>

<div id="main">
  <div id="zoom-bar">
    <button onclick="setZoom('fit')" title="Fit to screen">⤢ Fit</button>
    <button onclick="setZoom(1.0)" title="100% (1 image px = 1 screen px)">1:1</button>
    <button onclick="zoomBy(0.5)" title="Zoom out">−</button>
    <button onclick="zoomBy(2.0)" title="Zoom in">+</button>
    <button onclick="setZoom(2.0)">2x</button>
    <button onclick="setZoom(4.0)">4x</button>
    <button onclick="setZoom(8.0)">8x</button>
    <span id="zoom-info">Zoom: fit</span>
    <span style="margin-left: 12px;">·  Wheel = zoom, Shift+drag = pan, M = magnifier</span>
    <button id="mag-toggle" onclick="toggleMagnifier()" style="margin-left: auto;">Magnifier: OFF</button>
  </div>
  <div id="canvas-container">
    <div id="canvas-wrap">
      <canvas id="canvas"></canvas>
    </div>
    <div class="magnifier" id="magnifier">
      <canvas class="magnifier-canvas" id="mag-canvas" width="180" height="180"></canvas>
      <div class="crosshair-h"></div>
      <div class="crosshair-v"></div>
    </div>
  </div>
  <div id="controls">
    <span class="pt-info">P1: <span id="p1">none</span></span>
    <span class="pt-info">P2: <span id="p2">none</span></span>
    <span class="pt-info">Pixels: <span id="px">-</span></span>
    <label>Real distance (cm):</label>
    <input type="number" id="dist" step="0.1" min="0.1" placeholder="e.g. 10.0">
    <label>Object:</label>
    <input type="text" id="obj" placeholder="e.g. ArUco marker">
    <button onclick="resetPoints()" class="secondary">Reset</button>
    <button onclick="saveScale()">Save</button>
    <span id="status"></span>
    <span class="pt-info">Scale: <span class="scale-display" id="scale">-</span></span>
  </div>
</div>

<script>
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const wrap = document.getElementById('canvas-wrap');
const container = document.getElementById('canvas-container');
const zoomInfo = document.getElementById('zoom-info');
const magnifier = document.getElementById('magnifier');
const magCanvas = document.getElementById('mag-canvas');
const magCtx = magCanvas.getContext('2d');
const magToggle = document.getElementById('mag-toggle');
let img = null;
let p1 = null, p2 = null;
let currentImage = null;
let calibrations = {};
let zoomMode = 'fit';      // 'fit' or a number (1.0 = 1:1)
let currentZoom = 1.0;     // resolved numeric zoom factor (image px → screen px)
let panActive = false;
let panStart = null;
let scrollStart = null;
let magnifierOn = false;

function setZoom(z) {
  zoomMode = z;
  applyZoom();
}
function zoomBy(factor) {
  const z = (zoomMode === 'fit') ? currentZoom : zoomMode;
  setZoom(Math.max(0.05, Math.min(20, z * factor)));
}
function applyZoom() {
  if (!img) return;
  if (zoomMode === 'fit') {
    const cw = container.clientWidth;
    const ch = container.clientHeight;
    currentZoom = Math.min(cw / img.width, ch / img.height);
  } else {
    currentZoom = zoomMode;
  }
  wrap.style.width  = (img.width  * currentZoom) + 'px';
  wrap.style.height = (img.height * currentZoom) + 'px';
  wrap.style.transform = `scale(${currentZoom})`;
  // Use width/height on wrap so scrollbars work; zero out canvas's own size to image's natural size
  canvas.style.width  = img.width  + 'px';
  canvas.style.height = img.height + 'px';
  zoomInfo.textContent = 'Zoom: ' + (zoomMode === 'fit'
    ? `fit (${(currentZoom*100).toFixed(0)}%)` : `${(currentZoom*100).toFixed(0)}%`);
}
window.addEventListener('resize', () => { if (zoomMode === 'fit') applyZoom(); });

// Mouse wheel zoom centered on cursor
container.addEventListener('wheel', (e) => {
  if (e.ctrlKey) return;  // let browser handle Ctrl+wheel
  e.preventDefault();
  const rect = container.getBoundingClientRect();
  const mx = e.clientX - rect.left + container.scrollLeft;
  const my = e.clientY - rect.top + container.scrollTop;
  const oldZoom = currentZoom;
  const factor = e.deltaY < 0 ? 1.15 : 1/1.15;
  setZoom(Math.max(0.05, Math.min(20, oldZoom * factor)));
  // Adjust scroll so the point under the mouse stays stationary
  requestAnimationFrame(() => {
    const ratio = currentZoom / oldZoom;
    container.scrollLeft = mx * ratio - (e.clientX - rect.left);
    container.scrollTop  = my * ratio - (e.clientY - rect.top);
  });
}, { passive: false });

// Shift+drag = pan
container.addEventListener('mousedown', (e) => {
  if (e.shiftKey || e.button === 1) {
    panActive = true;
    panStart = { x: e.clientX, y: e.clientY };
    scrollStart = { x: container.scrollLeft, y: container.scrollTop };
    e.preventDefault();
  }
});
window.addEventListener('mousemove', (e) => {
  if (panActive) {
    container.scrollLeft = scrollStart.x - (e.clientX - panStart.x);
    container.scrollTop  = scrollStart.y - (e.clientY - panStart.y);
  }
});
window.addEventListener('mouseup', () => { panActive = false; });

// Magnifier toggle (M key or button)
function toggleMagnifier() {
  magnifierOn = !magnifierOn;
  magToggle.textContent = 'Magnifier: ' + (magnifierOn ? 'ON' : 'OFF');
  magnifier.style.display = magnifierOn ? 'block' : 'none';
}
window.addEventListener('keydown', (e) => {
  if (e.key === 'm' || e.key === 'M') toggleMagnifier();
});

// Magnifier follows mouse over canvas, showing 4x zoom of underlying image area
canvas.addEventListener('mousemove', (e) => {
  if (!magnifierOn || !img) return;
  const rect = canvas.getBoundingClientRect();
  const ix = (e.clientX - rect.left) * (canvas.width / rect.width);
  const iy = (e.clientY - rect.top) * (canvas.height / rect.height);
  const magZoom = 4;
  const magSize = 180;
  const srcSize = magSize / magZoom;  // source crop in image pixels
  magCtx.fillStyle = '#000';
  magCtx.fillRect(0, 0, magSize, magSize);
  magCtx.drawImage(img,
    ix - srcSize/2, iy - srcSize/2, srcSize, srcSize,
    0, 0, magSize, magSize);
  // Position magnifier near cursor but inside container
  const containerRect = container.getBoundingClientRect();
  let lx = e.clientX - containerRect.left + 20;
  let ly = e.clientY - containerRect.top + 20;
  if (lx + magSize > containerRect.width) lx = e.clientX - containerRect.left - magSize - 20;
  if (ly + magSize > containerRect.height) ly = e.clientY - containerRect.top - magSize - 20;
  magnifier.style.left = lx + 'px';
  magnifier.style.top = ly + 'px';
});
canvas.addEventListener('mouseleave', () => {
  if (magnifierOn) magnifier.style.display = 'none';
});
canvas.addEventListener('mouseenter', () => {
  if (magnifierOn) magnifier.style.display = 'block';
});

async function loadSourceInfo() {
  try {
    const r = await fetch('/api/source_info');
    const info = await r.json();
    pickerMode = info.mode || 'scale';
    const el = document.getElementById('track-method-info');
    let m = '';
    if (info.is_video) {
      m = '<b style="color:#4caf50">Video LK</b> · ';
    } else {
      m = '<b style="color:#ffaa00">ORB+RANSAC</b> · ';
    }
    if (pickerMode === 'seed') {
      m += '<b>Seed mode</b>: 1 point on belly per frame';
      // Hide distance/object inputs
      document.getElementById('dist').style.display = 'none';
      document.querySelectorAll('label').forEach(l => {
        if (l.textContent.includes('Real distance')) l.style.display = 'none';
      });
      // Rename Track button
      const tb = [...document.querySelectorAll('button')].find(b => b.textContent.includes('Track to all'));
      if (tb) tb.textContent = 'Track seed to all frames';
      const sv = [...document.querySelectorAll('button')].find(b => b.textContent === 'Save');
      if (sv) sv.textContent = 'Save seed';
    } else {
      m += '<b>Scale mode</b>: 2 points + real distance';
    }
    el.innerHTML = m;
  } catch(e) { /* ignore */ }
}

async function loadFiles() {
  const r = await fetch('/api/files');
  const files = await r.json();
  const r2 = await fetch('/api/calibrations');
  calibrations = await r2.json();
  const list = document.getElementById('image-list');
  list.innerHTML = '';
  files.forEach(f => {
    const li = document.createElement('li');
    li.textContent = f;
    li.className = calibrations[f] ? 'calibrated' : 'uncalibrated';
    li.onclick = () => selectImage(f);
    list.appendChild(li);
  });
  if (files.length > 0 && !currentImage) selectImage(files[0]);
}

function selectImage(name) {
  currentImage = name;
  document.querySelectorAll('.image-list li').forEach(li => {
    li.classList.toggle('active', li.textContent === name);
  });
  loadImage(name);
}

function loadImage(name) {
  img = new Image();
  img.onload = () => {
    canvas.width = img.width;
    canvas.height = img.height;
    applyZoom();
    redraw();
    // Restore previous calibration if any
    if (calibrations[name]) {
      p1 = calibrations[name].p1;
      p2 = calibrations[name].p2;
      document.getElementById('dist').value = calibrations[name].real_distance_cm || '';
      document.getElementById('obj').value = calibrations[name].object_description || '';
      redraw();
      updateInfo();
    } else {
      p1 = p2 = null;
      document.getElementById('dist').value = '';
      document.getElementById('obj').value = '';
      updateInfo();
    }
  };
  img.src = '/api/image/' + encodeURIComponent(name);
}

function redraw() {
  if (!img) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(img, 0, 0);
  if (p1) drawPoint(p1, '#ff5252', 'P1');
  if (p2) drawPoint(p2, '#4caf50', 'P2');
  if (p1 && p2) {
    ctx.strokeStyle = '#4cafef';
    ctx.lineWidth = Math.max(2, img.width / 800);
    ctx.beginPath();
    ctx.moveTo(p1[0], p1[1]);
    ctx.lineTo(p2[0], p2[1]);
    ctx.stroke();
  }
}

function drawPoint(p, color, label) {
  const r = Math.max(6, img.width / 200);
  ctx.fillStyle = color;
  ctx.strokeStyle = '#fff';
  ctx.lineWidth = Math.max(2, img.width / 1000);
  ctx.beginPath();
  ctx.arc(p[0], p[1], r, 0, 2 * Math.PI);
  ctx.fill();
  ctx.stroke();
  ctx.fillStyle = '#fff';
  ctx.font = 'bold ' + Math.max(14, img.width / 80) + 'px sans-serif';
  ctx.fillText(label, p[0] + r + 3, p[1] - r);
}

let pickerMode = 'scale';  // 'scale' or 'seed'

canvas.addEventListener('click', (e) => {
  if (e.shiftKey) return;  // Shift+click is pan, not point selection
  const rect = canvas.getBoundingClientRect();
  const x = (e.clientX - rect.left) * (canvas.width / rect.width);
  const y = (e.clientY - rect.top) * (canvas.height / rect.height);
  if (pickerMode === 'seed') {
    // Single-point mode: each click replaces the seed point
    p1 = [Math.round(x), Math.round(y)];
    p2 = null;
  } else {
    if (!p1) p1 = [Math.round(x), Math.round(y)];
    else if (!p2) p2 = [Math.round(x), Math.round(y)];
    else { p1 = [Math.round(x), Math.round(y)]; p2 = null; }
  }
  redraw();
  updateInfo();
});

function updateInfo() {
  document.getElementById('p1').textContent = p1 ? p1[0] + ', ' + p1[1] : 'none';
  document.getElementById('p2').textContent = p2 ? p2[0] + ', ' + p2[1] : 'none';
  if (p1 && p2) {
    const dx = p2[0] - p1[0], dy = p2[1] - p1[1];
    const px = Math.sqrt(dx*dx + dy*dy);
    document.getElementById('px').textContent = px.toFixed(1);
    const dist = parseFloat(document.getElementById('dist').value);
    if (dist && px > 0) {
      const scale = dist / px;
      document.getElementById('scale').textContent = scale.toFixed(5) + ' cm/px';
    } else {
      document.getElementById('scale').textContent = '(enter cm)';
    }
  } else {
    document.getElementById('px').textContent = '-';
    document.getElementById('scale').textContent = '-';
  }
}

document.getElementById('dist').addEventListener('input', updateInfo);

function resetPoints() {
  p1 = p2 = null;
  redraw();
  updateInfo();
}

async function saveScale() {
  if (!currentImage) return;
  if (pickerMode === 'seed') {
    if (!p1) { setStatus('Click 1 point on the belly', '#ff5252'); return; }
    const obj = document.getElementById('obj').value || 'belly';
    const data = { p1, object_description: obj, tracking: 'manual' };
    const r = await fetch('/api/save', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({image: currentImage, data: data}),
    });
    if (r.ok) {
      calibrations[currentImage] = data;
      setStatus('Saved ✓ seed=' + JSON.stringify(p1), '#4caf50');
      document.querySelectorAll('.image-list li').forEach(li => {
        if (li.textContent === currentImage) {
          li.classList.remove('uncalibrated');
          li.classList.add('calibrated');
        }
      });
    } else setStatus('Save failed', '#ff5252');
    return;
  }
  if (!p1 || !p2) { setStatus('Click 2 points first', '#ff5252'); return; }
  const dist = parseFloat(document.getElementById('dist').value);
  if (!dist || dist <= 0) { setStatus('Enter real distance in cm', '#ff5252'); return; }
  const obj = document.getElementById('obj').value || 'unknown';
  const dx = p2[0] - p1[0], dy = p2[1] - p1[1];
  const px = Math.sqrt(dx*dx + dy*dy);
  const scale = dist / px;
  const data = {
    p1, p2,
    real_distance_cm: dist,
    pixel_distance: px,
    scale_cm_per_pixel: scale,
    object_description: obj,
  };
  const r = await fetch('/api/save', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({image: currentImage, data: data})
  });
  if (r.ok) {
    calibrations[currentImage] = data;
    setStatus('Saved ✓ scale=' + scale.toFixed(5) + ' cm/px', '#4caf50');
    document.querySelectorAll('.image-list li').forEach(li => {
      if (li.textContent === currentImage) {
        li.classList.remove('uncalibrated');
        li.classList.add('calibrated');
      }
    });
  } else {
    setStatus('Save failed', '#ff5252');
  }
}

function setStatus(msg, color) {
  const el = document.getElementById('status');
  el.textContent = msg;
  el.style.color = color || '#4caf50';
  setTimeout(() => { el.textContent = ''; }, 4000);
}

async function trackToAll() {
  if (!currentImage) { setStatus('Select an image first', '#ff5252'); return; }
  // Seed mode: track a single point
  if (pickerMode === 'seed') {
    if (!p1) { setStatus('Click 1 point on belly first', '#ff5252'); return; }
    const obj = document.getElementById('obj').value || 'belly';
    setStatus('Tracking seed across all frames...', '#4cafef');
    const r = await fetch('/api/track_seed', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({src_image: currentImage, src_p: p1,
                            object_description: obj}),
    });
    if (!r.ok) { setStatus('Seed-tracking request failed', '#ff5252'); return; }
    const out = await r.json();
    setStatus('Seed tracked across ' + (out.n_tracked || 0) + ' frames', '#4caf50');
    const r2 = await fetch('/api/calibrations');
    calibrations = await r2.json();
    document.querySelectorAll('.image-list li').forEach(li => {
      const c = calibrations[li.textContent];
      li.classList.toggle('calibrated', !!c);
      li.classList.toggle('uncalibrated', !c);
    });
    return;
  }
  if (!p1 || !p2) { setStatus('Click 2 points on the source frame first', '#ff5252'); return; }
  const dist = parseFloat(document.getElementById('dist').value);
  if (!dist || dist <= 0) { setStatus('Enter real distance in cm', '#ff5252'); return; }
  const obj = document.getElementById('obj').value || 'unknown';

  // Build dst_images list from sidebar (excluding current source)
  const all = [...document.querySelectorAll('.image-list li')].map(li => li.textContent);
  const dst = all.filter(n => n !== currentImage);

  setStatus('Tracking ' + dst.length + ' frames...', '#4cafef');
  const r = await fetch('/api/track', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      src_image: currentImage,
      src_p1: p1,
      src_p2: p2,
      dst_images: dst,
      real_distance_cm: dist,
      object_description: obj,
    })
  });
  if (!r.ok) { setStatus('Tracking request failed', '#ff5252'); return; }
  const out = await r.json();
  const ok = out.n_tracked || 0;
  const fail = (out.results || []).filter(x => !x.tracking_ok);
  setStatus('Tracked ' + ok + '/' + (dst.length + 1) + ' frames'
            + (fail.length ? ', ' + fail.length + ' failed (manual click required)' : ''),
            fail.length ? '#ffaa00' : '#4caf50');

  // Build a detailed report
  let report = 'Tracking Results:\n' + '='.repeat(60) + '\n';
  for (const r of (out.results || [])) {
    if (r.tracking_ok) {
      const tag = r.tracking || '?';
      if (tag === 'manual') {
        report += `[SOURCE]  ${r.image}: ${r.pixel_distance.toFixed(1)}px → ${r.scale_cm_per_pixel.toFixed(5)} cm/px\n`;
      } else {
        report += `[OK]      ${r.image}: ${r.pixel_distance.toFixed(1)}px → ${r.scale_cm_per_pixel.toFixed(5)} cm/px`
               + ` (${r.n_inliers}/${r.n_good_matches} inliers ${(r.inlier_ratio*100).toFixed(0)}%, err ${r.median_reprojection_error_px}px)\n`;
      }
    } else {
      report += `[FAIL]    ${r.image}: ${r.error}\n`;
    }
  }
  report += '\nFor failed frames, please click the same object manually.';
  console.log(report);

  // Refresh calibrations and sidebar markers
  const r2 = await fetch('/api/calibrations');
  calibrations = await r2.json();
  document.querySelectorAll('.image-list li').forEach(li => {
    const c = calibrations[li.textContent];
    li.classList.toggle('calibrated', !!c);
    li.classList.toggle('uncalibrated', !c);
  });

  // Show failed frames in an alert if any
  if (fail.length) {
    let msg = 'Tracking failed for ' + fail.length + ' frame(s):\n\n';
    for (const f of fail) {
      msg += '  • ' + f.image + ': ' + f.error + '\n';
    }
    msg += '\nClick on each failed frame and manually click the 2 points there.';
    msg += '\n(See browser console for full report)';
    setTimeout(() => alert(msg), 100);
  }
}

async function downloadCalib() {
  const r = await fetch('/api/calibrations');
  const data = await r.json();
  const blob = new Blob([JSON.stringify(data, null, 2)], {type: 'application/json'});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = 'scale_calibration.json';
  a.click();
}

async function exportSummary() {
  const r = await fetch('/api/calibrations');
  const data = await r.json();
  let s = 'Scale Calibration Summary\n' + '='.repeat(60) + '\n';
  for (const [img, c] of Object.entries(data)) {
    s += img + ':\n';
    s += '  ' + c.real_distance_cm + ' cm = ' + c.pixel_distance.toFixed(1) + ' px\n';
    s += '  scale = ' + c.scale_cm_per_pixel.toFixed(5) + ' cm/px\n';
    s += '  object: ' + (c.object_description || 'unknown') + '\n\n';
  }
  alert(s);
}

loadFiles();
loadSourceInfo();
</script>
</body>
</html>
"""


def make_app(image_dir, output_path, mode="scale"):
    """Create the picker app.

    Args:
        image_dir: Directory of input images.
        output_path: Where to save the calibration JSON.
        mode: "scale" (default, 2 points + distance) or "seed" (1 point on belly).
    """
    app = Flask(__name__)
    app.config["IMAGE_DIR"] = os.path.abspath(image_dir)
    app.config["OUTPUT_PATH"] = os.path.abspath(output_path)
    app.config["MODE"] = mode

    # Detect a video manifest (set if frames were extracted via extract_frames.py).
    # If present, we can do precise frame-by-frame LK tracking on the original
    # video and only sample at the saved frame indices.
    manifest_path = os.path.join(app.config["IMAGE_DIR"], "_video_frame_manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
        app.config["VIDEO_PATH"] = manifest.get("video_path")
        app.config["VIDEO_FRAME_MAP"] = {
            entry["filename"]: int(entry["video_frame_index"])
            for entry in manifest.get("saved", [])
        }
        app.config["VIDEO_FPS"] = manifest.get("fps")
        print(f"Video manifest detected: {len(app.config['VIDEO_FRAME_MAP'])} frames "
              f"from {app.config['VIDEO_PATH']}")
    else:
        app.config["VIDEO_PATH"] = None
        app.config["VIDEO_FRAME_MAP"] = {}

    # Load existing calibration if present
    if os.path.exists(app.config["OUTPUT_PATH"]):
        with open(app.config["OUTPUT_PATH"]) as f:
            app.config["CALIBRATIONS"] = json.load(f)
    else:
        app.config["CALIBRATIONS"] = {}

    @app.route("/")
    def index():
        return render_template_string(HTML_TEMPLATE)

    @app.route("/api/files")
    def list_files():
        exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
        files = []
        for ext in exts:
            files.extend(glob.glob(os.path.join(app.config["IMAGE_DIR"], ext)))
            files.extend(glob.glob(os.path.join(app.config["IMAGE_DIR"], ext.upper())))
        files = sorted({os.path.basename(f) for f in files})
        return jsonify(files)

    @app.route("/api/image/<path:name>")
    def serve_image(name):
        # Prevent path traversal
        safe_name = os.path.basename(name)
        path = os.path.join(app.config["IMAGE_DIR"], safe_name)
        if not os.path.exists(path):
            return "Not found", 404
        return send_file(path)

    @app.route("/api/calibrations")
    def get_calibrations():
        return jsonify(app.config["CALIBRATIONS"])

    @app.route("/api/source_info")
    def source_info():
        return jsonify({
            "image_dir": app.config["IMAGE_DIR"],
            "is_video": app.config["VIDEO_PATH"] is not None,
            "video_path": app.config["VIDEO_PATH"],
            "n_frames": len(app.config["VIDEO_FRAME_MAP"]),
            "fps": app.config.get("VIDEO_FPS"),
            "mode": app.config.get("MODE", "scale"),
        })

    @app.route("/api/save", methods=["POST"])
    def save_calibration():
        body = request.get_json()
        image = body.get("image")
        data = body.get("data")
        if not image or not data:
            return jsonify({"error": "missing image or data"}), 400
        app.config["CALIBRATIONS"][image] = data

        # Persist to disk
        os.makedirs(os.path.dirname(app.config["OUTPUT_PATH"]) or ".", exist_ok=True)
        with open(app.config["OUTPUT_PATH"], "w") as f:
            json.dump(app.config["CALIBRATIONS"], f, indent=2)
        return jsonify({"ok": True})

    @app.route("/api/track_seed", methods=["POST"])
    def track_seed():
        """Track a single seed point across all frames.

        For video sources, uses frame-by-frame LK on the original video.
        For image directories, uses ORB+homography matching.
        """
        if not HAS_CV2:
            return jsonify({"error": "OpenCV not available"}), 500
        body = request.get_json()
        src_image = body.get("src_image")
        src_p = body.get("src_p")  # [x, y]
        object_description = body.get("object_description", "belly")

        if not src_image or src_p is None:
            return jsonify({"error": "missing src_image or src_p"}), 400

        # Use the same machinery as 2-point tracking, but with the same point
        # for both p1 and p2, then keep just the tracked p1.
        # This works because LK / ORB tracks each point independently anyway,
        # and we're only interested in the location.
        if app.config.get("VIDEO_PATH"):
            results, err = _track_video_lk(
                src_image, src_p, src_p, real_distance_cm=None,
                object_description=object_description,
            )
            if err:
                return jsonify({"error": err}), 500
            # Convert to seed format
            seed_results = []
            for r in results:
                if r.get("tracking_ok"):
                    seed_results.append({
                        "image": r["image"],
                        "point": r["p1"],
                        "tracking": r.get("tracking"),
                        "object_description": object_description,
                        "video_frame_index": r.get("video_frame_index"),
                    })
                    app.config["CALIBRATIONS"][r["image"]] = {
                        "p1": r["p1"],
                        "object_description": object_description,
                        "tracking": r.get("tracking"),
                        "video_frame_index": r.get("video_frame_index"),
                    }
                else:
                    seed_results.append({
                        "image": r["image"], "tracking_ok": False,
                        "error": r.get("error"),
                    })
            os.makedirs(os.path.dirname(app.config["OUTPUT_PATH"]) or ".",
                        exist_ok=True)
            with open(app.config["OUTPUT_PATH"], "w") as f:
                json.dump(app.config["CALIBRATIONS"], f, indent=2)
            return jsonify({
                "results": seed_results,
                "n_tracked": sum(1 for r in seed_results if "point" in r),
                "method": "video_lk",
            })
        else:
            return jsonify({
                "error": "Seed-tracking for image directories not implemented; "
                         "click each image manually",
            }), 400

    def _track_video_lk(src_image, src_p1, src_p2, real_distance_cm,
                         object_description):
        """Track points across the entire video using consecutive-frame LK,
        then sample at the extracted-frame indices.

        Why this works (and per-saved-frame ORB doesn't):
            - LK is accurate when frame-to-frame motion is small (< ~50 px),
              which is true at full video frame rate (30fps → ~3-15 px/frame).
            - We process EVERY video frame, so each LK step has tiny motion.
            - We only save the result at the N frames the user already extracted.
            - This is robust across arbitrary camera motion in the video.
        """
        video_path = app.config.get("VIDEO_PATH")
        frame_map = app.config.get("VIDEO_FRAME_MAP", {})
        if not video_path or not os.path.exists(video_path):
            return None, "video not available"
        if src_image not in frame_map:
            return None, f"{src_image} not in saved video frames"

        src_video_idx = frame_map[src_image]
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Read source frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, src_video_idx)
        ret, src_frame = cap.read()
        if not ret:
            cap.release()
            return None, "could not read source frame from video"
        src_gray = cv2.cvtColor(src_frame, cv2.COLOR_BGR2GRAY)

        lk_params = dict(
            winSize=(31, 31), maxLevel=4,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.001),
        )

        # Forward + backward tracking from src_video_idx
        # tracked[idx] = (p1, p2) at video frame idx
        tracked = {src_video_idx: (
            np.array(src_p1, dtype=np.float32),
            np.array(src_p2, dtype=np.float32),
        )}

        # Forward
        prev_gray = src_gray
        prev_pts = np.array([src_p1, src_p2], dtype=np.float32).reshape(-1, 1, 2)
        cap.set(cv2.CAP_PROP_POS_FRAMES, src_video_idx + 1)
        for fidx in range(src_video_idx + 1, total):
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            next_pts, status, err = cv2.calcOpticalFlowPyrLK(
                prev_gray, gray, prev_pts, None, **lk_params,
            )
            if (next_pts is None or status[0][0] == 0 or status[1][0] == 0
                    or err is None or err[0][0] > 50 or err[1][0] > 50):
                # Tracking lost — stop forward chain
                break
            tracked[fidx] = (
                next_pts[0][0].copy().astype(np.float32),
                next_pts[1][0].copy().astype(np.float32),
            )
            prev_gray = gray
            prev_pts = next_pts

        # Backward
        prev_gray = src_gray
        prev_pts = np.array([src_p1, src_p2], dtype=np.float32).reshape(-1, 1, 2)
        for fidx in range(src_video_idx - 1, -1, -1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            next_pts, status, err = cv2.calcOpticalFlowPyrLK(
                prev_gray, gray, prev_pts, None, **lk_params,
            )
            if (next_pts is None or status[0][0] == 0 or status[1][0] == 0
                    or err is None or err[0][0] > 50 or err[1][0] > 50):
                break
            tracked[fidx] = (
                next_pts[0][0].copy().astype(np.float32),
                next_pts[1][0].copy().astype(np.float32),
            )
            prev_gray = gray
            prev_pts = next_pts

        cap.release()

        # Save per-frame results at the saved-frame indices
        results = []
        # Sort frame_map by video frame index for orderly output
        sorted_frames = sorted(frame_map.items(), key=lambda kv: kv[1])
        for fname, vidx in sorted_frames:
            if vidx not in tracked:
                results.append({
                    "image": fname, "tracking_ok": False,
                    "error": f"LK tracking lost before reaching video frame {vidx}",
                })
                continue
            p1, p2 = tracked[vidx]
            px = float(np.linalg.norm(p1 - p2))
            scale = ((real_distance_cm / px) if real_distance_cm and px > 0
                     else None)
            entry = {
                "p1": [int(round(float(p1[0]))), int(round(float(p1[1])))],
                "p2": [int(round(float(p2[0]))), int(round(float(p2[1])))],
                "real_distance_cm": real_distance_cm,
                "pixel_distance": px,
                "scale_cm_per_pixel": scale,
                "object_description": object_description,
                "tracking": "video_lk" if vidx != src_video_idx else "manual",
                "video_frame_index": int(vidx),
            }
            app.config["CALIBRATIONS"][fname] = entry
            results.append({"image": fname, **entry, "tracking_ok": True})

        return results, None

    @app.route("/api/track", methods=["POST"])
    def track_to_all():
        """Track 2 reference points from a source image to all destination images.

        Method auto-selected:
            - If the session has a video manifest (extracted from a video file),
              use frame-by-frame Lucas-Kanade across the original video, then
              sample at the saved-frame indices. Most accurate.
            - Otherwise, use ORB feature matching + RANSAC homography between
              individual still images. Robust to large viewpoint changes but
              requires the reference object to be visible in each frame.

        Why this approach (and not LK optical flow):
            - Multi-view photos have LARGE camera motion between frames; LK only
              handles small motion (< ~50 px).
            - Feature descriptors are invariant to viewpoint, scale, rotation.
            - RANSAC homography is robust to outliers from clutter / moving people.
            - Always use frame 0 as the reference (no chain accumulation of error).
            - Bidirectional consistency check rejects bad tracks automatically.
        """
        if not HAS_CV2:
            return jsonify({"error": "OpenCV not available"}), 500

        body = request.get_json()
        src_image = body.get("src_image")
        src_p1 = body.get("src_p1")
        src_p2 = body.get("src_p2")
        dst_images = body.get("dst_images", [])
        real_distance_cm = body.get("real_distance_cm")
        object_description = body.get("object_description", "")

        if not src_image or src_p1 is None or src_p2 is None:
            return jsonify({"error": "missing src_image or points"}), 400

        # ── Method dispatch: video LK vs ORB homography ───────────────
        if app.config.get("VIDEO_PATH"):
            print(f"[track] Using video-LK tracking from {src_image}")
            results, err = _track_video_lk(
                src_image, src_p1, src_p2, real_distance_cm, object_description,
            )
            if err:
                return jsonify({"error": f"video tracking failed: {err}"}), 500
            os.makedirs(os.path.dirname(app.config["OUTPUT_PATH"]) or ".", exist_ok=True)
            with open(app.config["OUTPUT_PATH"], "w") as f:
                json.dump(app.config["CALIBRATIONS"], f, indent=2)
            return jsonify({
                "results": results,
                "n_tracked": sum(1 for r in results if r.get("tracking_ok")),
                "method": "video_lk",
            })

        # ── ORB+homography fallback for image directories ─────────────
        print(f"[track] Using ORB+homography tracking from {src_image}")
        src_path = os.path.join(app.config["IMAGE_DIR"], os.path.basename(src_image))
        if not os.path.exists(src_path):
            return jsonify({"error": f"source not found: {src_image}"}), 404

        src_img = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
        if src_img is None:
            return jsonify({"error": "could not read source image"}), 500

        H_src, W_src = src_img.shape[:2]

        # Save source image's calibration first
        results = []
        d12 = np.linalg.norm(np.array(src_p1) - np.array(src_p2))
        src_data = {
            "p1": [int(src_p1[0]), int(src_p1[1])],
            "p2": [int(src_p2[0]), int(src_p2[1])],
            "real_distance_cm": real_distance_cm,
            "pixel_distance": float(d12),
            "scale_cm_per_pixel": (real_distance_cm / d12) if real_distance_cm and d12 > 0 else None,
            "object_description": object_description,
            "tracking": "manual",
        }
        app.config["CALIBRATIONS"][src_image] = src_data
        results.append({"image": src_image, **src_data, "tracking_ok": True})

        # Define a region of interest around the clicked points in the source.
        # We'll only use features from this region — they're guaranteed to lie
        # on/near the reference object the user clicked.
        # Margin: ~3x the inter-point distance, or at least 250px.
        margin = max(250, int(d12 * 1.5))
        pts_arr = np.array([src_p1, src_p2])
        x_min = max(0, int(pts_arr[:, 0].min() - margin))
        y_min = max(0, int(pts_arr[:, 1].min() - margin))
        x_max = min(W_src, int(pts_arr[:, 0].max() + margin))
        y_max = min(H_src, int(pts_arr[:, 1].max() + margin))
        src_roi = src_img[y_min:y_max, x_min:x_max]

        # Use ORB for fast, robust features. nfeatures=2000 gives dense coverage.
        orb = cv2.ORB_create(nfeatures=2000, scaleFactor=1.2, nlevels=8,
                             edgeThreshold=15, fastThreshold=15)
        kp_src, des_src = orb.detectAndCompute(src_roi, None)

        if des_src is None or len(kp_src) < 20:
            return jsonify({
                "error": f"too few features near clicked points "
                         f"({0 if des_src is None else len(kp_src)} found)",
                "results": results,
                "n_tracked": 1,
            }), 200

        # Convert keypoint coordinates back to global (full-image) coords
        kp_src_global = np.array([(kp.pt[0] + x_min, kp.pt[1] + y_min)
                                    for kp in kp_src], dtype=np.float32)

        bf = cv2.BFMatcher(cv2.NORM_HAMMING)

        for dst in dst_images:
            if dst == src_image:
                continue
            dst_path = os.path.join(app.config["IMAGE_DIR"], os.path.basename(dst))
            if not os.path.exists(dst_path):
                results.append({"image": dst, "tracking_ok": False, "error": "not found"})
                continue
            dst_img = cv2.imread(dst_path, cv2.IMREAD_GRAYSCALE)
            if dst_img is None:
                results.append({"image": dst, "tracking_ok": False, "error": "could not read"})
                continue

            # Detect features in the WHOLE destination image
            kp_dst, des_dst = orb.detectAndCompute(dst_img, None)
            if des_dst is None or len(kp_dst) < 20:
                results.append({"image": dst, "tracking_ok": False,
                                "error": f"too few features in dst ({0 if des_dst is None else len(kp_dst)})"})
                continue

            # KNN match with Lowe's ratio test (strict 0.7 ratio)
            try:
                knn = bf.knnMatch(des_src, des_dst, k=2)
            except cv2.error as e:
                results.append({"image": dst, "tracking_ok": False, "error": str(e)})
                continue

            good = []
            for pair in knn:
                if len(pair) < 2:
                    continue
                m, n = pair
                if m.distance < 0.7 * n.distance:
                    good.append(m)

            MIN_GOOD = 25
            if len(good) < MIN_GOOD:
                results.append({"image": dst, "tracking_ok": False,
                                "error": f"only {len(good)} good matches "
                                         f"(need ≥{MIN_GOOD}) — reference object "
                                         f"likely not visible in this frame"})
                continue

            src_pts = np.array([kp_src_global[m.queryIdx] for m in good],
                                dtype=np.float32).reshape(-1, 1, 2)
            dst_pts = np.array([kp_dst[m.trainIdx].pt for m in good],
                                dtype=np.float32).reshape(-1, 1, 2)

            # RANSAC homography (3 px reprojection threshold)
            H_mat, inlier_mask = cv2.findHomography(src_pts, dst_pts,
                                                      cv2.RANSAC, 3.0)
            if H_mat is None or inlier_mask is None:
                results.append({"image": dst, "tracking_ok": False,
                                "error": "homography failed"})
                continue

            inlier_mask = inlier_mask.ravel().astype(bool)
            n_inliers = int(inlier_mask.sum())
            inlier_ratio = n_inliers / float(len(good))

            MIN_INLIERS = 25
            MIN_INLIER_RATIO = 0.30
            if n_inliers < MIN_INLIERS or inlier_ratio < MIN_INLIER_RATIO:
                results.append({"image": dst, "tracking_ok": False,
                                "error": f"unstable homography "
                                         f"({n_inliers}/{len(good)} inliers "
                                         f"= {inlier_ratio:.0%}, "
                                         f"need ≥{MIN_INLIERS} and ≥{MIN_INLIER_RATIO:.0%})"})
                continue

            # Compute reprojection error of the INLIER features
            # (this tells us how good the homography actually is — back-projecting
            # the 2 query points is meaningless because any 4 points can be exactly
            # fit by a homography).
            src_in = src_pts[inlier_mask]
            dst_in = dst_pts[inlier_mask]
            projected = cv2.perspectiveTransform(src_in, H_mat)
            reproj_errs = np.linalg.norm(dst_in - projected, axis=2).flatten()
            median_err = float(np.median(reproj_errs))
            max_err = float(np.max(reproj_errs))

            MAX_MEDIAN_ERR = 2.0
            if median_err > MAX_MEDIAN_ERR:
                results.append({"image": dst, "tracking_ok": False,
                                "error": f"high reprojection error "
                                         f"(median {median_err:.1f}px > {MAX_MEDIAN_ERR}px)"})
                continue

            # Apply homography to the original clicked points
            src_pt_arr = np.array([[src_p1, src_p2]], dtype=np.float32)
            dst_pt_arr = cv2.perspectiveTransform(src_pt_arr, H_mat)[0]
            p1_new = dst_pt_arr[0]
            p2_new = dst_pt_arr[1]

            # Sanity: tracked points should be within image bounds
            H_dst, W_dst = dst_img.shape[:2]
            in_bounds = (
                0 <= p1_new[0] <= W_dst and 0 <= p1_new[1] <= H_dst and
                0 <= p2_new[0] <= W_dst and 0 <= p2_new[1] <= H_dst
            )
            if not in_bounds:
                results.append({"image": dst, "tracking_ok": False,
                                "error": f"projected points outside image "
                                         f"({p1_new.tolist()}, {p2_new.tolist()} "
                                         f"vs {W_dst}x{H_dst})"})
                continue

            # Sanity: tracked points should be near the inlier features.
            # If far away, the homography is being extrapolated outside its
            # support region (unreliable).
            inlier_dst_pts = dst_in.reshape(-1, 2)
            dists_to_inliers = np.linalg.norm(
                inlier_dst_pts - p1_new[None, :], axis=1
            )
            min_dist_to_inlier_p1 = float(dists_to_inliers.min())
            dists_to_inliers = np.linalg.norm(
                inlier_dst_pts - p2_new[None, :], axis=1
            )
            min_dist_to_inlier_p2 = float(dists_to_inliers.min())
            MAX_EXTRAPOLATION = 250  # px
            if (min_dist_to_inlier_p1 > MAX_EXTRAPOLATION or
                min_dist_to_inlier_p2 > MAX_EXTRAPOLATION):
                results.append({"image": dst, "tracking_ok": False,
                                "error": f"projected points too far from inlier "
                                         f"features ({min_dist_to_inlier_p1:.0f}, "
                                         f"{min_dist_to_inlier_p2:.0f}px > "
                                         f"{MAX_EXTRAPOLATION}px) — extrapolation"})
                continue

            # Sanity: scale shouldn't differ from source by more than 4x in either
            # direction. If it does, the homography is probably degenerate.
            pixel_dist = float(np.linalg.norm(p1_new - p2_new))
            if pixel_dist < 1.0 or pixel_dist > 4.0 * d12 or pixel_dist < d12 / 4.0:
                results.append({"image": dst, "tracking_ok": False,
                                "error": f"degenerate distance "
                                         f"({pixel_dist:.1f}px vs source {d12:.1f}px)"})
                continue

            scale = ((real_distance_cm / pixel_dist)
                     if real_distance_cm and pixel_dist > 0 else None)

            entry = {
                "p1": [int(round(float(p1_new[0]))), int(round(float(p1_new[1])))],
                "p2": [int(round(float(p2_new[0]))), int(round(float(p2_new[1])))],
                "real_distance_cm": real_distance_cm,
                "pixel_distance": pixel_dist,
                "scale_cm_per_pixel": scale,
                "object_description": object_description,
                "tracking": "orb_homography",
                "n_inliers": n_inliers,
                "n_good_matches": len(good),
                "inlier_ratio": round(inlier_ratio, 3),
                "median_reprojection_error_px": round(median_err, 2),
                "max_reprojection_error_px": round(max_err, 2),
            }
            app.config["CALIBRATIONS"][dst] = entry
            results.append({"image": dst, **entry, "tracking_ok": True})

        # Persist
        os.makedirs(os.path.dirname(app.config["OUTPUT_PATH"]) or ".", exist_ok=True)
        with open(app.config["OUTPUT_PATH"], "w") as f:
            json.dump(app.config["CALIBRATIONS"], f, indent=2)

        return jsonify({
            "results": results,
            "n_tracked": sum(1 for r in results if r.get("tracking_ok")),
        })

    return app


def main():
    parser = argparse.ArgumentParser(description="Interactive scale/seed picker")
    parser.add_argument("--image_dir", help="Directory of input images")
    parser.add_argument("--video", help="Video file (frames will be extracted)")
    parser.add_argument("--n_frames", type=int, default=30,
                        help="Frames to extract from video (default: 30 — MUST "
                             "match belly_orchestrator's --n_frames so saved "
                             "clicks align with the orchestrator's frame names)")
    parser.add_argument("--output", required=True,
                        help="Output JSON path")
    parser.add_argument("--mode", default="scale", choices=["scale", "seed"],
                        help="'scale' (2 points + distance for cm/pixel) "
                             "or 'seed' (1 point on belly for SAM3)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    args = parser.parse_args()

    if args.video:
        # Extract frames from video into a sibling directory
        from calibration.extract_frames import extract_frames, is_video_file
        if not is_video_file(args.video):
            sys.exit(f"Not a video file (or unsupported extension): {args.video}")
        frames_dir = os.path.splitext(args.video)[0] + "_frames"
        print(f"Extracting frames from video → {frames_dir}")
        extract_frames(args.video, frames_dir, n_frames=args.n_frames)
        image_dir = frames_dir
    elif args.image_dir:
        image_dir = args.image_dir
    else:
        sys.exit("Must specify --image_dir or --video")

    if not os.path.isdir(image_dir):
        sys.exit(f"Image directory not found: {image_dir}")

    app = make_app(image_dir, args.output, mode=args.mode)
    print(f"Picker mode: {args.mode}")
    print(f"Serving images from: {image_dir}")
    print(f"Output JSON will be saved to: {args.output}")
    print(f"Open in browser: http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
