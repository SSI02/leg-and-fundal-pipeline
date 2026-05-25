"""Interactive browser picker for selecting the most-anterior frame.

Shows all extracted frames in a grid, ranked by an auto-computed
hip-X-separation heuristic so the most-anterior candidates are at the top.
The user clicks one frame; the selection is written to a JSON file and
consumed by the leg pipeline via --anterior_frame.

JSON output schema:
    {
        "anterior_frame":     "frame_009.jpg",
        "anterior_frame_idx": 9,
        "n_frames_total":     30,
        "auto_ranking_used":  true,
        "hip_sep_ratio":      0.27,
        "selected_at":        "2026-05-12T18:40:11"
    }

Usage (typical):
    # 1. Run scale picker first (extracts frames into <name>_frames/)
    bash scripts/run_scale_picker.sh data/input/patient.mp4 \\
         data/input/patient_scale.json
    # 2. Pick the anterior frame
    python src/calibration/anterior_picker.py \\
         --image_dir data/input/patient_frames \\
         --output    data/input/patient_anterior.json
    # → opens http://127.0.0.1:8091 in your browser
"""
import argparse
import io
import json
import os
import sys
import datetime
import glob

from flask import Flask, send_file, request, jsonify
from PIL import Image


SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


def _natural_sort_frames(image_dir):
    paths = sorted(glob.glob(os.path.join(image_dir, "frame_*.jpg")))
    if not paths:
        paths = sorted(
            p for p in glob.glob(os.path.join(image_dir, "*.jpg"))
            if not os.path.basename(p).startswith(".")
        )
    return paths


def _auto_rank_by_hip_sep(image_dir):
    """If we have access to a pose results JSON for these frames, rank by
    hip-X-separation so the user sees the best-anterior candidates first.

    Returns a list of (frame_name, hip_sep_ratio_or_None).
    """
    paths = _natural_sort_frames(image_dir)
    names = [os.path.basename(p) for p in paths]

    # Best-effort: look for a sibling pose_results.json if the user has
    # run the pipeline once before. Otherwise we present frames in
    # natural order with no ranking.
    candidates = []
    parent = os.path.dirname(image_dir.rstrip(os.sep))
    # Heuristic: data/input/<patient>_frames/  → data/output/<patient>/pose/pose_results.json
    base = os.path.basename(image_dir.rstrip(os.sep))
    if base.endswith("_frames"):
        patient_name = base[:-len("_frames")]
        pose_json = os.path.join(parent, "..", "output", patient_name,
                                  "pose", "pose_results.json")
        if os.path.exists(pose_json):
            with open(pose_json) as f:
                pr = json.load(f)
            from measurements.leg_metrics import compute_frame_view_quality_2d
            for name in names:
                person_data = pr.get(name, {})
                persons = person_data.get("persons", [])
                if not persons:
                    candidates.append((name, None))
                    continue
                p = max(persons, key=lambda x: x.get("mean_score", 0))
                q = compute_frame_view_quality_2d(p.get("leg_keypoints", {}))
                hip_sep = q.get("hip_sep_ratio") if q else None
                candidates.append((name, hip_sep))
            return candidates, True

    return [(n, None) for n in names], False


def make_app(image_dir, output_path):
    app = Flask(__name__)
    app.config["IMAGE_DIR"] = image_dir
    app.config["OUTPUT_PATH"] = output_path

    ranking, used_auto = _auto_rank_by_hip_sep(image_dir)
    # Sort so highest hip_sep first (None values go last)
    ranking_sorted = sorted(
        enumerate(ranking),
        key=lambda x: (-(x[1][1] if x[1][1] is not None else -1),)
    )
    app.config["RANKING"] = ranking
    app.config["RANKING_SORTED"] = ranking_sorted
    app.config["USED_AUTO"] = used_auto

    @app.route("/")
    def index():
        return _render_index_html(ranking_sorted, used_auto)

    @app.route("/frame/<name>")
    def frame(name):
        path = os.path.join(image_dir, name)
        if not os.path.exists(path):
            return ("not found", 404)
        # Send a downscaled JPEG for grid display
        img = Image.open(path)
        # Keep aspect, max width 360
        if img.width > 360:
            ratio = 360 / img.width
            img = img.resize((360, int(img.height * ratio)), Image.LANCZOS)
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=82)
        buf.seek(0)
        return send_file(buf, mimetype="image/jpeg")

    @app.route("/select", methods=["POST"])
    def select():
        data = request.get_json(force=True) or {}
        name = data.get("frame")
        if not name:
            return jsonify({"error": "missing frame"}), 400
        idx = next((i for i, (n, _) in enumerate(ranking) if n == name), None)
        if idx is None:
            return jsonify({"error": "frame not in directory"}), 400
        hip_sep = next((s for n, s in ranking if n == name), None)
        out = {
            "anterior_frame": name,
            "anterior_frame_idx": idx,
            "n_frames_total": len(ranking),
            "auto_ranking_used": used_auto,
            "hip_sep_ratio": hip_sep,
            "selected_at": datetime.datetime.now().isoformat(timespec="seconds"),
        }
        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".",
                    exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"  → wrote selection to {output_path}")
        return jsonify({"ok": True, "saved_to": output_path, **out})

    return app


def _render_index_html(ranking_sorted, used_auto):
    cards = []
    for sort_pos, (orig_idx, (name, hip_sep)) in enumerate(ranking_sorted):
        if hip_sep is None:
            badge = "—"
            color = "#888"
        else:
            if hip_sep >= 0.22:
                color = "#2e8b3a"; badge = "anterior"
            elif hip_sep >= 0.16:
                color = "#daa520"; badge = "near-anterior"
            else:
                color = "#c8302e"; badge = "oblique"
            badge = f"{badge} · {hip_sep:.2f}"
        rank_pip = f"#{sort_pos + 1}" if used_auto else f"frame {orig_idx}"
        cards.append(f"""
        <div class="card" data-frame="{name}" data-idx="{orig_idx}"
             onclick="pick(this)">
            <div class="rank-pip">{rank_pip}</div>
            <img src="/frame/{name}" loading="lazy" alt="{name}">
            <div class="meta">
              <div class="name">{name}  ·  idx {orig_idx}</div>
              <div class="badge" style="background:{color}">{badge}</div>
            </div>
        </div>
        """)

    ranking_note = (
        "Frames ranked by hip X-separation (cached pose). "
        "Top-of-list candidates are the most front-facing."
        if used_auto else
        "No prior pose run found — frames are in extraction order. "
        "Pick the frame where the patient looks most directly at the camera."
    )

    return f"""<!doctype html>
<html><head><meta charset="utf-8">
<title>Anterior frame picker</title>
<style>
  body {{ font-family: -apple-system, system-ui, sans-serif; margin: 0;
          background: #f3f3f3; color: #222; }}
  header {{ background: #222; color: white; padding: 14px 20px;
           position: sticky; top: 0; z-index: 10; }}
  h1 {{ margin: 0; font-size: 17px; }}
  .sub {{ font-size: 13px; color: #ccc; margin-top: 4px; }}
  #status {{ position: fixed; bottom: 12px; left: 50%; transform: translateX(-50%);
             background: #2e8b3a; color: white; padding: 10px 18px;
             border-radius: 6px; font-size: 14px; display: none;
             box-shadow: 0 2px 8px rgba(0,0,0,0.2); }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
            gap: 14px; padding: 18px; }}
  .card {{ background: white; border: 2px solid transparent;
            border-radius: 8px; cursor: pointer; overflow: hidden;
            transition: transform 0.12s, border-color 0.12s,
                        box-shadow 0.12s;
            position: relative; }}
  .card:hover {{ transform: translateY(-2px);
                 box-shadow: 0 4px 12px rgba(0,0,0,0.12);
                 border-color: #5a8de8; }}
  .card.selected {{ border-color: #2e8b3a;
                     box-shadow: 0 0 0 3px rgba(46,139,58,0.35); }}
  .card img {{ width: 100%; display: block; }}
  .meta {{ padding: 8px 10px; }}
  .name {{ font-size: 12px; color: #555; }}
  .badge {{ display: inline-block; color: white; padding: 2px 8px;
             border-radius: 4px; font-size: 11px; font-weight: bold;
             margin-top: 4px; }}
  .rank-pip {{ position: absolute; top: 6px; right: 6px;
                background: rgba(0,0,0,0.7); color: white;
                font-size: 11px; padding: 2px 7px; border-radius: 10px; }}
</style></head>
<body>
<header>
  <h1>Pick the most anterior (front-facing) frame</h1>
  <div class="sub">{ranking_note}</div>
</header>
<div class="grid">{''.join(cards)}</div>
<div id="status"></div>
<script>
  function pick(el) {{
    document.querySelectorAll('.card.selected').forEach(c =>
        c.classList.remove('selected'));
    el.classList.add('selected');
    const frame = el.dataset.frame;
    fetch('/select', {{
      method: 'POST',
      headers: {{ 'Content-Type': 'application/json' }},
      body: JSON.stringify({{ frame: frame }})
    }})
      .then(r => r.json())
      .then(d => {{
        const s = document.getElementById('status');
        s.textContent = '✓ Selected ' + frame + ' (saved to ' + d.saved_to + '). '
                      + 'You can close this tab.';
        s.style.display = 'block';
      }});
  }}
</script>
</body></html>
"""


def main():
    parser = argparse.ArgumentParser(
        description="Interactive picker for the most-anterior frame"
    )
    parser.add_argument("--image_dir", help="Directory of extracted frames")
    parser.add_argument("--video",
                          help="Video file (frames will be extracted if no "
                               "<name>_frames/ already exists)")
    parser.add_argument("--n_frames", type=int, default=30,
                          help="Frames to extract from video (default 30, "
                               "must match the leg pipeline's --n_frames)")
    parser.add_argument("--output", required=True,
                          help="JSON output path (saved on click)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8091)
    args = parser.parse_args()

    if args.video:
        from calibration.extract_frames import extract_frames, is_video_file
        if not is_video_file(args.video):
            sys.exit(f"Not a video file: {args.video}")
        frames_dir = os.path.splitext(args.video)[0] + "_frames"
        if not os.path.isdir(frames_dir):
            print(f"Extracting frames from video → {frames_dir}")
            extract_frames(args.video, frames_dir, n_frames=args.n_frames)
        image_dir = frames_dir
    elif args.image_dir:
        image_dir = args.image_dir
    else:
        sys.exit("Must specify --image_dir or --video")

    if not os.path.isdir(image_dir):
        sys.exit(f"Image directory not found: {image_dir}")

    app = make_app(image_dir, args.output)
    print(f"Anterior-frame picker")
    print(f"  serving:   {image_dir}")
    print(f"  output:    {args.output}")
    print(f"  open in browser: http://{args.host}:{args.port}/")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
