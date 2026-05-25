"""
Worker script: Run SAM3 segmentation on input images.
Executed inside the 'vv_sam3' conda environment.

Robustness features:
  1. Multi-prompt cascade: if the primary prompt fails, fall back to
     simpler prompts (e.g. "belly" → "abdomen" → "stomach" → "person").
  2. Adaptive confidence retry: each prompt is retried at progressively
     lower thresholds (0.4 → 0.25 → 0.15 → 0.08) until a good mask appears.
  3. Smart detection selection: when SAM3 returns multiple candidates we
     score each by `score * sqrt(area_fraction) * centrality` so the
     patient (large, central, high-confidence) wins over a bystander.
  4. Mask post-processing: keep largest connected component, fill holes,
     morphological close — eliminates fragmented or noisy masks.
  5. Quality gates: reject masks <2% or >90% of image, or with extreme
     aspect ratios (likely background or detector failure).
  6. Side-by-side debug overlays: every mask gets a JPG overlay for
     instant visual verification.

Usage:
    conda activate vv_sam3
    python src/pipeline/run_sam3.py \\
        --image_dir data/input/patient_001 \\
        --output_dir data/output/patient_001/segmentation \\
        [--prompt "person"] \\
        [--fallback_prompts "belly,abdomen,stomach"]

Outputs:
    - masks/             : Per-image binary masks (PNG, 0/255)
    - overlays/          : Per-image JPG overlays for quick visual check
    - segmentation.json  : Full metadata including quality scores
"""

import os
import sys
import json
import glob
import argparse
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
SAM3_DIR = os.path.join(PROJECT_DIR, "repos", "sam3")
sys.path.insert(0, SAM3_DIR)

import torch
from PIL import Image, ImageDraw

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


# ─── Mask post-processing ──────────────────────────────────────────────

def keep_largest_connected_component(mask):
    """Keep only the largest 4-connected blob in a binary mask.

    Mask fragmentation (multiple disconnected regions) is a common
    SAM3 failure mode, especially when the patient is partly occluded
    by their own arms or the camera frame. Keeping just the largest
    blob removes these spurious fragments.
    """
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    if HAS_CV2:
        n, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8,
        )
        if n <= 1:
            return mask
        # stats[:, cv2.CC_STAT_AREA] excluding background (idx 0)
        areas = stats[1:, cv2.CC_STAT_AREA]
        if len(areas) == 0:
            return mask
        biggest = int(np.argmax(areas)) + 1
        return ((labels == biggest).astype(np.uint8) * 255)
    # Fallback without OpenCV (slow but works)
    from scipy import ndimage
    lbl, n = ndimage.label(mask > 0)
    if n <= 1:
        return mask
    sizes = ndimage.sum(mask > 0, lbl, range(1, n + 1))
    keep = int(np.argmax(sizes)) + 1
    return ((lbl == keep).astype(np.uint8) * 255)


def fill_holes_and_smooth(mask, close_kernel=7):
    """Fill small holes and smooth jagged boundaries via morphology."""
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    if not HAS_CV2:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                         (close_kernel, close_kernel))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    # Fill internal holes by flood-filling from a known background pixel
    h, w = closed.shape
    flood = closed.copy()
    pad = np.zeros((h + 2, w + 2), dtype=np.uint8)
    pad[1:-1, 1:-1] = closed
    cv2.floodFill(pad, None, (0, 0), 255)
    bg = pad[1:-1, 1:-1] == 0  # pixels NOT reachable from outside = holes
    closed[bg] = 255
    return closed


def clean_mask(mask, fill_close=7):
    """Apply the full cleanup chain: largest CC → close → hole-fill."""
    m = (mask > 0).astype(np.uint8) * 255
    m = keep_largest_connected_component(m)
    m = fill_holes_and_smooth(m, close_kernel=fill_close)
    return m


def evaluate_mask_quality(mask, image_shape, box=None):
    """Score a mask for plausibility as the patient.

    Returns dict with:
        area_fraction: mask area / image area
        centrality: 1 - distance(mask centroid, image center) / max_distance
        compactness: 4πA / P²  (1.0 = circle, lower = more elongated)
        aspect_ratio_ok: bool, True if bbox aspect is reasonable
        passes: bool, True if mask is plausible enough to use
        fail_reason: str if not passing
    """
    h, w = image_shape[:2]
    img_area = h * w
    m_bool = mask > 0
    m_area = int(m_bool.sum())
    if m_area == 0:
        return {"area_fraction": 0, "passes": False, "fail_reason": "empty"}

    area_frac = m_area / float(img_area)

    # Centroid
    ys, xs = np.where(m_bool)
    cy_m = ys.mean(); cx_m = xs.mean()
    cx_i, cy_i = w / 2.0, h / 2.0
    dist = np.sqrt((cx_m - cx_i) ** 2 + (cy_m - cy_i) ** 2)
    max_dist = np.sqrt((w / 2.0) ** 2 + (h / 2.0) ** 2)
    centrality = 1.0 - (dist / max_dist)

    # Bounding box of mask
    bbox_w = xs.max() - xs.min() + 1
    bbox_h = ys.max() - ys.min() + 1
    aspect = bbox_w / max(bbox_h, 1)
    aspect_ok = 0.2 <= aspect <= 5.0

    # Compactness (Polsby-Popper)
    if HAS_CV2:
        contours, _ = cv2.findContours(
            (m_bool.astype(np.uint8) * 255), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        perimeter = sum(cv2.arcLength(c, True) for c in contours) or 1.0
    else:
        perimeter = 4 * (bbox_w + bbox_h)  # rough
    compactness = 4 * np.pi * m_area / (perimeter ** 2)

    # Quality gates
    # min area 0.005 (0.5%) — admits small/distant balloons, rejects single-
    #   pixel false positives. Was 0.02 (2%), which incorrectly rejected
    #   valid balloon masks at moderate distance.
    # max area 0.95 — admits a tightly-cropped scene where the subject fills
    #   most of the frame; rejects whole-image masks. Was 0.90.
    passes = True
    fail_reason = None
    if area_frac < 0.005:
        passes, fail_reason = False, f"area too small ({area_frac*100:.2f}%)"
    elif area_frac > 0.95:
        passes, fail_reason = False, f"area too large ({area_frac*100:.1f}%)"
    elif not aspect_ok:
        passes, fail_reason = False, f"extreme aspect ratio ({aspect:.2f})"

    return {
        "area_fraction": float(area_frac),
        "centrality": float(centrality),
        "compactness": float(compactness),
        "bbox_aspect_ratio": float(aspect),
        "centroid_xy": [float(cx_m), float(cy_m)],
        "passes": bool(passes),
        "fail_reason": fail_reason,
    }


# ─── Detection scoring (for picking the best of multiple candidates) ───

def score_detection(detection_score, mask_quality):
    """Combine model confidence + mask quality into a scalar.

    Heuristic:
        score = sqrt(detection_score) * sqrt(area_frac) * (0.5 + 0.5*centrality)
    Square roots flatten the influence of any single factor; centrality is
    blended so a near-edge person isn't penalized too heavily.
    """
    if not mask_quality.get("passes", False):
        return -1.0  # reject
    s = max(0.0, float(detection_score))
    a = max(0.0, mask_quality.get("area_fraction", 0.0))
    c = max(0.0, mask_quality.get("centrality", 0.0))
    return (s ** 0.5) * (a ** 0.5) * (0.5 + 0.5 * c)


# ─── Adaptive segmentation ─────────────────────────────────────────────

def adaptive_segment(processor, image, prompts, confidence_levels,
                     seed_point=None, seed_box_size=0.10):
    """Try each (prompt, confidence) combination until at least one
    candidate passes the quality gate.

    Iteration order: each prompt is tried at every confidence (descending),
    so a high-confidence answer with a primary prompt is preferred over
    a low-confidence answer with the same prompt. Once any prompt yields
    detections, we DO NOT continue to the next prompt — we stay with the
    best detection from the prompt that worked.

    Args:
        seed_point: Optional [x, y] in image pixel coords. When provided,
            adds a positive box prompt centered on this point so SAM3
            focuses on the object at that location (e.g., the belly).
            Combined with text prompt for best results.
        seed_box_size: Box width/height as fraction of min(W,H). 0.10 = 10%.
    """
    H, W = np.array(image).shape[:2]

    # Convert seed point to normalized [cx, cy, w, h] for add_geometric_prompt
    seed_box_norm = None
    if seed_point is not None:
        sx, sy = float(seed_point[0]), float(seed_point[1])
        side = seed_box_size * min(W, H)
        seed_box_norm = [sx / W, sy / H, side / W, side / H]

    # Run the (expensive) image backbone ONCE per image. The backbone output
    # doesn't depend on prompt or confidence, so caching it eliminates 8-32×
    # redundant ViT passes (one per (prompt, confidence) combination).
    with torch.cuda.amp.autocast(dtype=torch.float32):
        cached_state = processor.set_image(image)

    for prompt in prompts:
        candidates = []
        # Apply text prompt + (optional) seed box ONCE per prompt
        with torch.cuda.amp.autocast(dtype=torch.float32):
            state = {k: v for k, v in cached_state.items()}  # shallow clone
            state = processor.set_text_prompt(prompt, state)
            if seed_box_norm is not None:
                state = processor.add_geometric_prompt(
                    seed_box_norm, True, state,
                )

        # Now sweep confidence levels — only re-runs the cheap grounding head
        for conf in sorted(confidence_levels, reverse=True):
            processor.set_confidence_threshold(conf, state)
            masks = state.get("masks")
            scores = state.get("scores")
            boxes = state.get("boxes")
            if masks is None or len(masks) == 0:
                continue
            masks_np = masks.squeeze(1).cpu().numpy()
            scores_np = scores.cpu().numpy()
            boxes_np = boxes.cpu().numpy()
            # Free GPU tensors held in state after extracting numpy
            for k in ("masks", "masks_logits", "boxes", "scores"):
                if k in state:
                    del state[k]
            for i in range(len(masks_np)):
                candidates.append({
                    "raw_mask": masks_np[i],
                    "score": float(scores_np[i]),
                    "box": boxes_np[i].tolist(),
                    "confidence_level": float(conf),
                })
            # If we got at least one quality-passing detection, stop trying
            # lower confidence levels for this prompt.
            any_pass = False
            for c in candidates:
                cleaned = clean_mask(c["raw_mask"])
                qual = evaluate_mask_quality(cleaned, (H, W))
                if qual["passes"]:
                    any_pass = True
                    break
            if any_pass:
                break

        if not candidates:
            continue

        # Score each candidate with quality + centrality + size
        scored = []
        for c in candidates:
            cleaned = clean_mask(c["raw_mask"])
            qual = evaluate_mask_quality(cleaned, (H, W))
            sc = score_detection(c["score"], qual)
            scored.append({
                **c,
                "cleaned_mask": cleaned,
                "quality": qual,
                "combined_score": sc,
            })

        # Pick the best non-rejected candidate
        good = [s for s in scored if s["combined_score"] > 0]
        if not good:
            continue  # all rejected → try next prompt

        good.sort(key=lambda x: x["combined_score"], reverse=True)
        return prompt, good

    return None, []


# ─── Visualization ────────────────────────────────────────────────────

def save_overlay(image, mask, out_path, score=None, prompt=None,
                  quality=None, box=None):
    """Save an image with the mask outlined and labeled for quick inspection."""
    img = image.copy().convert("RGB")
    if not HAS_CV2:
        img.save(out_path)
        return
    arr = np.array(img)[:, :, ::-1].copy()  # to BGR
    m = (mask > 127).astype(np.uint8)

    # Magenta tint where the mask is positive
    overlay = arr.copy()
    overlay[m > 0] = (overlay[m > 0] * 0.55 + np.array([200, 0, 200]) * 0.45).astype(np.uint8)
    arr = overlay

    # Green contour for the mask boundary
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(arr, contours, -1, (0, 255, 0), 5)

    # Detection box (yellow)
    if box is not None and len(box) == 4:
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(arr, (x1, y1), (x2, y2), (0, 255, 255), 4)

    # Header text
    h = arr.shape[0]
    label = []
    if prompt: label.append(f"prompt='{prompt}'")
    if score is not None: label.append(f"score={score:.2f}")
    if quality:
        label.append(f"area={quality['area_fraction']*100:.1f}%")
        label.append(f"central={quality['centrality']*100:.0f}%")
    text = "  ".join(label)
    cv2.rectangle(arr, (0, 0), (arr.shape[1], 70), (0, 0, 0), -1)
    cv2.putText(arr, text, (12, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (255, 255, 255), 2)

    cv2.imwrite(out_path, arr)


# ─── Main pipeline ─────────────────────────────────────────────────────

def run_segmentation(
    image_dir,
    output_dir,
    text_prompt="person",
    confidence_threshold=0.25,
    checkpoint_path=None,
    fallback_prompts=None,
    confidence_levels=(0.40, 0.25, 0.15, 0.08),
    seed_points_path=None,
    seed_box_size=0.10,
):
    """Robust SAM3 segmentation across all images in image_dir.

    Args:
        image_dir: Directory containing input images.
        output_dir: Directory to save masks, overlays, metadata.
        text_prompt: Primary text prompt (default: "person").
        confidence_threshold: NOT used directly; see confidence_levels.
            Kept for backward compatibility.
        checkpoint_path: Path to SAM3 checkpoint (auto-detect if None).
        fallback_prompts: Prompts to try if the primary returns zero
            detections OR all detections fail quality gates.
        confidence_levels: Adaptive thresholds tried per prompt
            (descending order).
    """
    if fallback_prompts is None:
        fallback_prompts = []
    # NOTE: previously this code force-appended "person" as a final fallback.
    # That was a bug for non-person subjects (e.g. balloon): when the balloon
    # prompt occasionally failed, the cascade fell through to "person", which
    # segmented the whole patient holding the balloon instead of the balloon.
    # We now respect the caller's fallback list exactly — the orchestrator's
    # SUBJECT_PRESETS provide subject-appropriate fallbacks.

    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    os.makedirs(output_dir, exist_ok=True)
    masks_dir = os.path.join(output_dir, "masks")
    overlays_dir = os.path.join(output_dir, "overlays")
    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(overlays_dir, exist_ok=True)

    # Collect images
    image_extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(image_dir, ext)))
        image_files.extend(glob.glob(os.path.join(image_dir, ext.upper())))
    image_files = sorted(set(image_files))

    if not image_files:
        raise ValueError(f"No images found in {image_dir}")

    print(f"Found {len(image_files)} images in {image_dir}")
    print(f"Primary prompt: '{text_prompt}'")
    print(f"Fallback prompts: {fallback_prompts}")
    print(f"Confidence cascade: {list(confidence_levels)}")

    # Load model
    if checkpoint_path is None:
        checkpoint_path = os.path.join(SAM3_DIR, "checkpoints", "sam3.pt")

    if os.path.exists(checkpoint_path):
        print(f"Loading SAM3 from local checkpoint: {checkpoint_path}")
        model = build_sam3_image_model(
            checkpoint_path=checkpoint_path, load_from_HF=False,
            eval_mode=True, device="cuda",
        )
    else:
        print("Loading SAM3 from HuggingFace...")
        model = build_sam3_image_model(load_from_HF=True, eval_mode=True, device="cuda")

    model = model.float()
    # SAM3 requires its trained resolution (1008) — RoPE positional embeddings
    # are baked in. Memory savings come from caching the backbone and freeing
    # tensors aggressively (see adaptive_segment).
    processor = Sam3Processor(model, resolution=1008,
                                confidence_threshold=confidence_threshold)

    # Load seed points if provided
    seed_points = {}
    if seed_points_path and os.path.exists(seed_points_path):
        with open(seed_points_path) as f:
            seed_data = json.load(f)
        for fname, entry in seed_data.items():
            if isinstance(entry, dict) and "p1" in entry:
                seed_points[fname] = entry["p1"]
        print(f"Loaded {len(seed_points)} seed points from {seed_points_path}")

    all_results = {}
    prompt_cascade = [text_prompt] + list(fallback_prompts)

    for img_idx, img_path in enumerate(image_files):
        img_name = os.path.basename(img_path)
        print(f"\nProcessing: {img_name}")

        # Clear GPU cache between images so peak memory is per-image, not cumulative
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        image = Image.open(img_path).convert("RGB")
        width, height = image.size

        seed = seed_points.get(img_name)
        used_prompt, scored = adaptive_segment(
            processor, image, prompt_cascade, confidence_levels,
            seed_point=seed, seed_box_size=seed_box_size,
        )
        if seed is not None:
            print(f"  Using seed point ({seed[0]}, {seed[1]}) "
                  f"as geometric prompt")

        if not scored:
            print(f"  ✗ No quality-passing detection from {prompt_cascade}")
            all_results[img_name] = {
                "image_path": img_path,
                "width": width,
                "height": height,
                "num_detections": 0,
                "detections": [],
                "combined_mask_path": None,
                "combined_mask_fraction": 0.0,
                "prompts_tried": prompt_cascade,
                "status": "no_detection",
            }
            continue

        best = scored[0]
        cleaned = best["cleaned_mask"]
        qual = best["quality"]
        print(f"  ✓ prompt='{used_prompt}'  conf_level={best['confidence_level']:.2f}  "
              f"score={best['score']:.2f}  area={qual['area_fraction']*100:.1f}%  "
              f"central={qual['centrality']*100:.0f}%  combined={best['combined_score']:.3f}")
        if len(scored) > 1:
            print(f"    ({len(scored)} candidates total; top: " +
                  ", ".join(f"{c['combined_score']:.3f}" for c in scored[:3]) + ")")

        # Save individual candidate masks
        detections = []
        for di, cand in enumerate(scored):
            cleaned_i = cand["cleaned_mask"]
            mfn = f"{os.path.splitext(img_name)[0]}_det{di}.png"
            mp = os.path.join(masks_dir, mfn)
            Image.fromarray(cleaned_i).save(mp)
            detections.append({
                "detection_index": di,
                "score": cand["score"],
                "confidence_level": cand["confidence_level"],
                "box_xyxy": cand["box"],
                "mask_path": mp,
                "mask_pixels": int((cleaned_i > 0).sum()),
                "mask_fraction": cand["quality"]["area_fraction"],
                "centrality": cand["quality"]["centrality"],
                "compactness": cand["quality"]["compactness"],
                "bbox_aspect_ratio": cand["quality"]["bbox_aspect_ratio"],
                "combined_score": cand["combined_score"],
                "selected": di == 0,
            })

        # Combined mask = the chosen one
        combined_filename = f"{os.path.splitext(img_name)[0]}_person_mask.png"
        combined_path = os.path.join(masks_dir, combined_filename)
        Image.fromarray(cleaned).save(combined_path)

        # Overlay for visual inspection
        overlay_path = os.path.join(overlays_dir,
                                      f"{os.path.splitext(img_name)[0]}_overlay.jpg")
        save_overlay(image, cleaned, overlay_path,
                      score=best["score"], prompt=used_prompt,
                      quality=qual, box=best["box"])

        all_results[img_name] = {
            "image_path": img_path,
            "width": width,
            "height": height,
            "num_detections": len(detections),
            "detections": detections,
            "combined_mask_path": combined_path,
            "combined_mask_fraction": qual["area_fraction"],
            "overlay_path": overlay_path,
            "prompt_used": used_prompt,
            "prompts_tried": prompt_cascade[: prompt_cascade.index(used_prompt) + 1],
            "confidence_level_used": best["confidence_level"],
            "quality": qual,
            "status": "ok",
        }

    # Save summary
    meta_path = os.path.join(output_dir, "segmentation.json")
    with open(meta_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved segmentation results: {meta_path}")
    print(f"Saved overlays for visual inspection: {overlays_dir}")

    n_ok = sum(1 for r in all_results.values() if r.get("status") == "ok")
    print(f"Successfully segmented: {n_ok}/{len(image_files)} images")

    return meta_path


def main():
    parser = argparse.ArgumentParser(description="Robust SAM3 segmentation")
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="person",
                        help="Primary text prompt (default: person)")
    parser.add_argument("--fallback_prompts", type=str, default=None,
                        help="Comma-separated fallback prompts (e.g. 'belly,abdomen')")
    parser.add_argument("--confidence", type=float, default=0.25,
                        help="Initial confidence threshold (legacy; cascade overrides)")
    parser.add_argument("--confidence_levels", type=str,
                        default="0.40,0.25,0.15,0.08",
                        help="Adaptive confidence cascade, descending")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--seed_points", type=str, default=None,
                        help="Path to belly_seed.json from scale_picker --mode seed; "
                             "uses each frame's clicked point as a positive box "
                             "prompt to focus SAM3 on the belly region.")
    parser.add_argument("--seed_box_size", type=float, default=0.10,
                        help="Seed prompt box side as fraction of image (default 0.10)")
    args = parser.parse_args()

    fallbacks = ([s.strip() for s in args.fallback_prompts.split(",") if s.strip()]
                 if args.fallback_prompts else [])
    levels = tuple(float(s.strip()) for s in args.confidence_levels.split(",")
                   if s.strip())

    run_segmentation(
        args.image_dir, args.output_dir,
        text_prompt=args.prompt,
        confidence_threshold=args.confidence,
        checkpoint_path=args.checkpoint,
        fallback_prompts=fallbacks,
        confidence_levels=levels,
        seed_points_path=args.seed_points,
        seed_box_size=args.seed_box_size,
    )


if __name__ == "__main__":
    main()
