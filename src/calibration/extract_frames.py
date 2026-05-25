"""
Extract evenly-spaced frames from a video file.

Used by the scale picker and the main pipelines to handle video input.
"""

import os
import argparse
import numpy as np
import cv2


def extract_frames(video_path, output_dir, n_frames=8, prefix="frame",
                   return_indices=False):
    """Extract n_frames evenly-spaced frames from a video.

    Args:
        video_path: Path to video file.
        output_dir: Directory to save frame images.
        n_frames: Number of frames to extract (evenly spaced across video).
        prefix: Filename prefix (default 'frame').
        return_indices: If True, returns (paths, video_frame_indices) so the
            caller knows which video frame each saved image came from.

    Returns:
        List of extracted frame paths, or (paths, indices) if return_indices.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if total_frames <= 0:
        # fall back to streaming through
        total_frames = None
        print(f"Video metadata unavailable, falling back to streaming through entire file")

    print(f"Video: {video_path}")
    print(f"  {width}x{height} @ {fps:.1f}fps, {total_frames} total frames")

    os.makedirs(output_dir, exist_ok=True)

    if total_frames is not None and total_frames > 0:
        if n_frames >= total_frames:
            indices = list(range(total_frames))
        else:
            indices = np.linspace(0, total_frames - 1, n_frames, dtype=int).tolist()
    else:
        # streaming mode: collect all frames then sub-sample
        indices = None

    extracted = []
    saved_indices = []

    if indices is not None:
        for i, idx in enumerate(indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
            out_path = os.path.join(output_dir, f"{prefix}_{i:03d}.jpg")
            cv2.imwrite(out_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            extracted.append(out_path)
            saved_indices.append(int(idx))
            print(f"  Extracted frame {idx} → {os.path.basename(out_path)}")
    else:
        # Streaming fallback
        all_frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            all_frames.append(frame)
        total = len(all_frames)
        if total == 0:
            cap.release()
            raise RuntimeError("No frames could be read from video")
        if n_frames >= total:
            sample_indices = list(range(total))
        else:
            sample_indices = np.linspace(0, total - 1, n_frames, dtype=int).tolist()
        for i, idx in enumerate(sample_indices):
            out_path = os.path.join(output_dir, f"{prefix}_{i:03d}.jpg")
            cv2.imwrite(out_path, all_frames[idx], [cv2.IMWRITE_JPEG_QUALITY, 95])
            extracted.append(out_path)
            saved_indices.append(int(idx))

    cap.release()
    print(f"Extracted {len(extracted)} frames to {output_dir}")
    # Save a manifest mapping output filenames → original video frame indices.
    # This is essential for the scale picker to do frame-by-frame LK tracking
    # on the original video and then sample at the saved-frame timestamps.
    manifest_path = os.path.join(output_dir, "_video_frame_manifest.json")
    manifest = {
        "video_path": os.path.abspath(video_path),
        "total_video_frames": int(total_frames) if total_frames else None,
        "fps": float(fps) if fps else None,
        "saved": [
            {"filename": os.path.basename(p), "video_frame_index": idx}
            for p, idx in zip(extracted, saved_indices)
        ],
    }
    import json as _json
    with open(manifest_path, "w") as f:
        _json.dump(manifest, f, indent=2)
    print(f"Saved manifest: {manifest_path}")

    if return_indices:
        return extracted, saved_indices
    return extracted


def is_video_file(path):
    """Check if path looks like a video file."""
    if not os.path.isfile(path):
        return False
    ext = os.path.splitext(path)[1].lower()
    return ext in (".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".wmv", ".flv")


def main():
    parser = argparse.ArgumentParser(description="Extract frames from a video")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--output_dir", required=True, help="Directory for extracted frames")
    parser.add_argument("--n_frames", type=int, default=8, help="Number of frames")
    parser.add_argument("--prefix", default="frame", help="Filename prefix")
    args = parser.parse_args()

    extract_frames(args.video, args.output_dir, args.n_frames, args.prefix)


if __name__ == "__main__":
    main()
