"""
ArUco marker detection and metric scale calibration.

Detects ArUco markers in images and computes a scale factor to convert
arbitrary-scale 3D reconstructions to metric (real-world) measurements.

This module runs in the 'leg_pipeline' environment (opencv-contrib-python required).
"""

import os
import json
import glob
import numpy as np
import cv2


# ArUco dictionary types supported
ARUCO_DICTS = {
    "4x4_50": cv2.aruco.DICT_4X4_50,
    "4x4_100": cv2.aruco.DICT_4X4_100,
    "5x5_50": cv2.aruco.DICT_5X5_50,
    "5x5_100": cv2.aruco.DICT_5X5_100,
    "6x6_50": cv2.aruco.DICT_6X6_50,
    "6x6_100": cv2.aruco.DICT_6X6_100,
    "original": cv2.aruco.DICT_ARUCO_ORIGINAL,
}


def detect_aruco_markers(image_path, dict_type="4x4_50"):
    """Detect ArUco markers in an image.

    Args:
        image_path: Path to image file.
        dict_type: ArUco dictionary type (default: 4x4_50).

    Returns:
        List of detected markers, each with:
            - id: Marker ID
            - corners: 4 corner points as (4, 2) array
            - center: Center point (x, y)
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICTS[dict_type])
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    corners, ids, rejected = detector.detectMarkers(gray)

    markers = []
    if ids is not None:
        for i, marker_id in enumerate(ids.flatten()):
            corner_pts = corners[i][0]  # (4, 2) array
            center = corner_pts.mean(axis=0)
            markers.append(
                {
                    "id": int(marker_id),
                    "corners": corner_pts.tolist(),
                    "center": [float(center[0]), float(center[1])],
                }
            )

    return markers


def compute_marker_pixel_size(corners):
    """Compute the pixel side length of an ArUco marker from its 4 corners.

    Uses the average of all 4 side lengths for robustness.
    """
    corners = np.array(corners)
    side_lengths = []
    for i in range(4):
        p1 = corners[i]
        p2 = corners[(i + 1) % 4]
        side_lengths.append(np.linalg.norm(p2 - p1))
    return float(np.mean(side_lengths))


def compute_scale_factor(
    image_dir,
    marker_real_size_cm,
    dict_type="4x4_50",
    target_marker_id=None,
):
    """Compute the scale factor (cm per pixel) from ArUco markers in images.

    Searches all images in image_dir for ArUco markers and computes the
    average scale factor across all detections.

    Args:
        image_dir: Directory containing images.
        marker_real_size_cm: Real-world side length of the ArUco marker in cm.
        dict_type: ArUco dictionary type.
        target_marker_id: If specified, only use this marker ID.

    Returns:
        Dictionary with:
            - scale_cm_per_pixel: Average cm per pixel scale factor
            - detections: Per-image detection details
            - num_detections: Total number of valid detections
    """
    image_extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(image_dir, ext)))
        image_files.extend(glob.glob(os.path.join(image_dir, ext.upper())))
    image_files = sorted(set(image_files))

    detections = {}
    scale_factors = []

    for img_path in image_files:
        img_name = os.path.basename(img_path)
        markers = detect_aruco_markers(img_path, dict_type)

        img_detections = []
        for marker in markers:
            if target_marker_id is not None and marker["id"] != target_marker_id:
                continue

            pixel_size = compute_marker_pixel_size(marker["corners"])
            if pixel_size > 0:
                scale = marker_real_size_cm / pixel_size
                scale_factors.append(scale)
                img_detections.append(
                    {
                        "marker_id": marker["id"],
                        "center": marker["center"],
                        "pixel_size": pixel_size,
                        "scale_cm_per_pixel": scale,
                    }
                )

        if img_detections:
            detections[img_name] = img_detections

    if not scale_factors:
        return {
            "scale_cm_per_pixel": None,
            "per_image_scale": {},
            "detections": detections,
            "num_detections": 0,
            "error": "No ArUco markers detected in any image",
        }

    avg_scale = float(np.mean(scale_factors))
    std_scale = float(np.std(scale_factors)) if len(scale_factors) > 1 else 0.0

    # Build per-image scale map: image_name → best scale for that image
    # If multiple markers in one image, average them.
    # If no marker in an image, it won't have an entry.
    per_image_scale = {}
    for img_name, img_dets in detections.items():
        scales = [d["scale_cm_per_pixel"] for d in img_dets]
        per_image_scale[img_name] = float(np.mean(scales))

    return {
        "scale_cm_per_pixel": avg_scale,  # kept for backward compat
        "per_image_scale": per_image_scale,
        "scale_std": std_scale,
        "detections": detections,
        "num_detections": len(scale_factors),
        "marker_real_size_cm": marker_real_size_cm,
    }


def apply_scale_to_point_cloud(points, scale_factor):
    """Scale a point cloud from arbitrary units to metric (cm).

    Note: AMB3R already outputs metric-scale points, so this function
    is primarily used as a fallback for reference-object-based scaling
    when AMB3R's metric scale is not trusted.

    Args:
        points: (N, 3) array of 3D points.
        scale_factor: Multiplication factor to convert to cm.

    Returns:
        Scaled (N, 3) array.
    """
    return points * scale_factor


def generate_aruco_marker(marker_id=0, dict_type="4x4_50", size_pixels=200,
                          output_path="aruco_marker.png"):
    """Generate and save an ArUco marker image for printing.

    Args:
        marker_id: Marker ID (default: 0).
        dict_type: ArUco dictionary type.
        size_pixels: Size of the marker image in pixels.
        output_path: Path to save the marker image.
    """
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICTS[dict_type])
    marker_image = cv2.aruco.generateImageMarker(aruco_dict, marker_id, size_pixels)

    # Add white border for easier detection
    border = 50
    bordered = np.ones(
        (size_pixels + 2 * border, size_pixels + 2 * border), dtype=np.uint8
    ) * 255
    bordered[border : border + size_pixels, border : border + size_pixels] = marker_image

    cv2.imwrite(output_path, bordered)
    print(f"Saved ArUco marker (ID={marker_id}) to {output_path}")
    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ArUco marker detection & calibration")
    sub = parser.add_subparsers(dest="command")

    # Detect command
    detect_parser = sub.add_parser("detect", help="Detect markers in images")
    detect_parser.add_argument("--image_dir", type=str, required=True)
    detect_parser.add_argument("--dict_type", type=str, default="4x4_50")
    detect_parser.add_argument("--marker_size_cm", type=float, required=True,
                               help="Real-world marker side length in cm")
    detect_parser.add_argument("--output", type=str, default="aruco_calibration.json")

    # Generate command
    gen_parser = sub.add_parser("generate", help="Generate a printable ArUco marker")
    gen_parser.add_argument("--id", type=int, default=0)
    gen_parser.add_argument("--dict_type", type=str, default="4x4_50")
    gen_parser.add_argument("--size", type=int, default=200)
    gen_parser.add_argument("--output", type=str, default="aruco_marker.png")

    args = parser.parse_args()

    if args.command == "detect":
        result = compute_scale_factor(
            args.image_dir, args.marker_size_cm, args.dict_type
        )
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(json.dumps(result, indent=2))

    elif args.command == "generate":
        generate_aruco_marker(args.id, args.dict_type, args.size, args.output)

    else:
        parser.print_help()
