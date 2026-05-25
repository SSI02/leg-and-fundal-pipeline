"""
Debug visualization module.

Generates visual debug outputs at each pipeline stage so you can verify
landmarks, skeleton, measurements, and scale correctness.
"""

import os
import json
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ─── Color palette ────────────────────────────────────────────────
COLORS_BGR = {
    "left_hip": (0, 0, 255), "left_knee": (0, 100, 255), "left_ankle": (0, 200, 255),
    "right_hip": (255, 0, 0), "right_knee": (255, 100, 0), "right_ankle": (255, 200, 0),
}
COLORS_RGB = {
    "left_hip": [1, 0, 0], "left_knee": [1, 0.4, 0], "left_ankle": [1, 0.8, 0],
    "right_hip": [0, 0, 1], "right_knee": [0, 0.4, 1], "right_ankle": [0, 0.8, 1],
}
SKELETON = [
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
    ("left_hip", "right_hip"),
]
JOINT_LABELS_SHORT = {
    "left_hip": "L-Hip", "left_knee": "L-Knee", "left_ankle": "L-Ankle",
    "right_hip": "R-Hip", "right_knee": "R-Knee", "right_ankle": "R-Ankle",
}


# ─── Helper: dashed line ─────────────────────────────────────────
def _draw_dashed_line(img, p1, p2, color, thickness, dash_len=15):
    x1, y1 = p1; x2, y2 = p2
    dist = np.sqrt((x2-x1)**2 + (y2-y1)**2)
    if dist < 1: return
    dx, dy = (x2-x1)/dist*dash_len, (y2-y1)/dist*dash_len
    for i in range(0, int(dist/dash_len), 2):
        cv2.line(img, (int(x1+dx*i), int(y1+dy*i)),
                 (int(x1+dx*min(i+1, int(dist/dash_len))), int(y1+dy*min(i+1, int(dist/dash_len)))),
                 color, thickness)


# ─── Helper: cylinder mesh ───────────────────────────────────────
def _create_line_mesh(p1, p2, color, radius=0.003):
    import open3d as o3d
    p1, p2 = np.array(p1, float), np.array(p2, float)
    length = np.linalg.norm(p2-p1)
    if length < 1e-6: return o3d.geometry.TriangleMesh()
    cyl = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=length, resolution=8, split=1)
    cyl.paint_uniform_color(color)
    d = (p2-p1)/length
    z = np.array([0,0,1])
    ra = np.cross(z, d)
    rn = np.linalg.norm(ra)
    if rn < 1e-6:
        R = np.diag([1,-1,-1]) if d[2]<0 else np.eye(3)
    else:
        ra /= rn; a = np.arccos(np.clip(np.dot(z,d),-1,1))
        K = np.array([[0,-ra[2],ra[1]],[ra[2],0,-ra[0]],[-ra[1],ra[0],0]])
        R = np.eye(3)+np.sin(a)*K+(1-np.cos(a))*(K@K)
    cyl.rotate(R, center=[0,0,0]); cyl.translate((p1+p2)/2)
    return cyl


# ─── 2D Debug: Pose overlay ──────────────────────────────────────
def debug_pose_2d(image_path, pose_data, output_path, measurements_2d=None):
    img = cv2.imread(image_path)
    if img is None: return
    lk = pose_data["leg_keypoints"]

    skel_colors = {
        ("left_hip","left_knee"):(0,0,255), ("left_knee","left_ankle"):(0,100,255),
        ("right_hip","right_knee"):(255,0,0), ("right_knee","right_ankle"):(255,100,0),
        ("left_hip","right_hip"):(0,255,0),
    }
    for (j1,j2), c in skel_colors.items():
        if lk[j1]["score"]>0.3 and lk[j2]["score"]>0.3:
            cv2.line(img, (int(lk[j1]["x"]),int(lk[j1]["y"])), (int(lk[j2]["x"]),int(lk[j2]["y"])), c, 3)

    for name, kp in lk.items():
        if kp["score"]<0.3: continue
        x,y = int(kp["x"]), int(kp["y"])
        c = COLORS_BGR.get(name, (255,255,255))
        cv2.circle(img, (x,y), 8, c, -1)
        cv2.circle(img, (x,y), 8, (0,0,0), 2)
        cv2.putText(img, f"{JOINT_LABELS_SHORT.get(name,name)} ({kp['score']:.2f})", (x+12,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1)

    for side in ["left","right"]:
        h,a = lk[f"{side}_hip"], lk[f"{side}_ankle"]
        if h["score"]>0.3 and a["score"]>0.3:
            _draw_dashed_line(img, (int(h["x"]),int(h["y"])), (int(a["x"]),int(a["y"])), (128,128,128), 2)

    if measurements_2d:
        for sk in ["left_leg","right_leg"]:
            leg = measurements_2d.get(sk)
            if not leg: continue
            knee = lk[f"{leg['side']}_knee"]
            if knee["score"]<0.3: continue
            cls = leg.get("classification","?")
            tc = (0,255,0) if cls=="normal" else (0,165,255) if cls=="varus" else (255,0,255)
            cv2.putText(img, f"HKA={leg.get('hka_angle',0):.1f} dev={leg.get('hka_deviation',0):+.1f} [{cls}]",
                        (int(knee["x"])+15, int(knee["y"])+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, tc, 2)

    cv2.imwrite(output_path, img)
    print(f"  Debug 2D pose: {output_path}")


# ─── 3D Debug: Full annotated visualization ───────────────────────

def debug_3d_landmarks(point_cloud_path, joints_3d, output_dir, assessment_3d=None,
                       confidence_data=None):
    """
    Create comprehensive 3D debug with:
    - Labeled landmark spheres (with text label point clusters)
    - Skeleton bones
    - Mechanical axis lines
    - HKA angle arcs
    - Knee/ankle gap lines
    - Leg length measurement lines
    - Height line
    - Legend in matplotlib figure
    - Anatomical sanity checks
    """
    import open3d as o3d
    os.makedirs(output_dir, exist_ok=True)

    pcd = o3d.io.read_point_cloud(point_cloud_path)
    pcd_pts = np.asarray(pcd.points)
    print(f"  Debug 3D: {len(pcd_pts)} points")

    # ── Detect vertical axis (largest extent of hip→ankle) ──
    hip_mid = None
    ankle_mid = None
    if joints_3d.get("left_hip") and joints_3d.get("right_hip"):
        hip_mid = (np.array(joints_3d["left_hip"]) + np.array(joints_3d["right_hip"])) / 2
    if joints_3d.get("left_ankle") and joints_3d.get("right_ankle"):
        ankle_mid = (np.array(joints_3d["left_ankle"]) + np.array(joints_3d["right_ankle"])) / 2

    vert_axis = 1  # default Y
    if hip_mid is not None and ankle_mid is not None:
        vert_vec = hip_mid - ankle_mid
        vert_axis = int(np.argmax(np.abs(vert_vec)))
    axis_names = {0: "X", 1: "Y", 2: "Z"}
    print(f"  Detected vertical axis: {axis_names[vert_axis]}")

    # ── Confidence-based height calculation ──
    height_raw = (pcd_pts[:, vert_axis].max() - pcd_pts[:, vert_axis].min()) if len(pcd_pts) > 0 else 0

    # Use confidence if available
    if confidence_data is not None:
        flat_conf = confidence_data.reshape(-1)
        conf_thresh = np.percentile(flat_conf, 75)
        high_conf_mask = flat_conf > conf_thresh
        if high_conf_mask.sum() > 100:
            hi_pts = pcd_pts[:len(flat_conf)][high_conf_mask] if len(pcd_pts) >= len(flat_conf) else pcd_pts
            height_filtered = hi_pts[:, vert_axis].max() - hi_pts[:, vert_axis].min()
        else:
            height_filtered = height_raw
    else:
        # Fallback: use statistical outlier removal
        vals = pcd_pts[:, vert_axis]
        p5, p95 = np.percentile(vals, [5, 95])
        mask = (vals >= p5) & (vals <= p95)
        height_filtered = vals[mask].max() - vals[mask].min() if mask.sum() > 0 else height_raw

    # ── Anatomical sanity check ──
    sanity = _anatomical_sanity_check(joints_3d)

    # ── Build 3D meshes ──
    meshes = []

    # Joint spheres with LABELED point clusters
    for name, pos in joints_3d.items():
        if pos is None: continue
        color = COLORS_RGB.get(name, [1,1,1])
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.012)
        sphere.translate(pos)
        sphere.paint_uniform_color(color)
        meshes.append(sphere)

    # Skeleton bones
    bone_defs = [
        ("left_hip","left_knee",[1,0,0],0.005), ("left_knee","left_ankle",[1,0.5,0],0.005),
        ("right_hip","right_knee",[0,0,1],0.005), ("right_knee","right_ankle",[0,0.5,1],0.005),
        ("left_hip","right_hip",[0,1,0],0.004),
    ]
    for j1,j2,c,r in bone_defs:
        p1,p2 = joints_3d.get(j1), joints_3d.get(j2)
        if p1 and p2: meshes.append(_create_line_mesh(p1,p2,c,r))

    # Mechanical axis
    for side,c in [("left",[1,0.3,0.3]),("right",[0.3,0.3,1])]:
        h,a = joints_3d.get(f"{side}_hip"), joints_3d.get(f"{side}_ankle")
        if h and a: meshes.append(_create_line_mesh(h,a,c,0.002))

    # Knee gap
    lk,rk = joints_3d.get("left_knee"), joints_3d.get("right_knee")
    if lk and rk: meshes.append(_create_line_mesh(lk,rk,[1,0,1],0.003))

    # Ankle gap
    la,ra = joints_3d.get("left_ankle"), joints_3d.get("right_ankle")
    if la and ra: meshes.append(_create_line_mesh(la,ra,[0,1,0.5],0.003))

    # Save skeleton mesh
    combined = o3d.geometry.TriangleMesh()
    for m in meshes: combined += m
    o3d.io.write_triangle_mesh(os.path.join(output_dir, "debug_landmarks_skeleton.ply"), combined)

    # Save point cloud with LABELED landmarks (colored clusters with offset text-points)
    landmark_pcd = o3d.geometry.PointCloud()
    pts_list, cols_list = [], []
    for name, pos in joints_3d.items():
        if pos is None: continue
        color = COLORS_RGB.get(name, [1,1,1])
        p = np.array(pos)
        # Sphere cluster at landmark
        for _ in range(500):
            pts_list.append(p + np.random.randn(3)*0.004)
            cols_list.append(color)
        # "Label" trail: a line of bright points going outward from landmark
        # This creates a visible tag you can trace back to the landmark
        label_dir = np.array([0.03, 0.01, 0]) # offset direction for label
        for i in range(30):
            t = i / 30
            pts_list.append(p + label_dir * t + np.random.randn(3)*0.001)
            cols_list.append(color)
        # End tag: brighter cluster at label endpoint
        label_end = p + label_dir
        for _ in range(200):
            pts_list.append(label_end + np.random.randn(3)*0.003)
            cols_list.append([min(1, c+0.3) for c in color])  # brighter

    if pts_list:
        landmark_pcd.points = o3d.utility.Vector3dVector(np.array(pts_list))
        landmark_pcd.colors = o3d.utility.Vector3dVector(np.array(cols_list))
        merged = pcd + landmark_pcd
        o3d.io.write_point_cloud(os.path.join(output_dir, "debug_pcd_with_landmarks.ply"), merged)

    # ── Generate matplotlib figure ──
    _generate_measurement_figure(joints_3d, assessment_3d, pcd_pts, output_dir,
                                 vert_axis, height_filtered, sanity)

    # ── Save text summary ──
    _save_text_summary(joints_3d, assessment_3d, pcd_pts, output_dir,
                       vert_axis, height_raw, height_filtered, sanity)

    print(f"  Debug 3D outputs saved to: {output_dir}")


def _anatomical_sanity_check(joints_3d):
    """Check if measurements are anatomically plausible. Returns dict of checks."""
    checks = {}
    warnings = []

    # Expected ranges (meters)
    expected = {
        "inter_hip": (0.20, 0.40),      # 20-40cm
        "femur": (0.30, 0.55),           # 30-55cm
        "tibia": (0.28, 0.50),           # 28-50cm
        "total_leg": (0.60, 1.00),       # 60-100cm
        "inter_knee": (0.0, 0.30),       # 0-30cm
        "inter_ankle": (0.0, 0.35),      # 0-35cm
    }

    lh, rh = joints_3d.get("left_hip"), joints_3d.get("right_hip")
    lk, rk = joints_3d.get("left_knee"), joints_3d.get("right_knee")
    la, ra = joints_3d.get("left_ankle"), joints_3d.get("right_ankle")

    def d(a, b):
        if a is None or b is None: return None
        return float(np.linalg.norm(np.array(a)-np.array(b)))

    measurements = {
        "inter_hip": d(lh, rh),
        "left_femur": d(lh, lk),
        "right_femur": d(rh, rk),
        "left_tibia": d(lk, la),
        "right_tibia": d(rk, ra),
        "inter_knee": d(lk, rk),
        "inter_ankle": d(la, ra),
    }

    if measurements["left_femur"] and measurements["left_tibia"]:
        measurements["left_total_leg"] = measurements["left_femur"] + measurements["left_tibia"]
    if measurements["right_femur"] and measurements["right_tibia"]:
        measurements["right_total_leg"] = measurements["right_femur"] + measurements["right_tibia"]

    # Check each
    for key, val in measurements.items():
        if val is None: continue
        check_key = key.replace("left_","").replace("right_","")
        if check_key in expected:
            lo, hi = expected[check_key]
            ok = lo <= val <= hi
            checks[key] = {"value_m": val, "value_cm": val*100, "expected_cm": (lo*100, hi*100), "ok": ok}
            if not ok:
                if val < lo:
                    ratio = lo / val
                    warnings.append(f"{key}: {val*100:.1f}cm is {ratio:.1f}x too small (expected {lo*100:.0f}-{hi*100:.0f}cm). Scale likely off.")
                else:
                    ratio = val / hi
                    warnings.append(f"{key}: {val*100:.1f}cm is {ratio:.1f}x too large (expected {lo*100:.0f}-{hi*100:.0f}cm).")

    # Estimate scale correction factor
    scale_estimates = []
    for key in ["inter_hip", "left_femur", "right_femur", "left_tibia", "right_tibia"]:
        if key not in checks: continue
        check_key = key.replace("left_","").replace("right_","")
        if check_key in expected:
            mid_expected = (expected[check_key][0] + expected[check_key][1]) / 2
            scale_estimates.append(mid_expected / checks[key]["value_m"])

    avg_scale = float(np.median(scale_estimates)) if scale_estimates else 1.0

    return {
        "checks": checks,
        "warnings": warnings,
        "estimated_scale_correction": avg_scale,
        "scale_is_off": any(not c["ok"] for c in checks.values()),
        "measurements_raw": measurements,
    }


def _generate_measurement_figure(joints_3d, assessment_3d, pcd_pts, output_dir,
                                  vert_axis, height_filtered, sanity):
    fig = plt.figure(figsize=(24, 10))
    fig.suptitle("3D Clinical Measurements Debug", fontsize=16, fontweight="bold")

    # Subsample for plotting
    if len(pcd_pts) > 20000:
        idx = np.random.choice(len(pcd_pts), 20000, replace=False)
        pp = pcd_pts[idx]
    else:
        pp = pcd_pts

    ax_map = {0: "X", 1: "Y", 2: "Z"}
    # Choose axes for front view: horizontal = non-vertical non-depth
    axes_order = [i for i in range(3) if i != vert_axis]

    # ── Panel 1: Front view ──
    ax1 = fig.add_subplot(141)
    h_ax, d_ax = axes_order[0], axes_order[1]
    ax1.set_title(f"Front View ({ax_map[h_ax]}-{ax_map[vert_axis]})", fontsize=11)
    ax1.scatter(pp[:, h_ax], pp[:, vert_axis], s=0.1, c="gray", alpha=0.3)

    for name, pos in joints_3d.items():
        if pos is None: continue
        c = COLORS_RGB.get(name, [0.5,0.5,0.5])
        ax1.scatter(pos[h_ax], pos[vert_axis], s=120, c=[c], edgecolors="black", zorder=5, linewidths=1.5)
        ax1.annotate(JOINT_LABELS_SHORT.get(name, name),
                     (pos[h_ax], pos[vert_axis]), fontsize=8, fontweight="bold", ha="center",
                     xytext=(0, 10), textcoords="offset points",
                     bbox=dict(boxstyle="round,pad=0.2", fc="yellow", alpha=0.8),
                     arrowprops=dict(arrowstyle="->", color="black"))

    # Skeleton
    skel_colors = {"left_hip":"red","left_knee":"red","right_hip":"blue","right_knee":"blue"}
    for j1,j2 in SKELETON:
        p1,p2 = joints_3d.get(j1), joints_3d.get(j2)
        if p1 and p2:
            c = "red" if "left" in j1 else "blue" if "right" in j1 else "green"
            ax1.plot([p1[h_ax],p2[h_ax]], [p1[vert_axis],p2[vert_axis]], c=c, linewidth=2.5, zorder=4)

    # Mechanical axis
    for side,c in [("left","red"),("right","blue")]:
        h,a = joints_3d.get(f"{side}_hip"), joints_3d.get(f"{side}_ankle")
        if h and a:
            ax1.plot([h[h_ax],a[h_ax]], [h[vert_axis],a[vert_axis]], c=c, linewidth=1, linestyle="--", alpha=0.4)

    # Knee gap
    lk,rk = joints_3d.get("left_knee"), joints_3d.get("right_knee")
    if lk and rk:
        ax1.plot([lk[h_ax],rk[h_ax]], [lk[vert_axis],rk[vert_axis]], c="magenta", linewidth=2, linestyle=":")
        mid_h = (lk[h_ax]+rk[h_ax])/2; mid_v = (lk[vert_axis]+rk[vert_axis])/2
        gap = np.linalg.norm(np.array(lk)-np.array(rk))*100
        ax1.annotate(f"Knee gap\n{gap:.1f}cm", (mid_h,mid_v), fontsize=8, ha="center", color="magenta", fontweight="bold")

    # Ankle gap
    la,ra = joints_3d.get("left_ankle"), joints_3d.get("right_ankle")
    if la and ra:
        ax1.plot([la[h_ax],ra[h_ax]], [la[vert_axis],ra[vert_axis]], c="teal", linewidth=2, linestyle=":")
        mid_h = (la[h_ax]+ra[h_ax])/2; mid_v = (la[vert_axis]+ra[vert_axis])/2
        gap = np.linalg.norm(np.array(la)-np.array(ra))*100
        ax1.annotate(f"Ankle gap\n{gap:.1f}cm", (mid_h,mid_v), fontsize=8, ha="center", color="teal", fontweight="bold")

    ax1.set_aspect("equal"); ax1.set_xlabel(f"{ax_map[h_ax]} (m)"); ax1.set_ylabel(f"{ax_map[vert_axis]} (m)")

    # ── Panel 2: Side view ──
    ax2 = fig.add_subplot(142)
    ax2.set_title(f"Side View ({ax_map[d_ax]}-{ax_map[vert_axis]})", fontsize=11)
    ax2.scatter(pp[:, d_ax], pp[:, vert_axis], s=0.1, c="gray", alpha=0.3)
    for name, pos in joints_3d.items():
        if pos is None: continue
        c = COLORS_RGB.get(name, [0.5,0.5,0.5])
        ax2.scatter(pos[d_ax], pos[vert_axis], s=100, c=[c], edgecolors="black", zorder=5)
        ax2.annotate(JOINT_LABELS_SHORT.get(name,name), (pos[d_ax],pos[vert_axis]),
                     fontsize=7, ha="center", xytext=(0,8), textcoords="offset points",
                     bbox=dict(boxstyle="round,pad=0.2",fc="white",alpha=0.7))
    for j1,j2 in SKELETON:
        p1,p2 = joints_3d.get(j1), joints_3d.get(j2)
        if p1 and p2:
            c = "red" if "left" in j1 else "blue" if "right" in j1 else "green"
            ax2.plot([p1[d_ax],p2[d_ax]], [p1[vert_axis],p2[vert_axis]], c=c, linewidth=2)
    ax2.set_aspect("equal"); ax2.set_xlabel(f"{ax_map[d_ax]} (m)"); ax2.set_ylabel(f"{ax_map[vert_axis]} (m)")

    # ── Panel 3: Measurements table ──
    ax3 = fig.add_subplot(143)
    ax3.axis("off"); ax3.set_title("Measurements", fontsize=11)

    rows = []
    colors = []

    # Height
    rows.append(["Height (filtered)", f"{height_filtered*100:.1f} cm"])
    colors.append(["white","white"])

    if assessment_3d:
        for sk,label in [("left_leg","LEFT LEG"),("right_leg","RIGHT LEG")]:
            leg = assessment_3d.get(sk)
            if not leg: continue
            cls = leg.get("classification","?")
            bg = "#ddffdd" if cls=="normal" else "#ffdddd"
            rows.append([f"--- {label} ---", f"{cls} ({leg.get('severity','?')})"])
            colors.append([bg,bg])
            rows.append(["  HKA Angle", f"{leg.get('hka_angle_3d',0):.1f}°"])
            colors.append(["white","white"])
            rows.append(["  HKA Deviation", f"{leg.get('hka_deviation_3d',0):+.1f}°"])
            colors.append(["white","white"])
            rows.append(["  MAD", f"{leg.get('mad_3d',0)*100:.2f} cm"])
            colors.append(["white","white"])
            rows.append(["  Femur", f"{leg.get('femur_length_3d',0)*100:.1f} cm"])
            colors.append(["white","white"])
            rows.append(["  Tibia", f"{leg.get('tibia_length_3d',0)*100:.1f} cm"])
            colors.append(["white","white"])
            rows.append(["  Total Leg", f"{leg.get('total_leg_length_3d',0)*100:.1f} cm"])
            colors.append(["white","white"])
        rows.append(["Knee Gap", f"{assessment_3d.get('intercondylar_distance_3d',0)*100:.1f} cm"])
        colors.append(["#ffe0ff","#ffe0ff"])
        rows.append(["Ankle Gap", f"{assessment_3d.get('intermalleolar_distance_3d',0)*100:.1f} cm"])
        colors.append(["#e0ffff","#e0ffff"])
        rows.append(["Leg Len Diff", f"{assessment_3d.get('leg_length_difference_3d',0)*100:.1f} cm"])
        colors.append(["white","white"])
        oc = assessment_3d.get("overall_classification","?")
        bg = "#ddffdd" if "Normal" in oc else "#ffdddd"
        rows.append(["Overall", oc])
        colors.append([bg,bg])

    if rows:
        t = ax3.table(cellText=rows, cellColours=colors, colLabels=["Metric","Value"],
                       loc="upper center", cellLoc="left", colWidths=[0.5,0.4])
        t.auto_set_font_size(False); t.set_fontsize(8); t.scale(1,1.2)

    # ── Panel 4: Sanity checks + legend ──
    ax4 = fig.add_subplot(144)
    ax4.axis("off"); ax4.set_title("Sanity Checks & Legend", fontsize=11)

    y = 0.95
    ax4.text(0.05, y, "ANATOMICAL SANITY CHECKS:", transform=ax4.transAxes, fontsize=10, fontweight="bold")
    y -= 0.04

    if sanity["scale_is_off"]:
        ax4.text(0.05, y, f"⚠ SCALE LIKELY WRONG (correction ~{sanity['estimated_scale_correction']:.1f}x)",
                 transform=ax4.transAxes, fontsize=9, color="red", fontweight="bold")
        y -= 0.03

    for key, check in sanity["checks"].items():
        status = "✓" if check["ok"] else "✗"
        color = "green" if check["ok"] else "red"
        text = f"{status} {key}: {check['value_cm']:.1f}cm (expected {check['expected_cm'][0]:.0f}-{check['expected_cm'][1]:.0f}cm)"
        ax4.text(0.05, y, text, transform=ax4.transAxes, fontsize=7, color=color)
        y -= 0.025

    if sanity["warnings"]:
        y -= 0.02
        ax4.text(0.05, y, "WARNINGS:", transform=ax4.transAxes, fontsize=9, fontweight="bold", color="red")
        y -= 0.025
        for w in sanity["warnings"]:
            ax4.text(0.05, y, w, transform=ax4.transAxes, fontsize=6, color="red", wrap=True)
            y -= 0.03

    # Legend
    y -= 0.03
    ax4.text(0.05, y, "LEGEND:", transform=ax4.transAxes, fontsize=10, fontweight="bold")
    y -= 0.03
    legend = [
        ("red ●/line", "Left hip/femur"), ("orange line", "Left tibia"),
        ("blue ●/line", "Right hip/femur"), ("cyan line", "Right tibia"),
        ("green line", "Pelvis"), ("dashed line", "Mechanical axis"),
        ("magenta dotted", "Knee gap"), ("teal dotted", "Ankle gap"),
    ]
    for sym, desc in legend:
        ax4.text(0.08, y, f"{sym}: {desc}", transform=ax4.transAxes, fontsize=7)
        y -= 0.025

    y -= 0.02
    ax4.text(0.05, y, f"Vertical axis: {['X','Y','Z'][vert_axis]}", transform=ax4.transAxes, fontsize=8, fontstyle="italic")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "debug_3d_measurements.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Debug 3D figure saved")


def _save_text_summary(joints_3d, assessment_3d, pcd_pts, output_dir,
                        vert_axis, height_raw, height_filtered, sanity):
    with open(os.path.join(output_dir, "debug_3d_summary.txt"), "w") as f:
        f.write("=" * 60 + "\n")
        f.write("3D CLINICAL MEASUREMENTS DEBUG SUMMARY\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Vertical axis: {'XYZ'[vert_axis]}\n\n")

        f.write("LANDMARK POSITIONS:\n" + "-"*40 + "\n")
        for name in ["left_hip","right_hip","left_knee","right_knee","left_ankle","right_ankle"]:
            pos = joints_3d.get(name)
            label = JOINT_LABELS_SHORT.get(name, name)
            if pos:
                f.write(f"  {label:10s} ({name:15s}): [{pos[0]*100:+7.1f}, {pos[1]*100:+7.1f}, {pos[2]*100:+7.1f}] cm\n")
            else:
                f.write(f"  {label:10s} ({name:15s}): FAILED\n")

        f.write(f"\nHEIGHT:\n")
        f.write(f"  Raw (all points):        {height_raw*100:.1f} cm\n")
        f.write(f"  Filtered (high conf):    {height_filtered*100:.1f} cm\n")

        if assessment_3d:
            f.write(f"\nMEASUREMENTS:\n" + "-"*40 + "\n")
            for sk in ["left_leg","right_leg"]:
                leg = assessment_3d.get(sk)
                if not leg: continue
                f.write(f"\n{leg['side'].upper()} LEG:\n")
                f.write(f"  HKA angle:      {leg.get('hka_angle_3d',0):.2f}°\n")
                f.write(f"  HKA deviation:  {leg.get('hka_deviation_3d',0):+.2f}°\n")
                f.write(f"  MAD:            {leg.get('mad_3d',0)*100:.2f} cm\n")
                f.write(f"  Femur:          {leg.get('femur_length_3d',0)*100:.2f} cm\n")
                f.write(f"  Tibia:          {leg.get('tibia_length_3d',0)*100:.2f} cm\n")
                f.write(f"  Total leg:      {leg.get('total_leg_length_3d',0)*100:.2f} cm\n")
                f.write(f"  Classification: {leg.get('classification','?')} ({leg.get('severity','?')})\n")

            f.write(f"\nKnee gap:      {assessment_3d.get('intercondylar_distance_3d',0)*100:.2f} cm\n")
            f.write(f"Ankle gap:     {assessment_3d.get('intermalleolar_distance_3d',0)*100:.2f} cm\n")
            f.write(f"Leg len diff:  {assessment_3d.get('leg_length_difference_3d',0)*100:.2f} cm\n")
            f.write(f"Overall:       {assessment_3d.get('overall_classification','?')}\n")

        f.write(f"\n{'='*60}\n")
        f.write(f"ANATOMICAL SANITY CHECKS:\n{'='*60}\n\n")
        f.write(f"Scale correction estimate: {sanity['estimated_scale_correction']:.2f}x\n")
        f.write(f"Scale is off: {sanity['scale_is_off']}\n\n")
        for key, check in sanity["checks"].items():
            status = "OK" if check["ok"] else "FAIL"
            f.write(f"  [{status:4s}] {key}: {check['value_cm']:.1f}cm (expected {check['expected_cm'][0]:.0f}-{check['expected_cm'][1]:.0f}cm)\n")
        if sanity["warnings"]:
            f.write(f"\nWARNINGS:\n")
            for w in sanity["warnings"]:
                f.write(f"  ! {w}\n")


# ─── AMB3R debug ──────────────────────────────────────────────────
def debug_amb3r_pointmap(amb3r_npz_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    data = np.load(amb3r_npz_path, allow_pickle=True)
    pts, conf, imgs = data["points_per_frame"], data["conf_per_frame"], data["images_per_frame"]
    T = pts.shape[0]
    for t in range(T):
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        fig.suptitle(f"Frame {t} — AMB3R Debug", fontsize=14)
        axes[0].imshow(np.clip(imgs[t], 0, 1)); axes[0].set_title("Input"); axes[0].axis("off")
        depth = np.linalg.norm(pts[t], axis=-1)
        valid = depth > 0.01
        dv = np.zeros_like(depth)
        if valid.any():
            dv[valid] = depth[valid]
            lo,hi = np.percentile(dv[valid],[2,98])
            dv = np.clip((dv-lo)/(hi-lo+1e-8),0,1)
        axes[1].imshow(dv, cmap="viridis"); axes[1].set_title("Depth"); axes[1].axis("off")
        cf = conf[t][...,0] if conf[t].ndim==3 else conf[t]
        axes[2].imshow(cf, cmap="hot"); axes[2].set_title(f"Conf (mean={cf.mean():.2f})"); axes[2].axis("off")
        fp = pts[t].reshape(-1,3)
        fv = np.linalg.norm(fp,axis=-1)>0.01
        if fv.sum()>0:
            s = fp[fv]; s = s[np.random.choice(len(s),min(10000,len(s)),replace=False)] if len(s)>10000 else s
            axes[3].scatter(s[:,0],s[:,2],s=0.1,c=s[:,1],cmap="viridis")
            axes[3].set_title("Top-down XZ"); axes[3].set_aspect("equal")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"debug_amb3r_frame_{t:02d}.png"), dpi=120, bbox_inches="tight")
        plt.close()


# ─── Landmark projection debug ────────────────────────────────────
def debug_landmark_projection(image_path, pose_keypoints, joints_3d, output_path):
    img = cv2.imread(image_path)
    if img is None: return
    for name in ["left_hip","right_hip","left_knee","right_knee","left_ankle","right_ankle"]:
        kp = pose_keypoints.get(name)
        j3d = joints_3d.get(name)
        if kp is None or kp["score"]<0.3: continue
        px,py = int(kp["x"]), int(kp["y"])
        c = COLORS_BGR.get(name, (255,255,255))
        cv2.circle(img, (px,py), 8, c, -1)
        cv2.circle(img, (px,py), 8, (0,0,0), 2)
        label = JOINT_LABELS_SHORT.get(name, name)
        if j3d:
            txt = f"{label}: ({j3d[0]*100:.1f}, {j3d[1]*100:.1f}, {j3d[2]*100:.1f})cm"
            cv2.putText(img, txt, (px+12, py+5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, c, 1)
        else:
            cv2.putText(img, f"{label}: FAILED", (px+12, py+5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)
    cv2.imwrite(output_path, img)


# ─── Master debug ─────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════
#  Visualizations for new pipeline stages
# ════════════════════════════════════════════════════════════════════


# ─── Stage: ArUco marker detection ─────────────────────────────────

def viz_aruco_detection(image_path, calibration_json_path, output_dir):
    """For each image with detected ArUco markers, draw the marker outline,
    corners, and the computed pixel side length + cm/pixel scale."""
    if not os.path.exists(calibration_json_path):
        return None
    with open(calibration_json_path) as f:
        cal = json.load(f)
    detections = cal.get("detections", {})
    if not detections:
        return None
    os.makedirs(output_dir, exist_ok=True)

    saved = []
    for img_name, dets in detections.items():
        img_path = os.path.join(image_path, img_name) if os.path.isdir(image_path) else image_path
        if not os.path.exists(img_path):
            continue
        img = cv2.imread(img_path)
        if img is None:
            continue
        for det in dets:
            corners = np.array(cal.get("detections", {}).get(img_name, [{}])[0].get("center"))
            # We don't store all 4 corners in the cal JSON, only center + pixel_size.
            # Draw a circle at center and label.
            cx, cy = int(det["center"][0]), int(det["center"][1])
            ps = det.get("pixel_size", 0)
            scale = det.get("scale_cm_per_pixel", 0)
            cv2.circle(img, (cx, cy), max(8, int(ps * 0.5)), (0, 255, 0), 4)
            label = f"ID={det.get('marker_id', '?')}  {ps:.0f}px  {scale:.4f}cm/px"
            cv2.rectangle(img, (cx - 200, cy - ps - 60), (cx + 200, cy - ps - 20), (0, 0, 0), -1)
            cv2.putText(img, label, (cx - 195, cy - ps - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        out_path = os.path.join(output_dir, f"aruco_{img_name}.jpg")
        cv2.imwrite(out_path, img)
        saved.append(out_path)
    return saved


# ─── Stage: Manual scale picker ────────────────────────────────────

def viz_scale_calibration(image_path, calibration_json_path, output_dir):
    """For each image with manual scale calibration, show the 2 clicked
    points, the connecting line, and the computed cm/pixel scale."""
    if not os.path.exists(calibration_json_path):
        return None
    with open(calibration_json_path) as f:
        cal = json.load(f)
    if not cal:
        return None

    os.makedirs(output_dir, exist_ok=True)
    saved = []
    for img_name, c in cal.items():
        if "p1" not in c or "p2" not in c:
            continue
        img_path = os.path.join(image_path, img_name) if os.path.isdir(image_path) else image_path
        if not os.path.exists(img_path):
            continue
        img = cv2.imread(img_path)
        if img is None:
            continue
        p1 = tuple(int(v) for v in c["p1"])
        p2 = tuple(int(v) for v in c["p2"])
        cv2.line(img, p1, p2, (255, 200, 0), 4, cv2.LINE_AA)
        cv2.circle(img, p1, 12, (0, 0, 255), -1)
        cv2.circle(img, p1, 12, (255, 255, 255), 2)
        cv2.circle(img, p2, 12, (0, 200, 0), -1)
        cv2.circle(img, p2, 12, (255, 255, 255), 2)
        # Label
        midx = (p1[0] + p2[0]) // 2
        midy = (p1[1] + p2[1]) // 2
        d = float(c.get("real_distance_cm", 0))
        s = float(c.get("scale_cm_per_pixel", 0))
        obj = c.get("object_description", "")
        tag = c.get("tracking", "manual")
        text = f"{d:.2f}cm = {c.get('pixel_distance', 0):.0f}px ({s:.5f}cm/px) [{tag}] {obj}"
        cv2.rectangle(img, (midx - 400, midy - 40), (midx + 400, midy), (0, 0, 0), -1)
        cv2.putText(img, text, (midx - 395, midy - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        out_path = os.path.join(output_dir, f"scale_{img_name}.jpg")
        cv2.imwrite(out_path, img)
        saved.append(out_path)
    return saved


# ─── Stage: SAM3 segmentation overlay ──────────────────────────────

def viz_segmentation_overlay(image_dir, segmentation_dir, output_dir,
                             alpha=0.45, mask_color=(255, 0, 255)):
    """Overlay SAM3 masks on the original images for visual verification."""
    seg_json = os.path.join(segmentation_dir, "segmentation.json")
    if not os.path.exists(seg_json):
        return None
    with open(seg_json) as f:
        seg = json.load(f)

    os.makedirs(output_dir, exist_ok=True)
    saved = []
    for img_name, data in seg.items():
        img_path = data.get("image_path") or os.path.join(image_dir, img_name)
        mask_path = data.get("combined_mask_path")
        if not (img_path and mask_path and os.path.exists(img_path) and os.path.exists(mask_path)):
            continue
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            continue
        if mask.shape != img.shape[:2]:
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Build colored overlay
        overlay = img.copy()
        m = mask > 127
        overlay[m] = (np.array(mask_color) * alpha + img[m] * (1 - alpha)).astype(np.uint8)

        # Draw mask contour for clarity
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), 4)

        # Header text with detection details
        n = data.get("num_detections", 0)
        frac = data.get("combined_mask_fraction", 0) * 100
        scores = [round(d.get("score", 0), 2) for d in data.get("detections", [])]
        header = f"{n} detections  scores={scores}  mask={frac:.1f}% of frame"
        h, w = overlay.shape[:2]
        cv2.rectangle(overlay, (0, 0), (w, 60), (0, 0, 0), -1)
        cv2.putText(overlay, header, (10, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        out_path = os.path.join(output_dir, f"seg_{img_name}.jpg")
        cv2.imwrite(out_path, overlay)
        saved.append(out_path)
    return saved


# ─── Stage: 3D point cloud — multi-view rendering with matplotlib ──

def _render_pcd_matplotlib(points, colors=None, ax=None, sub=200_000, title=None,
                            highlight_pts=None, highlight_color="red", highlight_label=None,
                            extra_lines=None, view_elev=20, view_azim=-60):
    """Scatter-plot a point cloud on a 3D matplotlib axis."""
    if len(points) > sub:
        idx = np.random.RandomState(0).choice(len(points), sub, replace=False)
        points = points[idx]
        if colors is not None:
            colors = colors[idx]
    if ax is None:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection="3d")
    if colors is None:
        c_arg = points[:, 2]
        ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                   c=c_arg, cmap="viridis", s=0.3, alpha=0.5)
    else:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                   c=np.clip(colors, 0, 1), s=0.3, alpha=0.5)
    if highlight_pts is not None and len(highlight_pts):
        hp = np.atleast_2d(highlight_pts)
        ax.scatter(hp[:, 0], hp[:, 1], hp[:, 2], c=highlight_color, s=80,
                   edgecolors="black", linewidth=1, label=highlight_label or "highlight")
    if extra_lines:
        for (p1, p2, color) in extra_lines:
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                    color=color, linewidth=2)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.view_init(elev=view_elev, azim=view_azim)
    if title:
        ax.set_title(title, fontsize=10)
    return ax


def viz_pointcloud_views(pcd_path, output_dir, prefix="pc",
                         title_prefix="", highlight_pts=None,
                         highlight_label=None, extra_lines=None):
    """Render a point cloud from 3 standard views (front/side/top)."""
    import open3d as o3d
    if not os.path.exists(pcd_path):
        return None
    os.makedirs(output_dir, exist_ok=True)

    pcd = o3d.io.read_point_cloud(pcd_path)
    points = np.asarray(pcd.points)
    if len(points) == 0:
        return None
    colors = np.asarray(pcd.colors) if len(pcd.colors) else None

    views = [
        ("front", 10, -90, "Front (toward camera)"),
        ("side", 10, 0, "Side (X-axis)"),
        ("top", 89, -90, "Top-down"),
    ]
    saved = []
    fig = plt.figure(figsize=(15, 5))
    for i, (name, elev, azim, label) in enumerate(views, 1):
        ax = fig.add_subplot(1, 3, i, projection="3d")
        _render_pcd_matplotlib(
            points, colors, ax=ax,
            title=f"{title_prefix}{label}",
            highlight_pts=highlight_pts,
            highlight_label=highlight_label,
            extra_lines=extra_lines,
            view_elev=elev, view_azim=azim,
        )
    fig.suptitle(f"{title_prefix}{os.path.basename(pcd_path)} — {len(points)} points",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    out = os.path.join(output_dir, f"{prefix}_views.jpg")
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    saved.append(out)
    return saved


# ─── Stage: Before/after outlier removal comparison ────────────────

def viz_pointcloud_compare(before_path, after_path, output_dir, label_before="raw", label_after="cleaned"):
    """Side-by-side render of two point clouds (e.g., before vs. after outlier removal)."""
    import open3d as o3d
    if not (os.path.exists(before_path) and os.path.exists(after_path)):
        return None
    os.makedirs(output_dir, exist_ok=True)
    pcd_b = o3d.io.read_point_cloud(before_path)
    pcd_a = o3d.io.read_point_cloud(after_path)
    pts_b = np.asarray(pcd_b.points)
    pts_a = np.asarray(pcd_a.points)
    col_b = np.asarray(pcd_b.colors) if len(pcd_b.colors) else None
    col_a = np.asarray(pcd_a.colors) if len(pcd_a.colors) else None

    fig = plt.figure(figsize=(12, 5))
    for i, (pts, cols, label) in enumerate([(pts_b, col_b, label_before),
                                              (pts_a, col_a, label_after)], 1):
        ax = fig.add_subplot(1, 2, i, projection="3d")
        _render_pcd_matplotlib(pts, cols, ax=ax,
                               title=f"{label} — {len(pts)} points",
                               view_elev=10, view_azim=-90)
    plt.tight_layout()
    out = os.path.join(output_dir, "pointcloud_compare.jpg")
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out


# ─── Stage: Belly mesh + back plane + protrusion direction ─────────

def viz_belly_mesh_with_plane(mesh_path, plane_point, plane_normal,
                               output_dir, prefix="belly_mesh"):
    """Render the belly mesh from multiple angles with the fitted back plane
    and protrusion direction visible."""
    import open3d as o3d
    if not os.path.exists(mesh_path):
        return None
    os.makedirs(output_dir, exist_ok=True)

    mesh = o3d.io.read_triangle_mesh(mesh_path)
    V = np.asarray(mesh.vertices)
    F = np.asarray(mesh.triangles)
    if len(V) == 0:
        return None

    plane_point = np.asarray(plane_point)
    plane_normal = np.asarray(plane_normal)
    plane_normal = plane_normal / (np.linalg.norm(plane_normal) + 1e-12)

    # Compute height-above-plane field for color-coding
    heights = (V - plane_point) @ plane_normal
    h_norm = (heights - heights.min()) / (heights.max() - heights.min() + 1e-12)

    # Build a small grid of plane points (for visualization)
    extent = np.linalg.norm(V - plane_point, axis=1).max()
    u = np.cross(plane_normal, [1.0, 0.0, 0.0])
    if np.linalg.norm(u) < 1e-6:
        u = np.cross(plane_normal, [0.0, 1.0, 0.0])
    u = u / (np.linalg.norm(u) + 1e-12)
    v = np.cross(plane_normal, u)
    plane_grid = []
    for du in np.linspace(-extent, extent, 6):
        for dv in np.linspace(-extent, extent, 6):
            plane_grid.append(plane_point + du * u + dv * v)
    plane_grid = np.array(plane_grid)

    # Sub-sample faces (5000 max) for render speed
    n_face_max = 5000
    if len(F) > n_face_max:
        sub = np.random.RandomState(0).choice(len(F), n_face_max, replace=False)
    else:
        sub = np.arange(len(F))

    # Build face-color array once (height-above-plane → plasma colormap)
    from matplotlib import cm
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    cmap = cm.get_cmap("plasma")
    triangles = V[F[sub]]                  # (N_sub, 3, 3)
    avg_h_per_face = h_norm[F[sub]].mean(axis=1)  # (N_sub,)
    face_colors = cmap(avg_h_per_face)
    # Translucent
    face_colors[:, 3] = 0.7

    views = [("front", 10, -90), ("side", 10, 0), ("top", 89, -90)]
    fig = plt.figure(figsize=(15, 5))
    for i, (name, elev, azim) in enumerate(views, 1):
        ax = fig.add_subplot(1, 3, i, projection="3d")
        # Use Poly3DCollection — draws triangle faces directly. Avoids
        # per-face plot_trisurf calls that re-triangulate via qhull and
        # can fail with "singular input data" on degenerate (collinear)
        # triangle xy projections.
        coll = Poly3DCollection(triangles, facecolors=face_colors,
                                  edgecolors="none", linewidths=0)
        ax.add_collection3d(coll)
        # Set axis bounds so the mesh is actually visible
        ax.set_xlim(V[:, 0].min(), V[:, 0].max())
        ax.set_ylim(V[:, 1].min(), V[:, 1].max())
        ax.set_zlim(V[:, 2].min(), V[:, 2].max())
        # Plot fitted plane as scatter
        ax.scatter(plane_grid[:, 0], plane_grid[:, 1], plane_grid[:, 2],
                   c="cyan", s=8, alpha=0.6, label="back plane")
        # Plot protrusion direction
        center = V.mean(axis=0)
        tip = center + plane_normal * extent * 0.7
        ax.plot([center[0], tip[0]], [center[1], tip[1]], [center[2], tip[2]],
                color="lime", linewidth=3, label="protrusion dir")
        ax.scatter([tip[0]], [tip[1]], [tip[2]], c="lime", s=80, marker="^")
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(f"{name} — color = height above back plane")
        if i == 1:
            ax.legend(loc="upper left", fontsize=8)

    fig.suptitle(f"{prefix} | {len(V)} verts, {len(F)} faces", fontsize=11, y=1.02)
    plt.tight_layout()
    out = os.path.join(output_dir, f"{prefix}_with_plane.jpg")
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out


# ─── Stage: Belly button localization (gradient flow) ──────────────

def viz_belly_button(mesh_path, belly_button_3d, plane_point, plane_normal,
                     output_dir, prefix="belly_button",
                     belly_button_info=None):
    """Render the belly mesh with the belly button highlighted.

    Three panels: front / isometric / side views. Mesh is drawn with
    Poly3DCollection (face-based, qhull-safe) and colored by height above
    the back plane. The belly button is shown as a large red marker plus a
    visible "needle" showing protrusion direction.
    """
    import open3d as o3d
    if not os.path.exists(mesh_path) or belly_button_3d is None:
        return None
    os.makedirs(output_dir, exist_ok=True)

    mesh = o3d.io.read_triangle_mesh(mesh_path)
    V = np.asarray(mesh.vertices)
    F = np.asarray(mesh.triangles)
    if len(V) == 0 or len(F) == 0:
        return None

    bb = np.asarray(belly_button_3d)
    plane_point = np.asarray(plane_point)
    plane_normal = np.asarray(plane_normal)
    plane_normal = plane_normal / (np.linalg.norm(plane_normal) + 1e-12)
    heights = (V - plane_point) @ plane_normal
    h_norm = (heights - heights.min()) / (heights.max() - heights.min() + 1e-12)

    # Sub-sample faces for render speed
    n_face_max = 5000
    if len(F) > n_face_max:
        sub = np.random.RandomState(0).choice(len(F), n_face_max, replace=False)
    else:
        sub = np.arange(len(F))

    from matplotlib import cm
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    cmap = cm.get_cmap("plasma")
    triangles = V[F[sub]]
    avg_h_per_face = h_norm[F[sub]].mean(axis=1)
    face_colors = cmap(avg_h_per_face)
    face_colors[:, 3] = 0.8  # mostly opaque

    # Compute a "needle" along the protrusion direction at the belly-button
    extent = np.linalg.norm(V - V.mean(axis=0), axis=1).max()
    needle_back = bb - plane_normal * extent * 0.20
    needle_fwd  = bb + plane_normal * extent * 0.20

    fig = plt.figure(figsize=(15, 5))
    views = [("front", 10, -90), ("isometric", 25, -45), ("side", 10, 0)]
    for i, (name, elev, azim) in enumerate(views, 1):
        ax = fig.add_subplot(1, 3, i, projection="3d")
        coll = Poly3DCollection(triangles, facecolors=face_colors,
                                  edgecolors="none", linewidths=0)
        ax.add_collection3d(coll)
        ax.set_xlim(V[:, 0].min(), V[:, 0].max())
        ax.set_ylim(V[:, 1].min(), V[:, 1].max())
        ax.set_zlim(V[:, 2].min(), V[:, 2].max())
        # Protrusion-direction "needle" through the belly button
        ax.plot([needle_back[0], needle_fwd[0]],
                [needle_back[1], needle_fwd[1]],
                [needle_back[2], needle_fwd[2]],
                color="lime", linewidth=2.5, alpha=0.9, label="protrusion dir")
        # Belly button marker (large)
        ax.scatter([bb[0]], [bb[1]], [bb[2]], c="red", s=350,
                    edgecolors="white", linewidth=2, marker="o",
                    label="Belly button", zorder=10)
        ax.scatter([bb[0]], [bb[1]], [bb[2]], c="red", s=80, marker="x", zorder=11)
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(name)
        if i == 1:
            ax.legend(loc="upper left", fontsize=8)

    bb_cm = (bb * 100).round(1).tolist()
    title = f"Belly button at {bb_cm} cm"
    if belly_button_info:
        proto_h = belly_button_info.get("protrusion_height_cm")
        conv_frac = belly_button_info.get("convergence_fraction")
        method = belly_button_info.get("method", "?")
        n_lm = belly_button_info.get("n_local_maxima")
        bits = [f"method={method}"]
        if proto_h is not None:
            bits.append(f"max_protrusion={proto_h:.1f}cm")
        if conv_frac is not None:
            bits.append(f"flow_converged={conv_frac*100:.0f}% of verts")
        if n_lm is not None:
            bits.append(f"n_local_maxima={n_lm}")
        title += "  |  " + ", ".join(bits)

    fig.suptitle(title, fontsize=11, y=1.02)
    plt.tight_layout()
    out = os.path.join(output_dir, f"{prefix}.jpg")
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out


def viz_belly_button_height_field(mesh_path, belly_button_3d, plane_point,
                                    plane_normal, output_dir,
                                    prefix="belly_button_height"):
    """2D top-down 'topographic' map: project mesh vertices onto the back
    plane and color by height above the plane. The belly button stands out
    as a clear hot-spot. Easy to verify visually."""
    import open3d as o3d
    if not os.path.exists(mesh_path) or belly_button_3d is None:
        return None
    os.makedirs(output_dir, exist_ok=True)

    mesh = o3d.io.read_triangle_mesh(mesh_path)
    V = np.asarray(mesh.vertices)
    if len(V) == 0:
        return None

    bb = np.asarray(belly_button_3d)
    plane_point = np.asarray(plane_point)
    plane_normal = np.asarray(plane_normal)
    plane_normal = plane_normal / (np.linalg.norm(plane_normal) + 1e-12)

    # Build orthonormal basis (u, v) on the back plane
    u = np.cross(plane_normal, [1.0, 0.0, 0.0])
    if np.linalg.norm(u) < 1e-6:
        u = np.cross(plane_normal, [0.0, 1.0, 0.0])
    u = u / (np.linalg.norm(u) + 1e-12)
    v = np.cross(plane_normal, u)

    # Project vertices onto the plane (in plane coordinates)
    rel = V - plane_point
    u_coord = rel @ u
    v_coord = rel @ v
    h_coord = rel @ plane_normal  # height above plane

    # Project belly button onto the plane (its "shadow")
    bb_rel = bb - plane_point
    bb_u = bb_rel @ u
    bb_v = bb_rel @ v
    bb_h = bb_rel @ plane_normal

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # LEFT: scatter colored by height (heatmap)
    sc = axes[0].scatter(u_coord * 100, v_coord * 100, c=h_coord * 100,
                          cmap="plasma", s=4, alpha=0.7)
    plt.colorbar(sc, ax=axes[0], label="Height above back plane (cm)")
    axes[0].scatter([bb_u * 100], [bb_v * 100], s=400,
                     facecolors="red", edgecolors="white",
                     linewidth=2.5, marker="o",
                     label=f"Belly button (h={bb_h*100:.1f}cm)", zorder=10)
    axes[0].scatter([bb_u * 100], [bb_v * 100], s=120, c="red",
                     marker="x", zorder=11)
    axes[0].set_xlabel("u-axis on back plane (cm)")
    axes[0].set_ylabel("v-axis on back plane (cm)")
    axes[0].set_title("Top-down: vertex projection colored by height")
    axes[0].set_aspect("equal")
    axes[0].legend(loc="upper right", fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # RIGHT: contour plot of height (topographic)
    # Bin into a 2D grid
    try:
        from scipy.interpolate import griddata
        u_min, u_max = u_coord.min(), u_coord.max()
        v_min, v_max = v_coord.min(), v_coord.max()
        nu, nv = 80, 80
        ui = np.linspace(u_min, u_max, nu)
        vi = np.linspace(v_min, v_max, nv)
        UI, VI = np.meshgrid(ui, vi)
        HI = griddata((u_coord, v_coord), h_coord, (UI, VI), method="linear",
                       fill_value=h_coord.min())
        cf = axes[1].contourf(UI * 100, VI * 100, HI * 100,
                                levels=20, cmap="plasma")
        axes[1].contour(UI * 100, VI * 100, HI * 100, levels=10,
                         colors="black", alpha=0.3, linewidths=0.5)
        plt.colorbar(cf, ax=axes[1], label="Height above back plane (cm)")
    except Exception as e:
        axes[1].text(0.5, 0.5, f"contour failed: {e}",
                      transform=axes[1].transAxes, ha="center", va="center")
    axes[1].scatter([bb_u * 100], [bb_v * 100], s=400,
                     facecolors="red", edgecolors="white",
                     linewidth=2.5, marker="o", zorder=10)
    axes[1].scatter([bb_u * 100], [bb_v * 100], s=120, c="red",
                     marker="x", zorder=11)
    axes[1].set_xlabel("u-axis on back plane (cm)")
    axes[1].set_ylabel("v-axis on back plane (cm)")
    axes[1].set_title("Topographic contours")
    axes[1].set_aspect("equal")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(f"Belly button localization — height field on back plane "
                 f"(max height = {h_coord.max()*100:.1f}cm)", fontsize=11)
    plt.tight_layout()
    out = os.path.join(output_dir, f"{prefix}.jpg")
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out


def viz_belly_button_gradient_flow(mesh_path, belly_button_3d, plane_point,
                                     plane_normal, output_dir,
                                     prefix="belly_button_flow",
                                     n_arrows=200, max_steps=200):
    """Visualize where gradient flow converges, using arrows from random
    starting vertices toward the local maximum each one walks to.

    Algorithm matches find_belly_button() but instead of just reporting the
    mode, we draw an arrow from each starting vertex to its convergence
    point, plus a 2D top-down panel showing the basin of attraction.
    """
    import open3d as o3d
    if not os.path.exists(mesh_path) or belly_button_3d is None:
        return None
    os.makedirs(output_dir, exist_ok=True)

    mesh = o3d.io.read_triangle_mesh(mesh_path)
    V = np.asarray(mesh.vertices)
    F = np.asarray(mesh.triangles)
    n = len(V)
    if n == 0 or len(F) == 0:
        return None

    bb = np.asarray(belly_button_3d)
    plane_point = np.asarray(plane_point)
    plane_normal = np.asarray(plane_normal)
    plane_normal = plane_normal / (np.linalg.norm(plane_normal) + 1e-12)
    h = (V - plane_point) @ plane_normal

    # Build adjacency
    from collections import defaultdict
    adj = defaultdict(set)
    for tri in F:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        adj[a].add(b); adj[a].add(c)
        adj[b].add(a); adj[b].add(c)
        adj[c].add(a); adj[c].add(b)
    adj = {k: list(v) for k, v in adj.items()}

    # Compute full convergence map (vertex → which local-max it flows to)
    convergence = np.full(n, -1, dtype=int)
    for start in range(n):
        cur = start
        for _ in range(max_steps):
            neighbors = adj.get(cur, [])
            if not neighbors:
                break
            best = max(neighbors, key=lambda x: h[x])
            if h[best] <= h[cur]:
                break
            cur = best
        convergence[start] = cur

    # Stats: most popular local max
    unique, counts = np.unique(convergence[convergence >= 0], return_counts=True)
    mode_idx = int(unique[np.argmax(counts)])
    mode_count = int(counts.max())
    mode_frac = mode_count / float(n)
    n_local_maxima = int(len(unique))

    # Build orthonormal basis on back plane for top-down view
    u = np.cross(plane_normal, [1.0, 0.0, 0.0])
    if np.linalg.norm(u) < 1e-6:
        u = np.cross(plane_normal, [0.0, 1.0, 0.0])
    u = u / (np.linalg.norm(u) + 1e-12)
    v = np.cross(plane_normal, u)
    rel = V - plane_point
    U_v = rel @ u
    V_v = rel @ v

    # Sample starting vertices uniformly
    rng = np.random.RandomState(0)
    n_arrows = min(n_arrows, n)
    starts = rng.choice(n, n_arrows, replace=False)

    fig = plt.figure(figsize=(15, 5.5))

    # ── PANEL 1: 3D mesh + arrows from start → local max ──────────────
    from matplotlib import cm
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    cmap = cm.get_cmap("plasma")
    h_norm = (h - h.min()) / (h.max() - h.min() + 1e-12)

    ax = fig.add_subplot(1, 3, 1, projection="3d")
    # Sub-sample faces for render speed
    n_face_max = 3000
    if len(F) > n_face_max:
        sub = rng.choice(len(F), n_face_max, replace=False)
    else:
        sub = np.arange(len(F))
    triangles = V[F[sub]]
    avg_h_per_face = h_norm[F[sub]].mean(axis=1)
    face_colors = cmap(avg_h_per_face)
    face_colors[:, 3] = 0.45  # semi-transparent so arrows show through
    coll = Poly3DCollection(triangles, facecolors=face_colors,
                              edgecolors="none", linewidths=0)
    ax.add_collection3d(coll)
    ax.set_xlim(V[:, 0].min(), V[:, 0].max())
    ax.set_ylim(V[:, 1].min(), V[:, 1].max())
    ax.set_zlim(V[:, 2].min(), V[:, 2].max())
    # Draw arrows: from each start, a single straight arrow to its convergence point
    converged_to_main = 0
    for s in starts:
        end_idx = convergence[s]
        if end_idx < 0:
            continue
        if end_idx == mode_idx:
            converged_to_main += 1
            color = "#00cc44"  # green = goes to main apex
            alpha = 0.55
        else:
            color = "#ff6644"  # red = goes to a different local max
            alpha = 0.35
        ax.plot([V[s, 0], V[end_idx, 0]],
                [V[s, 1], V[end_idx, 1]],
                [V[s, 2], V[end_idx, 2]],
                color=color, linewidth=0.8, alpha=alpha)
    # Belly button + main apex
    ax.scatter([V[mode_idx, 0]], [V[mode_idx, 1]], [V[mode_idx, 2]],
                c="red", s=300, edgecolors="white", linewidth=2, marker="*",
                label="convergence apex", zorder=10)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.view_init(elev=20, azim=-60)
    ax.set_title(f"3D flow: {n_arrows} starts → apex\n"
                 f"green = converges to main apex, red = other local max")

    # ── PANEL 2: 2D top-down arrows (matplotlib quiver-style) ─────────
    ax2 = fig.add_subplot(1, 3, 2)
    end_pts_u = U_v[convergence[starts]]
    end_pts_v = V_v[convergence[starts]]
    start_u = U_v[starts]
    start_v = V_v[starts]
    dx = (end_pts_u - start_u) * 100
    dy = (end_pts_v - start_v) * 100
    main_mask = convergence[starts] == mode_idx
    other_mask = ~main_mask
    # Arrows that flow to the main apex (green)
    if main_mask.any():
        ax2.quiver(start_u[main_mask] * 100, start_v[main_mask] * 100,
                    dx[main_mask], dy[main_mask],
                    angles="xy", scale_units="xy", scale=1.0,
                    color="#00aa33", alpha=0.55, width=0.003,
                    label=f"→ main apex ({main_mask.sum()})")
    # Arrows that flow to another local max (red)
    if other_mask.any():
        ax2.quiver(start_u[other_mask] * 100, start_v[other_mask] * 100,
                    dx[other_mask], dy[other_mask],
                    angles="xy", scale_units="xy", scale=1.0,
                    color="#cc3322", alpha=0.55, width=0.003,
                    label=f"→ other local max ({other_mask.sum()})")
    # All local maxima marked
    for lm_idx, lm_count in zip(unique, counts):
        is_main = lm_idx == mode_idx
        ax2.scatter([U_v[lm_idx] * 100], [V_v[lm_idx] * 100],
                     s=280 if is_main else 100,
                     facecolors="red" if is_main else "orange",
                     edgecolors="white", linewidth=2,
                     marker="*" if is_main else "o", zorder=10,
                     label=("apex (main)" if is_main else None))
        ax2.annotate(f"{lm_count}",
                      (U_v[lm_idx] * 100, V_v[lm_idx] * 100),
                      xytext=(6, 6), textcoords="offset points",
                      fontsize=8, fontweight="bold",
                      color="white",
                      bbox=dict(boxstyle="round,pad=0.2",
                                  fc="black", alpha=0.7))
    ax2.set_aspect("equal")
    ax2.set_xlabel("u (cm)")
    ax2.set_ylabel("v (cm)")
    ax2.set_title(f"Top-down: arrows from start → local max\n"
                  f"{n_local_maxima} local maxima found, "
                  f"{mode_frac*100:.0f}% converge to main")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, alpha=0.3)

    # ── PANEL 3: Basin of attraction (every vertex colored by where it lands) ──
    ax3 = fig.add_subplot(1, 3, 3)
    # Color each vertex by its convergence target (each unique target = unique color)
    target_to_color = {}
    color_palette = plt.get_cmap("tab10").colors + plt.get_cmap("Set3").colors
    for i, tgt in enumerate(unique):
        if tgt == mode_idx:
            target_to_color[int(tgt)] = (1.0, 0.0, 0.0, 0.7)  # red for main
        else:
            target_to_color[int(tgt)] = (*color_palette[i % len(color_palette)], 0.6)
    vertex_colors = np.array([target_to_color.get(int(c), (0.5, 0.5, 0.5, 0.3))
                               for c in convergence])
    # Sub-sample for speed
    n_max = 6000
    if n > n_max:
        psub = rng.choice(n, n_max, replace=False)
    else:
        psub = np.arange(n)
    ax3.scatter(U_v[psub] * 100, V_v[psub] * 100,
                 c=vertex_colors[psub], s=4)
    # Mark main apex
    ax3.scatter([U_v[mode_idx] * 100], [V_v[mode_idx] * 100],
                 s=350, marker="*", c="red", edgecolors="white", linewidth=2,
                 zorder=10, label="main apex")
    ax3.set_aspect("equal")
    ax3.set_xlabel("u (cm)")
    ax3.set_ylabel("v (cm)")
    ax3.set_title(f"Basin of attraction\n"
                  f"red = vertices converging to main apex")
    ax3.legend(loc="upper right", fontsize=9)
    ax3.grid(True, alpha=0.3)

    fig.suptitle(
        f"Gradient flow convergence — "
        f"{mode_count}/{n} verts ({mode_frac*100:.1f}%) → main apex, "
        f"{n_local_maxima} local maxima total. "
        f"Higher % = more confident belly-button localization.",
        fontsize=10,
    )
    plt.tight_layout()
    out = os.path.join(output_dir, f"{prefix}.jpg")
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out


def viz_belly_button_cross_section(mesh_path, belly_button_3d, plane_point,
                                     plane_normal, output_dir,
                                     prefix="belly_button_section"):
    """Cross-section profile: slice the belly mesh through the belly-button
    along two orthogonal planes and plot the silhouette. Lets you see the
    bump shape and how far the belly button protrudes."""
    import open3d as o3d
    if not os.path.exists(mesh_path) or belly_button_3d is None:
        return None
    os.makedirs(output_dir, exist_ok=True)

    mesh = o3d.io.read_triangle_mesh(mesh_path)
    V = np.asarray(mesh.vertices)
    if len(V) == 0:
        return None

    bb = np.asarray(belly_button_3d)
    plane_point = np.asarray(plane_point)
    plane_normal = np.asarray(plane_normal)
    plane_normal = plane_normal / (np.linalg.norm(plane_normal) + 1e-12)

    # Build basis (u, v) on back plane, w = protrusion normal
    u = np.cross(plane_normal, [1.0, 0.0, 0.0])
    if np.linalg.norm(u) < 1e-6:
        u = np.cross(plane_normal, [0.0, 1.0, 0.0])
    u = u / (np.linalg.norm(u) + 1e-12)
    v = np.cross(plane_normal, u)

    rel = V - plane_point
    U_v = rel @ u
    V_v = rel @ v
    H_v = rel @ plane_normal

    bb_rel = bb - plane_point
    bb_u = bb_rel @ u
    bb_v = bb_rel @ v
    bb_h = bb_rel @ plane_normal

    # Take vertices within a slice of width 2cm centered on the belly button
    slice_thickness_m = 0.02

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Horizontal slice: V ≈ bb_v
    mask_h = np.abs(V_v - bb_v) < slice_thickness_m
    if mask_h.sum() > 5:
        order = np.argsort(U_v[mask_h])
        axes[0].plot(U_v[mask_h][order] * 100,
                      H_v[mask_h][order] * 100, "b.", markersize=3)
    axes[0].axvline(bb_u * 100, color="red", linestyle="--", alpha=0.7,
                     label=f"belly button (u={bb_u*100:.1f}cm, h={bb_h*100:.1f}cm)")
    axes[0].axhline(bb_h * 100, color="red", linestyle="--", alpha=0.4)
    axes[0].axhline(0, color="cyan", linestyle="-", alpha=0.5, label="back plane")
    axes[0].set_xlabel("u (cm)  — left ↔ right")
    axes[0].set_ylabel("Height above back plane (cm)")
    axes[0].set_title(f"Horizontal slice through belly button (±{slice_thickness_m*100:.0f}cm v-band)")
    axes[0].legend(loc="upper right", fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # Vertical slice: U ≈ bb_u
    mask_v = np.abs(U_v - bb_u) < slice_thickness_m
    if mask_v.sum() > 5:
        order = np.argsort(V_v[mask_v])
        axes[1].plot(V_v[mask_v][order] * 100,
                      H_v[mask_v][order] * 100, "g.", markersize=3)
    axes[1].axvline(bb_v * 100, color="red", linestyle="--", alpha=0.7,
                     label=f"belly button (v={bb_v*100:.1f}cm)")
    axes[1].axhline(bb_h * 100, color="red", linestyle="--", alpha=0.4)
    axes[1].axhline(0, color="cyan", linestyle="-", alpha=0.5, label="back plane")
    axes[1].set_xlabel("v (cm)  — top ↔ bottom")
    axes[1].set_ylabel("Height above back plane (cm)")
    axes[1].set_title(f"Vertical slice through belly button (±{slice_thickness_m*100:.0f}cm u-band)")
    axes[1].legend(loc="upper right", fontsize=9)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Belly cross-section profiles through the belly button",
                 fontsize=11)
    plt.tight_layout()
    out = os.path.join(output_dir, f"{prefix}.jpg")
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out


# ─── Stage: Belly button overlay on original images ────────────────

def _project_world_to_recon_pixel(point_3d, extrinsic, intrinsic):
    """Project a 3D world point to VGGT/AMB3R image-canvas pixel coords.

    Args:
        point_3d: (3,) world point.
        extrinsic: (3, 4) [R | t] world→camera transform.
        intrinsic: (3, 3) camera intrinsic matrix in canvas pixel coords.

    Returns:
        (px, py, depth) — pixel coords in the canvas + depth in camera frame.
        depth ≤ 0 means the point is behind the camera (not visible).
    """
    p_world = np.asarray(point_3d, dtype=float).reshape(3)
    R = np.asarray(extrinsic)[:3, :3]
    t = np.asarray(extrinsic)[:3, 3]
    p_cam = R @ p_world + t
    if p_cam[2] <= 1e-6:
        return None, None, p_cam[2]
    K = np.asarray(intrinsic)
    p_pix = K @ p_cam
    px = float(p_pix[0] / p_pix[2])
    py = float(p_pix[1] / p_pix[2])
    return px, py, float(p_cam[2])


def _recon_pixel_to_orig_pixel(canvas_xy, transform):
    """Invert the VGGT/AMB3R preprocess transform to map canvas pixels back
    to the original-image pixel coords."""
    cx, cy = canvas_xy
    if transform is None:
        return cx, cy
    scale = transform.get("scale", 1.0)
    if transform.get("mode") == "pad":
        ox = (cx - transform.get("pad_left", 0)) / scale
        oy = (cy - transform.get("pad_top", 0)) / scale
    else:  # crop
        ox = (cx + transform.get("crop_left", 0)) / scale
        oy = (cy + transform.get("crop_top", 0)) / scale
    return ox, oy


def viz_belly_button_overlays(amb3r_npz_path, recon_meta_path, image_dir,
                                belly_button_3d, output_dir,
                                prefix="belly_button_overlay"):
    """For each reconstruction frame, project the 3D belly button back into the
    ORIGINAL input image coordinates and overlay a marker on it.

    Lets you visually verify "where is the predicted belly button on each
    real-world frame" — most direct sanity check there is.

    Saves:
        <output_dir>/<prefix>/frame_NNN.jpg       — full-resolution overlay per frame
        <output_dir>/<prefix>_grid.jpg            — multi-frame thumbnail grid
    """
    if not (os.path.exists(amb3r_npz_path)
            and os.path.exists(recon_meta_path)
            and image_dir and belly_button_3d is not None):
        return None
    os.makedirs(output_dir, exist_ok=True)
    per_frame_dir = os.path.join(output_dir, prefix)
    os.makedirs(per_frame_dir, exist_ok=True)

    npz = np.load(amb3r_npz_path, allow_pickle=True)
    if "extrinsic" not in npz or "intrinsic" not in npz:
        print("  (no extrinsic/intrinsic in NPZ — cannot project)")
        return None
    extrinsics = npz["extrinsic"]   # (T, 3, 4)
    intrinsics = npz["intrinsic"]   # (T, 3, 3)
    canvas_imgs = npz["images_per_frame"]   # (T, H, W, 3) — what VGGT saw

    with open(recon_meta_path) as f:
        recon_meta = json.load(f)
    transforms = {t["filename"]: t for t in recon_meta.get("preprocess_transforms", [])}
    image_files_in_order = recon_meta.get("image_files_in_order", [])
    if not image_files_in_order:
        # Fall back to canvas-image-only overlays
        image_files_in_order = [f"frame_{i:03d}.jpg"
                                  for i in range(len(extrinsics))]

    bb = np.asarray(belly_button_3d, dtype=float)
    saved_paths = []
    grid_thumbs = []

    for frame_idx, fname in enumerate(image_files_in_order):
        if frame_idx >= len(extrinsics):
            break
        # Project to canvas-pixel coords
        canvas_xy_d = _project_world_to_recon_pixel(
            bb, extrinsics[frame_idx], intrinsics[frame_idx],
        )
        canvas_px, canvas_py, depth = canvas_xy_d
        canvas_h, canvas_w = canvas_imgs.shape[1], canvas_imgs.shape[2]
        in_canvas = (canvas_px is not None
                     and 0 <= canvas_px < canvas_w
                     and 0 <= canvas_py < canvas_h
                     and depth > 0)

        # Map back to original image pixels
        orig_path = os.path.join(image_dir, fname)
        if not os.path.exists(orig_path):
            # try recon_frames as a sibling fallback
            orig_path = None

        # Try to load the original image
        if orig_path and os.path.exists(orig_path):
            img = cv2.imread(orig_path)
            if img is None:
                continue
            transform = transforms.get(fname)
            orig_px, orig_py = _recon_pixel_to_orig_pixel(
                (canvas_px, canvas_py) if canvas_px is not None else (0, 0),
                transform,
            ) if canvas_px is not None else (None, None)
            display_img = img
            display_px, display_py = orig_px, orig_py
            in_image = (orig_px is not None
                        and 0 <= orig_px < img.shape[1]
                        and 0 <= orig_py < img.shape[0]
                        and depth > 0)
        else:
            # Fall back to drawing on the canvas image directly
            canvas_uint8 = (canvas_imgs[frame_idx] * 255).clip(0, 255).astype(np.uint8)
            display_img = cv2.cvtColor(canvas_uint8, cv2.COLOR_RGB2BGR)
            display_px, display_py = canvas_px, canvas_py
            in_image = in_canvas

        h_img, w_img = display_img.shape[:2]

        # Draw marker
        if in_image:
            cx_int, cy_int = int(round(display_px)), int(round(display_py))
            r = max(20, int(0.02 * min(w_img, h_img)))   # marker radius
            # Outer black ring
            cv2.circle(display_img, (cx_int, cy_int), r + 8, (0, 0, 0), 6)
            # Inner red filled
            cv2.circle(display_img, (cx_int, cy_int), r, (0, 0, 255), -1)
            # White center dot
            cv2.circle(display_img, (cx_int, cy_int), max(3, r // 5),
                        (255, 255, 255), -1)
            # Crosshair lines extending across the frame
            cv2.line(display_img, (0, cy_int), (w_img, cy_int),
                      (0, 0, 255), 2, cv2.LINE_AA)
            cv2.line(display_img, (cx_int, 0), (cx_int, h_img),
                      (0, 0, 255), 2, cv2.LINE_AA)
            label = (f"BELLY BUTTON  pixel=({cx_int},{cy_int})  "
                     f"depth={depth*100:.1f}cm  3D={(bb*100).round(1).tolist()}cm")
            color_box = (0, 0, 0)
            color_text = (255, 255, 255)
        else:
            label = (f"NOT VISIBLE in this frame  "
                     f"(canvas px={canvas_px}, depth={depth:.3f}m)")
            color_box = (0, 0, 80)
            color_text = (255, 255, 255)

        # Header band
        bar_h = max(60, h_img // 30)
        cv2.rectangle(display_img, (0, 0), (w_img, bar_h), color_box, -1)
        font_scale = max(0.7, w_img / 2400)
        cv2.putText(display_img, f"{fname}   {label}", (12, int(bar_h * 0.7)),
                     cv2.FONT_HERSHEY_SIMPLEX, font_scale, color_text, 2)

        # Save full-resolution overlay
        out_path = os.path.join(per_frame_dir, f"{os.path.splitext(fname)[0]}_overlay.jpg")
        cv2.imwrite(out_path, display_img)
        saved_paths.append(out_path)

        # Thumbnail for the grid
        thumb_w = 480
        thumb_h = int(h_img * thumb_w / w_img)
        thumb = cv2.resize(display_img, (thumb_w, thumb_h),
                            interpolation=cv2.INTER_AREA)
        grid_thumbs.append((fname, thumb, in_image))

    # Build a grid summary image
    if grid_thumbs:
        n = len(grid_thumbs)
        cols = min(5, n)
        rows = (n + cols - 1) // cols
        thumb_h = grid_thumbs[0][1].shape[0]
        thumb_w = grid_thumbs[0][1].shape[1]
        canvas = np.full((rows * thumb_h, cols * thumb_w, 3), 30, dtype=np.uint8)
        for i, (fname, thumb, in_image) in enumerate(grid_thumbs):
            r, c = i // cols, i % cols
            canvas[r * thumb_h:(r + 1) * thumb_h,
                   c * thumb_w:(c + 1) * thumb_w] = thumb
            # red border if not visible
            if not in_image:
                cv2.rectangle(canvas, (c * thumb_w + 2, r * thumb_h + 2),
                                ((c + 1) * thumb_w - 2, (r + 1) * thumb_h - 2),
                                (0, 0, 200), 4)
        grid_path = os.path.join(output_dir, f"{prefix}_grid.jpg")
        cv2.imwrite(grid_path, canvas)
        saved_paths.append(grid_path)

    return saved_paths


# ─── Stage: Belly + feet + distances scene ─────────────────────────

def viz_belly_scene(reconstruction_pcd_path, belly_button_3d, feet_info,
                    distances, output_dir, prefix="belly_scene"):
    """Render the full scene: full point cloud + belly button + ankles +
    distance lines from belly button to feet/ground."""
    import open3d as o3d
    if not os.path.exists(reconstruction_pcd_path):
        return None
    os.makedirs(output_dir, exist_ok=True)

    pcd = o3d.io.read_point_cloud(reconstruction_pcd_path)
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) if len(pcd.colors) else None
    if len(points) == 0:
        return None

    bb = np.asarray(belly_button_3d) if belly_button_3d is not None else None
    extra_lines = []
    highlights = []
    labels = []
    if bb is not None:
        highlights.append(bb)
        labels.append("Belly button")
    if feet_info:
        l = np.asarray(feet_info.get("left_ankle_3d", []))
        r = np.asarray(feet_info.get("right_ankle_3d", []))
        mid = np.asarray(feet_info.get("midfoot_3d", []))
        if len(l) == 3:
            highlights.append(l); labels.append("L ankle")
        if len(r) == 3:
            highlights.append(r); labels.append("R ankle")
        if bb is not None and len(mid) == 3:
            extra_lines.append((bb, mid, "yellow"))

    highlights = np.array(highlights) if highlights else None

    fig = plt.figure(figsize=(15, 5))
    views = [("front", 10, -90), ("side", 10, 0), ("isometric", 25, -45)]
    for i, (name, elev, azim) in enumerate(views, 1):
        ax = fig.add_subplot(1, 3, i, projection="3d")
        _render_pcd_matplotlib(
            points, colors, ax=ax,
            title=name, highlight_pts=highlights,
            highlight_color="red",
            extra_lines=extra_lines,
            view_elev=elev, view_azim=azim,
        )

    title = "Belly scene"
    if distances:
        d_feet = distances.get("distance_belly_to_midfeet_cm")
        if d_feet is not None:
            title += f" — to midfoot: {d_feet:.1f} cm"

    fig.suptitle(title, fontsize=11, y=1.02)
    plt.tight_layout()
    out = os.path.join(output_dir, f"{prefix}.jpg")
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out


# ─── Stage: belly point-cloud confidence histogram + spatial extent ──

def viz_belly_pointcloud_stats(belly_pcd_path, output_dir, prefix="belly_pc"):
    """Plot histograms of X/Y/Z extent and a 2D top-down outline of the belly cloud
    so you can sanity-check that the belly was correctly isolated."""
    import open3d as o3d
    if not os.path.exists(belly_pcd_path):
        return None
    os.makedirs(output_dir, exist_ok=True)
    pcd = o3d.io.read_point_cloud(belly_pcd_path)
    points = np.asarray(pcd.points)
    if len(points) == 0:
        return None

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    for i, (label, idx) in enumerate(zip(["X (cm)", "Y (cm)", "Z (cm)"], [0, 1, 2])):
        axes[i].hist(points[:, idx] * 100, bins=50, color="steelblue", edgecolor="black")
        axes[i].set_title(f"{label} distribution")
        axes[i].set_xlabel(label); axes[i].set_ylabel("# points")
    # Top-down silhouette
    axes[3].scatter(points[:, 0] * 100, points[:, 2] * 100, s=0.5, c="darkgreen", alpha=0.4)
    axes[3].set_title("Top-down (X vs Z, in cm)")
    axes[3].set_xlabel("X (cm)"); axes[3].set_ylabel("Z (cm)")
    axes[3].set_aspect("equal")

    fig.suptitle(f"{prefix} | {len(points)} points  | "
                 f"span X={(points[:,0].max()-points[:,0].min())*100:.1f}cm, "
                 f"Y={(points[:,1].max()-points[:,1].min())*100:.1f}cm, "
                 f"Z={(points[:,2].max()-points[:,2].min())*100:.1f}cm",
                 fontsize=11, y=1.02)
    plt.tight_layout()
    out = os.path.join(output_dir, f"{prefix}_stats.jpg")
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out


# ─── Top-level: belly pipeline visualizations ──────────────────────

def run_belly_debug(output_dir, image_dir=None):
    """Generate all debug visualizations for the belly pipeline."""
    debug_dir = os.path.join(output_dir, "debug")
    os.makedirs(debug_dir, exist_ok=True)

    print("\n" + "="*60 + "\nGENERATING BELLY DEBUG VISUALIZATIONS\n" + "="*60)

    # 1. Scale calibration
    # Look in input dir for scale_calibration.json
    if image_dir:
        cal_path = os.path.join(image_dir, "scale_calibration.json")
        if os.path.exists(cal_path):
            print("\n[1] Scale calibration overlay:")
            saved = viz_scale_calibration(image_dir, cal_path,
                                           os.path.join(debug_dir, "scale"))
            if saved:
                print(f"  Saved {len(saved)} scale-overlay images")

    # 2. SAM3 belly segmentation
    seg_dir = os.path.join(output_dir, "segmentation")
    if os.path.isdir(seg_dir):
        print("\n[2] SAM3 belly segmentation overlay:")
        saved = viz_segmentation_overlay(image_dir, seg_dir,
                                         os.path.join(debug_dir, "segmentation"),
                                         mask_color=(255, 0, 200))
        if saved:
            print(f"  Saved {len(saved)} segmentation overlays")

    # 3. Reconstruction point cloud — multi-view
    recon_ply = os.path.join(output_dir, "reconstruction", "point_cloud.ply")
    if os.path.exists(recon_ply):
        print("\n[3] Reconstruction point cloud — multi-view:")
        viz_pointcloud_views(recon_ply, os.path.join(debug_dir, "reconstruction"),
                             prefix="full_pc", title_prefix="Full scene — ")

    # 4. Belly point cloud — stats + multi-view
    belly_pc = os.path.join(output_dir, "belly", "belly_pointcloud.ply")
    if os.path.exists(belly_pc):
        print("\n[4] Belly point cloud — stats + views:")
        viz_belly_pointcloud_stats(belly_pc, os.path.join(debug_dir, "belly"),
                                    prefix="belly_pointcloud")
        viz_pointcloud_views(belly_pc, os.path.join(debug_dir, "belly"),
                             prefix="belly_pc", title_prefix="Belly only — ")

    # 5/6/7. Belly mesh + plane + button + scene
    belly_results = os.path.join(output_dir, "belly", "belly_results.json")
    belly_mesh = os.path.join(output_dir, "belly", "belly_mesh.ply")
    if os.path.exists(belly_results) and os.path.exists(belly_mesh):
        with open(belly_results) as f:
            br = json.load(f)
        plane_point = br.get("plane_point")
        protrusion_dir = br.get("protrusion_direction")
        bb = br.get("belly_button", {}).get("position_3d")
        if plane_point and protrusion_dir:
            print("\n[5] Belly mesh + back plane + protrusion direction:")
            viz_belly_mesh_with_plane(belly_mesh, plane_point, protrusion_dir,
                                      os.path.join(debug_dir, "belly"))
        if bb and plane_point and protrusion_dir:
            bb_info = br.get("belly_button", {})
            print("\n[6a] Belly button — 3D mesh views:")
            viz_belly_button(belly_mesh, bb, plane_point, protrusion_dir,
                             os.path.join(debug_dir, "belly"),
                             belly_button_info=bb_info)
            print("\n[6b] Belly button — top-down height field + topographic contours:")
            try:
                viz_belly_button_height_field(
                    belly_mesh, bb, plane_point, protrusion_dir,
                    os.path.join(debug_dir, "belly"),
                )
            except Exception as e:
                print(f"     (height field viz failed: {e})")
            print("\n[6c] Belly button — orthogonal cross-section profiles:")
            try:
                viz_belly_button_cross_section(
                    belly_mesh, bb, plane_point, protrusion_dir,
                    os.path.join(debug_dir, "belly"),
                )
            except Exception as e:
                print(f"     (cross-section viz failed: {e})")
            print("\n[6d] Belly button — gradient-flow convergence (arrows):")
            try:
                viz_belly_button_gradient_flow(
                    belly_mesh, bb, plane_point, protrusion_dir,
                    os.path.join(debug_dir, "belly"),
                )
            except Exception as e:
                print(f"     (gradient flow viz failed: {e})")
            print("\n[6e] Belly button — overlay on every original input image:")
            try:
                npz_path = os.path.join(output_dir, "reconstruction", "point_cloud.npz")
                rmeta_path = os.path.join(output_dir, "reconstruction", "reconstruction_meta.json")
                viz_belly_button_overlays(
                    npz_path, rmeta_path, image_dir, bb,
                    os.path.join(debug_dir, "belly"),
                )
            except Exception as e:
                print(f"     (overlay viz failed: {e})")

        print("\n[7] Full belly scene (point cloud + belly button + feet):")
        feet_info = br.get("distances", {})
        # feet_info from belly.compute_distance_to_ground returns left/right/midfoot
        feet_for_scene = {
            "left_ankle_3d": feet_info.get("left_ankle_3d"),
            "right_ankle_3d": feet_info.get("right_ankle_3d"),
            "midfoot_3d": feet_info.get("midfoot_3d"),
        } if feet_info else None
        viz_belly_scene(recon_ply, bb, feet_for_scene, feet_info,
                        os.path.join(debug_dir, "belly"), prefix="belly_scene")

    print(f"\nAll belly debug outputs: {debug_dir}")
    return debug_dir


def viz_leg_assessment_card(leg_assessment_path, output_dir,
                              prefix="leg_assessment_card"):
    """Render a single 'assessment card' summarising the leg pipeline output.

    Shows: HKA angle + classification per leg, MAD, lengths, LLD, overall
    diagnosis. Visual quick-glance summary suitable for clinical review.
    """
    if not os.path.exists(leg_assessment_path):
        return None
    os.makedirs(output_dir, exist_ok=True)

    with open(leg_assessment_path) as f:
        a = json.load(f)

    L = a.get("left", {})
    R = a.get("right", {})
    metric = a.get("metric_calibrated", False)

    fig, ax = plt.subplots(figsize=(11, 8))
    ax.axis("off")

    # Header
    ax.text(0.5, 0.97, "Leg Deformity Assessment",
             fontsize=18, ha="center", va="top", weight="bold",
             transform=ax.transAxes)
    ax.text(0.5, 0.93,
             f"Overall: {a.get('overall_assessment', 'unknown')}",
             fontsize=14, ha="center", va="top",
             transform=ax.transAxes,
             bbox=dict(boxstyle="round,pad=0.4",
                          fc="#222", ec="#666", alpha=0.9), color="white")

    def render_leg(panel_x, leg_dict, title):
        y0 = 0.85
        ax.text(panel_x, y0, title, fontsize=14, weight="bold",
                 transform=ax.transAxes)
        if leg_dict.get("hka_angle_deg_median") is None:
            ax.text(panel_x, y0 - 0.04, "(insufficient frames)",
                     fontsize=11, color="#888", transform=ax.transAxes)
            return
        cls = leg_dict.get("classification", "?")
        sev = leg_dict.get("severity", "?")
        conf = leg_dict.get("confidence", "?")
        hka = leg_dict.get("hka_angle_deg_median", 0)
        hka_iqr = leg_dict.get("hka_angle_deg_iqr", 0) or 0
        dev = leg_dict.get("hka_deviation_deg_median", 0)
        dev_iqr = leg_dict.get("hka_deviation_deg_iqr", 0) or 0
        n_used = leg_dict.get("n_frames_used", 0)
        n_total = leg_dict.get("n_frames_total", 0)

        color_map = {
            "normal": "#2e8b3a", "varus": "#c84a2e", "valgus": "#3060c8",
        }
        color = color_map.get(cls, "#444")

        text_lines = [
            (f"  {cls.upper()} ({sev})", 0.04, color, 12, "bold"),
            (f"  confidence: {conf}", 0.08, "#666", 10, "normal"),
            (f"  HKA angle: {hka:.2f}° (IQR {hka_iqr:.2f}°)", 0.13, "#000", 11, "normal"),
            (f"  deviation: {dev:+.2f}° (IQR {dev_iqr:.2f}°)", 0.17, "#000", 11, "normal"),
            (f"  frames used: {n_used}/{n_total}", 0.21, "#666", 10, "normal"),
        ]
        if metric and leg_dict.get("femur_length_cm_median") is not None:
            text_lines += [
                (f"  femur: {leg_dict['femur_length_cm_median']:.1f} cm", 0.26, "#000", 11, "normal"),
                (f"  tibia: {leg_dict['tibia_length_cm_median']:.1f} cm", 0.30, "#000", 11, "normal"),
                (f"  total leg: {leg_dict['total_leg_length_cm_median']:.1f} cm", 0.34, "#000", 11, "normal"),
                (f"  MAD: {leg_dict['mad_cm_median']:.2f} cm", 0.38, "#000", 11, "normal"),
            ]
        text_lines.append(
            (f"  femur/tibia ratio: {leg_dict.get('femur_tibia_ratio_median', 0):.3f}",
             0.43, "#000", 11, "normal"),
        )
        text_lines.append(
            (f"  {leg_dict.get('classification_note', '')[:80]}",
             0.48, "#666", 9, "normal"),  # "normal" weight (compact card)
        )

        for text, dy, c, sz, w in text_lines:
            ax.text(panel_x, y0 - dy, text, fontsize=sz, color=c, weight=w,
                     transform=ax.transAxes, va="top")

    render_leg(0.03, L, "LEFT LEG")
    render_leg(0.53, R, "RIGHT LEG")

    # Bilateral metrics
    y_bil = 0.35
    ax.text(0.5, y_bil, "Bilateral", fontsize=13, weight="bold",
             ha="center", transform=ax.transAxes)
    bil_lines = []
    if a.get("intercondylar_distance_cm") is not None:
        bil_lines.append(f"  Intercondylar (knee gap): "
                          f"{a['intercondylar_distance_cm']:.1f} cm")
    if a.get("intermalleolar_distance_cm") is not None:
        bil_lines.append(f"  Intermalleolar (ankle gap): "
                          f"{a['intermalleolar_distance_cm']:.1f} cm")
    if a.get("leg_length_difference_cm") is not None:
        bil_lines.append(f"  Leg length difference: "
                          f"{a['leg_length_difference_cm']:.2f} cm "
                          f"({a.get('leg_length_difference_pct', 0):.1f}% of avg)")
        bil_lines.append(f"  LLD class: {a.get('leg_length_classification', 'unknown')}")
    if not metric:
        bil_lines.append("  (distances in arbitrary units — no scale calibration)")
    for i, line in enumerate(bil_lines):
        ax.text(0.5, y_bil - 0.04 * (i + 1), line, fontsize=10, ha="center",
                 transform=ax.transAxes)

    # Flags
    flags = a.get("flags", [])
    if flags:
        y_flags = 0.13
        ax.text(0.5, y_flags, "Flags",
                 fontsize=13, weight="bold", ha="center",
                 transform=ax.transAxes, color="#c84a2e")
        for i, f in enumerate(flags[:4]):
            ax.text(0.5, y_flags - 0.03 * (i + 1), f"• {f}", fontsize=10,
                     ha="center", transform=ax.transAxes, color="#c84a2e")

    out = os.path.join(output_dir, f"{prefix}.jpg")
    plt.tight_layout()
    plt.savefig(out, dpi=130, bbox_inches="tight",
                  facecolor="white")
    plt.close(fig)
    return out


def viz_hka_per_frame_chart(leg_assessment_path, output_dir,
                              prefix="hka_per_frame"):
    """Plot HKA angle (or deviation) per frame for both legs.
    Helps spot frames with outlier pose detections."""
    if not os.path.exists(leg_assessment_path):
        return None
    os.makedirs(output_dir, exist_ok=True)
    with open(leg_assessment_path) as f:
        a = json.load(f)

    pf_l = a.get("per_frame_left", [])
    pf_r = a.get("per_frame_right", [])
    if not pf_l and not pf_r:
        return None

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    def plot_metric(ax, metric_key, ylabel, color_l, color_r):
        if pf_l:
            xs = [f["frame_idx"] for f in pf_l]
            ys = [f[metric_key] for f in pf_l]
            ax.plot(xs, ys, "o-", color=color_l, label="left", markersize=6, alpha=0.85)
            med = np.median(ys)
            ax.axhline(med, color=color_l, alpha=0.4, linestyle="--",
                        label=f"left median={med:.2f}")
        if pf_r:
            xs = [f["frame_idx"] for f in pf_r]
            ys = [f[metric_key] for f in pf_r]
            ax.plot(xs, ys, "s-", color=color_r, label="right", markersize=6, alpha=0.85)
            med = np.median(ys)
            ax.axhline(med, color=color_r, alpha=0.4, linestyle="--",
                        label=f"right median={med:.2f}")
        ax.set_ylabel(ylabel)
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)

    plot_metric(axes[0], "hka_angle_deg", "HKA angle (°)",
                  "#2e7d3a", "#2e7d8c")
    axes[0].axhline(180.0, color="black", alpha=0.5, linestyle=":",
                     label="180° (straight)")
    plot_metric(axes[1], "hka_deviation_deg",
                  "Deviation from 180° (signed, + varus / − valgus)",
                  "#c84a2e", "#cf6c20")
    # Threshold bands
    from measurements.leg_metrics import (
        NORMAL_MAX_DEG, BORDERLINE_MAX_DEG,
        MILD_MAX_DEG, MODERATE_MAX_DEG,
    )
    for thr, label, color in [
        (NORMAL_MAX_DEG, "normal", "#2e8b3a"),
        (BORDERLINE_MAX_DEG, "borderline", "#888"),
        (MILD_MAX_DEG, "mild", "#daa520"),
        (MODERATE_MAX_DEG, "moderate", "#c8602e"),
    ]:
        axes[1].axhline(thr, color=color, alpha=0.4, linestyle="-.")
        axes[1].axhline(-thr, color=color, alpha=0.4, linestyle="-.")

    axes[1].set_xlabel("frame index")
    fig.suptitle("Per-frame HKA angle and deviation\n"
                  "(dashed = median across frames, dot-dashed = classification thresholds)",
                  fontsize=12, y=1.02)
    plt.tight_layout()
    out = os.path.join(output_dir, f"{prefix}.jpg")
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


def viz_person_pointcloud(amb3r_npz_path, recon_meta_path, segmentation_dir,
                           output_dir, leg_assessment_path=None,
                           prefix="person_pointcloud"):
    """Apply SAM3 masks to the reconstruction and render the SEGMENTED PERSON
    point cloud from multiple views, with leg landmarks overlaid if available.

    This isolates just the patient (vs. the background) so the user can see
    the actual body shape that the pipeline measured.
    """
    if not os.path.exists(amb3r_npz_path):
        return None
    os.makedirs(output_dir, exist_ok=True)

    npz = np.load(amb3r_npz_path, allow_pickle=True)
    pts_per_frame = npz["points_per_frame"]
    imgs_per_frame = npz["images_per_frame"]
    T, H, W = pts_per_frame.shape[:3]

    # Load preprocess transforms for proper mask alignment
    transforms = {}
    image_order = None
    if recon_meta_path and os.path.exists(recon_meta_path):
        with open(recon_meta_path) as f:
            rm = json.load(f)
        for t in rm.get("preprocess_transforms", []):
            transforms[t["filename"]] = t
        image_order = rm.get("image_files_in_order")

    # Load segmentation
    person_pts = []
    person_cols = []
    if os.path.exists(os.path.join(segmentation_dir, "segmentation.json")):
        with open(os.path.join(segmentation_dir, "segmentation.json")) as f:
            seg = json.load(f)
        ordered = image_order if image_order else sorted(seg.keys())
        # Import the same helper the belly pipeline uses for mask alignment
        try:
            from measurements.belly import _transform_mask_to_recon_space
        except ImportError:
            _transform_mask_to_recon_space = None
        for t, name in enumerate(ordered[:T]):
            if name not in seg:
                continue
            mask_path = seg[name].get("combined_mask_path")
            if not mask_path or not os.path.exists(mask_path):
                continue
            from PIL import Image as PILImage
            orig_mask = np.array(PILImage.open(mask_path).convert("L"))
            tf = transforms.get(name)
            if tf is not None and _transform_mask_to_recon_space is not None:
                mask = _transform_mask_to_recon_space(orig_mask, tf, H, W)
            else:
                mask = np.array(PILImage.fromarray(orig_mask).resize(
                    (W, H), PILImage.NEAREST)) > 127
            pts = pts_per_frame[t][mask]
            cols = imgs_per_frame[t][mask]
            valid = np.linalg.norm(pts, axis=-1) > 0.001
            person_pts.append(pts[valid])
            person_cols.append(cols[valid])

    if not person_pts:
        # Fall back to all points (no segmentation)
        person_pts = [pts_per_frame.reshape(-1, 3)]
        person_cols = [imgs_per_frame.reshape(-1, 3)]

    pts = np.concatenate(person_pts, axis=0)
    cols = np.concatenate(person_cols, axis=0)
    cols = np.clip(cols, 0, 1)
    print(f"  Person point cloud: {len(pts):,} points")

    # Save the segmented person cloud as a PLY for separate inspection
    try:
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.colors = o3d.utility.Vector3dVector(cols)
        ply_path = os.path.join(output_dir, f"{prefix}.ply")
        o3d.io.write_point_cloud(ply_path, pcd)
    except Exception:
        ply_path = None

    # Optionally overlay leg landmarks from leg_assessment.json
    highlight_pts = None
    highlight_labels = None
    extra_lines = []
    if leg_assessment_path and os.path.exists(leg_assessment_path):
        with open(leg_assessment_path) as f:
            la = json.load(f)
        # Use medians of joint positions across frames as representative landmarks
        for side in ("left", "right"):
            pf = la.get(f"per_frame_{side}", [])
            if not pf:
                continue
            joints = {}
            for joint_key in ("hip_3d", "knee_3d", "ankle_3d"):
                arr = np.array([f[joint_key] for f in pf])
                joints[joint_key] = arr.mean(axis=0)
            if highlight_pts is None:
                highlight_pts = []
                highlight_labels = []
            highlight_pts.extend([joints["hip_3d"], joints["knee_3d"], joints["ankle_3d"]])
            highlight_labels.extend([
                f"{side[0].upper()}-Hip", f"{side[0].upper()}-Knee", f"{side[0].upper()}-Ankle",
            ])
            # Skeleton lines
            col = "#dd2222" if side == "left" else "#2244dd"
            extra_lines.append((joints["hip_3d"], joints["knee_3d"], col))
            extra_lines.append((joints["knee_3d"], joints["ankle_3d"], col))

    # Render 3 views
    views = [("front", 10, -90, "Front"),
             ("side",  10,   0, "Side"),
             ("iso",   25, -45, "Isometric")]
    fig = plt.figure(figsize=(16, 5.5))
    for i, (_, elev, azim, label) in enumerate(views, 1):
        ax = fig.add_subplot(1, 3, i, projection="3d")
        _render_pcd_matplotlib(
            pts, cols, ax=ax,
            title=f"{label} view  ({len(pts):,} pts)",
            highlight_pts=np.array(highlight_pts) if highlight_pts is not None else None,
            highlight_color="red",
            extra_lines=extra_lines if extra_lines else None,
            view_elev=elev, view_azim=azim,
        )
        # Optional landmark text labels
        if highlight_pts is not None and i == 3 and highlight_labels:
            hp = np.array(highlight_pts)
            for j, lab in enumerate(highlight_labels):
                ax.text(hp[j, 0], hp[j, 1], hp[j, 2], "  " + lab,
                         fontsize=7, color="black")

    fig.suptitle("Segmented person point cloud (SAM3 mask applied)",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    out = os.path.join(output_dir, f"{prefix}_views.jpg")
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out, ply_path


def viz_leg_classification_chart(leg_assessment_path, output_dir,
                                   prefix="leg_classification_chart"):
    """Build a number-line chart showing each leg's HKA deviation, its 95%
    bootstrap confidence interval, and the classification threshold zones.

    Lets the user see at a glance how reliable the classification is —
    if the CI spans multiple zones, the classification is uncertain.
    """
    if not os.path.exists(leg_assessment_path):
        return None
    os.makedirs(output_dir, exist_ok=True)
    with open(leg_assessment_path) as f:
        a = json.load(f)

    from measurements.leg_metrics import (
        NORMAL_MAX_DEG, BORDERLINE_MAX_DEG, MILD_MAX_DEG, MODERATE_MAX_DEG,
    )

    fig, axes = plt.subplots(2, 1, figsize=(13, 5.5), sharex=True)
    x_max = 15.0  # ±15° range

    for ax_i, (side, leg) in enumerate(zip(axes, [a.get("left", {}), a.get("right", {})])):
        side_name = ["LEFT", "RIGHT"][ax_i]
        # Threshold zone backgrounds
        zones = [
            (-x_max, -MODERATE_MAX_DEG, "#3060c8", 0.15, "severe valgus"),
            (-MODERATE_MAX_DEG, -MILD_MAX_DEG, "#5080d0", 0.15, "moderate valgus"),
            (-MILD_MAX_DEG, -BORDERLINE_MAX_DEG, "#80a0d8", 0.15, "mild valgus"),
            (-BORDERLINE_MAX_DEG, -NORMAL_MAX_DEG, "#a8bce0", 0.15, "borderline valgus"),
            (-NORMAL_MAX_DEG, NORMAL_MAX_DEG, "#2e8b3a", 0.18, "normal"),
            (NORMAL_MAX_DEG, BORDERLINE_MAX_DEG, "#dec0a8", 0.15, "borderline varus"),
            (BORDERLINE_MAX_DEG, MILD_MAX_DEG, "#dcaa72", 0.15, "mild varus"),
            (MILD_MAX_DEG, MODERATE_MAX_DEG, "#d08040", 0.15, "moderate varus"),
            (MODERATE_MAX_DEG, x_max, "#c84a2e", 0.15, "severe varus"),
        ]
        for lo, hi, c, a_v, lbl in zones:
            ax_i_obj = axes[ax_i]
            ax_i_obj.axvspan(lo, hi, color=c, alpha=a_v)
            mid = (lo + hi) / 2
            ax_i_obj.text(mid, 0.85, lbl, ha="center", va="bottom",
                           fontsize=7.5, color="#222")

        dev = leg.get("hka_deviation_deg_median")
        if dev is None:
            axes[ax_i].text(0, 0.5, f"{side_name}: insufficient data",
                              ha="center", va="center", fontsize=12, color="#888",
                              transform=axes[ax_i].transAxes)
            continue

        ci_low = leg.get("hka_deviation_ci_low_deg", dev)
        ci_high = leg.get("hka_deviation_ci_high_deg", dev)
        # Plot CI band
        axes[ax_i].fill_betweenx([0.30, 0.55], ci_low, ci_high,
                                    color="black", alpha=0.18,
                                    label="95% CI")
        # Plot median marker
        axes[ax_i].plot([dev], [0.425], "o", markersize=18,
                          markerfacecolor="black", markeredgecolor="white",
                          markeredgewidth=2, zorder=10)
        axes[ax_i].text(dev, 0.18,
                          f"median {dev:+.2f}°\nCI [{ci_low:+.2f}°, {ci_high:+.2f}°]",
                          ha="center", va="top", fontsize=9,
                          bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black"))
        # Reliability label
        rel = leg.get("reliability_label", "?")
        rel_score = leg.get("reliability_score", 0)
        cls = leg.get("classification", "?")
        sev = leg.get("severity", "?")
        title = (f"{side_name}: {cls.upper()} ({sev})  |  "
                 f"reliability={rel} ({rel_score:.2f})  |  "
                 f"{leg.get('n_frames_used', 0)}/{leg.get('n_frames_total', 0)} frames")
        axes[ax_i].set_title(title, fontsize=11, weight="bold")

        # If class_probabilities is informative, show top 3
        probs = leg.get("class_probabilities", {})
        if probs:
            top3 = sorted(probs.items(), key=lambda kv: -kv[1])[:3]
            prob_str = "Bootstrap P:  " + "  ·  ".join(
                f"{k}={v*100:.0f}%" for k, v in top3
            )
            axes[ax_i].text(0.5, -0.20, prob_str, transform=axes[ax_i].transAxes,
                              ha="center", va="top", fontsize=8.5, color="#444")

        axes[ax_i].set_ylim(0, 1.0)
        axes[ax_i].set_xlim(-x_max, x_max)
        axes[ax_i].set_yticks([])
        axes[ax_i].grid(True, axis="x", alpha=0.3)
        if ax_i == 1:
            axes[ax_i].set_xlabel("HKA deviation from 180°  (− valgus / + varus)")

    fig.suptitle("Per-leg HKA classification — median, 95% CI, and threshold zones",
                  fontsize=12, y=1.0)
    plt.tight_layout()
    out = os.path.join(output_dir, f"{prefix}.jpg")
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


def viz_leg_report(leg_assessment_path, output_dir, prefix="leg_report"):
    """Single-page image report summarising the leg-pipeline output.

    Layout uses a 4-row grid with non-overlapping columns and wrapped text
    so long classification/stance notes don't bleed across panels.
    """
    if not os.path.exists(leg_assessment_path):
        return None
    os.makedirs(output_dir, exist_ok=True)
    with open(leg_assessment_path) as f:
        a = json.load(f)

    import textwrap as _tw
    from measurements.leg_metrics import (
        GAP_NORMAL_MAX_CM, GAP_MILD_MAX_CM, GAP_MODERATE_MAX_CM,
    )

    metric = a.get("metric_calibrated", False)
    flags = a.get("flags", []) or []
    n_flags_shown = min(len(flags), 6)
    # Pre-wrap flags so we can size the flag panel to actual line count.
    _wrap_width = 110
    _wrapped_flags = [_tw.wrap(str(f), width=_wrap_width) or [""]
                      for f in flags[:n_flags_shown]]
    total_flag_lines = sum(len(w) for w in _wrapped_flags)
    flags_h = max(0.6, 0.20 * (total_flag_lines + 1))

    fig = plt.figure(figsize=(13, 11.0 + 0.22 * total_flag_lines))
    gs = fig.add_gridspec(
        4, 2,
        height_ratios=[0.55, 2.2, 1.3, flags_h],
        hspace=0.45, wspace=0.16,
        left=0.05, right=0.97, top=0.96, bottom=0.04,
    )

    # ── HEADER (title + status pill + meta strip) ─────────────────────
    ax_hdr = fig.add_subplot(gs[0, :])
    ax_hdr.axis("off")
    ax_hdr.text(0.5, 0.78, "Leg Deformity Assessment Report",
                ha="center", va="center", fontsize=22, weight="bold",
                transform=ax_hdr.transAxes)
    overall = a.get("overall_assessment", "unknown")
    overall_color = "#2e8b3a" if "normal" in (overall or "").lower() else "#222"
    ax_hdr.text(0.5, 0.40, overall, ha="center", va="center",
                fontsize=14, color="white", transform=ax_hdr.transAxes,
                bbox=dict(boxstyle="round,pad=0.5", fc=overall_color,
                          ec="none"))
    meta_line = (
        f"Subject: {a.get('subject', 'standing')}     "
        f"Calibration: {'metric (cm)' if metric else 'arbitrary units'}     "
        f"Frames used:  L = {a.get('n_frames_used_left', 0)}   "
        f"R = {a.get('n_frames_used_right', 0)}  "
        f"(of {a.get('n_frames_total', 0)})"
    )
    ax_hdr.text(0.5, 0.04, meta_line, ha="center", va="bottom",
                fontsize=10.5, color="#555", transform=ax_hdr.transAxes)

    # ── PER-LEG PANELS ─────────────────────────────────────────────────
    color_map = {"normal": "#2e8b3a", "varus": "#c84a2e", "valgus": "#3060c8"}

    def _fmt_signed(x, suffix=""):
        return "—" if x is None else f"{x:+.2f}{suffix}"
    def _fmt(x, fmt="{:.2f}", suffix=""):
        return "—" if x is None else (fmt.format(x) + suffix)

    for col_i, (side_key, side_label) in enumerate([("left", "LEFT LEG"),
                                                     ("right", "RIGHT LEG")]):
        ax = fig.add_subplot(gs[1, col_i])
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.axis("off")
        leg = a.get(side_key, {}) or {}
        cls = leg.get("classification") or "—"
        sev = leg.get("severity") or "—"
        cls_color = color_map.get(cls, "#444")

        # Side label
        ax.text(0.5, 0.97, side_label, ha="center", va="top",
                fontsize=13, weight="bold", color="#222",
                transform=ax.transAxes)

        # Classification banner
        ax.add_patch(plt.Rectangle((0.06, 0.78), 0.88, 0.13,
                                   facecolor="white", edgecolor=cls_color,
                                   lw=2.2, transform=ax.transAxes,
                                   joinstyle="round"))
        banner = cls.upper() if sev == "none" else f"{cls.upper()}  •  {sev}"
        ax.text(0.5, 0.846, banner, ha="center", va="center",
                fontsize=15, weight="bold", color=cls_color,
                transform=ax.transAxes)

        # Reliability bar
        rel = leg.get("reliability_score") or 0.0
        rel_label = leg.get("reliability_label") or "—"
        rel_color = {"high": "#2e8b3a", "medium": "#daa520",
                     "low": "#c8302e"}.get(rel_label, "#888")
        ax.text(0.06, 0.71, "Reliability", fontsize=9.5, color="#444",
                transform=ax.transAxes, va="center")
        ax.add_patch(plt.Rectangle((0.30, 0.695), 0.50, 0.03,
                                   facecolor="#eaeaea", transform=ax.transAxes,
                                   edgecolor="none"))
        if rel > 0:
            ax.add_patch(plt.Rectangle((0.30, 0.695), 0.50 * rel, 0.03,
                                       facecolor=rel_color,
                                       transform=ax.transAxes,
                                       edgecolor="none"))
        ax.text(0.94, 0.71, f"{rel_label} ({rel:.2f})",
                fontsize=9, ha="right", va="center",
                color=rel_color, transform=ax.transAxes)

        # Numbers table
        rows = []
        rows.append(("HKA angle",
                     _fmt(leg.get("hka_angle_deg_median"), suffix="°"),
                     f"IQR {_fmt(leg.get('hka_angle_deg_iqr'), suffix='°')}"))
        dev = leg.get("hka_deviation_deg_median")
        ci_low = leg.get("hka_deviation_ci_low_deg")
        ci_high = leg.get("hka_deviation_ci_high_deg")
        ci_str = (f"CI [{_fmt_signed(ci_low)},{_fmt_signed(ci_high)}]"
                  if ci_low is not None and ci_high is not None else "")
        rows.append(("HKA deviation", _fmt_signed(dev, suffix="°"), ci_str))
        if metric:
            rows.append(("Femur",
                         _fmt(leg.get("femur_length_cm_median"),
                              fmt="{:.1f}", suffix=" cm"), ""))
            rows.append(("Tibia",
                         _fmt(leg.get("tibia_length_cm_median"),
                              fmt="{:.1f}", suffix=" cm"), ""))
            rows.append(("Total leg",
                         _fmt(leg.get("total_leg_length_cm_median"),
                              fmt="{:.1f}", suffix=" cm"), ""))
            rows.append(("MAD",
                         _fmt(leg.get("mad_cm_median"), suffix=" cm"), ""))
        rows.append(("Femur / tibia",
                     _fmt(leg.get("femur_tibia_ratio_median"), fmt="{:.3f}"),
                     f"IQR {_fmt(leg.get('femur_tibia_ratio_iqr'), fmt='{:.3f}')}"))

        y0, dy = 0.62, 0.075
        for i, (k, v, extra) in enumerate(rows):
            y = y0 - i * dy
            ax.text(0.06, y, k, fontsize=10, color="#333",
                    transform=ax.transAxes, va="center")
            ax.text(0.40, y, v, fontsize=10.5, color="#000", weight="bold",
                    transform=ax.transAxes, va="center")
            if extra:
                ax.text(0.62, y, extra, fontsize=9, color="#777",
                        transform=ax.transAxes, va="center")

        # Class probabilities (top 3) — compact
        probs = leg.get("class_probabilities") or {}
        if probs:
            top3 = sorted(probs.items(), key=lambda kv: -kv[1])[:3]
            prob_str = "  ".join(f"{lbl} {p*100:.0f}%" for lbl, p in top3)
            ax.text(0.06, 0.05, "Top bootstrap outcomes",
                    fontsize=8.5, color="#555", transform=ax.transAxes,
                    va="bottom")
            ax.text(0.06, 0.013, prob_str, fontsize=9.5, color="#222",
                    transform=ax.transAxes, va="bottom")

    # ── BILATERAL PANEL ───────────────────────────────────────────────
    ax_bil = fig.add_subplot(gs[2, :])
    ax_bil.set_xlim(0, 1); ax_bil.set_ylim(0, 1)
    ax_bil.axis("off")
    ax_bil.text(0.02, 0.97, "Bilateral metrics & stance",
                fontsize=12.5, weight="bold", color="#222",
                transform=ax_bil.transAxes, va="top")

    bil_rows = []
    if a.get("intercondylar_distance_cm") is not None:
        bil_rows.append((
            "Knee gap (intercondylar)",
            f"{a['intercondylar_distance_cm']:.2f} cm",
            f"normal ≤ {GAP_NORMAL_MAX_CM}, mild ≤ {GAP_MILD_MAX_CM}, "
            f"mod ≤ {GAP_MODERATE_MAX_CM}",
        ))
    if a.get("intermalleolar_distance_cm") is not None:
        bil_rows.append((
            "Ankle gap (intermalleolar)",
            f"{a['intermalleolar_distance_cm']:.2f} cm",
            f"normal ≤ {GAP_NORMAL_MAX_CM}, mild ≤ {GAP_MILD_MAX_CM}, "
            f"mod ≤ {GAP_MODERATE_MAX_CM}",
        ))
    if a.get("leg_length_difference_cm") is not None:
        lld_class = a.get("leg_length_classification", "?")
        side = a.get("leg_length_discrepancy_side", "n/a")
        bil_rows.append((
            "Leg-length discrepancy",
            f"{a['leg_length_difference_cm']:.2f} cm "
            f"({a.get('leg_length_difference_pct', 0):.1f}%)",
            f"{side} shorter   ·   {lld_class}",
        ))
    genu_cls = a.get("genu_alignment_classification")
    if genu_cls:
        genu_sev = a.get("genu_alignment_severity", "?")
        genu_label = {
            "normal_alignment": "Normal stance",
            "genu_varum": "Genu varum (bow-legs)",
            "genu_valgum": "Genu valgum (knock-knees)",
            "ambiguous": "Ambiguous stance",
            "insufficient_data": "Insufficient data",
        }.get(genu_cls, genu_cls)
        bil_rows.append((
            "Stance",
            f"{genu_label}",
            f"severity: {genu_sev}",
        ))

    # 3-column layout: label | value | hint
    col_label_x, col_val_x, col_hint_x = 0.02, 0.32, 0.58
    row_h = 0.78 / max(len(bil_rows), 1)
    y_top = 0.86
    for i, (k, v, hint) in enumerate(bil_rows):
        y = y_top - i * row_h
        ax_bil.text(col_label_x, y, k, fontsize=10.5, color="#333",
                    transform=ax_bil.transAxes, va="center", weight="bold")
        ax_bil.text(col_val_x, y, v, fontsize=11, color="#000",
                    transform=ax_bil.transAxes, va="center", weight="bold")
        # Wrap hint into max ~52 chars to keep it inside its column
        hint_wrapped = "\n".join(_tw.wrap(hint, width=52)) if hint else ""
        ax_bil.text(col_hint_x, y, hint_wrapped, fontsize=9, color="#777",
                    transform=ax_bil.transAxes, va="center", style="italic")

    # ── FLAGS PANEL ───────────────────────────────────────────────────
    ax_flags = fig.add_subplot(gs[3, :])
    ax_flags.set_xlim(0, 1); ax_flags.set_ylim(0, 1)
    ax_flags.axis("off")
    if flags:
        ax_flags.text(0.02, 0.97, f"Flags  ({len(flags)})",
                      fontsize=12, weight="bold", color="#c8302e",
                      transform=ax_flags.transAxes, va="top")
        # Render all flag bullets as a single multi-line text block — let
        # matplotlib's own line metrics handle leading, so font size and
        # axis size don't have to agree manually.
        lines = []
        for wrapped in _wrapped_flags:
            lines.append("• " + wrapped[0])
            for extra in wrapped[1:]:
                lines.append("    " + extra)   # hanging indent
        body = "\n".join(lines)
        ax_flags.text(0.035, 0.80, body, fontsize=10, color="#a02018",
                      transform=ax_flags.transAxes, va="top",
                      linespacing=1.45)
        if len(flags) > n_flags_shown:
            ax_flags.text(0.035, 0.04,
                          f"… (+{len(flags) - n_flags_shown} more)",
                          fontsize=9, color="#a02018", style="italic",
                          transform=ax_flags.transAxes, va="bottom")
    else:
        ax_flags.text(0.02, 0.5, "✓ No critical flags",
                      fontsize=11.5, color="#2e8b3a", weight="bold",
                      transform=ax_flags.transAxes, va="center")

    out = os.path.join(output_dir, f"{prefix}.jpg")
    plt.savefig(out, dpi=140, facecolor="white")
    plt.close(fig)
    return out


# ════════════════════════════════════════════════════════════════════
#  Anterior-frame assessment visualization (PRIMARY classification)
# ════════════════════════════════════════════════════════════════════

def viz_anterior_assessment(leg_assessment_path, recon_imgs_dir,
                              output_dir, prefix="anterior_assessment"):
    """Big single-page visual report for the single-frame 2D HKA assessment.

    Layout:
      Top half: the chosen anterior frame with mechanical axis (dashed)
        + actual bones (severity-colored) + angle/dev labels on each leg.
      Bottom half: stacked soft-probability bars for each leg, showing the
        full Gaussian-over-bands probability distribution.
    """
    if not os.path.exists(leg_assessment_path):
        return None
    os.makedirs(output_dir, exist_ok=True)
    with open(leg_assessment_path) as f:
        a = json.load(f)
    afa = a.get("anterior_frame_assessment")
    if not afa:
        return None
    frame_name = afa.get("frame_name")
    img_path = os.path.join(recon_imgs_dir, frame_name) if frame_name else None
    if not img_path or not os.path.exists(img_path):
        # Fall back to scanning the recon imgs dir
        return None

    img = cv2.imread(img_path)
    if img is None:
        return None
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Severity-to-color (same scheme as the HKA-overlay viz)
    sev_color = {
        "none": "#2e8b3a", "borderline": "#888", "mild": "#daa520",
        "moderate": "#d8772d", "severe": "#c8302e",
    }
    cls_color = {"normal": "#2e8b3a", "varus": "#c84a2e", "valgus": "#3060c8"}

    fig = plt.figure(figsize=(14, 11))
    gs = fig.add_gridspec(3, 2, height_ratios=[0.45, 2.2, 1.0],
                            hspace=0.30, wspace=0.18,
                            left=0.04, right=0.97, top=0.96, bottom=0.05)

    # ── Header ─────────────────────────────────────────────────────────
    ax_hdr = fig.add_subplot(gs[0, :])
    ax_hdr.axis("off")
    title = "Leg Deformity Assessment — Single Anterior Frame"
    ax_hdr.text(0.5, 0.78, title, ha="center", va="center",
                fontsize=20, weight="bold", transform=ax_hdr.transAxes)
    overall = afa.get("overall_assessment", "?")
    o_color = "#2e8b3a" if "normal" in (overall or "").lower() else "#222"
    ax_hdr.text(0.5, 0.30, overall, ha="center", va="center",
                fontsize=14, color="white",
                transform=ax_hdr.transAxes,
                bbox=dict(boxstyle="round,pad=0.5", fc=o_color, ec="none"))
    sub = (f"Frame: {frame_name}   ·   view: {afa.get('view_quality_label')}"
           f"   ·   hip-sep ratio: {afa.get('hip_sep_ratio', 0):.2f}")
    asym = afa.get("leg_length_asymmetry_pct")
    if asym is not None:
        sub += f"   ·   leg-length asym: {asym:.0f}%"
    ax_hdr.text(0.5, -0.05, sub, ha="center", va="center",
                fontsize=10, color="#555", transform=ax_hdr.transAxes)

    # ── Image with overlay ────────────────────────────────────────────
    ax_img = fig.add_subplot(gs[1, :])
    ax_img.imshow(rgb)
    ax_img.set_xticks([]); ax_img.set_yticks([])
    ax_img.set_title(f"Mechanical-axis overlay on {frame_name}",
                      fontsize=11, pad=6)

    for side, side_letter, prefix2 in [("left", "L", "left"), ("right", "R", "right")]:
        hip = np.array(afa.get(f"{prefix2}_hip_xy", [0, 0]))
        knee = np.array(afa.get(f"{prefix2}_knee_xy", [0, 0]))
        ank = np.array(afa.get(f"{prefix2}_ankle_xy", [0, 0]))
        if not (np.linalg.norm(hip) > 0 and np.linalg.norm(knee) > 0 and
                np.linalg.norm(ank) > 0):
            continue
        cls = afa.get(f"{prefix2}_classification", "?")
        sev = afa.get(f"{prefix2}_severity", "?")
        col = sev_color.get(sev, "#444") if sev != "none" else "#2e8b3a"
        if cls == "valgus":
            col = "#3060c8" if sev == "severe" else col

        # Mechanical axis
        ax_img.plot([hip[0], ank[0]], [hip[1], ank[1]],
                     "--", color="#888", linewidth=1.8, alpha=0.85)
        # Bones
        ax_img.plot([hip[0], knee[0]], [hip[1], knee[1]],
                     "-", color=col, linewidth=3.2, solid_capstyle="round")
        ax_img.plot([knee[0], ank[0]], [knee[1], ank[1]],
                     "-", color=col, linewidth=3.2, solid_capstyle="round")
        # Joints
        for pt in [hip, knee, ank]:
            ax_img.scatter(pt[0], pt[1], s=55, c=[col],
                            edgecolors="white", linewidth=1.5, zorder=10)
        # Knee perpendicular drop
        mech = ank - hip
        mn = np.linalg.norm(mech)
        if mn > 1e-6:
            u = mech / mn
            foot = hip + ((knee - hip) @ u) * u
            ax_img.plot([knee[0], foot[0]], [knee[1], foot[1]],
                         ":", color=col, linewidth=1.6, alpha=0.75)

        hka = afa.get(f"{prefix2}_hka_deg")
        dev = afa.get(f"{prefix2}_hka_deviation_deg")
        if hka is not None and dev is not None:
            dx = 30 if side == "left" else -30
            label = (f"{side_letter}: HKA {hka:.1f}°   dev {dev:+.1f}°\n"
                     f"   {cls}  ·  {sev}")
            ax_img.annotate(
                label,
                xy=(knee[0], knee[1]),
                xytext=(knee[0] + dx, knee[1]),
                fontsize=10, color="white", weight="bold",
                ha="left" if side == "left" else "right", va="center",
                bbox=dict(boxstyle="round,pad=0.32", fc=col,
                            ec="white", lw=0.8, alpha=0.95),
            )

    leg_handles = [
        plt.Line2D([0], [0], color="#888", lw=1.8, ls="--",
                   label="mechanical axis (hip→ankle)"),
        plt.Line2D([0], [0], color="#444", lw=3.0, label="actual bones"),
        plt.Line2D([0], [0], color="#444", lw=1.4, ls=":",
                   label="knee deflection"),
    ]
    ax_img.legend(handles=leg_handles, loc="lower left", fontsize=8.5,
                   framealpha=0.92)

    if afa.get("view_warning"):
        ax_img.text(0.5, 1.005, "⚠ " + afa["view_warning"],
                     transform=ax_img.transAxes, ha="center", va="bottom",
                     fontsize=8.5, color="#a02018",
                     bbox=dict(boxstyle="round,pad=0.25", fc="#fff3f0",
                                 ec="#d8772d", alpha=0.95))

    # ── Soft-probability bars per leg ─────────────────────────────────
    band_order = [
        "valgus_severe", "valgus_moderate", "valgus_mild", "valgus_borderline",
        "normal",
        "varus_borderline", "varus_mild", "varus_moderate", "varus_severe",
    ]
    band_colors = {
        "valgus_severe":  "#3060c8", "valgus_moderate": "#5080d0",
        "valgus_mild":    "#80a0d8", "valgus_borderline": "#a8bce0",
        "normal":         "#2e8b3a",
        "varus_borderline": "#dec0a8", "varus_mild": "#dcaa72",
        "varus_moderate":   "#d08040", "varus_severe": "#c84a2e",
    }
    pretty = {
        "valgus_severe": "valgus sev", "valgus_moderate": "valgus mod",
        "valgus_mild": "valgus mild", "valgus_borderline": "valgus bord",
        "normal": "NORMAL",
        "varus_borderline": "varus bord", "varus_mild": "varus mild",
        "varus_moderate": "varus mod", "varus_severe": "varus sev",
    }

    for col_i, (side, prefix2) in enumerate([("LEFT", "left"), ("RIGHT", "right")]):
        ax = fig.add_subplot(gs[2, col_i])
        probs = afa.get(f"{prefix2}_class_probabilities") or {}
        # Defensive: probs may have keys we didn't anticipate; restrict.
        values = [probs.get(b, 0.0) for b in band_order]
        positions = np.arange(len(band_order))
        bars = ax.barh(positions, values,
                        color=[band_colors[b] for b in band_order],
                        edgecolor="white", linewidth=1)
        ax.set_yticks(positions)
        ax.set_yticklabels([pretty[b] for b in band_order], fontsize=9)
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.set_xlabel("probability", fontsize=9)
        ax.set_title(f"{side} — soft-classification probabilities",
                      fontsize=10.5, pad=4)
        # Annotate non-trivial values
        for p, v in zip(positions, values):
            if v >= 0.01:
                ax.text(min(v + 0.012, 0.96), p, f"{v * 100:.0f}%",
                        va="center", fontsize=8.5, color="#222")
        ax.grid(axis="x", color="#eee", linewidth=0.5, zorder=0)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        # Headline below bars
        dev = afa.get(f"{prefix2}_hka_deviation_deg")
        note = afa.get(f"{prefix2}_note", "")
        cls = afa.get(f"{prefix2}_classification", "?")
        sev = afa.get(f"{prefix2}_severity", "?")
        ax.text(0.5, -0.34,
                f"dev = {dev:+.2f}°   →   {cls} / {sev}\n{note}",
                ha="center", va="top", transform=ax.transAxes,
                fontsize=9, color="#333",
                bbox=dict(boxstyle="round,pad=0.35", fc="#fafafa",
                            ec="#ccc"))

    out = os.path.join(output_dir, f"{prefix}.jpg")
    fig.savefig(out, dpi=140, facecolor="white")
    plt.close(fig)
    return out


# ════════════════════════════════════════════════════════════════════
#  HKA / mechanical-axis overlay on per-frame images
# ════════════════════════════════════════════════════════════════════

def _severity_color_rgb(severity: str):
    """Map a severity label to an RGB color tuple in 0-1 range."""
    return {
        "none": (0.18, 0.55, 0.23),       # green
        "borderline": (0.55, 0.55, 0.55), # gray
        "mild": (0.86, 0.65, 0.13),       # amber
        "moderate": (0.85, 0.45, 0.18),   # orange
        "severe": (0.78, 0.18, 0.18),     # red
    }.get(severity, (0.4, 0.4, 0.4))


def viz_hka_overlay_per_frame(leg_assessment_path, pose_results_path,
                                recon_imgs_dir, output_dir,
                                max_frames=12):
    """For each frame, draw the mechanical axis (hip→ankle) and the actual
    bones (hip→knee + knee→ankle) with a HKA-angle arc at the knee, both legs.

    This is the clinically-most-readable overlay: a doctor can SEE the
    deflection of the knee from straight, the angle, and the severity colour
    on the actual patient image.
    """
    if not (os.path.exists(leg_assessment_path)
            and os.path.exists(pose_results_path)
            and os.path.isdir(recon_imgs_dir)):
        return None
    os.makedirs(output_dir, exist_ok=True)

    with open(leg_assessment_path) as f:
        la = json.load(f)
    with open(pose_results_path) as f:
        pr = json.load(f)

    # Index per-frame measurements by frame_idx for fast lookup
    by_idx_l = {pf["frame_idx"]: pf for pf in la.get("per_frame_left", [])}
    by_idx_r = {pf["frame_idx"]: pf for pf in la.get("per_frame_right", [])}

    # Use the recon's canonical image ordering so frame_idx (T-axis) matches
    # the JSON's frame_idx and the pose results lookup.
    recon_meta_path = os.path.join(
        os.path.dirname(os.path.dirname(recon_imgs_dir)),
        "reconstruction", "reconstruction_meta.json",
    )
    image_order = None
    if os.path.exists(recon_meta_path):
        with open(recon_meta_path) as f:
            rm = json.load(f)
        image_order = rm.get("image_files_in_order")
    if image_order is None:
        image_order = sorted(pr.keys())

    saved = []
    n_done = 0
    for t, name in enumerate(image_order):
        if n_done >= max_frames:
            break
        if name not in pr:
            continue
        img_path = os.path.join(recon_imgs_dir, name)
        if not os.path.exists(img_path):
            continue

        # Use most-confident person
        persons = pr[name].get("persons", [])
        if not persons:
            continue
        person = max(persons, key=lambda p: p.get("mean_score", 0))
        lk = person.get("leg_keypoints", {})
        try:
            joints = {k: (lk[k]["x"], lk[k]["y"]) for k in (
                "left_hip", "right_hip",
                "left_knee", "right_knee",
                "left_ankle", "right_ankle",
            )}
        except KeyError:
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(rgb)
        ax.set_xticks([]); ax.set_yticks([])

        l_pf = by_idx_l.get(t)
        r_pf = by_idx_r.get(t)
        # Aggregated severity labels (for color) come from the per-leg
        # aggregate. Per-frame deviation is the actual annotation value.
        l_sev = la.get("left", {}).get("severity") or "none"
        r_sev = la.get("right", {}).get("severity") or "none"

        for side, pf, sev in [("left", l_pf, l_sev), ("right", r_pf, r_sev)]:
            hip = np.array(joints[f"{side}_hip"])
            knee = np.array(joints[f"{side}_knee"])
            ank = np.array(joints[f"{side}_ankle"])
            col = _severity_color_rgb(sev)

            # 1. Mechanical axis — dashed grey from hip to ankle
            ax.plot([hip[0], ank[0]], [hip[1], ank[1]],
                    "--", color="#888", linewidth=1.8, alpha=0.85)

            # 2. Actual bones (hip→knee, knee→ankle) — solid, severity-colored
            ax.plot([hip[0], knee[0]], [hip[1], knee[1]],
                    "-", color=col, linewidth=3.0, solid_capstyle="round")
            ax.plot([knee[0], ank[0]], [knee[1], ank[1]],
                    "-", color=col, linewidth=3.0, solid_capstyle="round")

            # 3. Joint dots
            for pt in [hip, knee, ank]:
                ax.scatter(pt[0], pt[1], s=42, c=[col],
                           edgecolors="white", linewidth=1.2, zorder=10)

            # 4. Knee-deflection vector (knee → perpendicular foot on mech axis)
            mech = ank - hip
            mech_n = np.linalg.norm(mech)
            if mech_n > 1e-6:
                u = mech / mech_n
                t_proj = (knee - hip) @ u
                foot = hip + t_proj * u
                ax.plot([knee[0], foot[0]], [knee[1], foot[1]],
                        ":", color=col, linewidth=1.5, alpha=0.7)

            # 5. HKA angle arc + label
            if pf is not None:
                hka = pf.get("hka_angle_deg")
                dev = pf.get("hka_deviation_deg")
                if hka is not None and dev is not None:
                    # Place text on the OUTWARD side of the knee
                    # (away from the other leg). For left leg, that's +x;
                    # for right, that's -x.
                    dx = 22 if side == "left" else -22
                    txt = f"{side.upper()[0]}: HKA {hka:.1f}°  dev {dev:+.1f}°"
                    ax.annotate(
                        txt,
                        xy=(knee[0], knee[1]),
                        xytext=(knee[0] + dx, knee[1] + 6),
                        fontsize=8.5, color="white",
                        weight="bold",
                        ha="left" if side == "left" else "right",
                        bbox=dict(boxstyle="round,pad=0.25",
                                   fc=col, ec="white", lw=0.6, alpha=0.92),
                    )

        # Title strip
        overall = la.get("overall_assessment", "")
        view_lbl = la.get("view_label", "?")
        title = (f"Frame {t}    |    {name}    |    view: {view_lbl}    "
                 f"|    {overall}")
        ax.set_title(title, fontsize=10, pad=6)

        # Legend in the corner
        leg_handles = [
            plt.Line2D([0], [0], color="#888", lw=1.8, ls="--",
                       label="mechanical axis"),
            plt.Line2D([0], [0], color="#444", lw=3.0, label="actual bones"),
            plt.Line2D([0], [0], color="#444", lw=1.4, ls=":",
                       label="knee deflection"),
        ]
        ax.legend(handles=leg_handles, loc="lower left",
                  fontsize=8, framealpha=0.9)

        # Use explicit margins (no bbox_inches="tight") so the image size
        # and pixel-coord overlay stay in sync — bbox_inches="tight" can
        # asymmetrically crop the figure's whitespace and visually shift
        # the displayed image.
        fig.tight_layout(pad=0.6)
        out = os.path.join(output_dir, f"hka_overlay_{t:03d}_{os.path.splitext(name)[0]}.jpg")
        fig.savefig(out, dpi=110, facecolor="white")
        plt.close(fig)
        saved.append(out)
        n_done += 1
    print(f"  wrote {len(saved)} HKA-overlay frames to {output_dir}")
    return saved


# ════════════════════════════════════════════════════════════════════
#  Bilateral comparison: L vs R bar plot
# ════════════════════════════════════════════════════════════════════

def viz_bilateral_comparison(leg_assessment_path, output_dir,
                                prefix="bilateral_comparison"):
    """Side-by-side bar chart for the key per-leg metrics, with threshold
    bands so the bars are interpretable at a glance.

    Panels:
      A. HKA deviation (with normal / borderline / mild / moderate / severe
         bands as horizontal shading)
      B. Tibia length (cm)
      C. Femur length (cm)
      D. Lower-leg volume (cm³) with method-spread as error bars
    """
    if not os.path.exists(leg_assessment_path):
        return None
    os.makedirs(output_dir, exist_ok=True)
    with open(leg_assessment_path) as f:
        a = json.load(f)
    from measurements.leg_metrics import (
        NORMAL_MAX_DEG, BORDERLINE_MAX_DEG, MILD_MAX_DEG, MODERATE_MAX_DEG,
    )

    # Decide where to read HKA values from:
    #   1. Anterior mode: anterior_frame_assessment (single-frame 2D)
    #   2. Fallback: multi-frame medians from left/right LegAggregate
    afa = a.get("anterior_frame_assessment")
    use_anterior = afa is not None and afa.get("left_hka_deviation_deg") is not None

    # In anterior mode we only show 3 panels (HKA / tibia / volume) — the
    # femur length is a 3D-multi-frame derived metric that's not measured in
    # anterior mode. Fallback mode keeps the legacy 4-panel layout.
    n_panels = 3 if use_anterior else 4
    fig, axes = plt.subplots(1, n_panels,
                              figsize=(4 * n_panels + 1, 5),
                              gridspec_kw={"wspace": 0.30})
    sides = ["left", "right"]
    side_colors = {"left": "#c84a2e", "right": "#3060c8"}

    # ── A. HKA deviation ───────────────────────────────────────────
    ax = axes[0]
    devs = []
    if use_anterior:
        for side in sides:
            d = afa.get(f"{side}_hka_deviation_deg")
            devs.append(d if d is not None else 0.0)
    else:
        for side in sides:
            d = a.get(side, {}).get("hka_deviation_deg_median")
            devs.append(d if d is not None else 0.0)
    # Band shading
    span = max(abs(d) for d in devs) + 3
    span = max(span, MODERATE_MAX_DEG + 3)
    bands = [
        (-span, -MODERATE_MAX_DEG, "#3060c8", 0.16, "severe valgus"),
        (-MODERATE_MAX_DEG, -MILD_MAX_DEG, "#80a0d8", 0.16, "moderate"),
        (-MILD_MAX_DEG, -NORMAL_MAX_DEG, "#a8bce0", 0.16, "mild/borderline"),
        (-NORMAL_MAX_DEG, NORMAL_MAX_DEG, "#2e8b3a", 0.18, "normal"),
        (NORMAL_MAX_DEG, MILD_MAX_DEG, "#dec0a8", 0.16, "mild/borderline"),
        (MILD_MAX_DEG, MODERATE_MAX_DEG, "#dcaa72", 0.16, "moderate"),
        (MODERATE_MAX_DEG, span, "#c84a2e", 0.16, "severe varus"),
    ]
    for lo, hi, col, alpha, _lbl in bands:
        ax.axhspan(lo, hi, facecolor=col, alpha=alpha, edgecolor="none")
    ax.axhline(0, color="#222", lw=0.7)
    bars = ax.bar([0, 1], devs,
                   color=[side_colors[s] for s in sides],
                   edgecolor="white", linewidth=1.5, width=0.55)
    # Confidence intervals:
    #   anterior mode: derive 95% interval from soft-class Gaussian (±2σ from
    #                  measured value, where σ is implied by the probs spread)
    #   multi-frame:   use bootstrap CI from the JSON
    if use_anterior:
        # For anterior mode, draw a "soft uncertainty" band of ±4° (≈ 2σ
        # for the default σ=2°; widened in the measurement when the view
        # was sketchy). This visually communicates the SOFT boundary.
        for i in range(2):
            ax.errorbar(i, devs[i], yerr=[[4.0], [4.0]],
                         fmt="none", ecolor="#222", capsize=4, lw=1.2,
                         alpha=0.8)
    else:
        for i, side in enumerate(sides):
            lo = a.get(side, {}).get("hka_deviation_ci_low_deg")
            hi = a.get(side, {}).get("hka_deviation_ci_high_deg")
            if lo is not None and hi is not None:
                ax.errorbar(i, devs[i],
                             yerr=[[devs[i] - lo], [hi - devs[i]]],
                             fmt="none", ecolor="#222", capsize=4, lw=1.2)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["LEFT", "RIGHT"])
    ax.set_ylabel("HKA deviation (°)   +varus  /  −valgus")
    ax.set_title(
        ("HKA deviation\n(single anterior frame, ±2σ)" if use_anterior
         else "HKA deviation\nwith 95% CI"),
        fontsize=11,
    )
    ax.set_ylim(-span - 3, span + 3)
    for i, v in enumerate(devs):
        ax.text(i, v + (2 if v >= 0 else -2) * 1.0,
                f"{v:+.1f}°", ha="center",
                va="bottom" if v >= 0 else "top",
                fontsize=10, weight="bold", color="#222")

    # ── B. Tibia length ────────────────────────────────────────────
    ax = axes[1]
    # Anterior mode doesn't measure tibia length directly (it's a 2D angle
    # check), but the volume estimator stores tibia_length_cm from the
    # keypoint-derived 3D positions.
    if use_anterior:
        tibias = [
            (a.get(f"lower_leg_volume_{side}") or {}).get("tibia_length_cm")
            for side in sides
        ]
    else:
        tibias = [a.get(side, {}).get("tibia_length_cm_median") for side in sides]
    if all(t is not None for t in tibias):
        ax.bar([0, 1], tibias,
                color=[side_colors[s] for s in sides],
                edgecolor="white", linewidth=1.5, width=0.55)
        ax.axhspan(35, 45, facecolor="#2e8b3a", alpha=0.10,
                    edgecolor="none", label="typical adult (35-45cm)")
        for i, v in enumerate(tibias):
            ax.text(i, v + 0.8, f"{v:.1f} cm", ha="center", fontsize=10,
                    weight="bold", color="#222")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["LEFT", "RIGHT"])
        ax.set_ylabel("Tibia length (cm)")
        ax.set_title("Tibia length", fontsize=11)
        ax.set_ylim(0, max(tibias) * 1.18 + 2)
        ax.legend(fontsize=8, loc="lower right")
    else:
        ax.text(0.5, 0.5, "no scale calibration",
                ha="center", va="center", color="#888",
                transform=ax.transAxes)
        ax.set_axis_off()

    # ── C. Femur length (multi-frame mode only) ────────────────────
    if not use_anterior:
        ax = axes[2]
        femurs = [a.get(side, {}).get("femur_length_cm_median") for side in sides]
        if all(f is not None for f in femurs):
            ax.bar([0, 1], femurs,
                    color=[side_colors[s] for s in sides],
                    edgecolor="white", linewidth=1.5, width=0.55)
            ax.axhspan(40, 50, facecolor="#2e8b3a", alpha=0.10,
                        edgecolor="none", label="typical adult (40-50cm)")
            for i, v in enumerate(femurs):
                ax.text(i, v + 0.8, f"{v:.1f} cm", ha="center", fontsize=10,
                        weight="bold", color="#222")
            ax.set_xticks([0, 1]); ax.set_xticklabels(["LEFT", "RIGHT"])
            ax.set_ylabel("Femur length (cm)")
            ax.set_title("Femur length", fontsize=11)
            ax.set_ylim(0, max(femurs) * 1.18 + 2)
            ax.legend(fontsize=8, loc="lower right")
        else:
            ax.text(0.5, 0.5, "no scale calibration",
                    ha="center", va="center", color="#888",
                    transform=ax.transAxes)
            ax.set_axis_off()

    # ── D. Lower-leg volume (ellipse slab fit only) ───────────────
    ax = axes[2 if use_anterior else 3]
    vols = [None, None]
    for i, side in enumerate(sides):
        v = a.get(f"lower_leg_volume_{side}") or {}
        vols[i] = v.get("volume_cm3")
    if any(v is not None for v in vols):
        xs, ys, cs = [], [], []
        for i, side in enumerate(sides):
            if vols[i] is None:
                continue
            xs.append(i); ys.append(vols[i]); cs.append(side_colors[side])
        ax.bar(xs, ys, color=cs, edgecolor="white", linewidth=1.5, width=0.55)
        y_max = max(ys) * 1.22 + 100   # headroom for the label
        for i, x in enumerate(xs):
            ax.text(x, ys[i] + max(ys) * 0.04, f"{ys[i]:.0f} cm³",
                    ha="center", va="bottom",
                    fontsize=9.5, weight="bold", color="#222")
        ax.axhspan(1200, 2800, facecolor="#2e8b3a", alpha=0.10,
                    edgecolor="none", label="typical adult (1.2-2.8L)")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["LEFT", "RIGHT"])
        ax.set_ylabel("Volume (cm³)")
        ax.set_title("Lower-leg volume\n(ellipse slab fit)", fontsize=11)
        ax.set_ylim(0, y_max)
        ax.legend(fontsize=8, loc="lower right")
    else:
        ax.text(0.5, 0.5, "volume unavailable\n(no scale calibration)",
                ha="center", va="center", color="#888",
                transform=ax.transAxes)
        ax.set_axis_off()

    method_tag = ("single anterior frame" if use_anterior
                   else "multi-frame 3D aggregate")
    fig.suptitle(
        f"Bilateral comparison — {a.get('overall_assessment', '')}"
        f"   ({method_tag})",
        fontsize=12.5, weight="bold", y=0.99,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(output_dir, f"{prefix}.jpg")
    fig.savefig(out, dpi=140, facecolor="white")
    plt.close(fig)
    return out


# ════════════════════════════════════════════════════════════════════
#  Frame-quality dashboard
# ════════════════════════════════════════════════════════════════════

def viz_frame_quality_dashboard(leg_assessment_path, pose_results_path,
                                  output_dir, prefix="frame_quality"):
    """One-strip-per-frame dashboard showing:
      - Pose mean confidence (color column)
      - View-quality (hip-sep ratio) per frame
      - Per-frame HKA deviation, color-coded by severity band
      - Used / filtered / outlier-dropped status

    Helps you see at a glance which frames contributed to the result.
    """
    if not (os.path.exists(leg_assessment_path)
            and os.path.exists(pose_results_path)):
        return None
    os.makedirs(output_dir, exist_ok=True)
    with open(leg_assessment_path) as f:
        a = json.load(f)
    with open(pose_results_path) as f:
        pr = json.load(f)
    from measurements.leg_metrics import (
        MIN_KEYPOINT_SCORE, ANTERIOR_VIEW_MIN_HIP_SEP, ANTERIOR_VIEW_CLEAN_HIP_SEP,
        NORMAL_MAX_DEG, BORDERLINE_MAX_DEG, MILD_MAX_DEG, MODERATE_MAX_DEG,
    )

    # Reconstruct frame order
    recon_meta_path = os.path.join(
        os.path.dirname(os.path.dirname(pose_results_path)),
        "reconstruction", "reconstruction_meta.json",
    )
    image_order = None
    if os.path.exists(recon_meta_path):
        with open(recon_meta_path) as f:
            rm = json.load(f)
        image_order = rm.get("image_files_in_order")
    if image_order is None:
        image_order = sorted(pr.keys())

    # Compute per-frame stats
    rows = []
    for t, name in enumerate(image_order):
        rec = {"frame": t, "name": name}
        persons = pr.get(name, {}).get("persons", [])
        if not persons:
            rec["status"] = "no_person"
            rows.append(rec); continue
        p = max(persons, key=lambda x: x.get("mean_score", 0))
        rec["mean_score"] = p.get("mean_score", 0)
        lk = p.get("leg_keypoints", {})
        try:
            lh, rh = lk["left_hip"], lk["right_hip"]
            la_, ra_ = lk["left_ankle"], lk["right_ankle"]
            leg_h = max(abs(la_["y"] - lh["y"]), abs(ra_["y"] - rh["y"])) or 1
            rec["hip_sep"] = abs(lh["x"] - rh["x"]) / leg_h
            rec["min_score"] = min(lk[k]["score"] for k in lk)
        except (KeyError, ZeroDivisionError):
            rec["status"] = "missing_joint"
            rows.append(rec); continue
        rows.append(rec)

    # Mark per-frame status
    used_l = {pf["frame_idx"] for pf in a.get("per_frame_left", [])}
    used_r = {pf["frame_idx"] for pf in a.get("per_frame_right", [])}
    dev_by_idx = {pf["frame_idx"]: pf["hka_deviation_deg"]
                   for pf in a.get("per_frame_left", [])}
    dev_by_idx_r = {pf["frame_idx"]: pf["hka_deviation_deg"]
                     for pf in a.get("per_frame_right", [])}
    for r in rows:
        idx = r["frame"]
        if "status" in r:
            continue
        if idx in used_l or idx in used_r:
            r["status"] = "used"
        elif r.get("min_score", 0) < MIN_KEYPOINT_SCORE:
            r["status"] = "low_score"
        elif r.get("hip_sep", 0) < ANTERIOR_VIEW_MIN_HIP_SEP:
            r["status"] = "oblique"
        else:
            r["status"] = "filtered"

    n = len(rows)
    fig, axes = plt.subplots(4, 1, figsize=(max(12, n * 0.5), 6.5),
                              gridspec_kw={"hspace": 0.55})

    # Row 1 — status
    ax = axes[0]
    status_color = {
        "used": "#2e8b3a", "oblique": "#dba040", "low_score": "#c8302e",
        "filtered": "#888", "no_person": "#444", "missing_joint": "#444",
    }
    for i, r in enumerate(rows):
        ax.add_patch(plt.Rectangle((i - 0.4, 0), 0.8, 1,
                                       facecolor=status_color.get(r["status"], "#888"),
                                       edgecolor="white"))
        ax.text(i, 0.5, r["status"][0].upper(), ha="center", va="center",
                fontsize=8, color="white", weight="bold")
    ax.set_xlim(-0.5, n - 0.5); ax.set_ylim(0, 1)
    ax.set_yticks([0.5]); ax.set_yticklabels(["status"])
    ax.set_xticks(range(n))
    ax.set_xticklabels([str(r["frame"]) for r in rows], fontsize=7)
    ax.set_title("Per-frame status: U=used / O=oblique / L=low_score / F=filtered",
                  fontsize=10, loc="left")

    # Row 2 — hip X-sep ratio
    ax = axes[1]
    hips = [r.get("hip_sep", 0) for r in rows]
    ax.bar(range(n), hips, color=[
        "#2e8b3a" if h >= ANTERIOR_VIEW_CLEAN_HIP_SEP
        else "#daa520" if h >= ANTERIOR_VIEW_MIN_HIP_SEP
        else "#c8302e" for h in hips])
    ax.axhline(ANTERIOR_VIEW_MIN_HIP_SEP, color="#666", lw=1, ls="--")
    ax.axhline(ANTERIOR_VIEW_CLEAN_HIP_SEP, color="#222", lw=1, ls="--")
    ax.set_xticks(range(n))
    ax.set_xticklabels([str(r["frame"]) for r in rows], fontsize=7)
    ax.set_ylabel("hip X-sep")
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_title("View quality (anterior threshold = 0.16, clean = 0.22)",
                  fontsize=10, loc="left")

    # Row 3 — pose min score
    ax = axes[2]
    ms = [r.get("min_score", 0) for r in rows]
    ax.bar(range(n), ms, color=[
        "#2e8b3a" if v >= 0.7 else "#daa520" if v >= MIN_KEYPOINT_SCORE
        else "#c8302e" for v in ms])
    ax.axhline(MIN_KEYPOINT_SCORE, color="#666", lw=1, ls="--",
                label=f"min keypoint = {MIN_KEYPOINT_SCORE}")
    ax.set_xticks(range(n))
    ax.set_xticklabels([str(r["frame"]) for r in rows], fontsize=7)
    ax.set_ylabel("min pose score")
    ax.set_ylim(0, 1)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_title("Pose-detector min keypoint confidence",
                  fontsize=10, loc="left")
    ax.legend(fontsize=8, loc="lower right")

    # Row 4 — Per-frame HKA dev (left + right side-by-side)
    ax = axes[3]
    bw = 0.35
    devs_l = [dev_by_idx.get(r["frame"], np.nan) for r in rows]
    devs_r = [dev_by_idx_r.get(r["frame"], np.nan) for r in rows]
    def _dev_color(d):
        if d is None or (isinstance(d, float) and np.isnan(d)): return "#ccc"
        ad = abs(d)
        if ad <= NORMAL_MAX_DEG: return "#2e8b3a"
        if ad <= BORDERLINE_MAX_DEG: return "#a8bce0" if d < 0 else "#dec0a8"
        if ad <= MILD_MAX_DEG: return "#80a0d8" if d < 0 else "#dcaa72"
        if ad <= MODERATE_MAX_DEG: return "#5080d0" if d < 0 else "#d08040"
        return "#3060c8" if d < 0 else "#c8302e"
    xs = np.arange(n)
    ax.bar(xs - bw / 2, [d if not np.isnan(d) else 0 for d in devs_l],
            bw, color=[_dev_color(d) for d in devs_l], label="LEFT",
            edgecolor="#000", linewidth=0.4)
    ax.bar(xs + bw / 2, [d if not np.isnan(d) else 0 for d in devs_r],
            bw, color=[_dev_color(d) for d in devs_r], label="RIGHT",
            edgecolor="#000", linewidth=0.4)
    ax.axhline(0, color="#222", lw=0.6)
    for lvl in [NORMAL_MAX_DEG, -NORMAL_MAX_DEG,
                  MILD_MAX_DEG, -MILD_MAX_DEG,
                  MODERATE_MAX_DEG, -MODERATE_MAX_DEG]:
        ax.axhline(lvl, color="#888", lw=0.5, ls=":")
    ax.set_xticks(range(n))
    ax.set_xticklabels([str(r["frame"]) for r in rows], fontsize=7)
    ax.set_ylabel("HKA dev (°)")
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_xlabel("frame index")
    ax.set_title("Per-frame HKA deviation (after view+outlier filters)",
                  fontsize=10, loc="left")
    ax.legend(fontsize=8, loc="upper right")

    fig.suptitle("Frame quality dashboard", fontsize=12, weight="bold", y=0.995)
    out = os.path.join(output_dir, f"{prefix}.jpg")
    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


# ════════════════════════════════════════════════════════════════════
#  Lower-leg volume: slab visualization
# ════════════════════════════════════════════════════════════════════

def viz_lower_leg_volume_slabs(leg_assessment_path, npz_path,
                                  recon_meta_path, segmentation_dir,
                                  output_dir, prefix="lower_leg_volume_slabs"):
    """Render the slab-by-slab ellipse fits used to estimate lower-leg volume.

    Layout:
      Left half: 3D view of the cleaned person points, coloured by axial
        slab membership, with hip-knee-ankle landmarks.
      Right half (per leg): a 2D side projection along tibia axis with
        the fitted ellipse profile rendered as nested contours.
    """
    if not (os.path.exists(leg_assessment_path) and os.path.exists(npz_path)):
        return None
    os.makedirs(output_dir, exist_ok=True)
    with open(leg_assessment_path) as f:
        a = json.load(f)
    if not a.get("metric_calibrated"):
        return None

    # Need a person cloud. We'll rebuild it here to make this viz
    # standalone (no dependency on the orchestrator's transient state).
    try:
        from PIL import Image as PILImage
        from measurements.belly import _transform_mask_to_recon_space
    except ImportError:
        return None
    if not os.path.exists(os.path.join(segmentation_dir, "segmentation.json")):
        return None
    with open(os.path.join(segmentation_dir, "segmentation.json")) as f:
        seg = json.load(f)
    npz = np.load(npz_path, allow_pickle=True)
    pts_per_frame = npz["points_per_frame"]
    T, H, W = pts_per_frame.shape[:3]
    with open(recon_meta_path) as f:
        rm = json.load(f)
    tfm = {t["filename"]: t for t in rm.get("preprocess_transforms", [])}
    image_order = rm.get("image_files_in_order", sorted(seg.keys()))
    person_chunks = []
    for t, name in enumerate(image_order[:T]):
        if name not in seg: continue
        mp = seg[name].get("combined_mask_path")
        if not mp or not os.path.exists(mp): continue
        om = np.array(PILImage.open(mp).convert("L"))
        ttf = tfm.get(name)
        if ttf is not None:
            mask = _transform_mask_to_recon_space(om, ttf, H, W)
        else:
            mask = (np.array(PILImage.fromarray(om).resize((W, H), PILImage.NEAREST)) > 127)
        pts = pts_per_frame[t][mask]
        pts = pts[np.linalg.norm(pts, axis=-1) > 0.01]
        if len(pts) > 0:
            person_chunks.append(pts)
    if not person_chunks:
        return None
    person_pts = np.concatenate(person_chunks)

    # Auto-scale to cm if cloud is in metres
    med_r = float(np.median(np.linalg.norm(person_pts, axis=1)))
    if med_r < 5.0:
        person_pts = person_pts * 100.0

    from measurements.leg_metrics import clean_person_pointcloud
    person_pts = clean_person_pointcloud(person_pts)

    # Median per-leg joints
    def _median_3d(frames, key, scale):
        if not frames: return None
        arr = np.array([f[key] for f in frames])
        med = np.median(arr, axis=0)
        return med * scale
    scale = 100.0 if med_r < 5.0 else 1.0
    l_knee = _median_3d(a.get("per_frame_left", []), "knee_3d", scale)
    l_ank  = _median_3d(a.get("per_frame_left", []), "ankle_3d", scale)
    r_knee = _median_3d(a.get("per_frame_right", []), "knee_3d", scale)
    r_ank  = _median_3d(a.get("per_frame_right", []), "ankle_3d", scale)

    # For each side compute slab geometry (same algorithm as the volume fn)
    def _slab_geometry(knee, ank, other_knee):
        from measurements.leg_metrics import (
            _knn_outlier_filter, _robust_radial_filter, _fit_slab_ellipse,
            MAX_RADIAL_CM, DEFAULT_N_SLABS, MIN_POINTS_PER_SLAB,
            SLAB_AXIS_PCT,
        )
        tibia = ank - knee
        L = np.linalg.norm(tibia)
        if L < 1.0: return None
        tu = tibia / L
        t_lo, t_hi = 2.0, L - 2.0
        rel = person_pts - knee
        t = rel @ tu
        perp = rel - t[:, None] * tu
        radial = np.linalg.norm(perp, axis=1)
        keep = (t >= t_lo) & (t <= t_hi) & (radial < MAX_RADIAL_CM)
        if other_knee is not None:
            mid = 0.5 * (knee + other_knee)
            lat = (knee - other_knee)
            lat /= max(np.linalg.norm(lat), 1e-6)
            keep = keep & ((person_pts - mid) @ lat > -1.0)
        pk = person_pts[keep]; tk = t[keep]; perp_k = perp[keep]; radk = radial[keep]
        if len(pk) < 30: return None
        pk, tk, perp_k, radk = _knn_outlier_filter(
            pk, tk, perp_k, radk, k=12, std_ratio=2.0
        )
        ref = np.array([1, 0, 0]) if abs(tu[0]) < 0.9 else np.array([0, 1, 0])
        e1 = ref - tu * (ref @ tu); e1 /= np.linalg.norm(e1)
        e2 = np.cross(tu, e1)
        proj_2d = np.column_stack([perp_k @ e1, perp_k @ e2])
        edges = np.linspace(t_lo, t_hi, DEFAULT_N_SLABS + 1)
        slab_h = (t_hi - t_lo) / DEFAULT_N_SLABS
        slabs = []
        for i in range(DEFAULT_N_SLABS):
            m = (tk >= edges[i]) & (tk < edges[i + 1])
            s_radial = radk[m]; s_pts2 = proj_2d[m]
            if len(s_pts2) < MIN_POINTS_PER_SLAB:
                slabs.append(None); continue
            mask = _robust_radial_filter(s_radial)
            if mask.sum() < MIN_POINTS_PER_SLAB:
                mask = np.ones_like(s_radial, dtype=bool)
            a_ax, b_ax = _fit_slab_ellipse(s_pts2[mask], pct=SLAB_AXIS_PCT)
            slabs.append({
                "t_lo": float(edges[i]), "t_hi": float(edges[i + 1]),
                "a": a_ax, "b": b_ax, "slab_h": slab_h, "n": int(mask.sum()),
            })
        return {
            "tu": tu, "e1": e1, "e2": e2, "knee": knee, "ankle": ank,
            "t_lo": t_lo, "t_hi": t_hi,
            "slabs": slabs, "kept_points": pk,
        }

    l_geom = _slab_geometry(l_knee, l_ank, r_knee) if l_knee is not None else None
    r_geom = _slab_geometry(r_knee, r_ank, l_knee) if r_knee is not None else None

    # ── Render — 4 multi-view 3D panels + 2 slab profile panels ─────
    from mpl_toolkits.mplot3d import Axes3D  # noqa

    # Stack all kept lower-leg points to pick the 3D camera zoom
    kept_all = []
    for geom in (l_geom, r_geom):
        if geom is not None and len(geom["kept_points"]) > 0:
            kept_all.append(geom["kept_points"])
    kept_all = (np.concatenate(kept_all) if kept_all else None)

    # Pre-compute slab-ellipse polygons in 3D so we can draw them on every
    # 3D view. Each ellipse is sampled at 64 points on its plane.
    def _ellipse_polygons(geom):
        """Return a list of (N, 3) arrays — one polygon per slab."""
        if geom is None or "slabs" not in geom:
            return []
        knee = geom["knee"]; tu = geom["tu"]
        e1 = geom["e1"]; e2 = geom["e2"]
        theta = np.linspace(0, 2 * np.pi, 64, endpoint=True)
        out = []
        for s in geom["slabs"]:
            if s is None:
                continue
            t_center = 0.5 * (s["t_lo"] + s["t_hi"])
            center_3d = knee + t_center * tu
            a_ax, b_ax = s["a"], s["b"]
            polygon = (center_3d[None, :]
                       + a_ax * np.cos(theta)[:, None] * e1[None, :]
                       + b_ax * np.sin(theta)[:, None] * e2[None, :])
            out.append(polygon)
        return out

    l_ellipses = _ellipse_polygons(l_geom)
    r_ellipses = _ellipse_polygons(r_geom)

    def _render_3d_view(ax, elev, azim, title, draw_ellipses=True):
        # Faint grey body context
        sub = min(len(person_pts), 20_000)
        idx = np.random.default_rng(0).choice(len(person_pts), sub, replace=False)
        ax.scatter(person_pts[idx, 0], person_pts[idx, 1], person_pts[idx, 2],
                    s=0.4, c="#d0d0d0", alpha=0.15)

        for geom, color, ellipses in [
            (l_geom, "#c84a2e", l_ellipses),
            (r_geom, "#3060c8", r_ellipses),
        ]:
            if geom is None:
                continue
            kept = geom["kept_points"]
            if len(kept) > 0:
                ki = (np.random.default_rng(1).choice(
                    len(kept), min(len(kept), 4000), replace=False)
                    if len(kept) > 4000 else np.arange(len(kept)))
                ax.scatter(kept[ki, 0], kept[ki, 1], kept[ki, 2],
                            s=2.4, c=color, alpha=0.65, edgecolors="none")
            # Tibia axis
            ax.plot([geom["knee"][0], geom["ankle"][0]],
                     [geom["knee"][1], geom["ankle"][1]],
                     [geom["knee"][2], geom["ankle"][2]],
                     color="black", lw=1.8)
            # Joints
            ax.scatter(*geom["knee"], s=80, c="white",
                        edgecolors=color, linewidth=2.0)
            ax.scatter(*geom["ankle"], s=80, c="white",
                        edgecolors=color, linewidth=2.0)
            # Slab-ellipse contours — show every 2nd slab so it doesn't
            # become a dense mess.
            if draw_ellipses:
                for i, poly in enumerate(ellipses):
                    if i % 2 != 0:
                        continue
                    ax.plot(poly[:, 0], poly[:, 1], poly[:, 2],
                             color=color, lw=1.0, alpha=0.55)

        ax.set_title(title, fontsize=9.5)
        ax.set_xlabel("X (cm)", fontsize=7.5)
        ax.set_ylabel("Y (cm)", fontsize=7.5)
        ax.set_zlabel("Z (cm)", fontsize=7.5)
        ax.tick_params(labelsize=6)
        ax.view_init(elev=elev, azim=azim)

        # Zoom in on lower legs
        if kept_all is not None and len(kept_all) > 0:
            mins = kept_all.min(axis=0); maxs = kept_all.max(axis=0)
            center = (mins + maxs) / 2
            extent = max((maxs - mins).max() * 0.65, 20.0)
            ax.set_xlim(center[0] - extent, center[0] + extent)
            ax.set_ylim(center[1] - extent, center[1] + extent)
            ax.set_zlim(center[2] - extent, center[2] + extent)

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(
        2, 4,
        height_ratios=[1.4, 1.0],
        hspace=0.28, wspace=0.20,
        left=0.04, right=0.97, top=0.93, bottom=0.06,
    )

    # Top row — four 3D views: front, side, back, isometric
    views = [
        (8,   -90, "Front view"),
        (8,     0, "Side view (from patient's right)"),
        (8,    90, "Back view"),
        (22,  -50, "Isometric"),
    ]
    for i, (elev, azim, title) in enumerate(views):
        ax = fig.add_subplot(gs[0, i], projection="3d")
        _render_3d_view(ax, elev, azim, title)

    # Bottom row — per-leg side profiles (axis = ellipse semi-axes vs
    # axial position from knee) for both legs, then volume summary.
    def _draw_slab_panel(ax, geom, label, color):
        ax.set_title(f"{label} — slab ellipse profile  (knee → ankle)",
                      fontsize=9.5)
        if geom is None:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes); ax.set_axis_off(); return
        slabs = geom["slabs"]
        t_center, a_vals, b_vals = [], [], []
        for s in slabs:
            if s is None: continue
            t_center.append((s["t_lo"] + s["t_hi"]) / 2)
            a_vals.append(s["a"]); b_vals.append(s["b"])
        if not t_center:
            ax.text(0.5, 0.5, "no slab fits", ha="center", va="center",
                    transform=ax.transAxes); ax.set_axis_off(); return
        ax.fill_between(t_center, a_vals, [-v for v in a_vals],
                         color=color, alpha=0.20, label="major semi-axis (a)")
        ax.fill_between(t_center, b_vals, [-v for v in b_vals],
                         color=color, alpha=0.45, label="minor semi-axis (b)")
        ax.axhline(0, color="black", lw=0.9)
        ax.set_xlabel("axial distance from knee (cm)", fontsize=8.5)
        ax.set_ylabel("ellipse semi-axis (cm)", fontsize=8.5)
        ax.set_xlim(0, geom["t_hi"] + 1)
        ax.legend(fontsize=8, loc="lower center", framealpha=0.95)

    ax_lp = fig.add_subplot(gs[1, 0:2])
    _draw_slab_panel(ax_lp, l_geom, "LEFT", "#c84a2e")
    ax_rp = fig.add_subplot(gs[1, 2:4])
    _draw_slab_panel(ax_rp, r_geom, "RIGHT", "#3060c8")

    # Volume annotations as text boxes (no reliability label anymore)
    for ax, geom, label, color in [
        (ax_lp, l_geom, "left", "#c84a2e"),
        (ax_rp, r_geom, "right", "#3060c8"),
    ]:
        if geom is None: continue
        vol = a.get(f"lower_leg_volume_{label}", {}) or {}
        v_cm3 = vol.get("volume_cm3") or 0
        n_slabs = (f"{vol.get('n_slabs_with_data', 0)}"
                   f"/{vol.get('n_slabs_total', 0)}")
        kp_tibia = vol.get("tibia_length_cm") or 0
        cl_tibia = vol.get("tibia_length_cloud_cm") or 0
        max_circ = vol.get("max_circumference_cm") or 0
        mean_circ = vol.get("mean_circumference_cm") or 0
        ax.text(
            0.02, 0.97,
            f"V_ellipse  = {v_cm3:.0f} cm³\n"
            f"slabs      = {n_slabs}\n"
            f"tibia(kp)  = {kp_tibia:.1f} cm\n"
            f"tibia(cld) = {cl_tibia:.1f} cm\n"
            f"max circ   = {max_circ:.1f} cm\n"
            f"mean circ  = {mean_circ:.1f} cm",
            ha="left", va="top", transform=ax.transAxes,
            fontsize=9, family="monospace",
            bbox=dict(boxstyle="round,pad=0.32", fc="white",
                        ec=color, lw=1.2),
        )

    fig.suptitle(
        "Lower-leg volume — point-cloud multi-view + slab ellipse fits  "
        "(visual sanity check)",
        fontsize=12, weight="bold", y=0.985,
    )
    out = os.path.join(output_dir, f"{prefix}.jpg")
    fig.savefig(out, dpi=130, facecolor="white")
    plt.close(fig)
    return out


def run_leg_debug(output_dir, image_dir=None):
    """Generate all debug visualisations for the LEG pipeline.

    Two viz sets, depending on the JSON's primary_method:

      anterior_mode (PREFERRED):
        - Scale calibration overlay
        - SAM3 segmentation overlay
        - Reconstruction point cloud
        - Person point cloud
        - **Anterior assessment** (the chosen frame with mechanical axis +
          bones + per-leg soft-probability bars) — PRIMARY OUTPUT
        - Bilateral comparison (HKA from anterior, tibia + volume)
        - Lower-leg volume slabs

      multi-frame mode (FALLBACK — only when no anterior frame was chosen):
        Adds back: leg_report, classification chart, HKA per-frame chart,
        compact card, per-frame HKA overlays, frame-quality dashboard.

    All outputs are saved to <output>/debug/leg/.
    """
    debug_dir = os.path.join(output_dir, "debug")
    os.makedirs(debug_dir, exist_ok=True)
    leg_dir = os.path.join(debug_dir, "leg")
    os.makedirs(leg_dir, exist_ok=True)

    leg_out = os.path.join(output_dir, "leg_assessment.json")
    pose_results_path = os.path.join(output_dir, "pose", "pose_results.json")
    recon_imgs_dir = os.path.join(output_dir, "reconstruction", "amb3r_images")

    # Detect which mode the JSON was written in
    anterior_mode = False
    if os.path.exists(leg_out):
        try:
            with open(leg_out) as f:
                _la = json.load(f)
            anterior_mode = (_la.get("primary_method")
                             == "single_anterior_frame_2d")
        except Exception:
            pass

    mode_tag = ("ANTERIOR-FRAME mode" if anterior_mode
                else "MULTI-FRAME 3D mode (LEGACY)")
    print("\n" + "=" * 60)
    print(f"GENERATING LEG DEBUG VISUALIZATIONS — {mode_tag}")
    print("=" * 60)

    # ── Step 1: Stale-output cleanup ─────────────────────────────────
    # In anterior mode, we don't generate the multi-frame viz files; if
    # they exist from a prior run, remove them so the user isn't confused
    # by stale plots.
    if anterior_mode:
        import glob, shutil
        stale_files = [
            "leg_report.jpg",
            "leg_classification_chart.jpg",
            "hka_per_frame.jpg",
            "leg_assessment_card_compact.jpg",
            "frame_quality.jpg",
        ]
        for s in stale_files:
            p = os.path.join(leg_dir, s)
            if os.path.exists(p):
                os.remove(p)
                print(f"  [cleanup] removed stale {s}")
        # Per-frame pose images — in anterior mode the anterior_assessment.jpg
        # is the canonical per-frame view; the rest are noise.
        for p in glob.glob(os.path.join(leg_dir, "pose2d_frame_*.jpg")):
            os.remove(p)
        # Stale subdirectories from multi-frame runs
        for sub in ("hka_overlay", "3d_landmarks"):
            sp = os.path.join(leg_dir, sub)
            if os.path.isdir(sp):
                shutil.rmtree(sp)
                print(f"  [cleanup] removed stale {sub}/ directory")

    # ── [1] Scale calibration overlay ────────────────────────────────
    if image_dir:
        for cal_name in ("scale_calibration.json", "patient_scale.json"):
            cal_path = os.path.join(image_dir, cal_name)
            if os.path.exists(cal_path):
                print("\n[1] Scale calibration overlay:")
                viz_scale_calibration(image_dir, cal_path,
                                       os.path.join(debug_dir, "scale"))
                break

    # ── [2] SAM3 person segmentation overlay ─────────────────────────
    seg_dir = os.path.join(output_dir, "segmentation")
    if os.path.isdir(seg_dir) and image_dir:
        print("\n[2] SAM3 person segmentation overlay:")
        viz_segmentation_overlay(image_dir, seg_dir,
                                  os.path.join(debug_dir, "segmentation"),
                                  mask_color=(0, 200, 255))

    # ── [3] Reconstruction point cloud multi-view ────────────────────
    recon_ply = os.path.join(output_dir, "reconstruction", "point_cloud.ply")
    if os.path.exists(recon_ply):
        print("\n[3] Reconstruction point cloud — multi-view:")
        viz_pointcloud_views(recon_ply, os.path.join(debug_dir, "reconstruction"),
                              prefix="reconstruction", title_prefix="Reconstruction — ")

    # ── [4] Person point cloud (SAM3-segmented body with leg landmarks)
    npz_path = os.path.join(output_dir, "reconstruction", "point_cloud.npz")
    rmeta_path = os.path.join(output_dir, "reconstruction", "reconstruction_meta.json")
    if (os.path.exists(npz_path) and os.path.isdir(seg_dir)):
        print("\n[4] Person point cloud (SAM3-segmented):")
        try:
            viz_person_pointcloud(
                npz_path, rmeta_path, seg_dir, leg_dir,
                leg_assessment_path=leg_out if os.path.exists(leg_out) else None,
            )
        except Exception as e:
            print(f"     person point cloud viz failed: {e}")

    # ── [5] PRIMARY anterior-frame assessment (anterior mode only) ───
    if anterior_mode and os.path.isdir(recon_imgs_dir):
        print("\n[5] Anterior-frame assessment (PRIMARY):")
        try:
            viz_anterior_assessment(leg_out, recon_imgs_dir, leg_dir)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"     anterior assessment viz failed: {e}")

    # ── [6] Bilateral comparison bars ────────────────────────────────
    if os.path.exists(leg_out):
        print("\n[6] Bilateral comparison bars:")
        try:
            viz_bilateral_comparison(leg_out, leg_dir)
        except Exception as e:
            print(f"     bilateral comparison failed: {e}")

    # ── [7] Lower-leg volume slab visualisation ──────────────────────
    if (os.path.exists(leg_out) and os.path.exists(npz_path)
            and os.path.exists(rmeta_path) and os.path.isdir(seg_dir)):
        print("\n[7] Lower-leg volume slab visualisation:")
        try:
            viz_lower_leg_volume_slabs(leg_out, npz_path, rmeta_path,
                                          seg_dir, leg_dir)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"     slab viz failed: {e}")

    # ── MULTI-FRAME-MODE ONLY: extra diagnostic viz (skipped in anterior mode)
    if not anterior_mode:
        # [M1] 2D pose visualisations (each frame with skeleton overlay)
        if os.path.exists(pose_results_path) and os.path.isdir(recon_imgs_dir):
            print("\n[M1] 2D pose with skeleton overlay:")
            with open(pose_results_path) as f:
                pose_results = json.load(f)
            for img_name, img_data in pose_results.items():
                ip = img_data.get("image_path", "")
                if not os.path.exists(ip):
                    ip = os.path.join(recon_imgs_dir, img_name)
                    if not os.path.exists(ip):
                        continue
                for person in img_data.get("persons", []):
                    out_p = os.path.join(
                        leg_dir,
                        f"pose2d_{os.path.splitext(img_name)[0]}_p{person['person_index']}.jpg",
                    )
                    try:
                        debug_pose_2d(ip, person, out_p, None)
                    except Exception as e:
                        print(f"     pose2d viz failed for {img_name}: {e}")

        # [M2] 3D landmark + skeleton point cloud
        pcd_path = os.path.join(output_dir, "reconstruction", "point_cloud.ply")
        if os.path.exists(pcd_path) and os.path.exists(leg_out):
            print("\n[M2] 3D landmarks on point cloud:")
            with open(leg_out) as f:
                la = json.load(f)
            pf_l = la.get("per_frame_left", [])
            pf_r = la.get("per_frame_right", [])
            if pf_l and pf_r:
                joints = {
                    "left_hip":    pf_l[0]["hip_3d"],
                    "left_knee":   pf_l[0]["knee_3d"],
                    "left_ankle":  pf_l[0]["ankle_3d"],
                    "right_hip":   pf_r[0]["hip_3d"],
                    "right_knee":  pf_r[0]["knee_3d"],
                    "right_ankle": pf_r[0]["ankle_3d"],
                }
                try:
                    debug_3d_landmarks(pcd_path, joints,
                                        os.path.join(leg_dir, "3d_landmarks"),
                                        assessment_3d=None)
                except Exception as e:
                    print(f"     3d landmark viz failed: {e}")

        # [M3] Leg report (image-based, all metrics at a glance)
        if os.path.exists(leg_out):
            print("\n[M3] Leg report (multi-frame, image-based summary):")
            try:
                viz_leg_report(leg_out, leg_dir)
            except Exception as e:
                print(f"     leg report failed: {e}")

        # [M4] Classification chart with CI + threshold zones
        if os.path.exists(leg_out):
            print("\n[M4] Classification chart (CI + threshold zones):")
            try:
                viz_leg_classification_chart(leg_out, leg_dir)
            except Exception as e:
                print(f"     classification chart failed: {e}")

        # [M5] HKA per-frame chart (helps spot outlier frames)
        if os.path.exists(leg_out):
            print("\n[M5] HKA per-frame chart:")
            try:
                viz_hka_per_frame_chart(leg_out, leg_dir)
            except Exception as e:
                print(f"     HKA per-frame chart failed: {e}")

        # [M6] Per-frame HKA / mechanical-axis overlay
        if (os.path.exists(leg_out) and os.path.exists(pose_results_path)
                and os.path.isdir(recon_imgs_dir)):
            print("\n[M6] Per-frame HKA mechanical-axis overlay:")
            try:
                viz_hka_overlay_per_frame(leg_out, pose_results_path,
                                            recon_imgs_dir,
                                            os.path.join(leg_dir, "hka_overlay"),
                                            max_frames=8)
            except Exception as e:
                print(f"     HKA overlay failed: {e}")
        # [M7] Frame-quality dashboard
        if os.path.exists(leg_out) and os.path.exists(pose_results_path):
            print("\n[M7] Frame quality dashboard:")
            try:
                viz_frame_quality_dashboard(leg_out, pose_results_path, leg_dir)
            except Exception as e:
                print(f"     frame quality dashboard failed: {e}")

        # [M8] Compact older-style assessment card
        if os.path.exists(leg_out):
            print("\n[M8] Compact assessment card:")
            try:
                viz_leg_assessment_card(leg_out, leg_dir,
                                          prefix="leg_assessment_card_compact")
            except Exception as e:
                print(f"     compact assessment card failed: {e}")

    print(f"\nAll leg debug outputs: {debug_dir}")
    return debug_dir


def run_all_debug(output_dir, image_dir=None):
    debug_dir = os.path.join(output_dir, "debug")
    os.makedirs(debug_dir, exist_ok=True)

    print("\n" + "="*60 + "\nGENERATING DEBUG VISUALIZATIONS\n" + "="*60)

    # [0a] Scale calibration overlay (manual scale picker)
    if image_dir:
        cal_path = os.path.join(image_dir, "scale_calibration.json")
        if os.path.exists(cal_path):
            print("\n[0a] Scale calibration overlay:")
            saved = viz_scale_calibration(image_dir, cal_path,
                                           os.path.join(debug_dir, "scale"))
            if saved:
                print(f"  Saved {len(saved)} scale-overlay images")

    # [0b] ArUco detection overlay
    aruco_cal_path = os.path.join(output_dir, "calibration.json")
    if image_dir and os.path.exists(aruco_cal_path):
        try:
            with open(aruco_cal_path) as f:
                cd = json.load(f)
            if cd.get("source") == "aruco" or cd.get("detections"):
                print("\n[0b] ArUco detection overlay:")
                saved = viz_aruco_detection(image_dir, aruco_cal_path,
                                              os.path.join(debug_dir, "aruco"))
                if saved:
                    print(f"  Saved {len(saved)} ArUco overlays")
        except Exception as e:
            print(f"  (ArUco viz skipped: {e})")

    # [0c] SAM3 person segmentation overlay
    seg_dir = os.path.join(output_dir, "segmentation")
    if os.path.isdir(seg_dir) and image_dir:
        print("\n[0c] SAM3 person segmentation overlay:")
        saved = viz_segmentation_overlay(image_dir, seg_dir,
                                         os.path.join(debug_dir, "segmentation"),
                                         mask_color=(0, 255, 200))
        if saved:
            print(f"  Saved {len(saved)} segmentation overlays")

    # [0d] Reconstruction point cloud multi-view
    recon_ply = os.path.join(output_dir, "reconstruction", "point_cloud.ply")
    if os.path.exists(recon_ply):
        print("\n[0d] Reconstruction point cloud — multi-view:")
        viz_pointcloud_views(recon_ply, os.path.join(debug_dir, "reconstruction"),
                             prefix="reconstruction", title_prefix="Reconstruction — ")

    # [0e] Outlier-removal before/after comparison
    raw_ply = os.path.join(output_dir, "postprocessed", "point_cloud_person.ply")
    clean_ply = os.path.join(output_dir, "postprocessed", "point_cloud_clean.ply")
    if not os.path.exists(raw_ply):
        raw_ply = recon_ply
    if os.path.exists(raw_ply) and os.path.exists(clean_ply):
        print("\n[0e] Point cloud post-processing (before vs. after):")
        viz_pointcloud_compare(raw_ply, clean_ply,
                                os.path.join(debug_dir, "postprocess"),
                                label_before="raw / SAM3-filtered",
                                label_after="outlier-cleaned")

    amb3r_npz = os.path.join(output_dir, "reconstruction", "point_cloud.npz")
    if os.path.exists(amb3r_npz):
        print("\n[1] AMB3R debug:")
        debug_amb3r_pointmap(amb3r_npz, os.path.join(debug_dir, "amb3r"))

    pose_path = os.path.join(output_dir, "pose", "pose_results.json")
    meas_path = os.path.join(output_dir, "clinical_measurements.json")
    if os.path.exists(pose_path):
        print("\n[2] 2D Pose debug:")
        with open(pose_path) as f: pose_results = json.load(f)
        m2d = {}
        if os.path.exists(meas_path):
            with open(meas_path) as f: m2d = json.load(f)
        for img_name, img_data in pose_results.items():
            ip = img_data.get("image_path","")
            if not os.path.exists(ip): continue
            for person in img_data["persons"]:
                pm = None
                if img_name in m2d:
                    for pd in m2d[img_name]:
                        if pd.get("person_index")==person["person_index"] and "assessment" in pd:
                            pm = pd["assessment"]; break
                debug_pose_2d(ip, person, os.path.join(debug_dir, f"debug_pose2d_{img_name}_p{person['person_index']}.jpg"), pm)

    meas3d_path = os.path.join(output_dir, "clinical_measurements_3d.json")
    pcd_path = os.path.join(output_dir, "postprocessed", "point_cloud_clean.ply")
    if not os.path.exists(pcd_path):
        pcd_path = os.path.join(output_dir, "reconstruction", "point_cloud.ply")

    # Load confidence data for height calculation
    conf_data = None
    if os.path.exists(amb3r_npz):
        amb3r = np.load(amb3r_npz, allow_pickle=True)
        conf_data = amb3r["conf_per_frame"][0]
        if conf_data.ndim == 3: conf_data = conf_data[:,:,0]

    if os.path.exists(meas3d_path) and os.path.exists(pcd_path):
        print("\n[3] 3D Landmark + Measurement debug:")
        with open(meas3d_path) as f: meas_3d = json.load(f)
        for img_name, persons in meas_3d.items():
            for pd in persons:
                if "error" in pd: continue
                a = pd["assessment"]
                joints = {}
                for sk in ["left_leg","right_leg"]:
                    leg = a.get(sk)
                    if not leg: continue
                    s = leg["side"]
                    joints[f"{s}_hip"] = leg.get("hip_3d")
                    joints[f"{s}_knee"] = leg.get("knee_3d")
                    joints[f"{s}_ankle"] = leg.get("ankle_3d")
                debug_3d_landmarks(pcd_path, joints, os.path.join(debug_dir, "3d"), a,
                                   confidence_data=conf_data)
                break
            break

    # Projection debug
    amb3r_pose_path = os.path.join(output_dir, "pose_amb3r", "pose_results.json")
    if not os.path.exists(amb3r_pose_path): amb3r_pose_path = pose_path
    if os.path.exists(amb3r_pose_path) and os.path.exists(meas3d_path):
        print("\n[4] Projection debug:")
        with open(amb3r_pose_path) as f: pr = json.load(f)
        with open(meas3d_path) as f: m3 = json.load(f)
        for img_name in pr:
            if img_name not in m3: continue
            ip = pr[img_name].get("image_path","")
            if not os.path.exists(ip): continue
            for pp, mp in zip(pr[img_name]["persons"], m3[img_name]):
                if "error" in mp: continue
                a = mp["assessment"]
                joints = {}
                for sk in ["left_leg","right_leg"]:
                    leg = a.get(sk)
                    if not leg: continue
                    s = leg["side"]
                    joints[f"{s}_hip"] = leg.get("hip_3d")
                    joints[f"{s}_knee"] = leg.get("knee_3d")
                    joints[f"{s}_ankle"] = leg.get("ankle_3d")
                debug_landmark_projection(ip, pp["leg_keypoints"], joints,
                                          os.path.join(debug_dir, f"debug_projection_{img_name}.jpg"))
                break
            break

    print(f"\nAll debug outputs: {debug_dir}")
    return debug_dir


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--image_dir", default=None,
                        help="Original input image dir (for overlay visualizations)")
    parser.add_argument(
        "--mode", default="leg",
        choices=["leg", "leg_legacy", "belly", "both"],
        help="leg = new clean leg pipeline. leg_legacy = old leg pipeline. "
             "belly = belly pipeline. both = leg + belly.",
    )
    args = parser.parse_args()
    if args.mode == "leg":
        run_leg_debug(args.output_dir, image_dir=args.image_dir)
    if args.mode == "leg_legacy":
        run_all_debug(args.output_dir, image_dir=args.image_dir)
    if args.mode == "belly":
        run_belly_debug(args.output_dir, image_dir=args.image_dir)
    if args.mode == "both":
        run_leg_debug(args.output_dir, image_dir=args.image_dir)
        run_belly_debug(args.output_dir, image_dir=args.image_dir)
