"""
SMPL body model fitting to AMB3R point cloud.

Fits the SMPL parametric body model to the reconstructed 3D point cloud
to get anatomically correct joint positions and body measurements.

Pipeline:
1. Initialize SMPL with pose from 2D keypoints (lifted to 3D via point map)
2. Optimize SMPL parameters (pose, shape, translation, scale) to minimize
   chamfer distance between SMPL mesh and person point cloud
3. Extract joint positions from fitted model (these are anatomical joint
   CENTERS, not surface points)
4. Extract body measurements (height, limb lengths, circumferences)

Requirements:
    pip install smplx torch trimesh scipy
    Download SMPL model from https://smpl.is.tue.mpg.de
    Place as: data/body_models/smpl/SMPL_NEUTRAL.pkl
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))

# SMPL joint indices (24 joints)
SMPL_JOINTS = {
    "pelvis": 0, "left_hip": 1, "right_hip": 2, "spine1": 3,
    "left_knee": 4, "right_knee": 5, "spine2": 6,
    "left_ankle": 7, "right_ankle": 8, "spine3": 9,
    "left_foot": 10, "right_foot": 11, "neck": 12,
    "left_collar": 13, "right_collar": 14, "head": 15,
    "left_shoulder": 16, "right_shoulder": 17,
    "left_elbow": 18, "right_elbow": 19,
    "left_wrist": 20, "right_wrist": 21,
    "left_hand": 22, "right_hand": 23,
}

# The joints we need for leg deformity
LEG_JOINTS = {
    "left_hip": 1, "right_hip": 2,
    "left_knee": 4, "right_knee": 5,
    "left_ankle": 7, "right_ankle": 8,
}


def chamfer_distance_one_way(src, tgt, batch_size=10000):
    """Compute one-way chamfer distance: for each point in src, find nearest in tgt.

    Returns mean distance (not squared).
    """
    # Process in batches to avoid OOM
    dists = []
    for i in range(0, len(src), batch_size):
        src_batch = src[i:i + batch_size]  # (B, 3)
        diff = src_batch.unsqueeze(1) - tgt.unsqueeze(0)  # (B, N, 3)
        d = torch.norm(diff, dim=2)  # (B, N)
        min_d = d.min(dim=1)[0]  # (B,)
        dists.append(min_d)
    return torch.cat(dists).mean()


def fit_smpl_to_pointcloud(
    point_cloud,
    initial_joints_3d=None,
    smpl_model_path=None,
    n_iterations=200,
    lr=0.01,
    device="cuda",
):
    """Fit SMPL body model to a 3D point cloud.

    Args:
        point_cloud: (N, 3) numpy array of person's 3D points (metric)
        initial_joints_3d: Optional dict of joint_name -> [x,y,z] for initialization
        smpl_model_path: Path to SMPL model directory
        n_iterations: Optimization iterations
        lr: Learning rate
        device: torch device

    Returns:
        dict with fitted SMPL parameters, joints, vertices, measurements
    """
    try:
        import smplx
    except ImportError:
        return {"error": "smplx not installed. Run: pip install smplx"}

    if smpl_model_path is None:
        smpl_model_path = os.path.join(PROJECT_DIR, "data", "body_models")

    smpl_path = os.path.join(smpl_model_path, "smpl")
    if not os.path.isdir(smpl_path):
        return {
            "error": f"SMPL model not found at {smpl_path}. "
            "Download from https://smpl.is.tue.mpg.de and place "
            "SMPL_NEUTRAL.pkl in data/body_models/smpl/"
        }

    # Create SMPL model
    model = smplx.create(
        smpl_model_path,
        model_type="smpl",
        gender="NEUTRAL",
        num_betas=10,
        ext="pkl",
    ).to(device)

    # Target point cloud
    target_pts = torch.tensor(point_cloud, dtype=torch.float32, device=device)

    # Subsample target for speed
    if len(target_pts) > 15000:
        idx = torch.randperm(len(target_pts))[:15000]
        target_pts_sub = target_pts[idx]
    else:
        target_pts_sub = target_pts

    # Initialize parameters
    body_pose = torch.zeros(1, 69, device=device, requires_grad=True)  # 23 joints * 3 axis-angle
    betas = torch.zeros(1, 10, device=device, requires_grad=True)
    global_orient = torch.zeros(1, 3, device=device, requires_grad=True)
    transl = torch.zeros(1, 3, device=device, requires_grad=True)
    scale = torch.ones(1, device=device, requires_grad=True)

    # Better initialization: place SMPL at centroid of point cloud
    centroid = target_pts.mean(dim=0)
    transl.data[0] = centroid

    # Estimate initial scale from point cloud extent
    extent = target_pts.max(dim=0)[0] - target_pts.min(dim=0)[0]
    # SMPL default height is ~1.7m. Compare with point cloud's largest extent.
    max_extent = extent.max().item()
    if max_extent > 0.01:
        # Estimate: SMPL spans ~1.7m, point cloud spans max_extent
        init_scale = max_extent / 1.7
        scale.data[0] = init_scale

    # Detect vertical axis
    if initial_joints_3d:
        hip_mid = None
        ankle_mid = None
        lh, rh = initial_joints_3d.get("left_hip"), initial_joints_3d.get("right_hip")
        la, ra = initial_joints_3d.get("left_ankle"), initial_joints_3d.get("right_ankle")
        if lh and rh:
            hip_mid = (np.array(lh) + np.array(rh)) / 2
        if la and ra:
            ankle_mid = (np.array(la) + np.array(ra)) / 2
        if hip_mid is not None and ankle_mid is not None:
            vert_vec = hip_mid - ankle_mid
            vert_axis = int(np.argmax(np.abs(vert_vec)))
            # Initialize global_orient to align SMPL Y-up with the actual vertical axis
            if vert_axis == 0:  # X is vertical
                global_orient.data[0, 2] = np.pi / 2  # rotate 90° around Z
            elif vert_axis == 2:  # Z is vertical
                global_orient.data[0, 0] = np.pi / 2  # rotate 90° around X

    # Optimizer
    optimizer = torch.optim.Adam(
        [body_pose, betas, global_orient, transl, scale], lr=lr
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=80, gamma=0.5)

    print(f"  SMPL fitting: {n_iterations} iterations, {len(target_pts_sub)} target points")

    best_loss = float("inf")
    best_params = None

    for i in range(n_iterations):
        optimizer.zero_grad()

        output = model(
            body_pose=body_pose,
            betas=betas,
            global_orient=global_orient,
            transl=transl,
            return_verts=True,
        )

        # Scale SMPL vertices
        verts = output.vertices[0] * scale  # (6890, 3)

        # Subsample SMPL vertices for speed
        if len(verts) > 5000:
            vidx = torch.randperm(len(verts))[:5000]
            verts_sub = verts[vidx]
        else:
            verts_sub = verts

        # Chamfer distance (bidirectional)
        loss_s2t = chamfer_distance_one_way(verts_sub, target_pts_sub)
        loss_t2s = chamfer_distance_one_way(target_pts_sub, verts_sub)
        loss_chamfer = loss_s2t + loss_t2s

        # Regularization
        loss_pose = 0.001 * (body_pose ** 2).sum()
        loss_betas = 0.01 * (betas ** 2).sum()

        # Joint loss if initial joints provided
        loss_joints = torch.tensor(0.0, device=device)
        if initial_joints_3d:
            joints = output.joints[0] * scale  # (24, 3)
            for name, idx in LEG_JOINTS.items():
                if name in initial_joints_3d and initial_joints_3d[name] is not None:
                    target = torch.tensor(initial_joints_3d[name], dtype=torch.float32, device=device)
                    loss_joints += F.mse_loss(joints[idx], target)
            loss_joints *= 10.0

        loss = loss_chamfer + loss_pose + loss_betas + loss_joints
        loss.backward()
        optimizer.step()
        scheduler.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_params = {
                "body_pose": body_pose.detach().clone(),
                "betas": betas.detach().clone(),
                "global_orient": global_orient.detach().clone(),
                "transl": transl.detach().clone(),
                "scale": scale.detach().clone(),
            }

        if i % 50 == 0:
            print(f"    Iter {i:3d}: loss={loss.item():.6f} "
                  f"chamfer={loss_chamfer.item():.6f} "
                  f"joints={loss_joints.item():.6f} "
                  f"scale={scale.item():.4f}")

    # Get final output with best params
    with torch.no_grad():
        output = model(
            body_pose=best_params["body_pose"],
            betas=best_params["betas"],
            global_orient=best_params["global_orient"],
            transl=best_params["transl"],
            return_verts=True,
        )
        final_scale = best_params["scale"].item()
        final_verts = (output.vertices[0] * final_scale).cpu().numpy()
        final_joints = (output.joints[0] * final_scale).cpu().numpy()  # (24, 3) or more

    # Extract leg joint positions
    joint_positions = {}
    for name, idx in SMPL_JOINTS.items():
        if idx < len(final_joints):
            joint_positions[name] = final_joints[idx].tolist()

    # Body measurements from SMPL
    body_measurements = _compute_smpl_measurements(final_verts, final_joints, final_scale)

    result = {
        "joints": joint_positions,
        "leg_joints": {n: final_joints[i].tolist() for n, i in LEG_JOINTS.items()},
        "vertices": final_verts,  # (6890, 3)
        "betas": best_params["betas"][0].cpu().numpy().tolist(),
        "scale": final_scale,
        "best_loss": best_loss,
        "body_measurements": body_measurements,
    }

    print(f"  SMPL fit complete. Scale={final_scale:.4f}, Loss={best_loss:.6f}")
    return result


def _compute_smpl_measurements(vertices, joints, scale):
    """Compute body measurements from fitted SMPL."""
    measurements = {}

    # Height: from lowest to highest vertex along dominant axis
    # Detect vertical from hip→ankle direction
    hip_mid = (joints[1] + joints[2]) / 2
    ankle_mid = (joints[7] + joints[8]) / 2
    vert_vec = hip_mid - ankle_mid
    vert_axis = int(np.argmax(np.abs(vert_vec)))

    measurements["height_cm"] = float((vertices[:, vert_axis].max() - vertices[:, vert_axis].min()) * 100)
    measurements["vertical_axis"] = ["X", "Y", "Z"][vert_axis]

    # Limb lengths from joint positions
    def dist(a, b):
        return float(np.linalg.norm(a - b) * 100)

    measurements["left_femur_cm"] = dist(joints[1], joints[4])
    measurements["right_femur_cm"] = dist(joints[2], joints[5])
    measurements["left_tibia_cm"] = dist(joints[4], joints[7])
    measurements["right_tibia_cm"] = dist(joints[5], joints[8])
    measurements["left_leg_total_cm"] = measurements["left_femur_cm"] + measurements["left_tibia_cm"]
    measurements["right_leg_total_cm"] = measurements["right_femur_cm"] + measurements["right_tibia_cm"]
    measurements["inter_hip_cm"] = dist(joints[1], joints[2])
    measurements["inter_knee_cm"] = dist(joints[4], joints[5])
    measurements["inter_ankle_cm"] = dist(joints[7], joints[8])

    # Shoulder width
    measurements["shoulder_width_cm"] = dist(joints[16], joints[17])

    # Arm lengths
    measurements["left_upper_arm_cm"] = dist(joints[16], joints[18])
    measurements["right_upper_arm_cm"] = dist(joints[17], joints[19])
    measurements["left_forearm_cm"] = dist(joints[18], joints[20])
    measurements["right_forearm_cm"] = dist(joints[19], joints[21])

    return measurements


def save_smpl_mesh(vertices, faces, output_path):
    """Save fitted SMPL mesh as OBJ file."""
    import trimesh
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    mesh.export(output_path)
    print(f"  Saved SMPL mesh: {output_path}")


def run_smpl_fitting(
    point_cloud_path,
    joints_3d_path,
    output_dir,
    smpl_model_path=None,
    n_iterations=200,
):
    """Full SMPL fitting pipeline.

    Args:
        point_cloud_path: Path to PLY point cloud (person only)
        joints_3d_path: Path to clinical_measurements_3d.json
        output_dir: Where to save outputs
        smpl_model_path: Path to body model directory
        n_iterations: Optimization iterations
    """
    import open3d as o3d

    os.makedirs(output_dir, exist_ok=True)

    # Load point cloud
    pcd = o3d.io.read_point_cloud(point_cloud_path)
    pts = np.asarray(pcd.points)

    # Load initial joint positions
    initial_joints = None
    if os.path.exists(joints_3d_path):
        with open(joints_3d_path) as f:
            meas = json.load(f)
        # Get first person's joints from first frame
        for img_name, persons in meas.items():
            for pd in persons:
                if "error" in pd:
                    continue
                a = pd["assessment"]
                initial_joints = {}
                for sk in ["left_leg", "right_leg"]:
                    leg = a.get(sk)
                    if not leg:
                        continue
                    s = leg["side"]
                    initial_joints[f"{s}_hip"] = leg.get("hip_3d")
                    initial_joints[f"{s}_knee"] = leg.get("knee_3d")
                    initial_joints[f"{s}_ankle"] = leg.get("ankle_3d")
                break
            break

    # Fit SMPL
    device = "cuda" if torch.cuda.is_available() else "cpu"
    result = fit_smpl_to_pointcloud(
        pts,
        initial_joints_3d=initial_joints,
        smpl_model_path=smpl_model_path,
        n_iterations=n_iterations,
        device=device,
    )

    if "error" in result:
        print(f"  SMPL fitting failed: {result['error']}")
        with open(os.path.join(output_dir, "smpl_result.json"), "w") as f:
            json.dump(result, f, indent=2)
        return result

    # Save results
    result_json = {
        "joints": result["joints"],
        "leg_joints": result["leg_joints"],
        "betas": result["betas"],
        "scale": result["scale"],
        "best_loss": result["best_loss"],
        "body_measurements": result["body_measurements"],
    }
    with open(os.path.join(output_dir, "smpl_result.json"), "w") as f:
        json.dump(result_json, f, indent=2)

    # Save SMPL mesh as PLY
    try:
        import smplx
        model = smplx.create(
            smpl_model_path or os.path.join(PROJECT_DIR, "data", "body_models"),
            model_type="smpl", gender="NEUTRAL", num_betas=10, ext="pkl",
        )
        faces = model.faces
        verts_path = os.path.join(output_dir, "smpl_mesh.ply")
        save_smpl_mesh(result["vertices"], faces, verts_path)
    except Exception as e:
        print(f"  Could not save SMPL mesh: {e}")

    print(f"\n  SMPL Results:")
    bm = result["body_measurements"]
    print(f"    Height:         {bm['height_cm']:.1f} cm")
    print(f"    Left femur:     {bm['left_femur_cm']:.1f} cm")
    print(f"    Left tibia:     {bm['left_tibia_cm']:.1f} cm")
    print(f"    Left total leg: {bm['left_leg_total_cm']:.1f} cm")
    print(f"    Right femur:    {bm['right_femur_cm']:.1f} cm")
    print(f"    Right tibia:    {bm['right_tibia_cm']:.1f} cm")
    print(f"    Right total leg:{bm['right_leg_total_cm']:.1f} cm")
    print(f"    Inter-hip:      {bm['inter_hip_cm']:.1f} cm")
    print(f"    Scale factor:   {result['scale']:.4f}")

    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fit SMPL to point cloud")
    parser.add_argument("--point_cloud", required=True)
    parser.add_argument("--joints_3d", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--smpl_model_path", default=None)
    parser.add_argument("--iterations", type=int, default=200)
    args = parser.parse_args()

    run_smpl_fitting(
        args.point_cloud, args.joints_3d, args.output_dir,
        args.smpl_model_path, args.iterations,
    )
