"""
File: data/data_generate.py
Author: Tingyin Zhao @ SGroup, EE, Tsinghua
Date: 2026-02-14
Description: Cryo-EM simulation pipeline.
"""

import os
import json
import random
import numpy as np
import torch
import torch.fft as fft
import torch.distributed as dist
import scipy.constants as const
from Bio.PDB import MMCIFParser
from PIL import Image
from tqdm import tqdm
import torchvision.transforms.functional as F
import cv2
from scipy.interpolate import interp1d

from config.data_config import SIM_CONFIG


def parse_asu_coords(cif_path):
    """Parse atom coordinates from CIF file."""
    if not os.path.exists(cif_path):
        print(f"Error: CIF not found at {cif_path}")
        return None
    parser = MMCIFParser(QUIET=True)
    try:
        structure = parser.get_structure("STRUCTURE", cif_path)
        return np.array([atom.get_coord() for atom in structure[0].get_atoms()])
    except Exception as e:
        print(f"Error parsing CIF {cif_path}: {e}")
        return None


def precompute_fiber_atoms(fiber_config):
    """Pre-calculate helical stack atoms based on fiber configuration."""
    asu_coords = parse_asu_coords(fiber_config['cif_path'])
    if asu_coords is None:
        return None

    rise = fiber_config['helical_rise']
    num_units = fiber_config['num_units']
    asu_local = asu_coords - np.mean(asu_coords, axis=0)

    # Replicate units along the Z-axis
    all_xy = np.tile(asu_local[:, :2], (num_units, 1))
    atom_z = asu_local[:, 2]
    shifts = np.arange(num_units) * rise
    s_all = (atom_z[np.newaxis, :] + shifts[:, np.newaxis]).flatten()

    return {
        "name": fiber_config['name'],
        "local_xy": all_xy,
        "s_coords": s_all,
        "twist_deg": fiber_config['helical_twist'],
        "rise_ang": fiber_config['helical_rise'],
        "length_ang": num_units * rise
    }


def get_random_rotation_matrix(device):
    """Generate a random 3D rotation matrix."""
    angles = torch.rand(3, device=device) * 2.0 * torch.pi
    phi, theta, psi = angles[0], angles[1] / 2.0, angles[2]

    cp, sp = torch.cos(phi), torch.sin(phi)
    ct, st = torch.cos(theta), torch.sin(theta)
    cs, ss = torch.cos(psi), torch.sin(psi)

    rz = torch.tensor([[cp, -sp, 0.], [sp, cp, 0.], [0., 0., 1.]], device=device)
    ry = torch.tensor([[ct, 0., st], [0., 1., 0.], [-st, 0., ct]], device=device)
    rx = torch.tensor([[1., 0., 0.], [0., cs, -ss], [0., ss, cs]], device=device)

    return rz @ ry @ rx


def calculate_ctf_gpu(k2, wavelen, defocus, device):
    """Calculate the Contrast Transfer Function (CTF) on GPU."""
    cfg = SIM_CONFIG['Global']
    w = torch.as_tensor(wavelen, device=device, dtype=torch.float32)
    df = torch.as_tensor(defocus, device=device, dtype=torch.float32)
    pi = torch.pi
    cs = cfg['SPHERICAL_ABBERATION_MM']
    ac = cfg['AMPLITUDE_CONTRAST']

    chi = (pi * w * df * 1.0e4 * k2) + (0.5 * pi * cs * 1.0e7 * (w ** 3.0) * (k2 ** 2.0))
    amp = torch.sqrt(torch.as_tensor(1.0 - ac ** 2.0, device=device, dtype=torch.float32))

    return -(amp * torch.sin(chi)) - (ac * torch.cos(chi))


def apply_straight_fiber(l_xy, s_c, fiber_params, device):
    """Generate geometry for a straight fiber."""
    s = torch.as_tensor(s_c, device=device, dtype=torch.float32)
    l_xy_gpu = torch.as_tensor(l_xy, device=device, dtype=torch.float32)

    # Apply Twist
    twist_rad = (fiber_params['twist_deg'] * np.pi / 180.0) / fiber_params['rise_ang']
    theta = s * twist_rad
    c, sn = torch.cos(theta), torch.sin(theta)

    x = l_xy_gpu[:, 0] * c - l_xy_gpu[:, 1] * sn
    y = l_xy_gpu[:, 0] * sn + l_xy_gpu[:, 1] * c
    z = s

    atoms = torch.stack([x, y, z], dim=-1)

    # Spine (Straight Z-axis)
    len_ang = fiber_params['length_ang']
    steps = torch.linspace(0, len_ang, SIM_CONFIG['Bending']['SPINE_INTERPOLATION_POINTS'], device=device)
    spine = torch.stack([torch.zeros_like(steps), torch.zeros_like(steps), steps], dim=-1)

    return atoms, spine


def apply_wlc_bending(l_xy, s_c, fiber_params, device):
    """Generate geometry for a fiber bent using Worm-Like Chain (WLC) model."""
    b_cfg = SIM_CONFIG['Bending']
    len_ang = fiber_params['length_ang']
    n_pts = b_cfg['SPINE_INTERPOLATION_POINTS']
    ds = len_ang / (n_pts - 1)

    # Random Walk for Spine orientation
    persistence = np.random.uniform(*b_cfg['PERSISTENCE_RANGE'])
    sigma = np.sqrt(ds / persistence)
    d_theta = torch.randn(n_pts, device=device) * sigma
    d_theta[0] = np.random.uniform(*b_cfg['INIT_ANGLE_RANGE'])
    theta = torch.cumsum(d_theta, dim=0)

    sp_x = torch.cumsum(torch.cos(theta) * ds, dim=0)
    sp_z = torch.cumsum(torch.sin(theta) * ds, dim=0)
    sp_y = torch.zeros_like(sp_x)
    spine = torch.stack([sp_x, sp_y, sp_z], dim=-1)

    # Map Atoms to Spine
    s = torch.as_tensor(s_c, device=device, dtype=torch.float32)
    l_xy_gpu = torch.as_tensor(l_xy, device=device, dtype=torch.float32)

    idx = s / ds
    idx_f = torch.floor(idx).long().clamp(0, n_pts - 2)
    w = (idx - idx_f).unsqueeze(1)

    c_pos = spine[idx_f] * (1 - w) + spine[idx_f + 1] * w
    alpha = theta[idx_f] * (1 - w.squeeze()) + theta[idx_f + 1] * w.squeeze()

    # Apply Twist & Rotation relative to spine
    twist_rad = (fiber_params['twist_deg'] * np.pi / 180.0) / fiber_params['rise_ang']
    theta_twist = s * twist_rad
    ct, st = torch.cos(theta_twist), torch.sin(theta_twist)

    x_tw = l_xy_gpu[:, 0] * ct - l_xy_gpu[:, 1] * st
    y_tw = l_xy_gpu[:, 0] * st + l_xy_gpu[:, 1] * ct

    ca, sa = torch.cos(alpha), torch.sin(alpha)
    fx = c_pos[:, 0] - x_tw * sa
    fz = c_pos[:, 2] + x_tw * ca
    fy = c_pos[:, 1] + y_tw

    return torch.stack([fx, fy, fz], dim=-1), spine


def apply_sine_wave_bending(l_xy, s_c, fiber_params, device):
    """Generate geometry for a fiber bent using a sine-wave model."""
    b_cfg = SIM_CONFIG['Bending']
    amp = np.random.uniform(*b_cfg['AMPLITUDE_RANGE'])
    periods = np.random.uniform(*b_cfg['PERIODS_RANGE'])
    phase = np.random.uniform(*b_cfg['PHASE_RANGE'])

    len_ang = fiber_params['length_ang']
    k = periods * 2.0 * np.pi / len_ang
    twist_rad = (fiber_params['twist_deg'] * np.pi / 180.0) / fiber_params['rise_ang']

    s = torch.as_tensor(s_c, device=device, dtype=torch.float32)
    l_xy_gpu = torch.as_tensor(l_xy, device=device, dtype=torch.float32)

    # Spine derivative
    arg = k * s + phase
    c_x = amp * torch.sin(arg)
    dx_ds = amp * k * torch.cos(arg)
    alpha = torch.atan(dx_ds)

    # Twist
    theta = s * twist_rad
    ct, st = torch.cos(theta), torch.sin(theta)
    x_tw = l_xy_gpu[:, 0] * ct - l_xy_gpu[:, 1] * st
    y_tw = l_xy_gpu[:, 0] * st + l_xy_gpu[:, 1] * ct

    ca, sa = torch.cos(alpha), torch.sin(alpha)
    fx = x_tw * ca + c_x
    fy = y_tw
    fz = -x_tw * sa + s

    atoms = torch.stack([fx, fy, fz], dim=-1)

    # Spine
    steps = torch.linspace(0, len_ang, b_cfg['SPINE_INTERPOLATION_POINTS'], device=device)
    sp_x = amp * torch.sin(k * steps + phase)
    spine = torch.stack([sp_x, torch.zeros_like(steps), steps], dim=-1)

    return atoms, spine


def upgrade_annotations(raw_points, step_pixels=15.0):
    """
    Resample line points with a fixed distance interval for uniform density.

    Args:
        raw_points: Original list of points from the projection.
        step_pixels: Distance (in pixels) between sampled points.

    Returns:
        resampled: List of [x, y] coordinates (float).
        edges: Simple connectivity list (though usually implicit in line order).
    """
    pts = np.array(raw_points).reshape(-1, 2)

    # Need at least 2 points to define a line
    if len(pts) < 2:
        return pts.tolist(), []

    # Calculate cumulative distance along the curve
    dists = np.sqrt(np.sum(np.diff(pts, axis=0) ** 2, axis=1))
    cum_dist = np.concatenate(([0], np.cumsum(dists)))
    total_len = cum_dist[-1]

    # Handle degenerate zero-length lines
    if total_len <= 1e-6:
        return pts.tolist(), []

    # Calculate number of points needed for the target step size
    num_points = max(2, int(np.ceil(total_len / step_pixels)))

    try:
        # Interpolate uniformly along the cumulative distance
        f = interp1d(cum_dist / total_len, pts, axis=0)
        resampled = f(np.linspace(0, 1, num_points))

        # Keep float precision (rounded to 2 decimals) for accurate centerline prediction
        resampled = np.round(resampled, 2).tolist()
    except ValueError:
        resampled = pts.tolist()

    # Generate sequential edges (0-1, 1-2, etc.)
    edges = [[i, i + 1] for i in range(len(resampled) - 1)]

    return resampled, edges


def generate_single_image_gpu(image_id, fiber_db, device):
    """
    Generate one synthetic image according to SIM_CONFIG.
    Uses Z-score normalization for physical consistency and guarantees structural noise.
    """
    cfg = SIM_CONFIG['Global']
    dset_cfg = SIM_CONFIG['Dataset']
    res = cfg['IMAGE_RESOLUTION']

    # === Config Base Values ===
    base_snr = cfg['SNR']
    base_fiber_min, base_fiber_max = dset_cfg['NUM_FIBERS_RANGE']
    base_df_min, base_df_max = cfg['DEFOCUS_RANGE_UM']
    base_apix_min, base_apix_max = cfg.get('PIXEL_SIZE_RANGE', (2.0, 4.0))

    # Randomized Parameters within Config Ranges
    num_fibers = random.randint(base_fiber_min, base_fiber_max)
    target_snr = base_snr
    defocus_val = random.uniform(base_df_min, base_df_max)
    target_apix = random.uniform(base_apix_min, base_apix_max)

    # Calculate physical bound (Field of View) using the dynamic target_apix
    # Higher Apix = Larger physical field of view mapped to same pixels = Fibers appear smaller
    bound = (res * target_apix) / 2.0

    # 1. Place Fibers
    atoms_all, spines_all = [], []

    for _ in range(num_fibers):
        f_type = random.choice(fiber_db)
        is_straight = np.random.rand() < SIM_CONFIG['Bending'].get('STRAIGHT_RATIO', 0.0)

        if is_straight:
            helix, spine = apply_straight_fiber(f_type['local_xy'], f_type['s_coords'], f_type, device)
        elif SIM_CONFIG['Bending']['MODE'] == 'wlc':
            helix, spine = apply_wlc_bending(f_type['local_xy'], f_type['s_coords'], f_type, device)
        else:
            helix, spine = apply_sine_wave_bending(f_type['local_xy'], f_type['s_coords'], f_type, device)

        # Random Pose
        rot = get_random_rotation_matrix(device)
        trans = (torch.rand(3, device=device) - 0.5) * (bound * 1.6)

        helix_c = helix - helix.mean(dim=0)
        spine_c = spine - helix.mean(dim=0)

        atoms_all.append(helix_c @ rot.T + trans)
        spines_all.append(spine_c @ rot.T + trans)

    # Fallback if empty (should be rare)
    if not atoms_all:
        return np.zeros((res, res), dtype=np.uint8), np.zeros((res, res), dtype=np.uint8), [], defocus_val

    # 2. Project Atoms to Grid
    combined = torch.cat(atoms_all, dim=0)
    idx = ((combined[:, :2] + bound) / (2.0 * bound) * res).long()

    mask = (idx[:, 0] >= 0) & (idx[:, 0] < res) & (idx[:, 1] >= 0) & (idx[:, 1] < res)
    idx = idx[mask]

    hist = torch.zeros((res, res), device=device)
    hist.index_put_((idx[:, 1], idx[:, 0]), torch.tensor(1.0, device=device), accumulate=True)

    # === Add structured artifacts ===
    # Probability is 1.0 to ensure every image has structural noise (robustness).
    trash_prob = 1.0
    trash_map = np.zeros((res, res), dtype=np.float32)
    has_trash = False

    # A. Aggregates (Ice contamination blobs)
    if np.random.rand() < trash_prob:
        has_trash = True
        for _ in range(np.random.randint(5, 30)):
            c = (np.random.randint(0, res), np.random.randint(0, res))
            r = np.random.randint(5, 20)
            cv2.circle(trash_map, c, r, (np.random.uniform(8.0, 18.0),), -1)

    # B. False Positives (Linear artifacts)
    if np.random.rand() < trash_prob:
        has_trash = True
        for _ in range(np.random.randint(5, 20)):
            p1 = (np.random.randint(0, res), np.random.randint(0, res))
            angle = np.random.uniform(0, 2 * np.pi)
            length = np.random.randint(20, 80)
            p2 = (int(p1[0] + length * np.cos(angle)), int(p1[1] + length * np.sin(angle)))
            cv2.line(trash_map, p1, p2, (np.random.uniform(2.0, 5.0),), np.random.randint(3, 8))

    # C. Carbon Edge (High contrast gradients)
    if np.random.rand() < (trash_prob * 0.5):
        has_trash = True
        p1 = (np.random.randint(-100, res + 100), np.random.randint(-100, res + 100))
        p2 = (np.random.randint(-100, res + 100), np.random.randint(-100, res + 100))
        cv2.line(trash_map, p1, p2, (np.random.uniform(5.0, 15.0),), np.random.randint(50, 300))

    if has_trash:
        hist += torch.tensor(trash_map, device=device)

    # 3. Simulate Physics (Gaussian Blur -> CTF -> Noise)
    sigma = (1.2 / target_apix)
    k = int(2 * 3 * sigma + 1) | 1
    hist = F.gaussian_blur(hist.unsqueeze(0), [k, k], [sigma]).squeeze(0)

    # CTF Calculation
    v = cfg['VOLTAGE_KV'] * 1000.0
    wl = (const.h / np.sqrt(2 * const.m_e * const.e * v * (1 + (const.e * v) / (2 * const.m_e * const.c ** 2)))) * 1e10

    freq = torch.fft.fftfreq(res, d=target_apix, device=device)
    ky, kx = torch.meshgrid(freq, freq, indexing='ij')
    k2 = fft.fftshift(kx ** 2 + ky ** 2)

    spec = fft.fftshift(fft.fft2(hist))
    ctf = calculate_ctf_gpu(k2, wl, defocus_val, device)
    img = fft.ifft2(fft.ifftshift(spec * ctf * torch.exp(-cfg['B_FACTOR'] * k2 / 4.0))).real

    # Add White Noise
    # Correct SNR logic: Calculate variance of the *entire* image (signal + artifacts + background contrast)
    if target_snr > 0:
        signal_power = img.var()
        if signal_power > 0:
            noise_std = torch.sqrt(signal_power / target_snr)
            img += torch.randn_like(img) * noise_std

    # === Normalization (Physically Consistent) ===
    # Use Z-score based clamping (Mean +/- 3 std) instead of Min-Max.
    # This prevents outliers from destroying contrast and maintains consistency across images.
    mu = img.mean()
    std = img.std()

    # Clip extreme values (e.g. ice contamination or carbon edges)
    clamp_min = mu - 3 * std
    clamp_max = mu + 3 * std
    img = torch.clamp(img, clamp_min, clamp_max)

    # Map clamped range to 0-255
    img_norm = (img - clamp_min) / (clamp_max - clamp_min + 1e-8)
    img = (img_norm * 255.0).to(torch.uint8)

    # 4. Generate Annotations (GT)
    mask_img = np.zeros((res, res), dtype=np.uint8)
    anns = []

    for sp in spines_all:
        # Get float coordinates for high precision (keep on CPU for numpy ops)
        pts_float = ((sp[:, :2] + bound) / (2 * bound) * res).cpu().numpy()

        # Simple bounds check
        valid_mask = (pts_float[:, 0] >= 0) & (pts_float[:, 0] < res) & \
                     (pts_float[:, 1] >= 0) & (pts_float[:, 1] < res)

        # Need enough points to form a line segment
        if valid_mask.sum() < 2:
            continue

        valid_pts = pts_float[valid_mask]

        # Draw mask for visualization (requires int)
        # Using a slightly wider line to represent fiber thickness roughly
        pts_int = valid_pts.round().astype(int)
        cv2.polylines(mask_img, [pts_int.reshape(-1, 1, 2)], False, (255,), 5)

        # Resample centerline for Skeleton prediction (Fixed step size)
        resampled, edges = upgrade_annotations(valid_pts, step_pixels=15.0)

        if len(resampled) < 2:
            continue

        # Calculate BBox from the resampled skeleton points
        # (Used as auxiliary info, but 'line' is the main target)
        x_pts = [p[0] for p in resampled]
        y_pts = [p[1] for p in resampled]
        x0, y0 = int(min(x_pts)), int(min(y_pts))
        w, h = int(max(x_pts) - x0), int(max(y_pts) - y0)

        anns.append({
            "id": 0,
            "image_id": image_id,
            "category_id": 1,
            "bbox": [x0, y0, max(1, w), max(1, h)],
            "area": w * h,
            "line": resampled,  # Contains float coordinates with uniform density
            "iscrowd": 0,
            "resampled_line": resampled,
            "edges": edges,
            "seed_point": resampled[0]
        })

    return img.cpu().numpy(), mask_img, anns, defocus_val


def generate_synthetic_dataset(output_root, total_images, rank=0, world_size=1):
    """
    Main loop to generate the synthetic dataset (train/val/test splits).
    """
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
    print(f"[Rank {rank}] Loading fibers...")

    if total_images is None:
        total_images = SIM_CONFIG['Dataset'].get('TOTAL_IMAGES', 200)

    fiber_db = []
    for f in SIM_CONFIG['FiberTypes']:
        d = precompute_fiber_atoms(f)
        if d:
            fiber_db.append(d)

    if not fiber_db:
        print("Error: No fibers loaded.")
        return False

    ratios = SIM_CONFIG['Dataset']['SPLIT_RATIOS']
    counts = {k: int(total_images * v) for k, v in ratios.items()}
    # Add remainder to train set
    counts["train"] += total_images - sum(counts.values())

    for split, n_total in counts.items():
        # Calculate local chunk size for distributed generation
        n_local = n_total // world_size + (1 if rank == world_size - 1 else 0) * (n_total % world_size)
        start_id = rank * (n_total // world_size)

        if n_local <= 0:
            continue

        base = f"{split}{SIM_CONFIG['Global']['YEAR']}"
        img_dir, mask_dir = os.path.join(output_root, "images", base), os.path.join(output_root, "masks", base)
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)
        os.makedirs(os.path.join(output_root, "annotations"), exist_ok=True)

        coco = {"images": [], "annotations": [], "categories": [{"id": 1, "name": "helix"}]}
        gid = start_id + 1
        aid = start_id * 20 + 1

        for _ in tqdm(range(n_local), desc=f"Rank {rank}|{split}", ascii=True):
            fname = f"{rank}_{gid:06d}.png"

            img, mask, anns, df = generate_single_image_gpu(gid, fiber_db, device)

            # Save Image (8-bit Grayscale)
            Image.fromarray(img, mode='L').save(os.path.join(img_dir, fname))
            # Save Mask
            cv2.imwrite(os.path.join(mask_dir, fname), mask)

            coco["images"].append({
                "id": gid,
                "file_name": fname,
                "width": SIM_CONFIG['Global']['IMAGE_RESOLUTION'],
                "height": SIM_CONFIG['Global']['IMAGE_RESOLUTION'],
                "defocus": float(df)
            })

            for a in anns:
                a["id"] = aid
                coco["annotations"].append(a)
                aid += 1
            gid += 1

        # Save per-rank annotation file
        jname = f"instances_{split}{SIM_CONFIG['Global']['YEAR']}_rank{rank}.json"
        with open(os.path.join(output_root, "annotations", jname), 'w') as f:
            json.dump(coco, f)

    # Wait for all processes if running in distributed mode
    if world_size > 1:
        dist.barrier()

    # Rank 0 merges all annotation files
    if rank == 0:
        print(">> Merging annotations...")
        for split in counts.keys():
            merged = {"info": {"year": 2025}, "images": [], "annotations": [],
                      "categories": [{"id": 1, "name": "helix"}]}
            for r in range(world_size):
                p = os.path.join(output_root, "annotations",
                                 f"instances_{split}{SIM_CONFIG['Global']['YEAR']}_rank{r}.json")
                if os.path.exists(p):
                    with open(p, 'r') as f:
                        d = json.load(f)
                        merged["images"].extend(d["images"])
                        merged["annotations"].extend(d["annotations"])
                    os.remove(p)

            final_p = os.path.join(output_root, "annotations",
                                   f"instances_{split}{SIM_CONFIG['Global']['YEAR']}.json")
            with open(final_p, 'w') as f:
                json.dump(merged, f)
            print(f"Saved {final_p}")

    return True


if __name__ == "__main__":
    # Example usage: Generate 100 images locally
    generate_synthetic_dataset("cryosim", 100)
