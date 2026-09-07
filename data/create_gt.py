"""
File: data/create_gt.py
Author: Tingyin Zhao @ SGroup, EE, Tsinghua
Date: 2025-12-27
Description: Generates multi-task training targets including Gaussian heatmaps, distance fields,
and angle fields for fiber extraction.
"""

import numpy as np
import cv2
import math


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)
    x, y = int(center[0]), int(center[1])
    height, width = heatmap.shape[0:2]
    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def gaussian_radius(det_size, min_overlap=0.7):
    height, width = det_size
    a1 = 1
    b1 = height + width
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2
    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2
    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return min(r1, r2, r3)


def resample_polyline(poly, num_points=128):
    """Resample polyline to equidistant points."""
    if len(poly) < 2:
        return np.zeros((num_points, 2), dtype=np.float32)

    diffs = poly[1:] - poly[:-1]
    dists = np.linalg.norm(diffs, axis=1)
    cum_dist = np.concatenate(([0], np.cumsum(dists)))
    total_len = cum_dist[-1]

    if total_len == 0:
        return np.resize(poly, (num_points, 2))

    target_dists = np.linspace(0, total_len, num_points)
    xs = np.interp(target_dists, cum_dist, poly[:, 0])
    ys = np.interp(target_dists, cum_dist, poly[:, 1])

    return np.stack([xs, ys], axis=1).astype(np.float32)


def proposal_target_generator(gt_labels, output_size, num_classes, stride=4, contour_len=128, max_fibers=30):
    """
    Generate targets for proposal, contour evolution, and auxiliary fields.
    """
    w, h = output_size

    # 1. Detection Targets
    hm = np.zeros((num_classes, h, w), dtype=np.float32)
    reg = np.zeros((2, h, w), dtype=np.float32)  # Center offset
    ends = np.zeros((4, h, w), dtype=np.float32)  # Endpoints: [dx1, dy1, dx2, dy2]
    ind_masks = np.zeros((h, w), dtype=np.float32)  # Valid center mask

    # 2. Aux Targets (DF & AF)
    gt_df = np.ones((1, h, w), dtype=np.float32) * 50.0  # Default bg distance
    gt_af = np.zeros((2, h, w), dtype=np.float32)  # [cos, sin]
    af_mask = np.zeros((1, h, w), dtype=np.float32)  # Valid AF region mask

    # 3. Contour Targets
    gt_contours = np.zeros((max_fibers, contour_len, 2), dtype=np.float32)
    contour_masks = np.zeros(max_fibers, dtype=np.float32)

    # Scale polygons
    scaled_polys = [np.array(ann['points']) / stride for ann in gt_labels]

    # --- Generate Aux Fields ---
    skeleton = np.zeros((h, w), dtype=np.uint8)

    polys_int = [p.astype(np.int32) for p in scaled_polys]
    cv2.polylines(skeleton, polys_int, False, (1,), 1)

    # Distance Field (DF)
    gt_df[0] = cv2.distanceTransform(1 - skeleton, cv2.DIST_L2, 5)

    # Angle Field (AF)
    for poly in scaled_polys:
        for i in range(len(poly) - 1):
            p1, p2 = poly[i], poly[i + 1]
            vec = p2 - p1
            length = np.linalg.norm(vec) + 1e-6
            direction = vec / length  # [cos, sin]

            pt1 = (int(p1[0]), int(p1[1]))
            pt2 = (int(p2[0]), int(p2[1]))

            color_cos = (float(direction[0]),)
            color_sin = (float(direction[1]),)

            cv2.line(gt_af[0], pt1, pt2, color_cos, 2)
            cv2.line(gt_af[1], pt1, pt2, color_sin, 2)
            cv2.line(af_mask[0], pt1, pt2, (1,), 2)

    # --- Generate Proposal & Contour Targets ---
    fiber_idx = 0
    for poly in scaled_polys:
        if fiber_idx >= max_fibers:
            break

        center = poly[len(poly)//2]
        ct_int = center.astype(np.int32)

        # Check if center is within bounds
        if not (0 <= ct_int[0] < w and 0 <= ct_int[1] < h):
            continue

        radius = 2
        draw_gaussian(hm[0], ct_int, radius)

        # Regression & Endpoints
        reg[:, ct_int[1], ct_int[0]] = center - ct_int
        ind_masks[ct_int[1], ct_int[0]] = 1
        ends[0:2, ct_int[1], ct_int[0]] = poly[0] - center
        ends[2:4, ct_int[1], ct_int[0]] = poly[-1] - center

        # Contour GT
        gt_contours[fiber_idx] = resample_polyline(poly, contour_len)
        contour_masks[fiber_idx] = 1

        fiber_idx += 1

    return hm, reg, ends, ind_masks, gt_df, gt_af, af_mask, gt_contours, contour_masks
