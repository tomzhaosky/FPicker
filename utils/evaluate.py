"""
File: utils/evaluate.py
Author: Tingyin Zhao @ SGroup, EE, Tsinghua
Date: 2026-02-08
Description: Performance evaluation.
"""

import numpy as np


def chamfer_dist(p1, p2):
    """Computes bidirectional Chamfer distance."""
    d1 = np.min(np.linalg.norm(p1[:, None, :] - p2[None, :, :], axis=2), axis=1)
    d2 = np.min(np.linalg.norm(p2[:, None, :] - p1[None, :, :], axis=2), axis=1)
    return np.mean(d1) + np.mean(d2)


def compute_tangent_error(p1, p2):
    """Computes Mean Tangent Error theta_err with 180-degree symmetry."""

    def get_tangents(p):
        if len(p) < 2:
            return np.zeros((1, 2))
        v = np.diff(p, axis=0)
        norm = np.linalg.norm(v, axis=1, keepdims=True)
        return v / (norm + 1e-8)

    v1, v2 = get_tangents(p1), get_tangents(p2)
    m1 = np.mean(v1, axis=0)
    m1 /= (np.linalg.norm(m1) + 1e-8)
    m2 = np.mean(v2, axis=0)
    m2 /= (np.linalg.norm(m2) + 1e-8)

    cos_theta = np.abs(np.sum(m1 * m2))
    return np.degrees(np.arccos(np.clip(cos_theta, 0, 1)))


def get_length(points):
    """Calculates the cumulative arc-length of a filament."""
    if points.ndim > 2:
        points = points.reshape(-1, points.shape[-1])
    if len(points) < 2:
        return 0.0
    diffs = np.diff(points, axis=0)
    return np.sum(np.linalg.norm(diffs, axis=1))


def compute_ap_ar(all_results, all_gts, top_k, tau_theta=15.0, min_len=80.0):
    """
    Computes Spatio-Angular AP/AR with strict length grouping.
    Includes Fragmentation Analysis and Instance-based Gap Rate.
    """
    thresh_list = [5.0, 10.0, 15.0, 20.0, 25.0]

    # S/M/L buckets
    area_ranges = {
        'all': [80, 1e10],
        'small': [80, 200],
        'medium': [200, 400],
        'large': [400, 1e10]
    }

    max_dets_list = [1, 10, top_k]
    results = {}

    print(">> Pre-computing Distance and Angular matrices...")

    # Filter global noise first
    all_results = [p for p in all_results if get_length(p['poly']) >= min_len]

    # Pre-calculate lengths for filtering
    for p in all_results:
        if 'length' not in p:
            p['length'] = get_length(p['poly'])

    img_ids = set([g['img_id'] for g in all_gts] + [p['img_id'] for p in all_results])
    gts_by_img = {iid: [g for g in all_gts if g['img_id'] == iid] for iid in img_ids}
    preds_by_img = {
        iid: sorted(
            [p for p in all_results if p['img_id'] == iid],
            key=lambda x: x['score'], reverse=True
        ) for iid in img_ids
    }

    cache = {}

    # Accumulators for topology metrics
    total_cldice = []
    total_frag_index = []

    for iid in img_ids:
        p_list, g_list = preds_by_img[iid], gts_by_img[iid]
        d_mat = np.zeros((len(p_list), len(g_list)))
        ang_mat = np.zeros((len(p_list), len(g_list)))

        for g in g_list:
            if 'length' not in g:
                g['length'] = get_length(g['poly'])

        for i, p in enumerate(p_list):
            for j, g in enumerate(g_list):
                d_mat[i, j] = chamfer_dist(p['poly'], g['poly'])
                ang_mat[i, j] = compute_tangent_error(p['poly'], g['poly'])

        cache[iid] = {'preds': p_list, 'gts': g_list, 'dist_mat': d_mat, 'ang_mat': ang_mat}

        # --- Topology Metrics Calculation (clDice & Fragmentation) ---
        if g_list:
            tau_d = 5.0
            p_pts = np.concatenate([p['poly'] for p in p_list], axis=0) if p_list else None
            g_pts = np.concatenate([g['poly'] for g in g_list], axis=0)

            # 1. clDice (Raw)
            if p_pts is not None:
                dist_p2g = np.min(np.linalg.norm(p_pts[:, None, :] - g_pts[None, :, :], axis=2), axis=1)
                t_prec = np.mean(dist_p2g < tau_d)
                dist_g2p = np.min(np.linalg.norm(g_pts[:, None, :] - p_pts[None, :, :], axis=2), axis=1)
                t_sens = np.mean(dist_g2p < tau_d)
                total_cldice.append(2 * t_prec * t_sens / (t_prec + t_sens + 1e-8))
            else:
                total_cldice.append(0.0)

            # 2. Fragmentation Index
            if len(p_list) > 0:
                # Count how many preds are associated with each GT (approximate via distance)
                is_close = d_mat < 5.0
                preds_per_gt = np.sum(is_close, axis=0)

                # Only average over GTs that were detected at least once
                matched_counts = preds_per_gt[preds_per_gt > 0]
                if len(matched_counts) > 0:
                    total_frag_index.append(np.mean(matched_counts))
                else:
                    total_frag_index.append(1.0)  # Neutral if nothing detected
            else:
                total_frag_index.append(1.0)

    # Initialize Extra Metrics
    mean_cldice = np.mean(total_cldice) if total_cldice else 0.0
    mean_frag = np.mean(total_frag_index) if total_frag_index else 1.0
    # Strict clDice: penalize by fragmentation
    gamma = 0.1
    safe_frag = max(1.0, mean_frag)
    penalized_cldice = mean_cldice * np.exp(-gamma * (safe_frag - 1))

    results['extra'] = {
        'clDice': mean_cldice,
        'FragIndex': mean_frag,
        'clDice_Penalized': penalized_cldice,
        # 'GapRate' will be calculated after AR is computed
    }

    print(">> Calculating AP/AR with Spatio-Angular constraint...")

    # Store AR values to calculate Gap Rate later
    ar_values_for_gap_rate = []

    for t in thresh_list:
        for area_name, a_range in area_ranges.items():
            for m_det in max_dets_list:
                total_gt_in_range, tp_list, fp_list, score_list = 0, [], [], []
                image_theta_errors = []  # Only for current threshold/setting

                for iid in img_ids:
                    data = cache[iid]
                    p_list_all, g_list = data['preds'], data['gts']
                    d_mat, ang_mat = data['dist_mat'], data['ang_mat']

                    # 1. Filter GTs by range
                    valid_gt_indices = [
                        j for j, g in enumerate(g_list)
                        if a_range[0] <= g['length'] < a_range[1]
                    ]
                    total_gt_in_range += len(valid_gt_indices)

                    # 2. Filter Preds by range
                    valid_pred_indices = [
                        i for i, p in enumerate(p_list_all)
                        if a_range[0] <= p['length'] < a_range[1]
                    ]
                    # Sort by score
                    valid_pred_indices = sorted(valid_pred_indices, key=lambda p: p_list_all[p]['score'], reverse=True)
                    active_pred_indices = valid_pred_indices[:m_det]

                    gt_matched = {j: False for j in valid_gt_indices}

                    for i in active_pred_indices:
                        best_d, best_gi = t, -1
                        for gi in valid_gt_indices:
                            if gt_matched[gi]:
                                continue
                            if d_mat[i, gi] < best_d and ang_mat[i, gi] < tau_theta:
                                best_d, best_gi = d_mat[i, gi], gi

                        if best_gi != -1:
                            tp_list.append(1)
                            fp_list.append(0)
                            gt_matched[best_gi] = True
                            image_theta_errors.append(ang_mat[i, best_gi])
                        else:
                            tp_list.append(0)
                            fp_list.append(1)
                        score_list.append(p_list_all[i]['score'])

                    # If this is the main reporting config (t=5, all, top_k), calculate strict error
                    if t == 5.0 and area_name == 'all' and m_det == top_k:
                        # Find missed GTs
                        missed_count = len([gi for gi in valid_gt_indices if not gt_matched[gi]])
                        # Penalize missed GTs with 90 degrees
                        image_theta_errors.extend([90.0] * missed_count)

                # Save Theta Error to extra if this is the standard setting
                if t == 5.0 and area_name == 'all' and m_det == top_k:
                    results['extra']['theta_err'] = (
                        np.mean(image_theta_errors) if image_theta_errors else 90.0
                    )

                if total_gt_in_range == 0:
                    results[(t, area_name, m_det)] = {'ap': 0.0, 'ar': 0.0}
                    # Also record 0 AR for Gap Rate calculation
                    if area_name == 'all' and m_det == top_k:
                        ar_values_for_gap_rate.append(0.0)
                    continue

                if not score_list:
                    results[(t, area_name, m_det)] = {'ap': 0.0, 'ar': 0.0}
                    if area_name == 'all' and m_det == top_k:
                        ar_values_for_gap_rate.append(0.0)
                    continue

                stack = sorted(zip(score_list, tp_list, fp_list), key=lambda x: x[0], reverse=True)
                tp, fp = np.array([x[1] for x in stack]), np.array([x[2] for x in stack])

                rec = np.cumsum(tp) / total_gt_in_range
                prec = np.cumsum(tp) / (np.cumsum(tp) + np.cumsum(fp) + 1e-6)

                p_interp = [
                    np.max(prec[rec >= q]) if any(rec >= q) else 0
                    for q in np.linspace(0, 1, 101)
                ]

                final_ap = np.mean(p_interp)
                final_ar = np.max(rec)

                results[(t, area_name, m_det)] = {'ap': final_ap, 'ar': final_ar}

                # Record AR for Strict Gap Rate (using max_dets=top_k, area='all')
                if area_name == 'all' and m_det == top_k:
                    ar_values_for_gap_rate.append(final_ar)

    # Gap Rate = 1.0 - Mean Average Recall (mAR) over all thresholds
    if ar_values_for_gap_rate:
        mean_instance_recall = np.mean(ar_values_for_gap_rate)
        results['extra']['GapRate'] = 1.0 - mean_instance_recall
    else:
        results['extra']['GapRate'] = 1.0

    return results, thresh_list


def print_eval_matrix(evaluation_results, threshold_list, max_detection_count):
    """Prints AP/AR table with Strict Metrics."""

    def get_mean(metric, area, m_det):
        vals = [
            evaluation_results.get((t, area, m_det), {}).get(metric, 0.0)
            for t in threshold_list
        ]
        return np.mean(vals) if vals else 0.0

    range_str = f"Chamfer={int(min(threshold_list))}:{int(max(threshold_list))}"
    title = "FINAL EVALUATION MATRIX (Strict: AR-GapRate, Penalized Metrics)"
    print("\n" + "=" * 60 + "\n" + " " * 10 + title + "\n" + "=" * 60)

    for metric_name, label in [('ap', 'Precision'), ('ar', 'Recall')]:
        for area_level in ['all', 'small', 'medium', 'large']:
            if area_level == 'all' and metric_name == 'ar':
                for dl in [1, 10, max_detection_count]:
                    v = get_mean('ar', 'all', dl)
                    print(
                        f" Average Recall    (AR) @[ {range_str:<14} | "
                        f"len={area_level:6s} | maxDets={dl:2d} ] = {v:.3f}"
                    )
            else:
                v = get_mean(metric_name, area_level, max_detection_count)
                print(
                    f" Average {label:9s} ({metric_name.upper()}) @[ {range_str:<14} | "
                    f"len={area_level:6s} | maxDets={max_detection_count:2d} ] = {v:.3f}"
                )

    extra = evaluation_results.get('extra', {})
    print("-" * 60)
    print(f" Topological Fidelity (clDice)     : {extra.get('clDice', 0.0):.4f}")
    print(
        f" Frag. Penalty        (clDice/Frag): {extra.get('clDice_Penalized', 0.0):.4f} (Raw: {extra.get('clDice', 0.0):.4f}, Frag: {extra.get('FragIndex', 1.0):.2f})")
    print(f" Geometric Precision  (theta_err)  : {extra.get('theta_err', 90.0):.3f} deg (Strict: Missed=90deg)")
    print(f" Continuity Analysis  (Gap Rate)   : {extra.get('GapRate', 1.0):.4f} (Strict: 1 - mAR)")
    print("=" * 60)
