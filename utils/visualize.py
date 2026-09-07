"""
File: utils/visualize.py
Author: Tingyin Zhao @ SGroup, EE, Tsinghua
Date: 2025-12-24
Description: Provides visualization functions to draw predicted fibers and confidence scores on original images.
"""

import cv2
import numpy as np
import colorsys


def get_distinct_color(index):
    """
    Generates a visually distinct color (BGR) based on the index.
    Uses the Golden Ratio to sample hues from the HSV color space,
    ensuring colors are distinct and non-repetitive.
    """
    # Golden ratio conjugate to disperse hues
    golden_ratio_conjugate = 0.618033988749895

    # Calculate Hue: varies with index to ensure difference
    hue = (index * golden_ratio_conjugate) % 1.0

    # High Saturation and Value for bright, vivid colors
    saturation = 0.85
    value = 0.95

    # Convert HSV to RGB (returns floats 0-1)
    r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)

    # Convert to BGR format (integers 0-255) for OpenCV
    return int(b * 255), int(g * 255), int(r * 255)


def draw_fiber_predictions(image, contours, scores, threshold=0.3, conf_flag=False, thickness=2):
    canvas = image.copy()

    if contours is None or len(contours) == 0:
        return canvas

    num_predictions = min(len(contours), len(scores))

    for i in range(num_predictions):
        if scores[i] < threshold:
            continue

        pts = np.round(contours[i]).astype(np.int32)

        pts = np.ascontiguousarray(pts).reshape((-1, 1, 2))

        # Get distinct color for this fiber
        color = get_distinct_color(i)

        # Draw the central axis (contour)
        cv2.polylines(canvas, [pts], False, color, thickness)

        if conf_flag:
            # Draw confidence score
            if len(pts) > 0:
                mid_idx = len(pts) // 2
                # pts shape is (N, 1, 2), so we need pts[mid_idx][0]
                txt_pos = tuple(pts[mid_idx][0])
                cv2.putText(canvas, f"{scores[i]:.2f}", txt_pos,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), thickness)

    return canvas
