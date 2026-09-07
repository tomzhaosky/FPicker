"""
File: demo.py
Author: Tingyin Zhao @ SGroup, EE, Tsinghua
Date: 2026-01-22
Description: Demo/Visualization codes for FPicker.
"""

import argparse
import sys
import torch
import cv2
import numpy as np
from collections import OrderedDict

try:
    torch.multiprocessing.set_sharing_strategy('file_system')
except RuntimeError:
    pass

from models.fpicker import FPicker
from backbone.resnet import resnet50
from backbone.dla import dla34
from backbone.swint import swin_t
from utils.visualize import draw_fiber_predictions
from config.net_config import model_cfg


def parse_args(args_list=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, required=True, help='Path to test image')
    parser.add_argument('--model_path', default='./checkpoints/model_last.pth', type=str)

    parser.add_argument('--arch', default='resnet50',
                        choices=['resnet50', 'dla34', 'swin_t'],
                        help='Backbone architecture: resnet50 | dla34 | swin_t')

    parser.add_argument('--output_path', default='demo_result.jpg', type=str)
    parser.add_argument('--threshold', default=0.5, type=float)
    return parser.parse_args(args_list) if args_list else parser.parse_args()


def demo(opt):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_size = model_cfg.get('train_size', 512)

    # Model Initialization (Use FPicker)
    print(f"==> Building model with backbone: {opt.arch}")

    if opt.arch == 'resnet50':
        backbone = resnet50(pretrained=True)
    elif opt.arch == 'dla34':
        backbone = dla34(pretrained=True, return_levels=True)
    elif opt.arch == 'swin_t':
        backbone = swin_t(pretrained=True)
    else:
        raise NotImplementedError(f"Backbone {opt.arch} not implemented!")

    model = FPicker(
        backbone=backbone,
        heads=model_cfg['heads'],
        head_conv=64,
        contour_len=model_cfg.get('contour_len', 128)
    ).to(device)

    # Weight Loading
    print(f">> Loading checkpoint from {opt.model_path}...")
    checkpoint = torch.load(opt.model_path, map_location=device)

    if 'ema_state_dict' in checkpoint and checkpoint['ema_state_dict'] is not None:
        raw_state_dict = checkpoint['ema_state_dict']
        print(">> Using EMA weights.")
    else:
        raw_state_dict = checkpoint.get('state_dict', checkpoint)
        print(">> Using standard weights.")

    new_state_dict = OrderedDict()
    for k, v in raw_state_dict.items():
        name = k.replace('module.', '') if k.startswith('module.') else k
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict, strict=True)
    model.eval()

    # Image Preprocessing
    img_orig = cv2.imread(opt.image_path)
    if img_orig is None:
        print(f"Error: Could not read image {opt.image_path}")
        return

    orig_h, orig_w = img_orig.shape[:2]

    # Resize to network input size
    img_resized = cv2.resize(img_orig, (input_size, input_size))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

    # Normalization (RGB Mean/Std)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape((1, 1, 3))
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape((1, 1, 3))

    inp = img_rgb.astype(np.float32) / 255.0
    inp = (inp - mean) / std
    inp = inp.transpose(2, 0, 1)  # [C, H, W]
    inp_tensor = torch.from_numpy(inp).unsqueeze(0).to(device)

    # Inference
    print(f"Running inference on {opt.image_path}...")
    with torch.no_grad():
        # One-Line Inference
        preds = model(inp_tensor, mode='test')

        # Extract Results
        pred_contours = preds['contours']  # [B, K, 128, 2]
        scores = preds['scores']  # [B, K]

    # Coordinate Restoration
    if pred_contours is not None:
        # Restore coordinates from feature-map scale to input-image scale
        pred_contours_input = pred_contours[0].cpu().numpy() * 4.0
        scores_np = scores[0].cpu().numpy()

        # Scale factors: network input -> original resolution (H, W)
        scale_x = orig_w / float(input_size)
        scale_y = orig_h / float(input_size)

        final_contours = pred_contours_input.copy()
        final_contours[..., 0] *= scale_x
        final_contours[..., 1] *= scale_y
    else:
        final_contours = np.array([])
        scores_np = np.array([])

    # Visualization
    canvas = draw_fiber_predictions(img_orig, final_contours, scores_np, threshold=opt.threshold)

    cv2.imwrite(opt.output_path, canvas)
    print(f"Visualization result saved to: {opt.output_path}")


if __name__ == '__main__':
    options = parse_args()
    demo(options)
