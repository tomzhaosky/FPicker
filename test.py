"""
File: test.py
Author: Tingyin Zhao @ SGroup, EE, Tsinghua
Date: 2026-01-22
Description: Test/Visualization codes for FPicker.
"""

import os
import sys
import time
import argparse
from collections import OrderedDict
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import cv2
import numpy as np

try:
    torch.multiprocessing.set_sharing_strategy('file_system')
except RuntimeError:
    pass

# Modularized Imports
from models.fpicker import FPicker
from backbone.resnet import resnet50
from backbone.dla import dla34
from backbone.swint import swin_t
from data.cryo_fiber import CryoFiberDataset
from utils.visualize import draw_fiber_predictions
from config.net_config import model_cfg


def parse_args(args_list=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='./dataset', type=str)
    parser.add_argument('--model_path', default='./checkpoints/model_last.pth', type=str)

    parser.add_argument('--arch', default='resnet50',
                        choices=['resnet50', 'dla34', 'swin_t'],
                        help='Backbone architecture: resnet50 | dla34 | swin_t')

    parser.add_argument('--save_dir', default='./results/vis', type=str)
    parser.add_argument('--top_k', default=30, type=int)
    parser.add_argument('--gpus', default='0', type=str)
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--threshold', default=0.5, type=float)

    if args_list is not None:
        return parser.parse_args(args_list)
    return parser.parse_args()


def test(opt):
    # 1. GPU and Distributed Setup
    if opt.gpus:
        os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus
    distributed = 'WORLD_SIZE' in os.environ or 'LOCAL_RANK' in os.environ
    if distributed:
        dist.init_process_group(backend='nccl')
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        local_rank = 0

    # 2. Model Initialization and Weight Loading
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

    if local_rank == 0:
        print(f">> Loading checkpoint from {opt.model_path}...")

    checkpoint = torch.load(opt.model_path, map_location=device, weights_only=False)

    if 'ema_state_dict' in checkpoint and checkpoint['ema_state_dict'] is not None:
        raw_state_dict = checkpoint['ema_state_dict']
        if local_rank == 0:
            print(">> Using EMA weights.")
    else:
        raw_state_dict = checkpoint.get('state_dict', checkpoint)
        if local_rank == 0:
            print(">> Using standard weights.")

    new_state_dict = OrderedDict()
    for k, v in raw_state_dict.items():
        name = k.replace('module.', '') if k.startswith('module.') else k
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)

    if distributed:
        model = DDP(model, device_ids=[local_rank])
    model.eval()

    # 3. Dataset and Loader
    dataset = CryoFiberDataset(data_dir=opt.data_dir, split='test')
    sampler = DistributedSampler(dataset, shuffle=False) if distributed else None
    # Batch size is fixed to 1 for visualization scripts
    loader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=4)

    if local_rank == 0:
        os.makedirs(opt.save_dir, exist_ok=True)

    net_stride, input_size = 4, model_cfg.get('train_size', 512)

    # 4. Inference and Visualization Loop
    with torch.no_grad():
        for i, batch in enumerate(loader):
            start_time = time.time()
            imgs, meta = batch['image'].to(device), batch['meta']
            real_file_name = meta['file_name'][0]

            m = model.module if distributed else model

            preds = m(imgs, mode='test')

            pred_contours = preds['contours']
            scores = preds['scores']

            # Coordinate Restoration Logic
            orig_img_path = os.path.join(opt.data_dir, 'images', 'test2025', real_file_name)
            vis_img = cv2.imread(orig_img_path)

            # Calculate scaling factor based on original image size
            scale = torch.tensor(
                [float(meta['orig_size'][0][0]) / input_size, float(meta['orig_size'][0][1]) / input_size],
                device=device) if 'orig_size' in meta else torch.ones(2, device=device)

            if vis_img is None:
                # Fallback if image path is incorrect
                h_orig, w_orig = int(meta['orig_size'][0][1]), int(meta['orig_size'][0][0])
                vis_img = np.zeros((h_orig, w_orig, 3), dtype=np.uint8)

            # Convert predicted points to original image coordinates
            final_contours_np = (pred_contours * net_stride * scale).cpu().numpy()
            scores_np = scores[0].cpu().numpy()

            # 5. Use modularized visualization tool
            vis_img = draw_fiber_predictions(vis_img, final_contours_np[0], scores_np, threshold=opt.threshold)

            # 6. Save result
            save_path = os.path.join(opt.save_dir, f"{real_file_name.split('.')[0]}_res.jpg")
            cv2.imwrite(save_path, vis_img)

            if local_rank == 0:
                print(
                    f"[Test][Iter {i + 1}/{len(loader)}][Saved: {save_path} || time: {time.time() - start_time:.2f}s]")


if __name__ == '__main__':
    options = parse_args()
    test(options)
