"""
File: eval.py
Author: Tingyin Zhao @ SGroup, EE, Tsinghua
Date: 2025-12-24
Description: Evaluation codes for FPicker.
"""

import os
import sys
import argparse
from collections import OrderedDict
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np

try:
    torch.multiprocessing.set_sharing_strategy('file_system')
except RuntimeError:
    pass

from models.fpicker import FPicker
from backbone.resnet import resnet50
from backbone.dla import dla34
from backbone.swint import swin_t
from data.cryo_fiber import CryoFiberDataset
from utils.evaluate import get_length, compute_ap_ar, print_eval_matrix
from config.net_config import model_cfg


def parse_args(args_list=None):
    parser = argparse.ArgumentParser(description='FPicker Evaluation')
    parser.add_argument('--data_dir', default='./dataset', type=str)
    parser.add_argument('--model_path', default='./checkpoints/model_last.pth', type=str)

    parser.add_argument('--arch', default='resnet50',
                        choices=['resnet50', 'dla34', 'swin_t'],
                        help='Backbone architecture: resnet50 | dla34 | swin_t')

    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--top_k', default=150, type=int)
    parser.add_argument('--gpus', default='0', type=str)
    parser.add_argument('--local_rank', default=-1, type=int)
    return parser.parse_args(args_list) if args_list else parser.parse_args()


def evaluate(opt):
    # Setup GPU and distributed backend
    if opt.gpus:
        os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus

    distributed = 'WORLD_SIZE' in os.environ or 'LOCAL_RANK' in os.environ
    if distributed:
        dist.init_process_group(backend='nccl')
        l_rank = int(os.environ.get('LOCAL_RANK', 0))
        torch.cuda.set_device(l_rank)
        device = torch.device('cuda', l_rank)
        world_size = dist.get_world_size()
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        l_rank, world_size = 0, 1

    # Initialize model
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

    # Load checkpoint
    if l_rank == 0:
        print(f">> Loading checkpoint from {opt.model_path}...")

    ckpt = torch.load(opt.model_path, map_location=device, weights_only=False)

    # Handle EMA weights vs Standard weights
    if 'ema_state_dict' in ckpt and ckpt['ema_state_dict'] is not None:
        raw_state_dict = ckpt['ema_state_dict']
        if l_rank == 0:
            print(">> Loading EMA weights...")
    else:
        raw_state_dict = ckpt.get('state_dict', ckpt)
        if l_rank == 0:
            print(">> Loading standard weights...")

    # Clean state dict keys (remove 'module.' prefix)
    new_state_dict = OrderedDict()
    for k, v in raw_state_dict.items():
        name = k.replace('module.', '') if k.startswith('module.') else k
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)

    if distributed:
        model = DDP(model, device_ids=[l_rank])

    model.eval()

    # Data loading
    dataset = CryoFiberDataset(data_dir=opt.data_dir, split='test')
    sampler = DistributedSampler(dataset, shuffle=False) if distributed else None
    loader = DataLoader(
        dataset,
        batch_size=opt.batch_size,
        sampler=sampler,
        num_workers=opt.num_workers,
        pin_memory=True
    )

    all_preds, all_gts = [], []
    stride, input_sz = 4, model_cfg.get('train_size', 512)
    total_samples = len(dataset)
    print_interval = max(1, total_samples // 10)

    # Inference loop
    with torch.no_grad():
        for i, batch in enumerate(loader):
            imgs = batch['image'].to(device)
            m = model.module if distributed else model

            preds = m(imgs, mode='test')

            pred_s = preds['contours']
            scores = preds['scores']

            for b in range(imgs.size(0)):
                iid = batch['meta']['img_id'][b].item()
                # Compute scaling factors to restore original image coordinates
                sc = np.array([float(batch['meta']['orig_size'][b][0]) / input_sz,
                               float(batch['meta']['orig_size'][b][1]) / input_sz])

                sc_tensor_gpu = torch.from_numpy(sc).to(device)

                # Collect predictions
                for k in range(opt.top_k):
                    if scores[b, k] < 0.01:
                        continue

                    poly = (pred_s[b, k] * stride * sc_tensor_gpu).cpu().numpy()

                    all_preds.append({
                        'img_id': iid,
                        'score': scores[b, k].item(),
                        'poly': poly
                    })

                # Collect ground truths
                gs, ms = batch['gt_contours'][b], batch['contour_masks'][b]
                for j in range(gs.size(0)):
                    if ms[j] > 0:
                        valid_poly = gs[j]
                        g_poly = (valid_poly * stride * torch.from_numpy(sc)).numpy()

                        all_gts.append({
                            'img_id': iid,
                            'poly': g_poly,
                            'length': get_length(g_poly)
                        })

            # Progress logging
            samples_done = (i + 1) * opt.batch_size * world_size
            if l_rank == 0 and (
                    samples_done % print_interval < opt.batch_size * world_size or samples_done >= total_samples):
                print(f"[Eval: {min(samples_done, total_samples)} / {total_samples}]")

    # Compute metrics
    if l_rank == 0:
        results_matrix, threshold_list = compute_ap_ar(all_preds, all_gts, opt.top_k)
        print_eval_matrix(results_matrix, threshold_list, opt.top_k)


if __name__ == '__main__':
    options = parse_args()
    evaluate(options)
