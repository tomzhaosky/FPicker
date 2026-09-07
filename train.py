"""
File: train.py
Author: Tingyin Zhao @ SGroup, EE, Tsinghua
Date: 2026-01-22
Description: Train codes for FPicker.
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from datetime import timedelta

from models.fpicker import FPicker
from backbone.resnet import resnet50
from backbone.dla import dla34
from backbone.swint import swin_t
from data.cryo_fiber import CryoFiberDataset
from utils.loss import EvolutionLoss

from utils.train_utils import ModelEMA
from utils.evaluate import get_length, compute_ap_ar, print_eval_matrix
from config.net_config import model_cfg, train_strategy
from config.data_config import SIM_CONFIG
from data.data_generate import generate_synthetic_dataset


def parse_args(args_list=None):
    parser = argparse.ArgumentParser(description='FPicker Training')
    parser.add_argument('--data_dir', default='./dataset', type=str)
    parser.add_argument('--batch_size', default=train_strategy.get('batch_size', 6), type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--lr', default=train_strategy.get('lr', 4e-5), type=float)
    parser.add_argument('--epochs', default=train_strategy.get('max_epoch', 200), type=int)
    parser.add_argument('--save_dir', default='./checkpoints', type=str)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--print_iter', default=10, type=int)
    parser.add_argument('--val_interval', default=1, type=int)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--gpus', default='0', type=str)
    parser.add_argument('--gen_data', action='store_true')
    parser.add_argument('--gen_num', default=200, type=int)
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--warmup_epochs', default=train_strategy.get('warmup_epoch', 3), type=int)
    parser.add_argument('--phase1_end', default=train_strategy.get('phase1_end', 50), type=int)
    parser.add_argument('--phase2_end', default=train_strategy.get('phase2_end', 150), type=int)

    parser.add_argument('--arch', default='resnet50',
                        choices=['resnet50', 'dla34', 'swin_t'],
                        help='Backbone architecture: resnet50 | dla34 | swin_t')

    parser.add_argument('--teach_path', default=None, type=str,
                        help='Path to pretrained model (loads weights only, resets optimizer)')

    if args_list:
        return parser.parse_args(args_list)
    return parser.parse_args()


def validate_simulation_inputs(rank=0):
    invalid_cifs = []
    for fiber in SIM_CONFIG.get('FiberTypes', []):
        name = fiber.get('name', 'unnamed_fiber')
        cif_path = fiber.get('cif_path', '')
        normalized = cif_path.replace('\\', '/')
        is_placeholder = (
            not cif_path
            or normalized.startswith('path/to/')
            or normalized.startswith('to/path/of/')
        )
        if is_placeholder or not os.path.exists(cif_path):
            invalid_cifs.append((name, cif_path, is_placeholder))

    if not invalid_cifs:
        return True

    if rank == 0:
        print("==> Synthetic data generation requires valid mmCIF file paths.")
        for name, cif_path, is_placeholder in invalid_cifs:
            status = "placeholder" if is_placeholder else "not found"
            print(f"    - {name}: {cif_path or '<empty>'} ({status})")
        print("==> Please edit config/data_config.py and set each")
        print("    SIM_CONFIG['FiberTypes'][i]['cif_path'] to a valid local mmCIF file")
        print("    before running train.py --gen_data.")
    return False


def train(opt):
    # Set CUDA visible devices if specified
    if opt.device == 'cuda' and opt.gpus is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus

    distributed = 'WORLD_SIZE' in os.environ or 'LOCAL_RANK' in os.environ
    if distributed:
        dist.init_process_group(backend='nccl', timeout=timedelta(minutes=120))
        l_rank = int(os.environ.get('LOCAL_RANK', 0))
        torch.cuda.set_device(l_rank)
        device = torch.device('cuda', l_rank)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        l_rank, device, rank, world_size = 0, torch.device('cuda' if torch.cuda.is_available() else 'cpu'), 0, 1

    if rank == 0:
        if not os.path.exists(opt.save_dir):
            os.makedirs(opt.save_dir, exist_ok=True)
            print(f"==> Created save directory: {opt.save_dir}")
        else:
            print(f"==> Save directory already exists: {opt.save_dir}")

    if opt.gen_data:
        if not validate_simulation_inputs(rank):
            if distributed:
                dist.destroy_process_group()
            sys.exit(1)
        generate_synthetic_dataset(opt.data_dir, opt.gen_num, rank=rank, world_size=world_size)

    if distributed:
        dist.barrier()

    # Load Data
    train_dataset = CryoFiberDataset(
        opt.data_dir,
        'train',
        contour_len=model_cfg.get('contour_len', 128)
    )
    train_sampler = DistributedSampler(train_dataset) if distributed else None

    def worker_init_fn(worker_id):
        np.random.seed(np.random.get_state()[1][0] + worker_id)

    train_loader = DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=opt.num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn
    )
    val_loader = DataLoader(CryoFiberDataset(opt.data_dir, 'val'), batch_size=opt.batch_size, shuffle=False)
    test_loader = DataLoader(CryoFiberDataset(opt.data_dir, 'test'), batch_size=1, shuffle=False)

    # Initialize Base Model (FPicker)
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

    optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=train_strategy.get('lr_epoch', (100, 150)),
        gamma=0.1
    )

    # TEACH BLOCK
    if opt.teach_path is not None and not opt.resume:
        if os.path.exists(opt.teach_path):
            if rank == 0:
                print(f"==> [TEACH] Loading pretrained weights from {opt.teach_path}")

            ckpt = torch.load(opt.teach_path, map_location=device, weights_only=False)
            state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt

            if any(k.startswith('module.') for k in state_dict.keys()):
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

            model.load_state_dict(state_dict, strict=True)

            if rank == 0:
                print("==> [TEACH] Weights loaded successfully. Optimizer & Scheduler RESET.")
        else:
            if rank == 0:
                print(f"==> [TEACH] Warning: File {opt.teach_path} not found! Training from scratch.")

    # RESUME BLOCK
    start_epoch = 0
    if opt.resume:
        path = os.path.join(opt.save_dir, 'model_last.pth')
        if os.path.exists(path):
            if rank == 0:
                print(f"==> Resuming training from {path}")
            ckpt = torch.load(path, map_location=device, weights_only=False)

            state_dict = ckpt['state_dict']
            if any(k.startswith('module.') for k in state_dict.keys()):
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

            model.load_state_dict(state_dict)
            optimizer.load_state_dict(ckpt['optimizer'])

            if 'scheduler' in ckpt:
                scheduler.load_state_dict(ckpt['scheduler'])
                if rank == 0:
                    print("==> Scheduler state loaded.")

            start_epoch = ckpt['epoch']

        else:
            if rank == 0:
                print(f"==> Warning: No checkpoint found at {path}. Starting from scratch.")

    if distributed:
        model = DDP(model, device_ids=[l_rank], output_device=l_rank)

    ema = ModelEMA(model) if rank == 0 else None
    criterion = EvolutionLoss(
        lambda_hm=train_strategy.get('lambda_hm', 20.0),
        lambda_evolution=train_strategy.get('lambda_evolution', 1.0),
        lambda_aux=train_strategy.get('lambda_aux', 0.1),
        lambda_ends=train_strategy.get('lambda_ends', 0.5)
    )

    epoch_size = len(train_loader)
    base_lr = opt.lr

    for epoch in range(start_epoch, opt.epochs):
        if distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        model.train()

        # --- Phase transition for Bridge Probability (P_gt) ---
        if epoch < opt.phase1_end:
            p_gt = 1.0
        elif epoch < opt.phase2_end:
            p_gt = 1.0 - (epoch - opt.phase1_end) / (opt.phase2_end - opt.phase1_end)
        else:
            p_gt = 0.0

        # --- Phase transition for Dynamic Loss Weights ---
        if epoch < 20:
            criterion.lambda_evolution = 0.0
            criterion.lambda_aux = 0.0
            criterion.lambda_hm = 2.5 * train_strategy.get('lambda_hm', 1.0)
        else:
            criterion.lambda_evolution = train_strategy.get('lambda_evolution', 1.0)
            criterion.lambda_aux = train_strategy.get('lambda_aux', 0.1)
            criterion.lambda_hm = train_strategy.get('lambda_hm', 1.0)

        for i, batch in enumerate(train_loader):
            t_start = time.time()
            if epoch < opt.warmup_epochs:
                cur_lr = base_lr * pow((i + epoch * epoch_size) / (opt.warmup_epochs * epoch_size), 4)
                for pg in optimizer.param_groups:
                    pg['lr'] = cur_lr
            elif epoch == opt.warmup_epochs and i == 0:
                for pg in optimizer.param_groups:
                    pg['lr'] = base_lr

            imgs = batch['image'].to(device)
            targets = [batch[k].to(device) for k in
                       ['hm', 'reg', 'ends', 'ind_masks', 'gt_df', 'gt_af', 'af_mask', 'gt_contours', 'contour_masks']]

            gt_contours = targets[7]
            contour_masks = targets[8]

            optimizer.zero_grad()
            m_core = model.module if distributed else model

            # FPicker Forward Call
            preds_dict = m_core(
                imgs,
                gt_contours=gt_contours,
                contour_masks=contour_masks,
                p_gt=p_gt,
                mode='train'
            )
            loss, stats = criterion(preds_dict, targets)

            if torch.isnan(loss):
                print("Warning: Loss is NaN, skipping iteration.")
                continue

            loss.backward()
            optimizer.step()

            if ema:
                ema.update(model)

            if rank == 0 and i % opt.print_iter == 0:
                progress = (i + 1) / epoch_size * 100
                log_info = (f"[Epoch {epoch + 1}/{opt.epochs}][Iter {i + 1}/{epoch_size} ({progress:.1f}%)]"
                            f"[P_gt {p_gt:.2f}][lr {optimizer.param_groups[0]['lr']:.6f}] "
                            f"Total Loss: {loss.item():.4f} ")
                loss_details = " | ".join([f"{k.upper()}: {v:.3f}" for k, v in stats.items()])
                print(f"{log_info} [{loss_details}] ({time.time() - t_start:.2f}s)")

        # E2E Validation
        if rank == 0 and (epoch + 1) % opt.val_interval == 0:
            eval_model = ema.ema if ema else model
            eval_model.eval()

            v_total = 0
            v_stats_acc = {'hm': 0.0, 'reg': 0.0, 'ends': 0.0, 'aux': 0.0, 'evolution': 0.0}

            with torch.no_grad():
                for v_batch in val_loader:
                    v_imgs = v_batch['image'].to(device)
                    v_t = [v_batch[k].to(device) for k in
                           ['hm', 'reg', 'ends', 'ind_masks', 'gt_df', 'gt_af', 'af_mask', 'gt_contours', 'contour_masks']]
                    core = eval_model.module if hasattr(eval_model, 'module') else eval_model

                    v_preds = core(
                        v_imgs,
                        gt_contours=v_t[7],
                        contour_masks=v_t[8],
                        p_gt=0.0,  # Auto-Regression
                        mode='train'
                    )

                    v_loss, v_stats_batch = criterion(v_preds, v_t)

                    v_total += v_loss.item()

                    for k, v in v_stats_batch.items():
                        if k in v_stats_acc:
                            v_stats_acc[k] += v

            num_v_batches = len(val_loader)
            avg_loss = v_total / num_v_batches if num_v_batches > 0 else 0
            avg_stats = {k: v / num_v_batches for k, v in v_stats_acc.items()} if num_v_batches > 0 else v_stats_acc

            loss_details = " | ".join([f"{k.upper()}: {v:.3f}" for k, v in avg_stats.items()])
            print(f"--- [VAL] Epoch {epoch + 1}: Total Loss {avg_loss:.4f} | [{loss_details}] ---")

        # Save Checkpoint
        if rank == 0:
            m_state = model.module.state_dict() if distributed else model.state_dict()
            save_path = os.path.join(opt.save_dir, 'model_last.pth')
            torch.save({
                'epoch': epoch + 1,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'state_dict': m_state,
                'ema_state_dict': ema.ema.state_dict() if ema else None
            }, save_path)

            if (epoch + 1) % 10 == 0:
                torch.save(m_state, os.path.join(opt.save_dir, f'model_epoch_{epoch + 1}.pth'))

        scheduler.step()

    # Final AP Evaluation
    if rank == 0:
        print("\nStarting Final Test Evaluation...")
        model_final = ema.ema if ema else model
        model_final.eval()

        all_preds, all_gts = [], []
        stride = 4
        input_sz = model_cfg.get('train_size', 512)
        top_k = 100

        with torch.no_grad():
            for batch in test_loader:
                core = model_final.module if hasattr(model_final, 'module') else model_final
                imgs = batch['image'].to(device)

                preds = core(imgs, mode='test')

                # Extract outputs from FPicker
                pred_s = preds['contours']  # [B, K, 128, 2]
                scores = preds['scores']  # [B, K]

                for b in range(imgs.size(0)):
                    iid = batch['meta']['img_id'][b].item()
                    sc = np.array([float(batch['meta']['orig_size'][b][0]) / input_sz,
                                   float(batch['meta']['orig_size'][b][1]) / input_sz])
                    sc_gpu = torch.from_numpy(sc).to(device)

                    # Collect predictions
                    if pred_s is not None:
                        # Ensure we don't access out of bounds if K < top_k
                        actual_k = pred_s.shape[1]
                        loop_k = min(top_k, actual_k)

                        for k in range(loop_k):
                            if scores[b, k] < 0.01:
                                continue
                            poly = (pred_s[b, k] * stride * sc_gpu).cpu().numpy()
                            all_preds.append({'img_id': iid, 'score': scores[b, k].item(), 'poly': poly})

                    # Collect ground truths
                    gs, ms = batch['gt_contours'][b], batch['contour_masks'][b]
                    for j in range(gs.size(0)):
                        if ms[j] > 0:
                            g_poly = (gs[j] * stride * torch.from_numpy(sc)).numpy()
                            all_gts.append({'img_id': iid, 'poly': g_poly, 'length': get_length(g_poly)})

        # Compute and print metrics
        results, thresholds = compute_ap_ar(all_preds, all_gts, top_k)
        print_eval_matrix(results, thresholds, top_k)


if __name__ == '__main__':
    options = parse_args()
    train(options)
