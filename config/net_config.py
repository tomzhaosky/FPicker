"""
File: config/net_config.py
Author: Tingyin Zhao @ SGroup, EE, Tsinghua
Date: 2025-12-26
Description: Contains global configurations for model input sizes and training strategies.
"""

model_cfg = {
    'train_size': 1024,
    'test_size': 1024,

    'down_ratio': 4,

    'contour_len': 128,  # Number of points along the fiber contour
    'heads': {
        'hm': 1,       # Heatmap for fiber centers
        'reg': 2,      # Local center offset regression
        'ends': 4,     # Long-range endpoint regression (dx1, dy1, dx2, dy2)
        'aux': 3       # Auxiliary head (1 for Distance Field, 2 for Angle Field)
    },
}

train_strategy = {
    # Basic Training Hyperparameters
    'lr': 3.125e-5,
    'batch_size': 8,
    'max_epoch': 200,
    'warmup_epoch': 5,
    'lr_epoch': (120, 160),  # Epochs where learning rate decays

    # 3-Phase Bridge Strategy (Scheduled Sampling)
    'phase1_end': 10,  # Pure Teacher Forcing (Warm-up the contour)
    'phase2_end': 120,  # Mixed Sampling (Bridge the gap)
    # Beyond 'phase2_end': Pure End-to-End (Autonomous evolution)

    # Multi-task Loss Weights
    'lambda_hm': 1.0,  # Weight for Heatmap (Topological Centroid) loss
    'lambda_evolution': 1.0,  # Weight for contour evolution loss
    'lambda_aux': 0.1,  # Weight for Auxiliary (Distance/Angle Field) loss
    'lambda_ends': 0.05,  # Initial weight for endpoint regression (can be tuned)
}
