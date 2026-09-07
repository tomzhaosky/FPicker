"""
File: config/data_config.py
Author: Tingyin Zhao @ SGroup, EE, Tsinghua
Date: 2026-01-01
Description: Configuration parameters for the Cryo-EM simulation pipeline.
             Defines physics settings, dataset splits, and fiber structural properties.
"""

import numpy as np

SIM_CONFIG = {
    # --- Microscope & Physics Settings ---
    "Global": {
        "IMAGE_RESOLUTION": 1024,
        "VOLTAGE_KV": 300.0,
        "SPHERICAL_ABBERATION_MM": 2.7,
        "AMPLITUDE_CONTRAST": 0.1,
        "DEFOCUS_RANGE_UM": (1.5, 3.0),
        "B_FACTOR": 150.0,
        "SNR": 0.05,
        "SCENE_PADDING_FACTOR": 2.0,

        "YEAR": "2025",

        # 2.0 = Standard Small
        # 4.0 = Very Tiny (Hard)
        "PIXEL_SIZE_RANGE": (2.0, 4.0)

    },

    # --- Dataset Generation Settings ---
    "Dataset": {
        "SPLIT_RATIOS": {"train": 0.8, "val": 0.1, "test": 0.1},
        "NUM_FIBERS_RANGE": (2, 10),
    },

    # --- Bending & Geometry ---
    "Bending": {
        # Ratio of completely straight fibers (0.0 to 1.0).
        # e.g., 0.5 means 50% of fibers will be straight, 50% will be bent (WLC/Sine).
        "STRAIGHT_RATIO": 0,

        # Bending Mode for non-straight fibers: 'wlc' (Recommended) or 'sine_wave'
        "MODE": "wlc",

        "SPINE_INTERPOLATION_POINTS": 2000,

        # [WLC Parameters]
        # Persistence Factor: Higher = Stiffer; Lower = More flexible.
        "PERSISTENCE_RANGE": (1000.0, 3000.0),
        "INIT_ANGLE_RANGE": (0.0, 2 * np.pi),

        # [Sine Wave Parameters]
        "AMPLITUDE_RANGE": (20.0, 150.0),
        "PERIODS_RANGE": (0.5, 2.5),
        "PHASE_RANGE": (0.0, 2 * np.pi)
    },

    # --- Fiber Types ---
    "FiberTypes": [
        {
            "name": "9hgr_standard",
            "cif_path": "path/to/fiber.cif",
            "helical_rise": 4.77,
            "helical_twist": -0.73,
            "num_units": 300,
        }
    ]
}
