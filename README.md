# FPicker: Topology-Guided Evolution for Filament Tracing in Low-SNR Microscopy

[![Conference](https://img.shields.io/badge/ECCV-2026-blue.svg)](https://eccv2026.ecva.net/)
[![Code License: GPLv3](https://img.shields.io/badge/Code%20License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Dataset License: CC BY-NC 4.0](https://img.shields.io/badge/Dataset%20License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

This is the official PyTorch implementation of the **ECCV 2026** paper:

**FPicker: Topology-Guided Evolution for Filament Tracing in Low-SNR Microscopy**  
*Tingyin Zhao, Mingtao Huang, Yuan Shen*  
Department of Electronic Engineering, Tsinghua University

The source code is released under the **GNU General Public License v3.0 (GPLv3)**.  
The datasets, annotations, masks, and pretrained weights are released for **non-commercial research use** under the **Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)**.

## News

- **Coming soon:** The full codebase, pretrained weights, and synthetic Cryo-EM datasets will be released.

## Abstract

Automating filament tracing in Cryo-Electron Microscopy (Cryo-EM) is essential for 3D helical reconstruction but remains challenging under intersecting topologies and extremely low signal-to-noise ratios. Existing paradigms suffer from severe topological fragmentation, center drift, error accumulation, or artificial closed-curve constraints.

To address these challenges, we propose **FPicker**, a topology-guided framework for filament tracing in low-SNR microscopy. FPicker unifies perception through a center-endpoint representation and an open-curve evolution module, explicitly modeling non-cyclic filament connectivity. On simulated low-SNR Cryo-EM benchmarks, FPicker achieves strong tracing accuracy and improved topological continuity. It also demonstrates robust transfer potential on real-world Cryo-EM data.

## Repository Structure

```text
FPicker/
|-- backbone/              # Backbone networks: ResNet, DLA, and Swin Transformer
|-- config/                # Model, training, and synthetic data generation configs
|-- data/                  # Dataset loader, augmentation, target generation, and simulator
|-- models/                # FPicker model, decoder, CenterNet-style head, and snake module
|-- utils/                 # Losses, training utilities, evaluation, and visualization
|-- demo.py                # Single-image inference demo
|-- eval.py                # Quantitative evaluation on the test split
|-- test.py                # Test-set inference and visualization
|-- train.py               # Training and optional synthetic data generation
|-- README.md
`-- LICENSE
```

Before public release, remove local runtime cache folders such as `__pycache__/`.

## Installation

```bash
git clone https://github.com/tomzhaosky/FPicker.git
cd FPicker

conda create -n fpicker python=3.10
conda activate fpicker
```

Install PyTorch following the CUDA version on your machine, then install the remaining dependencies:

```bash
pip install numpy opencv-python scipy pillow biopython tqdm torchvision
```

If a `requirements.txt` file is provided in a later release, use:

```bash
pip install -r requirements.txt
```

## Dataset

FPicker expects a COCO-style synthetic Cryo-EM dataset with the following layout:

```text
cryosim/
|-- annotations/
|   |-- instances_train2025.json
|   |-- instances_val2025.json
|   `-- instances_test2025.json
|-- images/
|   |-- train2025/
|   |-- val2025/
|   `-- test2025/
`-- masks/
    |-- train2025/
    |-- val2025/
    `-- test2025/
```

Each annotation file contains topology-aware filament annotations, including ordered centerline points, sequential edges, bounding boxes, seed points, and image-level simulation parameters such as defocus.

The default dataset path in the scripts is `./cryosim`. If you use the current release layout where `FPicker/` is placed next to `annotations/`, `images/`, and `masks/`, run commands from inside `FPicker/` with:

```bash
--data_dir ..
```

## Training

Train FPicker with one GPU:

```bash
python train.py --data_dir ./cryosim --arch resnet50 --gpus 0 --save_dir ./checkpoints
```

If your dataset is the parent folder of `FPicker/`, use:

```bash
python train.py --data_dir .. --arch resnet50 --gpus 0 --save_dir ./checkpoints
```

Supported backbones:

```text
resnet50 | dla34 | swin_t
```

Resume from the last checkpoint:

```bash
python train.py --data_dir ./cryosim --resume --save_dir ./checkpoints
```

Initialize from pretrained weights while resetting the optimizer and scheduler:

```bash
python train.py --data_dir ./cryosim --teach_path path/to/pretrained.pth
```

Generate a small synthetic dataset before training:

```bash
python train.py --gen_data --gen_num 200 --data_dir ./cryosim
```

## Evaluation

Run quantitative evaluation on the test split:

```bash
python eval.py --data_dir ./cryosim --model_path ./checkpoints/model_last.pth --arch resnet50
```

## Visualization

Run test-set inference and save visualized predictions:

```bash
python test.py --data_dir ./cryosim --model_path ./checkpoints/model_last.pth --save_dir ./results/vis --threshold 0.5
```

Run a single-image demo:

```bash
python demo.py --image_path path/to/image.png --model_path ./checkpoints/model_last.pth --output_path demo_result.jpg
```

## Citation

If you find FPicker useful in your research, please cite:

```bibtex
@inproceedings{zhao2026fpicker,
