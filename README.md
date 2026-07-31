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

~~~text
FPicker/
|-- configs/
|-- datasets/
|-- models/
|-- tools/
|-- utils/
|-- README.md
`-- LICENSE
~~~

## Installation

~~~bash
git clone https://github.com/tomzhaosky/FPicker.git
cd FPicker

conda create -n fpicker python=3.10
conda activate fpicker

pip install -r requirements.txt
~~~

## Dataset

The synthetic Cryo-EM datasets used by FPicker follow a COCO-style layout:

~~~text
cryosim_snr_0.05/
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
~~~

Each annotation file contains topology-aware filament annotations, including ordered centerline points, sequential edges, bounding boxes, seed points, and image-level simulation parameters such as defocus.

The datasets, annotations, masks, and pretrained weights are licensed under **CC BY-NC 4.0**. Commercial use is not permitted without prior written permission from the authors.

## Usage

Training and evaluation instructions will be provided with the full code release.

Example commands:

~~~bash
python train.py --config configs/fpicker.yaml
python test.py --config configs/fpicker.yaml --checkpoint path/to/checkpoint.pth
~~~

## Citation

If you find FPicker useful in your research, please cite:

~~~bibtex
@inproceedings{zhao2026fpicker,
  title={FPicker: Topology-Guided Evolution for Filament Tracing in Low-SNR Microscopy},
  author={Zhao, Tingyin and Huang, Mingtao and Shen, Yuan},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2026}
}
~~~

## License

This repository uses separate licenses for code and data-related assets.

**Source code:**  
The source code is licensed under the **GNU General Public License v3.0 (GPLv3)**. See [LICENSE](LICENSE) for details.

**Datasets, annotations, masks, and pretrained weights:**  
The datasets, annotations, masks, and pretrained weights are licensed under the **Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)**.

Commercial use of the datasets, annotations, masks, or pretrained weights is not permitted without prior written permission from the authors.
