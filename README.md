# FPicker: Topology-Guided Evolution for Filament Tracing in Low-SNR Microscopy

[![Conference](https://img.shields.io/badge/ECCV-2026-blue.svg)](https://eccv2026.ecva.net/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status](https://img.shields.io/badge/Status-Coming%20Soon-orange.svg)]()

This is the official PyTorch implementation of the **ECCV 2026** paper:  
**"FPicker: Topology-Guided Evolution for Filament Tracing in Low-SNR Microscopy"**  
*Tingyin Zhao, Mingtao Huang, Yuan Shen*  
*(Tsinghua University)*

---

## 🚀 News & Updates
- **[Coming Soon]** The full codebase (including training/inference scripts), pre-trained weights, and the simulated/real-world datasets (Cryo-Sim & Custom-EMPIAR) will be released here prior to the ECCV 2026 conference. Stay tuned! 

## 📖 Abstract

Automating filament tracing in Cryo-Electron Microscopy (Cryo-EM) is essential for 3D helical reconstruction but challenged by intersecting topologies and extremely low Signal-to-Noise Ratios (SNR < 0.1 or -10 dB). Existing paradigms fail: pixel-wise segmenters suffer from severe topological fracturing, box-based detectors face ghost center drift, sequential trackers derail due to error accumulation, and traditional active contours collapse under artificial closed-curve constraints.

To resolve these bottlenecks, we present **FPicker**, the first topology-guided framework reconciling these incompatibilities. It unifies perception via a center-endpoint representation and an open-curve evolution module to explicitly model non-cyclic connectivity. On simulated benchmarks, FPicker outperforms top baselines by over 40% relative gain in mean spatio-angular precision (mSAP) and reduces topological gap rates by over 60% under extreme noise (-20 dB).

## 🔗 Citation

If you find our work or this repository useful, please consider citing our paper:

```bibtex
@inproceedings{zhao2026fpicker,
  title={FPicker: Topology-Guided Evolution for Filament Tracing in Low-SNR Microscopy},
  author={Zhao, Tingyin and Huang, Mingtao and Shen, Yuan},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2026}
}
