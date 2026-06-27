# FPicker: Topology-Guided Evolution for Filament Tracing in Low-SNR Microscopy

[![Conference](https://img.shields.io/badge/ECCV-2026-blue.svg)](https://eccv2026.ecva.net/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This is the official PyTorch implementation of the **ECCV 2026** paper:  
**"FPicker: Topology-Guided Evolution for Filament Tracing in Low-SNR Microscopy"**  
*Tingyin Zhao, Mingtao Huang, Yuan Shen*  
*(Department of Electronic Engineering, Tsinghua University)*

---

## 🚀 News & Updates
- **[Coming Soon]** The full codebase, pre-trained weights, and the datasets will be released here prior to the ECCV 2026 conference. Stay tuned! 

## 📖 Abstract

Automating filament tracing in Cryo-Electron Microscopy (Cryo-EM) is essential for 3D helical reconstruction but challenged by intersecting topologies and extremely low Signal-to-Noise Ratios ($\text{SNR} = \sigma_s^2/\sigma_n^2$ < 0.1 or -10 dB). Existing paradigms fail:  pixel-wise segmenters suffer from severe topological fracturing, box-based detectors face ghost center drift, sequential trackers derail due to error accumulation, and traditional active contours collapse under artificial closed-curve constraints. To resolve these bottlenecks, we present \textbf{FPicker}, the first topology-guided framework reconciling these incompatibilities. It unifies perception via a center-endpoint representation and an open-curve evolution module to explicitly model non-cyclic connectivity. On simulated benchmarks, FPicker outperforms top baselines by over $40\%$ relative gain in mean spatio-angular precision (mSAP) and reduces topological gap rates by over $60\%$ under extreme noise ($-20\text{ dB}$). By learning intrinsic physical geometry rather than local texture, FPicker demonstrates strong potential as a resilient geometric backbone. Its zero-shot performance on the real-world EMPIAR dataset exhibits robust topological resistance, achieving a state-of-the-art 82.9\% mSAP upon fine-tuning. Our results also suggest modeling physical priors is a highly robust path toward bridging the sim-to-real gap in signal-starved scientific imaging.

## 🔗 Citation

If you find our work or this repository useful, please consider citing our paper:

```bibtex
@inproceedings{zhao2026fpicker,
  title={FPicker: Topology-Guided Evolution for Filament Tracing in Low-SNR Microscopy},
  author={Zhao, Tingyin and Huang, Mingtao and Shen, Yuan},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2026}
}
