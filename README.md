# HCD-DETR

Official implementation of **HCD-DETR: A Hierarchically Calibrated and Dual-Domain DETR for Enhanced Real-Time Object Detection in UAV Imagery**.

## Paper Overview

UAV images contain complex backgrounds, large scale variation, and many small and densely distributed objects. These factors make it difficult for a general-purpose detector to preserve fine details while maintaining real-time performance. HCD-DETR extends RT-DETR with a feature-processing pipeline designed for this setting:

- **DDGM** (dual-domain denoising-guided module) uses spatial and frequency-domain information to suppress background interference and strengthen target representations.
- **HBCN** (hierarchical bidirectional calibration network) performs bidirectional cross-scale calibration and adaptive feature weighting for multi-scale fusion.
- **DDAD** (dynamic detail-aware downsampling) preserves spatial details during downsampling while improving semantic feature extraction through complementary branches.

The paper evaluates HCD-DETR on VisDrone2019, UAVDT, RSOD, and NWPU VHR-10. Results reported in the manuscript show improved detection performance and generalization with a compact real-time detector.

## Requirements

Install PyTorch for your CUDA environment first. Some commonly used dependencies are shown below; this is not a complete dependency list.

```bash
pip install torch torchvision opencv-python PyYAML numpy
```

## Citation

If you use this implementation, please cite:

> Junsan Zhang, Tianyi Liu, Xiuxuan Shen, Ming Cheng, Zehan Jin, Zongquan Yao, and Yao Zhang. *HCD-DETR: A Hierarchically Calibrated and Dual-Domain DETR for Enhanced Real-Time Object Detection in UAV Imagery*.
