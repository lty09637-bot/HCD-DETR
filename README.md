# HCD-DETR

Official implementation of **HCD-DETR: A Hierarchically Calibrated and Dual-Domain DETR for Enhanced Real-Time Object Detection in UAV Imagery**.

## Paper Overview

UAV images contain complex backgrounds, large scale variation, and many small and densely distributed objects. These factors make it difficult for a general-purpose detector to preserve fine details while maintaining real-time performance. HCD-DETR extends RT-DETR with a feature-processing pipeline designed for this setting:

- **DDGM** (dual-domain denoising-guided module) uses spatial and frequency-domain information to suppress background interference and strengthen target representations.
- **HBCN** (hierarchical bidirectional calibration network) performs bidirectional cross-scale calibration and adaptive feature weighting for multi-scale fusion.
- **DDAD** (dynamic detail-aware downsampling) preserves spatial details during downsampling while improving semantic feature extraction through complementary branches.

The paper evaluates HCD-DETR on VisDrone2019, UAVDT, RSOD, and NWPU VHR-10. Results reported in the manuscript show improved detection performance and generalization with a compact real-time detector.

## Requirements

Install PyTorch for your CUDA environment first. Some commonly used dependencies are listed below; this is not a complete dependency list.

- `torch>=1.8.2`
- `torchvision>=0.9.2`
- `opencv-python>=4.6.0`
- `numpy>=1.18.5`
- `PyYAML>=5.3.1`
- `requests>=2.23.0`
- `scipy>=1.4.1`
- `psutil>=5.8.0` - system utilization
- `py-cpuinfo>=8.0.0` - CPU information
- `thop>=0.1.1` - FLOPs computation

## Citation

If you use this implementation, please cite:

```bibtex
@misc{zhang2026hcddetr,
  title  = {HCD-DETR: A Hierarchically Calibrated and Dual-Domain DETR for Enhanced Real-Time Object Detection in UAV Imagery},
  author = {Junsan Zhang and Tianyi Liu and Xiuxuan Shen and Ming Cheng and Zehan Jin and Zongquan Yao and Yao Zhang},
  year   = {2026},
  note   = {Manuscript}
}
```
