# HCD-DETR

Official implementation of **HCD-DETR: A Hierarchically Calibrated and Dual-Domain DETR for Enhanced Real-Time Object Detection in UAV Imagery**.

## Paper Overview

UAV images contain complex backgrounds, large scale variation, and many small and densely distributed objects. These factors make it difficult for a general-purpose detector to preserve fine details while maintaining real-time performance. HCD-DETR extends RT-DETR with a feature-processing pipeline designed for this setting:

- **DDGM** (dual-domain denoising-guided module) uses spatial and frequency-domain information to suppress background interference and strengthen target representations.
- **HBCN** (hierarchical bidirectional calibration network) performs bidirectional cross-scale calibration and adaptive feature weighting for multi-scale fusion.
- **DDAD** (dynamic detail-aware downsampling) preserves spatial details during downsampling while improving semantic feature extraction through complementary branches.

The paper evaluates HCD-DETR on VisDrone2019, UAVDT, RSOD, and NWPU VHR-10. Results reported in the manuscript show improved detection performance and generalization with a compact real-time detector.

## Repository Contents

```text
hcd/       HCD-DETR runtime, model definitions, and custom modules
train.py   Minimal training entry point
```

The paper model configuration is:

```text
hcd/config/models/rt-detr/hcd-detr-paper.yaml
```

## Quick Start

Prepare a Python environment with a PyTorch installation suitable for your hardware and the usual runtime packages required by the included Ultralytics-style codebase. This repository intentionally does not pin every dependency; use versions compatible with your CUDA/PyTorch setup.

Clone the repository and run the training entry point from its root:

```bash
git clone https://github.com/lty09637-bot/HCD-DETR.git
cd HCD-DETR
python train.py
```

`train.py` reads the dataset and run settings from environment variables. At minimum, set the path to a detection dataset YAML file:

```bash
export HCD_DATA=/path/to/data.yaml
export HCD_DEVICE=0          # use cpu for CPU training
export HCD_EPOCHS=300
export HCD_BATCH=8
python train.py
```

Optional variables include `HCD_IMGSZ`, `HCD_WORKERS`, `HCD_PROJECT`, and `HCD_NAME`. Training outputs are written below the selected project directory.

## Python API

The model can also be constructed directly:

```python
from hcd import RTDETR

model = RTDETR("hcd/config/models/rt-detr/hcd-detr-paper.yaml")
model.train(data="/path/to/data.yaml", epochs=300, imgsz=640, batch=8)
```

The dataset YAML should follow the standard object-detection format expected by the RT-DETR/Ultralytics data loader, including `train`, `val`, and class-name definitions.

## Citation

If you use this implementation, please cite:

> Junsan Zhang, Tianyi Liu, Xiuxuan Shen, Ming Cheng, Zehan Jin, Zongquan Yao, and Yao Zhang. *HCD-DETR: A Hierarchically Calibrated and Dual-Domain DETR for Enhanced Real-Time Object Detection in UAV Imagery*.

