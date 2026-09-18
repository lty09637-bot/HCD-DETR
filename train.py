import os
import warnings
from pathlib import Path

import torch

warnings.filterwarnings('ignore')
from hcd import RTDETR

ROOT = Path(__file__).resolve().parent
MODEL_CONFIG = ROOT / 'hcd/config/models/rt-detr/hcd-detr.yaml'


if __name__ == '__main__':
    data = os.getenv('HCD_DATA', 'data.yaml')
    device = os.getenv('HCD_DEVICE', '0' if torch.cuda.is_available() else 'cpu')

    model = RTDETR(str(MODEL_CONFIG))
    # model.load('/data1/liutianyi/RTDETR-main-2/runs/train/exp2/weights/best.pt') # loading pretrain weights
    model.train(data=data,
                cache=False,
                imgsz=int(os.getenv('HCD_IMGSZ', '640')),
                epochs=int(os.getenv('HCD_EPOCHS', '300')),
                batch=int(os.getenv('HCD_BATCH', '8')),
                workers=int(os.getenv('HCD_WORKERS', '4')),
                device=device,
                # resume='true', # last.pt path
                project=os.getenv('HCD_PROJECT', 'runs/lambda'),
                name=os.getenv('HCD_NAME', '去掉cfo-visdrone2019'),
                )
