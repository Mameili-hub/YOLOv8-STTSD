import gc
import torch
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
import sys
sys.path.append("/root/ultralytics")
from ultralytics import YOLO

if __name__ == '__main__':
# Load a model
  model = YOLO(r'yolov8-sttsd.yaml').load("yolov8n.pt") 

# Predict with the model
  model.train(
    
    data=r'',
    epochs=300,
    batch=16,
    amp=True,
    workers=8,
    optimizer='SGD',
    patience=50,
    imgsz=320,
    save=True,
    save_period=-1,
    cache=True,
    device='',
    project='runs/train',  
    name='exp',
    exist_ok=False,
    pretrained=True,
    verbose=True,
    seed=0,
    deterministic=True,
    single_cls=False,
    rect=False,
    cos_lr=False,
    close_mosaic=0,
    resume=False,
    fraction=1.0,
    profile=False,
    overlap_mask=True,
    mask_ratio=4,
    dropout=0.0,
    lr0=0.01,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3.0,
    warmup_momentum=0.8,
    warmup_bias_lr=0.1,
    box=7.5,
    cls=0.5,
    dfl=1.5,
    pose=12.0,
    kobj=1.0,
    label_smoothing=0.0,
    nbs=64,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,  
    degrees=0.0,
    translate=0.1,
    scale=0.5,
    shear=0.0,
    perspective=0.0,
    flipud=0.0,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.0,
    copy_paste=0.0,   
 )

  # 安全清理内存，防止 weakref 报错
  gc.collect()
  if torch.cuda.is_available():
      torch.cuda.empty_cache()
