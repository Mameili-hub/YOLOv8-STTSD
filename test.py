import sys
sys.path.append("/root/ultralytics")
from ultralytics import YOLO

if __name__ == '__main__':
# Load a model
  model = YOLO(r'yolov8n.pt')  # load an official model
  model.predict(
    data=r'',
    save=True,
    imgsz=640,
    conf=0.25,
    iou=0.45,
    show=False,
    project='runs/predict',
    name='exp',
    save_txt=False,
    save_conf=True,
    save_crop=False,
    show_labels=True,
    show_conf=True,
    vid_stride=1,
    line_width=3,
    visualize=False,
    augment=False,
    agnostic_nms=False,
    retina_masks=False,
    boxes=True,
  )
