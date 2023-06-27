from multiprocessing import freeze_support
from ultralytics import YOLO

# select model
# ----Settings------------------------------------------------
# (yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt)
# small & fast & unreliable -----------> big & slow & reliable
# Suffix -seg, -cls,
# ------------------------------------------------------------
model = YOLO('net/yolov8m-seg.pt')

if __name__ == '__main__':
    freeze_support()

    # train model on custom data
    model.train(data='Yolo_net\data.yaml', epochs=100, imgsz=640, device=0)

    # validate model
    model.val()

    # export model
    model.export(format='pt') # choose format
