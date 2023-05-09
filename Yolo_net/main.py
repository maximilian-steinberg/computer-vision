from ultralytics import YOLO

# select model
# ------------------------------------------------------------
# (yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt)
# small & fast & unreliable -----------> big & slow & reliable
# ------------------------------------------------------------
model = YOLO('yolov8m.pt')

# train model on custom data
model.train(data='Yolo_net\data.yaml', epochs=50, imgsz=640, device=0)

# validate model
model.val()

# test model
# inputs = 'Label_Detection.v2i.yolov8\test\images'
results = model.predict()

for result in results:
    boxes = result.boxes  # Boxes object for bbox outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    probs = result.probs  # Class probabilities for classification outputs
