from ultralytics import YOLO


# select model
model = YOLO('runs/segment/train/weights/best.pt')

# modify result
results = model.predict('C:/Repositories/datasets/test/images/',
                        show=True,
                        save=True,
                        save_crop=True,
                        max_det=1,
                        retina_masks=True)

# preview result
for result in results:
    boxes = result.boxes  # Boxes object for bbox outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    probs = result.probs  # Class probabilities for classification outputs
