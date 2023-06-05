from get_image_rotation import get_image_rotation
import cv2 as cv
import numpy as np
from scipy import ndimage
from ultralytics import YOLO

# Select model
model = YOLO('runs/segment/train/weights/best.pt')

# Read image 
img_path = 'C:/Users/Dimster/Desktop/TestImage.jpg'
img = cv.imread(img_path)

# Predict on input data 
results = model.predict(img,
                        max_det=1,
                        retina_masks=True)

# Get mask from result
mask = (results[0].masks.data[0].cpu().numpy() * 255).astype('uint8')

# Apply mask to image
img_masked = cv.bitwise_and(img, img, mask=mask)

# Crop image
rect = cv.boundingRect(mask)
img_cropped = img_masked[rect[1]:(rect[1]+rect[3]), rect[0]:(rect[0]+rect[2])]
mask_cropped = mask[rect[1]:(rect[1]+rect[3]), rect[0]:(rect[0]+rect[2])]

# Get image rotation
rotation = get_image_rotation(img_cropped)
img_rotated = ndimage.rotate(img_cropped, rotation)
mask_rotated = ndimage.rotate(mask_cropped, rotation)

# Crop result
rect_result = cv.boundingRect(mask_rotated)
img_cropped_further = img_rotated[rect_result[1]:(rect_result[1]+rect_result[3]), rect_result[0]:(rect_result[0]+rect_result[2])]

cv.imshow("Img cropped final", img_cropped_further)

# Preview result
cv.waitKey(0)