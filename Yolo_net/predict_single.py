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
cv.imshow("img_masked", img_masked)

# Get image rotation
rotation = get_image_rotation(img_masked)
img_rotated = ndimage.rotate(img_masked, rotation)
mask_rotated = ndimage.rotate(mask, rotation)

cv.imshow("img_rotated", img_rotated)
cv.imshow("mask_rotated", mask_rotated)

# Crop image
rect = cv.boundingRect(mask_rotated)
img_cropped = img_rotated[rect[1]:(rect[1]+rect[3]), rect[0]:(rect[0]+rect[2])]

# Preview result
cv.imshow("img_cropped", img_cropped)
cv.waitKey(0)