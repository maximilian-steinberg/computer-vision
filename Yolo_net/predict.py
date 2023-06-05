import cv2 
import numpy as np
from scipy import ndimage
from ultralytics import YOLO
from get_image_rotation import get_image_rotation


test = True # Remove in production
img = None # Remove in production

# select model
model = YOLO('runs/segment/train/weights/best.pt')


if test: # Remove in production
    img = 'C:/Users/Dimster/Documents/6. Semester/Computer Vision/datasets/test/images' # Remove in production
else: # Remove in production
    # read image 
    img_path = 'C:/Users/Dimster/Desktop/TestImage.jpg'
    img = cv2.imread(img_path)


# predict on input data 
results = model.predict(img,
                        max_det=1,
                        retina_masks=True)


for result in results: 
# get mask from result
    mask = (result.masks.data[0].cpu().numpy() * 255).astype('uint8')
        # show mask
        # cv2.imshow('Mask', mask)

    # apply mask to imgage
    img = result.orig_img
    img_masked = cv2.bitwise_and(img, img, mask=mask)
        # show masked image
        # cv2.imshow('Masked', masked)

    # Crop image
    rect = cv2.boundingRect(mask)
    img_cropped = img_masked[rect[1]:(rect[1]+rect[3]), rect[0]:(rect[0]+rect[2])]
    mask_cropped = mask[rect[1]:(rect[1]+rect[3]), rect[0]:(rect[0]+rect[2])]

    # Get image rotation
    rotation = get_image_rotation(img_cropped)
    img_rotated = ndimage.rotate(img_cropped, rotation)
    mask_rotated = ndimage.rotate(mask_cropped, rotation)

    # Crop result
    rect_result = cv2.boundingRect(mask_rotated)
    img_cropped_further = img_rotated[rect_result[1]:(rect_result[1]+rect_result[3]), rect_result[0]:(rect_result[0]+rect_result[2])]

    # Preview result
    cv2.imshow("Img cropped final", img_cropped_further)
    cv2.waitKey(0)
    

cv2.destroyAllWindows()
cv2.waitKey(0)