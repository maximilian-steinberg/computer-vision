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

    rotation = get_image_rotation(img_masked)
    img_rotated = ndimage.rotate(img_masked, rotation)
    mask_rotated = ndimage.rotate(mask, rotation)

    # Crop image
    rect = cv2.boundingRect(mask_rotated)
    img_cropped = img_rotated[rect[1]:(rect[1]+rect[3]), rect[0]:(rect[0]+rect[2])]

    # Preview result
    cv2.imshow("img_cropped", img_cropped)
    

cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(0)




    # crop image


    # save image



# modify result (with crops)
# results = model.predict('C:/Users/Dimster/Documents/6. Semester/Computer Vision/datasets/test/images',
#                         show=True,
#                         save=True,
#                         save_crop=True,
#                         max_det=1,
#                         retina_masks=True)

# preview result
# for result in results:
#     boxes = result.boxes  # Boxes object for bbox outputs
#     masks = result.masks  # Masks object for segmentation masks outputs
#     probs = result.probs  # Class probabilities for classification outputs
