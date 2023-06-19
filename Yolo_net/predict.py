from get_image_rotation import get_image_rotation, get_approx, order_points, four_point_transform
import cv2 as cv
import numpy as np
import os
import imutils
from scipy import ndimage
from ultralytics import YOLO
from get_image_rotation import get_image_rotation


test = True # Remove in production
img = None # Remove in production

directory = os.getcwd() + "/Results/"

# select model
model = YOLO('runs/segment/train/weights/best.pt')


if test: # Remove in production
    img = 'C:/Users/Dimster/Documents/6. Semester/Computer Vision/datasets/test/images' # Remove in production
else: # Remove in production
    # read image 
    img_path = 'C:/Users/Dimster/Desktop/TestImage.jpg'
    img = cv.imread(img_path)


# predict on input data 
results = model.predict(img,
                        max_det=1,
                        retina_masks=True)
names = model.names

for result in results: 

    cv.destroyAllWindows()
    # get mask from result
    mask = (result.masks.data[0].cpu().numpy() * 255).astype('uint8')
        # show mask
        # cv2.imshow('Mask', mask)

    # apply mask to imgage
    img = result.orig_img
    img_masked = cv.bitwise_and(img, img, mask=mask)
        # show masked image
        # cv2.imshow('Masked', masked)

    # Find the 4 label corners 
    contours = cv.findContours(mask.copy(), cv.RETR_EXTERNAL,
	cv.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    max_area = max(contours, key=cv.contourArea)
    approx = get_approx(max_area)
    approx_points = approx.ravel().reshape(-1, 2)
    ordered_points = order_points(approx_points)

    # Draw corners & countours of the label
    pointed_img = img.copy()
    cv.drawContours(pointed_img, [approx], -1, (0, 255, 0), 3)

    iteration = 1
    for (x,y) in approx_points:
        cv.circle(pointed_img, (x,y), radius=10, color=(0,0,255),thickness=-1)
        cv.putText(pointed_img, str(iteration), (x-5,y+5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        iteration += 1

    # Show Bounding Rect with points 
    (x, y, w, h) = cv.boundingRect(max_area)
    text = "original, num_pts={}".format(len(approx))
    cv.putText(pointed_img, text, (x, y - 15), cv.FONT_HERSHEY_SIMPLEX,
        0.9, (0, 255, 0), 2)


    # Specify top-left, top-right, bottom-right, and bottom-left order
    output2 = img.copy()
    warped_img = four_point_transform(output2, approx_points)

    # rotate to horizontal
    img_height = warped_img.shape[0]
    img_width = warped_img.shape[1]

    if img_width < img_height:
        warped_img = cv.rotate(warped_img, cv.ROTATE_90_CLOCKWISE)

    # Save image
    ## get image classification
    classification = names[int(result.boxes.cls[0])]

    ## get original image name
    filename = str(os.path.split(result.path)[1])
    print(filename)
    
    if not os.path.exists(directory + f"/{classification}"):
        os.mkdir(directory + f"/{classification}")


    # directory = f"C:/Users/Images/{classification}/"
    

    # if not os.path.exists(directory):
    #     os.umask(0)
    #     os.makedirs(directory, mode=0o666)

    os.chdir(directory + f"/{classification}")
    cv.imwrite(filename, warped_img)
    print(os.listdir(directory + f"/{classification}"))
    print('Successfully saved')

    # cv.imshow("Original", img)
    # cv.imshow("Pointed", pointed_img)
    # cv.imshow("Warped", warped_img)
    # cv.waitKey(0)

    # # Crop image
    # rect = cv2.boundingRect(mask)
    # img_cropped = img_masked[rect[1]:(rect[1]+rect[3]), rect[0]:(rect[0]+rect[2])]
    # mask_cropped = mask[rect[1]:(rect[1]+rect[3]), rect[0]:(rect[0]+rect[2])]

    # # Get image rotation
    # rotation = get_image_rotation(img_cropped)
    # img_rotated = ndimage.rotate(img_cropped, rotation)
    # mask_rotated = ndimage.rotate(mask_cropped, rotation)

    # # Crop result
    # rect_result = cv2.boundingRect(mask_rotated)
    # img_cropped_further = img_rotated[rect_result[1]:(rect_result[1]+rect_result[3]), rect_result[0]:(rect_result[0]+rect_result[2])]

    # # Preview result
    # cv2.imshow("Img cropped final", img_cropped_further)
    # cv2.waitKey(0)
    

cv.destroyAllWindows()
cv.waitKey(0)