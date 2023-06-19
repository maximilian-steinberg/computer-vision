from get_image_rotation import get_image_rotation, get_approx, order_points, four_point_transform
import cv2 as cv
import imutils
import numpy as np
from scipy import ndimage
from ultralytics import YOLO

# Select model
model = YOLO('runs/segment/train/weights/best.pt')

# Read image 
img_path = 'C:/Users/Dimster/Desktop/TestImage5.jpg'
img = cv.imread(img_path)

# Predict on input data 
results = model.predict(img,
                        max_det=1,
                        retina_masks=True)

# Get mask from result
mask = (results[0].masks.data[0].cpu().numpy() * 255).astype('uint8')

# Apply mask to image
img_masked = cv.bitwise_and(img, img, mask=mask)

# Transform perspective
contours = cv.findContours(mask.copy(), cv.RETR_EXTERNAL,
	cv.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)
max_area = max(contours, key=cv.contourArea)
approx = get_approx(max_area)

approx_points = approx.ravel().reshape(-1, 2)

pointed_img = img.copy()
cv.drawContours(pointed_img, [approx], -1, (0, 255, 0), 3)

# Draw Red Cirles for Points
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
# ordered_points = order_points(approx_points)
# print(ordered_points)

output2 = img.copy()
warped_img = four_point_transform(output2, approx_points)

cv.imshow("Original", img)
cv.imshow("Pointed", pointed_img)
cv.imshow("Warped", warped_img)
cv.waitKey(0)
# rows,cols,ch = img.shape

# first_point = approx_points[1]
# second_point = approx_points[0]
# third_point = approx_points[2]

# pts1 = np.float32([first_point, second_point, third_point])
# pts2 = np.float32([first_point, [second_point[0],first_point[1]], [first_point[0],third_point[1]]])

# M = cv.getAffineTransform(pts1,pts2)
# dst = cv.warpAffine(img,M,(cols,rows))

# cv.imshow("Affined Image", dst)
# cv.waitKey(0)



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



# cv.imshow("Img cropped final", img_cropped_further)
cv.imshow("Rotated mask", img_rotated)
cv.waitKey(0)
cv.imshow("Rotated mask", mask_rotated)

# Preview result
cv.waitKey(0)
