"""" Predict
    Functions for image prediction.
"""

import os
import argparse
from pathlib import Path
from predict_extra import get_approx, four_point_transform, is_image_file
import cv2 as cv
import imutils
from ultralytics import YOLO

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, required=True, help='Path to the images')
args = parser.parse_args()

# Select model
model = YOLO('net/best.pt')
NAMES = model.names

def predict_all(path):
    """
    Predicts the output for all images in the specified folder.

    Parameters
    ----------
    path : str
        The path to the folder containing the images.

    Returns
    -------
    None
        This function does not return any value.

    Raises
    ------
    ValueError
        If the specified path is invalid or does not exist.

    Notes
    -----
    This function iterates over all the image files in the specified folder,
    calls the `predict_single` function for each image, and performs the prediction.
    The `predict_single` function takes the path of each image file as an argument
    and performs the prediction for that image.

    Example
    -------
    >>> python predict.py --path "C:/Users/Dimster/Documents/6. Semester/Computer Vision/datasets/test/images"
    """

    # Get a list of files in the specified folder
    files = os.listdir(path)

    # Filter the list to include only image files
    image_files = [file for file in files if is_image_file(os.path.join(path, file))]

    # Iterate over the image files and call the process_image function
    for image_file in image_files:
        image_path = os.path.join(path, image_file)
        predict_single(image_path)

def predict_single(path):
    """
    Predicts and processes a single image for label recognition.

    Args:
        path (str): The file path of the input image.

    Returns:
        None
    """

    # Get current directory
    directory = Path(os.getcwd()) / "Results/"

    # Read image 
    img = cv.imread(path)

    # Predict on input data 
    results = model.predict(img,
                        max_det=1,
                        retina_masks=True)

    # get mask from result
    mask = (results[0].masks.data[0].cpu().numpy() * 255).astype('uint8')


    # Find the label countours and the 4 label corners 
    contours = cv.findContours(mask.copy(), cv.RETR_EXTERNAL,
	cv.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    max_area = max(contours, key=cv.contourArea)
    approx_points = get_approx(max_area)

    # Draw corners & countours of the label
    pointed_img = img.copy()
    cv.drawContours(pointed_img, [approx_points], -1, (0, 255, 0), 3)
    iteration = 1
    for (x,y) in approx_points:
        cv.circle(pointed_img, (x,y), radius=10, color=(0,0,255),thickness=-1)
        cv.putText(pointed_img, str(iteration), (x-5,y+5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        iteration += 1

    # Put text to the pointed image with points number 
    (x, y, w, h) = cv.boundingRect(max_area)
    text = "original, num_pts={}".format(len(approx_points))
    cv.putText(pointed_img, text, (x, y - 15), cv.FONT_HERSHEY_SIMPLEX,
        0.9, (0, 255, 0), 2)

    # Wrap the image
    output2 = img.copy()
    warped_img = four_point_transform(output2, approx_points)

    # rotate to horizontal if the width is smaller than height
    img_height = warped_img.shape[0]
    img_width = warped_img.shape[1]
    if img_width < img_height:
        warped_img = cv.rotate(warped_img, cv.ROTATE_90_CLOCKWISE)


    # Save image
    ## get image classification
    classification = NAMES[int(results[0].boxes.cls[0])]

    ## get original image name
    filename = str(os.path.split(path)[1])

    ## check if the classification directory already exists
    ## if not, make a new one  
    if not os.path.exists(directory / f"{classification}"):
        os.mkdir(directory / f"{classification}")

    ## save the image in the corresponding directory
    os.chdir(directory / f"{classification}")
    cv.imwrite(filename, warped_img)

    os.chdir("../..")

    # FOR DEBUG - Show image preview
    # cv.imshow("Original", img)
    # cv.imshow("Pointed", pointed_img)
    # cv.imshow("Warped", warped_img)
    # cv.waitKey(0)

    # cv.destroyAllWindows()
    # cv.waitKey(0)


if __name__ == '__main__':
    predict_all(args.path)
