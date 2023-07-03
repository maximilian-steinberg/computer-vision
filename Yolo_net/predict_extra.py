import cv2 as cv
import numpy as np
from PIL import Image


def get_approx(area):
    """
    Finds a 4-point approximation of a contour area by varying the epsilon value.
    The epsilon value controls the level of approximation.

    Parameters:
        area (numpy.ndarray): Contour area to approximate.

    Returns:
        numpy.ndarray: Array of 4 points representing the approximation.
    """

    # Loop over a range of epsilon values
    for eps in np.linspace(0.001, 0.05, 10):

        # Calculate the perimeter of the contour
        peri = cv.arcLength(area, True)

        # Approximate the contour using the epsilon value
        approx = cv.approxPolyDP(area, eps * peri, True)

        # Check if the approximation has 4 points
        if len(approx) == 4:
            # Reshape the points into a 2D array
            return approx.ravel().reshape(-1, 2)

    # If no 4-point approximation is found, return the last approximation
    return approx.ravel().reshape(-1, 2)


def order_points(pts):
    """
    Orders the input points in a clockwise manner starting from the top-left point.

    Parameters:
        pts (numpy.ndarray): Input points to be ordered.

    Returns:
        numpy.ndarray: Array of ordered points.
    """

    # Compute the convex hull of the input points
    hull = cv.convexHull(pts)

    # Select the first four points from the convex hull as border points
    border_points = hull[:4]

    # Remove any single-dimensional entries from the border points
    border_points = np.squeeze(border_points)

    # Rotate the border points by shifting them by three positions clockwise
    rotated_arr = np.roll(border_points, 3, axis=0)

    # Initialize an empty array to store the ordered points
    rect = np.zeros((4, 2), dtype="float32")

    # Assign the rotated points to the corresponding indices in the rect array
    rect[0] = rotated_arr[0]
    rect[1] = rotated_arr[1]
    rect[2] = rotated_arr[2]
    rect[3] = rotated_arr[3]

    # Return the rect array as the final ordered points
    return np.array(rect)


def four_point_transform(image, pts):
    """
    Applies a perspective transform to an image based on the four given points.

    Parameters:
        image (numpy.ndarray): Input image to be transformed.
        pts (numpy.ndarray): Four points representing the region of interest.

    Returns:
        numpy.ndarray: Warped image after perspective transformation.
    """

    # Order the input points using the order_points function
    rect = order_points(pts)

    # Extract the four ordered points (top-left, top-right, bottom-right, bottom-left)
    (tl, tr, br, bl) = rect

    # Calculate the width and height of the transformed image based on the ordered points
    width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    width_max = max(int(width_a), int(width_b))

    height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    height_max = max(int(height_a), int(height_b))

    # Construct the destination points for the transformed image
    dst = np.array([
        [0, 0],
        [width_max - 1, 0],
        [width_max - 1, height_max - 1],
        [0, height_max - 1]], dtype="float32")

    # Retrieve the perspective transformation matrix using cv.getPerspectiveTransform
    matrix = cv.getPerspectiveTransform(rect, dst)

    # Apply the perspective transformation to the input image using cv.warpPerspective
    warped = cv.warpPerspective(image, matrix, (width_max, height_max))

    # Return the warped image as the result of the transformation
    return warped


def is_image_file(file_path):
    """
    Check if the given file_path is an image file.

    Parameters
    ----------
    file_path : str
        The path to the file to check.

    Returns
    -------
    bool
        Returns True if the file is an image file, otherwise False.

    Raises
    ------
    IOError
        If the file_path is not accessible.

    Example
    -------
    >>> is_image_file('C:/Users/Dimster/Documents/6. Semester/Computer Vision/datasets/test/images/image1.jpg')
    True
    >>> is_image_file('C:/Users/Dimster/Documents/6. Semester/Computer Vision/datasets/test/textfile.txt')
    False
    """

    try:
        Image.open(file_path)
        return True
    except IOError:
        return False
