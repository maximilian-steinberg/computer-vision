import cv2 as cv
import numpy as np

def get_approx(area):
    for eps in np.linspace(0.001, 0.05, 10):
        # approximate the contour
        peri = cv.arcLength(area, True)
        approx = cv.approxPolyDP(area, eps * peri, True)
        if len(approx) == 4:
            print("FOUND: eps={:.4f}, num_pts={}".format(eps, len(approx)))
            #print(approx)
            print("Unsortiert: {:}".format(approx.ravel().reshape(-1, 2)))
            return approx
    print ("NOT FOUND")
    return approx

def order_points(pts):
    # rect = np.zeros((4, 2), dtype = "float32")

    # s = pts.sum(axis = 1)
    # rect[0] = pts[np.argmin(s)]
    # rect[2] = pts[np.argmax(s)]

    # diff = np.diff(pts, axis = 1)
    # rect[1] = pts[np.argmin(diff)]
    # rect[3] = pts[np.argmax(diff)]
    # print("Sorted: {:}".format(rect))


    # return rect
    hull = cv.convexHull(pts)
    # Get the four border points
    border_points = hull[:4]
    # Convert to the desired format
    border_points = np.squeeze(border_points)

    rotated_arr = np.roll(border_points, 3, axis=0)
    

    rect = np.zeros((4, 2), dtype = "float32")
    
    rect[0] = rotated_arr[0]
    rect[1] = rotated_arr[1]
    rect[2] = rotated_arr[2]
    rect[3] = rotated_arr[3]

    print("pts[0]: {:}".format(pts[0]))
    print("arr[0]: {:}".format(rotated_arr[0]))

    print("Sorted: {:}".format(rect))
    return np.array(rect)
    
     

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")

    matrix = cv.getPerspectiveTransform(rect, dst)
    warped = cv.warpPerspective(image, matrix, (maxWidth, maxHeight))

    return warped


def get_image_rotation(img):
    # Edge detection
    dst = cv.Canny(cv.cvtColor(img, cv.COLOR_BGR2GRAY), 50, 200, None, 3)
    
    # Copy edges to the images that will display the results in BGR
    cdst = img
    cdstP = np.copy(cdst)

    #  Standard Hough Line Transform
    linesP = cv.HoughLinesP(dst, 1, np.pi / 180, 50, None, 50, 10)

    length = 0
    longest_line = None
    winkel = 0

    if linesP is not None:
        for i in range(0, len(linesP)):
            lineLength = None
            rho = linesP[i][0][0]
            theta = linesP[i][0][1]

            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho

            # Berechne die Endpunkte der Linie
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            # Berechne die Länge der Linie
            lineLength = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

            if lineLength > length:
                length = lineLength
                longest_line = i
                x1, y1, x2, y2 = linesP[i][0]
                winkel = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi

    l = linesP[longest_line][0]

    # cv.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv.LINE_AA)
    # cv.imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP)
    # cv.waitKey(0)
    return winkel


