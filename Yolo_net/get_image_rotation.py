import cv2 as cv
import numpy as np

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


