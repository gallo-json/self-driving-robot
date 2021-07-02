import cv2
import numpy as np

def empty(a): pass

def stackImages(scale, imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver

red = cv2.imread("pictures/red-cropped.jpg")
yellow = cv2.imread("pictures/yellow-cropped.jpg")
green = cv2.imread("pictures/green-cropped.jpg")

cv2.namedWindow("Trackbars")
cv2.resizeWindow("Trackbars", 640, 240)

cv2.createTrackbar("Hue min", "Trackbars", 0, 179, empty) # default start value: 0
cv2.createTrackbar("Hue max", "Trackbars", 179, 179, empty) # default start value: 179

cv2.createTrackbar("Saturation min", "Trackbars", 0, 255, empty) # default start value: 0
cv2.createTrackbar("Saturation max", "Trackbars", 255, 255, empty) # default start value: 255

cv2.createTrackbar("Value min", "Trackbars", 0, 255, empty) # default start value: 0
cv2.createTrackbar("Value max", "Trackbars", 255, 255, empty) # default start value: 255

while True:
    red_blurred = cv2.GaussianBlur(red, (11, 11), 0)
    red_hsv = cv2.cvtColor(red_blurred, cv2.COLOR_BGR2HSV)

    yellow_blurred = cv2.GaussianBlur(yellow, (11, 11), 0)
    yellow_hsv = cv2.cvtColor(yellow_blurred, cv2.COLOR_BGR2HSV)

    green_blurred = cv2.GaussianBlur(green, (11, 11), 0)
    green_hsv = cv2.cvtColor(green_blurred, cv2.COLOR_BGR2HSV)

    hue_min = cv2.getTrackbarPos("Hue min", "Trackbars")
    hue_max = cv2.getTrackbarPos("Hue max", "Trackbars")

    sat_min = cv2.getTrackbarPos("Saturation min", "Trackbars")
    sat_max = cv2.getTrackbarPos("Saturation max", "Trackbars")

    val_min = cv2.getTrackbarPos("Value min", "Trackbars")
    val_max = cv2.getTrackbarPos("Value max", "Trackbars")

    lower = np.array([hue_min, sat_min, val_min])
    upper = np.array([hue_max, sat_max, val_max])

    red_mask = cv2.inRange(red_hsv, lower, upper)
    yellow_mask = cv2.inRange(yellow_hsv, lower, upper)
    green_mask = cv2.inRange(green_hsv, lower, upper)

    cv2.imshow("Out", stackImages(0.5, ([red, red_hsv, red_mask], [yellow, yellow_hsv, yellow_mask], [green, green_hsv, green_mask])))

    if cv2.waitKey(1) & 0xFF == ord('n'):
        break