import cv2
import numpy as np


def temp(a):
    pass

cv2.namedWindow("Trackbars", cv2.WINDOW_NORMAL)
cv2.resizeWindow('Trackbars', 640, 240)
# cv2.imshow('Trackbars', np.ones((3,3), dtype='uint8'))
# cv2.waitKey(0)
# cv2.destroyAllWindows()
cv2.createTrackbar('h min', 'Trackbars', 0, 179, temp)
cv2.createTrackbar('h max', 'Trackbars', 0, 179, temp)
cv2.createTrackbar('s min', 'Trackbars', 0, 255, temp)
cv2.createTrackbar('s max', 'Trackbars', 0, 255, temp)
cv2.createTrackbar('v min', 'Trackbars', 0, 255, temp)
cv2.createTrackbar('v max', 'Trackbars', 0, 255, temp)


while True:
    img = cv2.imread('Session8/S9_Lambo.png')
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    h_min = cv2.getTrackbarPos('h min', "Trackbars")
    h_max = cv2.getTrackbarPos('h max', "Trackbars")
    s_min = cv2.getTrackbarPos('s min', "Trackbars")
    s_max = cv2.getTrackbarPos('s max', "Trackbars")
    v_min = cv2.getTrackbarPos('v min', "Trackbars")
    v_max = cv2.getTrackbarPos('v max', "Trackbars")

    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])

    mask = cv2.inRange(hsv, lower, upper)

    segmented = cv2.bitwise_and(img, img, mask=mask)

    cv2.imshow('original', img)
    cv2.imshow('hsv', hsv)
    cv2.imshow('mask', mask)
    cv2.imshow('segmented', segmented)

    cv2.waitKey(1)
    
cv2.destroyAllWindows()
