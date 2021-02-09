import numpy as np
import cv2

def border_orig(image):        
    Area = []
    img = image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (17, 17), 32)
    ret,thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    contours,hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(contours, key=cv2.contourArea)

    cv2.drawContours(img, c, -1, (0, 0, 255), 3)
    for c in contours:
        area = cv2.contourArea(c)
        Area.append(area)
    
    for d in Area:
        if d > area:
            area = d
    
    return area


def border_rect(image):
    Area = []
    rectangle = []
    img_line = image
    
    gray_line = cv2.cvtColor(img_line, cv2.COLOR_BGR2GRAY)
    blur_line = cv2.GaussianBlur(gray_line, (17, 17), 32)
    ret,thresh_line = cv2.threshold(blur_line,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)


    contours_line,hierarchy = cv2.findContours(thresh_line, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(contours_line, key=cv2.contourArea)

    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)

    # contour = box

    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])

    cv2.line(img_line,extLeft,extTop,(255,0,0),2)
    cv2.line(img_line,extTop,extRight,(255,0,0),2)
    cv2.line(img_line,extRight,extBot,(255,0,0),2)
    cv2.line(img_line,extBot,extLeft,(255,0,0),2)

    rectangle.append(extLeft)
    rectangle.append(extRight)
    rectangle.append(extTop)
    rectangle.append(extBot)
    contour = [np.array([rectangle], dtype=np.int32)]

    
    for c in contour:
        area = cv2.contourArea(c)
        Area.append(area)
  
    return Area


def border_main(pass_image):
    rect_area = border_rect(pass_image)
    contour_area = border_orig(pass_image)

    border_irreg = contour_area - rect_area[0]
    print('Irregular area = ', border_irreg)
    return border_irreg
