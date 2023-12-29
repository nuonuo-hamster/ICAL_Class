import numpy as np
import cv2 as cv

def generate_image(img, img_name):
    img_modified = cv.resize(img, (80, 120), interpolation=cv.INTER_AREA)
    # cv.imshow("Initial", img_modified)

    # 01-Gray
    img_gray = cv.cvtColor(img_modified, cv.COLOR_BGR2GRAY)
    img_gray = cv.cvtColor(img_gray, cv.COLOR_GRAY2BGR)
    # cv.imshow("Gray", img_gray)

    # 02-ZoomIn
    img_zoomin = cv.resize(img_modified, (240, 360), interpolation=cv.INTER_CUBIC)
    img_zoomInCut = img_zoomin[ 120:240, 80:160 ] 
    # cv.imshow("ZoomIn", img_zoomInCut)

    # 03-ZoomOut
    background = cv.imread("./1011HWPhoto/background.png")
    img_zoomout = cv.resize(img_modified, (40, 60), interpolation=cv.INTER_AREA)
    background[ 30 : 90, 20 : 60] = img_zoomout
    # cv.imshow('ZoomOut', background)

    # 04-Binary_type1 
    ret, thresh_type1 = cv.threshold(img_gray, 150, 255, cv.THRESH_TOZERO)
    # cv.imshow('type1', thresh_type1)

    # 05-Binary_type2
    ret, thresh_type2 = cv.threshold(img_gray, 150, 255, cv.THRESH_TRUNC)
    # cv.imshow('type2', thresh_type2 )

    # 06-flip
    image_filp = cv.flip(img_modified , 1)
    # cv.imshow('flip', image_filp)

    # horizontal
    image_h1 = cv.hconcat([img_gray, img_zoomInCut, background])
    image_h2 = cv.hconcat([thresh_type1, thresh_type2, image_filp])
    # vertical
    Result_Img = cv.vconcat([image_h1, image_h2])

    cv.imshow('Result', Result_Img)
    cv.imwrite("./result_img/{}.jpg".format(img_name),Result_Img)

    # print ("Width: {}\nHeight: {}\nChannel: {}".format(img.shape[0],img.shape[1],img.shape[2]))
    cv.waitKey(0)
    cv.destroyAllWindows()

# NO.1
image01 = cv.imread("./1011HWPhoto/AOI_2.png")
generate_image(image01,"result_01")

# NO.2
image02 = cv.imread("./1011HWPhoto/Fighter_2.jpg")
generate_image(image02,"result_02")

# NO.3
image03 = cv.imread("./1011HWPhoto/live_2.jpg")
generate_image(image03,"result_03")

# NO.4
image04 = cv.imread("./1011HWPhoto/Med_1.bmp")
generate_image(image04,"result_04")

# NO.5
image05 = cv.imread("./1011HWPhoto/street_2.jpg")
generate_image(image05,"result_05")