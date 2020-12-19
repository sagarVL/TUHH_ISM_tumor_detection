import os.path
import os
from os import path
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import skew
import skimage    
from skimage.feature import greycomatrix, greycoprops
from skimage.draw import ellipse
import string
import csv
import image_preprocessor as processor
import time
import multiprocessing
import border
import pandas as pd


def diameter(image):
    
    y = 384
    x = 512
    w = 730
    h = 600 
    Area = []
    Dia = []
    diameter = 0
    area = 0
    
    img = image    
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray,(5,5),0)
    # cropped = blur[y:y+h, x:x+w]

    ret,thresh = cv.threshold(blur, 70, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    for c in contours:
        area = cv.contourArea(c)
        Area.append(area)
        equi_dia = np.sqrt(4*area/np.pi) 
        Dia.append(equi_dia)

    for d in Dia:
        if d > diameter:
            diameter = d    

    return diameter
def image_feature_extraction():
  
  with open("D:\Documents\GitHub\TUHH_ISM_tumor_detection\csv_features\\features_validation_19122020.csv","a",newline='') as csvfile:
      writer =csv.writer(csvfile,delimiter=',', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
      writer.writerow(['imageID', 'diameter', 'border'])
  
  reader = pd.read_csv("D:\Documents\GitHub\TUHH_ISM_tumor_detection\csv_features\groundtruth_val.csv")
  
  for row in range(len(reader)):
      idno = reader.values[row][0]
      print(idno)
      PATH='G:\ISM\Processed_Images\\processed_image'+idno +'.jpg'
      #PATH_1='G:\ISM\Processed_Images\\'+idno +'_downsampled.jpg'

      if path.exists(PATH):
        imageID=idno
        imageSelected = cv.imread(PATH)
        print(imageID)
        print(PATH)
        print("Started")

        #Diameter

        diameter_data = diameter(imageSelected)

        #Border
        border_data = border.border_main(imageSelected)


        with open("D:\Documents\GitHub\TUHH_ISM_tumor_detection\csv_features\\features_validation_19122020.csv","a",newline='') as csvfile:
          writer =csv.writer(csvfile,delimiter=',', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
          writer.writerow([imageID, diameter_data, border_data])

      else:
            print('no file found:'+'processed_image'+idno )

if __name__ == '__main__':
    image_feature_extraction()