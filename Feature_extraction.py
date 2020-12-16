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

timer =[]

def colorHistogram(imageID,idno, imageSelected):
  histogram_data =[]
  for i in range(3):
    histogram =cv.calcHist([imageSelected],[i],None,[256],[0,256])  
    histogram_mean = histogram.mean()
    histogram_skew = skew(histogram)
    histogram_std = histogram.std()
    histogram_entropy = skimage.measure.shannon_entropy(imageSelected)
    histogram_data.append(histogram_mean)
    histogram_data.append(histogram_skew[0])
    histogram_data.append(histogram_std)
    histogram_data.append(histogram_entropy)

  return histogram_data
    
def Moments(imageID, imageSelected):
    moments =[]
    ret,thresh = cv.threshold(imageSelected,128,255,cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
    m=cv.moments(thresh)
    moments.append(m["m00"])
    moments.append(m["m10"])
    moments.append(m["m01"])
    moments.append(m["m20"])
    moments.append(m["m11"])
    moments.append(m["m02"])
    moments.append(m["m30"])
    moments.append(m["m21"])
    moments.append(m["m12"])
    moments.append(m["m03"])

    # moments.append(m["mu20"])
    # moments.append(m["mu11"])
    # moments.append(m["mu02"])
    # moments.append(m["mu30"])
    # moments.append(m["mu21"])
    # moments.append(m["mu12"])
    # moments.append(m["mu03"])

    # moments.append(m["nu20"])
    # moments.append(m["nu11"])
    # moments.append(m["nu02"])
    # moments.append(m["nu30"])
    # moments.append(m["nu21"])
    # moments.append(m["nu12"])
    # moments.append(m["nu03"])

    return moments

def HuMoments(imageID,imageSelected):
  huMoments =[]
  ret,thresh = cv.threshold(imageSelected,128,255,cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
  m=cv.moments(thresh)
  hu= cv.HuMoments(m)
  huMoments.append(hu[0][0])
  huMoments.append(hu[1][0])
  huMoments.append(hu[2][0])
  huMoments.append(hu[3][0])
  huMoments.append(hu[4][0])
  huMoments.append(hu[5][0])
  huMoments.append(hu[6][0])

  return huMoments
  
def CoOccuranceMatrix(imageID, imageSelected):
  cmmatrix=[]
  gray_image = cv.cvtColor(imageSelected, cv.COLOR_BGR2GRAY)
  co_matrix=greycomatrix(gray_image, [1,2,4,8], [0, np.pi/4, np.pi/2, 3*np.pi/4],levels=256)
  contrast=greycoprops(co_matrix,'contrast').flatten()
  
  for i in range(16):
    cmmatrix.append(contrast[i])
  
  return cmmatrix

def image_feature_extraction():
  
  with open("D:\Education\College\Semester V\ISM\Project\csv_features\Features_training_15122020.csv","a",newline='') as csvfile:
      writer =csv.writer(csvfile,delimiter=',', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
      writer.writerow(['imageID','mean-R','skew-R','std-R','entropy-R','mean-G','skew-G','std-G','entropy-G','mean-B','skew-B','std-B','entropy-B',
                       't1','t2','t3','t4','t5','t6','t7','t8','t9','t10','t11','t12','t13','t14','t15','t16',
                        'hu1','hu2','hu3','hu4','hu5','hu6','hu7',
                        'm00','m10','m01','m20','m11','m02','m30','m21','m12','m03'])
  
  with open('groundtruth_train.csv', mode='r') as csv_file:
    for row in csv.reader(csv_file, delimiter = ','):
      idno = row[0]
      PATH='G:\ISM\Processed_Images\processed_image'+idno +'.jpg'
      PATH_1='G:\ISM\Processed_Images\processed_image'+idno +'_downsampled.jpg'

      if path.exists(PATH):
        imageID='ISIC_'+idno
        imageSelected = cv.imread(PATH)
        print(imageID)
        print(PATH)
        print("Started")
        
        colorhist_data = colorHistogram(imageID, idno, imageSelected)      
        texture_data = CoOccuranceMatrix(imageID, imageSelected)

        segmented_image, seg_data = processor.k_means_segementation(imageSelected, 5)
        hu_moments_data = HuMoments(imageID, segmented_image)
        moments_data = Moments(imageID,segmented_image)

        #Moments(imageID, imageSelected
        #HuMoments(imageID, imageSelected)
        #CoOccuranceMatrix(imageID, imageSelected)
        #print("Started")
        #hair_removed_img = processor.hai_removal(PATH)
        #print("Hair Removal Done")
        #c_mask_img = processor.create_circular_mask(imageSelected)
        #print("Masking done")
        #contrast_img = processor.change_contrast(hair_removed_img,5)
        #print("Constrasting Done and saved")

        #k_img= processor.k_means_segementation(imageSelected,5)

        #cv.imwrite('G:\ISM\Processed_Images\test\processed_image'+imageID+'.jpg',k_img)
        with open("D:\Education\College\Semester V\ISM\Project\csv_features\Features_training_15122020.csv","a",newline='') as csvfile:
          writer =csv.writer(csvfile,delimiter=',', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
          writer.writerow([imageID, colorhist_data[0],colorhist_data[1],colorhist_data[2],colorhist_data[3]
          ,colorhist_data[4],colorhist_data[5],colorhist_data[6],colorhist_data[7],colorhist_data[8],colorhist_data[9]  
          ,colorhist_data[10],colorhist_data[11], 
          texture_data[0],texture_data[1],texture_data[2],texture_data[3],texture_data[4],texture_data[5],texture_data[6],texture_data[7],
          texture_data[8],texture_data[9],texture_data[10],texture_data[11],texture_data[12],texture_data[13],texture_data[14],texture_data[15],
          hu_moments_data[0],hu_moments_data[1],hu_moments_data[2],hu_moments_data[3],hu_moments_data[4],hu_moments_data[5],hu_moments_data[6],
          moments_data[0],moments_data[1],moments_data[2],moments_data[3],moments_data[4],moments_data[5],moments_data[6],
          moments_data[7],moments_data[8],moments_data[9]])

        
    

      elif path.exists(PATH_1):
        imageID='ISIC_'+idno+'_downsampled' 
        imageSelected = cv.imread(PATH_1)
        print(imageID)
        print(PATH_1)
        print("Started")

        colorhist_data = colorHistogram(imageID, idno, imageSelected)      
        texture_data = CoOccuranceMatrix(imageID, imageSelected)

        segmented_image, seg_data = processor.k_means_segementation(imageSelected, 5)
        hu_moments_data = HuMoments(imageID, segmented_image)
        moments_data = Moments(imageID,segmented_image)

        #hair_removed_img = processor.hair_removal(PATH_1)
        #print("Hair Removal Done")
        #c_mask_img = processor.create_circular_mask(imageSelected)
        #print("Masking done")
        #contrast_img = processor.change_contrast(hair_removed_img,5)
        #print("Constrasting Done and saved")
        #k_img = processor.k_means_segementation(imageSelected,10)
        #cv.imwrite('G:\ISM\Processed_Images\test\processed_image'+imageID+'.jpg',k_img)

        #colorHistogram(imageID, idno, imageSelected)
        #Moments(imageID, imageSelected)
        #HuMoments(imageID, imageSelected)
        #CoOccuranceMatrix(imageID, imageSelected)
        with open("D:\Education\College\Semester V\ISM\Project\csv_features\Features_training_15122020.csv","a",newline='') as csvfile:
          writer =csv.writer(csvfile,delimiter=',', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
          writer.writerow([imageID, colorhist_data[0],colorhist_data[1],colorhist_data[2],colorhist_data[3]
          ,colorhist_data[4],colorhist_data[5],colorhist_data[6],colorhist_data[7],colorhist_data[8],colorhist_data[9]  
          ,colorhist_data[10],colorhist_data[11], 
          texture_data[0],texture_data[1],texture_data[2],texture_data[3],texture_data[4],texture_data[5],texture_data[6],texture_data[7],
          texture_data[8],texture_data[9],texture_data[10],texture_data[11],texture_data[12],texture_data[13],texture_data[14],texture_data[15],
          hu_moments_data[0],hu_moments_data[1],hu_moments_data[2],hu_moments_data[3],hu_moments_data[4],hu_moments_data[5],hu_moments_data[6],
          moments_data[0],moments_data[1],moments_data[2],moments_data[3],moments_data[4],moments_data[5],moments_data[6],
          moments_data[7],moments_data[8],moments_data[9]])


      else:
        print('no file found:'+'processed_imageISIC_'+idno )

if __name__ == '__main__':
  image_feature_extraction()
