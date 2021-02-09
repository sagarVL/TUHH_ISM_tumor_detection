
import os
from glob import glob

import time
import random

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

from keras.preprocessing.image import ImageDataGenerator
from keras import layers
from keras import Model, Input
from keras.applications.inception_v3 import InceptionV3
from keras.applications.densenet import DenseNet201

from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping

import cv2
import csv

import matplotlib.pyplot as plt

input_shape = [500,400,3]
model_input = Input(shape=input_shape)

denseNet = DenseNet201(input_shape=input_shape, input_tensor=model_input, include_top=False, weights=None)


for layer in denseNet.layers:
    layer.trainable = True


denseNet_last_layer = denseNet.get_layer('relu')
print('last layer output shape:', denseNet_last_layer.output_shape)
denseNet_last_output = denseNet_last_layer.output


# Flatten the output layer to 1 dimension
x_denseNet = layers.GlobalMaxPooling2D()(denseNet_last_output)
# Add a fully connected layer with 512 hidden units and ReLU activation
x_denseNet = layers.Dense(512, activation='relu')(x_denseNet)
# Add a dropout rate of 0.7
x_denseNet = layers.Dropout(0.5)(x_denseNet)
# Add a final sigmoid layer for classification
x_denseNet = layers.Dense(8, activation='softmax')(x_denseNet)

# Configure and compile the model

denseNet_model = Model(model_input, x_denseNet)
optimizer = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)
denseNet_model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

denseNet_model.load_weights("DenseNet_Full_26epoch.h5")    


denseNet_model.summary()

train_dir= 'D:\Education\College\Semester V\ISM\Image_Data\Training_Input'
test_dir= 'D:\Education\College\Semester V\ISM\Image_Data\Training_Input'


train_df_path = 'D:\Education\College\Semester V\ISM\Image_Data\CSV file\label_groundtruth_validation.csv'

train_df = pd.read_csv(train_df_path)

test_df = pd.read_csv(train_df_path)

train_df["image"] = train_df["image"].apply(lambda x: x+".jpg")
print(train_df.head())


train_df["label"] = train_df["label"].apply(lambda x: str(x))
print(train_df.head())

pic_idx = random.sample(range(1,len(test_df)),50)

def get_image(i=0):
    data = test_df.iloc[[i]].values[0]
    img_name = os.path.join(train_dir,data[0]+'.jpg')
    print(img_name)
    img = cv2.imread(img_name)
    img = img/255.0
    img = cv2.resize(img,(500,400))
    img = np.reshape(img,[1,500,400,3])
    dataval=data[0]
    return img, dataval



with open('validation_data_20012021_Retrain_Inception_26Epochs.csv','a',newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter =',',quoting=csv.QUOTE_MINIMAL, lineterminator='\n' )
    writer.writerow(['ID','MEL','NV','BCC','AK','BKL','DF','VASC','SCC','UNK'])

for i in range (0,2535):
    img,dataval = get_image(i)
    start = time.time()
    pred = denseNet_model.predict(img)
    #print("Predicted with simple model in", time.time()-start)
    #print(pred)
    maxIndex = 0
    for i in range(0,7):
        newnumber = pred[0][i]
        if newnumber > pred[0][maxIndex]:
            maxIndex=i
    print(dataval, maxIndex+1)

    with open('validation_data_20012021_Retrain_Inception_26Epochs.csv','a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter =',',quoting=csv.QUOTE_MINIMAL, lineterminator='\n' )
        writer.writerow([dataval,pred[0][0],pred[0][1],pred[0][2],pred[0][3],pred[0][4],pred[0][5],pred[0][6],pred[0][7],'0.00000000'])

