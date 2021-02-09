import os
from glob import glob

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

from keras.preprocessing.image import ImageDataGenerator
from keras import layers
from keras import Model
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping

import matplotlib.pyplot as plt

pre_trained_model = InceptionV3(input_shape=(500, 400, 3), include_top=False, weights="imagenet")

for layer in pre_trained_model.layers:
    print(layer.name)
    layer.trainable = False
    
print(len(pre_trained_model.layers))


last_layer = pre_trained_model.get_layer('mixed10')
print('last layer output shape:', last_layer.output_shape)
last_output = last_layer.output


# Flatten the output layer to 1 dimension
x = layers.GlobalMaxPooling2D()(last_output)
# Add a fully connected layer with 512 hidden units and ReLU activation
x = layers.Dense(512, activation='relu')(x)
# Add a dropout rate of 0.7
x = layers.Dropout(0.5)(x)
# Add a final sigmoid layer for classification
x = layers.Dense(8, activation='softmax')(x)

# Configure and compile the model

model = Model(pre_trained_model.input, x)
optimizer = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

# optimizer = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
# model.compile(loss='categorical_crossentropy',
#               optimizer=optimizer,
#               metrics=['acc'])

model.summary()

train_datagen = ImageDataGenerator(rotation_range=60, width_shift_range=0.2, height_shift_range=0.2,
                                   shear_range=0.2, zoom_range=0.2, fill_mode='nearest', validation_split=0.25)

train_dir= 'D:\Education\College\Semester V\ISM\Image_Data\Training_Input'
test_dir= 'D:\Education\College\Semester V\ISM\Image_Data\Test_Input'


train_df_path = 'D:\Education\College\Semester V\ISM\Image_Data\CSV file\label_groundtruth_train.csv'

train_df = pd.read_csv(train_df_path)

train_df["image"] = train_df["image"].apply(lambda x: x+".jpg")
print(train_df.head())


train_df["label"] = train_df["label"].apply(lambda x: str(x))
print(train_df.head())


batch_size = 50
epochs = 3

train_generator=train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=train_dir,
    target_size=(500, 400),
    x_col='image',
    y_col='label',
    subset="training",
    batch_size=batch_size,
    seed=21)

valid_generator=train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=train_dir,
    target_size=(500, 400),
    x_col='image',
    y_col='label',
    subset="validation",
    batch_size=batch_size,
    seed=21)

STEP_TRAIN=train_generator.n//batch_size
STEP_VALID=valid_generator.n//batch_size

history = model.fit_generator(train_generator,
                              epochs = epochs,
                              steps_per_epoch=STEP_TRAIN,
                              validation_data = valid_generator,
                              validation_steps=STEP_VALID,
                              verbose=1)

for layer in pre_trained_model.layers:
    layer.trainable = True



learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', patience=3, verbose=1, factor=0.5, 
                                            min_lr=0.000001, cooldown=2)


model.summary()

batch_size = 50
epochs = 4

STEP_TRAIN=train_generator.n//batch_size
STEP_VALID=valid_generator.n//batch_size


history = model.fit_generator(train_generator,
                              epochs = epochs,
                              steps_per_epoch=STEP_TRAIN,
                              validation_data = valid_generator,
                              validation_steps=STEP_VALID,
                              verbose=1,
                              callbacks=[learning_rate_reduction])

                              
model.save("InceptionV3_Full.h5")