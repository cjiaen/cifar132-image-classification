# -*- coding: utf-8 -*-
#
"""
Created on Thu Sep 21 15:08:25 2017

@author: cjiaen
"""

import os
import keras
import numpy as np
import pandas as pd
from keras.utils import np_utils
from keras.optimizers import SGD, Adam
from keras.preprocessing import image
from keras.models import Model,Sequential
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import load_model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten
from keras.applications.inception_v3 import InceptionV3, preprocess_input
import h5py #needed for loading keras weights
import tensorflow as tf
import re

print(tf.__version__)
'''
#Home config
TRAIN_DIR = "/home/calvin/Documents/NNDL/Data/Training_data"
VALIDATION_DIR = "/home/calvin/Documents/NNDL/Data/Validation_data"
MODEL_PATH = "/home/calvin/Documents/NNDL/Models/incepv3_do_10_cnn2.h5"
LOG_DIR = "/home/calvin/Documents/NNDL/Log/"
CSV_DIR = "/home/calvin/Documents/NNDL/CSV/"
#base_model = keras.models.load_model("/home/calvin/Documents/NNDL/Models/model0.h5")
'''
#nscc config
TRAIN_DIR = "/home/users/nus/e0227268/kaggle/Data/Train/"
VALIDATION_DIR = "/home/users/nus/e0227268/kaggle/Data/Validate/"
TEST_DIR = "/home/users/nus/e0227268/kaggle/Data/Validate/Test/"
MODEL_PATH = "/home/users/nus/e0227268/kaggle/Models/incepv3_do_30_cnn6.h5"
LOG_DIR = "/home/users/nus/e0227268/kaggle/Logs/"
CSV_DIR = "/home/users/nus/e0227268/kaggle/CSV/"
#base_model = keras.models.load_model("/home/users/nus/e0227268/kaggle/Models/model0.h5")

#Variables
CSV_NAME = 'incepv3_do_30_cnn6.h5.csv'
BATCH_SIZE = 16
NUM_CLASSES = 132
DROPOUT_RATE = 0.3
IMG_H = 299
IMG_W = 299

model1 = keras.models.load_model(MODEL_PATH)

#base_model = InceptionV3(weights='imagenet', include_top=False)
#base_model.summary()
#base_model.get_config()
#base_model.save("/home/calvin/Documents/NNDL/Models/model0.h5")

#model.load_weights('cache/inception_v3_weights_th_dim_ordering_th_kernels.h5')

##preprocessing
#create image data generator, specify preprocessing
train_datagen = image.ImageDataGenerator(
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        rotation_range=90,
        preprocessing_function=preprocess_input)

#generate batches of data for training
train_data_flow = train_datagen.flow_from_directory(
        directory = TRAIN_DIR,
        target_size = (IMG_H, IMG_W),
        class_mode = "categorical",
        batch_size = BATCH_SIZE)

MINI_BATCHES = int(np.floor(train_data_flow.n/BATCH_SIZE)) + 1
#MINI_BATCHES = 5

#create image data generator for validation data
validation_datagen = image.ImageDataGenerator(preprocessing_function=preprocess_input)

#generate batches of data for training
validation_data_flow = validation_datagen.flow_from_directory(
        directory = VALIDATION_DIR,
        target_size = (IMG_H, IMG_W),
        class_mode = "categorical",
        batch_size = BATCH_SIZE)

VALIDATION_STEPS = int(np.floor(validation_data_flow.n/BATCH_SIZE))
#VALIDATION_STEPS = 5

#training on bottleneck features (from last layer of the CNN)
#bottleneck_features_train = new_model.predict_generator(train_datagen, steps = 10, max_queue_size=2,
#                                                         workers = 1, use_multiprocessing = False, verbose = 1)
#np.save(open(file="bottleneck_features.npy", mode='wb'), bottleneck_features_train)

########################
# Model 1: Create new FC layer, input using bottleneck features
#########################
##training a dense network using bottleneck features
#train_data = np.load(open('bottleneck_features.npy', mode='rb'))

seed = np.random.randint(0, high=1000, size=1)

'''
#Create new model
x = base_model.output
x = GlobalAveragePooling2D()(x) #batch_size*2048
x = Dense(2048, activation='relu')(x) #batch_size*2048
x = Dropout(rate=DROPOUT_RATE, noise_shape=((BATCH_SIZE,2048)), seed=seed)(x)
predictions = Dense(NUM_CLASSES, activation='softmax')(x) #batch size*132
model1 = Model(inputs=base_model.input, outputs=predictions)
'''

#setup callbacks
checkpointer = keras.callbacks.ModelCheckpoint(filepath=MODEL_PATH,
                                               verbose=1,
                                               save_best_only=True)
early_stop = keras.callbacks.EarlyStopping(patience=5,
                                           verbose=1)
csvlogger = keras.callbacks.CSVLogger(os.path.join(CSV_DIR,CSV_NAME),
                                      separator=',',
                                      append=False)
'''
#No dropout
#freeze all convolutional layers
for layer in model1.layers[:-3]:
    layer.trainable = False

#Train FC only
for layer in model1.layers[-3:]:
    layer.trainable = True

#Train 2 CNN blocks
for layer in model1.layers[249:]:
    layer.trainable = True
    print(layer)

#Train 3 CNN blocks
for layer in model1.layers[228:]:
    layer.trainable = True
    print(layer)

#Train 4 CNN blocks
for layer in model1.layers[196:]:
    layer.trainable = True
    print(layer)    
#Train 5 CNN blocks
for layer in model1.layers[164:]:
    layer.trainable = True
    print(layer)
'''
#Train 6 CNN blocks
for layer in model1.layers[132:]:
    layer.trainable = True
    print(layer)


#train
#adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-6)
sgd = SGD(lr=0.0001, momentum=0.9, decay=1e-4, nesterov=True)
#Model compile
model1.compile(optimizer=sgd, loss='categorical_crossentropy',metrics=['accuracy'])
#Model train
model1.fit_generator(train_data_flow,steps_per_epoch=MINI_BATCHES,epochs=10,verbose=1,callbacks=[checkpointer,early_stop,csvlogger],validation_data=validation_data_flow, validation_steps=VALIDATION_STEPS)

model1.save(MODEL_PATH)
print("model saved")
