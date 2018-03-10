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

#PATHS
#os.chdir(r"C:/Users/cjiaen/Documents/Sem1/CS5242_NNDL/Kaggle")
#TRAIN_DIR = r"C:\Users\cjiaen\Documents\Sem1\CS5242_NNDL\Kaggle\Training_data"
#VALIDATION_DIR = r"C:\Users\cjiaen\Documents\Sem1\CS5242_NNDL\Kaggle\Validation_data"
#MODEL_PATH = r"C:/Users/cjiaen/Documents/Sem1/CS5242_NNDL/Kaggle/Model/model1.h5"
#LOG_DIR = r"C:/Users/cjiaen/Documents/Sem1/CS5242_NNDL/Kaggle/log_files/model1"
#CSV_DIR = r"C:/Users/cjiaen/Documents/Sem1/CS5242_NNDL/Kaggle/csv_output"
#CSV_NAME = 'model1.csv'

TRAIN_DIR = "/home/users/nus/e0227268/kaggle/data/train/"
VALIDATION_DIR = "/home/users/nus/e0227268/kaggle/data/validate/"
MODEL_PATH = "/home/users/nus/e0227268/kaggle/models/model1.h5"
LOG_DIR = "/home/users/nus/e0227268/kaggle/log_files/"
CSV_DIR = "/home/users/nus/e0227268/kaggle/csv_files/"
CSV_NAME = 'model1.csv'
    
#Variables
BATCH_SIZE = 32
NUM_CLASSES = 132

IMG_H = 299
IMG_W = 299

#load model if not already loaded
#if 'new_model' not in locals():
#    new_model = keras.models.load_model(MODEL_PATH)

#base_model = InceptionV3(weights='imagenet', include_top=False)
#base_model = InceptionV3(weights='imagenet', include_top=False)
#base_model.summary()
#base_model.get_config()
#base_model.save(r"C:/Users/cjiaen/Documents/Sem1/CS5242_NNDL/Kaggle/Model/model0.h5")


#model.load_weights('cache/inception_v3_weights_th_dim_ordering_th_kernels.h5')
##preprocessing
#create image data generator, specify preprocessing
train_datagen = image.ImageDataGenerator(
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        preprocessing_function=preprocess_input)

#generate batches of data for training
train_data_flow = train_datagen.flow_from_directory(
        directory = TRAIN_DIR,
        target_size = (IMG_H, IMG_W),
        class_mode = "categorical",
        batch_size = BATCH_SIZE,
        shuffle = True,
        seed = 10)

MINI_BATCHES = int(np.floor(train_data_flow.n/BATCH_SIZE))
#MINI_BATCHES = 50

#create image data generator for validation data
validation_datagen = image.ImageDataGenerator(
        fill_mode='nearest',
        preprocessing_function=preprocess_input)

#generate batches of data for training
validation_data_flow = validation_datagen.flow_from_directory(
        directory = VALIDATION_DIR,
        target_size = (IMG_H, IMG_W),
        class_mode = "categorical",
        batch_size = BATCH_SIZE,
        shuffle = True,
        seed = 10)

#VALIDATION_STEPS = int(np.floor(validation_data_flow.n/BATCH_SIZE))
VALIDATION_STEPS = 10

#training on bottleneck features (from last layer of the CNN)
#bottleneck_features_train = new_model.predict_generator(train_datagen, steps = 10, max_queue_size=2,
#                                                         workers = 1, use_multiprocessing = False, verbose = 1)
#np.save(open(file="bottleneck_features.npy", mode='wb'), bottleneck_features_train)

########################
# Model 1: Create new FC layer, input using bottleneck features
#########################
##training a dense network using bottleneck features
#train_data = np.load(open('bottleneck_features.npy', mode='rb'))

#Create new model
#x = base_model.output
#x = GlobalAveragePooling2D()(x)
#x = Dense(2048, activation='relu')(x)
#predictions = Dense(NUM_CLASSES, activation='softmax')(x)
#model1 = Model(inputs=base_model.input, outputs=predictions)

#setup callbacks
checkpointer = keras.callbacks.ModelCheckpoint(filepath=MODEL_PATH,
                                               verbose=1,
                                               save_best_only=True)
early_stop = keras.callbacks.EarlyStopping(patience=5,
                                           verbose=1)
csvlogger = keras.callbacks.CSVLogger(os.path.join(CSV_DIR,CSV_NAME),
                                      separator=',',
                                      append=False)

#freeze all convolutional layers
#for layer in base_model.layers:
#    #print(layer)
#    layer.trainable = False

#load best model
model1 = keras.models.load_model(MODEL_PATH)
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-6)
#sgd = SGD(lr=0.1, momentum=0.9, decay=1e-6, nesterov=True)
#Model compile
model1.compile(optimizer=adam, loss='categorical_crossentropy',metrics=['accuracy'])
#Model train
model1.fit_generator(train_data_flow,
                        steps_per_epoch=MINI_BATCHES,
                        epochs=5,
                        verbose=1,
                        callbacks=[checkpointer,early_stop,csvlogger],
                        validation_data=validation_data_flow,
                        validation_steps=VALIDATION_STEPS)

#model1.save(MODEL_PATH)

#######################
# Model 2: Retrain top convolutional layer
#######################
#freeze all except last convolutional block (layer 282 onwards)



