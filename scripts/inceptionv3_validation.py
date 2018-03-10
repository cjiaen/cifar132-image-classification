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
from keras.preprocessing.image import img_to_array, load_img

print(tf.__version__)
print(keras.__version__)
'''
#Home config
TRAIN_DIR = "/home/calvin/Documents/NNDL/Data/Training_data"
#VALIDATION_DIR = "/home/calvin/Documents/NNDL/Data/Validation_data"
TEST_DIR = r"/home/calvin/Documents/NNDL/Data/temp_test/"
MODEL_PATH2 = r"/home/calvin/Documents/NNDL/Models/irv2_all_40.h5"
#LOG_DIR = r"C:/Users/cjiaen/Documents/Sem1/CS5242_NNDL/Kaggle/Log/"
#CSV_DIR = r"C:/Users/cjiaen/Documents/Sem1/CS5242_NNDL/Kaggle/CSV/"
#base_model = keras.models.load_model("/home/calvin/Documents/NNDL/Models/model0.h5")
'''
#nscc config
TRAIN_DIR = "/home/users/nus/e0227268/kaggle/Data/Train/"
VALIDATION_DIR = "/home/users/nus/e0227268/kaggle/Data/Validate/"
TEST_DIR = "/home/users/nus/e0227268/kaggle/Data/Test/"
MODEL_PATH1 = "/home/users/nus/e0227268/kaggle/Models/incepv3_do_20_all.h5"
MODEL_PATH2 = "/home/users/nus/e0227268/kaggle/Models/incepv3_do_30_all.h5"
MODEL_PATH3 = "/home/users/nus/e0227268/kaggle/Models/incepv3_do_40_all.h5"
#MODEL_PATH4 = "/home/users/nus/e0227268/kaggle/Models/irv2_cnn6.h5"
LOG_DIR = "/home/users/nus/e0227268/kaggle/Logs/"
CSV_DIR = "/home/users/nus/e0227268/kaggle/CSV/"

#Variables
PREDICTION_FILE_NAME = 'predictions_incep20_30_40.csv'
PREDICTION_FILE_NAME1 = 'predictions_incep_20.csv'
PREDICTION_FILE_NAME2 = 'predictions_incep_30.csv'
PREDICTION_FILE_NAME3 = 'predictions_incep_40.csv'
MODEL1_RAW_OUTPUT = 'raw_pred_incep_20.csv'
MODEL2_RAW_OUTPUT = 'raw_pred_incep_30.csv'
MODEL3_RAW_OUTPUT = 'raw_pred_incep_40.csv'
BATCH_SIZE = 128
NUM_CLASSES = 132
IMG_H = 299
IMG_W = 299

#load models
#model1 = keras.models.load_model(MODEL_PATH1)
#model2 = keras.models.load_model(MODEL_PATH2)
model3 = keras.models.load_model(MODEL_PATH3)
#model4 = keras.models.load_model(MODEL_PATH4)

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

mapping_dict = train_data_flow.class_indices
name_to_index_map = dict(map(reversed, mapping_dict.items()))

valid_datagen = image.ImageDataGenerator(preprocessing_function=preprocess_input)

#generate batches of data for training
valid_data_flow = valid_datagen.flow_from_directory(
        directory = VALIDATION_DIR,
        target_size = (IMG_H, IMG_W),
        class_mode = None,
        batch_size = BATCH_SIZE)

#######################
#generate test results
#######################
'''
test_datagen1 = image.ImageDataGenerator(preprocessing_function=preprocess_input)
#generate batches of data for training
test_data_flow1 = test_datagen1.flow_from_directory(
        directory = TEST_DIR,
        target_size = (IMG_H, IMG_W),
        class_mode = None,
        batch_size = BATCH_SIZE,
        shuffle = False)

test_steps = int(np.floor(test_data_flow1.n/BATCH_SIZE))+1
#test_steps = 1
image_ID1 = test_data_flow1.filenames
predicted_results1 = model1.predict_generator(test_data_flow1, steps=test_steps, verbose = 1)
#np.savetxt(MODEL1_RAW_OUTPUT, predicted_results1, delimiter=',')

labels = {}
for i in range(len(predicted_results1)):
    f = image_ID1[i][image_ID1[i].rindex("/")+1:]
    index = int(f.split(".")[0])
    #temp = predicted_results1[i] + predicted_results2[i]
    category = name_to_index_map[predicted_results1[i].argmax()]
    labels[index]=(f,category)
    #print("Image:{}, Category {}".format(image_ID[i],category))
f = open(PREDICTION_FILE_NAME1,"w")
f.write("image_name,category\n")
keys = labels.keys()
keys = sorted(keys)
for i in keys:
    (name,category)=labels[i]
    f.write("%s,%s\n"%(name,category))
f.close()

#format image ID
for image_name in range(len(image_ID1)):
    image_ID1[image_name] = re.sub(r"transferred_test/","",image_ID1[image_name])
df1 = pd.DataFrame(data=predicted_results1,
                   index=image_ID1,
                   columns=range(132))

df1.columns = name_to_index_map.values()
df1.to_csv(MODEL1_RAW_OUTPUT)

test_datagen2 = image.ImageDataGenerator(preprocessing_function=preprocess_input)
#generate batches of data for training
test_data_flow2 = test_datagen2.flow_from_directory(
        directory = TEST_DIR,
        target_size = (IMG_H, IMG_W),
        class_mode = None,
        batch_size = BATCH_SIZE,
        shuffle = False)

test_steps = int(np.floor(test_data_flow2.n/BATCH_SIZE))+1
#test_steps = 1
image_ID2 = test_data_flow2.filenames
predicted_results2 = model2.predict_generator(test_data_flow2, steps=test_steps, verbose = 1)
#np.savetxt(MODEL2_RAW_OUTPUT, predicted_results2, delimiter=',')

labels = {}
for i in range(len(predicted_results2)):
    f = image_ID2[i][image_ID2[i].rindex("/")+1:]
    index = int(f.split(".")[0])
    #temp = predicted_results1[i] + predicted_results2[i]
    category = name_to_index_map[predicted_results2[i].argmax()]
    labels[index]=(f,category)
    #print("Image:{}, Category {}".format(image_ID[i],category))
f = open(PREDICTION_FILE_NAME2,"w")
f.write("image_name,category\n")
keys = labels.keys()
keys = sorted(keys)
for i in keys:
    (name,category)=labels[i]
    f.write("%s,%s\n"%(name,category))
f.close()

#format image ID
for image_name in range(len(image_ID2)):
    image_ID2[image_name] = re.sub(r"transferred_test/","",image_ID2[image_name]) 
    
df2 = pd.DataFrame(data=predicted_results2,
                   index=image_ID2,
                   columns=range(132))

df2.columns = name_to_index_map.values()
df2.to_csv(MODEL2_RAW_OUTPUT)


test_datagen3 = image.ImageDataGenerator(preprocessing_function=preprocess_input)
#generate batches of data for training
test_data_flow3 = test_datagen3.flow_from_directory(
        directory = TEST_DIR,
        target_size = (IMG_H, IMG_W),
        class_mode = None,
        batch_size = BATCH_SIZE,
        shuffle = False)
'''
validation_steps = int(np.floor(valid_data_flow.n/BATCH_SIZE))+1
#test_steps = 10
image_ID3 = valid_data_flow.filenames
predicted_results3 = model3.predict_generator(valid_data_flow, steps=validation_steps, verbose = 1)

labels = {}
for i in range(len(predicted_results3)):
    f = image_ID3[i][image_ID3[i].rindex("/")+1:]
    index = int(f.split(".")[0])
    #temp = predicted_results1[i] + predicted_results2[i]
    category = name_to_index_map[predicted_results3[i].argmax()]
    labels[index]=(f,category)
    #print("Image:{}, Category {}".format(image_ID[i],category))
f = open(PREDICTION_FILE_NAME3,"w")
f.write("image_name,category\n")
keys = labels.keys()
keys = sorted(keys)
for i in keys:
    (name,category)=labels[i]
    f.write("%s,%s\n"%(name,category))
f.close()

'''
#format image ID
for image_name in range(len(image_ID3)):
    image_ID3[image_name] = re.sub(r"transferred_test/","",image_ID3[image_name]) 
    
df3 = pd.DataFrame(data=predicted_results3,
                   index=image_ID3,
                   columns=range(132))

df3.columns = name_to_index_map.values()
df3.to_csv(MODEL3_RAW_OUTPUT)
'''
##################
# Combined predictions
##################
'''
labels = {}
for i in range(len(predicted_results1)):
    f = image_ID1[i][image_ID1[i].rindex("/")+1:]
    index = int(f.split(".")[0])
    temp = predicted_results1[i] + predicted_results2[i] + predicted_results3[i]
    category = name_to_index_map[temp.argmax()]
    labels[index]=(f,category)
    #print("Image:{}, Category {}".format(image_ID[i],category))
f = open(PREDICTION_FILE_NAME,"w")
f.write("image_name,category\n")
keys = labels.keys()
keys = sorted(keys)
for i in keys:
    (name,category)=labels[i]
    f.write("%s,%s\n"%(name,category))
f.close()
'''