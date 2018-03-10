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

#Home config
TRAIN_DIR = "/home/calvin/Documents/NNDL/Data/Training_data"
VALIDATION_DIR = "/home/calvin/Documents/NNDL/Data/Validation_data"
TEST_DIR = r"/home/calvin/Documents/NNDL/Data/temp_test/"
MODEL_PATH1 = r"/home/calvin/Documents/NNDL/Models/incep_40.h5"
#LOG_DIR = r"C:/Users/cjiaen/Documents/Sem1/CS5242_NNDL/Kaggle/Log/"
#CSV_DIR = r"C:/Users/cjiaen/Documents/Sem1/CS5242_NNDL/Kaggle/CSV/"

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

#Save model without weights
MODEL_PATH = r'C:\Users\cjiaen\Documents\Sem1\CS5242_NNDL\Kaggle\Models\incepv3_do_40_all.h5'
model1 = keras.models.load_model(MODEL_PATH)
for layer in model1.layers:
    layer.trainable = False
sgd = SGD(lr=0.0001, momentum=0.9, decay=1e-6, nesterov=True)
model1.compile(optimizer=sgd, loss='categorical_crossentropy',metrics=['accuracy'])
model1.save("incep_40.h5")
'''

#Variables
#PREDICTION_FILE_NAME = 'predictions_incep20_30_40.csv'
PREDICTION_FILE_NAME1 = 'predictions_incep_40.csv'
MODEL1_RAW_OUTPUT = 'raw_pred_incep_40.csv'
BATCH_SIZE = 128
NUM_CLASSES = 132
IMG_H = 299
IMG_W = 299

#load models
model1 = keras.models.load_model(MODEL_PATH1)


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

#mapping_dict = train_data_flow.class_indices
#name_to_index_map = dict(map(reversed, mapping_dict.items()))
name_to_index_map = {0: '0', 1: '1', 2: '10', 3: '100', 4: '101', 5: '102', 6: '103', 7: '104', 8: '105', 9: '106', 10: '107', 11: '108', 12: '109', 13: '11', 14: '110', 15: '111', 16: '112', 17: '113', 18: '114', 19: '115', 20: '116', 21: '117', 22: '118', 23: '119', 24: '12', 25: '120', 26: '121', 27: '122', 28: '123', 29: '124', 30: '125', 31: '126', 32: '127', 33: '128', 34: '129', 35: '13', 36: '130', 37: '131', 38: '14', 39: '15', 40: '16', 41: '17', 42: '18', 43: '19', 44: '2', 45: '20', 46: '21', 47: '22', 48: '23', 49: '24', 50: '25', 51: '26', 52: '27', 53: '28', 54: '29', 55: '3', 56: '30', 57: '31', 58: '32', 59: '33', 60: '34', 61: '35', 62: '36', 63: '37', 64: '38', 65: '39', 66: '4', 67: '40', 68: '41', 69: '42', 70: '43', 71: '44', 72: '45', 73: '46', 74: '47', 75: '48', 76: '49', 77: '5', 78: '50', 79: '51', 80: '52', 81: '53', 82: '54', 83: '55', 84: '56', 85: '57', 86: '58', 87: '59', 88: '6', 89: '60', 90: '61', 91: '62', 92: '63', 93: '64', 94: '65', 95: '66', 96: '67', 97: '68', 98: '69', 99: '7', 100: '70', 101: '71', 102: '72', 103: '73', 104: '74', 105: '75', 106: '76', 107: '77', 108: '78', 109: '79', 110: '8', 111: '80', 112: '81', 113: '82', 114: '83', 115: '84', 116: '85', 117: '86', 118: '87', 119: '88', 120: '89', 121: '9', 122: '90', 123: '91', 124: '92', 125: '93', 126: '94', 127: '95', 128: '96', 129: '97', 130: '98', 131: '99'}

#######################
#generate test results
#######################

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
    image_ID1[image_name] = re.sub(r"0/","",image_ID1[image_name])
df1 = pd.DataFrame(data=predicted_results1,
                   index=image_ID1,
                   columns=range(132))

df1.columns = name_to_index_map.values()
df1.to_csv(MODEL1_RAW_OUTPUT)

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
