# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 16:38:14 2017

@author: cjiaen

##Moves images into corresponding subfolders based on category
"""

import pandas as pd
import os
import glob
import re
import shutil
import numpy as np

#os.chdir(r"C:\Users\cjiaen\Documents\Sem1\CS5242_NNDL\Kaggle")
TRAIN_LABEL_PATH = "/home/users/nus/e0227268/kaggle/data/train.csv"
SOURCE_PATH = "/home/users/nus/e0227268/kaggle/data/train/transferred_train/"
TARGET_PATH = "/home/users/nus/e0227268/kaggle/data/train/"
VALIDATION_PATH = "/home/users/nus/e0227268/kaggle/data/validate/"
VALIDATION_RATIO = 0.1
#TRAIN_LABEL_PATH = r"C:/Users/cjiaen/Documents/Sem1/CS5242_NNDL/Kaggle/train.csv"
#SOURCE_PATH = r"C:/Users/cjiaen/Documents/Sem1/CS5242_NNDL/origin/"
#TARGET_PATH = r"C:/Users/cjiaen/Documents/Sem1/CS5242_NNDL/move_to/"
#VALIDATION_PATH = r"C:/Users/cjiaen/Documents/Sem1/CS5242_NNDL/test_to/"

##Examine distribution of labels
labels = pd.read_csv(TRAIN_LABEL_PATH)
label_count = labels.groupby("category").count()

##move images to respective subfolders based on category
for images in glob.glob(SOURCE_PATH + '*.jpg'):
    x = re.search('[0-9]*\.jpg', images).span()
    image_ID = images[x[0]:x[1]]
    category = np.asscalar(labels.loc[labels['image_name'] == image_ID,"category"])
    target = os.path.join(TARGET_PATH,str(category))
    if not os.path.exists(target):
        os.makedirs(target)
    shutil.move(os.path.join(SOURCE_PATH,image_ID), target)
    print("Moved file to folder {}".format(category))

#split images into training and validation set
subfolder_list = os.listdir(TARGET_PATH)
for subfolder_name in subfolder_list:
    total_num_files = len(os.listdir(os.path.join(TARGET_PATH,subfolder_name)))
    print("Total number of files in subfolder {} = {}".format(subfolder_name, total_num_files))
    validation_count = np.floor(VALIDATION_RATIO * total_num_files).astype(int)
    print(validation_count)
    image_list = os.listdir(os.path.join(TARGET_PATH,str(subfolder_name)))
    #randomly select images for validation
    np.random.shuffle(image_list)
    validation_list = image_list[:validation_count]
    for image in validation_list:
        destination = os.path.join(VALIDATION_PATH,str(subfolder_name))
        if not os.path.exists(destination):
            os.makedirs(destination)
        origin = os.path.join(TARGET_PATH, str(subfolder_name), image)
        shutil.move(origin, destination)
        print("Moved image {} to {}".format(image,destination))

