import os
import datetime
import math
import keras
import numpy as np
from keras.utils import np_utils
from keras.optimizers import SGD
from keras.preprocessing import image
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
import h5py #needed for loading keras weights
print(keras.__version__)

TIMESTAMP = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
NUM_CLASSES = 132
SAVED_MODEL = "/home/users/nus/e0227268/kaggle/models/model1.h5"

model = keras.models.load_model(SAVED_MODEL)