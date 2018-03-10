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
SAVED_MODEL = "/home/users/nus/e0146089/food_v2a.h5"
TRAIN_DIR1 = "/home/users/nus/e0146089/train"
TRAIN_DIR2 = "/home/users/nus/e0146089/transferred_train_sorted"
EPOCH_CSV = "/home/users/nus/e0146089/train_v2a_epoch-%s.log"%TIMESTAMP

model = keras.models.load_model(SAVED_MODEL)

# Step 1 and Step 2 only used one time to initiate model for further training

#train_datagen = image.ImageDataGenerator(preprocessing_function=preprocess_input,rotation_range=45, width_shift_range=0.25, height_shift_range=0.25, horizontal_flip=True,vertical_flip=False,zoom_range=[.8, 1])
#batch_size = 128
#train_generator = train_datagen.flow_from_directory( TRAIN_DIR,target_size=(299, 299), batch_size=batch_size, class_mode='categorical')
#test_size = len(train_generator.filenames)
#steps_per_epoch = test_size//batch_size+1
#print("Step1")
#print("test_size=%d,batch_size=%d,steps_per_epoch=%d"%(test_size,batch_size,steps_per_epoch))
#model.fit_generator(train_generator,  epochs=10, verbose=1, class_weight=None, initial_epoch=0, steps_per_epoch=steps_per_epoch)
#model.save(SAVED_MODEL)
#model.save("/home/users/nus/e0146089/food_v2_toplayer.h5")

#print("Unfreezing")
#for layer in model.layers[:172]:
#   layer.trainable = False
#for layer in model.layers[172:]:
#   layer.trainable = True
#model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
#model.save(SAVED_MODEL)
#
#print("step2")

csvlogger = keras.callbacks.CSVLogger(EPOCH_CSV, separator=',', append=False)
checkpointer = keras.callbacks.ModelCheckpoint(filepath=SAVED_MODEL, verbose=1, save_best_only=False)

# total run time ~24 houts

# run training on preprocessed images. 30x the number of images in original training images. so only 1 epoch
# est time: 45697 seconds/epoch
batch_size = 64
train_datagen = image.ImageDataGenerator(preprocessing_function=preprocess_input,rotation_range=45, width_shift_range=0.25, height_shift_range=0.25, horizontal_flip=True,vertical_flip=False)
train_generator = train_datagen.flow_from_directory( TRAIN_DIR2,target_size=(299, 299), batch_size=batch_size, class_mode='categorical')
test_size = len(train_generator.filenames)
steps_per_epoch = int(math.ceil(test_size/float(batch_size)))
print("FIT 1: test_size=%d,batch_size=%d,steps_per_epoch=%d"%(test_size,batch_size,steps_per_epoch))
model.fit_generator(train_generator,  epochs=1, verbose=1, class_weight=None, initial_epoch=0, steps_per_epoch=steps_per_epoch, callbacks=[checkpointer,csvlogger])

# Do a short run on non-preprocessed training images
# est time: 1400 seconds/epoch
batch_size = 128
train_datagen = image.ImageDataGenerator(preprocessing_function=preprocess_input,rotation_range=45, width_shift_range=0.25, height_shift_range=0.25, horizontal_flip=True,vertical_flip=False,zoom_range=0.2)
train_generator = train_datagen.flow_from_directory( TRAIN_DIR1,target_size=(299, 299), batch_size=batch_size, class_mode='categorical')
test_size = len(train_generator.filenames)
steps_per_epoch = int(math.ceil(test_size/float(batch_size)))
print("FIT 2: test_size=%d,batch_size=%d,steps_per_epoch=%d"%(test_size,batch_size,steps_per_epoch))
model.fit_generator(train_generator,  epochs=30, verbose=1, class_weight=None, initial_epoch=1, steps_per_epoch=steps_per_epoch, callbacks=[checkpointer,csvlogger])
model.save(SAVED_MODEL)
