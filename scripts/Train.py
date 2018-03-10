import os
import datetime
import keras
import numpy as np
from keras.utils import np_utils
from keras.optimizers import SGD
from keras.preprocessing import image
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
import h5py #needed for loading keras weights
print(keras.__version__)

TIMESTAMP = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
NUM_CLASSES = 132
SAVED_MODEL = "/home/users/nus/e0146089/food.h5"
TRAIN_DIR = "/home/users/nus/e0146089/train"
EPOCH_CSV = "/home/users/nus/e0146089/train_epoch-%s.log"%TIMESTAMP

# IF USING NSCC: need to pre-create the model locally and upload to nscc. cannot download inception-v3 checkpoint from the internet when running in nscc
V3 = None
if os.path.exists(SAVED_MODEL):
    V3 = keras.models.load_model(SAVED_MODEL)
else:
    base_model = InceptionV3(weights='imagenet', include_top=False)
    # 132 categories
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(NUM_CLASSES, activation='softmax')(x)
    V3 = Model(inputs=base_model.input, outputs=predictions)
    #Only allowing these new layest to be trained
    for layer in base_model.layers: layer.trainable = False
    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    V3.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    
train_datagen = image.ImageDataGenerator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True)
val_datagen = image.ImageDataGenerator()
train_generator = train_datagen.flow_from_directory( TRAIN_DIR,target_size=(299, 299), batch_size=32, class_mode='categorical')
val_generator = val_datagen.flow_from_directory( TRAIN_DIR, target_size=(299,299), batch_size=32, class_mode='categorical')
csvlogger = keras.callbacks.CSVLogger(EPOCH_CSV, separator=',', append=False)
checkpointer = keras.callbacks.ModelCheckpoint(filepath=SAVED_MODEL, verbose=1, save_best_only=True)
# steps_per_epoch: Total number of steps (batches of samples) to yield from generator before declaring one epoch finished and starting the next epoch. It should typically be equal to the number of unique samples of your dataset divided by the batch size.
# about 30 minutes per epoch
V3.fit_generator(train_generator,  epochs=40, verbose=1, validation_data=val_generator, class_weight=None, initial_epoch=0, steps_per_epoch=1536, validation_steps=800, callbacks=[checkpointer,csvlogger])
#V3.save(SAVED_MODEL)

