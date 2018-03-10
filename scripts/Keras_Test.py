
# coding: utf-8

#updates
#1) added scaling of pixel values
#2) 
# In[19]:


import os
import keras
import numpy as np
from keras.utils import np_utils
from keras.optimizers import SGD
from keras.preprocessing import image
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import load_model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
import h5py #needed for loading keras weights
print(keras.__version__)



# In[2]:


base_model = InceptionV3(weights='imagenet', include_top=False)


# In[3]:


# 132 categories
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
predictions = Dense(NUM_CLASSES, activation='softmax')(x)
V3 = Model(inputs=base_model.input, outputs=predictions)
#Only allowing these new layest to be trained
for layer in base_model.layers: layer.trainable = False
sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
V3.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
datagen = image.ImageDataGenerator(
        rescale = 1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode = "nearest")


# In[ ]:


if os.path.exists(SAVED_MODEL):
    V3 = load_model(SAVED_MODEL)


# In[10]:


# does not work yet, using datagen.flow_from_directory instead
def load_image(img_path):
    #inception v3 input size is 299X299
    img = image.load_img(img_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = x.reshape(x.shape)
    x = x.reshape((1,) + x.shape)
    return x

def load_train_xy(img_path,category):
    x_train = load_image(img_path)
    y_train = np_utils.to_categorical(category, NUM_CLASSES)
    return (x_train,y_train)


# In[5]:


# does not work yet, using datagen.flow_from_directory instead
def load_train_set(csv_path):
    f = open(csv_path,"r")
    x_train = []
    y_train = []
    for l in f.readlines():
        arr = l.strip().split(",")
        filename = os.path.join(TRAIN_DIR,arr[0])
        category = int(arr[1])
        x,y = load_train_xy(filename,category)
        x_train.append(x)
        y_train.append(y)
    f.close()
    return (x_train,y_train)


# In[18]:


#(x_train,y_train) = load_train_set("subset_train.csv")
train_generator = datagen.flow_from_directory(
        directory = TRAIN_DIR,
        target_size=(299,299),
        batch_size=32,
        class_mode='categorical')
#do some simple training, parameters are artifically small so that my laptop don't die
#steps_per_epoch: Total number of steps (batches of samples) to yield from generator before declaring one epoch finished and starting the next epoch. It should typically be equal to the number of unique samples of your dataset divided by the batch size.
V3.fit_generator(train_generator,  epochs=3, verbose=1, callbacks=None, validation_data=None, shuffle=True, class_weight=None, initial_epoch=0, steps_per_epoch=10, validation_steps=800)
V3.save(SAVED_MODEL)


# In[23]:

#channel last array format (samples, height, width, channels)
x_test = load_image(os.path.join(TRAIN_DIR,"0.jpg"))
predictions = V3.predict(x_test, batch_size=1, verbose=1)
predicted_category = np.argmax(predictions)
print(predicted_category)


# In[24]:
#Extract weights from upper layers in InceptionV3 model

