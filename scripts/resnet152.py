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
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Flatten, Activation, add, Dropout
from keras.applications.inception_v3 import preprocess_input
import h5py #needed for loading keras weights
import tensorflow as tf
import re
import pickle
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras import initializers
from keras.engine import Layer, InputSpec
from keras import backend as K

print(tf.__version__)
print(keras.__version__)
'''
#Home config
TRAIN_DIR = "/home/calvin/Documents/NNDL/Data/Training_data"
VALIDATION_DIR = "/home/calvin/Documents/NNDL/Data/Validation_data"
MODEL_PATH = "/home/calvin/Documents/NNDL/Models/res152_fc_40.h5"
LOG_DIR = "/home/calvin/Documents/NNDL/Log/"
CSV_DIR = "/home/calvin/Documents/NNDL/CSV/"
WEIGHTS_PATH = "/home/calvin/Documents/NNDL/Models/resnet152_fc_40_weights.h5"
NEW_WEIGHTS_PATH = '/home/calvin/Documents/NNDL/Models/resnet152_res4b28_40_weights.h5'
'''
#nscc config
TRAIN_DIR = "/home/users/nus/e0227268/kaggle/Data/Train/"
VALIDATION_DIR = "/home/users/nus/e0227268/kaggle/Data/Validate/"
TEST_DIR = "/home/users/nus/e0227268/kaggle/Data/Validate/Test/"
MODEL_PATH = "/home/users/nus/e0227268/kaggle/Models/res152_res4b28_40.h5"
LOG_DIR = "/home/users/nus/e0227268/kaggle/Logs/"
CSV_DIR = "/home/users/nus/e0227268/kaggle/CSV/"
WEIGHTS_PATH = '/home/users/nus/e0227268/kaggle/Models/resnet152_res4b20_40_weights.h5'
NEW_WEIGHTS_PATH = '/home/users/nus/e0227268/kaggle/Models/resnet152_all_40_weights.h5'

#Variables
CSV_NAME = 'res152_all_40.csv'
BATCH_SIZE = 16
NUM_CLASSES = 132
DROPOUT_RATE = 0.4
IMG_H = 224
IMG_W = 224

class Scale(Layer):
    '''Custom Layer for ResNet used for BatchNormalization.
    
    Learns a set of weights and biases used for scaling the input data.
    the output consists simply in an element-wise multiplication of the input
    and a sum of a set of constants:

        out = in * gamma + beta,

    where 'gamma' and 'beta' are the weights and biases larned.

    # Arguments
        axis: integer, axis along which to normalize in mode 0. For instance,
            if your input tensor has shape (samples, channels, rows, cols),
            set axis to 1 to normalize per feature map (channels axis).
        momentum: momentum in the computation of the
            exponential average of the mean and standard deviation
            of the data, for feature-wise normalization.
        weights: Initialization weights.
            List of 2 Numpy arrays, with shapes:
            `[(input_shape,), (input_shape,)]`
        beta_init: name of initialization function for shift parameter
            (see [initializers](../initializers.md)), or alternatively,
            Theano/TensorFlow function to use for weights initialization.
            This parameter is only relevant if you don't pass a `weights` argument.
        gamma_init: name of initialization function for scale parameter (see
            [initializers](../initializers.md)), or alternatively,
            Theano/TensorFlow function to use for weights initialization.
            This parameter is only relevant if you don't pass a `weights` argument.
    '''
    def __init__(self, weights=None, axis=-1, momentum = 0.9, beta_init='zero', gamma_init='one', **kwargs):
        self.momentum = momentum
        self.axis = axis
        self.beta_init = initializers.get(beta_init)
        self.gamma_init = initializers.get(gamma_init)
        self.initial_weights = weights
        super(Scale, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        shape = (int(input_shape[self.axis]),)

        self.gamma = K.variable(self.gamma_init(shape), name='%s_gamma'%self.name)
        self.beta = K.variable(self.beta_init(shape), name='%s_beta'%self.name)
        self.trainable_weights = [self.gamma, self.beta]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def call(self, x, mask=None):
        input_shape = self.input_spec[0].shape
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[self.axis] = input_shape[self.axis]

        out = K.reshape(self.gamma, broadcast_shape) * x + K.reshape(self.beta, broadcast_shape)
        return out

    def get_config(self):
        config = {"momentum": self.momentum, "axis": self.axis}
        base_config = super(Scale, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def identity_block(input_tensor, kernel_size, filters, stage, block):
    '''The identity_block is the block that has no conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    '''
    eps = 1.1e-5
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    scale_name_base = 'scale' + str(stage) + block + '_branch'

    x = Conv2D(nb_filter1, (1, 1), name=conv_name_base + '2a', use_bias=False)(input_tensor)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2a')(x)
    x = Activation('relu', name=conv_name_base + '2a_relu')(x)

    x = ZeroPadding2D((1, 1), name=conv_name_base + '2b_zeropadding')(x)
    x = Conv2D(nb_filter2, (kernel_size, kernel_size), name=conv_name_base + '2b', use_bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2b')(x)
    x = Activation('relu', name=conv_name_base + '2b_relu')(x)

    x = Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c', use_bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2c')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2c')(x)

    x = add([x, input_tensor], name='res' + str(stage) + block)
    x = Activation('relu', name='res' + str(stage) + block + '_relu')(x)
    return x

def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    '''conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
    And the shortcut should have subsample=(2,2) as well
    '''
    eps = 1.1e-5
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    scale_name_base = 'scale' + str(stage) + block + '_branch'

    x = Conv2D(nb_filter1, (1, 1), strides=strides, name=conv_name_base + '2a', use_bias=False)(input_tensor)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2a')(x)
    x = Activation('relu', name=conv_name_base + '2a_relu')(x)

    x = ZeroPadding2D((1, 1), name=conv_name_base + '2b_zeropadding')(x)
    x = Conv2D(nb_filter2, (kernel_size, kernel_size),
                      name=conv_name_base + '2b', use_bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2b')(x)
    x = Activation('relu', name=conv_name_base + '2b_relu')(x)

    x = Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c', use_bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2c')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2c')(x)

    shortcut = Conv2D(nb_filter3, (1, 1), strides=strides,
                             name=conv_name_base + '1', use_bias=False)(input_tensor)
    shortcut = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '1')(shortcut)
    shortcut = Scale(axis=bn_axis, name=scale_name_base + '1')(shortcut)

    x = add([x, shortcut], name='res' + str(stage) + block)
    x = Activation('relu', name='res' + str(stage) + block + '_relu')(x)
    return x

def resnet152_model(weights_path=None):
    '''Instantiate the ResNet152 architecture,
    # Arguments
        weights_path: path to pretrained weight file
    # Returns
        A Keras model instance.
    '''
    eps = 1.1e-5

    # Handle Dimension Ordering for different backends
    global bn_axis
    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
        img_input = Input(shape=(224, 224, 3), name='data')
    else:
        bn_axis = 1
        img_input = Input(shape=(3, 224, 224), name='data')
            
    x = ZeroPadding2D((3, 3), name='conv1_zeropadding')(img_input)
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', use_bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name='bn_conv1')(x)
    x = Scale(axis=bn_axis, name='scale_conv1')(x)
    x = Activation('relu', name='conv1_relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), name='pool1')(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    for i in range(1,8):
        x = identity_block(x, 3, [128, 128, 512], stage=3, block='b'+str(i))

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    for i in range(1,36):
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b'+str(i))

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    x_fc = AveragePooling2D((7, 7), name='avg_pool')(x)
    x_fc = Flatten()(x_fc)
    x_fc = Dense(132, activation='softmax', name='fc132')(x_fc)

    model = Model(img_input, x_fc)
    
    # load weights
    if weights_path:
        model.load_weights(weights_path, by_name=True)

    return model

#retrieve weights from model file
model1 = keras.models.load_model(MODEL_PATH,custom_objects={"Scale": Scale()})
model1.save_weights(WEIGHTS_PATH)

#model1 = resnet152_model(weights_path=r'/home/calvin/Documents/NNDL/Models/resnet152_weights_tf.h5')
model1 = resnet152_model(weights_path=WEIGHTS_PATH)

seed = np.random.randint(0, high=1000, size=1)[0]

'''
model1.summary()
model1.layers.pop()
model1.layers.pop()
model1.layers.pop()
model1.outputs = model1.layers[-1].output
x = model1.outputs
x= AveragePooling2D((7, 7), name='avg_pool')(x)
x = Flatten()(x)
x = Dropout(rate=DROPOUT_RATE, noise_shape=((BATCH_SIZE,2048)), seed=seed)(x)
x = Dense(NUM_CLASSES, activation='softmax', name='fc132')(x)
model2= Model(inputs=model1.input, outputs=x)
model2.summary()
model2.save_weights(NEW_WEIGHTS_PATH)
'''


##preprocessing
#create image data generator, specify preprocessing
train_datagen = image.ImageDataGenerator(
        width_shift_range=0.5,
        height_shift_range=0.5,
        shear_range=0.3,
        zoom_range=0.3,
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
#MINI_BATCHES = 1

#create image data generator for validation data
validation_datagen = image.ImageDataGenerator(preprocessing_function=preprocess_input)

#generate batches of data for training
validation_data_flow = validation_datagen.flow_from_directory(
        directory = VALIDATION_DIR,
        target_size = (IMG_H, IMG_W),
        class_mode = "categorical",
        batch_size = BATCH_SIZE)

VALIDATION_STEPS = int(np.floor(validation_data_flow.n/BATCH_SIZE))
#VALIDATION_STEPS = 1

#setup callbacks
checkpointer = keras.callbacks.ModelCheckpoint(filepath=MODEL_PATH,
                                               verbose=1,
                                               save_best_only=True)
early_stop = keras.callbacks.EarlyStopping(patience=20,
                                           verbose=1)
csvlogger = keras.callbacks.CSVLogger(os.path.join(CSV_DIR,CSV_NAME),
                                      separator=',',
                                      append=False)

#No dropout
#freeze all convolutional layers
for layer in model1.layers:
    layer.trainable = True
'''
#Train FC only
#for layer in model1.layers[-148:]:
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

#Train 8 CNN blocks
for layer in model1.layers[-121:]:
    layer.trainable = True
    print(layer)

#4b28add
for layer in model1.layers[-153:]:
    layer.trainable = True
    print(layer)
#4b26add
for layer in model1.layers[-181:]:
    layer.trainable = True
    print(layer)
#4b20add
for layer in model1.layers[-265:]:
    layer.trainable = True
    print(layer)
'''
#train
#adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-6)
sgd = SGD(lr=0.0001, momentum=0.9, decay=1e-6, nesterov=True)
#Model compile
model1.compile(optimizer=sgd, loss='categorical_crossentropy',metrics=['accuracy'])
#Model train
model1.fit_generator(train_data_flow,steps_per_epoch=MINI_BATCHES,epochs=30,verbose=1,callbacks=[checkpointer,early_stop,csvlogger],validation_data=validation_data_flow, validation_steps=VALIDATION_STEPS)

#model1.save(MODEL_PATH)
model1.save_weights(NEW_WEIGHTS_PATH)
print("model saved")
