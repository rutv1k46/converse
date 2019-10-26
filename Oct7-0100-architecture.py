# This file is used to load back the architecture of the model
# the create_model() function recreates the model trained at a specifc time and specific date
# This date-time is specified by the filename.

#importing all the necessary libraries...
import numpy as np
from matplotlib import image
import matplotlib.pyplot as plt
import scipy.misc
import scipy
from scipy import ndimage

import argparse
import sys
import os,datetime
from IPython.display import SVG

#for preprocessing images...
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.python.framework import ops


#libraries and modules required to build and compile the model architecture...
from tensorflow.keras.models import Sequential

#All the layers that may be added/removed to form the architecture...
from tensorflow.keras.layers import Activation,Flatten
from tensorflow.keras.layers import LeakyReLU,GlobalAveragePooling2D
from tensorflow.keras.layers import Dense,Dropout,Conv2D,MaxPooling2D,BatchNormalization

#testing a bunch of optimizers and loss functions for better results...
from tensorflow.keras.optimizers import SGD,Adam
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.losses import softmax_cross_entropy

#finally the callbacks...

from tensorflow.keras.callbacks import ModelCheckpoint,TensorBoard,ReduceLROnPlateau




#Specifying the input dimensions...
#The ASL datset has 200x200 color images in RGB channels thus inputDims = 200x200x3
inputDims=(200,200,3)

#load this architecture first...
def create_model():
    model = Sequential()
    
    model.add(Conv2D(128,(3,3),input_shape=(200,200,3)))
    model.add(LeakyReLU(alpha=0.05))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.1))
    
    model.add(Conv2D(64,(3,3),input_shape=(200,200,3)))
    model.add(LeakyReLU(alpha=0.05))
    #model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.1))
    
    model.add(Conv2D(128,kernel_size=(3,3),strides=(2,2),input_shape=(200,200,3)))
    model.add(LeakyReLU(alpha=0.05))
    model.add(MaxPooling2D(pool_size=(3,3)))
    model.add(Dropout(0.3))

    model.add(Flatten())
    
    #model.add(Dense(128))
    #model.add(Activation("relu"))
    #model.add(Dropout(0.3))
    
    model.add(Dense(32))
    model.add(Activation("relu"))
    model.add(Dropout(0.2))
    
    model.add(Dense(29))
    model.add(Activation("softmax"))
    
    return model

#once the model is loaded with the weights,
def train_model(model):
    #[TODO] : import this from tf.losses...
    #[DONE]
    model.compile(loss=categorical_cross_entropy, metrics=['accuracy'], optimizer='Adadelta')
    
    #checkpoiniting models to get the best accuracy...
    checkpointer = ModelCheckpoint(filepath="/tmp/weights2.hdf5", verbose=1, save_best_only=True)
    
    #setting up a log directory for running tensorboard...
    log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    
    #creating a tensorboard callback to be passed to fit() or fit_generator()...
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    
    #creating a list of callbacks...
    callbacks = [tensorboard_callback,checkpointer]
    
    #training for 10 epocs with separate validation data and a set of callbacks...
    model.fit_generator(train,epochs=10,validation_data=test,callback = callbacks)
    
def save_trained_model(model):
    model.save("model-3Conv-2Dense-10Epoch.h5")

def load_trained_model(weights_path):
    model = create_model()
    model.load_weights(weights_path)
    return model
