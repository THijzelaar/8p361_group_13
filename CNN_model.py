'''
TU/e BME Project Imaging 2021
Convolutional neural network for PCAM
Author: Suzanne Wetstein

Adapted by group 13 for the course 8P361
This file trains the CNN model described in the report for each parameter variation
'''




# disable overly verbose tensorflow logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}   
import tensorflow as tf

import numpy as np

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.layers import Conv2D, MaxPool2D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

# the size of the images in the PCAM dataset


import tensorflow as tf
print(tf.test.is_built_with_cuda())
print(tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None))
gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus, tf.version)
tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)





def get_model(kernel_size=(3,3), pool_size=(4,4), first_filters=32, second_filters=64, learning_rate = 5e-5):

     # build the model
     model = Sequential()
     model.add(tf.keras.Input((IMAGE_SIZE, IMAGE_SIZE, 3)))
     model.add(Conv2D(first_filters, kernel_size, activation = 'relu', padding = 'same', input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)))
     model.add(MaxPool2D(pool_size = pool_size))

     model.add(Conv2D(second_filters, kernel_size, activation = 'relu', padding = 'same'))
     model.add(MaxPool2D(pool_size = pool_size))


     model.add(Flatten())
     model.add(Dense(75, activation = 'relu'))
     model.add(Dense(1, activation = 'sigmoid'))


     # compile the model
     model.compile(SGD(learning_rate=learning_rate, momentum=0.95), loss = 'binary_crossentropy', metrics=['accuracy'])

     return model





# train model for 10, 20 epochs. lr of 5e-5, 1e-4 and batchsize of 32 and 64

for i in range(len(settings)):
     model_name = name[i]
     
     model_filepath = model_name + '.json'
     weights_filepath = model_name + '_weights.hdf5'
     model_filepath = os.path.join(save_dir, model_name + '.json')
     weights_filepath = os.path.join(save_dir, model_name + '_weights.hdf5')

     # get the model
     model = get_model(learning_rate=settings[i][1])
     print('Model name: ', model_name)



     # get the data generators
     train_gen, val_gen = get_pcam_generators(data_path, train_batch_size=settings[i][2], val_batch_size=settings[i][2])


     model_json = model.to_json() # serialize model to JSON
     with open(model_filepath, 'w') as json_file:
          json_file.write(model_json)


     # define the model checkpoint and Tensorboard callbacks
     checkpoint = ModelCheckpoint(weights_filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
     tensorboard = TensorBoard(os.path.join('logs', model_name))
     callbacks_list = [checkpoint, tensorboard]


     # train the model
     train_steps = size_train//settings[i][2]
     val_steps = size_val//settings[i][2]

     history = model.fit(train_gen, steps_per_epoch=train_steps,
                         validation_data=val_gen,
                         validation_steps=val_steps,
                         epochs=settings[i][0],
                         callbacks=callbacks_list,
                         batch_size=settings[i][2])
     print('Training done for ', model_name)
