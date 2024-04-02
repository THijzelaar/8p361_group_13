'''
TU/e BME Project Imaging 2021
Convolutional neural network for PCAM
Author: Suzanne Wetstein
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

# unused for now, to be used for ROC analysis
from sklearn.metrics import roc_curve, auc


# the size of the images in the PCAM dataset
IMAGE_SIZE = 96

import tensorflow as tf
print(tf.test.is_built_with_cuda())
print(tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None))
gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus, tf.version)
tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)


def get_pcam_generators(base_dir, train_batch_size=32, val_batch_size=32, rand_aug=True):
        # Data generator objects (Taken from course github 8P361)
        # dataset parameters
        train_path = os.path.join(base_dir, 'train+val', 'train')
        valid_path = os.path.join(base_dir, 'train+val', 'valid')


        RESCALING_FACTOR = 1./255

        # instantiate data generators
        
        MAX_DELTA = 2.0
    
        def generator(image, label):
            return (image, label), (label, image)
    
        def random_brightness(x, y):
            return tf.image.random_brightness(x, max_delta=MAX_DELTA), y
        
        def random_flip_hor(x, y):
            return tf.image.random_flip_left_right(x), y # 50% of flipping
        
        def random_flip_vert(x,y):
            return tf.image.random_flip_up_down(x), y # 50% of flipping
   
        
        train_gen = tf.keras.utils.image_dataset_from_directory(train_path,
                                                image_size=(IMAGE_SIZE, IMAGE_SIZE),
                                                batch_size=train_batch_size,
                                                label_mode='binary')

        val_gen = tf.keras.utils.image_dataset_from_directory(valid_path,
                                                image_size=(IMAGE_SIZE, IMAGE_SIZE),
                                                batch_size=val_batch_size,
                                                label_mode='binary')
        
        
        
        train_gen = train_gen.map(lambda x, y: (x * RESCALING_FACTOR, y))

        # Random data augmentations
        if rand_aug:
            train_gen = train_gen.map(random_brightness)
            train_gen = train_gen.map(random_flip_hor)
            train_gen = train_gen.map(random_flip_vert)
        train_gen = train_gen.map(generator)
        
        
        val_gen = val_gen.map(lambda x, y: (x * RESCALING_FACTOR, y))
        val_gen = val_gen.map(generator)

        train_gen = train_gen.prefetch(-1)
        val_gen = val_gen.prefetch(-1)
        

        return train_gen, val_gen



def get_model(kernel_size=(3,3), pool_size=(4,4), first_filters=32, second_filters=64, third_filters=128, learning_rate = 5e-5):

     # build the model
     model = Sequential()

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


data_path = r'C:\Users\20203080\Documents\GitHub\8p361_group_13\Data'
save_dir = './bin/'
# save the model and weights
model_name = save_dir + 'cnn_model_'
model_filepath = model_name + '.json'
weights_filepath = model_name + '_weights.hdf5'

# train model for 10, 20 epochs. lr of 5e-5, 1e-4 and batchsize of 32 and 64
settings = [(10, 5e-5, 32), (10, 5e-5, 64), (10, 1e-4, 32), (20, 5e-5, 32)]
name = ['base', 'bs_64', 'lr_1e-4', '20_epochs']
for i in range(4):
     model_name = name[i]
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
     train_steps = 144000//train_gen.batch_size
     val_steps = 16000//val_gen.batch_size

     history = model.fit(train_gen, steps_per_epoch=train_steps,
                         validation_data=val_gen,
                         validation_steps=val_steps,
                         epochs=settings[i][0],
                         callbacks=callbacks_list)
     print('Training done for ', model_name)