# Copyright 2021  Vittorio Mazzia & Francesco Salvetti. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""
Adapted code for the course 8P361 year 2024 at the technical university of Eindhoven

Group 13
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import json
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class Dataset(object):
    """
    A class used to share common dataset functions and attributes.
    
    ...
    
    Attributes
    ----------
    model_name: str
        name of the model (Ex. 'MNIST')
    config_path: str
        path configuration file
    
    Methods
    -------
    load_config():
        load configuration file
    get_dataset():
        load the dataset defined by model_name and pre_process it
    get_tf_data():
        get a tf.data.Dataset object of the loaded dataset. 
    """
    def __init__(self, model_name, image_size=96, config_path='config.json'):
        self.model_name = model_name
        self.config_path = config_path
        self.config = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.class_names = None
        self.X_test_patch = None
        self.load_config()
        self.image_size = image_size
        

    def load_config(self):
        """
        Load config file
        """
        with open(self.config_path) as json_data_file:
            self.config = json.load(json_data_file)
        
    def get_pcam_generators(self, base_dir, train_batch_size=32, val_batch_size=32, rand_aug=True):
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
                                                image_size=(self.image_size, self.image_size),
                                                batch_size=train_batch_size,
                                                label_mode='categorical')

        val_gen = tf.keras.utils.image_dataset_from_directory(valid_path,
                                                image_size=(self.image_size, self.image_size),
                                                batch_size=val_batch_size,
                                                label_mode='categorical')
        
        
        
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

    def get_tf_data(self):
        # Load the datasets without the need of inputting any variables.
        dataset_train, dataset_test = self.get_pcam_generators(self.config['pcam_path'], self.config['batch_size'], self.config['batch_size'])
        return dataset_train, dataset_test
