# Copyright 2021 Vittorio Mazzia & Francesco Salvetti. All Rights Reserved.
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

import numpy as np
import tensorflow as tf
from utils.layers import PrimaryCaps, FCCaps, Length
from utils.tools import get_callbacks, marginLoss, multiAccuracy
from utils.dataset import Dataset
from models import efficient_capsnet_graph_dynamic_routing, efficient_capsnet_graph_self_attention, 
import os
import json
from tqdm.notebook import tqdm


class Model(object):
    """
    A class used to share common model functions and attributes.
    
    ...
    
    Attributes
    ----------
    model_name: str
        name of the model (Ex. 'MNIST')
    mode: str
        model modality (Ex. 'test')
    config_path: str
        path configuration file
    verbose: bool
    
    Methods
    -------
    load_config():
        load configuration file
    load_graph_weights():
        load network weights
    predict(dataset_test):
        use the model to predict dataset_test
    evaluate(X_test, y_test):
        comute accuracy and test error with the given dataset (X_test, y_test)
    save_graph_weights():
        save model weights
    """
    def __init__(self, model_name, mode='test', config_path='config.json', verbose=True):
        self.model_name = model_name
        self.model = None
        self.mode = mode
        self.config_path = config_path
        self.config = None
        self.verbose = verbose
        self.load_config()


    def load_config(self):
        """
        Load config file
        """
        with open(self.config_path) as json_data_file:
            self.config = json.load(json_data_file)
    

    def load_graph_weights(self):
        try:
            self.model.load_weights(self.model_path)
        except Exception as e:
            print("[ERRROR] Graph Weights not found")
            
        
    def predict(self, dataset_test):
        return self.model.predict(dataset_test)
    

    def evaluate(self, X_test, y_test):
        # Evaluate the model
        print('-'*30 + f'{self.model_name} Evaluation' + '-'*30)
        # Predict the test set
        y_pred, X_gen =  self.model.predict(X_test)
        # Calculate accuracy
        acc = np.sum(np.argmax(y_pred, 1) == np.argmax(y_test, 1))/y_test.shape[0]
        # Conver to error rate
        test_error = 1 - acc
        print('Test acc:', acc)
        print(f"Test error [%]: {(test_error):.4%}")
        print(f"N° misclassified images: {int(test_error*len(y_test))} out of {len(y_test)}")


    def save_graph_weights(self):
        self.model.save_weights(self.model_path)



class EfficientCapsNet(Model):
    """
    A class used to manage an Efficiet-CapsNet model. 'model_name' and 'mode' define the particular architecure and modality of the 
    generated network.
    
    ...
    
    Attributes
    ----------
    model_name: str
        name of the model (Ex. 'MNIST')
    mode: str
        model modality (Ex. 'test')
    config_path: str
        path configuration file
    custom_path: str
        custom weights path
    verbose: bool
    
    Methods
    -------
    load_graph():
        load the network graph given the model_name
    train(dataset, initial_epoch)
        train the constructed network with a given dataset. All train hyperparameters are defined in the configuration file

    """
    def __init__(self, model_name, mode='test', config_path='config.json', custom_path=None, verbose=True):
        Model.__init__(self, model_name, mode, config_path, verbose)
        if custom_path != None:
            self.model_path = custom_path
        else:
            self.model_path = os.path.join(self.config['saved_model_dir'], f"efficient_capsnet_{self.model_name}.h5")
        self.model_path_new_train = os.path.join(self.config['saved_model_dir'], f"efficient_capsnet{self.model_name}_new_train.h5")
        self.tb_path = os.path.join(self.config['tb_log_save_dir'], f"efficient_capsnet_{self.model_name}")
        self.load_graph()
    

    def load_graph(self):
        # Load the network graph
        if self.model_name == 'self_attention':
            self.model = efficient_capsnet_graph_self_attention.build_graph(self.config['8P361_INPUT_SHAPE'], self.mode, self.verbose)
        elif self.model_name == 'dynamic_routing':
            self.model = efficient_capsnet_graph_dynamic_routing.build_graph(self.config['8P361_INPUT_SHAPE'], self.mode, self.verbose)  
        else:
            raise ValueError(f"Model {self.model_name} not implemented")
            
    def train(self, dataset=None, initial_epoch=0):
        # Get the callbacks to track the training, save the model and update learning rate
        callbacks = get_callbacks(self.tb_path, self.model_path_new_train, self.config['lr_dec'], self.config['lr'])

        if dataset == None:
            dataset = Dataset(self.model_name, self.config_path)
        # Load dataset
        dataset_train, dataset_val = dataset.get_tf_data()    

        # Compile the model settings
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.config['lr']),
            loss=[marginLoss, 'mse'],
            loss_weights=[1., self.config['lmd_gen']],
            metrics={'Efficient_CapsNet': 'accuracy'})
        steps=None

        print('-'*30 + f'{self.model_name} train' + '-'*30)

        history = self.model.fit(dataset_train,
          epochs=self.config[f'epochs'], steps_per_epoch=steps,
          validation_data=dataset_val, batch_size=self.config['batch_size'], initial_epoch=initial_epoch,
          callbacks=callbacks)
        
        return history

            
        