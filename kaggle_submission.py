'''
TU/e BME Project Imaging 2021
Submission code for Kaggle PCAM
Author: Suzanne Wetstein
'''

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}   
import tensorflow as tf

import numpy as np

import glob
import pandas as pd
from matplotlib.pyplot import imread
from models import EfficientCapsNet
from tensorflow.keras.models import model_from_json

#Change these variables to point at the locations and names of the test dataset and your models.
TEST_PATH = r"./Data/test/"
models = ['base', 'bs_64', 'ep_20', 'lr_10e-5']
MODEL_WEIGHTS_FILEPATH = r"C:\Users\20203080\Documents\GitHub\8p361_group_13\bin\\"
model_type = 'CNN' # CNN or capsule

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(gpus[0], 'GPU')


# open the test set in batches (as it is a very big dataset) and make predictions
test_files = glob.glob(os.path.join(TEST_PATH, '*.tif'))



with tf.device('/GPU:0'):
    file_batch = 750
    max_idx = len(test_files)

    if model_type == 'capsule':
        for model_var in models:
            
            model_filepath = MODEL_WEIGHTS_FILEPATH + model_var + '.h5'
            model_name = "self_attention"
            model_test = EfficientCapsNet(model_name, mode='test', verbose=True)
            model_test.model.load_weights(model_filepath)
            model = model_test.model
            submission = pd.DataFrame()
            for idx in range(0, max_idx, file_batch):

                print('Indexes: %i - %i'%(idx, idx+file_batch))

                test_df = pd.DataFrame({'path': test_files[idx:idx+file_batch]})


                # get the image id 
                test_df['id'] = test_df.path.map(lambda x: x.split(os.sep)[-1].split('.')[0])
                test_df['image'] = test_df['path'].map(imread)
                
                
                K_test = np.stack(test_df['image'].values)
                
                # apply the same preprocessing as during draining
                K_test = K_test.astype('float')/255.0
                
                predictions = model.predict(K_test)[0]
                
                test_df['label'] = predictions[:,1]
                submission = pd.concat([submission, test_df[['id', 'label']]])


            # save your submission
            submission.head()
            submission.to_csv(f'submission_{model_var}.csv', index = False, header = True)

    elif model_type == 'CNN':
        #models = ['base', 'bs_64', '20_epochs', 'lr_10e-5']    
        models = [ 'bs_64', '20_epochs', 'lr_10e-5']    

        file_batch = 500
        max_idx = len(test_files)
        for model_var in models:
            # load model and model weights
            MODEL_FILEPATH = MODEL_WEIGHTS_FILEPATH + model_var + '.json'
            weights = MODEL_WEIGHTS_FILEPATH + model_var + '_weights.hdf5'
            json_file = open(MODEL_FILEPATH, 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            model = model_from_json(loaded_model_json)


            # load weights into new model
            model.load_weights(weights)
            submission = pd.DataFrame()
            for idx in range(0, max_idx, file_batch):
                
                print('Indexes: %i - %i'%(idx, idx+file_batch))

                test_df = pd.DataFrame({'path': test_files[idx:idx+file_batch]})


                # get the image id 
                test_df['id'] = test_df.path.map(lambda x: x.split(os.sep)[-1].split('.')[0])
                test_df['image'] = test_df['path'].map(imread)
                
                
                K_test = np.stack(test_df['image'].values)
                
                # apply the same preprocessing as during draining
                K_test = K_test.astype('float')/255.0
                
                predictions = model.predict(K_test)
                
                test_df['label'] = predictions
                submission = pd.concat([submission, test_df[['id', 'label']]])


            # save your submission
            submission.head()
            submission.to_csv(f'submission_{model_var}.csv', index = False, header = True)