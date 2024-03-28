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
MODEL_WEIGHTS_FILEPATH = r'C:\Users\20203080\Documents\GitHub\8p361_group_13\bin\efficient_capsnetdynamic_routing_new_train.h5'
model_name = "dynamic_routing"
model_test = EfficientCapsNet(model_name, mode='test', verbose=True)
# load model and model weights


# load weights into new model
model_test.model.load_weights(MODEL_WEIGHTS_FILEPATH)
model = model_test.model

# open the test set in batches (as it is a very big dataset) and make predictions
test_files = glob.glob(os.path.join(TEST_PATH, '*.tif'))

submission = pd.DataFrame()


file_batch = 1000
max_idx = len(test_files)

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
    predictions = np.argmax(predictions, axis = 1)
    test_df['label'] = predictions
    submission = pd.concat([submission, test_df[['id', 'label']]])


# save your submission
submission.head()
submission.to_csv('submission.csv', index = False, header = True)