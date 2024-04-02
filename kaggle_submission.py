'''
TU/e BME Project Imaging 2021
Submission code for Kaggle PCAM
Author: Suzanne Wetstein

Adapted by group 13 for the course 8P361
This file generates a submission file for the Kaggle PCAM challenge for the capsule network model
'''

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}   
import tensorflow as tf

import numpy as np

import glob
import pandas as pd
from matplotlib.pyplot import imread
from models import EfficientCapsNet

#Change these variables to point at the locations and names of the test dataset and your models.
TEST_PATH = r"./Data/test/"

MODEL_WEIGHTS_FILEPATH = r"./bin/efficient_capsnetself_attention_base.h5"

model_name = "self_attention" # chose self_attention or dynamic_routing
model_test = EfficientCapsNet(model_name, mode='test', verbose=True)
model_test.model.load_weights(MODEL_WEIGHTS_FILEPATH)
model = model_test.model



gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
file_batch = 750


submission = pd.DataFrame()

# open the test set in batches (as it is a very big dataset) and make predictions
test_files = glob.glob(os.path.join(TEST_PATH, '*.tif'))
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
                
                test_df['label'] = predictions[:,1]
                submission = pd.concat([submission, test_df[['id', 'label']]])
# save your submission
submission.head()
submission.to_csv(f'submission.csv', index = False, header = True)