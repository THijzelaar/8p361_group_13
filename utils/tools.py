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

"""
Adapted code for the course 8P361 year 2024 at the technical university of Eindhoven

Group 13
"""

import numpy as np
import tensorflow as tf

# Create a learning rate scheduler, 
# that decreases the learning rate by a factor of lr_dec every epoch
def learn_scheduler(lr_dec, lr):
    def learning_scheduler_fn(epoch):
        lr_new = lr * (lr_dec ** epoch)
        return lr_new if lr_new >= 5e-7 else 5e-7
    return learning_scheduler_fn

# Create tensorflow checkpoints, tensorboard logs and learning rate scheduler
def get_callbacks(tb_log_save_path, saved_model_path, lr_dec, lr):
    tb = tf.keras.callbacks.TensorBoard(log_dir=tb_log_save_path, histogram_freq=0)

    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(saved_model_path, monitor='val_Efficient_CapsNet_accuracy',
                                           save_best_only=True, save_weights_only=True, verbose=1)

    lr_decay = tf.keras.callbacks.LearningRateScheduler(learn_scheduler(lr_dec, lr))

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_CapsNet_accuracy', factor=0.9,
                              patience=4, min_lr=0.00001, min_delta=0.0001, mode='max')

    return [tb, model_checkpoint, lr_decay]

# Create a custom loss function
# The margin loss is designed to increase the margin between the true class and the other classes,
# so that the model can learn to distinguish between different classes.
def marginLoss(y_true, y_pred):
    # This is the lambda parameter, which is set to 0.5 in this code. 
    # It's a hyperparameter that controls the influence of the margin loss compared to the cross-entropy loss.
    lbd = 0.5
    # upper and lower margin
    m_plus = 0.9
    m_minus = 0.1
    
    L = y_true * tf.square(tf.maximum(0., m_plus - y_pred)) + \
    lbd * (1 - y_true) * tf.square(tf.maximum(0., y_pred - m_minus))

    return tf.reduce_mean(tf.reduce_sum(L, axis=1))

# Create a custom accuracy function
# This function calculates the accuracy of the model,
# by comparing the predicted label with the true label.

def multiAccuracy(y_true, y_pred):
    label_pred = tf.argsort(y_pred,axis=-1)[:,-2:]
    label_true = tf.argsort(y_true,axis=-1)[:,-2:]
    
    acc = tf.reduce_sum(tf.cast(label_pred[:,:1]==label_true,tf.int8),axis=-1) + \
          tf.reduce_sum(tf.cast(label_pred[:,1:]==label_true,tf.int8),axis=-1)
    acc /= 2
    return tf.reduce_mean(acc,axis=-1)
