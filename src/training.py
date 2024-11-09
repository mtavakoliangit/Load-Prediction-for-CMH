# -*- coding: utf-8 -*-
"""training.ipynb"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import holidays
import os
import json
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tqdm import tqdm
import tensorflow as tf
import random
from sklearn.metrics import mean_squared_error, r2_score

class Training:
    def __init__(self, lag):
        self.lag = lag
        self.model_dir = 'Models'

    def lr_schedule(self, epoch):
        initial_lr = 0.001
        drop_rate = 0.9
        return initial_lr * (drop_rate ** epoch)

    def develop_ann_model(self, X, y, X_valid, y_valid, model_name):
        # Set the random seed for reproducibility
        seed = 142
        np.random.seed(seed)
        tf.random.set_seed(seed)
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)

        # Create the learning rate scheduler callback
        lr_scheduler = tf.keras.callbacks.LearningRateScheduler(self.lr_schedule)

        # Define the ANN model architecture
        ann_model = tf.keras.Sequential([
            tf.keras.layers.Dense(2048, activation='relu', input_shape=(X.shape[1],)),
            tf.keras.layers.Dense(1024, activation='relu'),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(8, activation='relu'),
            tf.keras.layers.Dense(1)
        ])

        # Compile the model with the initial learning rate
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
        ann_model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mean_squared_error'])

        # Train the ANN model with the learning rate scheduler
        history = ann_model.fit(X, y, epochs=100, batch_size=self.lag, validation_data=(X_valid, y_valid), callbacks=[lr_scheduler], verbose=1)

        """Save the model:"""
        model_path = self.model_dir + '/' + model_name + str(weather_lag) + str(pred_hr) + str(lag) + '.keras'
        ann_model.save(model_path)

        return ann_model
