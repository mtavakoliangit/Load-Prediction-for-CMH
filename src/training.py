# -*- coding: utf-8 -*-
"""training.py"""

from src.read_data import ReadData
from src.prep_data import PrepDataForTraining
from src.prediction import PredictionAndQAPerformance

# Ignore all warning messages
import warnings
warnings.filterwarnings("ignore")

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
    def __init__(self):
        self.model_dir = 'Models'
        self.results_dir = 'Results'

    def lr_schedule(self, epoch):
        initial_lr = 0.001
        drop_rate = 0.9
        return initial_lr * (drop_rate ** epoch)

    def develop_ann_model(self, X, y, X_valid, y_valid, model_name, lag):
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
        history = ann_model.fit(X, y, epochs=100, batch_size=lag, validation_data=(X_valid, y_valid), callbacks=[lr_scheduler], verbose=1)

        """Save the model:"""
        # Create a folder name based on the current datetime
        folder_name = datetime.now().strftime('%Y-%m-%d')
        new_folder_path = os.path.join(self.model_dir, folder_name)

        # Create the new folder for the newly developed model
        os.makedirs(new_folder_path, exist_ok=True)
        print(f'Folder path {new_folder_path} created for the new model.')
        
        model_path = new_folder_path + '/' + model_name + '.keras'
        ann_model.save(model_path)
        print(f'New model saved in path {model_path}.')

        return ann_model

    def build_model_feed(self, category, lag, weather_lag, pred_hr):
        data = ReadData('SourceData/HourlyConsumptionAgg.csv')
        residential_load = data.aggregate_category_data('Residential')
        commercial_load = data.aggregate_category_data('Commercial')
        industrial_load = data.aggregate_category_data('Industrial')

        # Filter the rows where ' ReadValue ' is less than 100kWh
        commercial_load = commercial_load[commercial_load[' ReadValue '] > 100]
        industrial_load = industrial_load[industrial_load[' ReadValue '] > 100]
        residential_load = residential_load[residential_load[' ReadValue '] > 100]

        # Rename the ' ReadValue ' column in commercial_load to 'Commercial'
        commercial_load.rename(columns={' ReadValue ': 'Commercial'}, inplace=True)

        # Rename the ' ReadValue ' column in residential_load to 'Residential'
        residential_load.rename(columns={' ReadValue ': 'Residential'}, inplace=True)

        # Rename the ' ReadValue ' column in industrial_load to 'Industrial'
        industrial_load.rename(columns={' ReadValue ': 'Industrial'}, inplace=True)

        # Merge residential_load and commercial_load and industrial_load on the 'DateTime' column
        load = residential_load.merge(commercial_load, on='DateTime', how='inner')
        load = load.merge(industrial_load, on='DateTime', how='inner')

        # We choose certain categories of our targets
        categories = ['Residential', 'Commercial', 'Industrial']

        # Sum up all categories
        load['ResCom'] = load['Residential'] + load['Commercial']
        load['ResComInd'] = load['ResCom'] + load['Industrial']

        weather_pars = ['mslPres', 'windSpd', 'dewPt', 'relHum', 'temp']
        selected_columns = ['Date', 'MonthDay', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'WORKING'] + weather_pars + [category]

        generated_timeseries = PrepDataForTraining(weather_pars)
        timeseries_train, timeseries_valid, timeseries_test, norm_scaler_output, std_scaler_output, testing_data, testing_head_index = generated_timeseries.generate_timeseries(load, 2011, datetime.now().year+1, selected_columns, lag, weather_lag, pred_hr, category)
        
        # Make some variables global which will be used by other functions too
        self.timeseries_test = timeseries_test
        self.load = load
        self.testing_head_index = testing_head_index
        self.norm_scaler_output = norm_scaler_output
        self.std_scaler_output = std_scaler_output
        self.testing_data = testing_data
        
        # Separate input and output features
        X = timeseries_train.drop(columns=[category]).values
        y = timeseries_train[category].values
        X_valid = timeseries_valid.drop(columns=[category]).values
        y_valid = timeseries_valid[category].values

        model_name = category + '_deepFuture_' + str(lag) + str(weather_lag) + str(pred_hr)

        return X, y, X_valid, y_valid, model_name

    def dev_and_evaluate_models_for_categories(self):
        indices = range(82, 1800, 24)

        for category in ['Residential', 'Commercial', 'Industrial', 'ResCom', 'ResComInd']:

            results_dir = os.path.join(self.results_dir+'/'+category, datetime.now().strftime('%Y-%m-%d'))
            # Create the new folder for the newly generated results
            os.makedirs(results_dir, exist_ok=True)
            print(f'Folder path {results_dir} created to save the new results.')

            for lag in [72]:
                for pred_hr in [72]:
                    for weather_lag in [6]:
                        X, y, X_valid, y_valid, model_name = self.build_model_feed(category, lag, weather_lag, pred_hr)
                        ann_model = self.develop_ann_model(X, y, X_valid, y_valid, model_name, lag)

                        # Predict on test set
                        test_predictions_ann = ann_model.predict(self.timeseries_test.drop(columns=[category]))

                        # Plot actual vs predicted values for test set
                        plt.figure(figsize=(10, 6))
                        plt.scatter(self.timeseries_test[category], test_predictions_ann, color='purple', label='Test Data')
                        plt.xlabel('Actual Normalized-Standardized Load')
                        plt.ylabel('Predicted Normalized-Standardized Load')
                        plt.title('ANN: Actual vs Predicted (Test)')
                        plt.legend()
                        plt.grid(True)
                        plt.savefig(results_dir + '/' + 'pred_' + model_name + '.png')
                        plt.close()

                        prediction = PredictionAndQAPerformance(self.load, self.testing_head_index, results_dir)

                        for idx in indices:
                            prediction.print_rand_sample(self.timeseries_test, ann_model, model_name, idx, lag, pred_hr, self.norm_scaler_output, self.std_scaler_output, self.testing_data, category)

                        # Create empty DataFrames
                        benchmark_columns = ['index', 'Date', 'Hour', 'Temp', 'WORKING'] + [f'actual_{i}' for i in range(1, pred_hr + 1)] + [f'pred_{i}' for i in range(1, pred_hr + 1)]
                        residual_columns = ['index', 'Month'] + [f'residual_{i}' for i in range(1, pred_hr + 1)]

                        model_df = pd.DataFrame(columns=benchmark_columns)

                        for idx in tqdm(indices):
                            # Get actual, predicted, residual values, and the month
                            actuals, preds, residuals, month, date, hour, temp, working = prediction.evaluate_preds_at_diff_hrs(self.timeseries_test, ann_model, idx, lag, pred_hr, self.norm_scaler_output, self.std_scaler_output, self.testing_data, category)

                            # Create new row for model_df
                            model_row = {
                                'index': idx,
                                'Date': date,
                                'Hour': hour,
                                'Temp': temp,
                                'WORKING': working,
                                **{f'actual_{i}': actuals[f't{i}'] for i in range(1, pred_hr + 1)},
                                **{f'pred_{i}': preds[f't{i}'] for i in range(1, pred_hr + 1)}
                            }

                            # Append new rows to the DataFrames
                            model_df = pd.concat([model_df, pd.DataFrame([model_row])], ignore_index=True)

                            # Save the DataFrames to CSV files
                            csv_filename = '{}/{}_withTemp.csv'.format(results_dir, model_name)
                            model_df.to_csv(csv_filename, index=False)
