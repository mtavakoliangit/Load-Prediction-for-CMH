# -*- coding: utf-8 -*-
"""prediction.py"""

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

class PredictionAndQAPerformance:
    def __init__(self, load, testing_head_index, results_dir):
        self.load = load
        self.testing_head_index = testing_head_index
        self.results_dir = results_dir

    def multiple_prediction_with_known_index(self, timeseries, lag, pred_hr, model, idx, category):
        # Choose a random starting index
        random_start_index = idx

        # Select the next rows following the random starting index
        random_rows = timeseries.copy().iloc[random_start_index:random_start_index + pred_hr]

        # Save the actual load before it changes
        actual_load = random_rows[category].copy()

        for hr in range(pred_hr):
            pred = model.predict(pd.DataFrame(random_rows.iloc[hr]).transpose().drop(columns=[category]), verbose=0)
            random_rows[category][random_start_index + hr] = pred
            hr_bk = random_start_index + hr
            for future_hr in range(1, lag+1):
                hr_bk = hr_bk + 1
                if hr_bk < random_start_index + pred_hr:
                    random_rows['load {}hr bk'.format(str(future_hr))][hr_bk] = pred

        return (actual_load, random_rows, random_start_index)

    def print_rand_sample(self, timeseries, model, model_name, idx, lag, pred_hr, norm_scaler_output, std_scaler_output, testing_data, category):
        actual_load, prediction, random_start_index = self.multiple_prediction_with_known_index(timeseries, lag, pred_hr, model, idx, category)
        print(f'Random index: {random_start_index}')

        # Extract the start_date from testing_data using the random_start_index
        start_date = self.load.iloc[self.testing_head_index:]['DateTime'][self.testing_head_index + random_start_index]

        # Date range for predictions
        date_range = pd.date_range(start=start_date, periods=pred_hr, freq='H')

        # Create the x-axis values starting from the first point of time and extending to pred_hr hours more
        x_values = pd.date_range(start=start_date, periods=pred_hr, freq='H')

        actual_load_dnormalized = norm_scaler_output.inverse_transform(pd.DataFrame(actual_load))
        prediction_dnormalized = norm_scaler_output.inverse_transform(pd.DataFrame(prediction[category]))

        actual_load_original = std_scaler_output.inverse_transform(actual_load_dnormalized) / 1000
        actual_load_original = pd.DataFrame(actual_load_original).transpose()
        actual_load_original.columns = ['t{}'.format(i) for i in range(1, pred_hr+1)]

        prediction_original = std_scaler_output.inverse_transform(prediction_dnormalized) / 1000
        prediction_original = pd.DataFrame(prediction_original).transpose()
        prediction_original.columns = ['t{}'.format(i) for i in range(1, pred_hr+1)]

        # Extract temperature data for the given time range
        temperature_data = testing_data['temp'].iloc[random_start_index + lag : random_start_index + lag + pred_hr].values

        # Create a new DataFrame with x_values, actual, pred, and temp columns
        data_df = pd.DataFrame({
            'x_values': x_values,
            'actual': actual_load_original.iloc[0].values,
            'pred': prediction_original.iloc[0].values,
            'temp': temperature_data
        })

        # Save the DataFrame to a CSV file, naming it with the first date value in x_values
        csv_filename = f"{self.results_dir}/{model_name}_{x_values[0].strftime('%Y-%m-%d_%H-%M')}.csv"
        data_df.to_csv(csv_filename, index=False)

        # Plot actual vs. predicted loads for hours ahead
        fig, ax1 = plt.subplots()

        # Primary y-axis for Load
        ax1.plot(x_values, actual_load_original.iloc[0], label='Actual Load', color='blue')
        ax1.plot(x_values, prediction_original.iloc[0], label='Predicted Load', color='red')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Load (MW)', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')

        # Secondary y-axis for Temperature
        ax2 = ax1.twinx()
        ax2.plot(x_values, temperature_data, label='Temperature', color='green', linestyle='--')
        ax2.set_ylabel('Temperature (°C)', color='green')
        ax2.tick_params(axis='y', labelcolor='green')

        # Legends for both y-axes
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')

        # Title and save the plot
        plt.title('Actual vs Pred Load ({}) and Temp. - {}'.format(category, start_date))
        plt.savefig(self.results_dir + '/' + model_name + '_rand_pred_withTemp' + str(idx) + '.png')
        plt.close()

    def evaluate_preds_at_diff_hrs(self, timeseries, model, idx, lag, pred_hr, norm_scaler_output, std_scaler_output, testing_data, category):
        actual_load, prediction, random_start_index = self.multiple_prediction_with_known_index(timeseries, lag, pred_hr, model, idx, category)
        print(f'Random index: {random_start_index}')

        # Extract the start_date from testing_data using the random_start_index
        start_date = testing_data.index[random_start_index]
        date_range = pd.date_range(start=start_date, periods=pred_hr, freq='H')

        # Normalize and inverse transform to get the original values
        actual_load_dnormalized = norm_scaler_output.inverse_transform(pd.DataFrame(actual_load))
        prediction_dnormalized = norm_scaler_output.inverse_transform(pd.DataFrame(prediction[category]))

        actual_load_original = std_scaler_output.inverse_transform(actual_load_dnormalized)
        actual_load_original = pd.DataFrame(actual_load_original).transpose()
        actual_load_original.columns = ['t{}'.format(i) for i in range(1, pred_hr + 1)]

        prediction_original = std_scaler_output.inverse_transform(prediction_dnormalized)
        prediction_original = pd.DataFrame(prediction_original).transpose()
        prediction_original.columns = ['t{}'.format(i) for i in range(1, pred_hr + 1)]

        # Calculate residuals (Actual - Predicted)
        residuals = actual_load_original - prediction_original
        residuals.columns = ['residual_{}'.format(i) for i in range(1, pred_hr + 1)]

        # Extract the month from the random_start_index
        month = testing_data['Date'].iloc[random_start_index].month
        date = testing_data['Date'].iloc[random_start_index]
        hour = (np.arctan2(testing_data['hour_sin'], testing_data['hour_cos']) * 24 / (2 * np.pi)) % 24
        temp = testing_data['temp'].iloc[random_start_index]
        working = testing_data['WORKING'].iloc[random_start_index]

        return actual_load_original.iloc[0], prediction_original.iloc[0], residuals.iloc[0], month, date, hour, temp, working
