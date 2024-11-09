# -*- coding: utf-8 -*-
"""main.ipynb"""

from src.read_data import ReadData
from src.prep_data import PrepDataForTraining
from src.training import Training
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

def main():

    indices = range(82, 1800, 24)
    results_dir = 'Results'

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

    for category in ['Residential', 'Commercial', 'Industrial', 'ResCom', 'ResComInd']:

        selected_columns = ['Date', 'MonthDay', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'WORKING'] + weather_pars + [category]

        for weather_lag in [6]:

            for pred_hr in [72]:

                for lag in [72]:

                    generated_timeseries = PrepDataForTraining(weather_pars)
                    timeseries_train, timeseries_valid, timeseries_test, norm_scaler_output, std_scaler_output, testing_data, testing_head_index = generated_timeseries.generate_timeseries(load, 2011, datetime.now().year+1, selected_columns, lag, weather_lag, pred_hr, category)
                    # Separate input and output features
                    X = timeseries_train.drop(columns=[category]).values
                    y = timeseries_train[category].values
                    X_valid = timeseries_valid.drop(columns=[category]).values
                    y_valid = timeseries_valid[category].values

                    model_name = category + '_deepFuture_' + str(weather_lag) + str(pred_hr) + str(lag)

                    trained_model = Training(lag)
                    ann_model = trained_model.develop_ann_model(X, y, X_valid, y_valid, model_name)

                    # Predict on test set
                    test_predictions_ann = ann_model.predict(timeseries_test.drop(columns=[category]))

                    # Plot actual vs predicted values for test set
                    plt.figure(figsize=(10, 6))
                    plt.scatter(timeseries_test[category], test_predictions_ann, color='purple', label='Test Data')
                    plt.xlabel('Actual Normalized-Standardized Load')
                    plt.ylabel('Predicted Normalized-Standardized Load')
                    plt.title('ANN: Actual vs Predicted (Test)')
                    plt.legend()
                    plt.grid(True)
                    plt.savefig(results_dir + '/' + 'pred_' + model_name + '.png')
                    plt.close()

                    prediction = PredictionAndQAPerformance(load, testing_head_index, results_dir)

                    for idx in indices:
                        prediction.print_rand_sample(timeseries_test, ann_model, model_name, idx, lag, pred_hr, norm_scaler_output, std_scaler_output, testing_data, category)

                    # Create empty DataFrames
                    benchmark_columns = ['index', 'Date', 'Hour', 'Temp', 'WORKING'] + [f'actual_{i}' for i in range(1, pred_hr + 1)] + [f'pred_{i}' for i in range(1, pred_hr + 1)]
                    residual_columns = ['index', 'Month'] + [f'residual_{i}' for i in range(1, pred_hr + 1)]

                    model_df = pd.DataFrame(columns=benchmark_columns)

                    for idx in tqdm(indices):
                        # Get actual, predicted, residual values, and the month
                        actuals, preds, residuals, month, date, hour, temp, working = prediction.evaluate_preds_at_diff_hrs(timeseries_test, ann_model, idx, lag, pred_hr, norm_scaler_output, std_scaler_output, testing_data, category)

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

if __name__ == "__main__":
    main()
