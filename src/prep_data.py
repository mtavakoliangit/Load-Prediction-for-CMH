# -*- coding: utf-8 -*-
"""prep_data.py"""

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

class PrepDataForTraining:
    def __init__(self, weather_pars):
        self.city = 'Medicine Hat'
        self.weather_pars = weather_pars

    def ab_holidays(self):
        # Get the bank holidays in Alberta for the years 2011 to 2024
        ab_holidays = holidays.Canada(years=range(2011, datetime.now().year), prov='AB')

        # Convert bank holidays to a pandas DataFrame
        ab_holidays_df = pd.DataFrame(ab_holidays.items(), columns=['DATE', 'HOLIDAY'])

        # Convert 'DATE' column to datetime
        ab_holidays_df['DATE'] = pd.to_datetime(ab_holidays_df['DATE'])

        # Return a list of AB holidays
        return ab_holidays_df['DATE']

    def is_weekend(self, date):
        return date.dayofweek in [5, 6]

    def prep_data_for_model_dev(self, df, yr_start, yr_end, selected_columns):
        df['Date'] = df['DateTime'].dt.date
        df['Hour'] = df['DateTime'].dt.hour
        df['Day'] = df['DateTime'].dt.dayofweek + 1

        df = df.drop(columns=['DateTime'])

        # Sort DataFrame based on 'Date' and then 'Hour'
        df.sort_values(by=['Date', 'Hour'], inplace=True)

        # Define the directory path
        directory_path = f"weathersource/"

        # Initialize an empty list to store DataFrames
        weather_dfs = []

        # Iterate over each year from 2011 to 2024
        for year in range(yr_start, yr_end):
              # Define the file path for the current year
              city_nospace = self.city.replace(' ','')
              file_path = os.path.join(directory_path, '{}-{}.json'.format(city_nospace, year))

              # Read the JSON file
              with open(file_path, 'r') as file:
                  data = json.load(file)

              # Extract data associated with the 'history' key
              history_data = data['history']

              # Convert history_data to DataFrame
              history_df = pd.DataFrame(history_data)

              # Split timestamp into date and time components
              history_df[['DATE', 'TIME']] = history_df['timestamp'].str.split('T', expand=True)
              history_df['HOUR'] = pd.to_datetime(history_df['TIME']).dt.hour

              # Drop the 'timestamp' and 'TIME' columns
              history_df.drop(columns=['timestamp', 'TIME'], inplace=True)

              # Reorder the columns to have 'DATE' and 'HOUR' as the first two columns
              history_df = history_df[['DATE', 'HOUR'] + [col for col in history_df.columns if col not in ['DATE', 'HOUR']]]

              # Append the DataFrame to the list
              weather_dfs.append(history_df)

        # Concatenate all DataFrames in the list
        city_weather_df = pd.concat(weather_dfs, ignore_index=True)

        # Convert 'DATE' column to datetime
        city_weather_df['DATE'] = pd.to_datetime(city_weather_df['DATE'])

        # Sort DataFrame by 'DATE'
        city_weather_df.sort_values(by=['DATE', 'HOUR'], inplace=True)

        # Reset index
        city_weather_df.reset_index(drop=True, inplace=True)

        # Convert 'DATE' column in city_load to datetime if it's not already
        df['Date'] = pd.to_datetime(df['Date'])

        # Join city_load with city_weather_df based on 'DATE' and 'Hour'
        city_df = pd.merge(df, city_weather_df, left_on=['Date', 'Hour'], right_on=['DATE', 'HOUR'], how='inner')

        # Create a new column 'MonthDay' to move alongth months smoothly
        city_df['MonthDay'] = city_df['DATE'].dt.month + city_df['DATE'].dt.day / 31

        # Create a new column 'YEAR' containing the month component
        city_df['YEAR'] = city_df['DATE'].dt.year

        # Get the index of the 'DATE' column
        date_index = city_df.columns.get_loc('DATE')

        # Move the 'YEAR' column to be right after the 'DATE' column
        city_df.insert(date_index + 1, 'YEAR', city_df.pop('YEAR'))

        # Move the 'MonthDay' column to be right after the 'YEAR' column
        city_df.insert(date_index + 2, 'MonthDay', city_df.pop('MonthDay'))

        # Rename the column
        city_df.rename(columns={self.city: 'Load'}, inplace=True)

        # Create a new column 'WORKING' with default value 1
        city_df['WORKING'] = 1

        # Set 'WORKING' to 0 for weekends
        city_df.loc[city_df['DATE'].apply(self.is_weekend), 'WORKING'] = 2

        # Set 'WORKING' to 0 for bank holidays
        city_df.loc[city_df['DATE'].isin(self.ab_holidays()), 'WORKING'] = 3

        # Get the index of the 'DATE' column
        date_index = city_df.columns.get_loc('Date')

        # Move the 'MONTH' column to be right after the 'DATE' column
        city_df.insert(date_index + 1, 'WORKING', city_df.pop('WORKING'))

        # Apply sine and cosine transformations
        city_df['hour_sin'] = np.sin(2 * np.pi * city_df['Hour'] / 24)
        city_df['hour_cos'] = np.cos(2 * np.pi * city_df['Hour'] / 24)
        city_df['day_sin'] = np.sin(2 * np.pi * city_df['Day'] / 7)
        city_df['day_cos'] = np.cos(2 * np.pi * city_df['Day'] / 7)

        return city_df[selected_columns]

    def convert_to_dataframe(self, data, category):
        return data.to_frame(name=category)

    def prep_data(self, theCity, category):
        # Split the data into training, validation, and testing sets
        training_data = theCity[(theCity['Date'].dt.year <= 2023)]
        validation_data = theCity[(theCity['Date'].dt.year == 2024) & (theCity['Date'].dt.month <= 4)]
        testing_data = theCity[(theCity['Date'].dt.year == 2024) & (theCity['Date'].dt.month > 4)]

        # Separate features (X) and target variable (y) for training, validation, and testing sets
        X_train = training_data.drop(columns=[category])
        y_train = training_data[category]
        X_validation = validation_data.drop(columns=[category])
        y_validation = validation_data[category]
        X_test = testing_data.drop(columns=[category])
        y_test = testing_data[category]

        # Drop the Date column as we don't need it during ML process
        X_train = X_train.drop(columns=['Date'])
        X_validation = X_validation.drop(columns=['Date'])
        X_test = X_test.drop(columns=['Date'])

        # Convert y_train to DataFrame
        y_train = self.convert_to_dataframe(y_train, category)

        # Convert y_validation to DataFrame
        y_validation = self.convert_to_dataframe(y_validation, category)

        # Convert y_test to DataFrame
        y_test = self.convert_to_dataframe(y_test, category)

        # Extract input and output names
        input_cols = X_train.columns
        output_col = y_train.columns

        # Standardize the input features
        std_scaler_input = StandardScaler()

        # Fit and transform input features for training set (Standardization)
        train_data_standardized_input = std_scaler_input.fit_transform(X_train[input_cols])

        # Transform input features for testing and validation sets (Standardization)
        test_data_standardized_input = std_scaler_input.transform(X_test[input_cols])
        validation_data_standardized_input = std_scaler_input.transform(X_validation[input_cols])

        # Normalize the standardized input features
        norm_scaler_input = MinMaxScaler()

        # Fit and transform input features for training set (Normalization)
        train_data_normalized_input = norm_scaler_input.fit_transform(train_data_standardized_input)

        # Transform input features for testing and validation sets (Normalization)
        test_data_normalized_input = norm_scaler_input.transform(test_data_standardized_input)
        validation_data_normalized_input = norm_scaler_input.transform(validation_data_standardized_input)

        # Standardize the output vector
        std_scaler_output = StandardScaler()

        # Fit and transform output feature for training set (Standardization)
        train_data_standardized_output = std_scaler_output.fit_transform(y_train[output_col])

        # Transform output feature for testing and validation sets (Standardization)
        test_data_standardized_output = std_scaler_output.transform(y_test[output_col])
        validation_data_standardized_output = std_scaler_output.transform(y_validation[output_col])

        # Normalize the standardized output vector
        norm_scaler_output = MinMaxScaler()

        # Fit and transform output feature for training set (Normalization)
        train_data_normalized_output = norm_scaler_output.fit_transform(train_data_standardized_output)

        # Transform output feature for testing and validation sets (Normalization)
        test_data_normalized_output = norm_scaler_output.transform(test_data_standardized_output)
        validation_data_normalized_output = norm_scaler_output.transform(validation_data_standardized_output)

        # Columns to build the timeseries dataframes
        input_cols = input_cols.tolist()

        return train_data_normalized_input, validation_data_normalized_input, test_data_normalized_input, input_cols, training_data, validation_data, testing_data, train_data_normalized_output, test_data_normalized_output, validation_data_normalized_output, norm_scaler_output, std_scaler_output

    def create_lagged_features(self, data, lag, leading=False):
        if leading:
            # Shift forward (future) correctly
            lagged_features = [data.shift(-i) for i in range(1, lag + 1)]
        else:
            # Shift backward (past) correctly
            lagged_features = [data.shift(i) for i in range(1, lag + 1)]
        return pd.concat([data] + lagged_features, axis=1)

    def fill_timeseries_incl_weatherHist_df(self, input, output, lag, weather_lag, pred_hr, input_cols, weather_pars, category):
        # Shifted dataframe for past load data
        lagged_load = self.create_lagged_features(pd.DataFrame(output)[0], lag)
        lagged_load = lagged_load.transpose().iloc[::-1].iloc[:, lag:]

        # Create a dictionary to store past weather parameters
        lagged_weather_pars_past = pd.DataFrame(columns=weather_pars, index=[0])
        for param in weather_pars:
            # Past weather parameters with the specified lag
            lagged_weather_past = self.create_lagged_features(pd.DataFrame(input, columns=input_cols)[param], lag)
            lagged_weather_past = lagged_weather_past.transpose().iloc[::-1].iloc[:, lag:]
            lagged_weather_pars_past[param][0] = lagged_weather_past

        # Create an empty DataFrame for timeseries
        timeseries_df = pd.DataFrame(columns=['load ' + str(i) + 'hr bk' for i in range(lag, 0, -1)] + input_cols)

        # Number of iterations
        num_iterations = int((output.shape[0] - lag) / lag)

        # Define an empty list to store DataFrames column-wise
        dfs = []

        # Fill timeseries_train column-wise
        for iteration in tqdm(range(num_iterations)):
            df_list = []
            for i in range(iteration * lag + 1, (iteration + 1) * lag + 1):
                # Create a dictionary to store column data for each row
                column_data = {input_cols[param_no]: input[i-1+lag][param_no] for param_no in range(input.shape[1])}
                column_data[category] = output[i-1+lag][0]

                # Fill load columns from lagged_load
                for j in range(lag):
                    column_data['load ' + str(j + 1) + 'hr bk'] = lagged_load.iloc[lag - j - 1, i - 1]

                # Fill past and future weather columns
                for j in range(weather_lag):
                    for param in weather_pars:
                        column_data[param + '_past_' + str(j + 1) + 'hr bk'] = lagged_weather_pars_past[param][0].iloc[lag - j - 1, i - 1]

                # Append the column data to the list
                df_list.append(pd.DataFrame([column_data]))

            # Concatenate the DataFrames in the list
            df_concat = pd.concat(df_list, ignore_index=True)

            # Append the concatenated DataFrame to the list
            dfs.append(df_concat)

        # Concatenate all DataFrames in the list along axis 0 (rows)
        timeseries_df = pd.concat(dfs, axis=0, ignore_index=True)

        # Create future weather columns
        for param in weather_pars:
            for k in range(weather_lag):
                timeseries_df[param + '_future_' + str(k + 1) + 'hr fwd'] = timeseries_df[param].shift(-k-1)

        # Remove rows with NaN data as a result of shifting
        timeseries_df.dropna(inplace=True)

        return timeseries_df

    def generate_timeseries(self, load, yr_start, yr_end, selected_columns, lag, weather_lag, pred_hr, category):
        theCity = self.prep_data_for_model_dev(load, yr_start, yr_end, selected_columns)

        train_data_normalized_input, validation_data_normalized_input, test_data_normalized_input, input_cols, training_data, validation_data, testing_data, train_data_normalized_output, test_data_normalized_output, validation_data_normalized_output, norm_scaler_output, std_scaler_output = self.prep_data(theCity, category)

        testing_head_index = training_data.shape[0] + validation_data.shape[0] + lag

        timeseries_train = self.fill_timeseries_incl_weatherHist_df(train_data_normalized_input, train_data_normalized_output, lag, weather_lag, pred_hr, input_cols, self.weather_pars, category)
        timeseries_valid = self.fill_timeseries_incl_weatherHist_df(validation_data_normalized_input, validation_data_normalized_output, lag, weather_lag, pred_hr, input_cols, self.weather_pars, category)
        timeseries_test = self.fill_timeseries_incl_weatherHist_df(test_data_normalized_input, test_data_normalized_output, lag, weather_lag, pred_hr, input_cols, self.weather_pars, category)

        return timeseries_train, timeseries_valid, timeseries_test, norm_scaler_output, std_scaler_output, testing_data, testing_head_index
