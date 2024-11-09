# -*- coding: utf-8 -*-
"""read_data.py"""

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

class ReadData:

    def __init__(self, path):
        self.file_path = path
        self.df = pd.read_csv(self.file_path)

        # Convert ReadValue to numeric, replacing non-convertible values with NaN, then replace NaN with 0
        self.df[' ReadValue '] = pd.to_numeric(self.df[' ReadValue '].str.replace(',', ''), errors='coerce').fillna(0)

        # Convert Datesk and TimeSk to a single datetime column
        self.df['DateTime'] = pd.to_datetime(self.df['Datesk'].astype(str), format='%Y%m%d') + pd.to_timedelta(self.df['TimeSk'] - 1, unit='h')

        # Aggregate the ReadValue by DateTime
        self.load_agg = self.df.groupby('DateTime')[' ReadValue '].sum().reset_index()

        # Sort DataFrame based on 'Date' and then 'Hour'
        self.df.sort_values(by=['DateTime'], inplace=True)

        # Filter out rows where ReadValue is 0
        self.df = self.df[self.df[' ReadValue '] != 0]

        # Get unique categories
        self.categories = self.df['Category'].unique()

    def aggregate_category_data(self, category):
        # Filter the dataframe by the current category
        df_category = self.df[self.df['Category'] == category]

        # Aggregate the load data based on DateTime
        load_cat_agg = df_category.groupby('DateTime')[' ReadValue '].sum().reset_index()

        return load_cat_agg
