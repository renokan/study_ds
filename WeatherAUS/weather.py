""" Predict next-day rain (on the target variable RainTomorrow).
The dataset contains about 10 years of daily weather
observations from many locations across Australia.

Link: https://www.kaggle.com/jsphyg/weather-dataset-rattle-package

RainTomorrow is the target variable to predict.
It means - did it rain the next day, Yes or No? This colum is Yes if
the rain for that day was 1mm or more."""

import os
import sys
import json
import numpy as np
import pandas as pd

import utils_weather as we

from datetime import datetime


def get_settings(file_name):
    """ \o/ """
    try:
        with open(file_name, "r") as read_file:
            data = json.load(read_file)
    except Exception:
        print("\nIncorrect program launch, settings file not found.\n")
        sys.exit()
    else:    
        correct_settings = data.get('path_to_csv')

    if not correct_settings:
        print("\nIncorrect settings file.\n")
        sys.exit()
    
    return data


def get_features(settings):
    """ \o/ """
    return settings['features']


def set_column(settings, utils):
    """ \o/ """
    date = settings['columns']['date']
    city = settings['columns']['city']

    utils.COLUMN_DATE = date
    utils.COLUMN_CITY = city


def save_preprocessing(who_save, shape=None, info=""):
    """ \o/ """
    rows = cols = None
    now = datetime.now()
    dt = now.strftime("%H:%M:%S / %f")

    if shape:
        rows = shape[0]
        cols = shape[1]
    
    result = {"time": dt,
              "rows": rows,
              "cols": cols,
              "info": info}

    report['preprocessing'][who_save] = result


def load_data(settings):
    """ \o/ """    
    path_to = settings['path_to_csv']
    data = pd.read_csv(path_to)

    data_size = data.size
    isna_sum = data.isna().sum().sum()
    info = "Size {} / Missing values {}".format(data_size, isna_sum)
    save_preprocessing("load_data", data.shape, info)

    return data


def convert_dt(settings, data):
    """ \o/ """
    dt_column = settings['columns']['date']
    data[dt_column] = pd.to_datetime(data[dt_column])

    info = "Dtype {}".format(data[dt_column].dtype)
    save_preprocessing("convert_dt", data.shape, info)

    return data


def to_dropna_rows(settings, data):
    """ \o/ """
    dropna = settings['dropna']
    
    info = False
    if dropna['status'] == True:
        x_thresh = dropna['thresh']

        if x_thresh and x_thresh > 0:
            data = data.dropna(thresh=x_thresh)
        else:
            data = data.dropna()

    save_preprocessing("to_dropna_rows", data.shape, info)

    return data


def to_dropna_cols(features, data):
    """ \o/ """
    list_subset = []
    info = ""

    for feature in features.values():
        if feature['dropna'] == True:
            column = feature['name']

            if column in data.columns:
                list_subset.append(column)

    if list_subset:
        data = data.dropna(subset=list_subset)
        info = "{}\tDropna: {}".format(data.shape, ", ".join(list_subset))

    save_preprocessing("to_dropna_cols", data.shape, info)
    
    return data


def to_drop(features, data):
    """ \o/ """
    list_drop = []
    info = ""
    
    for feature in features.values():
        column = feature['name']
        is_drop = feature['drop']
        
        if is_drop == True and column in data.columns:
            del data[column]
            list_drop.append(column)

    if list_drop:
        info = "Drop: {}".format(", ".join(list_drop))

    save_preprocessing("to drop", data.shape, info)
 
    return data


def to_fillna(features, data):
    """ \o/ """
    for feature in features.values(): 
        column = feature['name']
        method = feature['fillna']
        
        if column not in data.columns:
            continue
        
        data[column] = we.try_fillna_column(data, column)
    
    na_sum = data.isna().sum().sum()
    info = "Missing values {}".format(na_sum)
    save_preprocessing("to_fillna", data.shape, info)

    return data


def to_upd_outlier(features, data, asymmetric=0.5, upd_all=False):
    """ \o/ """
    correct_columns = data.skew().index.to_list()
    result = {}

    for key, feature in features.items():
        column = feature['name']
        outlier = feature.get('outlier')
        
        if not outlier:
            continue

        if column not in correct_columns:
            continue
        
        is_update = outlier.get('update')
        method = outlier.get('method')
        
        if upd_all == True:
            is_update = True

        if is_update == True:
            data = we.try_outlier_upd(data, column, asymmetric, method)  
            result[key] = "Feature '{}' by '{}' method".format(column, method)
     
    save_preprocessing("to_upd_outlier", data.shape, result)

    return data


def to_encode(features, data):
    """ \o/ """
    result = {}

    for key, feature in features.items():
        column = feature['name']
        encoder = feature.get('encoder')
        
        if not encoder or column not in data.columns:
            continue

        is_update = encoder.get('update')
        method = encoder.get('method')
        
        if is_update == True:
            data = we.try_encode(data, column, method)  
            result[key] = "Feature '{}' by '{}' method".format(column, method)
     
    save_preprocessing("to_encode", data.shape, result)

    return data


def save_report(report, dir_name=None):
    """ \o/ """
    now = datetime.now()
    file_to_save = now.strftime("report_%Y-%m-%d___%H-%M-%S.json")

    if dir_name and os.path.isdir(dir_name):
        file_to_save = os.path.join(dir_name, file_to_save)

    with open(file_to_save, "a") as report_file:
        json.dump(report, report_file, indent=4)        


def save_old_settings(arguments, save_to):
    """ \o/ """
    input_arguments = arguments[1:]
    if input_arguments:
        report_file = input_arguments[0]

        try:
            with open(report_file, "r") as read_file:
                data = json.load(read_file)
        except Exception:
            print("\nInvalid input argument, report file not found.\n")
            sys.exit()
        else:
            settings = data.get('settings')
            
            if settings:
                with open(save_to, "w") as write_file:
                    json.dump(settings, write_file, indent=4)        

                print("\nThe settings have been extracted and saved.\n")
                sys.exit()


if __name__ == '__main__':
    # Extracting and saving settings from a report file
    # python weather.py report/report_[...].json
    save_old_settings(sys.argv, save_to='settings_OLD.json')

    settings = get_settings('settings.json')

    # Updating we.COLUMN_DATE / we.COLUMN_CITY
    set_column(settings, we)

    report = {'preprocessing': {}}
       
    # === Working with a dataset ===
    data = load_data(settings)  # Step 0
    data = convert_dt(settings, data) # Step 1
    data = to_dropna_rows(settings, data)  # Step 2
    # data.to_pickle("data/data-2.pkl")
   
    # === Working with features ===
    features = get_features(settings)

    data = to_drop(features, data)  # Step 3
    data = to_dropna_cols(features, data)  # Step 4
    data = to_fillna(features, data)  # Step 5
    # data.to_pickle("data/data-5.pkl")
    data = to_upd_outlier(features, data)  # Step 6
    data.to_pickle("data/data-6.pkl")
    data = to_encode(features, data)  # Step 7
    data.to_pickle("data/data-7.pkl")

    report['settings'] = settings
    save_report(report, dir_name='report')

    print("\nThe program has ended correctly.\n")
    