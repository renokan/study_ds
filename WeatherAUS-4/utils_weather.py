""" Utils """

import json
import pandas as pd


def get_columns(assortment=None):
    """ \o/ """
    try:
        with open("columns_param.json", "r") as read_file:
            columns_dict = json.load(read_file)
    except Exception:
        raise Exception("Incorrect read file...")
    else:
        if not assortment:
            return list(columns_dict.keys())
        else:
            return columns_dict.get(assortment)

        
def get_search(assortment=None):
    """ \o/ """
    try:
        with open("search_param.json", "r") as read_file:
            search_dict = json.load(read_file)
    except Exception:
        raise Exception("Incorrect read file...")
    else:
        if not assortment:
            return search_dict.keys()
        else:
            return search_dict.get(assortment)


def get_3group_features(df):
    """ \o/ """
    bin_features = []
    num_features = []
    cat_features = []

    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            if df[column].dropna().isin([1, 0]).all():
                bin_features.append(column)
            else:
                num_features.append(column)
        else:
            cat_features.append(column)

    return bin_features, num_features, cat_features
    

def _create_data(df, date_type='month_name'):
    """ \o/ """
    data = pd.DataFrame()

    if date_type == 'month_name':
        data['Date'] = pd.to_datetime(df['Date']).dt.month_name()
    elif date_type == 'month':
        data['Date'] = pd.to_datetime(df['Date']).dt.month()
    else:
        pass
        
    data['Location'] = df['Location']

    data['MinTemp'] = df['MinTemp']
    data['MaxTemp'] = df['MaxTemp']
    data['Diff_Temp'] = (df['MaxTemp'] - df['MinTemp']).abs()

    data['Temp9am'] = df['Temp9am']
    data['Temp3pm'] = df['Temp3pm']
    
    data['Rainfall'] = df['Rainfall']
    data['Evaporation'] = df['Evaporation']
    data['Sunshine'] = df['Sunshine']

    data['WindGustDir'] = df['WindGustDir']
    data['WindDir9am'] = df['WindDir9am']
    data['WindDir3pm'] = df['WindDir3pm']
    data['WindDir_Change'] = (df['WindGustDir'] == \
                              df['WindDir3pm']).map({True: 1, False: 0})

    data['WindDir_Change_short'] = (df['WindGustDir'].str.slice(stop=1) == \
                                    df['WindDir3pm'].str.slice(stop=1)
                                    ).map({True: 1, False: 0})    

    data['WindGustSpeed'] = df['WindGustSpeed']
    data['WindSpeed9am'] = df['WindSpeed9am']
    data['WindSpeed3pm'] = df['WindSpeed3pm']
    data['WindSpeed_Diff'] = (df['WindGustSpeed'] - df['WindSpeed3pm']).abs()

    data['Humidity9am'] = df['Humidity9am']
    data['Humidity3pm'] = df['Humidity3pm']
    data['Humidity_Diff'] = (df['Humidity3pm'] - df['Humidity9am']).abs()

    data['Pressure9am'] = df['Pressure9am']
    data['Pressure3pm'] = df['Pressure3pm']
    data['Pressure_Diff'] = (df['Pressure3pm'] - df['Pressure9am']).abs()
    
    data['Cloud9am'] = df['Cloud9am']
    data['Cloud3pm'] = df['Cloud3pm']

    mapping = {'Yes': 1, 'No': 0}
    data['RainToday'] = df['RainToday'].map(mapping)
    data['RainTomorrow'] = df['RainTomorrow'].map(mapping)

    return data


def get_data(df, columns=None, target=None):
    """ \o/ """
    data = _create_data(df)

    if not columns:
        columns = 'origin'

    features_list = get_columns(assortment=columns)
    
    if target and target in features_list and target in data.columns:
        features_list.remove(target)
        y = data[target]
    else:
        y = None
    
    features = [feature for feature in features_list if feature in data.columns]
    
    X = data[features]

    return X, y
    
    
    
