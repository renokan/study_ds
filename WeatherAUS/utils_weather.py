""" Custom utilities for working with weather data. """


import numpy as np
import pandas as pd
import category_encoders as ce

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
# FutureWarning: is_categorical is deprecated and will be removed
# in a future version.  Use is_categorical_dtype instead


COLUMN_CITY = 'Location'
COLUMN_DATE = 'Date'
COLUMN_TARGET = 'RainTomorrow'

# General functions


def _get_date_sets(df, date_column):
    """ \o/ """
    result = {}
    result['year'] = df[date_column].dt.year   
    result['month'] = df[date_column].dt.month
    result['week'] = df[date_column].dt.isocalendar().week
    
    return result


# === Working with missing values ===


def isna_stats(df, columns=None, n=None, more_than=None):
    """ The percentage of missing values in the dataset.
    There is a choice of columns, filter by top values / more than. """
    data = df
    if columns:
        data = data[columns]

    result = data.isna().mean().mul(100).round(2)

    if more_than:
        result = result[result > more_than]
    
    if n:
        result = result.nlargest(n)

    return result.map("{} %".format)


def notna_column(df, column, period='month', total=False):
    """ Table of values by columns (grouped by location and period).
    Period - year or month or month-year. And the total amount. """
    city = df[COLUMN_CITY]
    
    date_sets = _get_date_sets(df, COLUMN_DATE)
    year = date_sets.get('year')
    month = date_sets.get('month')

    table_index = city

    if period == 'all':
        table_index = [city, year]
        table_columns = month
    elif period == 'year':
        table_columns = year
    else:
        table_columns = month
    
    result = pd.pivot_table(df, values=column,
                                index=table_index,
                                columns=table_columns,
                                    aggfunc='count', fill_value=0)
    
    if total:
        result['Total'] = result.sum(axis=1)
        result = result.sort_values(by='Total', ascending=False)
    
    return result


def _fillna_g_ffill(df, column_name, x_limit=2):
    """ \o/ """
    data = df[column_name]
    city = df[COLUMN_CITY]

    date_sets = _get_date_sets(df, COLUMN_DATE)
    year = date_sets.get('year')
    month = date_sets.get('month')

    grouped = data.groupby([city, year, month])

    return grouped.ffill(limit=x_limit)


def _fillna_g_median(df, column_name):
    """ \o/ """
    data = df[column_name]
    city = df[COLUMN_CITY]

    date_sets = _get_date_sets(df, COLUMN_DATE)
    date = date_sets.get('month')

    grouped = data.groupby([city, date])

    fillna_by_group = lambda g: g.fillna(g.median())

    return grouped.apply(fillna_by_group)


def _fillna_g_mean(df, column_name):
    """ \o/ """
    data = df[column_name]
    city = df[COLUMN_CITY]

    date_sets = _get_date_sets(df, COLUMN_DATE)
    date = date_sets.get('month')

    grouped = data.groupby([city, date])

    fillna_by_group = lambda g: g.fillna(g.mean())

    return grouped.apply(fillna_by_group)


def _fillna_g_mode(df, column_name):
    """ \o/ """
    data = df[column_name]
    city = df[COLUMN_CITY]

    date_sets = _get_date_sets(df, COLUMN_DATE)
    date = date_sets.get('month')

    grouped = data.groupby([city, date])

    def fillna_by_group(group):
        g_mode = group.mode()
        if len(g_mode) > 0:
            return group.fillna(g_mode[0])
        else:
            return group

    return grouped.apply(fillna_by_group)


def _fillna_x_median(df, column_name):
    """ \o/ """
    data = df[column_name]
    x_median = data.median()
    
    return data.fillna(x_median)


def _fillna_x_mean(df, column_name):
    """ \o/ """
    data = df[column_name]
    x_mean = data.mean()
    
    return data.fillna(x_mean)


def _fillna_x_mode(df, column_name):
    """ \o/ """
    data = df[column_name]
    x_mode = data.mode()
    
    if len(x_mode) > 0:
        return data.fillna(x_mode[0])
    else:
        return data


def try_fillna_column(df, column_name, method='obj-easy'):
    """ Filling missing values with the chosen method. """    
    methods = {}

    methods['num-smart'] = [_fillna_g_ffill,_fillna_g_median, _fillna_x_median]
    methods['num-normal'] = [_fillna_g_median, _fillna_x_median]
    methods['num-easy'] = [_fillna_x_median]

    methods['obj-smart'] = [_fillna_g_ffill, _fillna_g_mode, _fillna_x_mode]
    methods['obj-normal'] = [_fillna_g_mode, _fillna_x_mode]
    methods['obj-easy'] = [_fillna_x_mode]

    data = df[column_name]
    
    for func_fillna in methods.get(method, []):
        if data.isna().any():
            data = func_fillna(df, column_name)

    return data


# === Working with outlines values ===


def _get_outlier_limit(series, how_search):
    """ \o/ """
    if how_search == 'std':
        data_3std = series.std() * 3
        data_mean = series.mean()

        lower_limit = data_mean - data_3std
        upper_limit = data_mean + data_3std
        
        return lower_limit, upper_limit

    if how_search == 'iqr':
        q_1, q_3 = series.quantile([0.25, 0.75])
        iqr = q_3 - q_1

        lower_limit = q_1 - (1.5 * iqr)
        upper_limit = q_3 + (1.5 * iqr)
        
        return lower_limit, upper_limit
        

def outlier_stats(df):
    """ The percentage of outlier values in the dataset.
    There is a choice of columns, filter by top values. """
    columns = df.select_dtypes(include='number').columns.to_list()
    data = df[columns]

    def column_stats(data, how_search):
        """ \o/ """
        std_limits = _get_outlier_limit(data, how_search)
        mask = (data < std_limits[0]) | (data > std_limits[1])
        std_sum = mask.sum()
        std_per = (mask.mean() * 100).round(3)
        
        return pd.Series([std_sum, std_per], index=['count', '%'])

    result = {}
    methods = ['std', 'iqr']

    for method in methods:
        result[method] = data.apply(column_stats, how_search=method).T

    return pd.concat(result.values(), keys=result.keys(), axis=1)


def try_outlier_upd(df, name_column, asymmetric, method):
    """ Filling outlier values with the chosen method. """
    data = df[name_column]
    data_asymmetric = data.skew()
    
    if data_asymmetric > asymmetric:
        search_type = 'iqr'
    else:
        search_type = 'std'
        
    lower_limit, upper_limit = _get_outlier_limit(data, search_type)

    outlier_low = data < lower_limit
    outlier_up = data > upper_limit

    outliers = outlier_low | outlier_up

    if method == 'drop':
        df = df[~outliers]
        
    elif method == 'border':
        df[name_column] = data.mask(outlier_low, lower_limit)
        df[name_column] = data.mask(outlier_up, upper_limit)
        
    elif method == 'median':
        found_median = data.median()
        df[name_column] = data.mask(outliers, found_median)
        
    else:
        pass  # if a missing method is passed, do nothing

    return df


# === Encoding categorial values ===


def _encode_yes_no(df, column_name):
    """ \o/ """
    mapping = {'Yes': 1, 'No': 0}
    
    data_correct = df[column_name].isin(['Yes', 'No']).all()
    
    if data_correct:
        df[column_name] = df[column_name].map(mapping)
    
    return df


def _encode_target(df, column_name, target_name):
    """ \o/ """
    encoder = ce.TargetEncoder(cols=column_name)
    
    encoder.fit(df, df[target_name])
    
    return encoder.transform(df)


def _use_encoder(df, name_column, encoder):
    """ \o/ """
    encoders = {'ordinal': ce.OrdinalEncoder,
                'binary': ce.BinaryEncoder,
                'onehot': ce.OneHotEncoder
               }
    
    encoder = encoders.get(encoder)

    if not encoder:
        return df
        
    encoder = encoder(cols=name_column)
        
    return encoder.fit_transform(df)


def _encode_location(df, column_name, method):
    """ \o/ """
    # I am thinking about this task
    
    return df


def _encode_date(df, column_name, method):
    """ \o/ """
    # I am thinking about this task
    
    return df


def try_encode(df, name_column, method):
    """ Encoding values with the chosen method. """
    if method == "yes_no":
        df = _encode_yes_no(df, name_column)

    elif method == "target":
        target_name = COLUMN_TARGET
        df = _encode_target(df, name_column, target_name)

    elif method == "ordinal":
        df = _use_encoder(df, name_column, 'ordinal')

    elif method == "binary":
        df = _use_encoder(df, name_column, 'binary')

    elif method == "onehot":
        df = _use_encoder(df, name_column, 'onehot')

    else:
        pass  # need to add a smart method
    
    return df


# === Test functions ===


def test_get_group_indx(df, location, x_year, x_month=None):
    """ Get a list of indexes by multi-index of the selected group.
    Possible options: location and year (required), month. """
    city = df[COLUMN_CITY]
    year = df[COLUMN_DATE].dt.year   

    result = df[(city == location) & (year == x_year)]

    if x_month:
        month = result[COLUMN_DATE].dt.month
        result = result[month == x_month]

    return result.index.tolist()


def test_compare_values(df, df_test, column_name, info=False, diff=False):
    """ Get old data (column) and new from the dataframes.
    Additional information / difference between columns. """
    result = {}
    data_new = df[column_name]
    data_old = df_test[column_name]

    if len(data_new) != len(data_old):
        raise Exception("Sorry, but ...")

    if info == True:
        result = {'Date': df[COLUMN_DATE],
                  'Location': df[COLUMN_CITY]}

    result['New'] = data_new
    result['Old'] = data_old    

    data = pd.DataFrame(result)
    
    if diff == True:
        data = data[data['New'].notna() & data['Old'].isna()]
    
    return data


def test_fillna_column(df, column_name):
    """ Trying all the methods for filling NaN values in a column. """
    methods = {'G Ffill': _fillna_g_ffill,
               'G Median': _fillna_g_median,
               'G Mean': _fillna_g_mean,
               'Median': _fillna_x_median,
               'Mean': _fillna_x_mean,
               'G Mode': _fillna_g_mode,
               'Mode': _fillna_x_mode
               }

    result = {'Date': df[COLUMN_DATE],
              'Location': df[COLUMN_CITY],
              'Origin': df[column_name]}

    for name_method, func_method in methods.items():
        try:
            result[name_method] = func_method(df, column_name)
        except Exception:
            pass
    
    return pd.DataFrame(result)


# Other functions

