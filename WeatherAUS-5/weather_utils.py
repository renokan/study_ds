import pandas as pd
import numpy as np
import scipy.stats as sts

from sklearn.base import BaseEstimator, TransformerMixin
import sklearn.metrics as metric


regions = {
    'Adelaide': '4', 'Albany': '6', 'Albury': '4', 'AliceSprings': '1',
    'BadgerysCreek': '4', 'Ballarat': '5', 'Bendigo': '3', 'Brisbane': '4',
    'Cairns': '6', 'Canberra': '3', 'Cobar': '2', 'CoffsHarbour': '6',
    'Dartmoor': '6', 'Darwin': '5',
    'GoldCoast': '5', 'Hobart': '5', 'Katherine': '3', 'Launceston': '4',
    'Melbourne': '5', 'MelbourneAirport': '4', 'Mildura': '2',
    'Moree': '2', 'MountGambier': '6', 'MountGinini': '6',
    'Newcastle': '5', 'Nhil': '2', 'NorahHead': '5',
    'NorfolkIsland': '6', 'Nuriootpa': '4',
    'PearceRAAF': '3', 'Penrith': '4', 'Perth': '4',
    'PerthAirport': '3', 'Portland': '7',
    'Richmond': '3',
    'Sale': '4', 'SalmonGums': '3', 'Sydney': '5', 'SydneyAirport': '5', 
    'Townsville': '3', 'Tuggeranong': '3',
    'Uluru': '1',
    'WaggaWagga': '3', 'Walpole': '7', 'Watsonia': '5', 'Williamtown': '5',
    'Witchcliffe': '6', 'Wollongong': '5', 'Woomera': '1'
}


def create_data(df):
    """ \o/ """
    data = pd.DataFrame()
    
    data['Month'] = pd.to_datetime(df['Date']).dt.month_name()

    data['Region'] = df['Location'].map(regions).fillna("0")
    
    data['MinTemp'] = df['MinTemp']
    data['MaxMin_Temp'] = (df['MaxTemp'] - df['MinTemp']).abs().round(1)

    data['Temp3pm'] = df['Temp3pm']
    data['AmPm_Temp'] = (df['Temp3pm'] - df['Temp9am']).abs().round(1)
    
    data['Rainfall_YesNo'] = df['Rainfall']
    data['Rainfall_YesNo'].mask(data['Rainfall_YesNo'] > 0, 1, inplace=True)
    
    rain_yes = ((df['RainToday'] == 'Yes') & \
                (df['Sunshine'].isna()))
    rain_no = ((df['RainToday'] == 'No') & \
               (df['Sunshine'].isna()))
    data['Sunshine_Clean'] = df['Sunshine']
    data['Sunshine_Clean'].mask(rain_yes, 0, inplace=True)
    data['Sunshine_Clean'].mask(rain_no, 10, inplace=True)
    data['Sunshine_Clean'].fillna(8, inplace=True)
    data['Sunshine_Clean'].mask(data['Sunshine_Clean'] > 12, 12, inplace=True)
    # missing values without rain equal 10
    # these values are considered 'normal'
    data['Sunshine_Types'] = pd.cut(data['Sunshine_Clean'],
                                    bins=[0, 3, 10, 12],
                                    labels=['small', 'normal', 'big'],
                                    include_lowest=True, right=True)
    
    data['WindDir3pm'] = df['WindDir3pm']

    data['WindGustSpeed'] = df['WindGustSpeed']
    data['WindSpeed3pm'] = df['WindSpeed3pm']

    data['Humidity9am'] = df['Humidity9am']
    data['Humidity3pm'] = df['Humidity3pm']

    data['Pressure3pm'] = df['Pressure3pm']
    data['Pressure_Diff'] = (df['Pressure3pm'] - df['Pressure9am']).abs()
    data['Pressure_Diff'].mask(data['Pressure_Diff'] > 5, 5, inplace=True)
    rain_yes = ((df['RainToday'] == 'Yes') & \
                (data['Pressure_Diff'].isna()))
    rain_no = ((df['RainToday'] == 'No') & \
               (data['Pressure_Diff'].isna()))
    data['Pressure_Diff'].mask(rain_yes, 0.5, inplace=True)
    data['Pressure_Diff'].mask(rain_no, 3.5, inplace=True)
    data['Pressure_Diff'].fillna(2.5, inplace=True)

    data['Cloud_YesNo'] = (
        df[['Cloud9am', 'Cloud3pm']] > 0
    ).any(axis=1).astype(int)

    mapping = {'Yes': 1, 'No': 0}
    data['RainToday'] = df['RainToday'].map(mapping)
    data['RainTomorrow'] = df['RainTomorrow'].map(mapping)

    return data


def get_train_test(data, column_name, column_value):
    """ \o/ """
    is_region = data[column_name] == column_value

    X_train = data[is_region]
    del X_train[column_name]

    X_test = data[~is_region]
    del X_test[column_name]

    return X_train, X_test


class OutliersTrim(BaseEstimator, TransformerMixin):
    """ \o/ """

    def __init__(self, method_search='smart', strategy=None):
        """ \o/ """
        self.method_search = method_search
        self.strategy = strategy

        if method_search and strategy:
            self.is_trim = True
        else:
            self.is_trim = False

    def __is_numpy(self, X):
        """ \o/ """
        return isinstance(X, np.ndarray)

    def __get_outlier_limit(self, data, method_search):
        """ \o/ """
        correct_search = ['smart', 'std', 'iqr']
        
        error_text = "Incorrect param 'method_search'..."
        assert method_search in correct_search, error_text
        
        if method_search == 'smart':
            data_asymmetric = sts.skew(data, nan_policy='omit')
            
            if data_asymmetric > 0.5:
                method_search = 'iqr'
            else:
                method_search = 'std'

        is_np = self.__is_numpy(data)
        if is_np:
            if method_search == 'std':
                data_3std = np.nanstd(data) * 3
                data_mean = np.nanmean(data)
                
                lower_limit = data_mean - data_3std
                upper_limit = data_mean + data_3std

            if method_search == 'iqr':
                q_1, q_3 = np.nanquantile(data, [0.25, 0.75])
                iqr = q_3 - q_1

                lower_limit = q_1 - (1.5 * iqr)
                upper_limit = q_3 + (1.5 * iqr)
        else:
            if method_search == 'std':
                data_3std = data.std() * 3
                data_mean = data.mean()

                lower_limit = data_mean - data_3std
                upper_limit = data_mean + data_3std

            if method_search == 'iqr':
                q_1, q_3 = data.quantile([0.25, 0.75])
                iqr = q_3 - q_1

                lower_limit = q_1 - (1.5 * iqr)
                upper_limit = q_3 + (1.5 * iqr)

        if lower_limit and upper_limit:
            return lower_limit, upper_limit

    def fit(self, X, y=None):
        """ \o/ """
        self._otliers_dict = {}

        if self.is_trim != True:
            return self
       
        is_np = self.__is_numpy(X)
        
        if len(X.shape) == 1:
            if is_np:
                X = X.reshape(-1, 1)
            else:
                X = pd.DataFrame(X)
        
        ncols = X.shape[1]
        
        strategy = self.strategy
        correct_strategy = ['border', 'mean', 'median', 'most_frequent', 'unique']

        error_text = "Incorrect param 'strategy'..."
        assert strategy in correct_strategy, error_text
        
        if is_np:
            for col in range(ncols):
                x_replace = None
                if strategy == 'mean':
                    x_replace = np.nanmean(X[:, col])
                if strategy == 'median':
                    x_replace = np.nanmedian(X[:, col])
                if strategy == 'most_frequent':
                    x_replace = sts.mode(X[:, col], nan_policy='omit').mode[0]
                if strategy == 'unique':
                    max_value = np.nanmax(X[:, col])
                    if max_value < 1111:
                        x_replace = 9999
                    else:
                        x_replace = max_value * 9
                
                limits = self.__get_outlier_limit(X[:, col], self.method_search)

                self._otliers_dict[col] = (*limits, x_replace)

        else:
            for col in X.columns:
                x_replace = None
                if strategy == 'mean':
                    x_replace = X[col].mean()
                if strategy == 'median':
                    x_replace = X[col].median()
                if strategy == 'most_frequent':
                    x_replace = X[col].mode()[0]
                if strategy == 'unique':
                    max_value = X[col].max()
                    if max_value < 1111:
                        x_replace = 9999
                    else:
                        x_replace = max_value * 9

                limits = self.__get_outlier_limit(X[col], self.method_search)

                self._otliers_dict[col] = (*limits, x_replace)

        return self

    def transform(self, X):
        """ \o/ """
        if self.is_trim != True:
            return X
        
        X = X.copy()
        
        is_np = self.__is_numpy(X)
        
        if len(X.shape) == 1:
            if is_np:
                X = X.reshape(-1, 1)
            else:
                X = pd.DataFrame(X)
            
        ncols = X.shape[1]
        
        if is_np:
            for col in range(ncols):
                lower_limit, upper_limit, x_replace = self._otliers_dict[col]

                outlier_low = X[:, col] < lower_limit
                outlier_up = X[:, col] > upper_limit
                outliers = outlier_low | outlier_up

                if self.strategy != 'border':
                    X[:, col] = np.where(outliers, x_replace, X[:, col])
                else:
                    X[:, col] = np.where(outlier_low, lower_limit, X[:, col])
                    X[:, col] = np.where(outlier_up, upper_limit, X[:, col])

        else:
            for col in X.columns:                
                lower_limit, upper_limit, x_replace = self._otliers_dict[col]

                outlier_low = X[col] < lower_limit
                outlier_up = X[col] > upper_limit
                outliers = outlier_low | outlier_up
                
                if self.strategy != 'border':
                    X[col] = X[col].mask(outliers, x_replace)
                else:
                    X[col] = X[col].mask(outlier_low, lower_limit)
                    X[col] = X[col].mask(outlier_up, upper_limit)
        
        return X


def print_stats(y_test, predict, metrics=False):
    """ \o/ """
    print()
    print("Confusion matrix:")
    print(metric.confusion_matrix(y_test, predict))
    print()
    print(metric.classification_report(y_test, predict, digits=3))

    if metrics:
        print()
        print("balanced_accuracy:", metric.balanced_accuracy_score(y_test, predict))
        print("accuracy:", metric.accuracy_score(y_test, predict))
        print("roc_auc:", metric.roc_auc_score(y_test, predict))
        print("recall:", metric.recall_score(y_test, predict))
        print("f1:", metric.f1_score(y_test, predict))    
    
