import pandas as pd

from imblearn.pipeline import Pipeline as PipelineIMB
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler, OneSidedSelection

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# Custom utilities for working with weather data
import weather_utils as we

ACTUAL_DATA = "data/actual_data.csv"
FUTURE_DATA = "data/future_data.csv"
TARGET_NAME = "RainTomorrow"

COLUMN_SPLIT = "Region"
COLUMN_VALUE = "5"

def print_stats(y_actual, y_predit, metrics=False):
    """ \o/ """
    we.print_stats(y_actual, y_predit, metrics=False)


def save_stats(y_actual, y_predit):
    """ \o/ """
    pass


def get_data(path):
    """ \o / """
    data = pd.read_csv(path)
    
    return we.create_data(data)


def get_notna_target(data, target_name):
    """ \o/ """    
    # data = data.dropna(subset=[target_name])
    return data.dropna(subset=[target_name])


def get_train_test(data):
    """ \o/ """
    column_name = COLUMN_SPLIT
    column_value = COLUMN_VALUE

    return we.get_train_test(data, column_name, column_value)


def get_x_y_data(data, target_name):
    """ \o/ """
    X = data
    y = data.pop(target_name)
    
    if COLUMN_SPLIT in X.columns:
        del X[COLUMN_SPLIT]
    
    return X, y


def get_3group_features(data):
    """ \o/ """
    bin_features = []
    num_features = []
    cat_features = []

    for column in data.columns:
        if pd.api.types.is_numeric_dtype(data[column]):
            if data[column].dropna().isin([1, 0]).all():
                bin_features.append(column)
            else:
                num_features.append(column)
        else:
            cat_features.append(column)

    return bin_features, num_features, cat_features


def get_estimator(model_type, features):
    """ \o/ """
    models = {
        'logreg': LogisticRegression(max_iter=10000),
        'knn': KNeighborsClassifier(),
        'svc': SVC()
    }

    model = models.get(model_type)
    
    bin_features, num_features, cat_features = features

    scaler = StandardScaler()
    encoder = OneHotEncoder()
    sampler = RandomUnderSampler(random_state=42)

    binary_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent'))
    ])

    numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('outlier', we.OutliersTrim(strategy='border')),
            ('scaler', scaler)
    ])

    categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', encoder)
    ])
    
    preprocessor = ColumnTransformer(transformers=[
            ('bin', binary_transformer, bin_features),
            ('num', numeric_transformer, num_features),
            ('cat', categorical_transformer, cat_features)
    ])

    estimator = PipelineIMB(steps=[
            ('preprocess', preprocessor),
            ('sampling', sampler),
            (model_type, model)
    ])
    
    return estimator


def get_param_grid(model_type, preprocess=False):
    """ \o/ """
    gridsearch_param = {
        'logreg': {
            'logreg__C': [0.1, 1, 10]
        },
        'knn': {
            'knn__n_neighbors': [9, 12, 15]
        },
        'svc': {
            'svc__C': [2], 'svc__gamma': ['scale'], 'svc__kernel': ['rbf']
        }
    }

    preprocess_param = {
        'preprocess__num__imputer__strategy': ['median', 'mean', 'most_frequent'],
        'preprocess__num__outlier__strategy': ['border', 'median']
    }
    
    if preprocess == False:
        return gridsearch_param.get(model_type)
    else:
        return [gridsearch_param.get(model_type), preprocess_param]


def get_search(estimator, param_grid, score='f1', param_cv=3, param_verbose=0):
    """ \o/ """
    search = GridSearchCV(estimator=estimator,
                      param_grid=param_grid,
                      cv=param_cv, scoring=score,
                      verbose=param_verbose)
    
    return search
    

if __name__ == '__main__':
    
    # 1. ACTUAL DATA

    data = get_data(ACTUAL_DATA)
    data = get_notna_target(data, TARGET_NAME)
    
    # Split data by column 'Region' value 5
    train_data, test_data = get_train_test(data)
    
    X_train, y_train = get_x_y_data(train_data, TARGET_NAME)
    X_test, y_test = get_x_y_data(test_data, TARGET_NAME)

    features = get_3group_features(X_train)
    
    model_type = "logreg"
    estimator = get_estimator(model_type, features)
    param_grid = get_param_grid(model_type)
    
    search = get_search(estimator, param_grid)
    search.fit(X_train, y_train)
    
    predict = search.predict(X_test)
    
    save_stats(y_test, predict)
    
    # 2. FUTURE DATA
    
    data = get_data(FUTURE_DATA)
    data = get_notna_target(data, TARGET_NAME)

    X_test, y_test = get_x_y_data(data, TARGET_NAME)
    
    predict = search.predict(X_test)
    
    save_stats(y_test, predict)

    print("The end")
    