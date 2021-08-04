import os
import pandas as pd

import category_encoders as ce
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
# FutureWarning: is_categorical is deprecated and will be removed
# in a future version.  Use is_categorical_dtype instead

from imblearn.pipeline import Pipeline as PipelineIMB
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler, OneSidedSelection

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Custom utilities for working with weather data
import weather_utils as we

ACTUAL_DATA = "actual_data.csv"
FUTURE_DATA = "future_data.csv"
TARGET_NAME = "RainTomorrow"

COLUMN_SPLIT = "Region"
COLUMN_VALUE = "5"

SCORE = "roc_auc"  # "f1" / "roc_auc" / "balanced_accuracy"

def print_stats(y_actual, y_predit, metrics=False):
    """ \o/ """
    we.print_stats(y_actual, y_predit, metrics=False)

    
def save_report(search, X=None, y=None, title=None, prefix=None, dir_report='report'):
    """ \o/ """
    if not os.path.isdir(dir_report):
        os.mkdir(dir_report)

    if isinstance(X, pd.DataFrame) and isinstance(y, pd.Series):
        score = search.score(X, y)
        file_to_save = "{}_score-{}.txt".format(search.scoring, score)        
    else:
        score = None
        file_to_save = "{}_score-{}.txt".format(search.scoring, search.best_score_)

    if prefix and isinstance(prefix, str):
        file_to_save = prefix + "__" + file_to_save

    path_to_save = os.path.join(dir_report, file_to_save)

    report = []
    
    if title:
        report = we.add_info(report, title)

    if score:
        report = we.add_info(report,
                             "Score={}".format(round(score, 5)))
    else:
        report = we.add_info(report,
                             "CV score={}".format(round(search.best_score_, 5)))

    report = we.add_info(report, search.best_params_)

    if score:
        report = we.add_stats(report, y, search.predict(X))
        
    with open(path_to_save, "w") as report_file:
        report_file.writelines("\n".join(report))


def get_data(file_name):
    """ \o / """
    path = file_name
    
    data_dir = 'data'
    if os.path.isdir(data_dir):
        path = os.path.join(data_dir, path)
    
    if os.path.isfile(path):
        data = pd.read_csv(path)
    
        return we.create_data(data)


def get_notna_target(data, target_name):
    """ \o/ """    
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


def _get_sampler(name_sampler):
    """ \o/ """
    if name_sampler == 'skip':
        return None
    
    random_seed = 42
    samplers = {
        'random_under': RandomUnderSampler(random_state=random_seed),
        'random_over': RandomOverSampler(random_state=random_seed),
        'smote': SMOTE()
    }
    
    sampler = samplers.get(name_sampler)

    if sampler:
        return ('sampler', sampler)
    else:
        raise NameError(f"Incorrect parameter: {name_sampler}")

    
def _get_scaler(name_scaler):
    """ \o/ """
    if name_scaler == 'skip':
        return None

    scalers = {
        'standard': StandardScaler(),
        'minmax': MinMaxScaler(),
        'robust': RobustScaler()
    }
    
    scaler = scalers.get(name_scaler)

    if scaler:
        return ('scaler', scaler)
    else:
        raise NameError(f"Incorrect parameter: {name_scaler}")


def _get_encoder(name_encoder):
    """ \o/ """
    if name_encoder == 'skip':
        return None

    encoders = {
        'ordinal': OrdinalEncoder(),
        'one_hot': OneHotEncoder(),
        'target': ce.TargetEncoder(handle_unknown='ignore'),
        'binary': ce.BinaryEncoder(),
        'count': ce.CountEncoder()
    }

    encoder = encoders.get(name_encoder)
    
    if encoder:
        return ('encoder', encoder)
    else:
        raise NameError(f"Incorrect parameter: {name_encoder}")


def _get_transformers(scaling, encoding):
    """ \o/ """
    
    # 1. binary transformer
    
    bin_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent'))
    ])

    # 2. numeric_transformer
    # steps = 'imputer' + 'outlier' + 'scaler' (scaling=[...])

    steps_num_transformer = [('imputer', SimpleImputer(strategy='median')),
                             ('outlier', we.OutliersTrim(strategy='border'))]

    scaler = _get_scaler(scaling)
    
    if scaler:
        steps_num_transformer.append(scaler)

    num_transformer = Pipeline(steps=steps_num_transformer)

    # 3. categorical_transformer
    # steps = 'imputer' + 'encoder' (encoding=[...])
    
    steps_cat_transformer = [('imputer', SimpleImputer(strategy='most_frequent'))]

    encoder = _get_encoder(encoding)
    
    if encoder:
        steps_cat_transformer.append(encoder)

    cat_transformer = Pipeline(steps=steps_cat_transformer)
    
    return bin_transformer, num_transformer, cat_transformer


def _select_estimator(preprocessor, sampling, model_type):
    """ \o/ """
    # First step: add 'preprocess'
    steps_estimator = [('preprocess', preprocessor)]

    sampler = _get_sampler(sampling)

    # Next step: add 'sampler'
    if sampler:
        steps_estimator.append(sampler)
        
    # Last step: add model_type
    models = {
        'logreg': LogisticRegression(max_iter=10000),
        'knn': KNeighborsClassifier(),
        'svc': SVC(),
        'tree': DecisionTreeClassifier(),
        'forest': RandomForestClassifier()
    }
    
    model = models.get(model_type)    
    steps_estimator.append((model_type, model))

    # Select Pipeline
    if sampler:
        # from imblearn.pipeline import Pipeline as PipelineIMB
        estimator = PipelineIMB(steps=steps_estimator)
    else:
        # from sklearn.pipeline import Pipeline
        estimator = Pipeline(steps=steps_estimator)
        
    return estimator

        
def get_estimator(model_type, features, scaling=None, encoding=None, sampling=None):
    """ \o/ """
    # Set default parameters:
    
    if not scaling:
        scaling = 'standard'
        
    if not encoding:
        encoding ='one_hot'
    
    if not sampling:
        sampling ='random_under'
    
    bin_transformer, num_transformer, cat_transformer = _get_transformers(scaling, encoding)

    bin_features, num_features, cat_features = features
    
    # Make preprocessor

    preprocessor = ColumnTransformer(transformers=[
            ('bin', bin_transformer, bin_features),
            ('num', num_transformer, num_features),
            ('cat', cat_transformer, cat_features)
    ])

    # Select estimator
    
    estimator = _select_estimator(preprocessor, sampling, model_type)
    
    return estimator


def get_param_grid(model_type, preprocess=False):
    """ \o/ """
    gridsearch_params = {
        'logreg': {
            'logreg__C': [0.1, 1, 10, 100, 1000]
        },
        'knn': {
            'knn__n_neighbors': [3, 5, 9, 12, 15]
        },
        'svc': {
            'svc__C': [1, 2], 'svc__kernel': ['rbf', 'poly']
        },
        'tree': {
            'tree__criterion': ['gini'], 'tree__max_depth': [5, 10],
            'tree__min_samples_split': [10, 20], 'tree__min_samples_leaf': [5, 10],
            'preprocess__num__outlier__strategy': ['border', 'median', 'unique', None]
        },
        'forest': {
            'forest__n_estimators': [100, 150, 200],
            'forest__max_features': [0.7, 0.8], 'forest__max_depth': [5],
            'forest__min_samples_split': [10, 20], 'forest__min_samples_leaf': [5, 10],
            'preprocess__num__outlier__strategy': ['border', 'median', 'unique', None]
        }
    }

    preprocess_param = {
        'preprocess__num__imputer__strategy': ['mean', 'most_frequent'],
        'preprocess__num__outlier__strategy': ['border', 'median', None]
    }
    
    gridsearch_param = gridsearch_params.get(model_type)
    
    if preprocess == True:
        for key, value in preprocess_param.items():
            gridsearch_param[key] = value
        
    return gridsearch_param


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
    
    search = get_search(estimator, param_grid, score=SCORE)
    search.fit(X_train, y_train)
    
    save_report(search, X_test, y_test, title=ACTUAL_DATA, prefix="test_data")
    
    # 2. FUTURE DATA
    
    data = get_data(FUTURE_DATA)
    data = get_notna_target(data, TARGET_NAME)

    X_test, y_test = get_x_y_data(data, TARGET_NAME)
    
    save_report(search, X_test, y_test, title=FUTURE_DATA, prefix="new_data")

    print("The end")
    