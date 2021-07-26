import os
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
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Custom utilities for working with weather data
import weather_utils as we

ACTUAL_DATA = "data/actual_data.csv"
FUTURE_DATA = "data/future_data.csv"
TARGET_NAME = "RainTomorrow"

COLUMN_SPLIT = "Region"
COLUMN_VALUE = "5"

SCORE = "roc_auc"  # "f1" / "roc_auc" / "balanced_accuracy"

def print_stats(y_actual, y_predit, metrics=False):
    """ \o/ """
    we.print_stats(y_actual, y_predit, metrics=False)

    
def save_report(search, X=None, y=None, title=None, dir_report='report'):
    """ \o/ """
    if not os.path.isdir(dir_report):
        os.mkdir(dir_report)

    if isinstance(X, pd.DataFrame) and isinstance(y, pd.Series):
        score = search.score(X, y)
        file_to_save = "{}_score-{}.txt".format(search.scoring, score)        
    else:
        score = None
        file_to_save = "{}_score-{}.txt".format(search.scoring, search.best_score_)

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


def get_data(path):
    """ \o / """
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


def get_estimator(model_type, features):
    """ \o/ """
    models = {
        'logreg': LogisticRegression(max_iter=10000),
        'knn': KNeighborsClassifier(),
        'svc': SVC(),
        'tree': DecisionTreeClassifier()
    }

    leaf_type_models = ['tree']  # add 'forest'
    
    model = models.get(model_type)
    
    bin_features, num_features, cat_features = features

    scaler = StandardScaler()
    encoder = OneHotEncoder()
    sampler = RandomUnderSampler(random_state=42)

    binary_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent'))
    ])

    if model_type in leaf_type_models:
        numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')),
                ('outlier', we.OutliersTrim(strategy=None))
        ])
    else:        
        numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')),
                ('outlier', we.OutliersTrim(strategy='border')),
                ('scaler', scaler)
        ])

    if model_type in leaf_type_models:
        categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OrdinalEncoder())
        ])
    else:
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
    leaf_type_models = ['tree']  # add 'forest'

    gridsearch_params = {
        'logreg': {
            'logreg__C': [0.1, 1, 10]
        },
        'knn': {
            'knn__n_neighbors': [9, 12, 15]
        },
        'svc': {
            'svc__C': [2], 'svc__gamma': ['scale'], 'svc__kernel': ['rbf']
        },
        'tree': {
            'tree__criterion': ['gini', 'entropy'], 'tree__max_depth': [None, 5, 15, 30],
            'tree__min_samples_split': [2, 20, 40], 'tree__min_samples_leaf': [1, 10, 20]
        }
    }

    if model_type in leaf_type_models:
        preprocess_param = {
            'preprocess__num__imputer__strategy': ['mean', 'most_frequent'],
            'preprocess__num__outlier__strategy': ['border', 'median', 'unique']
        }
    else:
        preprocess_param = {
            'preprocess__num__imputer__strategy': ['mean', 'most_frequent'],
            'preprocess__num__outlier__strategy': ['border', 'median']
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
    
    save_report(search, X_test, y_test, title=ACTUAL_DATA)
    
    # 2. FUTURE DATA
    
    data = get_data(FUTURE_DATA)
    data = get_notna_target(data, TARGET_NAME)

    X_test, y_test = get_x_y_data(data, TARGET_NAME)
    
    save_report(search, X_test, y_test, title=FUTURE_DATA)

    print("The end")
    