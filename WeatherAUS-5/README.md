# WeatherAUS - Predict the rain in Australia


The dataset contains about 10 years of daily weather
observations from many locations across Australia.

Link: https://www.kaggle.com/jsphyg/weather-dataset-rattle-package

**RainTomorrow** is the target variable to predict.
It means - did it rain the next day, Yes or No? This colum is Yes if
the rain for that day was 1mm or more.

The program logic and main functions are in `weather_project.py` file.
Additional functions for working with data are in the `weather_utils.py` file.

```
WeatherAUS-5/
 ├── data/
 │    ├── actual_data.csv
 │    ├── future_data.csv
 │    ├── weatherAUS.csv
 │    └── split_dataset.ipynb
 ├── report/
 │    ├── new_data__roc_auc_score-0.8418163108479612.txt
 │    ├── test_data__roc_auc_score-0.8673259680186647.txt
 │    └── [.......]__roc_auc_score-0.[..............].txt
 ├── info/
 │    ├── feature_analysis.ipynb
 │    ├── feature_engineering.ipynb
 │    ├── feature_importances.ipynb
 │    ├── model_logreg.ipynb
 │    ├── model_knn.ipynb
 │    ├── model_svc.ipynb
 │    ├── model_tree.ipynb
 │    ├── model_forest.ipynb
 │    ├── steps_outliers.ipynb
 │    ├── steps_param_grid.ipynb
 │    └── steps_sampling.ipynb
 ├── weather.py
 ├── weather_utils.py
 └── weather_project.ipynb
```

The main program code is in the `weather_project.py` file.

```python
# ... 

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

```

