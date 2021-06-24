# WeatherAUS - Predict the rain in Australia


The dataset contains about 10 years of daily weather
observations from many locations across Australia.

Link: https://www.kaggle.com/jsphyg/weather-dataset-rattle-package

**RainTomorrow** is the target variable to predict.
It means - did it rain the next day, Yes or No? This colum is Yes if
the rain for that day was 1mm or more.

Functions for working with data are in the `utils_weather.py` file.
Settings for working with data in `settings.json` file.

```
WeatherAUS/
 ├── data/
 │    ├── data-2.pkl
 │    ├── data-3.pkl
 │    ├── data-4.pkl
 │    └── data-7.pkl
 ├── info/
 │    ├── rains_info.ipynb
 │    ├── rains_data.csv
 │    └── img_climate_australia-[...].jpg
 ├── input/
 │    ├── weatherAUS.csv
 │    └── sample.csv
 ├── report/
 │    ├── report_2021-[...].json
 │    └── sample.json
 ├── weather.py
 ├── utils_weather.py
 ├── settings.json
 ├── weather-run.ipynb
 └── check-[...].ipynb
```

The main program code is in the `weather.py` file.

```python
# ... 

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

# ... 
```

