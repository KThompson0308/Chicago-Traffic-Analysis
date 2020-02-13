import os
import multiprocessing
from joblib import Parallel, delayed
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import scipy as stats

import time


# Retrieve all filepaths and place in a nested dictionary
def retrieve_filepaths(data_directory_name, filetypes=['csv']):
    data_path = get_datadirectory(data_directory_name)
    filepaths = dict.fromkeys(filetypes)
    for key in filepaths.keys():
        filepaths[key] = get_paths_for_filetype(key, data_path)
    return filepaths

def get_datadirectory(data_directory):
    WORKING_DIRECTORY_PATH = os.getcwd()
    DATA_DIRECTORY_PATH = os.path.join(WORKING_DIRECTORY_PATH, data_directory)
    data_path = Path(DATA_DIRECTORY_PATH)
    return data_path

def get_paths_for_filetype(filetype, path_object):
    files = dict.fromkeys([files.name for files in path_object.iterdir() if "".join(('.',filetype)) in files.name])
    paths = list(str(x.absolute()) for x in path_object.iterdir() if "".join(('.',filetype)) in str(x.absolute()))
    for filename in files.keys():
        files[filename] = [datapath for datapath in paths if filename in datapath][0]
    return files


start = time.time()
# Define data types of attributes
categoricals = ['CRASH_DATE_EST_I', 'TRAFFIC_CONTROL_DEVICE', 'DEVICE_CONDITION',
                'WEATHER_CONDITION', 'LIGHTING_CONDITION', 'TRAFFICWAY_TYPE',
                'FIRST_CRASH_TYPE', 'TRAFFICWAY_TYPE', 'ROADWAY_SURFACE_COND',
                'ROAD_DEFECT', 'REPORT_TYPE', 'CRASH_TYPE', 'INTERSECTION_RELATED_I',
                'NOT_RIGHT_OF_WAY_I', 'HIT_AND_RUN_I', 'DAMAGE', 'PRIM_CONTRIBUTORY_CAUSE',
                'SEC_CONTRIBUTORY_CAUSE', 'STREET_DIRECTION', 'STREET_NAME', 'PHOTOS_TAKEN_I',
                'STATEMENTS_TAKEN_I', 'DOORING_I', 'WORK_ZONE_I', 'WORK_ZONE_TYPE', 'WORKERS_PRESENT_I',
                'MOST_SEVERE_INJURY', 'BEAT_OF_OCCURRENCE']

dtypes = dict.fromkeys(categoricals, 'category')


filepaths = retrieve_filepaths('data', ['csv', 'geojson'])

# Import the datasets
beats = gpd.read_file(filepaths['geojson']['policebeats.geojson'])
weather = pd.read_csv(filepaths['csv']['ChicagoWeather.csv'], parse_dates = ['dt_iso'], usecols=['dt_iso', 'weather_main',
                                                                                           'weather_description'],
                     dtype={'weather_main':'category', 'weather_description':'category'})
weather['dt_iso'] = pd.to_datetime(weather['dt_iso'],format="%Y-%m-%d %H:00:00 +0000 UTC")
crashes = pd.read_csv(filepaths['csv']['TrafficCrashesChicago.csv'], dtype=dtypes, parse_dates = ['CRASH_DATE', 'DATE_POLICE_NOTIFIED'])

# Delete Data from before 2017
crashes = crashes[crashes.CRASH_DATE > '2018-01-01 00:00:00']
# Delete January 2020 Data because we don't have data for the entire month of January 
crashes = crashes[crashes.CRASH_DATE < '2020-01-01 00:00:00']

# Translate Crash Hours into Police Shifts
# Police Officer Shift Intervals. 1 starts at 6AM, 2 starts at 2PM, and 3 starts at 10PM.
shifts = dict.fromkeys([22, 23, 0, 1, 2, 3, 4, 5], 3)
shifts.update(dict.fromkeys([6, 7, 8, 9, 10, 11, 12, 13], 1))
shifts.update(dict.fromkeys([14, 15, 16, 17, 18, 19, 20, 21], 2))
crashes['SHIFT'] = crashes['CRASH_DATE'].dt.hour.map(shifts).astype('category')


# We are going to use RD_NO as a way to count the number of accidents
crashes = crashes.set_index(['CRASH_DATE', 'SHIFT', 'BEAT_OF_OCCURRENCE'])
crashes['RD_NO'] = 1

# Resample to Hourly
crashes_grouped = crashes.groupby([pd.Grouper(level=0, freq='H'), 'SHIFT', 'BEAT_OF_OCCURRENCE'])
columns_summed = ['NUM_UNITS', 'INJURIES_TOTAL', 'INJURIES_FATAL',
                  'INJURIES_INCAPACITATING', 'INJURIES_NON_INCAPACITATING', 'INJURIES_REPORTED_NOT_EVIDENT',
                  'INJURIES_NO_INDICATION', 'INJURIES_UNKNOWN', 'RD_NO']
columns_mode = ['weather_main', 'weather_description']
crashes_grouped_summed = crashes_grouped[columns_summed].sum()
crashes_grouped_summed[columns_summed] = crashes_grouped_summed[columns_summed].fillna(0)

# Merge weather and crashes dataset

merged_crashes = crashes_grouped_summed.reset_index().merge(weather,
                                                              how="left",
                                                              left_on='CRASH_DATE',
                                                              right_on='dt_iso')

merged_crashes['CRASH_DATE'] = pd.to_datetime(merged_crashes['CRASH_DATE'],format="%Y-%m-%d %H:00:00 +0000 UTC")
merged_crashes_resampled = merged_crashes.groupby([pd.Grouper(key='CRASH_DATE', freq='D'), 'SHIFT', 'BEAT_OF_OCCURRENCE'])
merged_crashes_summed = merged_crashes_resampled[columns_summed].sum()
merged_crashes_mode = merged_crashes_resampled[columns_mode].apply(lambda x: x.mode().iloc[0])
merged_crashes_daily = pd.concat([merged_crashes_summed, merged_crashes_mode], axis=1)
merged_crashes_daily = merged_crashes_daily.reset_index()

print(merged_crashes)
merged_crashes_daily.to_csv('data/final_dataset_daily.csv', index=False)




