import os
import multiprocessing
from joblib import Parallel, delayed
from pathlib import Path

import numpy as np
import geopandas as gdp
from pandas import read_csv


def applyParallel(grouped_df, func):
    max_cores = multiprocessing.cpu_count - 1
    result = Parallel(n_jobs=max_cores)(delayed(func)(group) for name, group in grouped_df)
    return pd.concat(result)

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
    data_path = Path(DATA_DIRECTORY)
    return data_path

def get_paths_for_filetype(filetype, path_object):
    files = dict.fromkeys([files.name for files in path_object.iterdir() if "".join(('.',filetype)) in files.name])
    paths = list(str(x.absolute()) for x in path_object.iterdir() if "".join(('.',filetype)) in str(x.absolute()))
    for filename in files.keys():
        files[filename] = [datapath for datapath in paths if filename in datapath]
    return files
