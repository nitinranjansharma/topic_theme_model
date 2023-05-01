import os

import pandas as pd


def get_csv(path):
    """ Get the dataframe with indexed data"""
    if os.path.exists(path):
        df = pd.read_csv(path)
        return df
    else:
        return pd.DataFrame()
