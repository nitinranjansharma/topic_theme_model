import os
import re
import pandas as pd


def get_csv(path):
    """ Get the dataframe with indexed data"""
    if os.path.exists(path):
        df = pd.read_csv(path)
        return df
    else:
        return pd.DataFrame()


def replace_enter(str):
    return re.sub(r'(\n+)(?=[A-Z])', r'.', str)
