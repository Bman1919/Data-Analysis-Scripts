import math

import numpy as np
import pandas as pd

def rolling_mean(data, window_size):
    return pd.Series(data).rolling(window=window_size, min_periods=1, center=True).mean().to_numpy()

def histogram_layout(num_channels):
    n_cols = min(4, max(1, num_channels))
    n_rows = max(1, math.ceil(num_channels / n_cols))
    return n_rows, n_cols

def get_probe_index_by_fullname(probe_name, probes_df):
    matching_probes = probes_df[probes_df['arm_code'] + probes_df['loc_code'].astype(str) == probe_name]
    if not matching_probes.empty:
        return matching_probes.index[0]
    else:
        return None

def get_probe_index_by_name(probe_arm_code, probe_loc_code, probes_df):
    return get_probe_index_by_fullname(probe_arm_code + probe_loc_code, probes_df)

def get_probe_name_by_index(probe_index, probes_df):
    if probe_index in probes_df.index:
        probe = probes_df.loc[probe_index]
        return f"{probe['arm_code']}", f"{probe['loc_code']}"
    else:
        return None

def get_probe_fullname_by_index(probe_index, probes_df):
    if probe_index in probes_df.index:
        probe = probes_df.loc[probe_index]
        return f"{probe['arm_code']}{probe['loc_code']}"
    else:
        return None