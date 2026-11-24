import math

import numpy as np
import pandas as pd

def rolling_mean(data, window_size):
    return pd.Series(data).rolling(window=window_size, min_periods=1, center=True).mean().to_numpy()

def histogram_layout(num_channels):
    n_cols = min(4, max(1, num_channels))
    n_rows = max(1, math.ceil(num_channels / n_cols))
    return n_rows, n_cols
