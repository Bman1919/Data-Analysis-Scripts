
from scipy.optimize import curve_fit
from analysis_helpers import *

def fit_newton_cooling(time_array,raw_temperature_array):
    temperature_values = raw_temperature_array.astype(float)

    mean_window = 10

    temperature = rolling_mean(temperature_values, mean_window)

    mean_w = 10

    average_temp = rolling_mean(temperature, mean_w)

    return temperature, average_temp

    # temperature_environment = np.zeros(len(temperature_values))
    # env_time = np.zeros(len(temperature_values))

    # fit_window = 100

    # for start in range(0, len(temperature_values) - fit_window + 1, fit_window):
    #     end = start + fit_window
    #     x = temperature[start:end]
    #     y = average_temp[start:end]

    #     alpha2, alpha1 = np.polyfit(x, y, 1) # Fit a linear model to the data, coefficients alpha and beta are the slope and intercept of the line, respectively

    #     temperature_environment[start] = alpha1 / (1 - alpha2) # Calculate the environmental temperature using the fitted linear model
    #     env_time[start] = time_array[start + fit_window // 2] # Store the corresponding time value for the environmental temperature
    
    # return env_time, temperature_environment

def numeric_columns(data_frame, column_names=None):
    if column_names is None:
        column_names = list(data_frame.columns)
    return [
        column_name
        for column_name in column_names
        if column_name in data_frame.columns and pd.api.types.is_numeric_dtype(data_frame[column_name])
    ]


def compute_channel_mean_differences(data_frame, channel_names):
    numeric = numeric_columns(data_frame, channel_names)
    if len(numeric) < 2:
        return pd.DataFrame(), numeric

    channel_mean = data_frame[numeric].mean(axis=1)
    diff_frame = data_frame[numeric].sub(channel_mean, axis=0)
    diff_frame.columns = [f"{column_name} - Mean" for column_name in numeric]
    return diff_frame, numeric

