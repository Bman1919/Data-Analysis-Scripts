import math

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

from analysis import compute_channel_mean_differences, histogram_layout, numeric_columns, fit_newton_cooling

def plot_full(args, dates_x, data=None, dataMatrix=None, column_names=None, no_hist=True, title="Temperature Measurement"):
    if data is not None:
        return std_plot_full(args, dates_x, column_names, data, no_hist=no_hist, title=title)
    if dataMatrix is not None:
        data = pd.DataFrame(data=dataMatrix, columns=column_names)
        return std_plot_full(args, dates_x, column_names, data, no_hist=no_hist, title=title)
    return None, None

def plotting_newton_cooling_fit(args, dates_x, data=None, dataMatrix=None, column_names=None, no_hist=False, title="Temperature Measurement"):
    if data is None:
        if dataMatrix is None:
            return None, None
        data = pd.DataFrame(data=dataMatrix, columns=column_names)
    result_columns = [f"Env Temp Fit ({column_name})" for column_name in column_names] + [f"Temperature Conducitivity ({column_name})" for column_name in column_names]
    resultant_data = pd.DataFrame(data=np.zeros((len(data), len(result_columns))), columns=result_columns)
    rolling_window = getattr(args, "rolling_window", 3000)
    sigma_multiplier = getattr(args, "sigma_multiplier", 3.0)
    verbose = getattr(args, "verbose", False)

    signal_columns = numeric_columns(data, column_names)

    new_columns = []

    date_values = np.asarray(dates_x)
    if len(date_values) > 1 and np.issubdtype(date_values.dtype, np.datetime64):
        sample_interval = float(
            np.median(np.diff(date_values) / np.timedelta64(1, "s"))
        )
    else:
        sample_interval = 1.0

    for column_name in signal_columns:
        newton_rolling_window = min(10, len(data))
        local_time = np.arange(newton_rolling_window, dtype=float) * sample_interval
        environment_values = np.full(len(data), np.nan)
        conductivity_values = np.full(len(data), np.nan)
        values = data[column_name].to_numpy(dtype=float)

        if newton_rolling_window >= 4:
            half_window = newton_rolling_window // 2
            for center in tqdm(range(half_window, len(values) - half_window), desc=f"Fitting Newton cooling model for {column_name}"):
                start = center - half_window
                stop = start + newton_rolling_window
                window = values[start:stop]
                valid = np.isfinite(window)
                if valid.sum() < 4:
                    continue
                try:
                    parameters, _ = fit_newton_cooling(
                        local_time[valid], window[valid]
                    )
                except (RuntimeError, ValueError, IndexError, FloatingPointError):
                    continue
                environment_values[center] = parameters[0]
                conductivity_values[center] = parameters[1]
                # if center % 10000 == 0 and verbose:
                #     print(f"Processed {center} of {len(values)} data points for {column_name}")

        environment_name = f"Env Temp Fit ({column_name})"
        conductivity_name = f"Temperature Conducitivity ({column_name})"
        resultant_data[environment_name] = environment_values
        resultant_data[conductivity_name] = conductivity_values
        new_columns.append(environment_name)
        new_columns.append(conductivity_name)

        valid_environment = environment_values[np.isfinite(environment_values)]
        if valid_environment.size:
            print(
                f"Fitted Newton cooling model for {column_name}: "
                f"T_env={np.mean(valid_environment):.2f}"
            )
        else:
            print(f"Warning: Could not fit Newton cooling model for {column_name}")

    return plot_full(
        args, dates_x, data=resultant_data, column_names=new_columns,
        no_hist=no_hist, title=title
    )

def std_plot_full(args, dates_x, column_names, data, no_hist=False, title="Temperature Measurement"):
    signal_columns = numeric_columns(data, column_names)
    if not signal_columns:
        return None, None

    working_data = data.copy()
    rolling_window = getattr(args, "rolling_window", 3000)
    sigma_multiplier = getattr(args, "sigma_multiplier", 3.0)
    newton_cooling_fit = getattr(args, "newton_cooling_fit", False)
    verbose = getattr(args, "verbose", False)

    if not args.no_mean:
        for column_name in tqdm(signal_columns,desc=f"Calculating the rolling mean and standard deviation {title}"):
            working_data[f"Rolling Mean ({column_name})"] = working_data[column_name].rolling(
                window=rolling_window, center=True
            ).mean()
            ch_residual = working_data[column_name] - working_data[f"Rolling Mean ({column_name})"]
            working_data[f"Rolling STD ({column_name})"] = ch_residual.rolling(
                window=rolling_window, center=True
            ).std()

    if verbose:
        print(f"Calculated rolling means and standard deviations for {title}")

    fig, ax = plt.subplots(figsize=(11, 8.5))
    plt.gcf().autofmt_xdate()

    ax.set_title(
        f"{title} (Mean +/- {sigma_multiplier:g} Sigma Noise)"
        if not args.no_mean
        else title
    )
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%dth %H:%M:%S"))
    ax.set_xlabel("Time")
    ax.grid(True)
    ax.set_ylabel("Temperature (°C)")

    for column_name in signal_columns:
        if newton_cooling_fit and f"Env Temp Fit ({column_name})" in working_data.columns:
            ax.plot(
                dates_x,
                working_data[f"Env Temp Fit ({column_name})"],
                color='orange',
                linewidth=1.0,
                label=f"Env Temp Fit ({column_name})",
            )
        elif not args.no_mean:
            rolling_mean = working_data[f"Rolling Mean ({column_name})"]
            rolling_std = working_data[f"Rolling STD ({column_name})"]
            lower_bound = rolling_mean - sigma_multiplier * rolling_std
            upper_bound = rolling_mean + sigma_multiplier * rolling_std
            ax.fill_between(
                dates_x,
                lower_bound,
                upper_bound,
                color='skyblue',
                alpha=0.5,
                linewidth=0,
            )
            ax.plot(dates_x, rolling_mean, linewidth=1, label=column_name)
        else:
            ax.plot(dates_x, working_data[column_name], label=column_name)

    if len(signal_columns) <= 8:
        ax.legend(loc='upper right')

    if no_hist:
        fig2 = None
    else:
        n_rows, n_cols = histogram_layout(len(signal_columns))
        fig2, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(n_cols * 3.0, n_rows * 2.6), squeeze=False)
        flat_axes = axes.flat

        all_values = working_data[signal_columns].to_numpy().ravel()
        valid_values = all_values[np.isfinite(all_values)]
        if valid_values.size:
            min_value = float(valid_values.min())
            max_value = float(valid_values.max())
            bin_step = 0.1
            bins = np.arange(
                np.floor(min_value / bin_step) * bin_step,
                np.ceil(max_value / bin_step) * bin_step + bin_step,
                bin_step,
            )
        else:
            bins = np.linspace(-1, 1, 11)

        for idx, column_name in tqdm(list(enumerate(signal_columns)), desc="Generating temperature histograms"):
            channel_ax = flat_axes[idx]
            channel_ax.grid(True)
            channel_ax.hist(working_data[column_name], bins=bins, color='steelblue', alpha=1)
            channel_ax.set_title(column_name)
            channel_ax.set_xlabel("Temperature (°C)")
            channel_ax.set_ylabel("Count")

        for axis in flat_axes[len(signal_columns):]:
            axis.set_visible(False)

        fig2.suptitle("Temperature Histogram")
        fig2.tight_layout(rect=[0, 0, 1, 0.98])

    return fig, fig2


def plot_channel_mean_differences(data_frame, dates_x, channel_names, title="Temperature Difference from Mean"):
    diff_frame, _ = compute_channel_mean_differences(data_frame, channel_names)
    if diff_frame.empty:
        return None, None

    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.grid(True)
    ax.axhline(0, color='black', linewidth=1, linestyle='--')
    ax.set_title(title)
    ax.set_ylabel("Temperature Difference (°C)")
    ax.set_xlabel("Time")

    for column_name in diff_frame.columns:
        ax.plot(dates_x, diff_frame[column_name], label=column_name)

    if len(diff_frame.columns) <= 8:
        ax.legend(loc='upper right', fontsize='x-small')

    n_rows = max(1, math.ceil(len(diff_frame.columns) / 4))
    fig2, axes = plt.subplots(nrows=n_rows, ncols=4, figsize=(15, 2.7 * n_rows), squeeze=False)
    flat_axes = axes.flat

    for idx, column_name in tqdm(enumerate(diff_frame.columns), desc="Generating temperature difference histograms"):
        channel_ax = flat_axes[idx]
        channel_ax.grid(True)
        values = diff_frame[column_name].dropna().to_numpy()
        if values.size > 0:
            bins = np.histogram_bin_edges(values, bins=30)
            channel_ax.hist(values, bins=bins, color='purple', alpha=0.7)
        channel_ax.axvline(0, color='black', linestyle='--', linewidth=1)
        channel_ax.set_title(column_name)
        channel_ax.set_xlabel("ΔT (°C)")
        channel_ax.set_ylabel("Count")

    for axis in flat_axes[len(diff_frame.columns):]:
        axis.set_visible(False)

    fig2.tight_layout(rect=[0, 0, 1, 0.98])
    return fig, fig2
