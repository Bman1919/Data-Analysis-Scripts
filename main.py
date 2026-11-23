# Imports
import re
from pathlib import Path

import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

# From Local Files
from analysis import compute_channel_mean_differences
from data_io import read_keyence_csv
from plotting import plot_channel_mean_differences, plot_full, plotting_newton_cooling_fit

# Global plot settings
plt.rcParams["timezone"] = 'America/New_York'

# Setup parser
parser = argparse.ArgumentParser(
    prog='Probe Data Analysis',
    description='Read a Keyence CSV file and create temperature-channel plots and statistics.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

# Parser arguments
parser.add_argument('filename', help='Input Keyence CSV file')
parser.add_argument('--plot_full', action='store_true', help='Plot all numeric channels over time and their histograms; this is the default action')
parser.add_argument(
    '--plot_phase_hist', '--plot_melt_hist',
    dest='plot_phase_hist',
    action='store_true',
    help='Plot the phase where channels are near the configured reference temperature',
)
parser.add_argument('--no_mean', action='store_true', help='Plot raw channel data without rolling mean and sigma band')
parser.add_argument('--calc_diff', action='store_true', help='Plot each numeric channel minus the row-wise mean of all numeric channels')
parser.add_argument('--hist_window', type=int, default=-1, help='Create a difference animation using this many samples per frame; disable when non-positive')
parser.add_argument('--rolling-window', type=int, default=3000, help='Number of samples in each rolling mean and standard-deviation window')
parser.add_argument('--sigma-multiplier', type=float, default=3.0, help='Distance from the rolling mean to each edge of the shaded sigma band')
parser.add_argument('--phase-center', type=float, default=0.0, help='Reference temperature used by --plot_phase_hist')
parser.add_argument('--phase-range', type=float, default=1.0, help='Allowed distance from --phase-center for --plot_phase_hist')
parser.add_argument('--title', type=str, default='Temperature Measurement', help='Title used for the full-data plot')
parser.add_argument('--newton-cooling-fit', action='store_true', help='Fit a Newton cooling model to the data and plot the result')
parser.add_argument('--verbose', action='store_true', help='Enable verbose output for debugging and information purposes')
parser.add_argument('--t_mean_plot',action='store_true', help='Plot the rolling mean of the temperature data')
args = parser.parse_args()

# Validate arguments
if args.rolling_window < 1:
    parser.error('--rolling-window must be a positive integer')
if args.sigma_multiplier <= 0:
    parser.error('--sigma-multiplier must be greater than zero')
if args.verbose:
    print("Verbose output enabled")
if args.newton_cooling_fit and not (args.plot_full or args.plot_phase_hist or args.calc_diff):
    parser.error('--newton-cooling-fit requires at least one of --plot_full, --plot_phase_hist, or --calc_diff') # TODO: VALIDATE that this works

# Default to plotting full data if no specific action is requested
if not (args.plot_full or args.plot_phase_hist or args.calc_diff):
    args.plot_full = True

# Refers to the main function in data_io.py to read the Keyence CSV file
data, time_array, dates_x, column_names = read_keyence_csv(args.filename)
if args.verbose:
    print(f"Data loaded successfully: {data.shape}")

# Removes invalid characters from the filename to create a safe inputt filename
safe_filename = Path(args.filename).stem
safe_filename = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", safe_filename).rstrip(" .")
if not safe_filename:
    raise ValueError("Input filename does not provide a valid output filename")
elif args.verbose:
    print(f"Safe input filename: {safe_filename}")

# Plots temperature_mean vs temperature
if args.t_mean_plot:
    signal_columns = [
        column_name
        for column_name in column_names
        if pd.api.types.is_numeric_dtype(data[column_name])
    ]
    if not signal_columns:
        raise ValueError("No numeric channels found for melt-phase plotting")
    if signal_columns is None:
        raise ValueError("No temperature column found in the data")
    
    fig, ax = plt.subplots(figsize=(10, 6))

    for column_name in tqdm(signal_columns, desc=f"Plotting rolling mean for temperature channels"):

        temperature_data = data[column_name].astype(float)
        temperature_data = temperature_data.rolling(window=10,min_periods=1,center=True).mean()

        rolling_mean_temp = temperature_data.rolling(window=50, min_periods=1, center=True).mean()

        ax.plot(temperature_data, rolling_mean_temp, label=f'Rolling Mean (window={5000})', alpha=0.5)

    ax.set_title('Temperature vs Rolling Mean')
    ax.set_xlabel('Temperature (°C)')
    ax.set_ylabel('Rolling Mean Temperature (°C)')
    ax.grid(True)
    plt.savefig(f"temperature_mean_plot_{safe_filename}.png", dpi=500)
    if args.verbose:
        print(f"Saved temperature mean plot as 'temperature_mean_plot_{safe_filename}.png'")

# Plotting the data for plot_full if requested
if(args.plot_full):
    if args.newton_cooling_fit:
        fig, fig2 = plotting_newton_cooling_fit(args, data=data, column_names=column_names, dates_x=dates_x, title=args.title)
    else:
        fig, fig2 = plot_full(args, data=data, column_names=column_names, dates_x=dates_x, title=args.title)
    if fig is not None:
        fig.savefig(f"temperature_figure_{safe_filename}.png", dpi=500)
        # Succesfully saved fig1 at f"temperature_figure_{safe_filename}.png"
        if args.verbose:
            print(f"Saved full data plot as 'temperature_figure_{safe_filename}.png'")
    if fig2 is not None:
        fig2.savefig(f"temperature_histogram_{safe_filename}.png", dpi=500)
        if args.verbose:
            print(f"Saved full data histogram as 'temperature_histogram_{safe_filename}.png'")

# Plot freezing phase histogram if requested
if(args.plot_phase_hist):
    signal_columns = [
        column_name
        for column_name in column_names
        if pd.api.types.is_numeric_dtype(data[column_name])
    ]
    if not signal_columns:
        raise ValueError("No numeric channels found for melt-phase plotting")

    phase_center = args.phase_center
    phase_range = args.phase_range
    if phase_range < 0:
        parser.error('--phase-range must be non-negative')
    melt_windows = {}

    for column_name in signal_columns:
        series = data[column_name].astype(float)
        within_window = (series - phase_center).abs() <= phase_range
        if within_window.any():
            start_idx = int(np.flatnonzero(within_window.to_numpy())[0])
            end_idx = int(np.flatnonzero(within_window.to_numpy())[-1])
            melt_windows[column_name] = (start_idx, end_idx)

    if not melt_windows:
        raise ValueError(
            f"No channel entered the phase window around {phase_center:g} °C"
        )

    n_rows = max(1, int(np.ceil(len(signal_columns) / 4)))
    fig2, axes = plt.subplots(nrows=n_rows, ncols=4, figsize=(15, 2.6 * n_rows), squeeze=False)
    flat_axes = axes.flat

    for idx, column_name in enumerate(signal_columns):
        start_idx, end_idx = melt_windows.get(column_name, (0, len(data) - 1))
        channel_slice = data.iloc[start_idx:end_idx + 1]
        channel_ax = flat_axes[idx]
        channel_ax.grid(True)
        channel_ax.hist(channel_slice[column_name], bins=np.arange(-1, 1.1, 0.1), color='skyblue', alpha=0.7)
        channel_ax.set_title(column_name)
        channel_ax.set_xlabel("Temperature (°C)")
        channel_ax.set_ylabel("Count")

    for axis in flat_axes[len(signal_columns):]:
        axis.set_visible(False)

    fig, ax1 = plt.subplots(figsize=(11, 8.5))
    ax1.grid(True)
    ax1.set_title(f"Phase Temperatures Near {phase_center:g} °C (Combined)")
    ax1.set_ylabel("Temperature (°C)")
    ax1.set_xlabel("Time (us)")

    for column_name in signal_columns:
        start_idx, end_idx = melt_windows.get(column_name, (0, len(data) - 1))
        ax1.plot(
            time_array.iloc[start_idx:end_idx + 1],
            data.iloc[start_idx:end_idx + 1][column_name],
            label=column_name,
        )

    ax1.axhline(
        y=phase_center,
        color='red',
        linestyle='--',
        label=f'Reference Temperature ({phase_center:g} °C)',
    )
    ax1.legend(loc='upper right', fontsize='x-small')

    plt.show()
    fig.savefig(f"phase_temperatures_{safe_filename}.png")
    if fig2 is not None:
        fig2.savefig(f"phase_histogram_{safe_filename}.png")

# Plotting the mean-difference analysis if requested
if(args.calc_diff):
    diff_frame, signal_columns = compute_channel_mean_differences(data, column_names)
    if diff_frame.empty:
        raise ValueError("No numeric channels available for mean-difference analysis")

    fig, fig2 = plot_channel_mean_differences(data, dates_x, column_names)

    if fig is not None:
        fig.savefig(f"temperature_difference_{safe_filename}.png")
    if fig2 is not None:
        fig2.savefig(f"temperature_difference_histogram_{safe_filename}.png")

    if(args.hist_window > 0):
        all_values = diff_frame.to_numpy().ravel()
        all_values = all_values[np.isfinite(all_values)]
        if all_values.size > 0:
            bin_width = 0.1
            bin_start = np.floor(np.nanmin(all_values) / bin_width) * bin_width
            bin_end = np.ceil(np.nanmax(all_values) / bin_width) * bin_width
            animation_bins = np.arange(bin_start, bin_end + bin_width, bin_width)
            x_vals = np.linspace(float(np.nanmin(all_values)), float(np.nanmax(all_values)), 500)

            animated_frames = []
            for i in range(len(data) // args.hist_window):
                start_idx = int(i * args.hist_window)
                end_idx = min(int((i + 1) * args.hist_window), len(data))
                window_frame = diff_frame.iloc[start_idx:end_idx].copy()
                if window_frame.empty:
                    continue

                fig_window, ax_window = plt.subplots(figsize=(11, 8.5))
                ax_window.grid(True)
                ax_window.axhline(0, color='black', linestyle='--')
                ax_window.set_title(f"Mean-centered differences ({start_idx} to {end_idx})")
                ax_window.set_ylabel("ΔT (°C)")
                ax_window.set_xlabel("Time index")
                for column_name in window_frame.columns:
                    ax_window.plot(np.arange(start_idx, end_idx), window_frame[column_name].to_numpy(), label=column_name)
                ax_window.legend(loc='upper right', fontsize='x-small')

                fig_window.canvas.draw()
                rgba_buffer = fig_window.canvas.buffer_rgba()
                img = Image.frombuffer('RGBA', fig_window.canvas.get_width_height(), rgba_buffer, 'raw', 'RGBA', 0, 1)
                animated_frames.append(img.convert("RGB"))
                plt.close(fig_window)

            if animated_frames:
                animated_frames[0].save(
                    f'temperature_difference_histogram_animation_{safe_filename}.gif',
                    save_all=True,
                    append_images=animated_frames[1:],
                    optimize=False,
                    duration=200,
                    loop=0,
                )
                print("Animation saved as 'temperature_difference_histogram_animation.gif'.")
            else:
                print("No frames were generated for the animation.")