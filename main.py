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
from data_io import read_keyence, read_probes_csv
from plotting import plot_channel_mean_differences, plot_full, plotting_newton_cooling_fit, plot_outside_comparison

# Global plot settings
plt.rcParams["timezone"] = 'America/New_York'

# Setup parser
parser = argparse.ArgumentParser(
    prog='Probe Data Analysis',
    description='Read a Keyence CSV file and create temperature-channel plots and statistics.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

# Parser arguments
parser.add_argument('--database', type=str, default='./Data', help='Input Keyence CSV file database')
parser.add_argument('--probes', type=str, default='./probes.csv', help='Input probes CSV file')
parser.add_argument('--dates', help="Dates for the data") # TODO: Not yet implemented

parser.add_argument('--reference_probe', type=int, default=32, help='Reference probe number for temperature analysis')
parser.add_argument('--sigma-multiplier', type=float, default=3.0, help='Multiplier for standard deviation thresholding')
parser.add_argument('--title', type=str, default='Temperature Measurement', help='Title used for the full-data plot')
parser.add_argument('--verbose', action='store_true', help='Enable verbose output for debugging and information purposes')

#Plot options
parser.add_argument('--outside_comparison', action='store_true', help='Subtracts outside probe from plots') # Assumes probe labelled AO is the outside probe
parser.add_argument('--plot_full', action='store_true', help='Plot the full data')
parser.add_argument('--newton_cooling_fit', action='store_true', help='Fit the Newton cooling model to the data') # TODO: NOT YET IMPLEMENTED
args = parser.parse_args()

# Validate arguments
if args.sigma_multiplier <= 0: parser.error('--sigma-multiplier must be greater than zero')
if args.verbose: print("Verbose output enabled")

# Refers to the main function in data_io.py to read the Keyence CSV file
data, time_array, dates_x, column_names = read_keyence(args)

start_date = dates_x[0] if len(dates_x) > 0 else None
end_date = dates_x[-1] if len(dates_x) > 0 else None
safe_filename = f"{np.datetime_as_string(start_date,unit='D').replace('-', '')}_{np.datetime_as_string(end_date,unit='D').replace('-', '')}" if start_date and end_date else "unknown_dates"

probes_df = read_probes_csv(args.probes)
if args.verbose:
    for probe in probes_df.itertuples():
        if column_names[probe.Index] == f"(NR-{probe.nr_num})TH-CH0{probe.ch_num}":
            print(f"{column_names[probe.Index]} X{probe.extension_num:02d} : {probe.arm_code}-{probe.loc_code}")

# Plotting the data for plot_full if requested
if(args.plot_full):
    if args.newton_cooling_fit:
        fig, fig2 = plotting_newton_cooling_fit(args, data=data, column_names=column_names, dates_x=dates_x, title=args.title)
    else:
        fig, fig2 = plot_full(args, data=data, column_names=column_names, dates_x=dates_x, title=args.title)
    if fig is not None:
        fig.savefig(f"./plots/temperature_figure_{safe_filename}.png", dpi=500)
        # Succesfully saved fig1 at f"temperature_figure_{safe_filename}.png"
        if args.verbose:
            print(f"Saved full data plot as 'temperature_figure_{safe_filename}.png'")
    if fig2 is not None:
        fig2.savefig(f"./plots/temperature_histogram_{safe_filename}.png", dpi=500)
        if args.verbose:
            print(f"Saved full data histogram as 'temperature_histogram_{safe_filename}.png'")

if args.outside_comparison:
    fig, fig2 = plot_outside_comparison(data_frame=data, probes_df=probes_df, dates_x=dates_x, channel_names=column_names, title="Temperature Difference from Outside Probe")
    if fig is not None:
        fig.savefig(f"./plots/outside_comparison_figure_{safe_filename}.png", dpi=500)
        if args.verbose:
            print(f"Saved outside comparison plot as 'outside_comparison_figure_{safe_filename}.png'")
    if fig2 is not None:
        fig2.savefig(f"./plots/outside_comparison_histogram_{safe_filename}.png", dpi=500)
        if args.verbose:
            print(f"Saved outside comparison histogram as 'outside_comparison_histogram_{safe_filename}.png'")