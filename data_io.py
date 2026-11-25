import csv
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


# Reading probes.csv file
def read_probes_csv(filename):
    if not Path(filename).is_file():
        raise ValueError(f"Probes CSV file '{filename}' does not exist")
    if not filename.lower().endswith(".csv"):
        raise ValueError(f"Probes CSV file '{filename}' is not a CSV file")

    probes_df = pd.read_csv(filename)

    probes_df.columns = probes_df.columns.str.strip()  # Strip whitespace from column names
    return probes_df

# Keyence CSV file reading functions
def read_keyence(args):
    if not args.database:
        raise ValueError("Database path is not provided")
    if not Path(args.database).is_dir():
        return read_keyence_csv(args.database)
    return read_keyence_dir(args)


def read_keyence_dir(args):
    data_files = list(Path(args.database).glob("*.csv"))
    if not data_files:
        raise ValueError("No CSV files found in the database directory")

    data_out, time_array_out, dates_x_out, signal_columns_out = pd.DataFrame(), pd.Series([], dtype=float), np.array([],dtype='datetime64[us]'), pd.Series([], dtype=str)
    for file in tqdm(data_files, desc="Reading data files"):
        data, time_array, dates_x, _ = read_keyence_csv(file)
        data_out = pd.concat([data_out, data], ignore_index=True)
        time_array_out = pd.concat([time_array_out, time_array], ignore_index=True)
        dates_x_out = np.append(dates_x_out, dates_x)
        # TODO: Consider putting termination around the dates argument

    signal_columns_out = data_out.columns[2:]  # Assuming the first two columns are Date and Time

    return data_out, np.array(time_array_out), np.array(dates_x_out), signal_columns_out

def read_keyence_csv(filename):
    skip_rows = None
    data_end = None
    column_names = None

    with open(filename, "r", encoding="utf-8", errors="ignore") as handle:
        for i, line in enumerate(handle):
            if line.startswith("#EndHeader,"):
                header_fields = next(csv.reader([line]))
                column_names = ["Date", "Time (µs)", *header_fields[2:]]
                skip_rows = i + 1
            elif skip_rows is not None and line.startswith("#BeginMark"):
                data_end = i
                break

    if skip_rows is None or data_end is None or column_names is None:
        raise ValueError("Could not find the measurement section in the CSV file")

    data = pd.read_csv(
        filename,
        encoding="utf-8",
        skiprows=skip_rows,
        nrows=data_end - skip_rows,
        header=None,
        names=column_names,
    )
    data.columns = data.columns.str.strip()

    date_format = "%m/%d/%Y %H:%M:%S"
    time_array = data["Time (µs)"] + data["Date"].apply(
        lambda value: datetime.strptime(value, date_format).timestamp() * 1e6
    )
    dates_x = np.array(time_array).astype("datetime64[us]")
    signal_columns = data.columns[2:]

    bad_values = [1372.0,-32768.0]
    min_possible = -10
    max_possible = 20

    for column_name in signal_columns:
        data[column_name] = pd.to_numeric(data[column_name], errors="coerce")
        data[column_name] = data[column_name].apply(lambda x: np.nan if x in bad_values or x < min_possible or x > max_possible else x)
    return data, time_array, dates_x, signal_columns
