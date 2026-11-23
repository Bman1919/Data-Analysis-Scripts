import csv
from datetime import datetime

import numpy as np
import pandas as pd


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

    for column_name in signal_columns:
        data[column_name] = pd.to_numeric(data[column_name], errors="coerce")
        data[column_name] = data[column_name].replace({val: None for val in bad_values})
    
    return time_array, dates_x, signal_columns
