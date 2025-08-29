import pandas as pd
import streamlit as st
from io import StringIO

def txt2csv(uploaded):
    # Read the file content directly from UploadedFile
    lines = uploaded.getvalue().decode("utf-8").splitlines()

    # Find the start of actual data (after "Time\tReal\tReal\tReal")
    start_index = None
    for i, line in enumerate(lines):
        if line.strip().startswith("Time"):
            start_index = i + 1
            break

    if start_index is None:
        st.warning("❌ Header 'Time' not found in file.")
        return None, None

    # Read the actual data into DataFrame
    data_str = '\n'.join(lines[start_index:])
    df = pd.read_csv(StringIO(data_str), sep='\t', engine='python', header=None)

    # Check number of columns and assign names accordingly
    if df.shape[1] >= 4:
        df = df.iloc[:, :4]
        df.columns = ['Time', 'DT1.HOR', 'DT1.VER', 'DT1.AXL']
    else:
        st.warning("⚠️ Not enough columns in file.")
        return None, None

    # Assign label y based on filename
    filename = uploaded.name
    if "Norm" in filename:
        df['y'] = 0
    elif "Bandul" in filename:
        df['y'] = 1
    else:
        df['y'] = -1

    # Convert filename
    csv_filename = filename.replace('.txt', '.csv')

    return df, csv_filename
