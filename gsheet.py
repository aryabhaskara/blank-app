import streamlit as st
import gspread
import pandas as pd
from gspread_dataframe import set_with_dataframe, get_as_dataframe
from google.oauth2.service_account import Credentials

def init_gspread(spreadsheet_name="FFT_Features_Output"):
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = Credentials.from_service_account_info(st.secrets["gcp_service_account"], scopes=scope)
    gc = gspread.authorize(creds)

    try:
        sh = gc.open(spreadsheet_name)
    except gspread.SpreadsheetNotFound:
        sh = gc.create(spreadsheet_name)

    ws_features = sh.sheet1
    try:
        ws_labels = sh.worksheet("Labels")
    except:
        ws_labels = sh.add_worksheet("Labels", rows=100, cols=5)

    try:
        df_existing = get_as_dataframe(ws_features).dropna(how='all')
    except:
        df_existing = pd.DataFrame()

    try:
        df_labels = get_as_dataframe(ws_labels).dropna(how='all')
    except:
        df_labels = pd.DataFrame()

    return sh, ws_features, ws_labels, df_existing, df_labels

def save_to_gsheet(ws_features, ws_labels, df_existing, df_labels, new_rows, new_labels, ordered_columns):
    import pandas as pd
    from gspread_dataframe import set_with_dataframe

    if new_rows:
        df_new = pd.DataFrame(new_rows, columns=ordered_columns)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        set_with_dataframe(ws_features, df_combined, resize=False)

    if new_labels:
        df_new_labels = pd.DataFrame(new_labels, columns=["filename", "y"])
        df_labels_combined = pd.concat([df_labels, df_new_labels], ignore_index=True)
        set_with_dataframe(ws_labels, df_labels_combined, resize=False)
