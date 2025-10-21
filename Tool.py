## Main tool

# Imports

from Formulas import main
import Formulas
import io
from datetime import datetime
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Prototype groep 8", layout="wide")

# Optimalisatie
# main(timetable, planning, distancematrix)

# Sidebar
st.sidebar.header("Menu")
st.sidebar.radio("", ["Main execution", "Advanced Options", "User Manual", "About Us"], label_visibility="collapsed")

# Header
st.title("Prototype group 8")
st.subheader("Upload existing bus plan here!")
# Accept only .xlsx to avoid xlrd dependency for .xls
uploaded = st.file_uploader("Upload Excel file (.xlsx)", type=["xlsx"])

# Required columns (exact labels, case-insensitive)
required_cols = [
    "start location", "end location", "start time", "end time",
    "activity", "line", "energy consumption", "bus"
]

# State for right-side/download
if "df_opt" not in st.session_state:
    st.session_state.df_opt = None
if "opt_filename" not in st.session_state:
    st.session_state.opt_filename = None

left, right = st.columns(2, gap="large")

