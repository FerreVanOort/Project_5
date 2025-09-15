# Main tool

# Imports
import pandas as pd
import numpy as np
import streamlit as st
import scipy.stats as sp
import datetime
import plotly.express as px

# Functions

# Function for energy constants
def energy_constants_usage():
    return {
        "min_energy_consumption": 0.7,
        "mid_energy_consumption": 1.2,
        "max_energy_consumption": 2.5,
        "stationary_consumption": 5
    }
# Min, mid, and max energy consumption in kWh/km and stationary in kWh

# Function to inspect the route
def inspect_route(row, df):
    if ((row['start_location'] == 'ehvapt' and row['end_location'] == 'ehvbst') or
        (row['start_location'] == 'ehvbst' and row['end_location'] == 'ehvapt')):
        match = df[
            (df['start_location'] == row['start_location']) &
            (df['start_time'] == row['departure_time']) &
            (df['end_location'] == row['end_location']) &
            (df['activity'] == 'service trip')
        ]
        return not match.empty
    return False


df = pd.read_excel('Bus Planning.xlsx')
df.columns = df.columns.str.replace(" ", "_")