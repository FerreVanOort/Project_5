## Formulas

# Imports
import pandas as pd
import numpy as np
import streamlit as st
import scipy.stats as sp
import datetime
import plotly.express as px

# Functions

def amount_of_busses(planning):
    planning = pd.DataFrame(planning)
    busses = planning[planning.columns[len(planning.columns) - 1]].unique()
    return busses

def length_activities(active_ride):
    active_ride[active_ride.columns[3]] = pd.to_datetime(active_ride.iloc[:,3], format = "%H:%M:%S")
    active_ride[active_ride.columns[4]] = pd.to_datetime(active_ride.iloc[:,4], format = "%H:%M:%S")
    active_ride["diff"] = active_ride[active_ride.columns[4]] - active_ride[active_ride.columns[3]]
    return active_ride

def cleanup_excel(active_ride:pd.Dataframe):
    active_ride.columns = active_ride.columns.str.replace(" ", "_")
    return active_ride

def check_format_excel(active_ride:pd.Dataframe):
    items = list(active_ride.columns)
    true_amount = items == ['start_location', 'end_location', 'start_time', 'end_time', 'activity', 'line', 'energy_consumption', 'bus']
    if true_amount:
        st.success("The uploaded file has the right format")
    if not true_amount:
        st.error("The uploaded file does not have the right format")
        
