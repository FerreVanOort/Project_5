## Formulas

# Imports
import pandas as pd
import numpy as np
import streamlit as st
import scipy.stats as sp
import datetime
import plotly.express as px


# Functions

def cleanup_excel(planning:pd.Dataframe) -> pd.DataFrame:
    """
    Input:
        Bus planning as a Pandas Dataframe

    Returns:
        Cleaned up bus planning as a Pandas DataFrame
    """
    # Replace spaces with underscores for easier coding
    planning.columns = planning.columns.str.replace(" ", "_")
    # Replaces all upper case letters with lower case ones
    planning.columns = planning.columns.str.lower()
    return planning

def check_format_excel(planning:pd.Dataframe):
    """
    Input:
        Bus planning as a Pandas DataFrame
        
    Returns:
        Success or fail message depending on if the format is correct
    """
    # Puts all columns names in a list
    items = list(planning.columns)
    # Checks if all columns names align with the needed input
    true_amount = items == ['start_location', 'end_location', 'start_time', 'end_time', 'activity', 'line', 'energy_consumption', 'bus']
    # Success or fail message dependent on true_amount
    if true_amount:
        st.success("The uploaded file has the right format")
    if not true_amount:
        st.error("The uploaded file does not have the right format")
        
def charging_check(planning:pd.DataFrame):
    """
    Input:
        Bus planning as a Pandas DataFrame
        
    Returns:
        Success or fail message depending on correct charging rates
    """
    # Filters all errors
    errors = planning[(planning['activity'] == 'charging') & (planning['energy_consumption'] >= 0)]
    # Success or error dependent on if there are any errors
    if errors.empty:
        st.success("Charging rates are correct")
    else:
        st.error("Charging rates have positive value")
        st.subheader("Error lines:")
        st.dataframe(errors, use_container_width = True)
        
def length_activities(planning:pd.DataFrame) -> pd.DataFrame:
    """
    Input:
        Bus planning as a Pandas Dataframe

    Returns:
        Bus planning as a Pandas DataFrame with an extra row with duration of activities
    """
    # Puts planning columns in similar datetime format
    planning[planning.columns[3]] = pd.to_datetime(planning.iloc[:,3], format = "%H:%M:%S")
    planning[planning.columns[4]] = pd.to_datetime(planning.iloc[:,4], format = "%H:%M:%S")
    # Adds new column to DataFrame
    planning["diff"] = planning[planning.columns[4]] - planning[planning.columns[3]]
    return planning

