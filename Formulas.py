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

def charge_time(planning:pd.DataFrame):
    """
    Input:
        Bus planning as a Pandas DataFrame

    Returns:
        Success or error message depending on sufficient charging
    """
    # Gathers all charging moments in planning
    charging_moments = planning[planning.iloc[:,5].str.contains("charging")]
    
    # Checks if charge time is longer than given minimum
    short_charge = charging_moments[charging_moments['diff'] < pd.Timedelta(minutes = 15)]
    
    # Success or error dependent on if there are any errors
    if len(short_charge) > 0:
        st.error(f"There are {len(short_charge)} times a bus is charged too short")
        with st.expander("More information on charging times"):
            st.write("Insufficient charge time")
            short_charge = short_charge[["start_time", "end_time", "activity"]]
            st.write(pd.DataFrame(short_charge))
    else:
        st.success("All buses have sufficient charging times")
        
def convert_to_time(value):
    """
    Input:
        Time value from any given columns
        
    Returns:
        Converted time value to datetime.time-object
    """
    # Checks if value is already in correct format
    if isinstance(value, datetime.time):
        return value
    try:
        return pd.to_datetime(value, format = '%H:%M').time()
    except ValueError:
        return pd.to_datetime(value, format = '%H:%M:%S').time()
    
def create_gannt_chart(planning:pd.DataFrame):
    """
    Input:
        Bus planning as a Pandas DataFrame

    Returns:
        Gannt chart of given bus planning
    """
    planning["start_time"] = pd.to_datetime(planning["start_time"])
    planning["end_time"] = pd.to_datetime(planning["end_time"])
    fig = px.timeline(planning, x_start = "start_time", x_end = "end_time", y = "Loop", color = "activity")
    fig.update_yaxes(tickmode = 'linear', tick0 = 1, dtick = 1, autorange = 'reversed', showgrid = True, gridcolor = 'w', gridwidth = 1)
    fig.update_xaxes(tickformat = '%H:%M', showgrid = True, gridcolor = 'w', gridwidth = 1)
    fig.update_layout(title = dict(text = 'Gantt chart of the given bus planning', font = dict(size = 30)))
    fig.update_layout(legend = dict(
    yanchor = 'bottom',
    y = 0.01,
    xanchor = 'right',
    x= 1.23
    ))
    return st.plotly_chart(fig)
