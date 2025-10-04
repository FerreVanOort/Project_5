## Formulas

# Imports
import pandas as pd
import numpy as np
import streamlit as st
import scipy.stats as sp
import datetime
from datetime import time
import plotly.express as px


# Functions

def cleanup_excel(planning:pd.DataFrame) -> pd.DataFrame:
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

def check_format_excel(planning:pd.DataFrame):
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

    Output:
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
    Turns a time value into a datetime.time-object.
    
    Input:
        Time as a string (for example '08:30' or '08:30:00'), or already a datetime.time object.
        
    Output:
        datetime.time object
    """
    if isinstance(value, time):
        return value
    
    if pd.isnull(value):
        raise ValueError("Null tijdwaarde aangetroffen")
    
    # Tries standard formats in order
    for fmt in ("%H:%M", "%H:%M:%S"):
        try:
            return pd.to_datetime(value, format=fmt).time()
        except (ValueError, TypeError):
            continue
    
    raise ValueError(f"Tijdformaat niet herkend: {value}")
    
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

def ensure_time_column(df : pd.DataFrame , column):
    """
    Uses the convert_to_time function

    Input:
        Pandas DataFrame

    Output:
        Pandas DataFrame with changed datetime columns
    """
    df[column] = df[column].apply(convert_to_time)

def check_timetable(timetable, planning):
    """
    Checks if all rides in Timetable are covered by the BusPlanning

    Input:
        Timetable and Bus Planning as Pandas DataFrames

    Output:
        Success or error statement dependent on ride coverage
    """

    # Makes sure all time columns are parsed
    ensure_time_column(timetable, 'departure_time')
    ensure_time_column(planning, 'start_time')
    ensure_time_column(planning, 'end_time')

    # Find rides not covered in planning
    unassigned_rides = []

    for _, ride in timetable.iterrows():
        matching = planning[
            (planning['bus'] == ride['line']) &
            (planning['start_location'] == ride['start']) &
            (planning['end_location'] == ride['end'])
        ]

        # Checks for a match in departure time
        is_covered = any(matching['start_time'] == ride['departure_time'])

        if not is_covered:
            unassigned_rides.append(ride)

    # Show result in Streamlit
    num_uncovered = len(unassigned_rides)

    if num_uncovered > 0:
        st.error(f"⚠️ There {'is' if num_uncovered == 1 else 'are'} {num_uncovered} ride{'s' if num_uncovered > 1 else ''} not being driven.")
        with st.expander("Click for more information on these rides"):
            st.write("The following rides are unassigned with the given Bus Planning:")
            st.write(pd.DataFrame(unassigned_rides))
    else:
        st.success("All rides are covered!")

def check_ride_duration(planning, distancematrix):
    """
    Checks if rides in the planning are within the allowed time alloted

    Input:
        Bus Planning and Distance Matrix as Pandas DataFrames

    Output:
        Success or error statement dependent on the rides being within the alloted time
    """

    # Find rides that take too long
    wrong_rides = []

    ensure_time_column(planning, 'start_time')
    ensure_time_column(planning, 'end_time')

    for _, ride in planning.iterrows():
        start = ride["start_time"]
        end = ride["end_time"]
        start_loc = ride["start_location"]
        end_loc = ride["end_location"]

        if start is None or end is None:
            continue

        actual_duration = (datetime.combine(datetime.today(), end) - datetime.combine(datetime.today(), start)).total_seconds() / 60

        match = distancematrix[(distancematrix['start'] == start_loc) & (distancematrix['end'] == end_loc)]

        if match.empty:
            continue

        max_time = match['max_travel_time'].values[0]
        min_time = match['min_travel_time'].values[0]

        if actual_duration > max_time or actual_duration < min_time:
            wrong_rides.append({
                'start_location': start_loc,
                'end_location': end_loc,
                'start_time': start,
                'end_time': end,
                'actual_duration': actual_duration,
                'min_travel_time': min_time,
                'max_travel_time': max_time
            })

    wrong =  pd.DataFrame(wrong_rides)

    if not wrong.empty:
        st.error(f"There are {len(wrong)} rides outside the allowed travel time")
        with st.expander("Click for more details"):
            st.write(wrong)
    else:
        st.success("All rides are within the allowed timeframe!")





def main(timetable, planning, distancematrix):
    """
    Does all necessary checks in Streamlit

    Input:
        Timetable, Bus Planning, and Distance matrix as Pandas DataFrames

    Output:
        Shows all checks in Streamlit
    """

    st.title("Bus Planning Control Center")

    st.header("Cleanup & Format Check")
    planning_clean = cleanup_excel(planning)
    check_format_excel(planning_clean)

    st.header("Length of Activities")
    planning_with_length = length_activities(planning_clean)

    st.header("Charging Check")
    charging_check(planning_clean)
    charge_time(planning_clean)

    st.header("Check Dienstregeling Coverage")
    check_timetable(timetable, planning_clean)

    st.header("Check Ride Duration vs Distance Matrix")
    check_ride_duration(planning_clean, distancematrix)
    
    st.header("Gantt Chart of Bus Planning")
    create_gannt_chart(planning_with_length)