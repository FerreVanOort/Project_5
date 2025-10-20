# Test bestand formules

## Formulas

# Imports
import pandas as pd
import numpy as np
import streamlit as st
import scipy.stats as sp
import datetime
from datetime import time
from datetime import datetime
from datetime import timedelta
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
        print("The uploaded file has the right format")
    if not true_amount:
        print("The uploaded file does not have the right format")
        
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
        print("Charging rates are correct")
    else:
        print("Charging rates have positive value")
        print("Error lines:")
        print(errors)
        
def length_activities(planning:pd.DataFrame, start_col="start_time", end_col="end_time") -> pd.DataFrame:
    """
    Input:
        Bus planning as a Pandas Dataframe

    Returns:
        Bus planning as a Pandas DataFrame with an extra row with duration of activities
    """
    # Converteer kolommen naar datetime met dummy datum
    start_dt = pd.to_datetime(planning[start_col], format="%H:%M:%S")
    end_dt = pd.to_datetime(planning[end_col], format="%H:%M:%S")

    # Nacht-overgang detectie: als end < start → 1 dag optellen
    end_dt = end_dt.where(end_dt >= start_dt, end_dt + pd.Timedelta(days=1))

    # Bereken diff en duration_min
    planning["diff"] = end_dt - start_dt
    planning["duration_min"] = planning["diff"].dt.total_seconds() / 60

    # Zet kolommen terug naar alleen tijd (verwijder dummy datum)
    planning[start_col] = start_dt.dt.time
    planning[end_col] = end_dt.dt.time

    print("✅ All time values are valid — 'diff' calculated with night-overrides!")
    return planning


def charge_time(planning:pd.DataFrame):
    """
    Input:
        Bus planning as a Pandas DataFrame

    Output:
        Success or error message depending on sufficient charging
    """
    # Gathers all charging moments in planning
    charging_moments = planning[planning.iloc[:,4].str.contains("charging")]
    
    # Checks if charge time is longer than given minimum
    short_charge = charging_moments[charging_moments['diff'] < pd.Timedelta(minutes = 15)]
    
    # Success or error dependent on if there are any errors
    if len(short_charge) > 0:
        print(f"There are {len(short_charge)} times a bus is charged too short")
        with print("More information on charging times"):
            print("Insufficient charge time")
            short_charge = short_charge[["start_time", "end_time", "activity"]]
            print(pd.DataFrame(short_charge))
    else:
        print("All buses have sufficient charging times")
        
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
    
def create_bus_gantt_chart(planning: pd.DataFrame):
    """
    Create a Gantt chart for bus planning.
    
    X-axis: Time of day
    Y-axis: Buses
    Color: Activity
    
    Handles night transitions automatically.
    
    Input:
        planning: DataFrame with at least columns:
            - 'bus' : bus identifier
            - 'start_time' : datetime.time
            - 'end_time' : datetime.time
            - 'activity' : activity type
            
    Returns:
        Plotly figure (also shows the figure in Python/Jupyter)
    """
    planning = planning.copy()
    
    # Combine time columns with dummy date
    planning["start_dt"] = planning["start_time"].apply(lambda t: datetime.combine(datetime(1900,1,1), t))
    planning["end_dt"] = planning["end_time"].apply(lambda t: datetime.combine(datetime(1900,1,1), t))
    
    # Handle night transitions (end < start)
    planning["end_dt"] = planning["end_dt"].where(
        planning["end_dt"] >= planning["start_dt"],
        planning["end_dt"] + timedelta(days=1)
    )
    
    # Create Gantt chart
    color_map = {
        "service trip": "blue",
        "material trip": "orange",
        "idle": "gray",
        "charging": "green"
    }
    
    fig = px.timeline(
        planning,
        x_start="start_dt",
        x_end="end_dt",
        y="bus",
        color="activity",
        color_discrete_map = color_map,
        hover_data=["start_time", "end_time", "activity"]
    )
    
    # Y-axis: reverse to match typical Gantt chart style
    fig.update_yaxes(autorange="reversed")
    
    # X-axis: show only hours and minutes
    fig.update_xaxes(tickformat="%H:%M")
    
    # Layout tweaks
    fig.update_layout(
        title="Bus Planning Gantt Chart",
        xaxis_title="Time of Day",
        yaxis_title="Bus",
        legend_title="Activity",
        height=600,
        width=2000
    )
    
    fig.show()
    return fig

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
            (planning['line'] == ride['line']) &
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
        print(f"⚠️ There {'is' if num_uncovered == 1 else 'are'} {num_uncovered} ride{'s' if num_uncovered > 1 else ''} not being driven.")
        with print("Click for more information on these rides"):
            print("The following rides are unassigned with the given Bus Planning:")
            print(pd.DataFrame(unassigned_rides))
    else:
        print("All rides are covered!")

def check_ride_duration(planning, distancematrix):
    """
    Checks if rides in the planning are within the allowed time alloted

    Input:
        Bus Planning and Distance Matrix as Pandas DataFrames

    Output:
        Success or error statement dependent on the rides being within the alloted time
    """

    # Zet min/max travel times om naar float
    distancematrix["min_travel_time"] = pd.to_numeric(distancematrix["min_travel_time"], errors='coerce')
    distancematrix["max_travel_time"] = pd.to_numeric(distancematrix["max_travel_time"], errors='coerce')

    # Bereken duration in minuten als nog niet aanwezig
    if "duration_min" not in planning.columns:
        planning["duration_min"] = planning["diff"].dt.total_seconds() / 60

    wrong_rides = []

    for _, ride in planning.iterrows():
        start_loc = ride["start_location"]
        end_loc = ride["end_location"]
        busline = ride["line"]
        actual_duration = ride["duration_min"]

        # Zoek bijbehorende reistijd
        match = distancematrix[
            (distancematrix["start"] == start_loc) &
            (distancematrix["end"] == end_loc) &
            (distancematrix["line"] == busline)
        ]

        if match.empty:
            # Geen afstandsinfo beschikbaar, overslaan
            continue

        max_time = match["max_travel_time"].values[0]
        min_time = match["min_travel_time"].values[0]

        # Sla ritten over als travel time info ontbreekt
        if pd.isna(max_time) or pd.isna(min_time):
            continue

        # Controleer of actual_duration buiten toegestaan interval valt
        if actual_duration < min_time or actual_duration > max_time:
            wrong_rides.append({
                "start_location": start_loc,
                "end_location": end_loc,
                "start_time": ride["start_time"],
                "end_time": ride["end_time"],
                "actual_duration_min": actual_duration,
                "min_travel_time": min_time,
                "max_travel_time": max_time
            })

    wrong_df = pd.DataFrame(wrong_rides)

    if not wrong_df.empty:
        print(f"⚠️  There are {len(wrong_df)} rides outside the allowed travel time!")
        print("Details of these rides:")
        print(wrong_df)
    else:
        print("✅ All rides are within the allowed timeframe!")

def SOC_periods(planning: pd.DataFrame) -> pd.DataFrame:
    """
    Find periods in which a bus is below the minimum SOC

    Input:
        Bus Planning as a Pandas Dataframe

    Output:
        Bus Planning with period/interval column added as a Pandas DataFrame
    """

    dataf = planning.copy()
    dataf = dataf.sort_values(['bus', 'start_time']).reset_index(drop = True)

    shifted_end = dataf['end_time'].shift(1)
    new_period = (dataf['start_time'] != shifted_end) | (dataf['bus'] != dataf['bus'].shift(1))
    dataf['period_id'] = new_period.cumsum()

    result = dataf.groupby('period_id').agg({
        'bus' : 'first',
        'start_time' : 'first',
        'end_time' : 'last'
    }).reset_index(drop = True)
    return result

def SOC_check(planning: pd.DataFrame, SOH, minbat, startbat):
    """
    Checks if the SOC gets below minimal value

    Input:
        Bus Planning as a Pandas DataFrame,
        the SOH of the battery in % (filled in as a float, e.g. 90 or 87.5),
        the minimal battery charge in % when the bus reached the charging station (filled in as a float, e.g. 10 or 12.5),
        the battery charge in % at the start (filled in as a float, e.g. 100 or 97.5)

    Output:
        Success or error statement dependent on if a bus gets below the minimal SOC
    """

    capacity = 300
    df = planning.copy()
    df['SOC (kW)'] = np.nan
    df['min_battery (kW)'] = np.nan

    df = df.sort_values(['bus', 'start_time']).reset_index(drop = True)

    for bus, group in df.groupby('bus'):
        idxs = group.index

        if not isinstance(SOH, (int, float)):
            raise ValueError("SOH moet een getal zijn (in procenten).")
        max_battery = float(SOH) / 100 * capacity

        battery_start = (startbat / 100) * max_battery
        min_battery = (minbat / 100) * max_battery

        usage = group['energy_consumption'].to_numpy()
        soc = np.empty(len(usage))
        soc[0] = battery_start
        soc[1:] = battery_start - np.cumsum(usage[:-1])

        df.loc[idxs, 'SOC (kW)'] = soc
        df.loc[idxs, 'min_battery (kW)'] = min_battery

    df['below_min_SOC'] = df['SOC (kW)'] < df['min_battery (kW)']
    soc_too_low = df.loc[df['below_min_SOC'], ['bus', 'start_time', 'end_time']]

    if not soc_too_low.empty:
        output = SOC_periods(soc_too_low)

        print(f"There are {len(output)} periods where a bus gets under the minimal SOC")
        print('Click for more information on these rides')
        print(output.set_index(output.columns[0]))

    else:
        print('All buses stay above the minimal SOC')


timetable = pd.read_excel('Timetable.xlsx')
distancematrix = pd.read_excel('DistanceMatrix.xlsx')
planning = pd.read_excel('BusPlanning.xlsx')


def main(timetable, planning, distancematrix):
    """
    Does all necessary checks in Streamlit

    Input:
        Timetable, Bus Planning, and Distance matrix as Pandas DataFrames

    Output:
        Shows all checks in Streamlit
    """

    print("Bus Planning Control Center")

    print("Cleanup & Format Check")
    planning_clean = cleanup_excel(planning)
    check_format_excel(planning_clean)

    print("Length of Activities")
    planning_with_length = length_activities(planning_clean)

    print("Charging Check")
    charging_check(planning_clean)
    charge_time(planning_clean)

    print("Check Dienstregeling Coverage")
    check_timetable(timetable, planning_clean)

    print("Check Ride Duration vs Distance Matrix")
    check_ride_duration(planning_clean, distancematrix)
    
    print("Gantt Chart of Bus Planning")
    create_bus_gantt_chart(planning_with_length)
    
main(timetable, planning, distancematrix)
