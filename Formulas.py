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

def cleanup_timetable(timetable: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans up timetable formatting
    
    Input:
        Timetable as a Pandas DataFrame
        
    Output:
        Cleaned timetable as a Pandas Dataframe
    """
    
    timetable = timetable.copy()
    timetable.columns = [str(c).strip().replace("\u00A0","").replace(" ","_").lower() for c in timetable.columns]
    return timetable

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
        
def length_activities(planning:pd.DataFrame, start_col = "start_time", end_col = "end_time") -> pd.DataFrame:
    """
    Input:
        Bus planning as a Pandas DataFrame

    Returns:
        Bus planning as a Pandas DataFrame with an extra row with duration of activities
    """
    # Converts columns to datetime with dummy date
    start_dt = pd.to_datetime(planning[start_col], format="%H:%M:%S")
    end_dt = pd.to_datetime(planning[end_col], format="%H:%M:%S")

    # Night-transition detection: if end < start → Add 1 day
    end_dt = end_dt.where(end_dt >= start_dt, end_dt + pd.Timedelta(days=1))

    # Calculate diff and duration_min
    planning["diff"] = end_dt - start_dt
    planning["duration_min"] = planning["diff"].dt.total_seconds() / 60

    # Remove dummy date
    planning[start_col] = start_dt.dt.time
    planning[end_col] = end_dt.dt.time
    
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
        raise ValueError("Null time value found")
    
    # Tries standard formats in order
    for fmt in ("%H:%M", "%H:%M:%S"):
        try:
            return pd.to_datetime(value, format=fmt).time()
        except (ValueError, TypeError):
            continue
    
    raise ValueError(f"Time format not recognized: {value}")

def fill_idle_periods(planning: pd.DataFrame, base_day: datetime = None):
    """
    Puts all non-mentioned idle times into an activity
    
    Input:
        Bus Planning as a Pandas Dataframe
        
    Output:
        Updated Bus Planning with added idle activities
    """
    
    planning = planning.copy()
    
    if base_day is None:
        base_day = datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)
    
    # Convert string to datetime.time
    planning["start_time"] = pd.to_datetime(planning["start_time"], format="%H:%M:%S").dt.time
    planning["end_time"]   = pd.to_datetime(planning["end_time"], format="%H:%M:%S").dt.time

    # Combine with dummy date
    planning["start_dt"] = planning["start_time"].apply(lambda t: datetime.combine(base_day.date(), t))
    planning["end_dt"]   = planning["end_time"].apply(lambda t: datetime.combine(base_day.date(), t))
    
    # Night-transition: if end < start, add 1 day
    planning.loc[planning["end_dt"] < planning["start_dt"], "end_dt"] += timedelta(days=1)
    
    # Move rides between 00:00 and 03:00 to next day
    mask_after_midnight = planning["start_dt"].dt.hour < 3
    planning.loc[mask_after_midnight, "start_dt"] += timedelta(days=1)
    planning.loc[mask_after_midnight, "end_dt"]   += timedelta(days=1)
    
    idle_rows = []

    # Loop per bus
    for bus, bus_df in planning.groupby("bus"):
        bus_df = bus_df.sort_values("start_dt").reset_index(drop=True)

        for i in range(len(bus_df) - 1):
            current_end = bus_df.loc[i, "end_dt"]
            next_start = bus_df.loc[i + 1, "start_dt"]

            # If there's a gap → add idle to it
            if next_start > current_end:
                idle_row = bus_df.loc[i].copy()
                idle_row["activity"] = "idle"
                idle_row["start_dt"] = current_end
                idle_row["end_dt"] = next_start
                idle_row["start_time"] = current_end.time()
                idle_row["end_time"] = next_start.time()
                
                idle_row["line"] = ""
                idle_row["start_location"] = bus_df.loc[i, "end_location"]
                idle_row["end_location"] = bus_df.loc[i, "end_location"]
                
                idle_rows.append(idle_row)
                
    # Add idle rows
    if idle_rows:
        planning = pd.concat([planning, pd.DataFrame(idle_rows)], ignore_index=True)
        planning = planning.sort_values(["bus", "start_dt"]).reset_index(drop=True)
    
    # Temporary columns can be kept or removed
    planning = planning.drop(columns=["start_dt", "end_dt"])  # optional
    
    return planning
def create_gannt_chart(planning: pd.DataFrame, base_day: datetime = None):
    """
    Maakt een Gantt chart waarbij:
    - service trip blokken een kleur krijgen per lijn (line 400, line 401, ...)
    - charging / idle / material trip houden vaste kleuren
    """

    planning = planning.copy()

    # 1. base_day bepalen
    if base_day is None:
        base_day = datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)

    planning["planning_day"] = base_day

    # 2. ritten tussen 00:00 en 03:00 horen bij 'volgende dag'
    start_cutoff = datetime.strptime("00:00:00", "%H:%M:%S").time()
    end_cutoff   = datetime.strptime("03:00:00", "%H:%M:%S").time()

    mask_next_day = (
        (planning["start_time"] >= start_cutoff) &
        (planning["start_time"] <= end_cutoff)
    )
    planning.loc[mask_next_day, "planning_day"] += pd.Timedelta(days=1)

    # 3. combineer datum + tijd
    planning["start_dt"] = planning.apply(
        lambda row: datetime.combine(row["planning_day"].date(), row["start_time"]),
        axis=1
    )
    planning["end_dt"] = planning.apply(
        lambda row: datetime.combine(row["planning_day"].date(), row["end_time"]),
        axis=1
    )

    # fix nachtovergang
    planning["end_dt"] = planning["end_dt"].where(
        planning["end_dt"] >= planning["start_dt"],
        planning["end_dt"] + timedelta(days=1)
    )

    # 4. duur in minuten
    planning["duration_min"] = (
        planning["end_dt"] - planning["start_dt"]
    ).dt.total_seconds() / 60.0

    # 5. display_group bepalen (dit gaat de kleur bepalen)
    #    - service trip → kleur per lijn (bv. "line 400")
    #    - anders gewoon activity ("charging", "idle", ...)
    def _pick_display_group(row):
        if row["activity"] == "service trip":
            if "line" in row and pd.notna(row["line"]) and str(row["line"]).strip() != "":
                return f"line {row['line']}"
            else:
                return "service trip (unknown line)"
        else:
            return row["activity"]

    planning["display_group"] = planning.apply(_pick_display_group, axis=1)

    # 6. colormap maken
    # vaste kleuren voor niet-service trip dingen
    base_colors = {
        "charging": "green",
        "idle": "gray",
        "material trip": "orange",
        "service trip (unknown line)": "blue",  # fallback
    }

    # kleurenlijstje om langs te cyclen voor lijnen
    palette_cycle = [
        "magenta", "blue", "red", "brown", "pink", "cyan", "olive", "purple"
    ]

    color_map = dict(base_colors)

    line_groups = [
        g for g in planning["display_group"].unique()
        if isinstance(g, str) and g.startswith("line ")
    ]

    for idx, g in enumerate(line_groups):
        color_map[g] = palette_cycle[idx % len(palette_cycle)]

    # 7. plotly timeline
    fig = px.timeline(
        planning,
        x_start="start_dt",
        x_end="end_dt",
        y="bus",
        color="display_group",            # <-- niet meer 'activity'
        color_discrete_map=color_map,
        hover_data=[
            "activity",
            "line",
            "start_time",
            "end_time",
            "duration_min",
            "start_location",
            "end_location",
            "energy_consumption",
        ]
    )

    # y-as omdraaien voor Gantt-stijl
    fig.update_yaxes(autorange="reversed")

    # x-as op HH:MM
    fig.update_xaxes(tickformat="%H:%M")

    # layout
    fig.update_layout(
        title="Bus Planning Gantt Chart",
        xaxis_title="Time of Day",
        yaxis_title="Bus",
        legend_title="Line / Activity",
        height=600,
        width=2000
    )

    return st.plotly_chart(fig)

def ensure_time_column(df: pd.DataFrame, column):
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

    required_cols = ["start", "end", "min_travel_time", "max_travel_time", "line"]
    missing = [c for c in required_cols if c not in distancematrix.columns]
    if missing:
        st.error(f"Distance matrix mist kolommen: {', '.join(missing)}")
        st.write("Looking for:", required_cols)
        st.write("Actually have:", list(distancematrix.columns))
        return
    
    # Put min/max travel times to a float
    distancematrix["min_travel_time"] = pd.to_numeric(distancematrix["min_travel_time"], errors='coerce')
    distancematrix["max_travel_time"] = pd.to_numeric(distancematrix["max_travel_time"], errors='coerce')

    # Calculate duration in minutes if not present
    if "duration_min" not in planning.columns:
        planning["duration_min"] = planning["diff"].dt.total_seconds() / 60
    
    # Find rides that take too long
    wrong_rides = []

    for _, ride in planning.iterrows():
        if ride.get("activity") == "idle":
            continue
        
        start_loc = ride["start_location"]
        end_loc = ride["end_location"]
        busline = ride["line"]
        actual_duration = ride["duration_min"]
        bus = ride["bus"]

        # Look for matching travel time
        match = distancematrix[
            (distancematrix["start"].astype(str) == str(start_loc)) &
            (distancematrix["end"].astype(str) == str(end_loc)) &
            (distancematrix["line"].astype(str) == str(busline))
        ]

        if match.empty:
            # No distance info available -> skip
            continue

        max_time = match["max_travel_time"].values[0]
        min_time = match["min_travel_time"].values[0]

        # Skip rides if travel time info is missing
        if pd.isna(max_time) or pd.isna(min_time):
            continue

        # Check if actual_duration falls outside allowed interval
        if actual_duration < min_time or actual_duration > max_time:
            wrong_rides.append({
                "start_location": start_loc,
                "end_location": end_loc,
                "start_time": ride["start_time"],
                "end_time": ride["end_time"],
                "actual_duration_min": actual_duration,
                "min_travel_time": min_time,
                "max_travel_time": max_time,
                "line": busline,
                "bus": bus
            })

    wrong_df = pd.DataFrame(wrong_rides)

    if not wrong_df.empty:
        st.error(f"There are {len(wrong_df)} rides outside the allowed travel time")
        with st.expander("Click for more details"):
            st.write(wrong_df)
    else:
        st.success("All rides are within the allowed timeframe!")

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

        st.error(f"There are {len(output)} periods where a bus gets under the minimal SOC")
        with st.expander('Click for more information on these rides'):
            st.write(output.set_index(output.columns[0]))

    else:
        st.success('All buses stay above the minimal SOC')

def calculate_energy_consumption(planning: pd.DataFrame, distancematrix: pd.DataFrame, driving_usage, idle_usage, charging_speed):
    """
    Creates a column with the newly calculated energy usage
    
    Input:
        Bus Planning as a Pandas DataFrame
        Distances in a dictionary
        Driving usage in kW/km
        Idle usage is a constant
        Charging speed in kW/h
        
    Output:
        Bus Planning with an added column with the newly calculated energy consumption
    """
    
    planning = planning.copy()
    distance = distancematrix.copy()
    
    distance.rename(columns = {"start": "start_location", "end": "end_location"}, inplace = True)
    
    for col in ["start_location", "end_location", "line"]:
        if col in planning.columns:
            planning[col] = planning[col].astype(str)
        if col in distance.columns:
            distance[col] = distance[col].astype(str)
    
    df = planning.merge(
        distance[["start_location", "end_location", "line", "distance_m"]],
        on = ["start_location", "end_location", "line"],
        how = "left"
    )
    
    if "diff_min" not in df.columns:
        df["diff_min"] = df["diff"].dt.total_seconds() / 60
        
    df["energy_consumption_new"] = np.select(
        condlist = [
            df["activity"].str.contains("idle", case = False, na = False),
            df["activity"].str.contains("charging", case = False, na = False)
        ],
        choicelist = [
            idle_usage,
            (df["diff_min"] * charging_speed * -1) / 60
        ],
        default = (df["distance_m"] / 1000) * driving_usage
    )
    
    return df


driving_usage = 1.2
idle_usage = 5
charging_speed = 450
SOH = 90
minbat = 10
startbat = 100


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
    
    st.header("Fill idle periods")
    planning_filled = fill_idle_periods(planning_clean)

    st.header("Length of Activities")
    planning_with_length = length_activities(planning_filled)

    st.header("Charging Check")
    charging_check(planning_with_length)
    charge_time(planning_with_length)

    st.header("Check Dienstregeling Coverage")
    check_timetable(timetable, planning_with_length)

    st.header("Check Ride Duration vs Distance Matrix")
    check_ride_duration(planning_with_length, distancematrix)
    
    st.header("Calculated energy consumption")
    planning_calculated = calculate_energy_consumption(planning_with_length, distancematrix, driving_usage, idle_usage, charging_speed)
    
    st.header("Check SOC")
    SOC_check(planning_calculated, SOH, minbat, startbat)
    
    st.header("Gantt Chart of Bus Planning")
    create_gannt_chart(planning_calculated)