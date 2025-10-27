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
    - service trips per lijnnummer worden gekleurd ("service trip line 400", ...)
    - andere activiteiten houden vaste kleuren
    """

    planning = planning.copy()

    # 1. Basisdatum
    if base_day is None:
        base_day = datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)
    planning["planning_day"] = base_day

    # 2. Ritjes tussen 00:00–03:00 → volgende dag
    start_cutoff = datetime.strptime("00:00:00", "%H:%M:%S").time()
    end_cutoff   = datetime.strptime("03:00:00", "%H:%M:%S").time()
    mask_next_day = (
        (planning["start_time"] >= start_cutoff) &
        (planning["start_time"] <= end_cutoff)
    )
    planning.loc[mask_next_day, "planning_day"] += pd.Timedelta(days=1)

    # 3. Combineer tijd met datum, fix nachtovergang
    planning["start_dt"] = planning.apply(
        lambda r: datetime.combine(r["planning_day"].date(), r["start_time"]),
        axis=1
    )
    planning["end_dt"] = planning.apply(
        lambda r: datetime.combine(r["planning_day"].date(), r["end_time"]),
        axis=1
    )
    planning["end_dt"] = planning["end_dt"].where(
        planning["end_dt"] >= planning["start_dt"],
        planning["end_dt"] + timedelta(days=1)
    )

    # duur in minuten
    planning["duration_min"] = (
        planning["end_dt"] - planning["start_dt"]
    ).dt.total_seconds() / 60.0

    # 4. Maak display_group (de legenda label + kleur key)
    def _pick_display_group(row):
        if row["activity"] == "service trip":
            # haal lijnnummer veilig op
            if "line" in row and pd.notna(row["line"]) and str(row["line"]).strip() != "":
                try:
                    # forceer geen .0
                    line_str = str(int(float(row["line"])))
                except Exception:
                    line_str = str(row["line"])
                return f"service trip line {line_str}"
            else:
                return "service trip (unknown line)"
        else:
            # charging / idle / material trip etc
            return row["activity"]

    planning["display_group"] = planning.apply(_pick_display_group, axis=1)

    

    # 5. Stel kleurmap samen
    base_colors = {
        "charging": "green",
        "idle": "gray",
        "material trip": "orange",
        "service trip (unknown line)": "blue",
    }

    palette_cycle = ["blue", "purple", "red", "brown", "pink", "cyan", "olive", "magenta"]
    color_map = dict(base_colors)

    # kleur per lijnlabel (service trip line xxx)
    line_groups = [
        g for g in planning["display_group"].unique()
        if isinstance(g, str) and g.startswith("service trip line ")
    ]
    for idx, g in enumerate(line_groups):
        color_map[g] = palette_cycle[idx % len(palette_cycle)]
    
    # 6. Plot
    fig = px.timeline(
        planning,
        x_start="start_dt",
        x_end="end_dt",
        y="bus",
        color="display_group",          # <-- we coloren nu echt op display_group
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
        ],
    )

    # assen en layout
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(tickformat="%H:%M")
    fig.update_layout(
        title="Bus Planning Gantt Chart",
        xaxis_title="Time of Day",
        yaxis_title="Bus",
        legend_title="Service trips / other activities",
        height=600,
        width=2000,
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
    Checks when the SOC drops below the allowed minimum.
    Reports:
    1. The first violating activity per bus (summary view)
    2. All violating activities (detailed view)

    - Violation = bus starts below limit OR crosses below during activity.
    - Rides between 00:00–03:00 count as 'next day', so they appear last per bus.
    """

    capacity = 300  # nominale batterijcapaciteit in kWh

    df = planning.copy()

    # --- Zorg dat tijdkolommen goed zijn ---
    df["start_time"] = pd.to_datetime(df["start_time"], format="%H:%M:%S").dt.time
    df["end_time"]   = pd.to_datetime(df["end_time"],   format="%H:%M:%S").dt.time

    # Basisdag en helper voor nachtverschuiving
    base_day = datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)

    def to_shifted_dt(t: time):
        dt = datetime.combine(base_day.date(), t)
        if t < time(3, 0):  # alles vóór 03:00 telt als 'volgende dag'
            dt += timedelta(days=1)
        return dt

    # Maak kolom voor correcte volgorde (nachtverschuiving)
    df["start_dt_shifted"] = df["start_time"].apply(to_shifted_dt)
    df = df.sort_values(["bus", "start_dt_shifted"]).reset_index(drop=True)

    # Voor debug en berekeningen
    df["SOC_start (kWh)"] = np.nan
    df["SOC_start (%)"] = np.nan
    df["SOC_after (kWh)"] = np.nan
    df["SOC_after (%)"] = np.nan
    df["min_battery (kWh)"] = np.nan
    df["min_battery (%)"] = np.nan
    df["violation"] = False

    # Resultaten
    first_violation_rows = []
    all_violations_rows = []

    # ---- Hoofdlus per bus ----
    for bus, group in df.groupby("bus"):
        idxs = group.index

        # Capaciteit & grenzen
        max_battery_kwh = (SOH / 100.0) * capacity
        start_soc_kwh = (startbat / 100.0) * max_battery_kwh
        min_soc_kwh = (minbat / 100.0) * max_battery_kwh
        min_soc_pct = (min_soc_kwh / max_battery_kwh) * 100.0

        usage = group["energy_consumption"].to_numpy()

        # SOC aan begin & eind
        soc_start_kwh = np.empty(len(usage))
        soc_start_kwh[0] = start_soc_kwh
        soc_start_kwh[1:] = start_soc_kwh - np.cumsum(usage[:-1])
        soc_after_kwh = soc_start_kwh - usage

        soc_start_pct = (soc_start_kwh / max_battery_kwh) * 100.0
        soc_after_pct = (soc_after_kwh / max_battery_kwh) * 100.0

        # Schrijf terug in df
        df.loc[idxs, "SOC_start (kWh)"] = soc_start_kwh
        df.loc[idxs, "SOC_start (%)"] = soc_start_pct
        df.loc[idxs, "SOC_after (kWh)"] = soc_after_kwh
        df.loc[idxs, "SOC_after (%)"] = soc_after_pct
        df.loc[idxs, "min_battery (kWh)"] = min_soc_kwh
        df.loc[idxs, "min_battery (%)"] = min_soc_pct

        # Check violations
        group_eval = df.loc[idxs].copy()
        group_eval["violation"] = (
            (group_eval["SOC_start (%)"] < group_eval["min_battery (%)"]) |
            (
                (group_eval["SOC_start (%)"] >= group_eval["min_battery (%)"]) &
                (group_eval["SOC_after (%)"] < group_eval["min_battery (%)"])
            )
        )

        df.loc[idxs, "violation"] = group_eval["violation"].to_numpy()
        bad_rows_all = group_eval[group_eval["violation"]]

        # ---- Alle overtredingen per bus ----
        for _, row in bad_rows_all.iterrows():
            start_t = row["start_time"]
            all_violations_rows.append({
                "bus": row["bus"],
                "time_when_SOC_too_low": start_t,
                "activity": row.get("activity", None),
                "start_location": row.get("start_location", None),
                "end_location": row.get("end_location", None),
                "SOC_at_start (kWh)": round(row["SOC_start (kWh)"], 2),
                "SOC_at_start (%)": round(row["SOC_start (%)"], 1),
                "SOC_after (kWh)": round(row["SOC_after (kWh)"], 2),
                "SOC_after (%)": round(row["SOC_after (%)"], 1),
                "minimum_allowed (kWh)": round(row["min_battery (kWh)"], 2),
                "minimum_allowed (%)": round(row["min_battery (%)"], 1),
                "_sort_key_dt": to_shifted_dt(start_t),  # nachtcorrectie
            })

        # ---- Eerste overtreding per bus ----
        if not bad_rows_all.empty:
            first_row = bad_rows_all.iloc[0]
            first_violation_rows.append({
                "bus": first_row["bus"],
                "time_when_SOC_too_low": first_row["start_time"],
                "activity": first_row.get("activity", None),
                "start_location": first_row.get("start_location", None),
                "end_location": first_row.get("end_location", None),
                "SOC_at_start (kWh)": round(first_row["SOC_start (kWh)"], 2),
                "SOC_at_start (%)": round(first_row["SOC_start (%)"], 1),
                "SOC_after (kWh)": round(first_row["SOC_after (kWh)"], 2),
                "SOC_after (%)": round(first_row["SOC_after (%)"], 1),
                "minimum_allowed (kWh)": round(first_row["min_battery (kWh)"], 2),
                "minimum_allowed (%)": round(first_row["min_battery (%)"], 1),
            })

    # ---- Streamlit output ----
    buses_in_trouble = {row["bus"] for row in first_violation_rows}
    num_buses_in_trouble = len(buses_in_trouble)

    if num_buses_in_trouble == 0:
        st.success("✅ All buses stay above the minimal SOC limit")
    else:
        st.error(f"⚠️ {num_buses_in_trouble} bus(es) drop below the minimal SOC limit")

    # 1️⃣ Eerste overtreding per bus
    if first_violation_rows:
        st.subheader("First violating activity per bus")
        summary_df = pd.DataFrame(first_violation_rows)
        st.dataframe(summary_df, use_container_width=True)

    # 2️⃣ Alle overtredingen, gesorteerd met nacht-correctie
    if all_violations_rows:
        with st.expander("All violating moments (bus can fail multiple times)"):
            all_df = pd.DataFrame(all_violations_rows)
            all_df = all_df.sort_values(["bus", "_sort_key_dt"]).reset_index(drop=True)
            all_df_display = all_df.drop(columns=["_sort_key_dt"])
            st.dataframe(all_df_display, use_container_width=True)

    # 3️⃣ Debug: volledige SOC-tijdlijn
    with st.expander("Full SOC timeline per bus (debug)"):
        debug_cols = [
            "bus", "start_time", "end_time", "activity",
            "SOC_start (kWh)", "SOC_start (%)",
            "SOC_after (kWh)", "SOC_after (%)",
            "min_battery (kWh)", "min_battery (%)",
            "violation", "start_location", "end_location",
            "energy_consumption"
        ]
        debug_view = df[debug_cols].copy()
        for col in ["SOC_start (kWh)", "SOC_after (kWh)", "min_battery (kWh)"]:
            debug_view[col] = debug_view[col].round(2)
        for col in ["SOC_start (%)", "SOC_after (%)", "min_battery (%)"]:
            debug_view[col] = debug_view[col].round(1)
        st.dataframe(debug_view, use_container_width=True)




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