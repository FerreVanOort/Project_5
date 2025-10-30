# Formulas

# Imports
import pandas as pd
import numpy as np
import streamlit as st
from datetime import time, datetime, timedelta
import plotly.express as px


# -------------------------------------------------
# Cleaning & preprocessing
# -------------------------------------------------

def cleanup_excel(planning: pd.DataFrame) -> pd.DataFrame:
    """
    Replace spaces in column names and puts it all in lowercase.
    
    Input:
        Bus planning as a Pandas DataFrame
        
    Output:
        Cleaned bus planning as a Pandas DataFrame
    """
    planning = planning.copy()
    planning.columns = planning.columns.str.replace(" ", "_")
    planning.columns = planning.columns.str.lower()
    return planning


def cleanup_timetable(timetable: pd.DataFrame) -> pd.DataFrame:
    """
    Clean timetable column names: replace spaces, non-breaking spaces, and all lowercase.
    
    Input:
        Timetable as a Pandas DataFrame
        
    Output:
        Cleaned timetable as a Pandas DataFrame
    """
    timetable = timetable.copy()
    timetable.columns = [
        str(c).strip().replace("\u00A0", "").replace(" ", "_").lower()
        for c in timetable.columns
    ]
    return timetable


def check_format_excel(planning: pd.DataFrame):
    """
    Checks if the uploaded bus planning has the required format.
    
    Input:
        Bus planning as Pandas DataFrame
        
    Output:
        Success or error statement dependent on the format
    """
    items = list(planning.columns)
    required = [
        'start_location', 'end_location',
        'start_time', 'end_time',
        'activity', 'line',
        'energy_consumption', 'bus'
    ]

    if items == required:
        st.success("The uploaded file has the right format")
    else:
        st.error("The uploaded file does not have the right format")
        st.write("Expected:", required)
        st.write("Got:", items)


# -------------------------------------------------
# Validation / checks
# -------------------------------------------------

def charging_check(planning: pd.DataFrame):
    """
    Checks if charging activities have correct (negative) energy values.

    Input:
        Bus planning as a Pandas DataFrame

    Output:
        Success or error statement dependent on correct charging energy values
    """
    # Filter for mistake
    errors = planning[
        (planning['activity'].str.lower() == 'charging') &
        (planning['energy_consumption'] >= 0)
    ]

    if errors.empty:
        st.success("All charging rates have correct (negative) energy values.")
    else:
        st.error(f"Charging rates have positive values in {len(errors)} line(s).")
        with st.expander("Show affected lines"):
            st.write("The following rows have incorrect (positive) charging values:")
            st.dataframe(
                errors[['bus', 'start_time', 'end_time', 'activity', 'energy_consumption']],
                use_container_width=True
            )



def length_activities(planning: pd.DataFrame,
                      start_col: str = "start_time",
                      end_col: str = "end_time") -> pd.DataFrame:
    """
    Add 'diff' and 'duration_min' columns for each activity.
    Handles overnight trips (end < start).
    
    Input:
        Bus planning as a Pandas DataFrame
        
    Output:
        Bus planning with added columns as a Pandas DataFrame
    """
    planning = planning.copy()

    start_dt = pd.to_datetime(planning[start_col], format="%H:%M:%S")
    end_dt = pd.to_datetime(planning[end_col], format="%H:%M:%S")

    # If end is "earlier" than start, assume it ends next day
    end_dt = end_dt.where(end_dt >= start_dt, end_dt + pd.Timedelta(days=1))

    planning["diff"] = end_dt - start_dt
    planning["duration_min"] = planning["diff"].dt.total_seconds() / 60

    # Store back only the times
    planning[start_col] = start_dt.dt.time
    planning[end_col] = end_dt.dt.time

    return planning


def charge_time(planning: pd.DataFrame):
    """
    Checks if each charging period is at least 15 minutes.
    
    Input:
        Bus planning as a Pandas DataFrame
        
    Output:
        Success or error statement dependent on sufficient charging times
    """
    planning = planning.copy()

    # activity-kolom eerst naar string om .str.contains veilig te doen
    planning["activity"] = planning["activity"].astype(str)

    charging_moments = planning[planning["activity"].str.contains("charging", case=False, na=False)]

    short_charge = charging_moments[charging_moments['diff'] < pd.Timedelta(minutes=15)]

    if len(short_charge) > 0:
        st.error(f"There are {len(short_charge)} times a bus is charged too short")
        with st.expander("More information on charging times"):
            st.write("Insufficient charge time")
            short_show = short_charge[["start_time", "end_time", "activity"]]
            st.write(short_show.reset_index(drop=True))
    else:
        st.success("All buses have sufficient charging times")


def convert_to_time(value):
    """
    Guarantees that a time value is a datetime.time object.
    Accepts '08:30', '08:30:00', or already a datetime.time.
    
    Input:
        Any value
        
    Output:
        Error if time format is not recognized
    """
    if isinstance(value, time):
        return value

    if pd.isnull(value):
        raise ValueError("Null time value found")

    for fmt in ("%H:%M", "%H:%M:%S"):
        try:
            return pd.to_datetime(value, format=fmt).time()
        except (ValueError, TypeError):
            continue

    raise ValueError(f"Time format not recognized: {value}")


def ensure_time_column(df: pd.DataFrame, column: str):
    """
    Convert a whole column to datetime.time using convert_to_time.
    
    Input:
        Bus planning as a Pandas DataFrame
        
    Output:
        Bus planning with applied convert_to_time function
    """
    df[column] = df[column].apply(convert_to_time)


def fill_idle_periods(planning: pd.DataFrame, base_day: datetime = None) -> pd.DataFrame:
    """
    Insert 'idle' rows for gaps between activities for each bus.
    Also handles after-midnight logic.
    
    Input:
        Bus planning as a Pandas DataFrame
        *Optional* Date on which the bus planning is executed -> If None: Date of today
        
    Output:
        Bus planning with all gaps in bus planning filled by idle period
    """
    planning = planning.copy()

    if base_day is None:
        base_day = datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)

    # Make sure start_time / end_time are datetime.time
    planning["start_time"] = pd.to_datetime(planning["start_time"], format="%H:%M:%S").dt.time
    planning["end_time"] = pd.to_datetime(planning["end_time"], format="%H:%M:%S").dt.time

    # Attach dummy date
    planning["start_dt"] = planning["start_time"].apply(lambda t: datetime.combine(base_day.date(), t))
    planning["end_dt"] = planning["end_time"].apply(lambda t: datetime.combine(base_day.date(), t))

    # Overnight shift: end before start means next day
    planning.loc[planning["end_dt"] < planning["start_dt"], "end_dt"] += timedelta(days=1)

    # Move trips starting between 00:00 and 03:00 to 'next day'
    mask_after_midnight = planning["start_dt"].dt.hour < 3
    planning.loc[mask_after_midnight, "start_dt"] += timedelta(days=1)
    planning.loc[mask_after_midnight, "end_dt"] += timedelta(days=1)

    idle_rows = []

    # Per bus, fill gaps with idle
    for bus_name, bus_df in planning.groupby("bus"):
        bus_df = bus_df.sort_values("start_dt").reset_index(drop=True)

        for i in range(len(bus_df) - 1):
            current_end = bus_df.loc[i, "end_dt"]
            next_start = bus_df.loc[i + 1, "start_dt"]

            if next_start > current_end:
                idle_row = bus_df.loc[i].copy()
                idle_row["activity"] = "idle"
                idle_row["start_dt"] = current_end
                idle_row["end_dt"] = next_start
                idle_row["start_time"] = current_end.time()
                idle_row["end_time"] = next_start.time()

                # Idle doesn't belong to a specific line, so an empty string
                idle_row["line"] = ""
                idle_row["start_location"] = bus_df.loc[i, "end_location"]
                idle_row["end_location"] = bus_df.loc[i, "end_location"]

                idle_rows.append(idle_row)

    if idle_rows:
        planning = pd.concat([planning, pd.DataFrame(idle_rows)], ignore_index=True)
        planning = planning.sort_values(["bus", "start_dt"]).reset_index(drop=True)

    # No more need for start_dt / end_dt outside of function
    planning = planning.drop(columns=["start_dt", "end_dt"])

    return planning


# -------------------------------------------------
# Gantt chart
# -------------------------------------------------

def create_gannt_chart(planning: pd.DataFrame, base_day: datetime = None):
    """
    Builds a Gantt chart in Streamlit.
    
    Input:
        Bus planning as a Pandas DataFrame
        
    Output:
        Gannt chart displayed in Streamlit tool
    """

    planning = planning.copy()

    # Reference day
    if base_day is None:
        base_day = datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)
    planning["planning_day"] = base_day

    # Anything starting between 00:00–03:00 belongs to 'next calendar day'
    start_cutoff = datetime.strptime("00:00:00", "%H:%M:%S").time()
    end_cutoff = datetime.strptime("03:00:00", "%H:%M:%S").time()
    mask_next_day = (
        (planning["start_time"] >= start_cutoff) &
        (planning["start_time"] <= end_cutoff)
    )
    planning.loc[mask_next_day, "planning_day"] += pd.Timedelta(days=1)

    # Combine date + time
    planning["start_dt"] = planning.apply(
        lambda r: datetime.combine(r["planning_day"].date(), r["start_time"]),
        axis=1
    )
    planning["end_dt"] = planning.apply(
        lambda r: datetime.combine(r["planning_day"].date(), r["end_time"]),
        axis=1
    )

    # Handle overnight (end before start)
    planning["end_dt"] = planning["end_dt"].where(
        planning["end_dt"] >= planning["start_dt"],
        planning["end_dt"] + timedelta(days=1)
    )

    # Duration in minutes
    planning["duration_min"] = (
        planning["end_dt"] - planning["start_dt"]
    ).dt.total_seconds() / 60.0

    # Build legend label / color group
    def _pick_display_group(row):
        if row["activity"] == "service trip":
            if "line" in row and pd.notna(row["line"]) and str(row["line"]).strip() != "":
                try:
                    # 401.0 -> "401"
                    line_str = str(int(float(row["line"])))
                except Exception:
                    line_str = str(row["line"])
                return f"service trip line {line_str}"
            else:
                return "service trip (unknown line)"
        else:
            return row["activity"]

    planning["display_group"] = planning.apply(_pick_display_group, axis=1)

    # Fixed colors for non-line activities
    base_colors = {
        "charging": "green",
        "idle": "gray",
        "material trip": "orange",
        "service trip (unknown line)": "blue",
    }

    # Palette to assign unique colors per line
    palette_cycle = [
        "blue", "purple", "red", "brown",
        "pink", "cyan", "olive", "magenta"
    ]
    color_map = dict(base_colors)

    # Add dynamic colors for "service trip line X"
    line_groups = [
        g for g in planning["display_group"].unique()
        if isinstance(g, str) and g.startswith("service trip line ")
    ]
    for idx, g in enumerate(line_groups):
        color_map[g] = palette_cycle[idx % len(palette_cycle)]

    # Build timeline chart
    fig = px.timeline(
        planning,
        x_start="start_dt",
        x_end="end_dt",
        y="bus",
        color="display_group",
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

    # Flip Y-axis so first bus is on top
    fig.update_yaxes(autorange="reversed")

    # X-axis HH:MM formatting
    fig.update_xaxes(tickformat="%H:%M")

    fig.update_layout(
        title="Bus Planning Gantt Chart",
        xaxis_title="Time of Day",
        yaxis_title="Bus",
        legend_title="Activity / Line",
        height=600,
        width=2000,
    )

    return st.plotly_chart(fig)


# -------------------------------------------------
# Coverage / timing checks
# -------------------------------------------------

def check_timetable(timetable: pd.DataFrame, planning: pd.DataFrame):
    """
    Check if every timetable ride is covered in the planning.
    
    Input:
        Timetable as a Pandas DataFrame
        Bus planning as a Pandas DataFrame
        
    Output:
        Success or error statement dependent on ride coverage
    """
    timetable = timetable.copy()
    planning = planning.copy()

    # Make sure time columns are really time objects
    ensure_time_column(timetable, 'departure_time')
    ensure_time_column(planning, 'start_time')
    ensure_time_column(planning, 'end_time')

    unassigned_rides = []

    for _, ride in timetable.iterrows():
        matching = planning[
            (planning['line'] == ride['line']) &
            (planning['start_location'] == ride['start']) &
            (planning['end_location'] == ride['end'])
        ]

        # Covered if any start_time matches the timetable departure_time
        is_covered = any(matching['start_time'] == ride['departure_time'])

        if not is_covered:
            unassigned_rides.append(ride)

    num_uncovered = len(unassigned_rides)

    if num_uncovered > 0:
        st.error(
            f"⚠️ There {'is' if num_uncovered == 1 else 'are'} "
            f"{num_uncovered} ride{'s' if num_uncovered > 1 else ''} not being driven."
        )
        with st.expander("Click for more information on these rides"):
            st.write("The following rides are unassigned with the given Bus Planning:")
            st.write(pd.DataFrame(unassigned_rides))
    else:
        st.success("All rides are covered!")


def check_ride_duration(planning: pd.DataFrame, distancematrix: pd.DataFrame):
    """
    Check if the duration of each (non-idle) ride is within allowed min/max
    from the distance matrix.
    
    Input:
        Bus planning as a Pandas DataFrame
        Distance matrix as a Pandas DataFrame
        
    Output:
        Success or error statement dependent on correct distance matrix coverage
    """
    planning = planning.copy()
    distancematrix = distancematrix.copy()

    required_cols = ["start", "end", "min_travel_time", "max_travel_time", "line"]
    missing = [c for c in required_cols if c not in distancematrix.columns]
    if missing:
        st.error(f"Distance matrix mist kolommen: {', '.join(missing)}")
        st.write("Looking for:", required_cols)
        st.write("Actually have:", list(distancematrix.columns))
        return

    # Force numeric travel times
    distancematrix["min_travel_time"] = pd.to_numeric(distancematrix["min_travel_time"], errors='coerce')
    distancematrix["max_travel_time"] = pd.to_numeric(distancematrix["max_travel_time"], errors='coerce')

    # Make sure we have duration_min
    if "duration_min" not in planning.columns:
        planning["duration_min"] = planning["diff"].dt.total_seconds() / 60

    wrong_rides = []

    for _, ride in planning.iterrows():
        if ride.get("activity") == "idle":
            continue

        start_loc = ride["start_location"]
        end_loc = ride["end_location"]
        busline = ride["line"]
        actual_duration = ride["duration_min"]
        bus = ride["bus"]

        # Find allowed time window for this leg+line
        match = distancematrix[
            (distancematrix["start"].astype(str) == str(start_loc)) &
            (distancematrix["end"].astype(str) == str(end_loc)) &
            (distancematrix["line"].astype(str) == str(busline))
        ]

        if match.empty:
            # No info -> skip
            continue

        max_time = match["max_travel_time"].values[0]
        min_time = match["min_travel_time"].values[0]

        if pd.isna(max_time) or pd.isna(min_time):
            continue

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


# -------------------------------------------------
# SOC check
# -------------------------------------------------

def SOC_periods(planning: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse consecutive below-min periods per bus into intervals.
    
    Input:
        Bus planning as a Pandas DataFrame
        
    Output:
        SOC intervals
    """
    dataf = planning.copy()
    dataf = dataf.sort_values(['bus', 'start_time']).reset_index(drop=True)

    shifted_end = dataf['end_time'].shift(1)
    new_period = (
        (dataf['start_time'] != shifted_end) |
        (dataf['bus'] != dataf['bus'].shift(1))
    )
    dataf['period_id'] = new_period.cumsum()

    result = dataf.groupby('period_id').agg({
        'bus': 'first',
        'start_time': 'first',
        'end_time': 'last'
    }).reset_index(drop=True)

    return result


def SOC_check(planning: pd.DataFrame, SOH, minbat, startbat):
    """
    Checks SOC behaviour for each bus.
    
    Input:
        Bus planning as a Pandas DataFrame
        SOH - State of Health of battery in percentages
        minbat - Minimum amount of battery that needs to remain in battery when reaching charging station
        startbat - Starting battery at the beginning of the day
        
    Output:
        Success or error statement dependent on correct SOC levels in battery
            Accompanied by DataFrames with error moments if there are any
    """

    capacity = 300  # Nominal battery capacity in kWh

    df = planning.copy()

    # Ensure times are datetime.time
    df["start_time"] = pd.to_datetime(df["start_time"], format="%H:%M:%S").dt.time
    df["end_time"]   = pd.to_datetime(df["end_time"],   format="%H:%M:%S").dt.time

    # Helper for "shift after midnight"
    base_day = datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)

    def to_shifted_dt(t: time):
        dt = datetime.combine(base_day.date(), t)
        if t < time(3, 0):  # Trips before 03:00 count as 'next day'
            dt += timedelta(days=1)
        return dt

    # Sort within each bus using shifted time
    df["start_dt_shifted"] = df["start_time"].apply(to_shifted_dt)
    df = df.sort_values(["bus", "start_dt_shifted"]).reset_index(drop=True)

    # Prep columns
    df["SOC_start (kWh)"] = np.nan
    df["SOC_start (%)"] = np.nan
    df["SOC_after (kWh)"] = np.nan
    df["SOC_after (%)"] = np.nan
    df["min_battery (kWh)"] = np.nan
    df["min_battery (%)"] = np.nan
    df["violation"] = False

    first_violation_rows = []
    all_violations_rows = []

    # Loop per bus
    for bus, group in df.groupby("bus"):
        idxs = group.index

        # Usable capacity = SOH% of nominal
        max_battery_kwh = (SOH / 100.0) * capacity

        # Initial SOC in kWh at the very first activity of the (shifted) day
        start_soc_kwh = (startbat / 100.0) * max_battery_kwh

        # Allowed minimum in kWh / %
        min_soc_kwh = (minbat / 100.0) * max_battery_kwh
        min_soc_pct = (min_soc_kwh / max_battery_kwh) * 100.0

        usage = group["energy_consumption"].to_numpy()  # positive=consumption, negative=charging

        # SOC at start of each activity
        soc_start_kwh = np.empty(len(usage))
        soc_start_kwh[0] = start_soc_kwh
        soc_start_kwh[1:] = start_soc_kwh - np.cumsum(usage[:-1])

        # SOC at end of each activity
        soc_after_kwh = soc_start_kwh - usage

        # Convert to %
        soc_start_pct = (soc_start_kwh / max_battery_kwh) * 100.0
        soc_after_pct = (soc_after_kwh / max_battery_kwh) * 100.0

        # Now estimate "minimum SOC during this activity"
        # If usage > 0 (we consume energy), SOC goes down over the activity → min is after.
        # If usage < 0 (charging), SOC goes up → min is before.
        usage_arr = usage
        min_during_kwh = np.where(
            usage_arr >= 0,
            soc_after_kwh,
            soc_start_kwh
        )
        min_during_pct = (min_during_kwh / max_battery_kwh) * 100.0

        # Write columns back for debug
        df.loc[idxs, "SOC_start (kWh)"]   = soc_start_kwh
        df.loc[idxs, "SOC_start (%)"]     = soc_start_pct
        df.loc[idxs, "SOC_after (kWh)"]   = soc_after_kwh
        df.loc[idxs, "SOC_after (%)"]     = soc_after_pct
        df.loc[idxs, "min_battery (kWh)"] = min_soc_kwh
        df.loc[idxs, "min_battery (%)"]   = min_soc_pct
        df.loc[idxs, "SOC_min_during (%)"] = min_during_pct
        df.loc[idxs, "SOC_min_during (kWh)"] = min_during_kwh

        # Define violation logic:
        # 1. If it already starts illegal (< limit), that's a violation
        starts_illegal = soc_start_pct < min_soc_pct

        # 2. If it starts legal but dips under limit at any point in the activity
        dips_during = (soc_start_pct >= min_soc_pct) & (min_during_pct < min_soc_pct)

        violation_mask = starts_illegal | dips_during

        df.loc[idxs, "violation"] = violation_mask

        # Collect all violations for this bus
        bad_rows_all = df.loc[idxs][violation_mask]

        for _, row in bad_rows_all.iterrows():
            start_t = row["start_time"]
            all_violations_rows.append({
                "bus": row["bus"],
                "time_when_SOC_too_low": start_t,
                "activity": row.get("activity", None),
                "start_location": row.get("start_location", None),
                "end_location": row.get("end_location", None),

                # SOC at start and the worst SOC reached in this activity
                "SOC_at_start (%)": round(row["SOC_start (%)"], 1),
                "SOC_min_during (%)": round(row["SOC_min_during (%)"], 1),
                "SOC_at_start (kWh)": round(row["SOC_start (kWh)"], 2),
                "SOC_min_during (kWh)": round(row["SOC_min_during (kWh)"], 2),

                # Limit
                "minimum_allowed (%)": round(row["min_battery (%)"], 1),
                "minimum_allowed (kWh)": round(row["min_battery (kWh)"], 2),

                # For proper ordering (00:00-03:00 goes last)
                "_sort_key_dt": to_shifted_dt(start_t),
            })

        # First violation for this bus = earliest violation row after sorting by start_dt_shifted
        if not bad_rows_all.empty:
            first_row = bad_rows_all.iloc[0]
            first_violation_rows.append({
                "bus": first_row["bus"],
                "time_when_SOC_too_low": first_row["start_time"],
                "activity": first_row.get("activity", None),
                "start_location": first_row.get("start_location", None),
                "end_location": first_row.get("end_location", None),

                "SOC_at_start (%)": round(first_row["SOC_start (%)"], 1),
                "SOC_min_during (%)": round(first_row["SOC_min_during (%)"], 1),
                "SOC_at_start (kWh)": round(first_row["SOC_start (kWh)"], 2),
                "SOC_min_during (kWh)": round(first_row["SOC_min_during (kWh)"], 2),

                "minimum_allowed (%)": round(first_row["min_battery (%)"], 1),
                "minimum_allowed (kWh)": round(first_row["min_battery (kWh)"], 2),
            })

    # -------- Streamlit output --------
    total_violations = len(all_violations_rows)

    if total_violations == 0:
        st.success("No violating moments detected — all buses stay above the minimum SOC limit.")
    elif total_violations == 1:
        st.error("1 violating moment detected across all buses.")
    else:
        st.error(f"{total_violations} violating moments detected across all buses.")

    # 1) Summary: First violation per bus
    if first_violation_rows:
        st.subheader("First violating activity per bus")
        summary_df = pd.DataFrame(first_violation_rows)
        st.dataframe(summary_df, use_container_width=True)

    # 2) Full list: All violation moments
    if all_violations_rows:
        with st.expander("All violating moments (bus can fail multiple times)"):
            all_df = pd.DataFrame(all_violations_rows)
            all_df = all_df.sort_values(["bus", "_sort_key_dt"]).reset_index(drop=True)
            all_df_display = all_df.drop(columns=["_sort_key_dt"])
            st.dataframe(all_df_display, use_container_width=True)

    # 3) Debug timeline
    with st.expander("Full SOC timeline per bus (debug)"):
        debug_cols = [
            "bus",
            "start_time",
            "end_time",
            "activity",
            "SOC_start (kWh)",
            "SOC_start (%)",
            "SOC_after (kWh)",
            "SOC_after (%)",
            "SOC_min_during (kWh)",
            "SOC_min_during (%)",
            "min_battery (kWh)",
            "min_battery (%)",
            "violation",
            "start_location",
            "end_location",
            "energy_consumption"
        ]
        debug_view = df[debug_cols].copy()
        for col in ["SOC_start (kWh)", "SOC_after (kWh)", "SOC_min_during (kWh)", "min_battery (kWh)"]:
            debug_view[col] = debug_view[col].round(2)
        for col in ["SOC_start (%)", "SOC_after (%)", "SOC_min_during (%)", "min_battery (%)"]:
            debug_view[col] = debug_view[col].round(1)
        st.dataframe(debug_view, use_container_width=True)


# -------------------------------------------------
# Energy calculation
# -------------------------------------------------

def calculate_energy_consumption(planning: pd.DataFrame,
                                 distancematrix: pd.DataFrame,
                                 driving_usage,
                                 idle_usage,
                                 charging_speed):
    """
    Recalculate energy consumption using:
    - driving_usage [kW/km]
    - idle_usage [kW constant]
    - charging_speed [kW/h] (negative, so charging adds energy)
    
    Input:
        Bus planning as a Pandas DataFrame
        Distance matrix as a Pandas DataFrame
        Driving_usage in kW/km
        Idle_usage in kW (constant)
        Charging speed in kW/h
        
    Output:
        Bus planning with added recalculated energy
    """

    planning = planning.copy()
    distance = distancematrix.copy()

    # Match column names for merge
    distance = distance.copy()
    distance.rename(columns={"start": "start_location", "end": "end_location"}, inplace=True)

    # Ensure consistent types
    for col in ["start_location", "end_location", "line"]:
        if col in planning.columns:
            planning[col] = planning[col].astype(str)
        if col in distance.columns:
            distance[col] = distance[col].astype(str)

    df = planning.merge(
        distance[["start_location", "end_location", "line", "distance_m"]],
        on=["start_location", "end_location", "line"],
        how="left"
    )

    if "diff_min" not in df.columns:
        df["diff_min"] = df["diff"].dt.total_seconds() / 60

    df["energy_consumption_new"] = np.select(
        condlist=[
            df["activity"].str.contains("idle", case=False, na=False),
            df["activity"].str.contains("charging", case=False, na=False)
        ],
        choicelist=[
            idle_usage,
            (df["diff_min"] * charging_speed * -1) / 60
        ],
        default=(df["distance_m"] / 1000) * driving_usage
    )

    # Very important: also use the new calculation in SOC_check
    df["energy_consumption"] = df["energy_consumption_new"]

    return df


# -------------------------------------------------
# Demo main (for direct Streamlit use)
# -------------------------------------------------

# Default constants
driving_usage = 1.2
idle_usage = 5
charging_speed = 450
SOH = 90
minbat = 10
startbat = 100


def main(timetable: pd.DataFrame, planning: pd.DataFrame, distancematrix: pd.DataFrame):
    """
    Run all checks in order and show output in Streamlit.
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
    planning_calculated = calculate_energy_consumption(
        planning_with_length,
        distancematrix,
        driving_usage,
        idle_usage,
        charging_speed
    )

    st.header("Check SOC")
    SOC_check(planning_calculated, SOH, minbat, startbat)

    st.header("Gantt Chart of Bus Planning")
    create_gannt_chart(planning_calculated)
