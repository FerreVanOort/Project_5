# Formulas.py

import pandas as pd
import numpy as np
import streamlit as st
import datetime
from datetime import time
from datetime import datetime, timedelta
import plotly.express as px


def cleanup_excel(planning: pd.DataFrame) -> pd.DataFrame:
    planning = planning.copy()
    planning.columns = planning.columns.str.replace(" ", "_")
    planning.columns = planning.columns.str.lower()
    return planning


def cleanup_timetable(timetable: pd.DataFrame) -> pd.DataFrame:
    timetable = timetable.copy()
    timetable.columns = [
        str(c).strip().replace("\u00A0", "").replace(" ", "_").lower()
        for c in timetable.columns
    ]
    return timetable


def check_format_excel(planning: pd.DataFrame):
    """
    Checker verwacht exact deze kolommen:
    ['start_location','end_location','start_time','end_time',
     'activity','line','energy_consumption','bus']
    """
    cols = list(planning.columns)
    expected = [
        'start_location',
        'end_location',
        'start_time',
        'end_time',
        'activity',
        'line',
        'energy_consumption',
        'bus'
    ]
    if cols == expected:
        st.success("The uploaded file has the right format")
    else:
        st.error("The uploaded file does not have the right format")
        st.write("Expected:", expected)
        st.write("Got:", cols)


def charging_check(planning: pd.DataFrame):
    errors = planning[
        (planning['activity'].str.lower() == 'charging') &
        (planning['energy_consumption'] >= 0)
    ]
    if errors.empty:
        st.success("Charging rates are correct (negative energy = charging).")
    else:
        st.error("Charging rows have positive energy_consumption (should be negative).")
        st.subheader("Error lines:")
        st.dataframe(errors, use_container_width=True)


def length_activities(planning: pd.DataFrame,
                      start_col="start_time",
                      end_col="end_time") -> pd.DataFrame:
    """
    Voeg diff + duration_min toe.
    Houd rekening met activiteiten die over middernacht gaan.
    """
    planning = planning.copy()

    # Zet tijden om naar datetime met dummy datum
    start_dt = pd.to_datetime(planning[start_col], format="%H:%M:%S")
    end_dt   = pd.to_datetime(planning[end_col],   format="%H:%M:%S")

    # Over-middernacht fix
    end_dt = end_dt.where(end_dt >= start_dt, end_dt + pd.Timedelta(days=1))

    planning["diff"] = end_dt - start_dt
    planning["duration_min"] = planning["diff"].dt.total_seconds() / 60

    planning[start_col] = start_dt.dt.time
    planning[end_col]   = end_dt.dt.time

    return planning


def charge_time(planning: pd.DataFrame):
    """
    Check of elke charging activity minstens 15 minuten duurt.
    """
    charging_moments = planning[planning['activity'].str.lower().str.contains("charging")]

    too_short = charging_moments[charging_moments['diff'] < pd.Timedelta(minutes=15)]

    if len(too_short) > 0:
        st.error(f"There are {len(too_short)} charging blocks shorter than 15 minutes.")
        with st.expander("More info on charging times"):
            short_charge = too_short[["start_time", "end_time", "activity"]]
            st.write(pd.DataFrame(short_charge))
    else:
        st.success("All charging moments are >= 15 minutes.")


def convert_to_time(value):
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


def fill_idle_periods(planning: pd.DataFrame, base_day: datetime = None) -> pd.DataFrame:
    """
    Vul expliciete idle-gaten in als 'idle', tussen activiteiten van dezelfde bus.
    N.B. Maker exporteert al idle blokken, maar als user handmatig plant zonder idle,
         vullen we ze hier alsnog.
    """
    planning = planning.copy()
    
    if base_day is None:
        base_day = datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)
    
    # zet kolommen naar datetime.time
    planning["start_time"] = pd.to_datetime(planning["start_time"], format="%H:%M:%S").dt.time
    planning["end_time"]   = pd.to_datetime(planning["end_time"],   format="%H:%M:%S").dt.time

    planning["start_dt"] = planning["start_time"].apply(lambda t: datetime.combine(base_day.date(), t))
    planning["end_dt"]   = planning["end_time"].apply(lambda t: datetime.combine(base_day.date(), t))

    # fix over-middernacht
    planning.loc[planning["end_dt"] < planning["start_dt"], "end_dt"] += timedelta(days=1)

    # ritten tussen 00:00 en 03:00 horen eigenlijk 'volgende dag' te zijn
    mask_after_midnight = planning["start_dt"].dt.hour < 3
    planning.loc[mask_after_midnight, "start_dt"] += timedelta(days=1)
    planning.loc[mask_after_midnight, "end_dt"]   += timedelta(days=1)
    
    idle_rows = []

    for bus, bus_df in planning.groupby("bus"):
        bus_df = bus_df.sort_values("start_dt").reset_index(drop=True)

        for i in range(len(bus_df) - 1):
            current_end = bus_df.loc[i, "end_dt"]
            next_start = bus_df.loc[i + 1, "start_dt"]

            if next_start > current_end:
                # voeg idle block toe
                idle_row = {
                    "start_location": bus_df.loc[i, "end_location"],
                    "end_location":   bus_df.loc[i, "end_location"],
                    "start_time":     current_end.time(),
                    "end_time":       next_start.time(),
                    "activity":       "idle",
                    "line":           "",
                    "energy_consumption": 0.0,  # idle verbruik vullen we niet hier exact
                    "bus":            bus,
                    "start_dt":       current_end,
                    "end_dt":         next_start,
                }
                idle_rows.append(idle_row)
                
    if idle_rows:
        planning = pd.concat([planning, pd.DataFrame(idle_rows)], ignore_index=True)
        planning = planning.sort_values(["bus", "start_dt"]).reset_index(drop=True)

    # ruim hulpkolommen op
    planning = planning.drop(columns=["start_dt", "end_dt"], errors="ignore")
    
    return planning


def create_gannt_chart(planning: pd.DataFrame, base_day: datetime = None):
    """
    Maakt Gantt chart met kleuren per activity:
    - 'service trip line XXX' (unieke kleur per lijn)
    - 'material trip'
    - 'charging'
    - 'idle'
    """

    planning = planning.copy()

    # basisdatum
    if base_day is None:
        base_day = datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)
    planning["planning_day"] = base_day

    # verschuif start_tijden tussen 00:00–03:00 naar 'volgende dag'
    cutoff_start = datetime.strptime("00:00:00", "%H:%M:%S").time()
    cutoff_end   = datetime.strptime("03:00:00", "%H:%M:%S").time()
    mask_next_day = (
        (planning["start_time"] >= cutoff_start) &
        (planning["start_time"] <= cutoff_end)
    )
    planning.loc[mask_next_day, "planning_day"] += pd.Timedelta(days=1)

    # combineer times met datum
    planning["start_dt"] = planning.apply(
        lambda r: datetime.combine(r["planning_day"].date(), r["start_time"]),
        axis=1
    )
    planning["end_dt"] = planning.apply(
        lambda r: datetime.combine(r["planning_day"].date(), r["end_time"]),
        axis=1
    )

    # fix over-midnight
    planning["end_dt"] = planning["end_dt"].where(
        planning["end_dt"] >= planning["start_dt"],
        planning["end_dt"] + timedelta(days=1)
    )

    # duur
    planning["duration_min"] = (
        planning["end_dt"] - planning["start_dt"]
    ).dt.total_seconds() / 60.0

    def _pick_display_group(row):
        if row["activity"] == "service trip":
            # label per lijn
            line_val = str(row.get("line", "")).strip()
            if line_val == "" or line_val.lower() == "nan":
                return "service trip (unknown line)"
            try:
                line_str = str(int(float(line_val)))
            except Exception:
                line_str = line_val
            return f"service trip line {line_str}"
        else:
            return row["activity"]

    planning["display_group"] = planning.apply(_pick_display_group, axis=1)

    base_colors = {
        "charging": "green",
        "idle": "gray",
        "material trip": "orange",
        "service trip (unknown line)": "blue",
    }

    palette_cycle = ["blue", "purple", "red", "brown", "pink", "cyan", "olive", "magenta"]
    color_map = dict(base_colors)

    line_groups = [
        g for g in planning["display_group"].unique()
        if isinstance(g, str) and g.startswith("service trip line ")
    ]
    for idx, g in enumerate(line_groups):
        color_map[g] = palette_cycle[idx % len(palette_cycle)]
    
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
    df[column] = df[column].apply(convert_to_time)


def check_timetable(timetable, planning):
    """
    Controleert of alle ritten in timetable ook in planning zitten.
    timetable kolommen: line, start, end, departure_time
    planning kolommen: line, start_location, end_location, start_time
    """
    ensure_time_column(timetable, 'departure_time')
    ensure_time_column(planning, 'start_time')
    ensure_time_column(planning, 'end_time')

    unassigned_rides = []

    for _, ride in timetable.iterrows():
        matching = planning[
            (planning['line'].astype(str) == str(ride['line'])) &
            (planning['start_location'].astype(str) == str(ride['start'])) &
            (planning['end_location'].astype(str) == str(ride['end']))
        ]

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


def check_ride_duration(planning, distancematrix):
    """
    Controleer of de geplande service trips binnen de max_travel_time vallen
    volgens de distance matrix.
    """
    required_cols = ["start", "end", "min_travel_time", "max_travel_time", "line"]
    missing = [c for c in required_cols if c not in distancematrix.columns]
    if missing:
        st.error(f"Distance matrix mist kolommen: {', '.join(missing)}")
        st.write("Looking for:", required_cols)
        st.write("Actually have:", list(distancematrix.columns))
        return
    
    distancematrix = distancematrix.copy()
    distancematrix["min_travel_time"] = pd.to_numeric(distancematrix["min_travel_time"], errors='coerce')
    distancematrix["max_travel_time"] = pd.to_numeric(distancematrix["max_travel_time"], errors='coerce')

    if "duration_min" not in planning.columns:
        planning["duration_min"] = planning["diff"].dt.total_seconds() / 60
    
    wrong_rides = []

    for _, ride in planning.iterrows():
        if str(ride.get("activity", "")).lower() == "idle":
            continue
        if str(ride.get("activity", "")).lower() == "charging":
            continue

        start_loc = ride["start_location"]
        end_loc   = ride["end_location"]
        busline   = ride["line"]
        actual_duration = ride["duration_min"]
        bus = ride["bus"]

        match = distancematrix[
            (distancematrix["start"].astype(str) == str(start_loc)) &
            (distancematrix["end"].astype(str) == str(end_loc)) &
            (distancematrix["line"].astype(str) == str(busline))
        ]

        if match.empty:
            # geen entry → slaan we over
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


def SOC_check(planning: pd.DataFrame, SOH, minbat, startbat):
    """
    SOC-check:
    - pakt planning in volgorde per bus (in tijd, met 00:xx na 23:xx)
    - rekent kWh over de dag
    - laat zien waar SOC onder min gaat
    """

    capacity_nominal = 300  # kWh fysiek
    df = planning.copy()

    # parse tijden terug naar datetime.time voor sorteren
    df["start_time"] = pd.to_datetime(df["start_time"], format="%H:%M:%S").dt.time
    df["end_time"]   = pd.to_datetime(df["end_time"],   format="%H:%M:%S").dt.time

    base_day = datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)

    def shift_time_to_seq(t: time):
        dt = datetime.combine(base_day.date(), t)
        # alles voor 03:00 beschouwen als volgende dag
        if t < time(3,0):
            dt += timedelta(days=1)
        return dt

    df["start_dt_shifted"] = df["start_time"].apply(shift_time_to_seq)
    df = df.sort_values(["bus", "start_dt_shifted"]).reset_index(drop=True)

    df["SOC_start (kWh)"] = np.nan
    df["SOC_start (%)"] = np.nan
    df["SOC_after (kWh)"] = np.nan
    df["SOC_after (%)"] = np.nan
    df["min_battery (kWh)"] = np.nan
    df["min_battery (%)"] = np.nan
    df["violation"] = False

    first_violation_rows = []
    all_violations_rows = []

    for bus_id, group in df.groupby("bus"):
        idxs = group.index

        max_battery_kwh = (SOH / 100.0) * capacity_nominal
        start_soc_kwh   = (startbat / 100.0) * max_battery_kwh
        min_soc_kwh     = (minbat / 100.0) * max_battery_kwh
        min_soc_pct     = (min_soc_kwh / max_battery_kwh) * 100.0

        usage = group["energy_consumption"].to_numpy()

        soc_start_kwh = np.empty(len(usage))
        soc_start_kwh[0] = start_soc_kwh
        soc_start_kwh[1:] = start_soc_kwh - np.cumsum(usage[:-1])
        soc_after_kwh = soc_start_kwh - usage

        soc_start_pct = (soc_start_kwh / max_battery_kwh) * 100.0
        soc_after_pct = (soc_after_kwh / max_battery_kwh) * 100.0

        df.loc[idxs, "SOC_start (kWh)"] = soc_start_kwh
        df.loc[idxs, "SOC_start (%)"] = soc_start_pct
        df.loc[idxs, "SOC_after (kWh)"] = soc_after_kwh
        df.loc[idxs, "SOC_after (%)"] = soc_after_pct
        df.loc[idxs, "min_battery (kWh)"] = min_soc_kwh
        df.loc[idxs, "min_battery (%)"] = min_soc_pct

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

        for _, row in bad_rows_all.iterrows():
            st_t = row["start_time"]
            all_violations_rows.append({
                "bus": row["bus"],
                "time_when_SOC_too_low": st_t,
                "activity": row.get("activity", None),
                "start_location": row.get("start_location", None),
                "end_location": row.get("end_location", None),
                "SOC_at_start (kWh)": round(row["SOC_start (kWh)"], 2),
                "SOC_at_start (%)": round(row["SOC_start (%)"], 1),
                "SOC_after (kWh)": round(row["SOC_after (kWh)"], 2),
                "SOC_after (%)": round(row["SOC_after (%)"], 1),
                "minimum_allowed (kWh)": round(row["min_battery (kWh)"], 2),
                "minimum_allowed (%)": round(row["min_battery (%)"], 1),
                "_sort_key_dt": shift_time_to_seq(st_t),
            })

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

    total_violations = len(all_violations_rows)

    if total_violations == 0:
        st.success("✅ No violating moments detected — all buses stay above the minimum SOC limit.")
    else:
        st.error(f"⚠️ There are {total_violations} violating moments detected across all buses.")

    if first_violation_rows:
        st.subheader("First violating activity per bus")
        summary_df = pd.DataFrame(first_violation_rows)
        st.dataframe(summary_df, use_container_width=True)

    if all_violations_rows:
        with st.expander("All violating moments (bus can fail multiple times)"):
            all_df = pd.DataFrame(all_violations_rows)
            all_df = all_df.sort_values(["bus", "_sort_key_dt"]).reset_index(drop=True)
            all_df_display = all_df.drop(columns=["_sort_key_dt"])
            st.dataframe(all_df_display, use_container_width=True)

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


def calculate_energy_consumption(planning: pd.DataFrame,
                                 distancematrix: pd.DataFrame,
                                 driving_usage,
                                 idle_usage,
                                 charging_speed):
    """
    Herbereken verbruik per activity als je een bestaande handmatige planning uploadt.
    (In Maker doen we dit al zelf, maar dit is voor Checker van user-input.)
    """
    planning = planning.copy()
    distance = distancematrix.copy()
    
    distance.rename(columns={"start": "start_location", "end": "end_location"}, inplace=True)
    
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
            (df["diff_min"] / 60.0) * idle_usage,        # idle verbruik
            (df["diff_min"] * charging_speed * -1) / 60  # charging (negatief)
        ],
        default=(df["distance_m"] / 1000.0) * driving_usage
    )
    
    return df


def main(timetable, planning, distancematrix):
    """
    Niet meer gebruikt direct in Tool.py, maar laat ik staan als losse debugger.
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
        driving_usage=1.2,
        idle_usage=5,
        charging_speed=450
    )
    
    st.header("Check SOC")
    SOC_check(planning_calculated, 90, 10, 100)
    
    st.header("Gantt Chart of Bus Planning")
    create_gannt_chart(planning_calculated)
