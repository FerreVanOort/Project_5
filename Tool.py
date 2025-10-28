# Tool.py / Main tool

import io
import base64
from pathlib import Path
from datetime import timedelta
import pandas as pd
import streamlit as st

import Formulas as fm
import PlanningMaker as pm


st.set_page_config(page_title="Prototype groep 8", layout="wide")

# Sidebar navigation
st.sidebar.header("Menu")
page = st.sidebar.radio(
    "Go to page:",
    ["Planning Checker", "Planning Maker", "Advanced Options", "User Manual", "About Us"],
    label_visibility="collapsed"
)

# Session defaults
if "driving_usage" not in st.session_state:
    st.session_state.driving_usage = 1.2
if "idle_usage" not in st.session_state:
    st.session_state.idle_usage = 5.0
if "charging_speed" not in st.session_state:
    st.session_state.charging_speed = 450.0
if "soh" not in st.session_state:
    st.session_state.soh = 90.0
if "minbat" not in st.session_state:
    st.session_state.minbat = 10.0
if "startbat" not in st.session_state:
    st.session_state.startbat = 100.0


# -------------------------------------------------
# Helper for About Us cards
# -------------------------------------------------
BASE_DIR = Path(__file__).parent
IMG_DIR = BASE_DIR / "images"

def team_member(filename: str, name: str, linkedin_url: str):
    p = IMG_DIR / filename
    try:
        img_bytes = p.read_bytes()
    except Exception as e:
        st.error(f"Kon afbeelding niet laden: {p} ({e})")
        return ""

    ext = p.suffix.lower()
    mime = "image/png" if ext == ".png" else "image/jpeg"
    b64 = base64.b64encode(img_bytes).decode("utf-8")

    return f"""
    <div style="
        background-color:#6e6e6e;
        padding:15px;
        border-style:solid;
        border-width:2px;
        border-color:#404040;
        border-radius:10px;
        text-align:center;
        ">
        <img src="data:{mime};base64,{b64}"
            style="border-radius:50%;
                width:120px;
                height:120px;
                object-fit:cover;
                margin-bottom:10px;">
        <h5 style="color:white; margin:0; margin-bottom:8px;">{name}</h5>
        <a href="{linkedin_url}" target="_blank"
            style="text-decoration:none; color:#0077b5; font-weight:bold;">
            LinkedIn
        </a>
    </div>
    """


# -------------------------------------------------
# Page 1 - Planning Checker
# -------------------------------------------------
if page == "Planning Checker":
    st.title("Prototype Group 8 - Bus Planning Check", anchor='group 8')
    
    st.subheader("Upload Bus Planning, Timetable, and Distance Matrix!")
    uploaded_planning = st.file_uploader("Upload planning (.xlsx)", type=["xlsx"], key="plan_upload")
    uploaded_distances = st.file_uploader("Upload distance matrix (.xlsx)", type=["xlsx"], key="dist_upload")
    uploaded_timetable = st.file_uploader("Upload timetable (.xlsx)", type=["xlsx"], key="time_upload")

    if uploaded_planning and uploaded_distances and uploaded_timetable:
        try:
            planning = pd.read_excel(uploaded_planning, engine="openpyxl")
            distancematrix = pd.read_excel(uploaded_distances, engine="openpyxl")
            timetable = pd.read_excel(uploaded_timetable, engine="openpyxl")
            
            st.success("All files succesfully loaded!")
            
            # --- Cleanup and format check ---
            st.header("Cleanup & Format Check")
            planning_clean = fm.cleanup_excel(planning)
            timetable = fm.cleanup_timetable(timetable)
            distancematrix = fm.cleanup_timetable(distancematrix)
            fm.check_format_excel(planning_clean)
            
            # --- Fill idle periods ---
            planning_filled = fm.fill_idle_periods(planning_clean)
            
            # --- Length of activities ---
            planning_length = fm.length_activities(planning_filled)
            
            # --- Charging check ---
            st.header("Checked charging moments")
            fm.charging_check(planning_length)
            fm.charge_time(planning_length)
            
            # --- Timetable coverage check ---
            st.header("Checked coverage of timetable")
            fm.check_timetable(timetable, planning_length)
            
            # --- Ride duration check ---
            st.header("Checked ride duration times")
            fm.check_ride_duration(planning_length, distancematrix)
            
            # --- Energy calculation ---
            planning_energy = fm.calculate_energy_consumption(
                planning_length,
                distancematrix,
                driving_usage = st.session_state.driving_usage,
                idle_usage = st.session_state.idle_usage,
                charging_speed = st.session_state.charging_speed
            )
            
            # --- State of Charge check ---
            st.header("State of Charge check")
            fm.SOC_check(
                planning_energy,
                SOH = st.session_state.soh,
                minbat = st.session_state.minbat,
                startbat = st.session_state.startbat
            )
            
            # --- Gantt Chart ---
            st.header("Gantt Chart of uploaded Bus Plan")
            fm.create_gannt_chart(planning_energy)
            
        except Exception as e:
            st.error(f"Something went wrong with the processing of files: {e}")
            st.exception(e)
        
    else:
        st.info("Upload a bus plan (use the CheckerInput sheet from Planning Maker), timetable, and distance matrix to start!")


# -------------------------------------------------
# Page 2 - Planning Maker
# -------------------------------------------------
elif page == "Planning Maker":
    st.title("Prototype Group 8 - Bus Planning Maker", anchor='group 8')
    
    st.subheader("Upload Timetable and Distance Matrix")
    uploaded_timetable_build = st.file_uploader("Upload timetable (.xlsx)", type=["xlsx"], key="time_build")
    uploaded_distances_build = st.file_uploader("Upload distance matrix (.xlsx)", type=["xlsx"], key="dist_build")

    if uploaded_timetable_build and uploaded_distances_build:
        try:
            timetable_build = pd.read_excel(uploaded_timetable_build, engine="openpyxl")
            distancematrix_build = pd.read_excel(uploaded_distances_build, engine="openpyxl")
            
            st.success("All files successfully loaded!")
            
            # --- Configuration section ---
            st.header("Planning Configuration")
            col1, col2 = st.columns(2)
            
            with col1:
                charging_station = st.text_input(
                    "Charging Station Name",
                    value="ehvgar", 
                    help="Name of charging station (must match distance matrix)"
                )
                garage_location = st.text_input(
                    "Garage Location Name",
                    value="ehvgar",
                    help="Where buses start from (must match distance matrix)"
                )
            
            with col2:
                st.info("⚡ Battery & energy settings (from Advanced Options)")
                usable_cap = 300 * (st.session_state.soh / 100.0)
                st.write(f"Nominal pack: 300 kWh")
                st.write(f"SOH: {st.session_state.soh:.1f}% → usable ≈ {usable_cap:.1f} kWh")
                st.write(f"Start SOC: {st.session_state.startbat:.1f}% of usable")
                st.write(f"Min SOC: {st.session_state.minbat:.1f}% of usable")
                st.write(f"Drive consumption: {st.session_state.driving_usage:.2f} kWh/km")
                st.write(f"Idle usage: {st.session_state.idle_usage:.2f} kW")
                st.write(f"Charging power: {st.session_state.charging_speed:.1f} kW")
                st.write("Charging only allowed at garage, min 15 min session.")
            
            # --- Generate Planning Button ---
            if st.button("Generate Bus Planning", type="primary"):
                
                # 1. sync naar pm.BusConstants
                pm.BusConstants.SOH_PERCENT        = st.session_state.soh
                pm.BusConstants.START_BAT_PERCENT  = st.session_state.startbat
                pm.BusConstants.MIN_BAT_PERCENT    = st.session_state.minbat
                pm.BusConstants.CHARGING_POWER_KW  = st.session_state.charging_speed
                pm.BusConstants.CONSUMPTION_PER_KM = st.session_state.driving_usage
                pm.BusConstants.IDLE_USAGE_KW      = st.session_state.idle_usage
                pm.BusConstants.GARAGE_NAME        = garage_location
                
                with st.spinner("Generating bus planning..."):
                    
                    # 2. load data
                    st.write("📊 Loading distance matrix...")
                    loader = pm.DataLoader()
                    distance_dict, time_dict, distance_df = loader.load_distance_matrix_from_df(distancematrix_build)
                    
                    st.write("📅 Loading timetable...")
                    rides = loader.load_timetable_from_df(timetable_build, distance_df)
                    
                    if not rides:
                        st.error("❌ No rides could be loaded from timetable!")
                    else:
                        st.success(f"✅ Loaded {len(rides)} rides from timetable")
                        
                        # 3. init scheduling
                        st.write("🔧 Initializing planning system...")
                        distance_matrix = pm.DistanceMatrix(distance_dict, time_dict)
                        charging_planner = pm.ChargingPlanner(charging_station)
                        scheduler = pm.BusScheduler(distance_matrix, charging_planner, garage_location)
                        
                        # 4. initial fleet
                        first_start_time = min(r.start_time for r in rides)
                        initial_buses = [
                            pm.Bus(
                                bus_id="BUS_1",
                                current_location=garage_location,
                                battery_kwh=pm.BusConstants.start_energy_kwh(),
                                current_time=first_start_time - timedelta(hours=1),
                                history=[],
                                occupied=[]
                            )
                        ]
                        
                        # 5. schedule all rides
                        st.write("🚌 Scheduling rides to buses...")
                        assignments = scheduler.schedule_all_rides(rides, initial_buses)
                        
                        st.success(f"✅ Planning complete! {len(assignments)} rides processed")
                        
                        # 6. overzicht per service rit
                        usable_cap_after_soh = pm.BusConstants.usable_capacity_kwh()
                        schedule_rows = []
                        for a in assignments:
                            pct_before = (
                                a.battery_before / usable_cap_after_soh * 100.0
                                if usable_cap_after_soh > 0 else float("nan")
                            )
                            pct_after = (
                                a.battery_after / usable_cap_after_soh * 100.0
                                if usable_cap_after_soh > 0 else float("nan")
                            )
                            
                            schedule_rows.append({
                                "Bus_ID": a.bus_id,
                                "Line": a.ride.line,
                                "Start_Location": a.ride.start_stop,
                                "End_Location": a.ride.end_stop,
                                "Start_Time": a.ride.start_time.strftime("%H:%M"),
                                "End_Time": a.ride.end_time.strftime("%H:%M"),
                                "Battery_Before_[%usable]": round(pct_before, 1),
                                "Battery_After_[%usable]": round(pct_after, 1),
                            })
                        
                        df_planning = pd.DataFrame(schedule_rows)
                        
                        # 7. bouw ruwe tijdsblokken (gantt_rows)
                        gantt_rows = []
                        for a in assignments:
                            # charging_before
                            if a.charging_before is not None:
                                charge_start_dt, charge_end_dt, bat_before_ch, bat_after_ch = a.charging_before
                                charged_kwh = bat_after_ch - bat_before_ch
                                gantt_rows.append({
                                    "bus": a.bus_id,
                                    "activity": "charging",
                                    "start_time": charge_start_dt,
                                    "end_time": charge_end_dt,
                                    "start_location": pm.BusConstants.GARAGE_NAME,
                                    "end_location": pm.BusConstants.GARAGE_NAME,
                                    "line": "",
                                    "energy_consumption": -(charged_kwh),
                                    "travel_min_used": None,  # niet van toepassing
                                })

                            # deadhead_before
                            if a.deadhead_before is not None:
                                from_loc, to_loc, dist_km, dep_dt, arr_dt = a.deadhead_before
                                energy_deadhead = dist_km * pm.BusConstants.CONSUMPTION_PER_KM
                                gantt_rows.append({
                                    "bus": a.bus_id,
                                    "activity": "deadhead",
                                    "start_time": dep_dt,
                                    "end_time": arr_dt,
                                    "start_location": from_loc,
                                    "end_location": to_loc,
                                    "line": "",
                                    "energy_consumption": energy_deadhead,
                                    "travel_min_used": (arr_dt - dep_dt).total_seconds() / 60.0,
                                })

                            # idle_before
                            if a.idle_before is not None:
                                idle_start_dt, idle_end_dt, idle_energy_used = a.idle_before
                                if idle_end_dt > idle_start_dt:
                                    gantt_rows.append({
                                        "bus": a.bus_id,
                                        "activity": "idle",
                                        "start_time": idle_start_dt,
                                        "end_time": idle_end_dt,
                                        "start_location": a.ride.start_stop,
                                        "end_location": a.ride.start_stop,
                                        "line": "",
                                        "energy_consumption": idle_energy_used,
                                        "travel_min_used": (idle_end_dt - idle_start_dt).total_seconds() / 60.0,
                                    })

                            # service trip
                            energy_service = a.ride.distance_km * pm.BusConstants.CONSUMPTION_PER_KM
                            gantt_rows.append({
                                "bus": a.bus_id,
                                "activity": "service",
                                "start_time": a.ride.start_time,
                                "end_time": a.ride.end_time,
                                "start_location": a.ride.start_stop,
                                "end_location": a.ride.end_stop,
                                "line": a.ride.line,
                                "energy_consumption": energy_service,
                                "travel_min_used": a.ride.travel_min_used,  # <- uit distance matrix voor die lijn!
                            })

                        df_gantt = pd.DataFrame(gantt_rows)

                        # 8. CheckerInput export met rijke info
                        def to_hms(dtval):
                            return dtval.strftime("%H:%M:%S")

                        def to_hm(dtval):
                            return dtval.strftime("%H:%M")

                        def minutes_since_service_day(dtval):
                            # zelfde nachtverschuiving als DataLoader
                            mins = dtval.hour * 60 + dtval.minute
                            if dtval.hour < 3:
                                mins += 24 * 60
                            return mins

                        def map_logical_line(line_val):
                            # groepeer evt. lijnen met dezelfde corridor
                            s = str(line_val) if line_val is not None else ""
                            if s in ["400", "401"]:
                                return "400_401"
                            return s

                        # helper om bij service blocks de originele Ride terug te vinden
                        def find_matching_ride(bus_id, start_dt, end_dt, line_val):
                            for a in assignments:
                                if (
                                    a.bus_id == bus_id and
                                    a.ride.start_time == start_dt and
                                    a.ride.end_time == end_dt and
                                    str(a.ride.line) == str(line_val)
                                ):
                                    return a.ride
                            return None

                        checker_rows = []
                        for row in gantt_rows:
                            is_srv = (row["activity"] == "service")

                            start_dt = row["start_time"]
                            end_dt   = row["end_time"]

                            start_abs = minutes_since_service_day(start_dt)
                            end_abs   = minutes_since_service_day(end_dt)
                            planned_duration_min = end_abs - start_abs

                            # probeer de juiste Ride te vinden voor extra metadata
                            ride_obj = None
                            if is_srv:
                                ride_obj = find_matching_ride(
                                    row["bus"],
                                    row["start_time"],
                                    row["end_time"],
                                    row["line"]
                                )

                            expected_min_travel_time_from_matrix = None
                            if ride_obj is not None:
                                expected_min_travel_time_from_matrix = ride_obj.travel_min_used

                            checker_rows.append({
                                # kolommen die de Checker tot nu toe verwacht
                                "start_location":     row["start_location"],
                                "end_location":       row["end_location"],
                                "start_time":         to_hms(start_dt),   # "HH:MM:SS"
                                "end_time":           to_hms(end_dt),     # "HH:MM:SS"
                                "activity":           row["activity"],   # service/deadhead/idle/charging
                                "line":               row["line"],
                                "energy_consumption": row["energy_consumption"],
                                "bus":                row["bus"],

                                # extra matching info / debug info
                                "is_service_trip":    1 if is_srv else 0,
                                "assigned_bus":       row["bus"] if is_srv else "",
                                "start_time_short":   to_hm(start_dt),    # "HH:MM"
                                "end_time_short":     to_hm(end_dt),      # "HH:MM"
                                "start_abs_min":      start_abs,
                                "end_abs_min":        end_abs,
                                "planned_duration_min": planned_duration_min,

                                # corridor info
                                "logical_line":       map_logical_line(row["line"]),
                                "corridor_key":       f"{row['start_location']}->{row['end_location']}",
                                "corridor_travel_min": planned_duration_min,

                                # belangrijkste toevoeging:
                                # de rijtijd zoals gekozen uit de juiste distance-matrixregel voor deze lijn
                                "expected_min_travel_time_from_matrix": expected_min_travel_time_from_matrix,
                            })

                        df_checker_input = pd.DataFrame(
                            checker_rows,
                            columns=[
                                "start_location",
                                "end_location",
                                "start_time",
                                "end_time",
                                "activity",
                                "line",
                                "energy_consumption",
                                "bus",
                                "is_service_trip",
                                "assigned_bus",
                                "start_time_short",
                                "end_time_short",
                                "start_abs_min",
                                "end_abs_min",
                                "planned_duration_min",
                                "logical_line",
                                "corridor_key",
                                "corridor_travel_min",
                                "expected_min_travel_time_from_matrix",
                            ]
                        ).sort_values(
                            by=["bus", "start_abs_min", "end_abs_min"]
                        ).reset_index(drop=True)

                        # 9. Checks / KPIs
                        st.header("Planning Results")
                        
                        def battery_after_pct_for_row(row):
                            for a in assignments:
                                same_bus = (a.bus_id == row["Bus_ID"])
                                same_start = (a.ride.start_time.strftime("%H:%M") == row["Start_Time"])
                                same_end = (a.ride.end_time.strftime("%H:%M") == row["End_Time"])
                                if same_bus and same_start and same_end:
                                    if usable_cap_after_soh > 0:
                                        return a.battery_after / usable_cap_after_soh * 100.0
                                    else:
                                        return float("nan")
                            return float("nan")
                        
                        df_planning["Battery_After_[%usable]_calc"] = df_planning.apply(
                            battery_after_pct_for_row,
                            axis=1
                        )

                        negative_mask = df_planning["Battery_After_[%usable]_calc"] < 0
                        negative_batteries = df_planning[negative_mask]
                        
                        if not negative_batteries.empty:
                            st.error(f"⚠️ WARNING: {len(negative_batteries)} rides have impossible battery (<0%).")
                            with st.expander("Show problematic rides"):
                                st.dataframe(negative_batteries, use_container_width=True)
                        else:
                            st.success("✅ No impossible battery states detected.")
                        
                        col1_kpi, col2_kpi, col3_kpi = st.columns(3)
                        with col1_kpi:
                            st.metric("Total Service Trips", len(df_planning))
                        with col2_kpi:
                            st.metric("Buses Used", df_planning['Bus_ID'].nunique())
                        with col3_kpi:
                            min_battery_after = df_planning["Battery_After_[%usable]_calc"].min()
                            st.metric(
                                "Minimum Battery After Ride",
                                f"{min_battery_after:.1f}%",
                                delta="OK" if min_battery_after >= st.session_state.minbat else "CRITICAL",
                                delta_color="normal" if min_battery_after >= st.session_state.minbat else "inverse"
                            )
                        
                        # 10. tabel en gantt-visual
                        st.subheader("Generated Planning (per service ride)")
                        st.dataframe(df_planning, use_container_width=True)

                        def label_for_chart(row):
                            if row["activity"] == "deadhead":
                                return "material trip"
                            if row["activity"] == "service":
                                if row["line"] is not None and row["line"] != "":
                                    return f"service {row['line']}"
                                return "service"
                            return row["activity"]  # charging / idle

                        df_gantt_for_chart = df_gantt.copy()
                        df_gantt_for_chart["activity"] = df_gantt_for_chart.apply(label_for_chart, axis=1)
                        df_gantt_for_chart["start_time"] = df_gantt_for_chart["start_time"].dt.time
                        df_gantt_for_chart["end_time"]   = df_gantt_for_chart["end_time"].dt.time

                        st.subheader("Gantt Chart of Generated Bus Plan")
                        fm.create_gannt_chart(
                            df_gantt_for_chart.rename(columns={
                                "bus": "bus",
                                "activity": "activity",
                                "start_time": "start_time",
                                "end_time": "end_time",
                                "start_location": "start_location",
                                "end_location": "end_location",
                                "line": "line",
                                "energy_consumption": "energy_consumption",
                            })
                        )

                        # 11. Checker Input Preview
                        st.subheader("Checker Input Preview (upload this in the Checker)")
                        st.caption(
                            "is_service_trip=1 ⇒ passagiersrit uit de dienstregeling.\n"
                            "assigned_bus ⇒ toegewezen bus.\n"
                            "planned_duration_min ⇒ hoe lang de rit volgens de planning duurt (met middernacht-correctie).\n"
                            "expected_min_travel_time_from_matrix ⇒ minimale rijtijd volgens distance matrix voor die specifieke lijn.\n"
                            "Als die twee overeenkomen is de rit qua duur geldig.\n"
                            "start_time_short ⇒ tijd zoals in de timetable (HH:MM, zonder seconden) om makkelijk te matchen."
                        )
                        st.dataframe(df_checker_input, use_container_width=True)

                        # 12. Download direct CheckerInput
                        st.subheader("Download file to use directly in the Checker")
                        output_xlsx = io.BytesIO()
                        with pd.ExcelWriter(output_xlsx, engine='openpyxl') as writer:
                            df_checker_input.to_excel(writer, sheet_name='CheckerInput', index=False)
                        output_xlsx.seek(0)

                        st.download_button(
                            label="📥 Download CheckerInput (Excel)",
                            data=output_xlsx,
                            file_name="checker_input.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
            
        except Exception as e:
            st.error(f"Something went wrong with the processing of files: {e}")
            st.exception(e)
    else:
        st.info("Upload a timetable and distance matrix to start generating a bus planning!")
     

# -------------------------------------------------
# Page 3 - Advanced Options
# -------------------------------------------------
elif page == "Advanced Options":
    st.title("Advanced Options", anchor='group 8')
    st.markdown("Change values to impact energy usage")
    
    st.session_state.driving_usage = st.number_input(
        "Driving usage (kWh/km) [Please round to 1 decimal point]",
        min_value=0.0,
        max_value=100.0,
        value=st.session_state.driving_usage,
        step=0.1,
        format="%.2f",
        help="Average energy usage while driving (kWh per km)."
    )

    st.session_state.idle_usage = st.number_input(
        "Idle usage (kW constant) [Please round to 1 decimal point]",
        min_value=0.0,
        max_value=500.0,
        value=st.session_state.idle_usage,
        step=0.1,
        format="%.1f",
        help="Constant usage while idle."
    )

    st.session_state.charging_speed = st.number_input(
        "Charging speed (kW/h) [Please round to a whole number]",
        min_value=0.0,
        max_value=2000.0,
        value=st.session_state.charging_speed,
        step=1.0,
        format="%.1f",
        help="Charging speed in kWh per hour."
    )

    st.session_state.soh = st.slider(
        "State of Health of battery [%]",
        min_value=0.0,
        max_value=100.0,
        value=st.session_state.soh,
        step=0.5,
    )

    st.session_state.minbat = st.slider(
        "Minimum battery when reaching garage [%]",
        min_value=0.0,
        max_value=100.0,
        value=st.session_state.minbat,
        step=0.5,
    )

    st.session_state.startbat = st.slider(
        "Starting battery when the day starts [%]",
        min_value=0.0,
        max_value=100.0,
        value=st.session_state.startbat,
        step=0.5,
    )
    
    st.info("⚙️ These options are automatically implemented in the calculations about the bus plan.")


# -------------------------------------------------
# Page 4 - User Manual
# -------------------------------------------------
elif page == "User Manual":
    st.title("User Manual", anchor='group 8')
    st.write("""
This tool has two main modules:

1. Planning Maker
   - Upload timetable + distance matrix
   - Configure energy/charging settings
   - Generate an operating plan with assigned buses
   - Export that plan

2. Planning Checker
   - Upload a planning (use the 'CheckerInput' sheet from Planning Maker)
   - Upload timetable + distance matrix
   - Verify coverage, travel times, charging rules, and SOC
   - Visualize full-day Gantt

Workflow:
DistanceMatrix.xlsx + Timetable.xlsx → Planning Maker → download Excel → upload 'CheckerInput' sheet into Planning Checker.
""")


# -------------------------------------------------
# Page 5 - About Us
# -------------------------------------------------
elif page == "About Us":
    st.title("About Us", anchor='group 8')
    st.header('The team')
    st.write("""
We are a team of enthusiastic students - Ferre, Mirthe and Fea - from Eindhoven, studying Applied mathematics at the Fontys University of Applied Sciences. Ferre takes the lead in dividing the tasks and ensuring everything runs smoothly. Ferre also focuses on coding the planning maker, while Fea is responsible for coding the planning checker. Mirthe is responsible for the streamlit interface. Together, we bring our unique skills and perspectives to create innovative solutions in the field of applied mathematics.
""")

    st.header('Project Planning Checker and Maker for Electric Bus Fleets')
    st.write("""
The PlanningChecker verifies whether your bus schedule is complete and accurate. Whether you’re new to this type of software or already experienced, the tool helps you evaluate and improve your planning.

With the growing shift toward electric buses, scheduling now involves stricter requirements. PlanningChecker ensures that each bus plan complies with these modern standards.

It checks if all routes are covered, whether each bus has sufficient charging time (at least 15 minutes), and if any bus’s State of Charge (SOC) drops below the allowed minimum. If any conditions aren’t met, PlanningChecker clearly indicates where the issues occur, making it easier for planners to identify problems, correct them efficiently, and reduce errors in the overall scheduling process.
""")

    st.header("Meet the team")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            team_member(
                "fea.jpg",
                "Fea Sanders",
                "https://www.linkedin.com/in/fea-sanders-04626b24b"
            ),
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            team_member(
                "ferre.jpg",
                "Ferre van Oort",
                "https://www.linkedin.com/in/ferrevanoort"
            ),
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            team_member(
                "mirthe.jpg",
                "Mirthe Termeulen",
                "https://www.linkedin.com/in/mirthetermeulen"
            ),
            unsafe_allow_html=True
        )

# End of Tool.py
