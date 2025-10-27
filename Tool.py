## Main tool

# Imports
from fileinput import filename
import Formulas as fm
import PlanningMaker as pm
import io
from datetime import datetime
from datetime import timedelta
import pandas as pd
import streamlit as st
import base64
from pathlib import Path

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
            st.header("Gannt Chart of uploaded Bus Plan")
            fm.create_gannt_chart(planning_energy)
            
        except Exception as e:
            st.error(f"Something went wrong with the processing of files {e}")
        
    else:
        st.info("Upload a bus plan, timetable, and distance matrix to start!")


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
                st.write(f"Charging power: {st.session_state.charging_speed:.1f} kW")
                st.write("Charging only allowed at garage, min 15 min session.")
            
            # --- Generate Planning Button ---
            if st.button("Generate Bus Planning", type="primary"):
                
                # -----------------------------
                # 1. SYNC PARAMETERS NAAR pm.BusConstants
                # -----------------------------
                pm.BusConstants.SOH_PERCENT        = st.session_state.soh
                pm.BusConstants.START_BAT_PERCENT  = st.session_state.startbat
                pm.BusConstants.MIN_BAT_PERCENT    = st.session_state.minbat
                pm.BusConstants.CHARGING_POWER_KW  = st.session_state.charging_speed
                pm.BusConstants.CONSUMPTION_PER_KM = st.session_state.driving_usage
                pm.BusConstants.IDLE_USAGE_KW      = st.session_state.idle_usage
                pm.BusConstants.GARAGE_NAME        = garage_location  # bv. "ehvgar"
                
                with st.spinner("Generating bus planning..."):
                    
                    # -----------------------------
                    # 2. LOAD / PARSE INPUT DATA
                    # -----------------------------
                    st.write("📊 Loading distance matrix...")
                    loader = pm.DataLoader()
                    distance_dict, time_dict, distance_df = loader.load_distance_matrix_from_df(distancematrix_build)
                    
                    st.write("📅 Loading timetable...")
                    rides = loader.load_timetable_from_df(timetable_build, distance_df)
                    
                    if not rides:
                        st.error("❌ No rides could be loaded from timetable!")
                    else:
                        st.success(f"✅ Loaded {len(rides)} rides from timetable")
                        
                        # -----------------------------
                        # 3. INIT PLANNING OBJECTS
                        # -----------------------------
                        st.write("🔧 Initializing planning system...")
                        distance_matrix = pm.DistanceMatrix(distance_dict, time_dict)
                        charging_planner = pm.ChargingPlanner(charging_station)
                        scheduler = pm.BusScheduler(distance_matrix, charging_planner, garage_location)
                        
                        # -----------------------------
                        # 4. INIT FLEET / START BUS
                        # -----------------------------
                        first_start_time = min(r.start_time for r in rides)
                        initial_buses = [
                            pm.Bus(
                                bus_id="BUS_1",
                                current_location=garage_location,
                                battery_kwh=pm.BusConstants.start_energy_kwh(),
                                current_time=first_start_time - timedelta(hours=1),
                                history=[]
                            )
                        ]
                        
                        # -----------------------------
                        # 5. SCHEDULE ALL RIDES
                        # -----------------------------
                        st.write("🚌 Scheduling rides to buses...")
                        assignments = scheduler.schedule_all_rides(rides, initial_buses)
                        
                        st.success(f"✅ Planning complete! {len(assignments)} rides processed")
                        
                        # -----------------------------
                        # 6. BOUW TABELLEN VOOR WEERGAVE
                        # -----------------------------
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
                        
                        gantt_rows = []
                        for a in assignments:
                            # Charging block (before ride)
                            if a.charging_before is not None:
                                charge_arrival_dt, charge_leave_dt, bat_before_ch, bat_after_ch = a.charging_before
                                
                                gantt_rows.append({
                                    "bus": a.bus_id,
                                    "activity": "charging",
                                    "start_time": charge_arrival_dt.time(),
                                    "end_time": charge_leave_dt.time(),
                                    "start_dt": charge_arrival_dt,
                                    "end_dt": charge_leave_dt,
                                    "start_location": pm.BusConstants.GARAGE_NAME,
                                    "end_location": pm.BusConstants.GARAGE_NAME,
                                    "line": "",
                                    "energy_consumption": -(bat_after_ch - bat_before_ch),  # negatief = laden
                                })
                            
                            # Deadhead block (material trip)
                            if a.deadhead_before is not None:
                                from_loc, to_loc, dist_km, dep_dt, arr_dt = a.deadhead_before
                                energy_deadhead = dist_km * pm.BusConstants.CONSUMPTION_PER_KM
                                
                                gantt_rows.append({
                                    "bus": a.bus_id,
                                    "activity": "material trip",
                                    "start_time": dep_dt.time(),
                                    "end_time": arr_dt.time(),
                                    "start_dt": dep_dt,
                                    "end_dt": arr_dt,
                                    "start_location": from_loc,
                                    "end_location": to_loc,
                                    "line": "",
                                    "energy_consumption": energy_deadhead,
                                })
                            
                            # Service trip block (actual passenger trip)
                            energy_service = a.ride.distance_km * pm.BusConstants.CONSUMPTION_PER_KM
                            gantt_rows.append({
                                "bus": a.bus_id,
                                "activity": "service trip",
                                "start_time": a.ride.start_time.time(),
                                "end_time": a.ride.end_time.time(),
                                "start_dt": a.ride.start_time,
                                "end_dt": a.ride.end_time,
                                "start_location": a.ride.start_stop,
                                "end_location": a.ride.end_stop,
                                "line": a.ride.line,
                                "energy_consumption": energy_service,
                            })
                        
                        df_gantt = pd.DataFrame(gantt_rows)
                        
                        # -----------------------------
                        # 7. CHECKS / WARNINGS
                        # -----------------------------
                        st.header("Planning Results")
                        
                        # Markeer ritten waar batterij na afloop < 0% usable capacity
                        # (dit is echt onmogelijk → planner kon geen geldige oplossing vinden)
                        def battery_after_pct_for_row(row):
                            # vind bijhorende assignment
                            for a in assignments:
                                if (
                                    a.bus_id == row["Bus_ID"] and
                                    a.ride.start_time.strftime("%H:%M") == row["Start_Time"] and
                                    a.ride.end_time.strftime("%H:%M") == row["End_Time"]
                                ):
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
                        
                        # KPI blokje
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Rides", len(df_planning))
                        with col2:
                            st.metric("Buses Used", df_planning['Bus_ID'].nunique())
                        with col3:
                            min_battery_after = df_planning["Battery_After_[%usable]_calc"].min()
                            st.metric(
                                "Minimum Battery After Ride",
                                f"{min_battery_after:.1f}%",
                                delta="OK" if min_battery_after >= st.session_state.minbat else "CRITICAL",
                                delta_color="normal" if min_battery_after >= st.session_state.minbat else "inverse"
                            )
                        
                        # -----------------------------
                        # 8. SHOW TABLE + GANTT
                        # -----------------------------
                        st.subheader("Generated Planning (per ride)")
                        st.dataframe(df_planning, use_container_width=True)
                        
                        st.subheader("Gantt Chart of Generated Bus Plan")
                        # create_gannt_chart verwacht kolommen bus/activity/start_time/end_time/... etc.
                        fm.create_gannt_chart(df_gantt.rename(columns={
                            "bus": "bus",
                            "activity": "activity",
                            "start_time": "start_time",
                            "end_time": "end_time",
                            "start_location": "start_location",
                            "end_location": "end_location",
                            "line": "line",
                            "energy_consumption": "energy_consumption",
                        }))
                        
                        # -----------------------------
                        # 9. DOWNLOAD KNOP
                        # -----------------------------
                        st.subheader("Download Planning")
                        output = io.BytesIO()
                        with pd.ExcelWriter(output, engine='openpyxl') as writer:
                            df_planning.to_excel(writer, sheet_name='Schedule', index=False)
                            df_gantt.to_excel(writer, sheet_name='DetailedTimeline', index=False)
                        output.seek(0)
                        
                        st.download_button(
                            label="📥 Download Planning as Excel",
                            data=output,
                            file_name="generated_bus_planning.xlsx",
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
        "Driving usage (kW/km) [Please round to 1 decimal point]",
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

