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
# Helper functie voor About Us kaartjes
# -------------------------------------------------
BASE_DIR = Path(__file__).parent
IMG_DIR = BASE_DIR / "images"   # map waar je afbeeldingen staan

def team_member(filename: str, name: str, linkedin_url: str):
    """
    Bouwt een HTML-kaartje met foto + naam + LinkedIn.
    Verwacht dat er in ./images/ een bestand staat met de gegeven filename.
    """
    p = IMG_DIR / filename  # bv. images/fea.jpg
    try:
        img_bytes = p.read_bytes()
    except Exception as e:
        st.error(f"Kon afbeelding niet laden: {p} ({e})")
        return ""

    # Kies juiste MIME-type
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
                st.info("**Battery Settings**")
                st.write(f"Battery Capacity: {pm.BusConstants.BATTERY_CAPACITY} kWh")
                st.write(f"Min Battery: {pm.BusConstants.MIN_BATTERY_PERCENT*100}%")
                st.write(f"Consumption: {pm.BusConstants.CONSUMPTION_PER_KM} kW/km")
            
            # --- Generate Planning Button ---
            if st.button("Generate Bus Planning", type="primary"):
                with st.spinner("Generating optimal bus planning..."):
                    
                    # Step 1: Load distance matrix
                    st.write("📊 Loading distance matrix...")
                    loader = pm.DataLoader()
                    distance_dict, time_dict, distance_df = loader.load_distance_matrix_from_df(distancematrix_build)
                    
                    # Step 2: Load timetable
                    st.write("📅 Loading timetable...")
                    rides = loader.load_timetable_from_df(timetable_build, distance_df)
                    
                    if not rides:
                        st.error("❌ No rides could be loaded from timetable!")
                    else:
                        st.success(f"✅ Loaded {len(rides)} rides")
                        
                        # Step 3: Create planning objects
                        st.write("🔧 Initializing planning system...")
                        distance_matrix = pm.DistanceMatrix(distance_dict, time_dict)
                        charging_planner = pm.ChargingPlanner(charging_station)
                        scheduler = pm.BusScheduler(distance_matrix, charging_planner, garage_location)
                        
                        # Step 4: Create initial bus fleet
                        initial_buses = [
                            pm.Bus(
                                "BUS_1",
                                garage_location,
                                pm.BusConstants.BATTERY_CAPACITY, 
                                rides[0].start_time - timedelta(hours=1)
                            )
                        ]
                        
                        # Step 5: Schedule all rides
                        st.write("🚌 Scheduling rides...")
                        assignments = scheduler.schedule_all_rides(rides, initial_buses)
                        
                        st.success(f"✅ Planning complete! {len(assignments)} rides scheduled")
                        
                        # --- Create output dataframe for display ---
                        schedule_data = []
                        for assignment in assignments:
                            row = {
                                'Bus_ID': assignment.bus_id,
                                'Start_Location': assignment.ride.start_stop,
                                'End_Location': assignment.ride.end_stop,
                                'Start_Time': assignment.ride.start_time,
                                'End_Time': assignment.ride.end_time,
                                'Battery_Charge_After_Ride': round(
                                    assignment.battery_after / pm.BusConstants.BATTERY_CAPACITY * 100, 1
                                ),
                            }
                            schedule_data.append(row)
                        
                        df_planning = pd.DataFrame(schedule_data)
                        
                        # --- Create detailed dataframe for Gantt chart ---
                        gantt_data = []
                        for assignment in assignments:
                            # Add charging activity if present
                            if assignment.charging_before:
                                arrival, departure, before, after = assignment.charging_before
                                gantt_data.append({
                                    'bus': assignment.bus_id,
                                    'activity': 'charging',
                                    'start_time': arrival.time(),
                                    'end_time': departure.time(),
                                    'start_dt': arrival,
                                    'end_dt': departure
                                })
                            
                            # Add deadhead trip if present
                            if assignment.deadhead_before:
                                from_loc, to_loc, dist, dep, arr = assignment.deadhead_before
                                gantt_data.append({
                                    'bus': assignment.bus_id,
                                    'activity': 'material trip',
                                    'start_time': dep.time(),
                                    'end_time': arr.time(),
                                    'start_dt': dep,
                                    'end_dt': arr
                                })
                            
                            # Add service trip (the actual ride)
                            gantt_data.append({
                                'bus': assignment.bus_id,
                                'activity': 'service trip',
                                'start_time': assignment.ride.start_time.time(),
                                'end_time': assignment.ride.end_time.time(),
                                'start_dt': assignment.ride.start_time,
                                'end_dt': assignment.ride.end_time
                            })
                        
                        df_gantt = pd.DataFrame(gantt_data)
                        
                        # --- DEBUG: Check for negative batteries ---
                        st.header("Planning Results")
                        
                        negative_batteries = df_planning[df_planning['Battery_Charge_After_Ride'] < 0]
                        if not negative_batteries.empty:
                            st.error(f"⚠️ WARNING: {len(negative_batteries)} rides have negative battery!")
                            
                            with st.expander("Show rides with negative battery"):
                                st.dataframe(negative_batteries)
                            
                            # Show detailed info for first problematic ride
                            st.subheader("Debug Info: First Problematic Ride")
                            problem_idx = negative_batteries.index[0]
                            problem_assignment = assignments[problem_idx]
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write(f"**Bus:** {problem_assignment.bus_id}")
                                st.write(f"**Route:** {problem_assignment.ride.start_stop} → {problem_assignment.ride.end_stop}")
                                st.write(
                                    f"**Time:** {problem_assignment.ride.start_time.strftime('%H:%M')} → "
                                    f"{problem_assignment.ride.end_time.strftime('%H:%M')}"
                                )
                                st.write(f"**Distance:** {problem_assignment.ride.distance_km:.2f} km")
                            
                            with col2:
                                st.write(
                                    f"**Battery before:** {problem_assignment.battery_before:.2f} kWh "
                                    f"({problem_assignment.battery_before/pm.BusConstants.BATTERY_CAPACITY*100:.1f}%)"
                                )
                                st.write(
                                    f"**Battery after:** {problem_assignment.battery_after:.2f} kWh "
                                    f"({problem_assignment.battery_after/pm.BusConstants.BATTERY_CAPACITY*100:.1f}%)"
                                )
                                st.write(
                                    f"**Energy needed:** "
                                    f"{problem_assignment.ride.distance_km * pm.BusConstants.CONSUMPTION_PER_KM:.2f} kWh"
                                )
                            
                            if problem_assignment.charging_before:
                                st.info("**Charging session detected:**")
                                arrival, departure, before, after = problem_assignment.charging_before
                                st.write(f"- Arrival at charger: {arrival.strftime('%H:%M')}")
                                st.write(f"- Departure from charger: {departure.strftime('%H:%M')}")
                                st.write(f"- Charged from {before:.2f} kWh to {after:.2f} kWh")
                                st.write(
                                    f"- Charging duration: {(departure-arrival).total_seconds()/60:.1f} minutes"
                                )
                            else:
                                st.warning("**No charging session before this ride**")
                            
                            if problem_assignment.deadhead_before:
                                from_loc, to_loc, dist, dep, arr = problem_assignment.deadhead_before
                                st.info("**Deadhead trip detected:**")
                                st.write(f"- Route: {from_loc} → {to_loc}")
                                st.write(f"- Distance: {dist:.2f} km")
                                st.write(
                                    f"- Energy used: {dist * pm.BusConstants.CONSUMPTION_PER_KM:.2f} kWh"
                                )
                            else:
                                st.info("**No deadhead trip before this ride**")
                        
                        # --- Show statistics ---
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Rides", len(df_planning))
                        with col2:
                            st.metric("Buses Used", df_planning['Bus_ID'].nunique())
                        with col3:
                            min_battery = df_planning['Battery_Charge_After_Ride'].min()
                            st.metric(
                                "Minimum Battery",
                                f"{min_battery}%", 
                                delta="OK" if min_battery >= 10 else "CRITICAL",
                                delta_color="normal" if min_battery >= 10 else "inverse"
                            )
                        
                        # --- Show planning table ---
                        st.subheader("Generated Planning")
                        st.dataframe(df_planning, use_container_width=True)
                        
                        # --- Gantt Chart ---
                        st.subheader("Gantt Chart of Generated Bus Plan")
                        fm.create_gannt_chart(df_gantt)
                        
                        # --- Download button ---
                        st.subheader("Download Planning")
                        output = io.BytesIO()
                        with pd.ExcelWriter(output, engine='openpyxl') as writer:
                            df_planning.to_excel(writer, sheet_name='Schedule', index=False)
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

