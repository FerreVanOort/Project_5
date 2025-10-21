## Main tool

# Imports

import Formulas as fm
import PlanningMaker as pm
import io
from datetime import datetime
import pandas as pd
import streamlit as st

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
if "SOH" not in st.session_state:
    st.session_state.soh = 90.0
if "minbat" not in st.session_state:
    st.session_state.minbat = 10.0
if "startbat" not in st.session_state:
    st.session_state.startbat = 100.0
    
# Page 1 - Planning Checker
if page == "Planning Checker":
    st.title("Prototype Group 8 - Bus Planning Check")
    
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
            
            # --- Gannt Chart ---
            st.header("Gannt Chart of uploaded Bus Plan")
            fm.create_gannt_chart(planning_energy)
            
        except Exception as e:
            st.error(f"Something went wrong with the processing of files {e}")
        
    else:
        st.info("Upload a bus plan, timetable, and distance matrix to start!")
        

# Page 2 - Schedule Builder
elif page == "Planning Maker":
    st.title("Bus Planning Maker")
    st.write("Generate an optimized bus schedule from a timetable")
    
    st.subheader("Upload Timetable and Distance Matrix")
    uploaded_timetable_build = st.file_uploader("Upload timetable (.xlsx)", type=["xlsx"], key="time_build")
    uploaded_distances_build = st.file_uploader("Upload distance matrix (.xlsx)", type=["xlsx"], key="dist_build")
        

# Page 3 - Advanced Options
elif page == "Advanced Options":
    st.title("Advanced Options")
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


# Page 4 - User Manual
elif page == "User Manual":
    st.title("User Manual")


# Page 5 - About Us
elif page == "About Us":
    st.title("About Us")