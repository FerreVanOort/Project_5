## Main tool

# Imports
from fileinput import filename
import Formulas as fm
import PlanningMaker as pm
import io
from datetime import datetime, timedelta
import pandas as pd
import streamlit as st
import base64
from pathlib import Path

# Streamlit settings
st.set_page_config(page_title="Prototype groep 8", layout="wide")

# Sidebar navigation
st.sidebar.header("Menu")
page = st.sidebar.radio(
    "Go to page:",
    ["Planning Checker", "Advanced Options", "User Manual", "About Us"],
    label_visibility="collapsed"
)
# , "Planning Maker" TERUG TOEVOEGEN VOOR PLANNING MAKER DEADLINE

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
IMG_DIR = BASE_DIR / "images"  # map waar je afbeeldingen staan


def team_member(filename: str, name: str, linkedin_url: str):
    """Bouwt een HTML-kaartje met foto + naam + LinkedIn."""
    p = IMG_DIR / filename
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

            st.success("All files successfully loaded!")

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
                driving_usage=st.session_state.driving_usage,
                idle_usage=st.session_state.idle_usage,
                charging_speed=st.session_state.charging_speed
            )

            # --- State of Charge check ---
            st.header("State of Charge check")
            fm.SOC_check(
                planning_energy,
                SOH=st.session_state.soh,
                minbat=st.session_state.minbat,
                startbat=st.session_state.startbat
            )

            # --- Gantt Chart ---
            st.header("Gantt Chart of uploaded Bus Plan")

            

            fm.create_gannt_chart(planning_energy)

        except Exception as e:
            st.error(f"Something went wrong with the processing of files: {e}")
    else:
        st.info("Upload a bus plan, timetable, and distance matrix to start!")


# -------------------------------------------------
# Page 2 - Planning Maker
# -------------------------------------------------
elif page == "Planning Maker":
    st.title("Prototype Group 8 - Bus Planning Maker", anchor='group 8')
    st.info("⚠️ Planning Maker niet relevant voor de huidige check — deze sectie kan ongewijzigd blijven.")


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
    
    with open("usermanual.pdf", "rb") as file:
        pdf_data = file.read()
        
    st.download_button(
        label = "Download User Manual",
        data = pdf_data,
        file_name = "usermanual.pdf",
        mime = "application/pdf"
    )


# -------------------------------------------------
# Page 5 - About Us
# -------------------------------------------------
elif page == "About Us":
    st.title("About Us", anchor='group 8')
    st.header('The team')
    st.write("""
We are a team of enthusiastic students - Ferre, Mirthe and Fea - from Eindhoven, studying Applied Mathematics at Fontys University of Applied Sciences.
Ferre takes the lead in dividing tasks and ensuring everything runs smoothly.
Ferre also focuses on coding the Planning Maker, while Fea is responsible for coding the Planning Checker.
Mirthe is responsible for the Streamlit interface.
Together, we bring our unique skills and perspectives to create innovative solutions in the field of applied mathematics.
""")

    st.header('Project Planning Checker and Maker for Electric Bus Fleets')
    st.write("""
The PlanningChecker verifies whether your bus schedule is complete and accurate.
With the growing shift toward electric buses, scheduling now involves stricter requirements.
PlanningChecker ensures that each bus plan complies with these modern standards by checking routes, charging times, and State of Charge limits.
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
