## Formulas

# Imports
import pandas as pd
import numpy as np
import streamlit as st
import scipy.stats as sp
import datetime
import plotly.express as px

# Functions

def amount_of_busses(planning):
    planning = pd.DataFrame(planning)
    busses = planning[planning.columns[len(planning.columns) - 1]].unique()
    return busses

def length_activities(active_ride):
    active_ride[active_ride.columns[3]] = pd.to_datetime(active_ride.iloc[:,3], format = "%H:%M:%S")
    active_ride[active_ride.columns[4]] = pd.to_datetime(active_ride.iloc[:,4], format = "%H:%M:%S")
    active_ride["diff"] = active_ride[active_ride.columns[4]] - active_ride[active_ride.columns[3]]
    return active_ride
