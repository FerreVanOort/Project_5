# Improved planning

import pandas as pd
from datetime import datetime, timedelta

# Load in Excel files
timetable_df = pd.read_excel('Timetable.xlsx')
distance_df = pd.read_excel('DistanceMatrix.xlsx')

## Functions


# Function for energy constants
def energy_constants_usage():
    return {
        "min_energy_consumption": 0.7,
        "mid_energy_consumption": 1.2,
        "max_energy_consumption": 2.5,
        "stationary_consumption": 5
    }
# Min, mid, and max energy consumption in kWh/km and stationary in kWh

