## Eigen planning

# Imports
import pandas as pd
from datetime import timedelta
from datetime import datetime
from datetime import time

# Formules / Functies
def can_assign_ride(bus_state: dict, ride: pd.Series, next_ride_start: datetime,
                    min_battery_pct: float, battery_capacity: float,
                    driving_usage: float, idle_usage: float, charging_speed: float) -> bool:
    """
    Checks if a bus can take the next ride given time, location, and battery constraints.
    
    Inputs:
        bus_state: dict with keys 'end_time', 'location', 'battery'
        ride: the next ride as a Pandas Series
        next_ride_start: start time of the next ride
        min_battery_pct: minimum battery fraction (0-1)
        battery_capacity: maximum battery capacity in kWh
        driving_usage: energy consumption in kWh/km
        idle_usage: energy consumption while idle in kWh/h
        charging_speed: charging rate in kW
        
    Returns:
        True if the bus can take the ride, False otherwise
    """
    # Time constraint
    if bus_state['end_time'] > next_ride_start:
        return False
    
    # Location constraint
    if bus_state['location'] != ride['start']:
        return False
    
    # Battery constraint
    energy_needed = calculate_ride_energy(ride, driving_usage)
    
    # Idle energy during waiting
    wait_hours = (next_ride_start - bus_state['end_time']).total_seconds() / 3600
    idle_energy = wait_hours * idle_usage
    
    if bus_state['battery'] - idle_energy - energy_needed < min_battery_pct * battery_capacity:
        # Check if bus can charge enough
        energy_gap = battery_capacity - bus_state['battery']
        charging_time_needed = energy_gap / charging_speed
        if charging_time_needed > wait_hours:
            return False
    
    return True

def calculate_ride_energy(ride: pd.Series, driving_usage: float) -> float:
    """
    Calculate energy consumption for a ride in kWh
    
    Inputs:
        ride: Pandas Series with 'distance_m'
        driving_usage: kWh/km
        
    Returns:
        Energy required in kWh
    """
    distance_km = ride['distance_m'] / 1000
    return distance_km * driving_usage

def create_ride_entry(bus_id: str, ride: pd.Series) -> dict:
    """
    Create a dictionary entry for a ride in the schedule
    
    Inputs:
        bus_id: identifier for the bus
        ride: Pandas Series with ride info
        
    Returns:
        Dictionary representing the ride
    """
    return {
        'bus': bus_id,
        'activity': 'driving',
        'start_time': ride['departure_time'],
        'end_time': ride['arrival_time'],
        'start_location': ride['start'],
        'end_location': ride['end'],
        'line': ride['line'],
        'energy_consumption': calculate_ride_energy(ride, driving_usage=1)  # placeholder
    }

def create_idle_entry(bus_id: str, bus_state: dict, next_ride: pd.Series, idle_usage: float) -> dict:
    """
    Create a dictionary entry for an idle period
    
    Inputs:
        bus_id: identifier for the bus
        bus_state: dict with 'end_time', 'location', 'battery'
        next_ride: next ride as Pandas Series
        idle_usage: energy consumption in kWh/hour
        
    Returns:
        Dictionary representing idle period
    """
    idle_hours = (next_ride['departure_time'] - bus_state['end_time']).total_seconds() / 3600
    bus_state['battery'] -= idle_hours * idle_usage
    
    return {
        'bus': bus_id,
        'activity': 'idle',
        'start_time': bus_state['end_time'],
        'end_time': next_ride['departure_time'],
        'start_location': bus_state['location'],
        'end_location': bus_state['location'],
        'line': None,
        'energy_consumption': idle_hours * idle_usage
    }

def create_charging_entry(bus_id: str, bus_state: dict, next_ride: pd.Series,
                          battery_capacity: float, charging_speed: float, min_battery_pct: float) -> dict:
    """
    Create a dictionary entry for a charging period
    
    Inputs:
        bus_id: identifier for the bus
        bus_state: dict with 'end_time', 'location', 'battery'
        next_ride: next ride as Pandas Series
        battery_capacity: max battery capacity in kWh
        charging_speed: charging rate in kW
        min_battery_pct: minimum battery fraction to maintain (0-1)
        
    Returns:
        Dictionary representing the charging period
    """
    available_hours = (next_ride['departure_time'] - bus_state['end_time']).total_seconds() / 3600
    max_chargeable = charging_speed * available_hours
    energy_gap = battery_capacity - bus_state['battery']
    energy_to_charge = min(max_chargeable, energy_gap)
    
    charging_duration = energy_to_charge / charging_speed
    start_time = bus_state['end_time']
    end_time = start_time + timedelta(hours=charging_duration)
    
    bus_state['battery'] += energy_to_charge
    bus_state['end_time'] = end_time
    
    return {
        'bus': bus_id,
        'activity': 'charging',
        'start_time': start_time,
        'end_time': end_time,
        'start_location': bus_state['location'],
        'end_location': bus_state['location'],
        'line': None,
        'energy_consumption': -energy_to_charge  # negative because battery is increasing
    }



def create_bus_schedule(timetable: pd.DataFrame, 
                        distancematrix: pd.DataFrame,
                        battery_capacity: float,
                        min_battery_pct: float,
                        driving_usage: float,
                        idle_usage: float,
                        charging_speed: float,
                        depot_location: str = "Depot",
                        start_battery_pct: float = 1.0):
    """
    Generate a bus schedule based on timetable, distances, and battery constraints.
    
    Inputs:
        timetable: DataFrame with rides
        distancematrix: DataFrame with travel times and distances
        battery_capacity: Total battery capacity in kWh
        min_battery_pct: Minimum battery fraction before charging (0-1)
        driving_usage: Energy consumption in kWh/km
        idle_usage: Energy consumption while idle in kWh/hour
        charging_speed: Charging rate in kW
        depot_location: Starting location for new buses
        start_battery_pct: Starting battery level fraction (0-1)
    
    Output:
        DataFrame with columns: bus, activity, start_time, end_time,
                                start_location, end_location, line, energy_consumption
    """
    
    timetable = timetable.copy()
    distancematrix = distancematrix.copy()
    
    # Clean column names
    timetable.columns = [str(c).strip().replace("\u00A0","").replace(" ","_").lower() for c in timetable.columns]
    distancematrix.columns = [str(c).strip().replace("\u00A0","").replace(" ","_").lower() for c in distancematrix.columns]
        
    last_dt = pd.Timestamp("2025-01-01")
    departure_times = []
    
    timetable = timetable.sort_values('departure_time').reset_index(drop = True)

    def assign_datetime(ride_time: time, last_dt: datetime):
        """
        Converts a ride_time (datetime.time) to datetime after the last ride datetime.
        - If ride_time < last_dt.time(), assume it's the next day.
        """
        # If string, convert to datetime.time
        if isinstance(ride_time, str):
            try:
                ride_time = datetime.strptime(ride_time, "%H:%M:%S").time()
            except ValueError:
                ride_time = datetime.strptime(ride_time, "%H:%M").time()
        
        candidate_dt = datetime.combine(last_dt.date(), ride_time)
        if candidate_dt <= last_dt:
            candidate_dt += timedelta(days=1)
        return candidate_dt
    
    # Apply to departure times
    for t in timetable['departure_time']:
        full_dt = assign_datetime(t, last_dt)
        departure_times.append(full_dt)
        last_dt = full_dt  # update last_dt for next ride

    timetable['departure_time'] = departure_times
    
    # Ensure departure_time is datetime
    if timetable['departure_time'].dtype != 'datetime64[ns]':
        timetable['departure_time'] = pd.to_datetime(timetable['departure_time']).apply(lambda t: datetime.combine(last_dt, t.time()))
    
    # Merge travel info
    timetable = timetable.merge(
        distancematrix[['start', 'end', 'line', 'min_travel_time', 'distance_m']],
        left_on=['start', 'end', 'line'],
        right_on=['start', 'end', 'line'],
        how='left'
    )
    
    # Calculate arrival times
    timetable['arrival_time'] = timetable.apply(
    lambda row: row['departure_time'] + timedelta(minutes=row['min_travel_time']),
    axis=1
    )
    
    timetable = timetable.sort_values('departure_time').reset_index(drop=True)
    
    # Track buses
    buses = {}  # bus_id -> {'end_time', 'location', 'battery'}
    schedule = []
    bus_counter = 1
    
    for _, ride in timetable.iterrows():
        assigned = False
        
        for bus_id, bus_state in buses.items():
            if can_assign_ride(bus_state, ride, ride['departure_time'], min_battery_pct,
                             battery_capacity, driving_usage, idle_usage, charging_speed):
                
                # Add idle if needed
                if bus_state['end_time'] < ride['departure_time']:
                    idle_entry = create_idle_entry(bus_id, bus_state, ride, idle_usage)
                    schedule.append(idle_entry)
                    bus_state['end_time'] = ride['departure_time']
                
                # Add charging if needed
                energy_needed = calculate_ride_energy(ride, driving_usage)
                if bus_state['battery'] - energy_needed < min_battery_pct * battery_capacity:
                    charge_entry = create_charging_entry(bus_id, bus_state, ride, battery_capacity, charging_speed, min_battery_pct)
                    schedule.append(charge_entry)
                
                # Add driving
                schedule.append(create_ride_entry(bus_id, ride))
                
                # Update bus state
                buses[bus_id]['end_time'] = ride['arrival_time']
                buses[bus_id]['location'] = ride['end']
                buses[bus_id]['battery'] -= energy_needed
                
                assigned = True
                break
        
        # Create new bus if not assigned
        if not assigned:
            bus_id = f"Bus_{bus_counter}"
            bus_counter += 1
            
            # Add driving entry
            schedule.append(create_ride_entry(bus_id, ride))
            
            buses[bus_id] = {
                'end_time': ride['arrival_time'],
                'location': ride['end'],
                'battery': start_battery_pct * battery_capacity - calculate_ride_energy(ride, driving_usage)
            }
    
    return pd.DataFrame(schedule)
