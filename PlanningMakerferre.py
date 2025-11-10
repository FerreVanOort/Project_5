import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
from datetime import datetime, timedelta

# =====================================================================
# Bus constants
# =====================================================================
@dataclass
class BusConstants:
    ORIGINAL_BATTERY_CAPACITY = 300  # kWh
    AGING_FACTOR = 0.90
    BATTERY_CAPACITY = ORIGINAL_BATTERY_CAPACITY * AGING_FACTOR
    CONSUMPTION_PER_KM = 1.2  # kWh/km
    IDLE_CONSUMPTION_PER_HOUR = 5
    MIN_BATTERY_PERCENT = 0.10
    MIN_CHARGE_TIME = 15  # minutes
    FAST_CHARGE_RATE = 450
    SLOW_CHARGE_RATE = 60
    FAST_CHARGE_THRESHOLD = 0.90

# =====================================================================
# Rides, Buses, Events
# =====================================================================
@dataclass
class Ride:
    ride_id: str
    start_stop: str
    end_stop: str
    start_time: datetime
    end_time: datetime
    distance_meters: float
    line: str = ""

    @property
    def distance_km(self) -> float:
        return self.distance_meters / 1000

@dataclass
class Bus:
    bus_id: str
    current_location: str
    current_battery_kwh: float
    available_from: datetime

    @property
    def battery_percent(self) -> float:
        return self.current_battery_kwh / BusConstants.BATTERY_CAPACITY

@dataclass
class Event:
    event_type: str  # ride, charging, deadhead, idle
    bus_id: str
    start_time: datetime
    end_time: datetime
    start_location: str
    end_location: str
    battery_before: float
    battery_after: float
    distance_km: float = 0
    energy_consumed: float = 0
    ride_id: Optional[str] = None
    line: str = ""

    @property
    def duration_minutes(self) -> float:
        return (self.end_time - self.start_time).total_seconds() / 60

@dataclass
class Assignment:
    bus_id: str
    ride: Ride
    events: List[Event] = field(default_factory=list)

    @property
    def battery_after_ride(self) -> float:
        ride_event = next((e for e in self.events if e.event_type == 'ride'), None)
        return ride_event.battery_after if ride_event else 0

# =====================================================================
# Distance Matrix
# =====================================================================
class DistanceMatrix:
    def __init__(self, distance_dict: Dict[Tuple[str, str], float],
                 time_dict: Dict[Tuple[str, str], float]):
        self.distance_dict = distance_dict
        self.time_dict = time_dict

    def get_distance_km(self, from_stop: str, to_stop: str) -> float:
        if from_stop == to_stop:
            return 0.0
        return self.distance_dict.get((from_stop, to_stop), 10000) / 1000

    def get_travel_time_minutes(self, from_stop: str, to_stop: str) -> float:
        if from_stop == to_stop:
            return 0.0
        return self.time_dict.get((from_stop, to_stop), self.get_distance_km(from_stop, to_stop)/30*60)

    def get_energy_for_deadhead(self, from_stop: str, to_stop: str, consumption_per_km: float = None) -> float:
        if consumption_per_km is None:
            consumption_per_km = BusConstants.CONSUMPTION_PER_KM
        return calculate_energy_consumption(self.get_distance_km(from_stop, to_stop), consumption_per_km)

# =====================================================================
# Energy calculations
# =====================================================================
def calculate_energy_consumption(distance_km: float, consumption_per_km: float = None) -> float:
    if consumption_per_km is None:
        consumption_per_km = BusConstants.CONSUMPTION_PER_KM
    return distance_km * consumption_per_km

def calculate_idle_consumption(minutes: float, idle_per_hour: float = None) -> float:
    if idle_per_hour is None:
        idle_per_hour = BusConstants.IDLE_CONSUMPTION_PER_HOUR
    return (minutes/60) * idle_per_hour

def calculate_charging_time(current_kwh: float, target_kwh: float) -> float:
    threshold = BusConstants.BATTERY_CAPACITY * BusConstants.FAST_CHARGE_THRESHOLD
    total_minutes = 0
    if current_kwh < threshold:
        fast_amount = min(target_kwh, threshold) - current_kwh
        total_minutes += fast_amount / BusConstants.FAST_CHARGE_RATE * 60
    if target_kwh > threshold:
        slow_amount = target_kwh - max(current_kwh, threshold)
        total_minutes += slow_amount / BusConstants.SLOW_CHARGE_RATE * 60
    return max(total_minutes, BusConstants.MIN_CHARGE_TIME)

# =====================================================================
# Charging Planner
# =====================================================================
class ChargingPlanner:
    def __init__(self, charging_station: str):
        self.charging_station = charging_station

    def plan_charging_session(self, bus: Bus, target_kwh: float, available_minutes: float) -> Tuple[float, float]:
        required_minutes = calculate_charging_time(bus.current_battery_kwh, target_kwh)
        if required_minutes <= available_minutes:
            return target_kwh, required_minutes
        # partial charging
        charged = bus.current_battery_kwh + (available_minutes/60)*(BusConstants.FAST_CHARGE_RATE if bus.current_battery_kwh < BusConstants.BATTERY_CAPACITY*0.9 else BusConstants.SLOW_CHARGE_RATE)
        return min(charged, BusConstants.BATTERY_CAPACITY), available_minutes

# =====================================================================
# Bus Scheduler
# =====================================================================
class BusScheduler:
    def __init__(self, distance_matrix: DistanceMatrix, charging_planner: ChargingPlanner, garage_location: str,
                 consumption_per_km: float = None, idle_per_hour: float = None):
        self.distance_matrix = distance_matrix
        self.charging_planner = charging_planner
        self.garage_location = garage_location
        self.consumption_per_km = consumption_per_km or BusConstants.CONSUMPTION_PER_KM
        self.idle_per_hour = idle_per_hour or BusConstants.IDLE_CONSUMPTION_PER_HOUR

    def can_bus_serve_ride(self, bus: Bus, ride: Ride) -> bool:
        travel_time = self.distance_matrix.get_travel_time_minutes(bus.current_location, ride.start_stop)
        arrival = bus.available_from + timedelta(minutes=travel_time)
        return arrival <= ride.start_time

    def assign_ride_to_bus(self, bus: Bus, ride: Ride) -> Assignment:
        assignment = Assignment(bus.bus_id, ride)
        current_time = bus.available_from
        current_battery = bus.current_battery_kwh
        current_location = bus.current_location

        # DEADHEAD
        if current_location != ride.start_stop:
            deadhead_distance = self.distance_matrix.get_distance_km(current_location, ride.start_stop)
            deadhead_time = self.distance_matrix.get_travel_time_minutes(current_location, ride.start_stop)
            deadhead_energy = calculate_energy_consumption(deadhead_distance, self.consumption_per_km)
            start_deadhead = current_time
            end_deadhead = start_deadhead + timedelta(minutes=deadhead_time)
            assignment.events.append(Event(
                'deadhead', bus.bus_id, start_deadhead, end_deadhead,
                current_location, ride.start_stop, current_battery, current_battery-deadhead_energy,
                distance_km=deadhead_distance, energy_consumed=deadhead_energy
            ))
            current_time = end_deadhead
            current_battery -= deadhead_energy
            current_location = ride.start_stop

        # IDLE before ride
        if current_time < ride.start_time:
            idle_minutes = (ride.start_time - current_time).total_seconds()/60
            idle_energy = calculate_idle_consumption(idle_minutes, self.idle_per_hour)
            assignment.events.append(Event(
                'idle', bus.bus_id, current_time, ride.start_time,
                current_location, current_location, current_battery, current_battery-idle_energy,
                distance_km=0, energy_consumed=idle_energy
            ))
            current_battery -= idle_energy
            current_time = ride.start_time

        # RIDE
        ride_energy = calculate_energy_consumption(ride.distance_km, self.consumption_per_km)
        assignment.events.append(Event(
            'ride', bus.bus_id, ride.start_time, ride.end_time,
            ride.start_stop, ride.end_stop, current_battery, current_battery-ride_energy,
            distance_km=ride.distance_km, energy_consumed=ride_energy,
            ride_id=ride.ride_id, line=ride.line
        ))
        current_battery -= ride_energy
        current_time = ride.end_time
        current_location = ride.end_stop

        # CHARGING check after ride if battery < 30%
        if current_battery/BusConstants.BATTERY_CAPACITY < 0.3:
            charger_loc = self.charging_planner.charging_station
            travel_time_to_charger = self.distance_matrix.get_travel_time_minutes(current_location, charger_loc)
            energy_to_charger = self.distance_matrix.get_energy_for_deadhead(current_location, charger_loc, self.consumption_per_km)
            arrival_at_charger = current_time + timedelta(minutes=travel_time_to_charger)
            current_battery -= energy_to_charger
            current_time = arrival_at_charger
            current_location = charger_loc
            target_kwh = BusConstants.BATTERY_CAPACITY*0.8
            charged_to, charge_minutes = self.charging_planner.plan_charging_session(
                Bus(bus.bus_id, current_location, current_battery, current_time), target_kwh, 120
            )
            assignment.events.append(Event(
                'charging', bus.bus_id, current_time, current_time+timedelta(minutes=charge_minutes),
                current_location, current_location, current_battery, charged_to,
                distance_km=0, energy_consumed=-(charged_to-current_battery)
            ))
            current_battery = charged_to
            current_time += timedelta(minutes=charge_minutes)

        return assignment

    def schedule_all_rides(self, rides: List[Ride], buses: List[Bus]) -> List[Assignment]:
        assignments = []
        for ride in sorted(rides, key=lambda r: r.start_time):
            best_bus = None
            for bus in buses:
                if self.can_bus_serve_ride(bus, ride):
                    best_bus = bus
                    break
            if best_bus is None:
                # create new bus
                best_bus = Bus(f"BUS_{len(buses)+1}", self.garage_location, BusConstants.BATTERY_CAPACITY, ride.start_time-timedelta(minutes=30))
                buses.append(best_bus)
            assignment = self.assign_ride_to_bus(best_bus, ride)
            assignments.append(assignment)
            best_bus.current_battery_kwh = assignment.battery_after_ride
            best_bus.current_location = ride.end_stop
            best_bus.available_from = ride.end_time
        return assignments

# =====================================================================
# Export to DataFrame
# =====================================================================
def assignments_to_dataframe(assignments: List[Assignment]) -> pd.DataFrame:
    rows = []
    for assignment in assignments:
        for e in assignment.events:
            rows.append({
                'bus': e.bus_id,
                'activity': e.event_type,
                'start_time': e.start_time,
                'end_time': e.end_time,
                'start_location': e.start_location,
                'end_location': e.end_location,
                'distance_km': e.distance_km,
                'energy_consumption': round(e.energy_consumed,2),
                'line': e.line,
                'ride_id': e.ride_id
            })
    return pd.DataFrame(rows)

# =====================================================================
# Main function
# =====================================================================
def create_bus_planning(timetable_df: pd.DataFrame, distance_matrix_df: pd.DataFrame,
                        charging_station: str = "DEPOT", garage_location: str = "GARAGE",
                        driving_usage: float = 1.2, idle_usage: float = 5.0,
                        startbat: float = 100.0) -> pd.DataFrame:

    BusConstants.CONSUMPTION_PER_KM = driving_usage
    BusConstants.IDLE_CONSUMPTION_PER_HOUR = idle_usage
    BusConstants.BATTERY_CAPACITY = BusConstants.ORIGINAL_BATTERY_CAPACITY * BusConstants.AGING_FACTOR

    # Build distance/time dicts
    distance_dict = {}
    time_dict = {}
    for idx,row in distance_matrix_df.iterrows():
        start = str(row['start']).strip()
        end = str(row['end']).strip()
        distance_dict[(start,end)] = float(row['distance_m'])
        avg_time = (float(row['min_travel_time'])+float(row['max_travel_time']))/2
        time_dict[(start,end)] = avg_time

    distance_matrix = DistanceMatrix(distance_dict,time_dict)
    charging_planner = ChargingPlanner(charging_station)
    scheduler = BusScheduler(distance_matrix, charging_planner, garage_location, driving_usage, idle_usage)

    # Convert timetable to Ride objects including distance from matrix
    rides = []
    for idx,row in timetable_df.iterrows():
        start = str(row['start']).strip()
        end = str(row['end']).strip()
        line = str(row['line']).strip() if 'line' in row else ''
        key = (start,end)
        distance_m = distance_dict.get(key,5000)
        start_time = pd.to_datetime(row['departure_time'])
        avg_travel_time = time_dict.get(key,10)
        end_time = start_time + timedelta(minutes=avg_travel_time)
        ride_id = f"{line}_{start}_{start_time.strftime('%H%M')}_{idx}"
        rides.append(Ride(ride_id, start, end, start_time, end_time, distance_m, line))

    # Initial bus
    start_battery_kwh = BusConstants.BATTERY_CAPACITY * startbat/100
    initial_buses = [Bus("BUS_1", garage_location, start_battery_kwh, rides[0].start_time-timedelta(minutes=30))]

    assignments = scheduler.schedule_all_rides(rides, initial_buses)
    df = assignments_to_dataframe(assignments)
    return df
