import pandas as pd
from typing import List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta

# ====================================================================
# Constants and Helper Functions
# ====================================================================

@dataclass
class BusConstants:
    ORIGINAL_BATTERY_CAPACITY = 300  # kWh
    AGING_FACTOR = 0.90
    BATTERY_CAPACITY = ORIGINAL_BATTERY_CAPACITY * AGING_FACTOR
    CONSUMPTION_PER_KM = 1.2
    IDLE_CONSUMPTION_PER_HOUR = 5
    MIN_BATTERY_PERCENT = 0.10
    MIN_CHARGE_TIME = 15
    FAST_CHARGE_RATE = 450
    SLOW_CHARGE_RATE = 60
    FAST_CHARGE_THRESHOLD = 0.90  # 90%

def calculate_energy_consumption(distance_km: float, consumption_per_km: float = None) -> float:
    if consumption_per_km is None:
        consumption_per_km = BusConstants.CONSUMPTION_PER_KM
    return distance_km * consumption_per_km

def calculate_idle_consumption(minutes: float, idle_per_hour: float = None) -> float:
    if idle_per_hour is None:
        idle_per_hour = BusConstants.IDLE_CONSUMPTION_PER_HOUR
    return (minutes / 60) * idle_per_hour

def calculate_charging_time(current_kwh: float, target_kwh: float,
                            fast_rate: float = None, slow_rate: float = None) -> float:
    if fast_rate is None:
        fast_rate = BusConstants.FAST_CHARGE_RATE
    if slow_rate is None:
        slow_rate = BusConstants.SLOW_CHARGE_RATE
    if target_kwh <= current_kwh:
        return 0
    threshold_kwh = BusConstants.BATTERY_CAPACITY * BusConstants.FAST_CHARGE_THRESHOLD
    total_minutes = 0
    if current_kwh < threshold_kwh:
        fast_amount = min(target_kwh, threshold_kwh) - current_kwh
        total_minutes += fast_amount / fast_rate * 60
    if target_kwh > threshold_kwh and current_kwh < target_kwh:
        slow_start = max(current_kwh, threshold_kwh)
        total_minutes += (target_kwh - slow_start) / slow_rate * 60
    return max(total_minutes, BusConstants.MIN_CHARGE_TIME)

# ====================================================================
# Data Classes
# ====================================================================

@dataclass
class Ride:
    ride_id: str
    start_stop: str
    end_stop: str
    start_time: datetime
    end_time: datetime
    distance_km: float
    line: str = ""

@dataclass
class Bus:
    bus_id: str
    current_location: str
    current_battery_kwh: float
    available_from: datetime

@dataclass
class Event:
    event_type: str
    bus_id: str
    start_time: datetime
    end_time: datetime
    start_location: str
    end_location: str
    battery_before: float
    battery_after: float
    distance_km: float = 0.0
    energy_consumed: float = 0.0
    ride_id: Optional[str] = None
    line: str = ""

@dataclass
class Assignment:
    bus_id: str
    ride: Ride
    events: List[Event] = field(default_factory=list)

# ====================================================================
# Distance Matrix
# ====================================================================

class DistanceMatrix:
    def __init__(self, distance_dict, time_dict):
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

    def get_energy_for_deadhead(self, from_stop: str, to_stop: str) -> float:
        return calculate_energy_consumption(self.get_distance_km(from_stop, to_stop))

# ====================================================================
# Charging Planner
# ====================================================================

class ChargingPlanner:
    def __init__(self, station_location: str):
        self.station = station_location

    def plan_charging(self, bus: Bus, target_kwh: float, available_minutes: float) -> Tuple[float, float]:
        ideal_minutes = calculate_charging_time(bus.current_battery_kwh, target_kwh)
        if ideal_minutes <= available_minutes:
            return target_kwh, ideal_minutes
        # partial charge
        proportion = available_minutes / ideal_minutes
        achieved_kwh = bus.current_battery_kwh + (target_kwh - bus.current_battery_kwh) * proportion
        return achieved_kwh, available_minutes

# ====================================================================
# Bus Scheduler
# ====================================================================

class BusScheduler:
    def __init__(self, distance_matrix: DistanceMatrix, charging_planner: ChargingPlanner, garage_location: str):
        self.distance_matrix = distance_matrix
        self.charging_planner = charging_planner
        self.garage_location = garage_location

    def assign_ride_to_bus(self, bus: Bus, ride: Ride) -> Assignment:
        assignment = Assignment(bus.bus_id, ride)
        current_time = bus.available_from
        current_battery = bus.current_battery_kwh
        current_location = bus.current_location

        # Energy needed
        deadhead_energy = self.distance_matrix.get_energy_for_deadhead(current_location, ride.start_stop)
        ride_energy = calculate_energy_consumption(ride.distance_km)
        total_needed = deadhead_energy + ride_energy
        safe_needed = total_needed * 1.3  # 30% buffer

        # NEED CHARGING
        if current_battery < safe_needed:
            # Deadhead to charger if not already there
            if current_location != self.charging_planner.station:
                dh_dist = self.distance_matrix.get_distance_km(current_location, self.charging_planner.station)
                dh_time = self.distance_matrix.get_travel_time_minutes(current_location, self.charging_planner.station)
                dh_energy = calculate_energy_consumption(dh_dist)
                event = Event("deadhead", bus.bus_id, current_time, current_time+timedelta(minutes=dh_time),
                              current_location, self.charging_planner.station,
                              current_battery, current_battery-dh_energy, dh_dist, dh_energy)
                assignment.events.append(event)
                current_time += timedelta(minutes=dh_time)
                current_battery -= dh_energy
                current_location = self.charging_planner.station

            # Charging
            energy_needed = min(BusConstants.BATTERY_CAPACITY, safe_needed)
            available_minutes = max(0, (ride.start_time - current_time).total_seconds()/60)
            charged_to, charge_duration = self.charging_planner.plan_charging(Bus(bus.bus_id, current_location, current_battery, current_time),
                                                                             energy_needed, available_minutes)
            charge_event = Event("charging", bus.bus_id, current_time, current_time+timedelta(minutes=charge_duration),
                                 current_location, current_location, current_battery, charged_to, 0, -(charged_to-current_battery))
            assignment.events.append(charge_event)
            current_time += timedelta(minutes=charge_duration)
            current_battery = charged_to

        # Deadhead to ride start
        if current_location != ride.start_stop:
            dh_dist = self.distance_matrix.get_distance_km(current_location, ride.start_stop)
            dh_time = self.distance_matrix.get_travel_time_minutes(current_location, ride.start_stop)
            dh_energy = calculate_energy_consumption(dh_dist)
            event = Event("deadhead", bus.bus_id, current_time, current_time+timedelta(minutes=dh_time),
                          current_location, ride.start_stop,
                          current_battery, current_battery-dh_energy, dh_dist, dh_energy)
            assignment.events.append(event)
            current_time += timedelta(minutes=dh_time)
            current_battery -= dh_energy
            current_location = ride.start_stop

        # Idle if early
        if current_time < ride.start_time:
            idle_minutes = (ride.start_time - current_time).total_seconds()/60
            idle_energy = calculate_idle_consumption(idle_minutes)
            event = Event("idle", bus.bus_id, current_time, ride.start_time,
                          current_location, current_location,
                          current_battery, current_battery-idle_energy, 0, idle_energy)
            assignment.events.append(event)
            current_battery -= idle_energy
            current_time = ride.start_time

        # Ride
        ride_event = Event("ride", bus.bus_id, ride.start_time, ride.end_time,
                           ride.start_stop, ride.end_stop,
                           current_battery, current_battery-ride_energy,
                           ride.distance_km, ride_energy, ride.ride_id, ride.line)
        assignment.events.append(ride_event)
        current_battery -= ride_energy
        current_time = ride.end_time
        current_location = ride.end_stop

        return assignment

    def schedule_all_rides(self, rides: List[Ride], initial_buses: List[Bus]) -> List[Assignment]:
        rides_sorted = sorted(rides, key=lambda r: r.start_time)
        assignments = []
        buses = [Bus(b.bus_id, b.current_location, b.current_battery_kwh, b.available_from) for b in initial_buses]

        for ride in rides_sorted:
            # Choose best bus (earliest available & enough battery)
            best_bus = None
            for bus in buses:
                if bus.available_from <= ride.start_time:
                    best_bus = bus
                    break
            if best_bus is None:
                # Add new bus
                new_bus = Bus(f"BUS_{len(buses)+1}", self.garage_location, BusConstants.BATTERY_CAPACITY, ride.start_time-timedelta(minutes=30))
                buses.append(new_bus)
                best_bus = new_bus

            assignment = self.assign_ride_to_bus(best_bus, ride)
            assignments.append(assignment)

            # Update bus state
            best_bus.current_location = ride.end_stop
            best_bus.current_battery_kwh = assignment.events[-1].battery_after
            best_bus.available_from = ride.end_time

        return assignments

# ====================================================================
# Export Planning to DataFrame
# ====================================================================

def assignments_to_dataframe(assignments: List[Assignment]) -> pd.DataFrame:
    all_events = []
    for assignment in assignments:
        for e in assignment.events:
            all_events.append({
                'start_location': e.start_location,
                'end_location': e.end_location,
                'start_time': e.start_time,
                'end_time': e.end_time,
                'activity': e.event_type,
                'line': e.line,
                'energy_consumption': round(e.energy_consumed,2),
                'bus': e.bus_id
            })
    df = pd.DataFrame(all_events)
    return df

# ====================================================================
# Main Function
# ====================================================================

def create_bus_planning(timetable_df: pd.DataFrame,
                        distance_matrix_df: pd.DataFrame,
                        charging_station: str = "DEPOT",
                        garage_location: str = "GARAGE",
                        driving_usage: float = 1.2,
                        idle_usage: float = 5.0,
                        charging_speed: float = 450.0,
                        soh: float = 90.0,
                        startbat: float = 100.0) -> pd.DataFrame:

    # Update constants
    BusConstants.CONSUMPTION_PER_KM = driving_usage
    BusConstants.IDLE_CONSUMPTION_PER_HOUR = idle_usage
    BusConstants.FAST_CHARGE_RATE = charging_speed
    BusConstants.AGING_FACTOR = soh / 100
    BusConstants.BATTERY_CAPACITY = BusConstants.ORIGINAL_BATTERY_CAPACITY * BusConstants.AGING_FACTOR

    # Build distance and time dicts
    distance_dict = {}
    time_dict = {}
    for idx,row in distance_matrix_df.iterrows():
        start, end = row['start'], row['end']
        distance_dict[(start,end)] = row['distance_m']
        time_dict[(start,end)] = (row['min_travel_time'] + row['max_travel_time'])/2

    # Load rides with distance
    rides = []
    for idx,row in timetable_df.iterrows():
        start, end, line = row['start'], row['end'], row.get('line','')
        departure = pd.to_datetime(row['departure_time'])
        # get distance from distance matrix
        dist = distance_dict.get((start,end), 5000)/1000
        duration = timedelta(minutes=time_dict.get((start,end), 10))
        rides.append(Ride(f"{line}_{start}_{departure.strftime('%H%M')}_{idx}", start, end, departure, departure+duration, dist, line))

    # Initial bus
    initial_buses = [Bus("BUS_1", garage_location, BusConstants.BATTERY_CAPACITY*(startbat/100), rides[0].start_time-timedelta(hours=1))]

    distance_matrix = DistanceMatrix(distance_dict, time_dict)
    charging_planner = ChargingPlanner(charging_station)
    scheduler = BusScheduler(distance_matrix, charging_planner, garage_location)

    assignments = scheduler.schedule_all_rides(rides, initial_buses)
    planning_df = assignments_to_dataframe(assignments)
    return planning_df
