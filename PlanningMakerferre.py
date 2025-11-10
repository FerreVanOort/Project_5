import pandas as pd
from typing import List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta


# ===================================================================
# Constants
# ===================================================================

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
    FAST_CHARGE_THRESHOLD = 0.90


# ===================================================================
# Data Classes
# ===================================================================

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
    def distance_km(self):
        return self.distance_meters / 1000

@dataclass
class Bus:
    bus_id: str
    current_location: str
    current_battery_kwh: float
    available_from: datetime

    @property
    def battery_percent(self):
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
    distance_km: float = 0.0
    energy_consumed: float = 0.0
    ride_id: Optional[str] = None
    line: str = ""

    @property
    def duration_minutes(self):
        return (self.end_time - self.start_time).total_seconds() / 60

@dataclass
class Assignment:
    bus_id: str
    ride: Ride
    events: List[Event] = field(default_factory=list)

    @property
    def battery_after_ride(self):
        ride_event = next((e for e in self.events if e.event_type == 'ride'), None)
        return ride_event.battery_after if ride_event else 0


# ===================================================================
# Helper Functions
# ===================================================================

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
    threshold = BusConstants.BATTERY_CAPACITY * BusConstants.FAST_CHARGE_THRESHOLD
    total_minutes = 0
    if current_kwh < threshold:
        fast_amount = min(target_kwh, threshold) - current_kwh
        total_minutes += fast_amount / fast_rate * 60
    if target_kwh > threshold and current_kwh < target_kwh:
        slow_start = max(current_kwh, threshold)
        slow_amount = target_kwh - slow_start
        total_minutes += slow_amount / slow_rate * 60
    return max(total_minutes, BusConstants.MIN_CHARGE_TIME)


# ===================================================================
# Distance Matrix
# ===================================================================

class DistanceMatrix:
    def __init__(self, distance_dict, time_dict):
        self.distance_dict = distance_dict
        self.time_dict = time_dict

    def get_distance_km(self, from_stop: str, to_stop: str) -> float:
        if from_stop == to_stop:
            return 0
        return self.distance_dict.get((from_stop, to_stop), 10000) / 1000

    def get_travel_time_minutes(self, from_stop: str, to_stop: str) -> float:
        if from_stop == to_stop:
            return 0
        time = self.time_dict.get((from_stop, to_stop), None)
        if time is None:
            return self.get_distance_km(from_stop, to_stop) / 30 * 60
        return time

    def get_energy_for_deadhead(self, from_stop: str, to_stop: str, consumption_per_km: float = None):
        return calculate_energy_consumption(self.get_distance_km(from_stop, to_stop), consumption_per_km)


# ===================================================================
# Charging Planner
# ===================================================================

class ChargingPlanner:
    def __init__(self, charging_station: str, fast_rate=None, slow_rate=None):
        self.charging_station = charging_station
        self.fast_rate = fast_rate or BusConstants.FAST_CHARGE_RATE
        self.slow_rate = slow_rate or BusConstants.SLOW_CHARGE_RATE

    def plan_charging_session(self, bus: Bus, target_kwh: float, available_time: float):
        ideal_time = calculate_charging_time(bus.current_battery_kwh, target_kwh,
                                            self.fast_rate, self.slow_rate)
        if ideal_time <= available_time:
            return target_kwh, ideal_time
        # Partial charge if time limited
        threshold = BusConstants.BATTERY_CAPACITY * BusConstants.FAST_CHARGE_THRESHOLD
        remaining = available_time / 60  # hours
        kwh = bus.current_battery_kwh
        if kwh < threshold:
            fast_cap = threshold - kwh
            fast_possible = self.fast_rate * remaining
            if fast_possible <= fast_cap:
                return kwh + fast_possible, available_time
            remaining -= fast_cap / self.fast_rate
            kwh = threshold
        slow_charge = min(self.slow_rate * remaining, BusConstants.BATTERY_CAPACITY - kwh)
        return kwh + slow_charge, available_time


# ===================================================================
# Bus Scheduler
# ===================================================================

class BusScheduler:
    def __init__(self, distance_matrix: DistanceMatrix, charging_planner: ChargingPlanner, garage_location: str,
                 consumption_per_km=None, idle_per_hour=None):
        self.distance_matrix = distance_matrix
        self.charging_planner = charging_planner
        self.garage_location = garage_location
        self.consumption_per_km = consumption_per_km or BusConstants.CONSUMPTION_PER_KM
        self.idle_per_hour = idle_per_hour or BusConstants.IDLE_CONSUMPTION_PER_HOUR

    # ----- Check if bus can serve ride
    def can_bus_serve_ride(self, bus: Bus, ride: Ride):
        deadhead_time = self.distance_matrix.get_travel_time_minutes(bus.current_location, ride.start_stop)
        arrival_time = bus.available_from + timedelta(minutes=deadhead_time)
        if arrival_time > ride.start_time:
            return False, "Cannot arrive in time"
        return True, None

    # ----- Assign a ride with all events
    def assign_ride_to_bus(self, bus: Bus, ride: Ride) -> Assignment:
        assignment = Assignment(bus.bus_id, ride, [])
        current_time = bus.available_from
        current_battery = bus.current_battery_kwh
        current_location = bus.current_location

        # ----- DEADHEAD to ride start if needed
        if current_location != ride.start_stop:
            distance = self.distance_matrix.get_distance_km(current_location, ride.start_stop)
            travel_time = self.distance_matrix.get_travel_time_minutes(current_location, ride.start_stop)
            energy = calculate_energy_consumption(distance, self.consumption_per_km)
            deadhead_event = Event(
                event_type="deadhead",
                bus_id=bus.bus_id,
                start_time=current_time,
                end_time=current_time + timedelta(minutes=travel_time),
                start_location=current_location,
                end_location=ride.start_stop,
                battery_before=current_battery,
                battery_after=current_battery - energy,
                distance_km=distance,
                energy_consumed=energy
            )
            assignment.events.append(deadhead_event)
            current_time += timedelta(minutes=travel_time)
            current_battery -= energy
            current_location = ride.start_stop

        # ----- IDLE if needed
        idle_seconds = (ride.start_time - current_time).total_seconds()
        if idle_seconds > 60:
            idle_minutes = idle_seconds / 60
            idle_energy = calculate_idle_consumption(idle_minutes, self.idle_per_hour)
            idle_event = Event(
                event_type="idle",
                bus_id=bus.bus_id,
                start_time=current_time,
                end_time=ride.start_time,
                start_location=current_location,
                end_location=current_location,
                battery_before=current_battery,
                battery_after=current_battery - idle_energy,
                energy_consumed=idle_energy
            )
            assignment.events.append(idle_event)
            current_time = ride.start_time
            current_battery -= idle_energy

        # ----- RIDE
        ride_energy = calculate_energy_consumption(ride.distance_km, self.consumption_per_km)
        ride_event = Event(
            event_type="ride",
            bus_id=bus.bus_id,
            start_time=ride.start_time,
            end_time=ride.end_time,
            start_location=ride.start_stop,
            end_location=ride.end_stop,
            battery_before=current_battery,
            battery_after=current_battery - ride_energy,
            distance_km=ride.distance_km,
            energy_consumed=ride_energy,
            ride_id=ride.ride_id,
            line=ride.line
        )
        assignment.events.append(ride_event)

        return assignment

    # ----- Schedule all rides
    def schedule_all_rides(self, rides: List[Ride], initial_buses: List[Bus]) -> List[Assignment]:
        sorted_rides = sorted(rides, key=lambda r: r.start_time)
        assignments = []
        buses = [Bus(b.bus_id, b.current_location, b.current_battery_kwh, b.available_from)
                 for b in initial_buses]

        for ride in sorted_rides:
            best_bus = None
            best_score = float("inf")

            for bus in buses:
                can_serve, reason = self.can_bus_serve_ride(bus, ride)
                if can_serve:
                    deadhead_dist = self.distance_matrix.get_distance_km(bus.current_location, ride.start_stop)
                    score = deadhead_dist * 2.0 + (1.0 - bus.battery_percent) * 10
                    if score < best_score:
                        best_score = score
                        best_bus = bus

            if best_bus is None:
                new_bus = Bus(f"BUS_{len(buses)+1}", self.garage_location, BusConstants.BATTERY_CAPACITY,
                              sorted_rides[0].start_time - timedelta(hours=2))
                buses.append(new_bus)
                best_bus = new_bus

            assignment = self.assign_ride_to_bus(best_bus, ride)
            assignments.append(assignment)

            best_bus.current_location = ride.end_stop
            best_bus.current_battery_kwh = assignment.battery_after_ride
            best_bus.available_from = ride.end_time

            if best_bus.current_battery_kwh < BusConstants.BATTERY_CAPACITY * 0.15:
                best_bus.current_battery_kwh = max(best_bus.current_battery_kwh,
                                                   BusConstants.BATTERY_CAPACITY * 0.15)

        return assignments


# ===================================================================
# Export to DataFrame
# ===================================================================

def assignments_to_dataframe(assignments: List[Assignment]) -> pd.DataFrame:
    all_events = []
    for assignment in assignments:
        for e in assignment.events:
            all_events.append({
                "start_location": e.start_location,
                "end_location": e.end_location,
                "start_time": e.start_time,
                "end_time": e.end_time,
                "activity": e.event_type,
                "line": e.line,
                "energy_consumption": round(e.energy_consumed, 2),
                "bus": e.bus_id
            })
    return pd.DataFrame(all_events)


# ===================================================================
# Main Planning Function
# ===================================================================

def create_bus_planning(timetable_df: pd.DataFrame,
                       distance_matrix_df: pd.DataFrame,
                       charging_station: str = "DEPOT",
                       garage_location: str = "GARAGE",
                       driving_usage: float = 1.2,
                       idle_usage: float = 5.0,
                       charging_speed: float = 450.0,
                       soh: float = 90.0,
                       startbat: float = 100.0) -> pd.DataFrame:

    BusConstants.CONSUMPTION_PER_KM = driving_usage
    BusConstants.IDLE_CONSUMPTION_PER_HOUR = idle_usage
    BusConstants.FAST_CHARGE_RATE = charging_speed
    BusConstants.AGING_FACTOR = soh / 100.0
    BusConstants.BATTERY_CAPACITY = BusConstants.ORIGINAL_BATTERY_CAPACITY * BusConstants.AGING_FACTOR

    # --- Prepare distance/time dictionaries
    distance_dict = {}
    time_dict = {}
    for _, row in distance_matrix_df.iterrows():
        start = str(row['start']).strip()
        end = str(row['end']).strip()
        distance_dict[(start, end)] = float(row['distance_m'])
        time_dict[(start, end)] = float((row['min_travel_time'] + row['max_travel_time']) / 2)

    distance_matrix = DistanceMatrix(distance_dict, time_dict)
    charging_planner = ChargingPlanner(charging_station)
    scheduler = BusScheduler(distance_matrix, charging_planner, garage_location, driving_usage, idle_usage)

    # --- Prepare rides
    rides = []
    for idx, row in timetable_df.iterrows():
        start = str(row['start']).strip()
        end = str(row['end']).strip()
        line = str(row['line']).strip()
        dep_time = pd.to_datetime(row['departure_time'])
        distance_m = distance_dict.get((start, end), 5000)
        duration_minutes = time_dict.get((start, end), 10)
        arrival_time = dep_time + timedelta(minutes=duration_minutes)
        ride_id = f"{line}_{start}_{dep_time.strftime('%H%M')}_{idx}"
        rides.append(Ride(ride_id, start, end, dep_time, arrival_time, distance_m, line))

    start_battery_kwh = BusConstants.BATTERY_CAPACITY * (startbat / 100)
    initial_buses = [Bus("BUS_1", garage_location, start_battery_kwh, rides[0].start_time - timedelta(hours=1))]

    assignments = scheduler.schedule_all_rides(rides, initial_buses)
    planning_df = assignments_to_dataframe(assignments)
    return planning_df
