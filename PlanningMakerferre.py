from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Tuple
import pandas as pd

# =========================================
# Bus Constants
# =========================================
@dataclass
class BusConstants:
    ORIGINAL_BATTERY_CAPACITY: float = 300
    AGING_FACTOR: float = 0.9
    BATTERY_CAPACITY: float = ORIGINAL_BATTERY_CAPACITY * AGING_FACTOR
    CONSUMPTION_PER_KM: float = 1.2
    IDLE_CONSUMPTION_PER_HOUR: float = 5
    MIN_BATTERY_PERCENT: float = 0.10
    MIN_CHARGE_TIME: float = 15
    FAST_CHARGE_RATE: float = 450
    SLOW_CHARGE_RATE: float = 60
    FAST_CHARGE_THRESHOLD: float = 0.90

# =========================================
# Ride and Bus
# =========================================
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

# =========================================
# Event
# =========================================
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

# =========================================
# Distance Matrix
# =========================================
class DistanceMatrix:
    def __init__(self, distance_matrix: Dict[Tuple[str,str], float], time_matrix: Dict[Tuple[str,str], float]):
        self.distance_matrix = distance_matrix
        self.time_matrix = time_matrix

    def get_distance_km(self, from_stop: str, to_stop: str) -> float:
        if from_stop == to_stop:
            return 0
        return self.distance_matrix.get((from_stop,to_stop), 10000) / 1000

    def get_travel_time_minutes(self, from_stop: str, to_stop: str) -> float:
        if from_stop == to_stop:
            return 0
        return self.time_matrix.get((from_stop,to_stop), self.get_distance_km(from_stop,to_stop)/30*60)

    def get_energy_for_deadhead(self, from_stop: str, to_stop: str) -> float:
        distance_km = self.get_distance_km(from_stop, to_stop)
        return distance_km * BusConstants.CONSUMPTION_PER_KM

# =========================================
# Charging Planner
# =========================================
class ChargingPlanner:
    def __init__(self, charging_station: str):
        self.charging_station = charging_station

    def plan_charging_session(self, bus: Bus, target_kwh: float, available_minutes: float):
        if bus.current_battery_kwh >= target_kwh:
            return bus.current_battery_kwh, 0
        threshold = BusConstants.BATTERY_CAPACITY * BusConstants.FAST_CHARGE_THRESHOLD
        minutes = 0
        current = bus.current_battery_kwh

        # Fast charge
        if current < threshold:
            fast_needed = min(target_kwh, threshold) - current
            fast_time = fast_needed / BusConstants.FAST_CHARGE_RATE * 60
            minutes += fast_time
            current += fast_needed

        # Slow charge
        if current < target_kwh:
            slow_needed = target_kwh - current
            slow_time = slow_needed / BusConstants.SLOW_CHARGE_RATE * 60
            minutes += slow_time
            current += slow_needed

        return min(current, BusConstants.BATTERY_CAPACITY), min(minutes, available_minutes)

# =========================================
# Scheduler
# =========================================
class BusScheduler:
    def __init__(self, distance_matrix: DistanceMatrix, charging_planner: ChargingPlanner, garage_location: str):
        self.distance_matrix = distance_matrix
        self.charging_planner = charging_planner
        self.garage_location = garage_location

    def assign_ride_to_bus(self, bus: Bus, ride: Ride) -> List[Event]:
        events = []
        current_time = bus.available_from
        current_battery = bus.current_battery_kwh
        current_location = bus.current_location

        # Deadhead if needed
        if current_location != ride.start_stop:
            travel_time = self.distance_matrix.get_travel_time_minutes(current_location, ride.start_stop)
            distance_km = self.distance_matrix.get_distance_km(current_location, ride.start_stop)
            energy = distance_km * BusConstants.CONSUMPTION_PER_KM
            start_time = current_time
            end_time = current_time + timedelta(minutes=travel_time)
            events.append(Event(
                "deadhead", bus.bus_id, start_time, end_time,
                current_location, ride.start_stop, current_battery, current_battery-energy,
                distance_km, energy
            ))
            current_time = end_time
            current_battery -= energy
            current_location = ride.start_stop

        # Idle if early
        if current_time < ride.start_time:
            idle_minutes = (ride.start_time - current_time).total_seconds()/60
            energy = idle_minutes/60*BusConstants.IDLE_CONSUMPTION_PER_HOUR
            events.append(Event(
                "idle", bus.bus_id, current_time, ride.start_time,
                current_location, current_location, current_battery, current_battery-energy,
                0, energy
            ))
            current_battery -= energy
            current_time = ride.start_time

        # Charging if battery too low
        if current_battery < ride.distance_km*BusConstants.CONSUMPTION_PER_KM*1.3:
            target = max(BusConstants.BATTERY_CAPACITY*0.8, ride.distance_km*BusConstants.CONSUMPTION_PER_KM*1.3)
            available_time = (ride.start_time - current_time).total_seconds()/60
            charged_to, charge_minutes = self.charging_planner.plan_charging_session(
                Bus(bus.bus_id, self.charging_planner.charging_station, current_battery, current_time),
                target, available_time
            )
            start_time = current_time
            end_time = current_time + timedelta(minutes=charge_minutes)
            events.append(Event(
                "charging", bus.bus_id, start_time, end_time,
                current_location, self.charging_planner.charging_station,
                current_battery, charged_to,
                0, -(charged_to-current_battery)
            ))
            current_time = end_time
            current_battery = charged_to
            current_location = self.charging_planner.charging_station

        # Ride
        distance_km = ride.distance_km
        energy = distance_km * BusConstants.CONSUMPTION_PER_KM
        events.append(Event(
            "ride", bus.bus_id, ride.start_time, ride.end_time,
            ride.start_stop, ride.end_stop,
            current_battery, current_battery-energy,
            distance_km, energy, ride.ride_id, ride.line
        ))
        current_battery -= energy
        current_time = ride.end_time
        current_location = ride.end_stop

        # Update bus state
        bus.current_location = current_location
        bus.current_battery_kwh = current_battery
        bus.available_from = current_time

        return events

    def schedule_all_rides(self, rides: List[Ride], buses: List[Bus]) -> List[Event]:
        all_events = []
        for ride in sorted(rides, key=lambda r: r.start_time):
            # Find first available bus
            bus = min(buses, key=lambda b: b.available_from)
            events = self.assign_ride_to_bus(bus, ride)
            all_events.extend(events)
        return all_events

# =========================================
# Main Planning Function
# =========================================
def create_bus_planning(timetable_df: pd.DataFrame, distance_matrix_df: pd.DataFrame,
                        charging_station: str = "DEPOT", garage_location: str = "GARAGE",
                        driving_usage: float = 1.2, idle_usage: float = 5.0,
                        startbat: float = 100.0,
                        charging_speed: float = 450.0, soh: float = 90.0) -> pd.DataFrame:

    # Update constants
    BusConstants.CONSUMPTION_PER_KM = driving_usage
    BusConstants.IDLE_CONSUMPTION_PER_HOUR = idle_usage
    BusConstants.FAST_CHARGE_RATE = charging_speed
    BusConstants.AGING_FACTOR = soh/100
    BusConstants.BATTERY_CAPACITY = BusConstants.ORIGINAL_BATTERY_CAPACITY*BusConstants.AGING_FACTOR

    # Build rides
    rides = []
    for idx,row in timetable_df.iterrows():
        start_stop = row['start']
        end_stop = row['end']
        line = str(row.get('line',''))
        start_time = pd.to_datetime(row['departure_time'])
        # lookup distance from distance_matrix
        match = distance_matrix_df[
            (distance_matrix_df['start']==start_stop)&
            (distance_matrix_df['end']==end_stop)
        ]
        distance_m = float(match.iloc[0]['distance_m']) if not match.empty else 5000
        avg_travel_time = float(match.iloc[0]['min_travel_time']) if not match.empty else 10
        end_time = start_time + timedelta(minutes=avg_travel_time)
        ride_id = f"{line}_{start_stop}_{start_time.strftime('%H%M')}_{idx}"
        rides.append(Ride(ride_id, start_stop, end_stop, start_time, end_time, distance_m, line))

    # Build distance matrix
    distance_dict = {}
    time_dict = {}
    for idx,row in distance_matrix_df.iterrows():
        distance_dict[(row['start'], row['end'])] = float(row['distance_m'])
        time_dict[(row['start'], row['end'])] = (float(row['min_travel_time']) + float(row['max_travel_time']))/2

    distance_matrix = DistanceMatrix(distance_dict, time_dict)
    charging_planner = ChargingPlanner(charging_station)
    scheduler = BusScheduler(distance_matrix, charging_planner, garage_location)

    # Start buses
    initial_battery = BusConstants.BATTERY_CAPACITY*startbat/100
    buses = [Bus("BUS_1", garage_location, initial_battery, rides[0].start_time - timedelta(hours=1))]

    # Schedule
    events = scheduler.schedule_all_rides(rides, buses)

    # Convert to DataFrame
    all_events = []
    for e in events:
        all_events.append({
            'bus': e.bus_id,
            'activity': e.event_type,
            'line': e.line,
            'start_time': e.start_time,
            'end_time': e.end_time,
            'start_location': e.start_location,
            'end_location': e.end_location,
            'distance_km': e.distance_km,
            'energy_consumption': round(e.energy_consumed,2)
        })
    df = pd.DataFrame(all_events)
    return df
