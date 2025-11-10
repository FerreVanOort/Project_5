"""
PlanningMakerFerre.py
====================
Bus Planning Maker met volledige event tracking (rides, charging, deadhead, idle)
Voor gebruik in Streamlit applicatie
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta

# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class BusConstants:
    """Constants for bus specifications"""
    ORIGINAL_BATTERY_CAPACITY = 300  # kWh
    AGING_FACTOR = 0.90  # 90% of original capacity
    BATTERY_CAPACITY = ORIGINAL_BATTERY_CAPACITY * AGING_FACTOR
    CONSUMPTION_PER_KM = 1.2  # kWh/km
    IDLE_CONSUMPTION_PER_HOUR = 5  # kWh per hour
    MIN_BATTERY_PERCENT = 0.10
    MIN_CHARGE_TIME = 15  # minutes
    FAST_CHARGE_RATE = 450
    SLOW_CHARGE_RATE = 60
    FAST_CHARGE_THRESHOLD = 0.90

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

    @property
    def duration_minutes(self) -> float:
        return (self.end_time - self.start_time).total_seconds() / 60

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
    event_type: str  # 'ride', 'charging', 'deadhead', 'idle'
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
    def duration_minutes(self) -> float:
        return (self.end_time - self.start_time).total_seconds() / 60

@dataclass
class Assignment:
    bus_id: str
    ride: Ride
    events: List[Event] = field(default_factory=list)

# ============================================================================
# Energy / Charging Calculations
# ============================================================================

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

    # Fast charge phase
    if current_kwh < threshold_kwh:
        fast_charge_amount = min(target_kwh, threshold_kwh) - current_kwh
        total_minutes += fast_charge_amount / fast_rate * 60

    # Slow charge phase
    if target_kwh > threshold_kwh and current_kwh < target_kwh:
        slow_start = max(current_kwh, threshold_kwh)
        slow_charge_amount = target_kwh - slow_start
        total_minutes += slow_charge_amount / slow_rate * 60

    return max(total_minutes, BusConstants.MIN_CHARGE_TIME)

# ============================================================================
# Distance Matrix
# ============================================================================

class DistanceMatrix:
    def __init__(self, distance_matrix: Dict[Tuple[str, str], float],
                 time_matrix: Dict[Tuple[str, str], float]):
        self.distance_matrix = distance_matrix
        self.time_matrix = time_matrix

    def get_distance_km(self, from_stop: str, to_stop: str) -> float:
        if from_stop == to_stop:
            return 0.0
        return self.distance_matrix.get((from_stop, to_stop), 10000) / 1000

    def get_travel_time_minutes(self, from_stop: str, to_stop: str) -> float:
        if from_stop == to_stop:
            return 0.0
        t = self.time_matrix.get((from_stop, to_stop), None)
        if t is None:
            return self.get_distance_km(from_stop, to_stop) / 30 * 60
        return t

    def get_energy_for_deadhead(self, from_stop: str, to_stop: str, consumption_per_km: float = None) -> float:
        return calculate_energy_consumption(self.get_distance_km(from_stop, to_stop), consumption_per_km)

# ============================================================================
# Charging Planner
# ============================================================================

class ChargingPlanner:
    def __init__(self, charging_station_location: str, fast_rate: float = None, slow_rate: float = None):
        self.charging_station = charging_station_location
        self.fast_rate = fast_rate if fast_rate else BusConstants.FAST_CHARGE_RATE
        self.slow_rate = slow_rate if slow_rate else BusConstants.SLOW_CHARGE_RATE

    def plan_charging_session(self, bus: Bus, target_kwh: float, available_time: float) -> Tuple[float, float]:
        ideal_time = calculate_charging_time(bus.current_battery_kwh, target_kwh, self.fast_rate, self.slow_rate)
        if ideal_time <= available_time:
            return target_kwh, ideal_time
        # Partial charging if not enough time
        charged_kwh = self._charge_for_duration(bus.current_battery_kwh, available_time)
        return charged_kwh, available_time

    def _charge_for_duration(self, start_kwh: float, minutes: float) -> float:
        threshold_kwh = BusConstants.BATTERY_CAPACITY * BusConstants.FAST_CHARGE_THRESHOLD
        hours = minutes / 60
        current = start_kwh
        if current < threshold_kwh:
            fast_charge_cap = threshold_kwh - current
            fast_charge_possible = self.fast_rate * hours
            if fast_charge_possible <= fast_charge_cap:
                return current + fast_charge_possible
            time_for_fast = fast_charge_cap / self.fast_rate
            hours -= time_for_fast
            current = threshold_kwh
        slow_charge = self.slow_rate * hours
        return min(current + slow_charge, BusConstants.BATTERY_CAPACITY)

# ============================================================================
# Bus Scheduler
# ============================================================================

class BusScheduler:
    def __init__(self, distance_matrix: DistanceMatrix,
                 charging_planner: ChargingPlanner,
                 garage_location: str,
                 consumption_per_km: float = None,
                 idle_per_hour: float = None):
        self.distance_matrix = distance_matrix
        self.charging_planner = charging_planner
        self.garage_location = garage_location
        self.consumption_per_km = consumption_per_km if consumption_per_km else BusConstants.CONSUMPTION_PER_KM
        self.idle_per_hour = idle_per_hour if idle_per_hour else BusConstants.IDLE_CONSUMPTION_PER_HOUR

    def can_bus_serve_ride(self, bus: Bus, ride: Ride) -> Tuple[bool, Optional[str]]:
        deadhead_time = self.distance_matrix.get_travel_time_minutes(bus.current_location, ride.start_stop)
        arrival_time = bus.available_from + timedelta(minutes=deadhead_time)
        if arrival_time > ride.start_time:
            return False, "Cannot arrive in time"
        deadhead_energy = self.distance_matrix.get_energy_for_deadhead(bus.current_location, ride.start_stop, self.consumption_per_km)
        ride_energy = calculate_energy_consumption(ride.distance_km, self.consumption_per_km)
        if bus.current_battery_kwh >= (deadhead_energy + ride_energy):
            return True, None
        return True, None  # Can charge if needed

    def assign_ride_to_bus(self, bus: Bus, ride: Ride) -> Assignment:
        """Create all events without overlaps, with correct idle/charging placement."""
        assignment = Assignment(bus_id=bus.bus_id, ride=ride)
        current_time = max(bus.available_from, ride.start_time - timedelta(hours=2))
        current_battery = bus.current_battery_kwh
        current_location = bus.current_location

        # CHARGING
        battery_needed = calculate_energy_consumption(ride.distance_km, self.consumption_per_km)
        if current_battery < battery_needed * 1.3:
            charger = self.charging_planner.charging_station
            if current_location != charger:
                travel_to_charger = self.distance_matrix.get_travel_time_minutes(current_location, charger)
                current_battery -= self.distance_matrix.get_energy_for_deadhead(current_location, charger, self.consumption_per_km)
                arrival_at_charger = current_time + timedelta(minutes=travel_to_charger)
            else:
                arrival_at_charger = current_time
            available_time = max(0, (ride.start_time - arrival_at_charger).total_seconds() / 60)
            target_charge = min(BusConstants.BATTERY_CAPACITY, battery_needed * 1.5)
            charged_to, charge_duration = self.charging_planner.plan_charging_session(
                Bus(bus.bus_id, charger, current_battery, arrival_at_charger),
                target_charge, available_time
            )
            charging_event = Event(
                event_type='charging',
                bus_id=bus.bus_id,
                start_time=arrival_at_charger,
                end_time=arrival_at_charger + timedelta(minutes=charge_duration),
                start_location=charger,
                end_location=charger,
                battery_before=current_battery,
                battery_after=charged_to,
                energy_consumed=-(charged_to - current_battery)
            )
            assignment.events.append(charging_event)
            current_battery = charged_to
            current_location = charger
            current_time = charging_event.end_time

        # DEADHEAD
        if current_location != ride.start_stop:
            deadhead_time = self.distance_matrix.get_travel_time_minutes(current_location, ride.start_stop)
            deadhead_energy = self.distance_matrix.get_energy_for_deadhead(current_location, ride.start_stop, self.consumption_per_km)
            deadhead_event = Event(
                event_type='deadhead',
                bus_id=bus.bus_id,
                start_time=current_time,
                end_time=current_time + timedelta(minutes=deadhead_time),
                start_location=current_location,
                end_location=ride.start_stop,
                battery_before=current_battery,
                battery_after=current_battery - deadhead_energy,
                distance_km=self.distance_matrix.get_distance_km(current_location, ride.start_stop),
                energy_consumed=deadhead_energy
            )
            assignment.events.append(deadhead_event)
            current_battery -= deadhead_energy
            current_time = deadhead_event.end_time
            current_location = ride.start_stop

        # IDLE
        if current_time < ride.start_time:
            idle_minutes = (ride.start_time - current_time).total_seconds() / 60
            idle_energy = calculate_idle_consumption(idle_minutes, self.idle_per_hour)
            idle_event = Event(
                event_type='idle',
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
            current_battery -= idle_energy
            current_time = ride.start_time

        # RIDE
        ride_energy = calculate_energy_consumption(ride.distance_km, self.consumption_per_km)
        ride_event = Event(
            event_type='ride',
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
        current_battery -= ride_energy
        current_time = ride.end_time

        return assignment

# ============================================================================
# Schedule All Rides
# ============================================================================

def schedule_all_rides(rides: List[Ride], initial_buses: List[Bus], scheduler: BusScheduler) -> List[Assignment]:
    sorted_rides = sorted(rides, key=lambda r: r.start_time)
    assignments = []
    buses = [Bus(b.bus_id, b.current_location, b.current_battery_kwh, b.available_from) for b in initial_buses]

    for ride in sorted_rides:
        best_bus = None
        best_score = float('inf')
        for bus in buses:
            can_serve, _ = scheduler.can_bus_serve_ride(bus, ride)
            if can_serve:
                score = scheduler.distance_matrix.get_distance_km(bus.current_location, ride.start_stop) * 2 + (1 - bus.battery_percent) * 10
                if score < best_score:
                    best_score = score
                    best_bus = bus
        if best_bus is None:
            new_bus = Bus(
                bus_id=f"BUS_{len(buses)+1}",
                current_location=scheduler.garage_location,
                current_battery_kwh=BusConstants.BATTERY_CAPACITY,
                available_from=ride.start_time - timedelta(hours=1)
            )
            buses.append(new_bus)
            best_bus = new_bus
        assignment = scheduler.assign_ride_to_bus(best_bus, ride)
        assignments.append(assignment)
        last_event = assignment.events[-1]
        best_bus.current_location = last_event.end_location
        best_bus.current_battery_kwh = last_event.battery_after
        best_bus.available_from = last_event.end_time

    return assignments

# ============================================================================
# Export to DataFrame
# ============================================================================

def assignments_to_dataframe(assignments: List[Assignment]) -> pd.DataFrame:
    all_events = []
    for assignment in assignments:
        for event in assignment.events:
            all_events.append({
                'start_location': event.start_location,
                'end_location': event.end_location,
                'start_time': event.start_time,
                'end_time': event.end_time,
                'activity': event.event_type,
                'line': event.line,
                'energy_consumption': round(event.energy_consumed, 2),
                'bus': event.bus_id
            })
    return pd.DataFrame(all_events)[[
        'start_location', 'end_location', 'start_time', 'end_time', 'activity', 'line', 'energy_consumption', 'bus'
    ]]

# ============================================================================
# Main Planning Function
# ============================================================================

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
    BusConstants.AGING_FACTOR = soh / 100.0
    BusConstants.BATTERY_CAPACITY = BusConstants.ORIGINAL_BATTERY_CAPACITY * BusConstants.AGING_FACTOR

    # Load distance matrix
    distance_dict = {(row['start'], row['end']): float(row['distance_m']) for _, row in distance_matrix_df.iterrows()}
    time_dict = {(row['start'], row['end']): (float(row['min_travel_time']) + float(row['max_travel_time'])) / 2
                 for _, row in distance_matrix_df.iterrows()}
    distance_matrix = DistanceMatrix(distance_dict, time_dict)
    charging_planner = ChargingPlanner(charging_station, charging_speed, 60)
    scheduler = BusScheduler(distance_matrix, charging_planner, garage_location, driving_usage, idle_usage)

    # Load rides
    rides = []
    for idx, row in timetable_df.iterrows():
        start_time = pd.to_datetime(row['departure_time'])
        end_time = start_time + timedelta(minutes=(int(row['avg_travel_time']) if 'avg_travel_time' in row else 10))
        ride_id = f"{row['line']}_{row['start']}_{start_time.strftime('%H%M')}_{idx}"
        rides.append(Ride(
            ride_id=ride_id,
            start_stop=row['start'],
            end_stop=row['end'],
            start_time=start_time,
            end_time=end_time,
            distance_meters=float(row['distance_m']),
            line=row['line']
        ))

    start_battery_kwh = BusConstants.BATTERY_CAPACITY * (startbat / 100.0)
    initial_buses = [Bus("BUS_1", garage_location, start_battery_kwh, rides[0].start_time - timedelta(hours=1))]

    assignments = schedule_all_rides(rides, initial_buses, scheduler)
    planning_df = assignments_to_dataframe(assignments)
    return planning_df
