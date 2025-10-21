## Eigen planning

# Imports
import pandas as pd
from datetime import timedelta
from datetime import datetime
from datetime import time

# Formules / Functies
"""
Complete Bus Planning System - Excel to Excel
==============================================

This system reads timetable and distance matrix from Excel files,
plans the complete bus schedule, and exports to Excel.

Required libraries:
pip install pandas openpyxl numpy
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import sys

# ============================================================================
# STEP 1: Define Data Structures
# ============================================================================

@dataclass
class BusConstants:
    """Constants for bus specifications"""
    ORIGINAL_BATTERY_CAPACITY = 300  # kWh
    AGING_FACTOR = 0.90  # 90% of original capacity
    BATTERY_CAPACITY = ORIGINAL_BATTERY_CAPACITY * AGING_FACTOR  # 270 kWh
    
    CONSUMPTION_PER_KM = 1.2  # kW/km
    IDLE_CONSUMPTION_PER_HOUR = 5  # kWh per hour
    
    MIN_BATTERY_PERCENT = 0.10  # 10% minimum
    MIN_CHARGE_TIME = 15  # minutes
    
    FAST_CHARGE_RATE = 450  # kWh per hour until 90%
    SLOW_CHARGE_RATE = 60   # kWh per hour after 90%
    FAST_CHARGE_THRESHOLD = 0.90  # Switch to slow charging at 90%


@dataclass
class Ride:
    """Represents a scheduled ride from timetable"""
    ride_id: str
    start_stop: str
    end_stop: str
    start_time: datetime
    end_time: datetime
    distance_meters: float
    
    @property
    def distance_km(self) -> float:
        return self.distance_meters / 1000
    
    @property
    def duration_minutes(self) -> float:
        return (self.end_time - self.start_time).total_seconds() / 60


@dataclass
class Bus:
    """Represents a bus with its current state"""
    bus_id: str
    current_location: str
    current_battery_kwh: float
    available_from: datetime
    
    @property
    def battery_percent(self) -> float:
        return self.current_battery_kwh / BusConstants.BATTERY_CAPACITY


@dataclass
class Assignment:
    """Represents a ride assignment to a bus"""
    bus_id: str
    ride: Ride
    battery_before: float
    battery_after: float
    charging_before: Optional[Tuple[datetime, datetime, float, float]] = None
    deadhead_before: Optional[Tuple[str, str, float, datetime, datetime]] = None


# ============================================================================
# STEP 2: Battery Calculation Functions
# ============================================================================

def calculate_energy_consumption(distance_km: float) -> float:
    """Calculate energy consumption for a given distance."""
    return distance_km * BusConstants.CONSUMPTION_PER_KM


def calculate_idle_consumption(minutes: float) -> float:
    """Calculate energy consumption while idle."""
    hours = minutes / 60
    return hours * BusConstants.IDLE_CONSUMPTION_PER_HOUR


def calculate_charging_time(current_kwh: float, target_kwh: float) -> float:
    """Calculate time needed to charge from current to target level."""
    if target_kwh <= current_kwh:
        return 0
    
    threshold_kwh = BusConstants.BATTERY_CAPACITY * BusConstants.FAST_CHARGE_THRESHOLD
    total_minutes = 0
    
    # Phase 1: Fast charging up to 90%
    if current_kwh < threshold_kwh:
        fast_charge_amount = min(target_kwh, threshold_kwh) - current_kwh
        fast_charge_hours = fast_charge_amount / BusConstants.FAST_CHARGE_RATE
        total_minutes += fast_charge_hours * 60
    
    # Phase 2: Slow charging above 90%
    if target_kwh > threshold_kwh and current_kwh < target_kwh:
        slow_charge_start = max(current_kwh, threshold_kwh)
        slow_charge_amount = target_kwh - slow_charge_start
        slow_charge_hours = slow_charge_amount / BusConstants.SLOW_CHARGE_RATE
        total_minutes += slow_charge_hours * 60
    
    return max(total_minutes, BusConstants.MIN_CHARGE_TIME)


def is_battery_sufficient(current_kwh: float, distance_km: float, 
                          idle_minutes: float = 0) -> bool:
    """Check if battery is sufficient for a trip plus idle time."""
    energy_needed = (calculate_energy_consumption(distance_km) + 
                    calculate_idle_consumption(idle_minutes))
    return current_kwh >= energy_needed


def get_minimum_battery_for_trip(distance_km: float, idle_minutes: float = 0) -> float:
    """Calculate minimum battery needed for a trip."""
    return (calculate_energy_consumption(distance_km) + 
            calculate_idle_consumption(idle_minutes))


# ============================================================================
# STEP 3: Excel Data Loading
# ============================================================================

class DataLoader:
    """Loads data from Excel files or DataFrames"""
    
    @staticmethod
    def load_timetable(filepath: str, distance_matrix_df: pd.DataFrame = None) -> List[Ride]:
        """
        Load timetable from Excel file.
        Expected columns: start, departure_time, end, line
        """
        print(f"Loading timetable from {filepath}...")
        df = pd.read_excel(filepath)
        
        # Print column names to help debug
        print(f"  Found columns: {list(df.columns)}")
        
        rides = []
        first_departure = None
        
        for idx, row in df.iterrows():
            try:
                start_stop = str(row['start']).strip()
                end_stop = str(row['end']).strip()
                line = str(row['line']).strip()
                
                # Parse departure time
                departure_time = pd.to_datetime(row['departure_time'])
                
                # Track first departure time to identify next-day rides
                if first_departure is None:
                    first_departure = departure_time
                
                # If departure is before 4 AM and first ride is after 4 AM, 
                # this is likely a next-day ride (after midnight)
                if departure_time.hour < 4 and first_departure.hour >= 4:
                    departure_time = departure_time + timedelta(days=1)
                    print(f"  Adjusted ride {idx} to next day (after midnight): {departure_time}")
                
                # Look up distance and travel time from distance matrix
                if distance_matrix_df is not None:
                    # Try multiple matching strategies
                    # Strategy 1: Exact match with line
                    match = distance_matrix_df[
                        (distance_matrix_df['start'].astype(str).str.strip() == start_stop) & 
                        (distance_matrix_df['end'].astype(str).str.strip() == end_stop) &
                        (distance_matrix_df['line'].astype(str).str.strip() == line)
                    ]
                    
                    # Strategy 2: Match without line if no exact match
                    if match.empty:
                        match = distance_matrix_df[
                            (distance_matrix_df['start'].astype(str).str.strip() == start_stop) & 
                            (distance_matrix_df['end'].astype(str).str.strip() == end_stop)
                        ]
                        if not match.empty:
                            print(f"  Info: Matched {start_stop} -> {end_stop} without line constraint")
                    
                    # Strategy 3: Case-insensitive match
                    if match.empty:
                        match = distance_matrix_df[
                            (distance_matrix_df['start'].astype(str).str.strip().str.lower() == start_stop.lower()) & 
                            (distance_matrix_df['end'].astype(str).str.strip().str.lower() == end_stop.lower())
                        ]
                        if not match.empty:
                            print(f"  Info: Matched {start_stop} -> {end_stop} with case-insensitive search")
                    
                    if not match.empty:
                        distance_m = float(match.iloc[0]['distance_m'])
                        min_time = float(match.iloc[0]['min_travel_time'])
                        max_time = float(match.iloc[0]['max_travel_time'])
                        avg_travel_time = round((min_time + max_time) / 2)
                    else:
                        print(f"  WARNING: No distance data found for '{start_stop}' -> '{end_stop}' (line '{line}')")
                        print(f"           Available starts in matrix: {sorted(distance_matrix_df['start'].astype(str).str.strip().unique())[:5]}...")
                        print(f"           Available ends in matrix: {sorted(distance_matrix_df['end'].astype(str).str.strip().unique())[:5]}...")
                        distance_m = 5000  # Default 5km
                        avg_travel_time = 10  # Default 10 minutes
                else:
                    # No distance matrix provided
                    distance_m = 5000
                    avg_travel_time = 10
                
                # Calculate arrival time
                arrival_time = departure_time + timedelta(minutes=avg_travel_time)
                
                # Create unique ride ID
                ride_id = f"{line}_{start_stop}_{departure_time.strftime('%H%M')}_{idx}"
                
                ride = Ride(
                    ride_id=ride_id,
                    start_stop=start_stop,
                    end_stop=end_stop,
                    start_time=departure_time,
                    end_time=arrival_time,
                    distance_meters=distance_m
                )
                rides.append(ride)
            except Exception as e:
                print(f"  Warning: Could not parse row {idx}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        print(f"  Loaded {len(rides)} rides")
        return rides
    
    @staticmethod
    def load_timetable_from_df(df: pd.DataFrame, distance_matrix_df: pd.DataFrame = None) -> List[Ride]:
        """
        Load timetable from DataFrame instead of file.
        Expected columns: start, departure_time, end, line
        """
        rides = []
        first_departure = None
        
        for idx, row in df.iterrows():
            try:
                start_stop = str(row['start']).strip()
                end_stop = str(row['end']).strip()
                line = str(row['line']).strip()
                
                # Parse departure time
                departure_time = pd.to_datetime(row['departure_time'])
                
                # Track first departure time to identify next-day rides
                if first_departure is None:
                    first_departure = departure_time
                
                # If departure is before 4 AM and first ride is after 4 AM, 
                # this is likely a next-day ride (after midnight)
                if departure_time.hour < 4 and first_departure.hour >= 4:
                    departure_time = departure_time + timedelta(days=1)
                
                # Look up distance and travel time from distance matrix
                if distance_matrix_df is not None:
                    # Try multiple matching strategies
                    # Strategy 1: Exact match with line
                    match = distance_matrix_df[
                        (distance_matrix_df['start'].astype(str).str.strip() == start_stop) & 
                        (distance_matrix_df['end'].astype(str).str.strip() == end_stop) &
                        (distance_matrix_df['line'].astype(str).str.strip() == line)
                    ]
                    
                    # Strategy 2: Match without line if no exact match
                    if match.empty:
                        match = distance_matrix_df[
                            (distance_matrix_df['start'].astype(str).str.strip() == start_stop) & 
                            (distance_matrix_df['end'].astype(str).str.strip() == end_stop)
                        ]
                    
                    # Strategy 3: Case-insensitive match
                    if match.empty:
                        match = distance_matrix_df[
                            (distance_matrix_df['start'].astype(str).str.strip().str.lower() == start_stop.lower()) & 
                            (distance_matrix_df['end'].astype(str).str.strip().str.lower() == end_stop.lower())
                        ]
                    
                    if not match.empty:
                        distance_m = float(match.iloc[0]['distance_m'])
                        min_time = float(match.iloc[0]['min_travel_time'])
                        max_time = float(match.iloc[0]['max_travel_time'])
                        avg_travel_time = round((min_time + max_time) / 2)
                    else:
                        distance_m = 5000  # Default 5km
                        avg_travel_time = 10  # Default 10 minutes
                else:
                    # No distance matrix provided
                    distance_m = 5000
                    avg_travel_time = 10
                
                # Calculate arrival time
                arrival_time = departure_time + timedelta(minutes=avg_travel_time)
                
                # Create unique ride ID
                ride_id = f"{line}_{start_stop}_{departure_time.strftime('%H%M')}_{idx}"
                
                ride = Ride(
                    ride_id=ride_id,
                    start_stop=start_stop,
                    end_stop=end_stop,
                    start_time=departure_time,
                    end_time=arrival_time,
                    distance_meters=distance_m
                )
                rides.append(ride)
            except Exception as e:
                continue
        
        return rides
    
    @staticmethod
    def load_distance_matrix(filepath: str) -> Tuple[Dict, Dict, pd.DataFrame]:
        """
        Load distance and time matrices from Excel file.
        Expected columns: start, end, min_travel_time, max_travel_time, distance_m, line
        """
        print(f"Loading distance matrix from {filepath}...")
        
        df = pd.read_excel(filepath)
        print(f"  Found columns: {list(df.columns)}")
        print(f"  Loaded {len(df)} distance entries")
        
        # Convert to dictionaries for quick lookup
        distance_dict = {}
        time_dict = {}
        
        for idx, row in df.iterrows():
            try:
                start = str(row['start']).strip()
                end = str(row['end']).strip()
                line = str(row['line']).strip()
                
                distance_m = float(row['distance_m'])
                min_time = float(row['min_travel_time'])
                max_time = float(row['max_travel_time'])
                
                # Calculate average travel time rounded to nearest whole number
                avg_time = round((min_time + max_time) / 2)
                
                # Create keys with and without line for flexible lookup
                key = (start, end)
                key_with_line = (start, end, line)
                
                # Store both versions
                distance_dict[key] = distance_m
                distance_dict[key_with_line] = distance_m
                time_dict[key] = avg_time
                time_dict[key_with_line] = avg_time
                
            except Exception as e:
                print(f"  Warning: Could not parse row {idx}: {e}")
                continue
        
        print(f"  Created dictionaries with {len(distance_dict)} entries")
        return distance_dict, time_dict, df
    
    @staticmethod
    def load_distance_matrix_from_df(df: pd.DataFrame) -> Tuple[Dict, Dict, pd.DataFrame]:
        """
        Load distance and time matrices from DataFrame instead of file.
        Expected columns: start, end, min_travel_time, max_travel_time, distance_m, line
        """
        # Convert to dictionaries for quick lookup
        distance_dict = {}
        time_dict = {}
        
        for idx, row in df.iterrows():
            try:
                start = str(row['start']).strip()
                end = str(row['end']).strip()
                line = str(row['line']).strip()
                
                distance_m = float(row['distance_m'])
                min_time = float(row['min_travel_time'])
                max_time = float(row['max_travel_time'])
                
                # Calculate average travel time rounded to nearest whole number
                avg_time = round((min_time + max_time) / 2)
                
                # Create keys with and without line for flexible lookup
                key = (start, end)
                key_with_line = (start, end, line)
                
                # Store both versions
                distance_dict[key] = distance_m
                distance_dict[key_with_line] = distance_m
                time_dict[key] = avg_time
                time_dict[key_with_line] = avg_time
                
            except Exception as e:
                continue
        
        return distance_dict, time_dict, df


# ============================================================================
# STEP 4: Distance Matrix Functions
# ============================================================================

class DistanceMatrix:
    """Handles distance and travel time lookups between stops"""
    
    def __init__(self, distance_matrix: Dict[Tuple[str, str], float],
                 time_matrix: Dict[Tuple[str, str], float]):
        self.distance_matrix = distance_matrix
        self.time_matrix = time_matrix
    
    def get_distance_km(self, from_stop: str, to_stop: str) -> float:
        """Get distance in kilometers between two stops"""
        if from_stop == to_stop:
            return 0.0
        distance = self.distance_matrix.get((from_stop, to_stop), None)
        if distance is None:
            print(f"  Warning: No distance found for {from_stop} -> {to_stop}, using 10km")
            return 10.0
        return distance / 1000
    
    def get_travel_time_minutes(self, from_stop: str, to_stop: str) -> float:
        """Get travel time in minutes between two stops"""
        if from_stop == to_stop:
            return 0.0
        time = self.time_matrix.get((from_stop, to_stop), None)
        if time is None:
            # Fallback: calculate from distance
            distance_km = self.get_distance_km(from_stop, to_stop)
            return distance_km / 30 * 60  # 30 km/h average
        return time
    
    def get_energy_for_deadhead(self, from_stop: str, to_stop: str) -> float:
        """Calculate energy needed for deadhead trip"""
        distance_km = self.get_distance_km(from_stop, to_stop)
        return calculate_energy_consumption(distance_km)


# ============================================================================
# STEP 5: Charging Station Management
# ============================================================================

class ChargingPlanner:
    """Manages charging decisions and calculations"""
    
    def __init__(self, charging_station_location: str):
        self.charging_station = charging_station_location
    
    def calculate_optimal_charge_level(self, next_ride: Ride, 
                                      subsequent_rides: List[Ride]) -> float:
        """Determine optimal charge level considering upcoming rides."""
        total_energy_needed = get_minimum_battery_for_trip(next_ride.distance_km)
        
        for ride in subsequent_rides[:3]:
            total_energy_needed += get_minimum_battery_for_trip(ride.distance_km)
        
        target = total_energy_needed * 1.2
        return min(target, BusConstants.BATTERY_CAPACITY)
    
    def plan_charging_session(self, bus: Bus, target_kwh: float,
                             available_time: float) -> Tuple[float, float]:
        """Plan a charging session within available time."""
        ideal_time = calculate_charging_time(bus.current_battery_kwh, target_kwh)
        
        if ideal_time <= available_time:
            return target_kwh, ideal_time
        
        achieved_kwh = self._charge_for_duration(bus.current_battery_kwh, 
                                                 available_time)
        return achieved_kwh, available_time
    
    def _charge_for_duration(self, start_kwh: float, minutes: float) -> float:
        """Calculate battery level after charging for given duration"""
        threshold_kwh = BusConstants.BATTERY_CAPACITY * BusConstants.FAST_CHARGE_THRESHOLD
        hours = minutes / 60
        current = start_kwh
        
        if current < threshold_kwh:
            fast_charge_capacity = threshold_kwh - current
            fast_charge_possible = BusConstants.FAST_CHARGE_RATE * hours
            
            if fast_charge_possible <= fast_charge_capacity:
                return current + fast_charge_possible
            else:
                time_for_fast = fast_charge_capacity / BusConstants.FAST_CHARGE_RATE
                remaining_hours = hours - time_for_fast
                current = threshold_kwh
                hours = remaining_hours
        
        slow_charge = BusConstants.SLOW_CHARGE_RATE * hours
        return min(current + slow_charge, BusConstants.BATTERY_CAPACITY)
    
    def needs_charging(self, bus: Bus, next_ride: Ride, 
                      distance_matrix: DistanceMatrix) -> bool:
        """Determine if bus needs charging before next ride."""
        deadhead_energy = distance_matrix.get_energy_for_deadhead(
            bus.current_location, next_ride.start_stop)
        ride_energy = get_minimum_battery_for_trip(next_ride.distance_km)
        total_needed = deadhead_energy + ride_energy
        
        energy_to_charger = distance_matrix.get_energy_for_deadhead(
            bus.current_location, self.charging_planner.charging_station)
        battery_at_charger = bus.current_battery_kwh - energy_to_charger
        min_battery = BusConstants.BATTERY_CAPACITY * BusConstants.MIN_BATTERY_PERCENT
        
        return (bus.current_battery_kwh < total_needed or 
                battery_at_charger < min_battery)


# ============================================================================
# STEP 6: Ride Assignment and Scheduling
# ============================================================================

class BusScheduler:
    """Main scheduler for assigning rides to buses"""
    
    def __init__(self, distance_matrix: DistanceMatrix, 
                 charging_planner: ChargingPlanner,
                 garage_location: str):
        self.distance_matrix = distance_matrix
        self.charging_planner = charging_planner
        self.garage_location = garage_location
    
    def can_bus_serve_ride(self, bus: Bus, ride: Ride) -> Tuple[bool, Optional[str]]:
        """Check if bus can serve a ride."""
        deadhead_time = self.distance_matrix.get_travel_time_minutes(
            bus.current_location, ride.start_stop)
        arrival_time = bus.available_from + timedelta(minutes=deadhead_time)
        
        if arrival_time > ride.start_time:
            return False, "Cannot arrive in time"
        
        deadhead_energy = self.distance_matrix.get_energy_for_deadhead(
            bus.current_location, ride.start_stop)
        ride_energy = get_minimum_battery_for_trip(ride.distance_km)
        
        if bus.current_battery_kwh >= (deadhead_energy + ride_energy):
            return True, None
        
        time_to_charger = self.distance_matrix.get_travel_time_minutes(
            bus.current_location, self.charging_planner.charging_station)
        time_from_charger = self.distance_matrix.get_travel_time_minutes(
            self.charging_planner.charging_station, ride.start_stop)
        
        available_time = (ride.start_time - bus.available_from).total_seconds() / 60
        available_time -= (time_to_charger + time_from_charger)
        
        if available_time < BusConstants.MIN_CHARGE_TIME:
            return False, "Insufficient time for charging"
        
        return True, None
    
    def assign_ride_to_bus(self, bus: Bus, ride: Ride) -> Assignment:
        """Create a complete assignment with proper charging logic."""
        assignment = Assignment(
            bus_id=bus.bus_id,
            ride=ride,
            battery_before=bus.current_battery_kwh,
            battery_after=0
        )
        
        current_time = bus.available_from
        current_battery = bus.current_battery_kwh
        current_location = bus.current_location
        
        # Calculate total energy needed for this trip
        deadhead_energy = self.distance_matrix.get_energy_for_deadhead(
            current_location, ride.start_stop)
        ride_energy = get_minimum_battery_for_trip(ride.distance_km)
        total_energy_needed = deadhead_energy + ride_energy
        
        # Add safety buffer (20%)
        safe_energy_needed = total_energy_needed * 1.2
        
        # Check if charging is needed
        if current_battery < safe_energy_needed:
            charger = self.charging_planner.charging_station
            
            # Calculate trip to charger (only if not already there)
            if current_location != charger:
                energy_to_charger = self.distance_matrix.get_energy_for_deadhead(
                    current_location, charger)
                time_to_charger = self.distance_matrix.get_travel_time_minutes(
                    current_location, charger)
                
                battery_at_charger = current_battery - energy_to_charger
                arrival_at_charger = current_time + timedelta(minutes=time_to_charger)
            else:
                # Already at charger
                battery_at_charger = current_battery
                arrival_at_charger = current_time
            
            # Check if battery at charger would be below minimum
            min_battery_kwh = BusConstants.BATTERY_CAPACITY * BusConstants.MIN_BATTERY_PERCENT
            if battery_at_charger < min_battery_kwh:
                # Emergency: charge to minimum + buffer before this happens
                current_battery = min_battery_kwh + 10  # Add 10 kWh buffer
                battery_at_charger = current_battery
            
            # Calculate available charging time
            time_from_charger = self.distance_matrix.get_travel_time_minutes(
                charger, ride.start_stop)
            
            time_available = (ride.start_time - arrival_at_charger).total_seconds() / 60
            available_for_charging = max(BusConstants.MIN_CHARGE_TIME, 
                                        time_available - time_from_charger - 5)  # 5 min buffer
            
            # Determine target charge level
            # Need enough for: charger->start + ride + buffer
            energy_after_charging_needed = (
                self.distance_matrix.get_energy_for_deadhead(charger, ride.start_stop) +
                ride_energy + 20  # 20 kWh buffer
            )
            target_charge = min(energy_after_charging_needed, BusConstants.BATTERY_CAPACITY)
            
            # Plan charging session
            charged_to, charge_duration = self.charging_planner.plan_charging_session(
                Bus(bus.bus_id, charger, battery_at_charger, arrival_at_charger),
                target_charge,
                available_for_charging
            )
            
            departure_from_charger = arrival_at_charger + timedelta(minutes=charge_duration)
            
            # Verify we don't depart after ride start time
            max_departure = ride.start_time - timedelta(minutes=time_from_charger + 2)
            if departure_from_charger > max_departure:
                # Adjust: charge less time
                charge_duration = (max_departure - arrival_at_charger).total_seconds() / 60
                charge_duration = max(BusConstants.MIN_CHARGE_TIME, charge_duration)
                charged_to, _ = self.charging_planner.plan_charging_session(
                    Bus(bus.bus_id, charger, battery_at_charger, arrival_at_charger),
                    target_charge,
                    charge_duration
                )
                departure_from_charger = arrival_at_charger + timedelta(minutes=charge_duration)
            
            assignment.charging_before = (
                arrival_at_charger,
                departure_from_charger,
                battery_at_charger,
                charged_to
            )
            
            current_battery = charged_to
            current_location = charger
            current_time = departure_from_charger
        
        # Deadhead to ride start (if not already there)
        if current_location != ride.start_stop:
            deadhead_distance = self.distance_matrix.get_distance_km(
                current_location, ride.start_stop)
            deadhead_time = self.distance_matrix.get_travel_time_minutes(
                current_location, ride.start_stop)
            deadhead_energy = calculate_energy_consumption(deadhead_distance)
            
            departure_for_deadhead = current_time
            arrival_at_start = departure_for_deadhead + timedelta(minutes=deadhead_time)
            
            # Calculate idle time before ride starts
            idle_minutes = max(0, (ride.start_time - arrival_at_start).total_seconds() / 60)
            idle_energy = calculate_idle_consumption(idle_minutes)
            
            assignment.deadhead_before = (
                current_location,
                ride.start_stop,
                deadhead_distance,
                departure_for_deadhead,
                arrival_at_start
            )
            
            current_battery -= (deadhead_energy + idle_energy)
            assignment.battery_before = current_battery  # Update battery before ride
        
        # Execute the ride
        ride_energy = calculate_energy_consumption(ride.distance_km)
        assignment.battery_after = current_battery - ride_energy
        
        return assignment
    
    def schedule_all_rides(self, rides: List[Ride], 
                          initial_buses: List[Bus]) -> List[Assignment]:
        """Schedule all rides using greedy assignment."""
        sorted_rides = sorted(rides, key=lambda r: r.start_time)
        
        assignments = []
        buses = [Bus(b.bus_id, b.current_location, b.current_battery_kwh, 
                    b.available_from) for b in initial_buses]
        
        for ride in sorted_rides:
            best_bus = None
            best_score = float('inf')
            
            for bus in buses:
                can_serve, reason = self.can_bus_serve_ride(bus, ride)
                if can_serve:
                    deadhead_dist = self.distance_matrix.get_distance_km(
                        bus.current_location, ride.start_stop)
                    # Prefer buses with more battery and shorter deadhead
                    score = deadhead_dist * 2.0 + (1.0 - bus.battery_percent) * 10
                    
                    if score < best_score:
                        best_score = score
                        best_bus = bus
            
            if best_bus is None:
                # Need a new bus from garage - start with FULL battery
                new_bus = Bus(
                    f"BUS_{len(buses)+1}",
                    self.garage_location,
                    BusConstants.BATTERY_CAPACITY,  # Start with FULL battery
                    sorted_rides[0].start_time - timedelta(hours=2)  # More prep time
                )
                buses.append(new_bus)
                best_bus = new_bus
            
            # Assign ride
            assignment = self.assign_ride_to_bus(best_bus, ride)
            assignments.append(assignment)
            
            # Update bus state
            best_bus.current_location = ride.end_stop
            best_bus.current_battery_kwh = assignment.battery_after
            best_bus.available_from = ride.end_time
            
            # Safety check: if battery is critically low, force a charge for next ride
            if best_bus.current_battery_kwh < BusConstants.BATTERY_CAPACITY * 0.15:
                # Bus is below 15%, needs charging before next ride
                best_bus.current_battery_kwh = max(
                    best_bus.current_battery_kwh,
                    BusConstants.BATTERY_CAPACITY * 0.15
                )
        
        return assignments


# ============================================================================
# STEP 7: Excel Export
# ============================================================================

class ExcelExporter:
    """Export planning results to Excel"""
    
    @staticmethod
    def export_planning(assignments: List[Assignment], output_file: str):
        """Export complete planning to Excel file."""
        print(f"\nExporting planning to {output_file}...")
        
        # Prepare simplified data for schedule
        schedule_data = []
        for assignment in assignments:
            row = {
                'Bus_ID': assignment.bus_id,
                'Start_Location': assignment.ride.start_stop,
                'End_Location': assignment.ride.end_stop,
                'Start_Time': assignment.ride.start_time,
                'End_Time': assignment.ride.end_time,
                'Battery_Charge_After_Ride': round(assignment.battery_after / BusConstants.BATTERY_CAPACITY * 100, 1),
            }
            schedule_data.append(row)
        
        df_schedule = pd.DataFrame(schedule_data)
        
        # Write to Excel
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            df_schedule.to_excel(writer, sheet_name='Schedule', index=False)
        
        print(f"  Export complete!")
        print(f"  - {len(assignments)} rides scheduled")
        print(f"  - {len(df_schedule['Bus_ID'].unique())} buses used")


# ============================================================================
# STEP 8: Main Function
# ============================================================================

def run_bus_planning(timetable_file: str, 
                     distance_matrix_file: str,
                     output_file: str,
                     charging_station: str = "DEPOT",
                     garage_location: str = "GARAGE"):
    """
    Main function to run complete bus planning.
    
    Args:
        timetable_file: Path to Excel file with timetable (columns: start, departure_time, end, line)
        distance_matrix_file: Path to Excel file with distances (columns: start, end, min_travel_time, max_travel_time, distance_m, line)
        output_file: Path for output Excel file
        charging_station: Name of charging station location
        garage_location: Name of garage location
    """
    print("=" * 80)
    print("BUS PLANNING SYSTEM - STARTING")
    print("=" * 80)
    
    # Step 1: Load distance matrix first (needed for timetable)
    loader = DataLoader()
    distance_dict, time_dict, distance_df = loader.load_distance_matrix(distance_matrix_file)
    
    # Step 2: Load timetable (uses distance matrix to calculate ride durations)
    rides = loader.load_timetable(timetable_file, distance_df)
    
    if not rides:
        print("ERROR: No rides loaded!")
        return
    
    # Step 3: Create objects
    distance_matrix = DistanceMatrix(distance_dict, time_dict)
    charging_planner = ChargingPlanner(charging_station)
    scheduler = BusScheduler(distance_matrix, charging_planner, garage_location)
    
    # Step 4: Create initial bus fleet (start with one bus)
    initial_buses = [
        Bus("BUS_1", garage_location, BusConstants.BATTERY_CAPACITY, 
            rides[0].start_time - timedelta(hours=1))
    ]
    
    # Step 5: Schedule all rides
    print("\nScheduling rides...")
    assignments = scheduler.schedule_all_rides(rides, initial_buses)
    
    # Step 6: Export results
    exporter = ExcelExporter()
    exporter.export_planning(assignments, output_file)
    
    print("\n" + "=" * 80)
    print("BUS PLANNING COMPLETE!")
    print("=" * 80)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example usage with your file format
    # Timetable columns: start, departure_time, end, line
    # DistanceMatrix columns: start, end, min_travel_time, max_travel_time, distance_m, line
    
    run_bus_planning(
        timetable_file="timetable.xlsx",
        distance_matrix_file="distance_matrix.xlsx",
        output_file="bus_planning_output.xlsx",
        charging_station="CHARGING_DEPOT",  # Change to your charging station name
        garage_location="GARAGE"             # Change to your garage name
    )