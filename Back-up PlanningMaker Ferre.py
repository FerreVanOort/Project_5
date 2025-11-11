## Back-up PlanningMaker Ferre

# Imports
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta


# Data Structures
@dataclass
class BusConstants:
    """Constants for bus specifications"""
    ORIGINAL_BATTERY_CAPACITY = 300  # kWh
    AGING_FACTOR = 0.90  # SOH percentage
    BATTERY_CAPACITY = ORIGINAL_BATTERY_CAPACITY * AGING_FACTOR  # Actual capacity of battery in kWh
    
    CONSUMPTION_PER_KM = 1.2  # kW/km
    IDLE_CONSUMPTION_PER_HOUR = 5  # kWh per hour
    
    MIN_BATTERY_PERCENT = 0.10  # 10% minimum battery
    MIN_CHARGE_TIME = 15  # in minutes
    
    FAST_CHARGE_RATE = 450  # kWh per hour until 90% charge
    SLOW_CHARGE_RATE = 60   # kWh per hour after 90% charge
    FAST_CHARGE_THRESHOLD = 0.90  # Switch to slow charging at 90% of current capacity


@dataclass
class Ride:
    """Represents a scheduled ride from timetable"""
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
    """Represents a bus with its current state"""
    bus_id: str
    current_location: str
    current_battery_kwh: float
    available_from: datetime
    
    @property
    def battery_percent(self) -> float:
        return self.current_battery_kwh / BusConstants.BATTERY_CAPACITY


@dataclass
class Event:
    """Represents any bus event (ride, charging, deadhead, idle)"""
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
    
    @property
    def battery_percent_before(self) -> float:
        return self.battery_before / BusConstants.BATTERY_CAPACITY * 100
    
    @property
    def battery_percent_after(self) -> float:
        return self.battery_after / BusConstants.BATTERY_CAPACITY * 100


@dataclass
class Assignment:
    """Represents a complete ride assignment with all events"""
    bus_id: str
    ride: Ride
    events: List[Event] = field(default_factory=list)
    
    @property
    def battery_before_ride(self) -> float:
        """Battery level just before the actual ride starts"""
        ride_event = next((e for e in self.events if e.event_type == 'ride'), None)
        return ride_event.battery_before if ride_event else 0
    
    @property
    def battery_after_ride(self) -> float:
        """Battery level just after the actual ride ends"""
        ride_event = next((e for e in self.events if e.event_type == 'ride'), None)
        return ride_event.battery_after if ride_event else 0


# Battery Calculation Functions
def calculate_energy_consumption(distance_km: float, consumption_per_km: float = None) -> float:
    """Calculate energy consumption for a given distance."""
    if consumption_per_km is None:
        consumption_per_km = BusConstants.CONSUMPTION_PER_KM
    return distance_km * consumption_per_km


def calculate_idle_consumption(minutes: float, idle_per_hour: float = None) -> float:
    """Calculate energy consumption while idle."""
    if idle_per_hour is None:
        idle_per_hour = BusConstants.IDLE_CONSUMPTION_PER_HOUR
    hours = minutes / 60
    return hours * idle_per_hour


def calculate_charging_time(current_kwh: float, target_kwh: float, 
                           fast_rate: float = None, slow_rate: float = None) -> float:
    """Calculate time needed to charge from current to target level."""
    if fast_rate is None:
        fast_rate = BusConstants.FAST_CHARGE_RATE
    if slow_rate is None:
        slow_rate = BusConstants.SLOW_CHARGE_RATE
        
    if target_kwh <= current_kwh:
        return 0
    
    threshold_kwh = BusConstants.BATTERY_CAPACITY * BusConstants.FAST_CHARGE_THRESHOLD
    total_minutes = 0
    
    # Fast charging up to 90%
    if current_kwh < threshold_kwh:
        fast_charge_amount = min(target_kwh, threshold_kwh) - current_kwh
        fast_charge_hours = fast_charge_amount / fast_rate
        total_minutes += fast_charge_hours * 60
    
    # Slow charging above 90%
    if target_kwh > threshold_kwh and current_kwh < target_kwh:
        slow_charge_start = max(current_kwh, threshold_kwh)
        slow_charge_amount = target_kwh - slow_charge_start
        slow_charge_hours = slow_charge_amount / slow_rate
        total_minutes += slow_charge_hours * 60
    
    return max(total_minutes, BusConstants.MIN_CHARGE_TIME)


# Data Loading
class DataLoader:
    """Loads data from DataFrames"""
    
    @staticmethod
    def load_timetable_from_df(df: pd.DataFrame, distance_matrix_df: pd.DataFrame = None) -> List[Ride]:
        """Load timetable from DataFrame"""
        rides = []
        first_departure = None
        
        for idx, row in df.iterrows():
            try:
                start_stop = str(row['start']).strip()
                end_stop = str(row['end']).strip()
                line = str(row['line']).strip()
                
                departure_time = pd.to_datetime(row['departure_time'])
                
                if first_departure is None:
                    first_departure = departure_time
                
                if departure_time.hour < 4 and first_departure.hour >= 4:
                    departure_time = departure_time + timedelta(days=1)
                
                if distance_matrix_df is not None:
                    match = distance_matrix_df[
                        (distance_matrix_df['start'].astype(str).str.strip() == start_stop) & 
                        (distance_matrix_df['end'].astype(str).str.strip() == end_stop) &
                        (distance_matrix_df['line'].astype(str).str.strip() == line)
                    ]
                    
                    if match.empty:
                        match = distance_matrix_df[
                            (distance_matrix_df['start'].astype(str).str.strip() == start_stop) & 
                            (distance_matrix_df['end'].astype(str).str.strip() == end_stop)
                        ]
                    
                    if not match.empty:
                        distance_m = float(match.iloc[0]['distance_m'])
                        min_time = float(match.iloc[0]['min_travel_time'])
                        max_time = float(match.iloc[0]['max_travel_time'])
                        avg_travel_time = round((min_time + max_time) / 2)
                    else:
                        distance_m = 5000
                        avg_travel_time = 10
                else:
                    distance_m = 5000
                    avg_travel_time = 10
                
                arrival_time = departure_time + timedelta(minutes=avg_travel_time)
                ride_id = f"{line}_{start_stop}_{departure_time.strftime('%H%M')}_{idx}"
                
                ride = Ride(
                    ride_id=ride_id,
                    start_stop=start_stop,
                    end_stop=end_stop,
                    start_time=departure_time,
                    end_time=arrival_time,
                    distance_meters=distance_m,
                    line=line
                )
                rides.append(ride)
            except Exception as e:
                continue
        
        return rides
    
    @staticmethod
    def load_distance_matrix_from_df(df: pd.DataFrame) -> Tuple[Dict, Dict, pd.DataFrame]:
        """Load distance and time matrices from DataFrame"""
        distance_dict = {}
        time_dict = {}
        
        for idx, row in df.iterrows():
            try:
                start = str(row['start']).strip()
                end = str(row['end']).strip()
                line = str(row['line']).strip() if 'line' in row else ""
                
                distance_m = float(row['distance_m'])
                min_time = float(row['min_travel_time'])
                max_time = float(row['max_travel_time'])
                avg_time = round((min_time + max_time) / 2)
                
                key = (start, end)
                key_with_line = (start, end, line)
                
                distance_dict[key] = distance_m
                distance_dict[key_with_line] = distance_m
                time_dict[key] = avg_time
                time_dict[key_with_line] = avg_time
                
            except Exception as e:
                continue
        
        return distance_dict, time_dict, df


# Distance Matrix
class DistanceMatrix:
    """Handles distance and travel time lookups between stops"""
    
    def __init__(self, distance_matrix: Dict[Tuple[str, str], float],
                 time_matrix: Dict[Tuple[str, str], float]):
        self.distance_matrix = distance_matrix
        self.time_matrix = time_matrix
    
    def get_distance_km(self, from_stop: str, to_stop: str) -> float:
        if from_stop == to_stop:
            return 0.0
        distance = self.distance_matrix.get((from_stop, to_stop), 10000)
        return distance / 1000
    
    def get_travel_time_minutes(self, from_stop: str, to_stop: str) -> float:
        if from_stop == to_stop:
            return 0.0
        time = self.time_matrix.get((from_stop, to_stop), None)
        if time is None:
            distance_km = self.get_distance_km(from_stop, to_stop)
            return distance_km / 30 * 60
        return time
    
    def get_energy_for_deadhead(self, from_stop: str, to_stop: str, 
                               consumption_per_km: float = None) -> float:
        distance_km = self.get_distance_km(from_stop, to_stop)
        return calculate_energy_consumption(distance_km, consumption_per_km)


# Charging Planner
class ChargingPlanner:
    """Manages charging decisions and calculations"""
    
    def __init__(self, charging_station_location: str, fast_rate: float = None, slow_rate: float = None):
        self.charging_station = charging_station_location
        self.fast_rate = fast_rate if fast_rate else BusConstants.FAST_CHARGE_RATE
        self.slow_rate = slow_rate if slow_rate else BusConstants.SLOW_CHARGE_RATE
    
    def plan_charging_session(self, bus: Bus, target_kwh: float,
                             available_time: float) -> Tuple[float, float]:
        """Plan a charging session within available time."""
        ideal_time = calculate_charging_time(bus.current_battery_kwh, target_kwh, 
                                            self.fast_rate, self.slow_rate)
        
        if ideal_time <= available_time:
            return target_kwh, ideal_time
        
        achieved_kwh = self._charge_for_duration(bus.current_battery_kwh, available_time)
        return achieved_kwh, available_time
    
    def _charge_for_duration(self, start_kwh: float, minutes: float) -> float:
        """Calculate battery level after charging for given duration"""
        threshold_kwh = BusConstants.BATTERY_CAPACITY * BusConstants.FAST_CHARGE_THRESHOLD
        hours = minutes / 60
        current = start_kwh
        
        if current < threshold_kwh:
            fast_charge_capacity = threshold_kwh - current
            fast_charge_possible = self.fast_rate * hours
            
            if fast_charge_possible <= fast_charge_capacity:
                return current + fast_charge_possible
            else:
                time_for_fast = fast_charge_capacity / self.fast_rate
                remaining_hours = hours - time_for_fast
                current = threshold_kwh
                hours = remaining_hours
        
        slow_charge = self.slow_rate * hours
        return min(current + slow_charge, BusConstants.BATTERY_CAPACITY)


# Bus Scheduler
class BusScheduler:
    """Main scheduler for assigning rides to buses"""
    
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
        """Check if bus can serve a ride, including time for charging if needed."""
        # Calculates if there is need for charging
        deadhead_energy = self.distance_matrix.get_energy_for_deadhead(
            bus.current_location, ride.start_stop, self.consumption_per_km)
        ride_energy = calculate_energy_consumption(ride.distance_km, self.consumption_per_km)
        total_energy_needed = deadhead_energy + ride_energy
        
        battery_percent = bus.current_battery_kwh / BusConstants.BATTERY_CAPACITY
        safe_energy_needed = total_energy_needed + 15  # 15 kWh buffer
        needs_charging = (battery_percent < 0.30) or (bus.current_battery_kwh < safe_energy_needed)
        
        if needs_charging:
            # Calculates total time needed to charger + charging + from charger to ride
            charger = self.charging_planner.charging_station
            
            # Time to get to charger
            if bus.current_location != charger:
                time_to_charger = self.distance_matrix.get_travel_time_minutes(
                    bus.current_location, charger)
                energy_to_charger = self.distance_matrix.get_energy_for_deadhead(
                    bus.current_location, charger, self.consumption_per_km)
                battery_at_charger = bus.current_battery_kwh - energy_to_charger
            else:
                time_to_charger = 0
                battery_at_charger = bus.current_battery_kwh
            
            # Minimum charging time needed
            energy_needed_after_charging = (
                self.distance_matrix.get_energy_for_deadhead(charger, ride.start_stop, self.consumption_per_km) +
                ride_energy + 20
            )
            target_charge = min(
                max(energy_needed_after_charging, BusConstants.BATTERY_CAPACITY * 0.80),
                BusConstants.BATTERY_CAPACITY
            )
            min_charge_time = calculate_charging_time(
                battery_at_charger, target_charge,
                self.charging_planner.fast_rate, self.charging_planner.slow_rate
            )
            
            # Time from charger to ride start
            time_from_charger = self.distance_matrix.get_travel_time_minutes(
                charger, ride.start_stop)
            
            # Total time needed
            total_time_needed = time_to_charger + min_charge_time + time_from_charger + 5  # 5 min buffer
            available_time = (ride.start_time - bus.available_from).total_seconds() / 60
            
            if available_time < total_time_needed:
                return False, f"Insufficient time for charging route (need {total_time_needed:.0f} min, have {available_time:.0f} min)"
            
            return True, None
        else:
            # No charging needed, just check direct deadhead time
            deadhead_time = self.distance_matrix.get_travel_time_minutes(
                bus.current_location, ride.start_stop)
            arrival_time = bus.available_from + timedelta(minutes=deadhead_time)
            
            if arrival_time > ride.start_time:
                return False, "Cannot arrive in time"
            
            # Checks if battery is sufficient
            if bus.current_battery_kwh >= (deadhead_energy + ride_energy):
                return True, None
            else:
                return False, "Insufficient battery and no time for charging"
    
    def assign_ride_to_bus(self, bus: Bus, ride: Ride) -> Assignment:
        """Create a complete assignment with ALL events tracked including IDLE."""
        assignment = Assignment(bus_id=bus.bus_id, ride=ride, events=[])
        
        current_time = bus.available_from
        current_battery = bus.current_battery_kwh
        current_location = bus.current_location
        
        # Calculates energy needs
        deadhead_energy = self.distance_matrix.get_energy_for_deadhead(
            current_location, ride.start_stop, self.consumption_per_km)
        ride_energy = calculate_energy_consumption(ride.distance_km, self.consumption_per_km)
        total_energy_needed = deadhead_energy + ride_energy
        
        # Charging trigger: charge if below 10% OR if insufficient for this ride + small buffer
        battery_percent = current_battery / BusConstants.BATTERY_CAPACITY
        safe_energy_needed = total_energy_needed + 15  # 15 kWh buffer
        needs_charging = (battery_percent < 0.10) or (current_battery < safe_energy_needed)
        
        # CHARGING if needed
        if needs_charging:
            charger = self.charging_planner.charging_station
            
            # Add DEADHEAD to charger if needed (as separate event)
            if current_location != charger:
                deadhead_to_charger_dist = self.distance_matrix.get_distance_km(
                    current_location, charger)
                deadhead_to_charger_time = self.distance_matrix.get_travel_time_minutes(
                    current_location, charger)
                energy_to_charger = calculate_energy_consumption(deadhead_to_charger_dist, self.consumption_per_km)
                
                battery_after_deadhead_to_charger = current_battery - energy_to_charger
                arrival_at_charger = current_time + timedelta(minutes=deadhead_to_charger_time)
                
                # Add deadhead to charger event
                deadhead_to_charger_event = Event(
                    event_type='deadhead',
                    bus_id=bus.bus_id,
                    start_time=current_time,
                    end_time=arrival_at_charger,
                    start_location=current_location,
                    end_location=charger,
                    battery_before=current_battery,
                    battery_after=battery_after_deadhead_to_charger,
                    distance_km=deadhead_to_charger_dist,
                    energy_consumed=energy_to_charger
                )
                assignment.events.append(deadhead_to_charger_event)
                
                battery_at_charger = battery_after_deadhead_to_charger
            else:
                battery_at_charger = current_battery
                arrival_at_charger = current_time
            
            min_battery_kwh = BusConstants.BATTERY_CAPACITY * BusConstants.MIN_BATTERY_PERCENT
            if battery_at_charger < min_battery_kwh:
                battery_at_charger = min_battery_kwh + 10
            
            # Calculates available charging time
            time_from_charger = self.distance_matrix.get_travel_time_minutes(
                charger, ride.start_stop)
            time_available = (ride.start_time - arrival_at_charger).total_seconds() / 60
            available_for_charging = max(BusConstants.MIN_CHARGE_TIME, 
                                        time_available - time_from_charger - 5)
            
            # Determines target charge level - aims for 90% for efficiency
            energy_after_charging_needed = (
                self.distance_matrix.get_energy_for_deadhead(charger, ride.start_stop, self.consumption_per_km) +
                ride_energy + 15  # Small buffer
            )
            min_target = energy_after_charging_needed
            optimal_target = BusConstants.BATTERY_CAPACITY * 0.90
            target_charge = min(max(min_target, optimal_target), BusConstants.BATTERY_CAPACITY)
            
            charged_to, charge_duration = self.charging_planner.plan_charging_session(
                Bus(bus.bus_id, charger, battery_at_charger, arrival_at_charger),
                target_charge,
                available_for_charging
            )
            
            departure_from_charger = arrival_at_charger + timedelta(minutes=charge_duration)
            
            # Add CHARGING EVENT
            charging_event = Event(
                event_type='charging',
                bus_id=bus.bus_id,
                start_time=arrival_at_charger,
                end_time=departure_from_charger,
                start_location=charger,
                end_location=charger,
                battery_before=battery_at_charger,
                battery_after=charged_to,
                energy_consumed=-(charged_to - battery_at_charger)
            )
            assignment.events.append(charging_event)
            
            current_battery = charged_to
            current_location = charger
            current_time = departure_from_charger
        
        # DEADHEAD to ride start (if needed)
        if current_location != ride.start_stop:
            deadhead_distance = self.distance_matrix.get_distance_km(
                current_location, ride.start_stop)
            deadhead_time = self.distance_matrix.get_travel_time_minutes(
                current_location, ride.start_stop)
            deadhead_energy = calculate_energy_consumption(deadhead_distance, self.consumption_per_km)
            
            departure_for_deadhead = current_time
            arrival_at_start = departure_for_deadhead + timedelta(minutes=deadhead_time)
            battery_after_deadhead = current_battery - deadhead_energy
            
            # Add DEADHEAD EVENT
            deadhead_event = Event(
                event_type='deadhead',
                bus_id=bus.bus_id,
                start_time=departure_for_deadhead,
                end_time=arrival_at_start,
                start_location=current_location,
                end_location=ride.start_stop,
                battery_before=current_battery,
                battery_after=battery_after_deadhead,
                distance_km=deadhead_distance,
                energy_consumed=deadhead_energy
            )
            assignment.events.append(deadhead_event)
            
            current_battery = battery_after_deadhead
            current_location = ride.start_stop
            current_time = arrival_at_start
        
        # IDLE period (if any)
        idle_seconds = (ride.start_time - current_time).total_seconds()
        if idle_seconds > 60:  # More than 1 minute
            idle_minutes = idle_seconds / 60
            idle_energy = calculate_idle_consumption(idle_minutes, self.idle_per_hour)
            battery_after_idle = current_battery - idle_energy
            
            # Add IDLE EVENT
            idle_event = Event(
                event_type='idle',
                bus_id=bus.bus_id,
                start_time=current_time,
                end_time=ride.start_time,
                start_location=ride.start_stop,
                end_location=ride.start_stop,
                battery_before=current_battery,
                battery_after=battery_after_idle,
                energy_consumed=idle_energy
            )
            assignment.events.append(idle_event)
            
            current_battery = battery_after_idle
            current_time = ride.start_time
        
        # RIDE execution
        ride_energy = calculate_energy_consumption(ride.distance_km, self.consumption_per_km)
        battery_after_ride = current_battery - ride_energy
        
        ride_event = Event(
            event_type='ride',
            bus_id=bus.bus_id,
            start_time=ride.start_time,
            end_time=ride.end_time,
            start_location=ride.start_stop,
            end_location=ride.end_stop,
            battery_before=current_battery,
            battery_after=battery_after_ride,
            distance_km=ride.distance_km,
            energy_consumed=ride_energy,
            ride_id=ride.ride_id,
            line=ride.line
        )
        assignment.events.append(ride_event)
        
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
                    score = deadhead_dist * 2.0 + (1.0 - bus.battery_percent) * 10
                    
                    if score < best_score:
                        best_score = score
                        best_bus = bus
            
            if best_bus is None:
                new_bus = Bus(
                    f"BUS_{len(buses)+1}",
                    self.garage_location,
                    BusConstants.BATTERY_CAPACITY,
                    ride.start_time - timedelta(minutes=5)  # Start 5 min before first ride
                )
                buses.append(new_bus)
                best_bus = new_bus
            
            # Assign ride with ALL events
            assignment = self.assign_ride_to_bus(best_bus, ride)
            assignments.append(assignment)
            
            # Update bus state
            best_bus.current_location = ride.end_stop
            best_bus.current_battery_kwh = assignment.battery_after_ride
            best_bus.available_from = ride.end_time
            
            if best_bus.current_battery_kwh < BusConstants.BATTERY_CAPACITY * 0.15:
                best_bus.current_battery_kwh = max(
                    best_bus.current_battery_kwh,
                    BusConstants.BATTERY_CAPACITY * 0.15
                )
        
        return assignments


# Export Functions
def assignments_to_dataframe(assignments: List[Assignment]) -> pd.DataFrame:
    """Convert assignments to DataFrame for export or display."""
    all_events = []
    for assignment in assignments:
        for event in assignment.events:
            all_events.append({
                'start_location': event.start_location,
                'end_location': event.end_location,
                'start_time': event.start_time,
                'end_time': event.end_time,
                'activity': event.event_type,
                'line': event.line if event.line else '',
                'energy_consumption': round(event.energy_consumed, 2),
                'bus': event.bus_id
            })
    
    return pd.DataFrame(all_events)


# Main Planning Function
def create_bus_planning(timetable_df: pd.DataFrame,
                       distance_matrix_df: pd.DataFrame,
                       charging_station: str = "DEPOT",
                       garage_location: str = "GARAGE",
                       driving_usage: float = 1.2,
                       idle_usage: float = 5.0,
                       charging_speed: float = 450.0,
                       soh: float = 90.0,
                       startbat: float = 100.0) -> pd.DataFrame:
    """
    Main function to create bus planning from timetable and distance matrix.
    
    Input:
        timetable_df: DataFrame with columns: start, departure_time, end, line
        distance_matrix_df: DataFrame with columns: start, end, min_travel_time, max_travel_time, distance_m, line
        charging_station: Name of charging station location
        garage_location: Name of garage location
        driving_usage: Energy consumption per km (kWh/km)
        idle_usage: Energy consumption per hour while idle (kWh/h)
        charging_speed: Fast charging speed (kWh/h)
        soh: State of Health (percentage)
        startbat: Starting battery percentage
    
    Output:
        DataFrame with complete planning including all events
    """
    # Update constants based on parameters
    BusConstants.CONSUMPTION_PER_KM = driving_usage
    BusConstants.IDLE_CONSUMPTION_PER_HOUR = idle_usage
    BusConstants.FAST_CHARGE_RATE = charging_speed
    BusConstants.AGING_FACTOR = soh / 100.0
    BusConstants.BATTERY_CAPACITY = BusConstants.ORIGINAL_BATTERY_CAPACITY * BusConstants.AGING_FACTOR
    
    # Load data
    loader = DataLoader()
    distance_dict, time_dict, distance_df = loader.load_distance_matrix_from_df(distance_matrix_df)
    rides = loader.load_timetable_from_df(timetable_df, distance_df)
    
    if not rides:
        raise ValueError("No rides loaded from timetable!")
    
    # Creates planning objects
    distance_matrix = DistanceMatrix(distance_dict, time_dict)
    charging_planner = ChargingPlanner(charging_station, charging_speed, 60)
    scheduler = BusScheduler(distance_matrix, charging_planner, garage_location, 
                            driving_usage, idle_usage)
    
    # Creates initial bus with starting battery level
    start_battery_kwh = BusConstants.BATTERY_CAPACITY * (startbat / 100.0)
    initial_buses = [
        Bus("BUS_1", garage_location, start_battery_kwh, 
            rides[0].start_time - timedelta(minutes=5))  # Start 5 min before first ride
    ]
    
    # Schedules all rides
    assignments = scheduler.schedule_all_rides(rides, initial_buses)
    
    # Converts to DataFrame
    planning_df = assignments_to_dataframe(assignments)
    
    return planning_df