"""
Complete Bus Planning System met correcte Idle Tracking
=======================================================
Verbeteringen:
- Idle periods worden expliciet bijgehouden
- Uitgebreide Excel export met alle events (rides, charging, deadhead, idle)
- Gantt chart compatible output
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


# ============================================================================
# Battery Calculation Functions
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


# ============================================================================
# Data Loading
# ============================================================================

class DataLoader:
    """Loads data from Excel files or DataFrames"""
    
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
                    distance_meters=distance_m
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
                line = str(row['line']).strip()
                
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


# ============================================================================
# Distance Matrix
# ============================================================================

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
    
    def get_energy_for_deadhead(self, from_stop: str, to_stop: str) -> float:
        distance_km = self.get_distance_km(from_stop, to_stop)
        return calculate_energy_consumption(distance_km)


# ============================================================================
# Charging Planner
# ============================================================================

class ChargingPlanner:
    """Manages charging decisions and calculations"""
    
    def __init__(self, charging_station_location: str):
        self.charging_station = charging_station_location
    
    def plan_charging_session(self, bus: Bus, target_kwh: float,
                             available_time: float) -> Tuple[float, float]:
        """Plan a charging session within available time."""
        ideal_time = calculate_charging_time(bus.current_battery_kwh, target_kwh)
        
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


# ============================================================================
# Bus Scheduler
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
        ride_energy = calculate_energy_consumption(ride.distance_km)
        
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
        """Create a complete assignment with ALL events tracked including IDLE."""
        assignment = Assignment(bus_id=bus.bus_id, ride=ride, events=[])
        
        current_time = bus.available_from
        current_battery = bus.current_battery_kwh
        current_location = bus.current_location
        
        # Calculate energy needs
        deadhead_energy = self.distance_matrix.get_energy_for_deadhead(
            current_location, ride.start_stop)
        ride_energy = calculate_energy_consumption(ride.distance_km)
        total_energy_needed = deadhead_energy + ride_energy
        safe_energy_needed = total_energy_needed * 1.2
        
        # CHARGING if needed
        if current_battery < safe_energy_needed:
            charger = self.charging_planner.charging_station
            
            if current_location != charger:
                energy_to_charger = self.distance_matrix.get_energy_for_deadhead(
                    current_location, charger)
                time_to_charger = self.distance_matrix.get_travel_time_minutes(
                    current_location, charger)
                
                battery_at_charger = current_battery - energy_to_charger
                arrival_at_charger = current_time + timedelta(minutes=time_to_charger)
            else:
                battery_at_charger = current_battery
                arrival_at_charger = current_time
            
            min_battery_kwh = BusConstants.BATTERY_CAPACITY * BusConstants.MIN_BATTERY_PERCENT
            if battery_at_charger < min_battery_kwh:
                battery_at_charger = min_battery_kwh + 10
            
            time_from_charger = self.distance_matrix.get_travel_time_minutes(
                charger, ride.start_stop)
            time_available = (ride.start_time - arrival_at_charger).total_seconds() / 60
            available_for_charging = max(BusConstants.MIN_CHARGE_TIME, 
                                        time_available - time_from_charger - 5)
            
            energy_after_charging_needed = (
                self.distance_matrix.get_energy_for_deadhead(charger, ride.start_stop) +
                ride_energy + 20
            )
            target_charge = min(energy_after_charging_needed, BusConstants.BATTERY_CAPACITY)
            
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
                energy_consumed=-(charged_to - battery_at_charger)  # Negative = gained energy
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
            deadhead_energy = calculate_energy_consumption(deadhead_distance)
            
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
            idle_energy = calculate_idle_consumption(idle_minutes)
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
        ride_energy = calculate_energy_consumption(ride.distance_km)
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
            ride_id=ride.ride_id
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
                    sorted_rides[0].start_time - timedelta(hours=2)
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


# ============================================================================
# Excel Export met Gantt-compatible format
# ============================================================================

class ExcelExporter:
    """Export planning results to Excel with Gantt chart data"""
    
    @staticmethod
    def export_planning(assignments: List[Assignment], output_file: str):
        """Export complete planning including ALL events for Gantt chart."""
        print(f"\nExporting planning to {output_file}...")
        
        # Create detailed event log for Gantt chart
        all_events = []
        for assignment in assignments:
            for event in assignment.events:
                all_events.append({
                    'Bus_ID': event.bus_id,
                    'Event_Type': event.event_type,
                    'Start_Time': event.start_time,
                    'End_Time': event.end_time,
                    'Duration_Min': round(event.duration_minutes, 1),
                    'Start_Location': event.start_location,
                    'End_Location': event.end_location,
                    'Distance_km': round(event.distance_km, 2),
                    'Battery_Before_%': round(event.battery_percent_before, 1),
                    'Battery_After_%': round(event.battery_percent_after, 1),
                    'Battery_Before_kWh': round(event.battery_before, 1),
                    'Battery_After_kWh': round(event.battery_after, 1),
                    'Energy_Consumed_kWh': round(event.energy_consumed, 2),
                    'Ride_ID': event.ride_id if event.ride_id else ''
                })
        
        df_events = pd.DataFrame(all_events)
        
        # Create summary by bus
        summary_data = []
        for bus_id in df_events['Bus_ID'].unique():
            bus_events = df_events[df_events['Bus_ID'] == bus_id]
            
            rides = bus_events[bus_events['Event_Type'] == 'ride']
            charging = bus_events[bus_events['Event_Type'] == 'charging']
            deadhead = bus_events[bus_events['Event_Type'] == 'deadhead']
            idle = bus_events[bus_events['Event_Type'] == 'idle']
            
            summary_data.append({
                'Bus_ID': bus_id,
                'Total_Rides': len(rides),
                'Total_Distance_km': rides['Distance_km'].sum(),
                'Charging_Sessions': len(charging),
                'Total_Charging_Time_Min': charging['Duration_Min'].sum(),
                'Deadhead_Trips': len(deadhead),
                'Total_Deadhead_km': deadhead['Distance_km'].sum(),
                'Idle_Periods': len(idle),
                'Total_Idle_Time_Min': idle['Duration_Min'].sum(),
                'Total_Energy_Consumed_kWh': bus_events[bus_events['Energy_Consumed_kWh'] > 0]['Energy_Consumed_kWh'].sum()
            })
        
        df_summary = pd.DataFrame(summary_data)
        
        # Write to Excel with multiple sheets
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            df_events.to_excel(writer, sheet_name='All_Events_Gantt', index=False)
            df_summary.to_excel(writer, sheet_name='Bus_Summary', index=False)
        
        print(f"  Export complete!")
        print(f"  - {len(assignments)} rides scheduled")
        print(f"  - {len(df_summary)} buses used")
        print(f"  - {len(all_events)} total events tracked")
        print(f"    * {len([e for e in all_events if e['Event_Type'] == 'ride'])} rides")
        print(f"    * {len([e for e in all_events if e['Event_Type'] == 'charging'])} charging sessions")
        print(f"    * {len([e for e in all_events if e['Event_Type'] == 'deadhead'])} deadhead trips")
        print(f"    * {len([e for e in all_events if e['Event_Type'] == 'idle'])} idle periods")


# ============================================================================
# Main Function
# ============================================================================

def run_bus_planning_from_dataframes(timetable_df: pd.DataFrame,
                                     distance_matrix_df: pd.DataFrame,
                                     output_file: str,
                                     charging_station: str = "DEPOT",
                                     garage_location: str = "GARAGE"):
    """
    Run complete bus planning from DataFrames.
    Returns assignments for further analysis.
    """
    print("=" * 80)
    print("BUS PLANNING SYSTEM - STARTING")
    print("=" * 80)
    
    loader = DataLoader()
    distance_dict, time_dict, distance_df = loader.load_distance_matrix_from_df(distance_matrix_df)
    rides = loader.load_timetable_from_df(timetable_df, distance_df)
    
    if not rides:
        print("ERROR: No rides loaded!")
        return None
    
    distance_matrix = DistanceMatrix(distance_dict, time_dict)
    charging_planner = ChargingPlanner(charging_station)
    scheduler = BusScheduler(distance_matrix, charging_planner, garage_location)
    
    initial_buses = [
        Bus("BUS_1", garage_location, BusConstants.BATTERY_CAPACITY, 
            rides[0].start_time - timedelta(hours=1))
    ]
    
    print(f"\nScheduling {len(rides)} rides...")
    assignments = scheduler.schedule_all_rides(rides, initial_buses)
    
    exporter = ExcelExporter()
    exporter.export_planning(assignments, output_file)
    
    print("\n" + "=" * 80)
    print("BUS PLANNING COMPLETE!")
    print("=" * 80)
    
    return assignments
