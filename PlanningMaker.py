# PlanningMaker.py

from dataclasses import dataclass, field
from datetime import datetime, timedelta
import pandas as pd
import math


# =========================
# 1. Constants / Parameters
# =========================
class BusConstants:
    # Deze waardes kun je later updaten vanuit Streamlit voordat je gaat plannen
    BATTERY_CAPACITY_NOMINAL = 300.0        # kWh fysiek pack
    SOH_PERCENT = 90.0                      # %
    START_BAT_PERCENT = 100.0               # % van usable capaciteit bij start dag
    MIN_BAT_PERCENT  = 10.0                 # % van usable capaciteit ondergrens
    CONSUMPTION_PER_KM = 1.2                # kWh/km
    CHARGING_POWER_KW = 450.0               # kWh per uur
    IDLE_USAGE_KW = 5.0                     # kWh per uur (optioneel)
    MIN_CHARGE_MIN = 15                     # minimaal 15 min laden
    GARAGE_NAME = "ehvgar"

    @classmethod
    def usable_capacity_kwh(cls):
        # Beschikbare capaciteit afhankelijk van SOH
        return cls.BATTERY_CAPACITY_NOMINAL * (cls.SOH_PERCENT / 100.0)

    @classmethod
    def start_energy_kwh(cls):
        # startbat% sla je op t.o.v. usable_capacity
        return cls.usable_capacity_kwh() * (cls.START_BAT_PERCENT / 100.0)

    @classmethod
    def min_energy_kwh(cls):
        # minbat% sla je op t.o.v. usable_capacity
        return cls.usable_capacity_kwh() * (cls.MIN_BAT_PERCENT / 100.0)


# =========================
# 2. Data classes
# =========================
@dataclass
class Ride:
    line: str
    start_stop: str
    end_stop: str
    start_time: datetime
    end_time: datetime
    distance_km: float

@dataclass
class AssignmentRecord:
    bus_id: str
    ride: Ride
    battery_before: float
    battery_after: float
    deadhead_before: tuple | None = None
    charging_before: tuple | None = None
    # deadhead_before: (from_loc, to_loc, dist_km, depart_dt, arrive_dt)
    # charging_before: (arrival_dt, leave_dt, bat_before, bat_after)

@dataclass
class Bus:
    bus_id: str
    current_location: str
    battery_kwh: float
    current_time: datetime
    history: list[AssignmentRecord] = field(default_factory=list)

    def clone(self):
        # handig voor what-if
        return Bus(
            bus_id=self.bus_id,
            current_location=self.current_location,
            battery_kwh=self.battery_kwh,
            current_time=self.current_time,
            history=list(self.history)
        )


# =========================
# 3. Distance Matrix Helper
# =========================
class DistanceMatrix:
    """
    Houdt afstand (km) en reistijd (minuten) tussen locaties bij.
    """
    def __init__(self, distance_dict, time_dict):
        # distance_dict[(A,B)] = km
        # time_dict[(A,B)]     = minuten
        self.distance_dict = distance_dict
        self.time_dict = time_dict

    def get_distance_km(self, origin, dest):
        return self.distance_dict.get((origin, dest), None)

    def get_travel_minutes(self, origin, dest):
        return self.time_dict.get((origin, dest), None)


# =========================
# 4. Charger Planner
# =========================
class ChargingPlanner:
    """
    Verantwoordelijk voor opladen.
    Regels:
    - Alleen laden in de garage BusConstants.GARAGE_NAME
    - Minimaal MIN_CHARGE_MIN minuten
    - Laadsnelheid CHARGING_POWER_KW (kWh per uur)
    - Kan alleen laden vóór een rit als we daarna op tijd bij de rit kunnen zijn
    """
    def __init__(self, charger_location: str):
        self.charger_location = charger_location

    def can_charge_here(self, location: str) -> bool:
        return location == self.charger_location == BusConstants.GARAGE_NAME

    def compute_charge_session(
        self,
        bus: Bus,
        next_ride: Ride,
        dist_matrix: DistanceMatrix
    ):
        """
        Probeert een laadmoment in te plannen vóór next_ride.
        Geeft (arrival_dt, leave_dt, bat_before, bat_after) of None terug.
        We nemen aan dat de bus al op charger_location staat.
        """

        if not self.can_charge_here(bus.current_location):
            return None

        # tijd die we minimaal moeten vrijhouden om nog naar de startloc van de rit te komen
        travel_to_start_min = 0.0
        if bus.current_location != next_ride.start_stop:
            tmin = dist_matrix.get_travel_minutes(bus.current_location, next_ride.start_stop)
            if tmin is None:
                return None
            travel_to_start_min = tmin

        latest_depart_for_ride = next_ride.start_time - timedelta(minutes=travel_to_start_min)

        # We kunnen pas laden vanaf bus.current_time
        # Minimaal 15 minuten
        start_charge = max(bus.current_time, bus.current_time)
        end_charge   = start_charge + timedelta(minutes=BusConstants.MIN_CHARGE_MIN)

        if end_charge > latest_depart_for_ride:
            # zelfs minimale charge past niet
            return None

        # Als er méér slack is kunnen we langer laden
        max_end_charge = latest_depart_for_ride
        # eindlaadtijd mag niet voorbij max_end_charge
        # voor nu nemen we gewoon max_end_charge (vol laden)
        end_charge = max_end_charge

        duration_min = (end_charge - start_charge).total_seconds() / 60.0
        if duration_min < BusConstants.MIN_CHARGE_MIN:
            return None

        # hoeveel kWh erbij?
        added_kwh = (duration_min / 60.0) * BusConstants.CHARGING_POWER_KW
        bat_before = bus.battery_kwh
        bat_after = min(
            bat_before + added_kwh,
            BusConstants.usable_capacity_kwh()
        )

        return (start_charge, end_charge, bat_before, bat_after)


# =========================
# 5. Scheduler
# =========================
class BusScheduler:
    """
    Probeert alle rides in te plannen over een vloot bussen.
    """

    def __init__(self, distance_matrix: DistanceMatrix, charging_planner: ChargingPlanner, garage_location: str):
        self.distance_matrix = distance_matrix
        self.charging_planner = charging_planner
        self.garage_location = garage_location

    def energy_needed_for_distance(self, dist_km: float):
        return dist_km * BusConstants.CONSUMPTION_PER_KM

    def can_bus_do_ride_direct(self, bus: Bus, ride: Ride):
        """
        Check of de bus zonder opladen deze rit kan halen.
        Retourneert tuple (feasible:bool, new_bus_state:Bus, assignment_record:AssignmentRecord)
        of (False,None,None) als het niet kan.
        """

        # 1. Moet bus eerst deadheaden?
        deadhead_info = None
        travel_min = 0.0
        travel_dist_km = 0.0
        depart_dt = bus.current_time

        if bus.current_location != ride.start_stop:
            tmin = self.distance_matrix.get_travel_minutes(bus.current_location, ride.start_stop)
            dist = self.distance_matrix.get_distance_km(bus.current_location, ride.start_stop)
            if tmin is None or dist is None:
                return (False, None, None)

            # bus moet vertrekken niet later dan ride.start_time - tmin
            arrive_dt = bus.current_time + timedelta(minutes=tmin)
            if arrive_dt > ride.start_time:
                return (False, None, None)

            # energie voor deadhead
            energy_deadhead = self.energy_needed_for_distance(dist)

            # bus batterij check na deadhead
            bat_after_deadhead = bus.battery_kwh - energy_deadhead
            if bat_after_deadhead < BusConstants.min_energy_kwh():
                # direct al onder minimum
                return (False, None, None)

            deadhead_info = (
                bus.current_location,
                ride.start_stop,
                dist,
                bus.current_time,
                arrive_dt
            )

            depart_dt = arrive_dt
            travel_min = tmin
            travel_dist_km = dist
            current_battery = bat_after_deadhead
        else:
            # geen deadhead
            if bus.current_time > ride.start_time:
                return (False, None, None)
            depart_dt = bus.current_time
            current_battery = bus.battery_kwh

        # 2. Wachten tot rit start
        if depart_dt > ride.start_time:
            # we komen te laat
            return (False, None, None)

        # In deze basisversie gaan we ervan uit dat wachten zelf geen grote extra verbruik oplevert,
        # of dat IDLE_USAGE pas later meegenomen wordt.
        start_drive_dt = ride.start_time

        # 3. Energie voor de rit zelf
        energy_service = self.energy_needed_for_distance(ride.distance_km)
        bat_before_ride = current_battery
        bat_after_ride = bat_before_ride - energy_service

        # check SOC floor
        if bat_after_ride < BusConstants.min_energy_kwh():
            return (False, None, None)

        # 4. Bus status na rit
        new_bus = bus.clone()
        new_bus.current_location = ride.end_stop
        new_bus.current_time = ride.end_time
        new_bus.battery_kwh = bat_after_ride

        # 5. Maak AssignmentRecord
        asg = AssignmentRecord(
            bus_id=new_bus.bus_id,
            ride=ride,
            battery_before=bat_before_ride,
            battery_after=bat_after_ride,
            deadhead_before=deadhead_info,
            charging_before=None
        )
        new_bus.history.append(asg)

        return (True, new_bus, asg)

    def try_bus_with_charging(self, bus: Bus, ride: Ride):
        """
        Variant: eerst laden (alleen in garage), dan eventueel deadhead en dan rit.
        """
        # we mogen alleen laden als bus in garage staat
        if bus.current_location != BusConstants.GARAGE_NAME:
            return (False, None, None)

        # plan een laad sessie
        charge_session = self.charging_planner.compute_charge_session(
            bus,
            ride,
            self.distance_matrix
        )
        if charge_session is None:
            return (False, None, None)

        charge_start, charge_end, bat_before_charge, bat_after_charge = charge_session

        # update bus tijdelijk alsof hij geladen heeft
        charged_bus = bus.clone()
        charged_bus.current_time = charge_end
        charged_bus.battery_kwh = bat_after_charge

        # daarna zelfde routine als direct
        feasible, new_bus_after, asg_after = self.can_bus_do_ride_direct(charged_bus, ride)
        if not feasible:
            return (False, None, None)

        # we moeten nu de assignment uitbreiden met charging_before info
        asg_after.charging_before = (
            charge_start,
            charge_end,
            bat_before_charge,
            bat_after_charge
        )

        return (True, new_bus_after, asg_after)

    def assign_ride_to_existing_buses(self, ride: Ride, buses: list[Bus]):
        """
        Probeer deze ride toe te voegen aan een van de bestaande bussen.
        Kies de bus die haalbaar is, en die het minst 'moeilijk' is.
        In deze simpele versie: eerste die werkt.
        """

        best_option = None

        for i, bus in enumerate(buses):
            # optie 1: zonder laden
            feasible, new_bus_state, assignment_record = self.can_bus_do_ride_direct(bus, ride)
            if feasible:
                best_option = (i, new_bus_state, assignment_record)
                break  # greedy: pak de eerste die kan

            # optie 2: met laden (alleen garage en min 15 min)
            feasible_c, new_bus_state_c, assignment_record_c = self.try_bus_with_charging(bus, ride)
            if feasible_c and best_option is None:
                best_option = (i, new_bus_state_c, assignment_record_c)
                break

        return best_option  # of None

    def create_new_bus(self, bus_id: int, first_ride: Ride):
        """
        Maak een nieuwe bus die start in de garage met start-SOC,
        en plan dan eventueel deadhead + rit.
        Als dat niet lukt → None.
        """

        start_energy = BusConstants.start_energy_kwh()
        # Starttijd: 1 uur voor eerste rit, zoals je al had in tool.py
        bus_start_time = first_ride.start_time - timedelta(hours=1)

        new_bus = Bus(
            bus_id=f"BUS_{bus_id}",
            current_location=self.garage_location,
            battery_kwh=start_energy,
            current_time=bus_start_time,
            history=[]
        )

        # Kan hij de rit meteen rijden (evt deadhead)? zo niet → probeer met laden
        feasible, new_bus_state, assignment_record = self.can_bus_do_ride_direct(new_bus, first_ride)
        if feasible:
            return new_bus_state, assignment_record

        feasible_c, new_bus_state_c, assignment_record_c = self.try_bus_with_charging(new_bus, first_ride)
        if feasible_c:
            return new_bus_state_c, assignment_record_c

        return None, None

    def schedule_all_rides(self, rides: list[Ride], initial_buses: list[Bus] | None = None):
        """
        Hoofdfunctie: geef lijst AssignmentRecord terug in chronologische volgorde van ritten.
        """
        # sorteer ritten op start_time
        rides_sorted = sorted(rides, key=lambda r: r.start_time)

        buses: list[Bus] = []
        if initial_buses:
            buses = [b.clone() for b in initial_buses]

        assignments: list[AssignmentRecord] = []

        next_bus_id = len(buses) + 1 if buses else 1

        for ride in rides_sorted:
            # eerst proberen in bestaande bussen
            option = self.assign_ride_to_existing_buses(ride, buses)

            if option is not None:
                bus_index, updated_bus_state, assignment_record = option
                buses[bus_index] = updated_bus_state
                assignments.append(assignment_record)
                continue

            # anders nieuwe bus aanmaken
            new_bus_state, assignment_record = self.create_new_bus(next_bus_id, ride)
            if new_bus_state is None:
                # geen oplossing → dit is een harde fail in deze eerste versie
                # we maken hier een "dummy" assignment met negatieve batterij zodat Streamlit dit highlight
                fail_asg = AssignmentRecord(
                    bus_id=f"BUS_{next_bus_id}",
                    ride=ride,
                    battery_before=0.0,
                    battery_after=-1.0,
                    deadhead_before=None,
                    charging_before=None
                )
                assignments.append(fail_asg)
                # we pushen geen nieuwe bus want hij is eigenlijk niet feasible
                continue

            # sla de nieuwe bus op
            buses.append(new_bus_state)
            assignments.append(assignment_record)
            next_bus_id += 1

        # return alle assignments (al in ritvolgorde)
        return assignments


# =========================
# 6. Data Loader Helpers
# =========================
class DataLoader:
    """
    Leest de excel-dataframes (timetable en distance matrix) in en maakt Ride objects + distance dicts
    """

    def _parse_time_today(self, tstring: str, base_day: datetime | None = None):
        """
        tstring bv. "08:30:00"
        we plakken er een dummy datum aan (vandaag 00:00) zodat we echte datetime hebben
        """
        if base_day is None:
            base_day = datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)

        # probeer HH:MM:SS, dan HH:MM
        for fmt in ("%H:%M:%S", "%H:%M"):
            try:
                just_time = datetime.strptime(str(tstring), fmt).time()
                return datetime.combine(base_day.date(), just_time)
            except ValueError:
                pass

        raise ValueError(f"Unrecognized time format: {tstring}")

    def load_distance_matrix_from_df(self, df: pd.DataFrame):
        """
        Verwacht kolommen (case-insensitive):
            start, end, distance_m, min_travel_time
        min_travel_time = minuten reistijd
        """
        cols = {c.lower(): c for c in df.columns}
        start_col = cols.get("start", None)
        end_col   = cols.get("end", None)
        dist_col  = cols.get("distance_m", None)
        time_col  = cols.get("min_travel_time", None)

        if not all([start_col, end_col, dist_col, time_col]):
            raise ValueError("Distance matrix has missing required columns")

        distance_dict = {}
        time_dict = {}

        for _, row in df.iterrows():
            origin = str(row[start_col])
            dest   = str(row[end_col])
            dist_m = float(row[dist_col])
            dist_km = dist_m / 1000.0
            tmin = float(row[time_col])

            distance_dict[(origin, dest)] = dist_km
            time_dict[(origin, dest)] = tmin

        return distance_dict, time_dict, df

    def load_timetable_from_df(self, df: pd.DataFrame, distance_df: pd.DataFrame):
        """
        Timetable Excel bevat bv:
            line, start, end, departure_time, arrival_time
        We moeten distance_km bepalen via distance_df.
        """
        cols = {c.lower(): c for c in df.columns}
        line_col   = cols.get("line", None)
        start_col  = cols.get("start", None)
        end_col    = cols.get("end", None)
        dep_col    = cols.get("departure_time", None)
        arr_col    = cols.get("arrival_time", None)

        if not all([line_col, start_col, end_col, dep_col, arr_col]):
            return []

        # we willen voor elk (start,end,line) de afstand km pakken
        # (same merge-idee als in Formulas.calculate_energy_consumption)
        tmp = df.copy()
        tmp["start_tmp_key"] = tmp[start_col].astype(str)
        tmp["end_tmp_key"]   = tmp[end_col].astype(str)

        dist_tmp = distance_df.copy()
        dist_tmp["start_tmp_key"] = dist_tmp["start"].astype(str)
        dist_tmp["end_tmp_key"]   = dist_tmp["end"].astype(str)
        dist_tmp = dist_tmp[["start_tmp_key","end_tmp_key","distance_m"]].drop_duplicates()

        merged = pd.merge(
            tmp,
            dist_tmp,
            on=["start_tmp_key","end_tmp_key"],
            how="left"
        )

        rides: list[Ride] = []
        for _, row in merged.iterrows():
            try:
                distance_km = float(row["distance_m"]) / 1000.0
            except Exception:
                distance_km = math.nan

            start_dt = self._parse_time_today(row[dep_col])
            end_dt   = self._parse_time_today(row[arr_col])

            ride = Ride(
                line = str(row[line_col]),
                start_stop = str(row[start_col]),
                end_stop = str(row[end_col]),
                start_time = start_dt,
                end_time = end_dt,
                distance_km = distance_km
            )
            rides.append(ride)

        return rides
