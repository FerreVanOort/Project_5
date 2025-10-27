# PlanningMaker.py
#
# Deze module bouwt de planning uit timetable + distance matrix
# en levert:
# - BusConstants (batterij/energie parameters, wordt live geüpdatet door Tool.py)
# - DataLoader (leest je Excel dataframes)
# - BusScheduler (maakt de daadwerkelijke planning)
# - AssignmentRecord per ingeplande rit, inclusief deadhead/idle/charging info

from dataclasses import dataclass, field
from datetime import datetime, timedelta, time
import pandas as pd
import math


# =========================
# 1. Constants / Parameters
# =========================
class BusConstants:
    # Deze waardes worden overschreven door Tool.py
    BATTERY_CAPACITY_NOMINAL = 300.0        # kWh fysiek pack
    SOH_PERCENT = 90.0                      # [%] health → usable capacity
    START_BAT_PERCENT = 100.0               # [% van usable capacity aan het begin van de dag]
    MIN_BAT_PERCENT  = 10.0                 # [% van usable capacity als ondergrens]
    CONSUMPTION_PER_KM = 1.2                # kWh/km
    CHARGING_POWER_KW = 450.0               # kW (kWh per uur)
    IDLE_USAGE_KW = 5.0                     # kW verbruik tijdens idle
    MIN_CHARGE_MIN = 15                     # minimaal 15 min laden
    GARAGE_NAME = "ehvgar"                  # garage naam / laadlocatie

    @classmethod
    def usable_capacity_kwh(cls):
        """
        Hoeveel kWh effectief bruikbaar is (SOH toegepast).
        """
        return cls.BATTERY_CAPACITY_NOMINAL * (cls.SOH_PERCENT / 100.0)

    @classmethod
    def start_energy_kwh(cls):
        """
        Start SOC in kWh op begin van de dag, gebaseerd op usable capacity.
        """
        return cls.usable_capacity_kwh() * (cls.START_BAT_PERCENT / 100.0)

    @classmethod
    def min_energy_kwh(cls):
        """
        Minimale toegestane energie in kWh (vloer SOC).
        """
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
    """
    Eén geplande rit van een bus, incl info over wat er vlak vóór nodig was.
    """
    bus_id: str
    ride: Ride

    battery_before: float  # kWh voor de rit begint
    battery_after: float   # kWh na rit eindigt

    # deadhead_before:
    #   (from_loc, to_loc, dist_km, depart_dt, arrive_dt)
    deadhead_before: tuple | None = None

    # charging_before:
    #   (charge_start_dt, charge_end_dt, bat_before_charge, bat_after_charge)
    charging_before: tuple | None = None

    # idle_before:
    #   (idle_start_dt, idle_end_dt, idle_energy_used_kWh)
    idle_before: tuple | None = None


@dataclass
class Bus:
    bus_id: str
    current_location: str
    battery_kwh: float
    current_time: datetime
    history: list[AssignmentRecord] = field(default_factory=list)

    def clone(self):
        return Bus(
            bus_id=self.bus_id,
            current_location=self.current_location,
            battery_kwh=self.battery_kwh,
            current_time=self.current_time,
            history=list(self.history)
        )


# =========================
# 3. Helpers voor tijd
# =========================
def shifted_dt(base_day: datetime, t: time) -> datetime:
    """
    Maak een datetime (vandaag+tijd). Als tijd <03:00 → tel als 'volgende dag'.
    Dit zorgt dat 01:30 na 23:30 komt, niet vóór.
    """
    dt = datetime.combine(base_day.date(), t)
    if t < time(3, 0):
        dt += timedelta(days=1)
    return dt


# =========================
# 4. Distance Matrix Helper
# =========================
class DistanceMatrix:
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
# 5. Charging Planner
# =========================
class ChargingPlanner:
    """
    Regels:
    - Laden kan alleen in GARAGE_NAME
    - Minimaal MIN_CHARGE_MIN minuten
    - Als er tijd is voor de rit start
    - Laadsnelheid CHARGING_POWER_KW kW
    """

    def __init__(self, charger_location: str):
        self.charger_location = charger_location

    def can_charge_here(self, location: str) -> bool:
        return (
            location == self.charger_location ==
            BusConstants.GARAGE_NAME
        )

    def compute_charge_session(
        self,
        bus: Bus,
        next_ride: Ride,
        dist_matrix: DistanceMatrix,
    ):
        """
        Probeer een laadmoment vóór next_ride:
        Return (charge_start, charge_end, bat_before, bat_after) of None
        """

        if not self.can_charge_here(bus.current_location):
            return None

        # Hoeveel tijd hebben we tot de rit moet vertrekken locatie-wise?
        travel_to_start_min = 0.0
        if bus.current_location != next_ride.start_stop:
            tmin = dist_matrix.get_travel_minutes(bus.current_location, next_ride.start_stop)
            if tmin is None:
                return None
            travel_to_start_min = tmin

        latest_depart_for_ride = next_ride.start_time - timedelta(minutes=travel_to_start_min)

        # Kan pas laden vanaf bus.current_time
        charge_start = bus.current_time
        # Minimaal 15 min
        earliest_end = charge_start + timedelta(minutes=BusConstants.MIN_CHARGE_MIN)

        if earliest_end > latest_depart_for_ride:
            # Zelfs minimale laadbeurt past niet.
            return None

        # Er is slack -> voluit laden tot we echt weg moeten
        charge_end = latest_depart_for_ride
        duration_min = (charge_end - charge_start).total_seconds() / 60.0

        if duration_min < BusConstants.MIN_CHARGE_MIN:
            return None

        added_kwh = (duration_min / 60.0) * BusConstants.CHARGING_POWER_KW
        bat_before = bus.battery_kwh
        bat_after = min(
            bat_before + added_kwh,
            BusConstants.usable_capacity_kwh()
        )

        return (charge_start, charge_end, bat_before, bat_after)


# =========================
# 6. Bus Scheduler
# =========================
class BusScheduler:
    """
    Simpele greedy planner:
    - Sorteer rides op starttijd
    - Voor elke ride:
      1) probeer bestaande bussen (met idle/charging/deadhead indien nodig)
      2) anders start nieuwe bus
    """

    def __init__(self, distance_matrix: DistanceMatrix, charging_planner: ChargingPlanner, garage_location: str):
        self.distance_matrix = distance_matrix
        self.charging_planner = charging_planner
        self.garage_location = garage_location

    def energy_needed_for_distance(self, dist_km: float):
        return dist_km * BusConstants.CONSUMPTION_PER_KM

    def simulate_idle_until(self, bus: Bus, new_time: datetime):
        """
        Laat bus wachten (idle) tot new_time, verbruik IDLE_USAGE_KW over die tijd.
        Return:
            updated_bus (Bus clone)
            idle_tuple (idle_start_dt, idle_end_dt, idle_energy_used_kWh) of None
        """
        if new_time <= bus.current_time:
            return bus.clone(), None

        idle_start = bus.current_time
        idle_end = new_time

        idle_hours = (idle_end - idle_start).total_seconds() / 3600.0
        idle_energy_used = idle_hours * BusConstants.IDLE_USAGE_KW

        new_bus = bus.clone()
        new_bus.current_time = new_time
        new_bus.battery_kwh = new_bus.battery_kwh - idle_energy_used

        idle_info = (idle_start, idle_end, idle_energy_used)
        return new_bus, idle_info

    def try_travel_to_start(self, bus: Bus, ride: Ride):
        """
        Zorg dat de bus op tijd bij ride.start_stop komt.
        Dit kan deadhead vereisen.
        We modelleren:
          1) Als bus al op juiste locatie:
              - mogelijk idle tot rit start
          2) Zo niet:
              - deadhead trip -> check aankomst voor start, trek energie af
              - idle als we te vroeg aankomen
        Return (ok, new_bus, deadhead_info, idle_info_before_ride)
        """
        b = bus.clone()

        deadhead_info = None
        idle_info_total = None

        if b.current_location != ride.start_stop:
            tmin = self.distance_matrix.get_travel_minutes(b.current_location, ride.start_stop)
            dist_km = self.distance_matrix.get_distance_km(b.current_location, ride.start_stop)
            if tmin is None or dist_km is None:
                return (False, None, None, None)

            depart_dt = b.current_time
            arrive_dt = b.current_time + timedelta(minutes=tmin)

            # moet voor ride.start_time aankomen
            if arrive_dt > ride.start_time:
                return (False, None, None, None)

            # energieverbruik voor deadhead
            energy_deadhead = self.energy_needed_for_distance(dist_km)
            b.battery_kwh -= energy_deadhead
            if b.battery_kwh < BusConstants.min_energy_kwh():
                return (False, None, None, None)

            deadhead_info = (
                bus.current_location,
                ride.start_stop,
                dist_km,
                depart_dt,
                arrive_dt
            )

            # update locatie/tijd
            b.current_location = ride.start_stop
            b.current_time = arrive_dt

        # nu zijn we op ride.start_stop op tijd (of te vroeg)
        # als we te vroeg zijn -> idle wachten tot start_time
        if b.current_time < ride.start_time:
            b_after_idle, idle_info = self.simulate_idle_until(b, ride.start_time)
            if b_after_idle.battery_kwh < BusConstants.min_energy_kwh():
                return (False, None, None, None)
            b = b_after_idle
            idle_info_total = idle_info

        # te laat is al afgevangen
        return (True, b, deadhead_info, idle_info_total)

    def try_ride_energy(self, bus: Bus, ride: Ride):
        """
        Check of bus de eigenlijke rit kan rijden (service trip),
        trek energie af en update locatie/tijd.
        Return (ok, new_bus, bat_before_ride, bat_after_ride)
        """
        b = bus.clone()

        bat_before_ride = b.battery_kwh
        service_energy = self.energy_needed_for_distance(ride.distance_km)
        bat_after_ride = bat_before_ride - service_energy

        if bat_after_ride < BusConstants.min_energy_kwh():
            return (False, None, None, None)

        b.battery_kwh = bat_after_ride
        b.current_location = ride.end_stop
        b.current_time = ride.end_time

        return (True, b, bat_before_ride, bat_after_ride)

    def try_bus_for_ride_nocharge(self, bus: Bus, ride: Ride):
        """
        Probeer ride met bestaande SOC, met eventueel deadhead + idle,
        maar zonder vooraf te laden.
        """
        ok, b_after_position, deadhead_info, idle_info = self.try_travel_to_start(bus, ride)
        if not ok:
            return (False, None, None)

        ok2, b_after_ride, bat_before, bat_after = self.try_ride_energy(b_after_position, ride)
        if not ok2:
            return (False, None, None)

        # bouw assignmentrecord:
        asg = AssignmentRecord(
            bus_id=bus.bus_id,
            ride=ride,
            battery_before=bat_before,
            battery_after=bat_after,
            deadhead_before=deadhead_info,
            charging_before=None,
            idle_before=idle_info
        )

        new_bus_state = b_after_ride
        new_bus_state.history.append(asg)
        return (True, new_bus_state, asg)

    def try_bus_for_ride_withcharge(self, bus: Bus, ride: Ride):
        """
        Zelfde als hierboven, maar eerst proberen te laden in de garage.
        Voorwaarde:
          - bus moet in garage zijn
          - genoeg slack om te laden min. 15 min
        """
        if bus.current_location != BusConstants.GARAGE_NAME:
            return (False, None, None)

        charge_sess = self.charging_planner.compute_charge_session(bus, ride, self.distance_matrix)
        if charge_sess is None:
            return (False, None, None)

        charge_start, charge_end, bat_before_ch, bat_after_ch = charge_sess

        # maak tijdelijke bus alsof hij net geladen heeft
        charged_bus = bus.clone()
        charged_bus.current_time = charge_end
        charged_bus.battery_kwh = bat_after_ch

        # vanaf hier: hetzelfde traject als nocharge
        ok, b_after_position, deadhead_info, idle_info = self.try_travel_to_start(charged_bus, ride)
        if not ok:
            return (False, None, None)

        ok2, b_after_ride, bat_before, bat_after = self.try_ride_energy(b_after_position, ride)
        if not ok2:
            return (False, None, None)

        # assignment inkl. charging_before info
        asg = AssignmentRecord(
            bus_id=bus.bus_id,
            ride=ride,
            battery_before=bat_before,
            battery_after=bat_after,
            deadhead_before=deadhead_info,
            charging_before=(charge_start, charge_end, bat_before_ch, bat_after_ch),
            idle_before=idle_info
        )

        new_bus_state = b_after_ride
        new_bus_state.history.append(asg)
        return (True, new_bus_state, asg)

    def assign_ride_to_existing_buses(self, ride: Ride, buses: list[Bus]):
        """
        Probeer ride toe te wijzen aan bestaande bussen.
        Greedy: pak de eerste bus die haalbaar is.
        """
        for i, b in enumerate(buses):
            # probeer zonder laden
            ok, new_state, rec = self.try_bus_for_ride_nocharge(b, ride)
            if ok:
                return (i, new_state, rec)

            # probeer met laden
            ok2, new_state2, rec2 = self.try_bus_for_ride_withcharge(b, ride)
            if ok2:
                return (i, new_state2, rec2)

        return None

    def create_new_bus(self, bus_id_number: int, first_ride: Ride):
        """
        Start een nieuwe bus in de garage met start-SOC.
        Hij staat een uur voor de eerste rit klaar.
        Probeer zonder/ met charge.
        """
        start_energy = BusConstants.start_energy_kwh()
        bus_start_time = first_ride.start_time - timedelta(hours=1)

        new_bus = Bus(
            bus_id=f"BUS_{bus_id_number}",
            current_location=self.garage_location,
            battery_kwh=start_energy,
            current_time=bus_start_time,
            history=[]
        )

        # probeer zonder laden
        ok, state_nocharge, rec_nocharge = self.try_bus_for_ride_nocharge(new_bus, first_ride)
        if ok:
            return state_nocharge, rec_nocharge

        # probeer met laden
        ok2, state_charge, rec_charge = self.try_bus_for_ride_withcharge(new_bus, first_ride)
        if ok2:
            return state_charge, rec_charge

        # geen oplossing → return None
        return None, None

    def schedule_all_rides(self, rides: list[Ride], initial_buses: list[Bus] | None = None):
        """
        Sorteer rides op start_time en plan ze sequentieel.
        Return: lijst AssignmentRecord in rit-volgorde.
        """
        rides_sorted = sorted(rides, key=lambda r: r.start_time)

        # actieve busfleet
        buses: list[Bus] = []
        if initial_buses:
            buses = [b.clone() for b in initial_buses]

        assignments: list[AssignmentRecord] = []

        next_bus_id = len(buses) + 1 if buses else 1

        for ride in rides_sorted:
            # 1. probeer bestaande bussen
            option = self.assign_ride_to_existing_buses(ride, buses)
            if option is not None:
                idx, updated_state, record = option
                buses[idx] = updated_state
                assignments.append(record)
                continue

            # 2. anders nieuwe bus
            new_state, rec = self.create_new_bus(next_bus_id, ride)
            if new_state is None:
                # planning is mislukt → produceer een "falen" assignment met battery_after <0
                fail_rec = AssignmentRecord(
                    bus_id=f"BUS_{next_bus_id}",
                    ride=ride,
                    battery_before=0.0,
                    battery_after=-1.0,
                    deadhead_before=None,
                    charging_before=None,
                    idle_before=None
                )
                assignments.append(fail_rec)
                # geen bus toevoegen omdat hij eigenlijk niet haalbaar was
                next_bus_id += 1
                continue

            buses.append(new_state)
            assignments.append(rec)
            next_bus_id += 1

        return assignments


# =========================
# 7. Data Loader Helpers
# =========================
class DataLoader:
    """
    Leest timetable + distance matrix uit al ingelezen DataFrames.
    """

    def _parse_time_today(self, t_val, base_day: datetime | None = None):
        """
        Converteer een tijd zoals '05:07' of '18:33:00' naar datetime met dummy-datum
        en pas night-shift toe (00:00-02:59 → volgende dag).
        """
        if base_day is None:
            base_day = datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)

        # t_val kan bv. '05:07' of '18:33:00' of een pandas.Timestamp met tijd
        if isinstance(t_val, pd.Timestamp):
            just_time = t_val.time()
        elif isinstance(t_val, datetime):
            just_time = t_val.time()
        elif isinstance(t_val, str):
            parsed = None
            for fmt in ("%H:%M:%S", "%H:%M"):
                try:
                    parsed_dt = datetime.strptime(t_val, fmt)
                    parsed = parsed_dt.time()
                    break
                except ValueError:
                    pass
            if parsed is None:
                raise ValueError(f"Unrecognized time format: {t_val}")
            just_time = parsed
        else:
            # probeer via pandas to_datetime fallback
            try:
                parsed_dt = pd.to_datetime(t_val)
                just_time = parsed_dt.time()
            except Exception:
                raise ValueError(f"Unrecognized time type: {t_val}")

        # combine met base_day en verschuif nacht
        dt_full = datetime.combine(base_day.date(), just_time)
        if just_time < time(3,0):
            dt_full += timedelta(days=1)

        return dt_full


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
            raise ValueError("Distance matrix has missing required columns (need start, end, distance_m, min_travel_time)")

        distance_dict = {}
        time_dict = {}

        for _, row in df.iterrows():
            origin = str(row[start_col])
            dest   = str(row[end_col])
            dist_m = float(row[dist_col])
            tmin   = float(row[time_col])

            distance_km = dist_m / 1000.0

            distance_dict[(origin, dest)] = distance_km
            time_dict[(origin, dest)] = tmin

        return distance_dict, time_dict, df


    def load_timetable_from_df(self, df: pd.DataFrame, distance_df: pd.DataFrame):
        """
        Timetable dataframe heeft minimaal:
            start, departure_time, end, line
        We schatten arrival_time via distance matrix:
          - pak (start,end) uit distance_df -> min_travel_time (in minuten)
          - arrival = departure + min_travel_time

        We bouwen Ride(line, start_stop, end_stop, start_time, end_time, distance_km)
        """
        cols = {c.lower(): c for c in df.columns}
        start_col  = cols.get("start", None)
        dep_col    = cols.get("departure_time", None)
        end_col    = cols.get("end", None)
        line_col   = cols.get("line", None)

        if not all([start_col, dep_col, end_col, line_col]):
            return []

        # Bouw hulpkoppeling start-end → (distance_m, min_travel_time)
        dist_tmp = distance_df.copy()
        dist_tmp_cols = {c.lower(): c for c in dist_tmp.columns}

        # Vereiste kolommen in distance_df
        req_cols = ["start","end","distance_m","min_travel_time"]
        for rc in req_cols:
            if rc not in dist_tmp_cols:
                raise ValueError(f"Distance matrix is missing '{rc}' column for timetable mapping")

        start_dcol = dist_tmp_cols["start"]
        end_dcol   = dist_tmp_cols["end"]
        distm_col  = dist_tmp_cols["distance_m"]
        tmin_col   = dist_tmp_cols["min_travel_time"]

        # maak een lookup dict
        # key: (start_stop, end_stop) -> (distance_km, travel_min)
        lookup = {}
        for _, row in dist_tmp.iterrows():
            k = (str(row[start_dcol]), str(row[end_dcol]))
            distance_km = float(row[distm_col]) / 1000.0
            travel_min = float(row[tmin_col])
            lookup[k] = (distance_km, travel_min)

        rides: list[Ride] = []

        base_day = datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)

        for _, row in df.iterrows():
            start_stop = str(row[start_col])
            end_stop   = str(row[end_col])
            line_id    = str(row[line_col])

            # vertrek
            dep_dt = self._parse_time_today(row[dep_col], base_day)

            # zoek afstand en rijtijd
            key = (start_stop, end_stop)
            if key in lookup:
                distance_km, travel_min = lookup[key]
            else:
                # als we geen afstand kennen -> skip deze rit
                continue

            arr_dt = dep_dt + timedelta(minutes=travel_min)

            ride = Ride(
                line=line_id,
                start_stop=start_stop,
                end_stop=end_stop,
                start_time=dep_dt,
                end_time=arr_dt,
                distance_km=distance_km
            )
            rides.append(ride)

        return rides
