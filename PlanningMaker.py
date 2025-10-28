# PlanningMaker.py
#
# Bouwt een planning uit een timetable + distance matrix
# en levert:
# - BusConstants  (batterij/energie parameters; worden live geüpdatet door Tool.py)
# - DataLoader    (leest je Excel dataframes en maakt Ride-objecten)
# - ChargingPlanner
# - BusScheduler  (plant de ritten op bussen, incl. laden, idle en deadhead)
# - AssignmentRecord per geplande rit

from dataclasses import dataclass, field
from datetime import datetime, timedelta, time
import pandas as pd
import math


# =========================
# 1. Constants / Parameters
# =========================
class BusConstants:
    # Deze waardes worden tijdens runtime overschreven door Tool.py
    BATTERY_CAPACITY_NOMINAL = 300.0        # kWh fysiek pack
    SOH_PERCENT = 90.0                      # [%] health → usable capacity
    START_BAT_PERCENT = 100.0               # [% van usable capacity aan het begin van de dag]
    MIN_BAT_PERCENT  = 10.0                 # [% van usable capacity als ondergrens]
    CONSUMPTION_PER_KM = 1.2                # kWh/km tijdens rijden
    CHARGING_POWER_KW = 450.0               # kW (kWh per uur)
    IDLE_USAGE_KW = 5.0                     # kW verbruik tijdens idle
    MIN_CHARGE_MIN = 15                     # minimaal 15 min laden
    GARAGE_NAME = "ehvgar"                  # garage/charging locatie

    @classmethod
    def usable_capacity_kwh(cls):
        """
        Hoeveel kWh bruikbaar is rekening houdend met SOH.
        """
        return cls.BATTERY_CAPACITY_NOMINAL * (cls.SOH_PERCENT / 100.0)

    @classmethod
    def start_energy_kwh(cls):
        """
        Start SOC in kWh, op basis van usable capacity en START_BAT_PERCENT.
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
    travel_min_used: float  # rijtijd (in minuten) die daadwerkelijk voor deze rit is gebruikt


@dataclass
class AssignmentRecord:
    """
    Eén ingeplande service rit van een bus, met info over wat er vlak vóór zat
    (charging / deadhead / idle).
    """
    bus_id: str
    ride: Ride

    battery_before: float    # kWh vlak voor vertrek van de service rit
    battery_after: float     # kWh na aankomst van de service rit

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

    # lijst met (start_dt, end_dt) blokken die al geclaimd zijn door deze bus
    occupied: list[tuple[datetime, datetime]] = field(default_factory=list)

    def clone(self):
        return Bus(
            bus_id=self.bus_id,
            current_location=self.current_location,
            battery_kwh=self.battery_kwh,
            current_time=self.current_time,
            history=list(self.history),
            occupied=list(self.occupied),
        )


# =========================
# 3. Helpers voor tijd
# =========================
def shifted_dt(base_day: datetime, t: time) -> datetime:
    """
    Maak een datetime (base_day + t).
    Als t < 03:00 → telt als 'volgende dag'.
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
    - Laden kan alleen op BusConstants.GARAGE_NAME
    - Minimaal MIN_CHARGE_MIN minuten
    - Je mag laden vóór de volgende rit als je in de garage bent en er genoeg tijd is
    - CHARGING_POWER_KW is laadvermogen
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
        Probeer laden vóór next_ride (alleen als de bus al in de garage staat).
        Return:
          (charge_start, charge_end, bat_before, bat_after) of None
        """

        if not self.can_charge_here(bus.current_location):
            return None

        # Tijd nodig om naar de start van de rit te komen:
        travel_to_start_min = 0.0
        if bus.current_location != next_ride.start_stop:
            tmin = dist_matrix.get_travel_minutes(bus.current_location, next_ride.start_stop)
            if tmin is None:
                return None
            travel_to_start_min = tmin

        latest_depart_for_ride = next_ride.start_time - timedelta(minutes=travel_to_start_min)

        charge_start = bus.current_time
        earliest_end = charge_start + timedelta(minutes=BusConstants.MIN_CHARGE_MIN)

        # past minimaal laden?
        if earliest_end > latest_depart_for_ride:
            return None

        # laad tot we echt moeten vertrekken
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
    Planner:
    - Sorteer alle rides op starttijd
    - Voor elke ride:
      1) kijk of een bestaande bus 'm kan rijden (geen overlap in tijd + energie OK),
         met evt. deadhead, idle, en eventueel laden als hij bij de garage staat
      2) anders instantiëer een nieuwe bus
    """

    def __init__(self, distance_matrix: DistanceMatrix, charging_planner: ChargingPlanner, garage_location: str):
        self.distance_matrix = distance_matrix
        self.charging_planner = charging_planner
        self.garage_location = garage_location

    def energy_needed_for_distance(self, dist_km: float):
        return dist_km * BusConstants.CONSUMPTION_PER_KM

    def simulate_idle_until(self, bus: Bus, new_time: datetime):
        """
        Laat bus wachten (idle) tot new_time, verbruik IDLE_USAGE_KW constant.
        Return:
            updated_bus (clone),
            idle_tuple (idle_start_dt, idle_end_dt, idle_energy_used_kWh) of None
        """
        if new_time <= bus.current_time:
            return bus.clone(), None

        idle_start = bus.current_time
        idle_end   = new_time

        idle_hours = (idle_end - idle_start).total_seconds() / 3600.0
        idle_energy_used = idle_hours * BusConstants.IDLE_USAGE_KW

        new_bus = bus.clone()
        new_bus.current_time = new_time
        new_bus.battery_kwh -= idle_energy_used

        idle_info = (idle_start, idle_end, idle_energy_used)
        return new_bus, idle_info

    def time_interval_free(self, bus: Bus, start_dt: datetime, end_dt: datetime):
        """
        Check of bus vrij is tussen start_dt en end_dt t.o.v. bus.occupied
        occupied heeft tuples (occ_start, occ_end)
        We eisen dat [start_dt,end_dt] niet overlapt met bestaande blokken.
        """
        for (occ_start, occ_end) in bus.occupied:
            # overlap als start < occ_end en occ_start < end
            if start_dt < occ_end and occ_start < end_dt:
                return False
        return True

    def block_interval(self, bus: Bus, start_dt: datetime, end_dt: datetime):
        """
        Voeg interval (start_dt, end_dt) toe aan bus.occupied.
        """
        b2 = bus.clone()
        b2.occupied.append((start_dt, end_dt))
        return b2

    def try_travel_to_start(self, bus: Bus, ride: Ride):
        """
        Zorg dat bus fysiek en qua tijd op ride.start_stop komt voor ride.start_time.
        Dit kan deadhead en idle vereisen.
        Return:
          (ok, new_bus, deadhead_info, idle_info_before_ride)
        """
        b = bus.clone()

        deadhead_info = None
        idle_info_total = None

        # 1. Moet er deadhead gereden worden?
        if b.current_location != ride.start_stop:
            tmin = self.distance_matrix.get_travel_minutes(b.current_location, ride.start_stop)
            dist_km = self.distance_matrix.get_distance_km(b.current_location, ride.start_stop)
            if tmin is None or dist_km is None:
                return (False, None, None, None)

            depart_dt = b.current_time
            arrive_dt = b.current_time + timedelta(minutes=tmin)

            # moet vóór de rit aankomen
            if arrive_dt > ride.start_time:
                return (False, None, None, None)

            # check energiebudget deadhead
            energy_deadhead = self.energy_needed_for_distance(dist_km)
            new_batt = b.battery_kwh - energy_deadhead
            if new_batt < BusConstants.min_energy_kwh():
                return (False, None, None, None)

            # check of bus vrij is qua tijd voor deze deadhead
            # deadhead blok = depart_dt → arrive_dt
            if not self.time_interval_free(b, depart_dt, arrive_dt):
                return (False, None, None, None)

            # update bus
            b.battery_kwh = new_batt
            b.current_location = ride.start_stop
            b.current_time = arrive_dt
            b = self.block_interval(b, depart_dt, arrive_dt)

            deadhead_info = (
                bus.current_location,
                ride.start_stop,
                dist_km,
                depart_dt,
                arrive_dt
            )

        # 2. Idle wachten tot start (indien te vroeg)
        if b.current_time < ride.start_time:
            # check of bus vrij is voor idle? Idle is ook een geblokkeerde periode.
            if not self.time_interval_free(b, b.current_time, ride.start_time):
                return (False, None, None, None)

            b_after_idle, idle_info = self.simulate_idle_until(b, ride.start_time)
            if b_after_idle.battery_kwh < BusConstants.min_energy_kwh():
                return (False, None, None, None)

            # block interval
            b_after_idle = self.block_interval(b_after_idle, b.current_time, ride.start_time)

            idle_info_total = idle_info
            b = b_after_idle

        # Als we precies op tijd aankomen is dit ook OK
        # -> b.current_time == ride.start_time
        return (True, b, deadhead_info, idle_info_total)

    def try_ride_energy_and_block(self, bus: Bus, ride: Ride):
        """
        Check of de bus de eigenlijke service rit kan rijden.
        Trek energie af, update locatie/tijd.
        Reserveer de interval van de service rit in bus.occupied.
        """
        b = bus.clone()

        bat_before_ride = b.battery_kwh
        service_energy = self.energy_needed_for_distance(ride.distance_km)
        bat_after_ride = bat_before_ride - service_energy

        if bat_after_ride < BusConstants.min_energy_kwh():
            return (False, None, None, None)

        # check tijd overlap tijdens service rit
        if not self.time_interval_free(b, ride.start_time, ride.end_time):
            return (False, None, None, None)

        b.battery_kwh = bat_after_ride
        b.current_location = ride.end_stop
        b.current_time = ride.end_time
        b = self.block_interval(b, ride.start_time, ride.end_time)

        return (True, b, bat_before_ride, bat_after_ride)

    def try_bus_for_ride_nocharge(self, bus: Bus, ride: Ride):
        """
        Probeer rit te plannen met bestaande busstate, zonder eerst op te laden.
        - positie/time align (deadhead + idle)
        - rit rijden
        """
        ok, b_after_position, deadhead_info, idle_info = self.try_travel_to_start(bus, ride)
        if not ok:
            return (False, None, None)

        ok2, b_after_ride, bat_before, bat_after = self.try_ride_energy_and_block(b_after_position, ride)
        if not ok2:
            return (False, None, None)

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
        Als bus in garage staat en er slack is → probeer eerst laden (minimaal 15 min),
        daarna dezelfde flow als hierboven.
        """
        if bus.current_location != BusConstants.GARAGE_NAME:
            return (False, None, None)

        charge_sess = self.charging_planner.compute_charge_session(bus, ride, self.distance_matrix)
        if charge_sess is None:
            return (False, None, None)

        charge_start, charge_end, bat_before_ch, bat_after_ch = charge_sess

        # check overlap: mag je deze laadperiode claimen?
        if not self.time_interval_free(bus, charge_start, charge_end):
            return (False, None, None)

        charged_bus = bus.clone()
        charged_bus.current_time = charge_end
        charged_bus.battery_kwh  = bat_after_ch
        charged_bus = self.block_interval(charged_bus, charge_start, charge_end)

        ok, b_after_position, deadhead_info, idle_info = self.try_travel_to_start(charged_bus, ride)
        if not ok:
            return (False, None, None)

        ok2, b_after_ride, bat_before, bat_after = self.try_ride_energy_and_block(b_after_position, ride)
        if not ok2:
            return (False, None, None)

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
        Greedy: pak de eerste bus waarbij het lukt (zonder charge, anders met charge).
        """
        for i, b in enumerate(buses):
            # zonder laden
            ok, new_state, rec = self.try_bus_for_ride_nocharge(b, ride)
            if ok:
                return (i, new_state, rec)

            # met laden
            ok2, new_state2, rec2 = self.try_bus_for_ride_withcharge(b, ride)
            if ok2:
                return (i, new_state2, rec2)

        return None

    def create_new_bus(self, bus_id_number: int, first_ride: Ride):
        """
        Maak een nieuwe bus in de garage, met start-SOC. Starttijd = 1 uur voor
        de eerste rit.
        """
        start_energy = BusConstants.start_energy_kwh()
        bus_start_time = first_ride.start_time - timedelta(hours=1)

        new_bus = Bus(
            bus_id=f"BUS_{bus_id_number}",
            current_location=self.garage_location,
            battery_kwh=start_energy,
            current_time=bus_start_time,
            history=[],
            occupied=[]
        )

        # zonder laden
        ok, state_nocharge, rec_nocharge = self.try_bus_for_ride_nocharge(new_bus, first_ride)
        if ok:
            return state_nocharge, rec_nocharge

        # met laden
        ok2, state_charge, rec_charge = self.try_bus_for_ride_withcharge(new_bus, first_ride)
        if ok2:
            return state_charge, rec_charge

        return None, None

    def schedule_all_rides(self, rides: list[Ride], initial_buses: list[Bus] | None = None):
        """
        Sorteer rides op start_time en plan sequentially.
        Return: lijst AssignmentRecord in volgorde van ritten.
        """
        rides_sorted = sorted(rides, key=lambda r: r.start_time)

        if initial_buses:
            buses = [b.clone() for b in initial_buses]
        else:
            buses = []

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

            # 2. nieuwe bus
            new_state, rec = self.create_new_bus(next_bus_id, ride)
            if new_state is None:
                # Mislukte planning → markeer assignment als fail
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
    Leest timetable + distance matrix dataframes en zet om naar Ride-objects.
    """

    def _parse_time_today(self, t_val, base_day: datetime | None = None):
        """
        Converteer '05:07' of '18:33:00' etc. naar datetime met dummy-datum,
        en pas nachtverschuiving toe (<03:00 → volgende dag).
        """
        if base_day is None:
            base_day = datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)

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
            parsed_dt = pd.to_datetime(t_val)
            just_time = parsed_dt.time()

        dt_full = datetime.combine(base_day.date(), just_time)
        if just_time < time(3,0):
            dt_full += timedelta(days=1)

        return dt_full

    def load_distance_matrix_from_df(self, df: pd.DataFrame):
        """
        Verwacht kolommen:
           start, end, distance_m, min_travel_time
        Bouwt dicts:
           distance_dict[(A,B)], time_dict[(A,B)]
        """
        cols = {c.lower(): c for c in df.columns}
        start_col = cols.get("start", None)
        end_col   = cols.get("end", None)
        dist_col  = cols.get("distance_m", None)
        time_col  = cols.get("min_travel_time", None)

        if not all([start_col, end_col, dist_col, time_col]):
            raise ValueError("Distance matrix missing columns (need: start, end, distance_m, min_travel_time)")

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
        timetable df kolommen:
            start, departure_time, end, line
        distance_df kolommen:
            start, end, distance_m, min_travel_time, line

        We bouwen een Ride per regel in de timetable.
        Belangrijk:
        - dezelfde corridor (start,end) kan meerdere lijnen hebben (400, 401, ...)
          met verschillende min_travel_time.
        - er kunnen ook regels zonder 'line' zijn in distance_df (fallback).

        Lookup:
            lookup_exact[(start,end,line_str)] = (distance_km, travel_min)
            lookup_fallback[(start,end,None)]  = (distance_km, travel_min)

        Voor iedere timetable-regel proberen we eerst exact (met lijn),
        anders fallback.
        """

        cols = {c.lower(): c for c in df.columns}
        start_col  = cols.get("start", None)
        dep_col    = cols.get("departure_time", None)
        end_col    = cols.get("end", None)
        line_col   = cols.get("line", None)

        if not all([start_col, dep_col, end_col, line_col]):
            return []

        dist_tmp = distance_df.copy()
        dist_tmp_cols = {c.lower(): c for c in dist_tmp.columns}

        req_cols = ["start","end","distance_m","min_travel_time","line"]
        for rc in req_cols:
            if rc not in dist_tmp_cols:
                raise ValueError(f"Distance matrix is missing '{rc}' column for timetable mapping")

        start_dcol = dist_tmp_cols["start"]
        end_dcol   = dist_tmp_cols["end"]
        distm_col  = dist_tmp_cols["distance_m"]
        tmin_col   = dist_tmp_cols["min_travel_time"]
        line_dcol  = dist_tmp_cols["line"]

        lookup_exact: dict[tuple[str,str,str], tuple[float,float]] = {}
        lookup_fallback: dict[tuple[str,str,None], tuple[float,float]] = {}

        # Bouw lookup tabellen
        for _, rowd in dist_tmp.iterrows():
            origin = str(rowd[start_dcol])
            dest   = str(rowd[end_dcol])
            distance_km = float(rowd[distm_col]) / 1000.0
            travel_min  = float(rowd[tmin_col])

            # line kan NaN zijn
            raw_line_val = rowd[line_dcol]
            if pd.isna(raw_line_val):
                key_fb = (origin, dest, None)
                # kies de kortste fallback
                if key_fb not in lookup_fallback or travel_min < lookup_fallback[key_fb][1]:
                    lookup_fallback[key_fb] = (distance_km, travel_min)
            else:
                # zet bv 401.0 -> "401", 400.0 -> "400"
                if isinstance(raw_line_val, float) and raw_line_val.is_integer():
                    line_str = str(int(raw_line_val))
                else:
                    line_str = str(raw_line_val)

                key_ex = (origin, dest, line_str)
                if key_ex not in lookup_exact or travel_min < lookup_exact[key_ex][1]:
                    lookup_exact[key_ex] = (distance_km, travel_min)

        rides: list[Ride] = []
        base_day = datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)

        # Maak Ride objecten
        for _, rowt in df.iterrows():
            start_stop = str(rowt[start_col])
            end_stop   = str(rowt[end_col])

            raw_line_tt = rowt[line_col]
            # zet line in string-vorm
            if isinstance(raw_line_tt, float) and hasattr(raw_line_tt, "is_integer") and raw_line_tt.is_integer():
                line_id = str(int(raw_line_tt))
            else:
                line_id = str(raw_line_tt)

            dep_val = rowt[dep_col]

            # parse departure_time
            if isinstance(dep_val, pd.Timestamp):
                just_time = dep_val.time()
            elif isinstance(dep_val, datetime):
                just_time = dep_val.time()
            elif isinstance(dep_val, str):
                parsed_ok = False
                for fmt in ("%H:%M:%S", "%H:%M"):
                    try:
                        parsed_dt = datetime.strptime(dep_val, fmt)
                        just_time = parsed_dt.time()
                        parsed_ok = True
                        break
                    except ValueError:
                        pass
                if not parsed_ok:
                    raise ValueError(f"Unrecognized time format in timetable: {dep_val}")
            else:
                parsed_dt = pd.to_datetime(dep_val)
                just_time = parsed_dt.time()

            dep_dt = datetime.combine(base_day.date(), just_time)
            if just_time < time(3,0):
                dep_dt += timedelta(days=1)

            # zoek juiste rijtijd / afstand voor deze lijn
            key_exact = (start_stop, end_stop, line_id)
            key_fb    = (start_stop, end_stop, None)

            if key_exact in lookup_exact:
                distance_km, travel_min = lookup_exact[key_exact]
            elif key_fb in lookup_fallback:
                distance_km, travel_min = lookup_fallback[key_fb]
            else:
                # we kunnen deze rit niet mappen op distance matrix
                continue

            arr_dt = dep_dt + timedelta(minutes=travel_min)

            ride = Ride(
                line=line_id,
                start_stop=start_stop,
                end_stop=end_stop,
                start_time=dep_dt,
                end_time=arr_dt,
                distance_km=distance_km,
                travel_min_used=travel_min
            )
            rides.append(ride)

        return rides
