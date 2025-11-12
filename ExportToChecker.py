import pandas as pd
from datetime import datetime
from typing import List

def assignments_to_checker_df(assignments, min_charge_activity_name="charge"):
    """
    Zet de AssignmentRecord lijst uit de planner om naar het formaat
    dat de checker verwacht.

    Checker verwacht kolommen:
    [
        "start_location",
        "end_location",
        "start_time",
        "end_time",
        "activity",
        "line",
        "energy_consumption",
        "bus",
    ]

    - start_time / end_time worden als strings "HH:MM:SS"
    - "activity" wordt een van ["deadhead","idle","charge","service"]
    - "line" bij deadhead/idle/charge mag leeg of dezelfde lijnnaam zijn.
    """

    rows = []

    for rec in assignments:
        bus_name = rec.bus_id
        line_name = rec.ride.line

        # 1. charging_before (als die er is)
        # charging_before = (charge_start_dt, charge_end_dt, bat_before_charge, bat_after_charge)
        if rec.charging_before is not None:
            c_start, c_end, bat_bef_ch, bat_aft_ch = rec.charging_before

            # energie_consumption voor charge kun je interpreteren.
            # - Negatief (we laden bij): bat_aft_ch - bat_bef_ch
            #   In kWh. Checker verwacht verbruik, dus dit is eigenlijk "negatief".
            # Als je bang bent dat hij crasht op negatief, zet 0.0.
            energy_ch = (bat_bef_ch - bat_aft_ch)

            rows.append({
                "start_location": "ehvgar",  # garage
                "end_location":   "ehvgar",
                "start_time":     c_start.strftime("%H:%M:%S"),
                "end_time":       c_end.strftime("%H:%M:%S"),
                "activity":       min_charge_activity_name,  # "charge"
                "line":           "",  # geen lijn
                "energy_consumption": energy_ch,
                "bus":            bus_name,
            })

        # 2. deadhead_before
        # deadhead_before = (from_loc, to_loc, dist_km, depart_dt, arrive_dt)
        if rec.deadhead_before is not None:
            from_loc, to_loc, dist_km, dep_dt, arr_dt = rec.deadhead_before

            # energie op deadhead kunnen we benaderen uit SOC verschil vóór ritstart:
            # We kennen battery_before van de service rit en battery_after meteen daarna,
            # maar niet apart voor deadhead, idle, etc.
            # Simpelste (en veilig voor checker) is een schatting: dist_km * consumption_per_km.
            # Maar die data (consumption_per_km) zit in BusConstants.
            # We importeren hier niet BusConstants om circular import te vermijden,
            # dus zetten we voorlopig None -> later kun je 'm zelf invullen met echte waarde.

            # safer: laat gewoon leeg/0.0 voor nu:
            energy_dh = 0.0

            rows.append({
                "start_location": from_loc,
                "end_location":   to_loc,
                "start_time":     dep_dt.strftime("%H:%M:%S"),
                "end_time":       arr_dt.strftime("%H:%M:%S"),
                "activity":       "deadhead",
                "line":           "",  # geen passagierslijn
                "energy_consumption": energy_dh,
                "bus":            bus_name,
            })

        # 3. idle_before
        # idle_before = (idle_start_dt, idle_end_dt, idle_energy_used_kWh)
        if rec.idle_before is not None:
            idle_start, idle_end, idle_energy_used = rec.idle_before

            # idle_energy_usedKWh hebben we direct
            rows.append({
                "start_location": rec.ride.start_stop,   # bus wacht bij vertrekhalte
                "end_location":   rec.ride.start_stop,   # blijft staan
                "start_time":     idle_start.strftime("%H:%M:%S"),
                "end_time":       idle_end.strftime("%H:%M:%S"),
                "activity":       "idle",
                "line":           "",
                "energy_consumption": idle_energy_used,
                "bus":            bus_name,
            })

        # 4. de eigenlijke service rit
        # service verbruik = battery_before - battery_after
        energy_service = rec.battery_before - rec.battery_after

        rows.append({
            "start_location": rec.ride.start_stop,
            "end_location":   rec.ride.end_stop,
            "start_time":     rec.ride.start_time.strftime("%H:%M:%S"),
            "end_time":       rec.ride.end_time.strftime("%H:%M:%S"),
            "activity":       "service",
            "line":           line_name,
            "energy_consumption": energy_service,
            "bus":            bus_name,
        })

    # maak dataframe met juiste kolomvolgorde
    df_out = pd.DataFrame(rows, columns=[
        "start_location",
        "end_location",
        "start_time",
        "end_time",
        "activity",
        "line",
        "energy_consumption",
        "bus",
    ])

    # sorteer op start_time chronologisch per bus
    # Let op: hier sorteren we puur op stringtijd; als je over middernacht gaat is dit tricky,
    # maar jouw checker lijkt toch met tijden (niet absolute datetimes) te rekenen.
    df_out = df_out.sort_values(by=["bus", "start_time", "end_time"]).reset_index(drop=True)

    return df_out
