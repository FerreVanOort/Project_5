# app.py
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Prototype name + Group number", layout="wide")

# Sidebar
st.sidebar.header("Menu")
st.sidebar.radio("", ["Option 1", "Option 2", "Option 3", "Option 4"], label_visibility="collapsed")

# Header
st.title("Prototype name + Group number")
st.subheader("Hi! Upload existing bus plan here!")
uploaded = st.file_uploader("Upload Excel file (.xlsx / .xls)", type=["xlsx", "xls"])

# Vereiste kolommen (exacte labels, maar we matchen case-insensitive)
required_cols = [
    "start location", "end location", "start time", "end time",
    "activity", "line", "energy consumption", "bus"
]

def normalize_cols(cols):
    # lower-case en strip spaties aan de randen
    return [c.strip().lower() for c in cols]

left, right = st.columns(2, gap="large")

with left:
    st.subheader("Their bus plan, with notes on where optimization is possible")

    if not uploaded:
        st.info("Upload een Excel-bestand met kolommen: " + ", ".join(required_cols))
    else:
        try:
            df_raw = pd.read_excel(uploaded)
            norm = normalize_cols(df_raw.columns)
            colmap = {orig: n for orig, n in zip(df_raw.columns, norm)}

            # Check ontbrekende kolommen
            missing = [c for c in required_cols if c not in norm]
            if missing:
                st.error("Het bestand mist kolommen: " + ", ".join(missing))
            else:
                # Hernoem tijdelijk naar genormaliseerde namen voor verwerking
                df = df_raw.rename(columns={k: v for k, v in colmap.items() if v in required_cols})

                # Parse tijden (niet-fatal bij fouten)
                for tcol in ["start time", "end time"]:
                    df[tcol] = pd.to_datetime(df[tcol], errors="coerce")

                # Alleen de gewenste kolommen in vaste volgorde
                order = required_cols
                st.dataframe(df[order], use_container_width=True)

                # ----- Notes / simpele checks -----
                st.markdown("### Notes")

                # 1) Negatieve of ontbrekende tijden
                if df["start time"].isna().any() or df["end time"].isna().any():
                    st.write("- Er staan ongeldige/lege tijdstippen in start/end time.")
                neg = (df["end time"] - df["start time"]).dt.total_seconds()
                if neg.notna().any() and (neg < 0).any():
                    st.write("- Er zijn ritten met eindtijd vóór starttijd (controleer invoer).")

                # 2) Overlappingen per bus (eenvoudige check)
                if {"bus", "start time", "end time"}.issubset(df.columns):
                    overlaps_found = False
                    for bus_id, g in df.sort_values(["bus", "start time"]).groupby("bus"):
                        # Een overlap als volgende start < huidige eind
                        s = g["start time"].values
                        e = g["end time"].values
                        for i in range(len(g) - 1):
                            if pd.notna(s[i+1]) and pd.notna(e[i]) and s[i+1] < e[i]:
                                overlaps_found = True
                                break
                        if overlaps_found:
                            break
                    if overlaps_found:
                        st.write("- Mogelijke overlappende ritten op dezelfde bus.")

                # 3) Dubbele regels (heuristiek)
                if df.duplicated(subset=["start location", "end location", "start time", "line"]).any():
                    st.write("- Dubbele ritten gevonden (zelfde start/end/time/line).")

                # 4) Energie samenvatting (indien numeriek)
                if "energy consumption" in df.columns:
                    try:
                        energy = pd.to_numeric(df["energy consumption"], errors="coerce")
                        total = energy.sum(min_count=1)
                        if pd.notna(total):
                            st.write(f"- Totale energy consumption in bestand: **{total:.2f}**")
                    except Exception:
                        pass

                if "activity" in df.columns and df["activity"].nunique() > 1:
                    st.write(f"- Activiteiten gevonden: {', '.join(sorted(map(str, df['activity'].dropna().unique())))}")

                # Altijd ten minste één note tonen
                st.write("- Handmatige review van planning en capaciteit aanbevolen.")
        except Exception as e:
            st.error(f"Kon Excel niet lezen: {e}")

with right:
    st.subheader("Newly made bus plan, optimised by us")
    st.info("Hier komt straks jullie geoptimaliseerde busplan (algoritme/heuristiek).")
