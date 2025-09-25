# app.py
import io
from datetime import datetime
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Prototype groep 8", layout="wide")

# -------------------------------
# Placeholder voor toekomstige optimalisatie (niet actief)
# -------------------------------
def improve_plan(df_input: pd.DataFrame) -> pd.DataFrame:
    """
    TODO (niet-actief):
      - Hier komt later jouw algoritme/heuristiek die het busplan verbetert.
      - Verwacht input-kolommen:
        ["start location", "end location", "start time", "end time",
         "activity", "line", "energy consumption", "bus"]
      - Return: nieuwe/verbeterde DataFrame met de planning.

    Let op: Deze functie doet nu niks (kopie terug) zodat de app gewoon blijft draaien.
    """
    return df_input.copy()

# Sidebar
st.sidebar.header("Menu")
st.sidebar.radio("", ["Option 1", "Option 2", "Option 3", "Option 4"], label_visibility="collapsed")

# Header
st.title("Prototype groep 8")
st.subheader("Hi! Upload existing bus plan here!")
# Accept only .xlsx to avoid xlrd dependency for .xls
uploaded = st.file_uploader("Upload Excel file (.xlsx)", type=["xlsx"])

# Vereiste kolommen (exacte labels, maar we matchen case-insensitive)
required_cols = [
    "start location", "end location", "start time", "end time",
    "activity", "line", "energy consumption", "bus"
]

def normalize_cols(cols):
    # lower-case en strip spaties aan de randen
    return [c.strip().lower() for c in cols]

# State voor rechts/download
if "df_opt" not in st.session_state:
    st.session_state.df_opt = None
if "opt_filename" not in st.session_state:
    st.session_state.opt_filename = None

left, right = st.columns(2, gap="large")

with left:
    st.subheader("Their bus plan, with notes on where optimization is possible")

    if not uploaded:
        st.info("Upload een Excel-bestand met kolommen: " + ", ".join(required_cols))
    else:
        try:
            # Explicit engine; shows clear error if openpyxl is missing
            df_raw = pd.read_excel(uploaded, engine="openpyxl")
            norm = normalize_cols(df_raw.columns)
            colmap = {orig: n for orig, n in zip(df_raw.columns, norm)}

            # Check ontbrekende kolommen
            missing = [c for c in required_cols if c not in norm]
            if missing:
                st.error("Het bestand mist kolommen: " + ", ".join(missing))
                st.session_state.df_opt = None
                st.session_state.opt_filename = None
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

                # ---------------------------------------
                # Rechts (no-op) "optimalisatie" resultaat
                # ---------------------------------------
                df_opt = improve_plan(df[order])  # nu pass-through
                st.session_state.df_opt = df_opt

                # Bestandsnaam met timestamp
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                st.session_state.opt_filename = f"optimized_bus_plan_{ts}.xlsx"

        except ImportError:
            st.error(
                "Kon Excel niet lezen: ontbrekende dependency.\n\n"
                "Installeer aub: `pip install openpyxl` en probeer opnieuw."
            )
            st.session_state.df_opt = None
            st.session_state.opt_filename = None
        except Exception as e:
            st.error(f"Kon Excel niet lezen: {e}")
            st.session_state.df_opt = None
            st.session_state.opt_filename = None

with right:
    st.subheader("Newly made bus plan, optimised by us")

    # Alleen een NIET-uitvoerende placeholder tonen
    with st.expander("📄 Bekijk (niet-actieve) placeholder voor optimalisatiecode", expanded=False):
        placeholder_text = """
# --- PSEUDO/PLACEHOLDER (wordt NIET uitgevoerd) ---
# Voorbeeld-structuur van jouw algoritme (later invullen):
# def improve_plan(df):
#     # 1) Valideer input
#     # 2) Bouw graf/constraint model
#     # 3) Los optimalisatie/heuristiek
#     # 4) Construeer nieuw schema
#     # 5) Return nieuw DataFrame
#     return df_out
# ---------------------------------------------------
"""
        st.code(placeholder_text, language="python")

    if st.session_state.df_opt is None:
        st.info("Hier komt straks jullie geoptimaliseerde busplan (algoritme/heuristiek).")
    else:
        st.dataframe(st.session_state.df_opt, use_container_width=True)

# ------------------------------------------------------
# DOWNLOAD-KNOP ONDERAAN, GE-CENTREERD (RECHTER EXCEL)
# ------------------------------------------------------
st.divider()
c1, c2, c3 = st.columns([1, 2, 1])
with c2:
    if st.session_state.df_opt is not None and not st.session_state.df_opt.empty:
        # Schrijf DataFrame naar in-memory Excel
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            st.session_state.df_opt.to_excel(writer, index=False, sheet_name="OptimizedPlan")
        buffer.seek(0)

        st.download_button(
            label="Download optimized Excel",
            data=buffer,
            file_name=st.session_state.opt_filename or "optimized_bus_plan.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
    else:
        st.download_button(
            label="Download optimized Excel",
            data=b"",
            file_name="optimized_bus_plan.xlsx",
            disabled=True,
            help="Upload eerst een geldig bestand; de (toekomstige) optimalisatie verschijnt rechts.",
            use_container_width=True,
        )
