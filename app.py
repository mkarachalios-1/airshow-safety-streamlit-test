import streamlit as st
import pandas as pd
import numpy as np
import json
from pathlib import Path

# --------- resilient plotly import ----------
try:
    import plotly.graph_objects as go
except Exception:
    import sys, subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "plotly"])
    import plotly.graph_objects as go

st.set_page_config(page_title="Airshow Safety & Excellence Database", layout="wide")

# --------- paths ----------
DATA_DIR   = Path(__file__).parent / "data"
ACC_PATH   = DATA_DIR / "airshow_accidents.json"
RATES_PATH = DATA_DIR / "historical_rates.json"

# --------- helpers ----------

_CONTINENT_MAP = {
    # North America
    "united states": "North America", "usa": "North America", "u.s.a.": "North America", "us": "North America",
    "canada": "North America", "mexico": "North America",
    # South America
    "brazil": "South America", "argentina": "South America", "chile": "South America",
    "peru": "South America", "colombia": "South America", "uruguay": "South America",
    "paraguay": "South America", "bolivia": "South America",
    # Europe
    "united kingdom": "Europe", "uk": "Europe", "england": "Europe", "scotland": "Europe",
    "wales": "Europe", "northern ireland": "Europe", "great britain": "Europe", "britain": "Europe",
    "ireland": "Europe", "france": "Europe", "germany": "Europe", "italy": "Europe", "spain": "Europe",
    "portugal": "Europe", "netherlands": "Europe", "belgium": "Europe", "switzerland": "Europe",
    "austria": "Europe", "sweden": "Europe", "norway": "Europe", "finland": "Europe", "denmark": "Europe",
    "iceland": "Europe", "czech republic": "Europe", "czechia": "Europe", "poland": "Europe",
    "hungary": "Europe", "romania": "Europe", "bulgaria": "Europe", "greece": "Europe", "serbia": "Europe",
    "croatia": "Europe", "slovenia": "Europe", "slovakia": "Europe", "bosnia": "Europe",
    "north macedonia": "Europe", "albania": "Europe", "estonia": "Europe", "latvia": "Europe",
    "lithuania": "Europe", "ukraine": "Europe", "russia": "Europe", "turkey": "Europe",
    "türkiye": "Europe", "turkiye": "Europe",
    "cyprus": "Europe",
    # Asia
    "china": "Asia", "japan": "Asia", "south korea": "Asia", "korea, south": "Asia", "north korea": "Asia",
    "india": "Asia", "pakistan": "Asia", "bangladesh": "Asia", "sri lanka": "Asia", "nepal": "Asia",
    "myanmar": "Asia", "thailand": "Asia", "vietnam": "Asia", "malaysia": "Asia", "singapore": "Asia",
    "indonesia": "Asia", "philippines": "Asia", "taiwan": "Asia", "hong kong": "Asia", "mongolia": "Asia",
    "laos": "Asia", "cambodia": "Asia", "israel": "Asia", "jordan": "Asia", "lebanon": "Asia",
    "saudi arabia": "Asia", "united arab emirates": "Asia", "uae": "Asia", "qatar": "Asia", "bahrain": "Asia",
    "oman": "Asia", "kuwait": "Asia", "iran": "Asia", "iraq": "Asia", "yemen": "Asia",
    # Africa
    "south africa": "Africa", "egypt": "Africa", "morocco": "Africa", "kenya": "Africa", "ethiopia": "Africa",
    "nigeria": "Africa", "ghana": "Africa", "tunisia": "Africa", "algeria": "Africa", "tanzania": "Africa",
    "uganda": "Africa", "zambia": "Africa", "zimbabwe": "Africa", "namibia": "Africa", "botswana": "Africa",
    "senegal": "Africa", "cote d'ivoire": "Africa", "ivory coast": "Africa",
    # Oceania
    "australia": "Oceania", "new zealand": "Oceania",
}

def country_to_continent(country: str) -> str:
    if not isinstance(country, str) or not country.strip():
        return "Unknown"
    key = country.strip().lower()
    return _CONTINENT_MAP.get(key, "Unknown")

def numcol(df: pd.DataFrame, name: str) -> pd.Series:
    """Safe numeric accessor for optional columns."""
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce").fillna(0)
    return pd.Series(0, index=df.index, dtype="float64")

# --------- data loading & cleaning ----------

@st.cache_data
def load_data():
    # base dataset from repo
    if ACC_PATH.exists():
        df = pd.read_json(ACC_PATH)
    else:
        df = pd.DataFrame()

    if df.empty:
        hist = {"years": [], "BAAR": [], "AFR": [], "ACR": [], "AER": []}
        if RATES_PATH.exists():
            try:
                hist = json.loads(RATES_PATH.read_text())
            except Exception:
                pass
        return df, hist

    # Parse dates
    df["date_parsed"] = pd.to_datetime(df.get("date"), errors="coerce")

    # derive year robustly
    year_raw = pd.to_numeric(df.get("year"), errors="coerce")
    year_from_date = df["date_parsed"].dt.year
    df["year"] = year_raw.fillna(year_from_date)

    # ensure casualties
    for c in ["pilot_killed", "pilot_injured", "crew_kill", "crew_inj", "pax_kill", "pax_inj"]:
        if c not in df.columns:
            df[c] = 0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    if "fatalities" not in df.columns:
        df["fatalities"] = (df["pilot_killed"] + df["crew_kill"] + df["pax_kill"]).astype(int)
    else:
        df["fatalities"] = pd.to_numeric(df["fatalities"], errors="coerce").fillna(
            df["pilot_killed"] + df["crew_kill"] + df["pax_kill"]
        ).astype(int)

    if "casualties" not in df.columns:
        df["casualties"] = (df["fatalities"] + df["pilot_injured"] + df["crew_inj"] + df["pax_inj"]).astype(int)
    else:
        df["casualties"] = pd.to_numeric(df["casualties"], errors="coerce").fillna(
            df["fatalities"] + df["pilot_injured"] + df["crew_inj"] + df["pax_inj"]
        ).astype(int)

    # 5M flags
    for c in [
        "fit", "mac", "loc", "mechanical", "enviro",
        "man_factor", "machine_factor", "medium_factor", "mission_factor", "management_factor"
    ]:
        if c not in df.columns:
            df[c] = 0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    # continent
    df["country"]   = df.get("country", "").astype(str).str.strip()
    df["continent"] = df["country"].apply(country_to_continent)

    # Historical rates
    hist = {"years": [], "BAAR": [], "AFR": [], "ACR": [], "AER": []}
    if RATES_PATH.exists():
        try:
            hist = json.loads(RATES_PATH.read_text())
        except Exception:
            pass

    return df, hist

df, hist = load_data()

# --------- UI header ----------
st.title("Airshow Safety & Excellence Database")
st.markdown(
    "#### 5M-aligned repository of airshow accidents (1908–2025). "
    "Use search and filters below. Charts update as you filter."
)
st.markdown("### Barker Airshow Incident & Accident Database")

# Basic debug: how many records loaded
st.caption(f"Records loaded from JSON: {len(df)}")

# --------- Text search ----------
q = st.text_input("Search", placeholder="e.g. engine, loop, MAC, Duxford")

# --------- Year range ----------
if not df.empty and df["year"].dropna().size > 0:
    ymin, ymax = int(df["year"].min()), int(df["year"].max())
else:
    ymin, ymax = 1908, 2025

c_year1, c_year2 = st.columns(2)
year_from = c_year1.number_input("Year from", value=ymin, min_value=ymin, max_value=ymax, step=1)
year_to   = c_year2.number_input("to",        value=ymax, min_value=ymin, max_value=ymax, step=1)

# --------- Dropdown filters ----------
col_a, col_b, col_c = st.columns(3)

aircraft_options = sorted(
    [x for x in df.get("aircraft_type", pd.Series([], dtype=str)).dropna().unique() if str(x).strip()]
)
country_options = sorted(
    [x for x in df.get("country", pd.Series([], dtype=str)).dropna().unique() if str(x).strip()]
)
continent_options = sorted(
    [x for x in df.get("continent", pd.Series([], dtype=str)).dropna().unique() if str(x).strip()]
)

sel_aircraft  = col_a.multiselect("Aircraft type", aircraft_options, default=[])
sel_country   = col_b.multiselect("Country", country_options, default=[])
sel_continent = col_c.multiselect("Continent", continent_options, default=[])

col_d, col_e = st.columns(2)
manoeuvre_options = sorted(
    [x for x in df.get("manoeuvre", pd.Series([], dtype=str)).dropna().unique() if str(x).strip()]
)
sel_manoeuvre = col_d.multiselect("Manoeuvre", manoeuvre_options, default=[])

severity = col_e.selectbox(
    "Severity",
    [
        "Any",
        "Fatal (any)",
        "Non-fatal only",
        "Fatal: Pilot/Crew",
        "Fatal: Passengers",
        "Fatal: Spectators/Crowd",
    ],
    index=0,
)
severity = str(severity)

# --------- Event type & 5M toggles ----------
c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
acc_on = c1.checkbox("Accidents", True)
inc_on = c2.checkbox("Incidents", True)
m_man  = c3.checkbox("Man", True)
m_mach = c4.checkbox("Machine", True)
m_med  = c5.checkbox("Medium", True)
m_mis  = c6.checkbox("Mission", True)
m_mgmt = c7.checkbox("Management", True)

# --------- Apply filters ----------

# --------- Apply filters ----------
if not df.empty:
    # Make sure year is numeric; NaNs indicate missing or unparseable years
    year_series = pd.to_numeric(df.get("year"), errors="coerce")

    # Inside the selected range
    in_range = (year_series >= year_from) & (year_series <= year_to)

    # KEEP rows with missing year instead of silently dropping them
    f = df[in_range | year_series.isna()].copy()
else:
    f = df.copy()

if not f.empty and sel_aircraft:
    f = f[f["aircraft_type"].isin(sel_aircraft)]
if not f.empty and sel_country:
    f = f[f["country"].isin(sel_country)]
if not f.empty and sel_continent:
    f = f[f["continent"].isin(sel_continent)]
if not f.empty and sel_manoeuvre:
    f = f[f["manoeuvre"].isin(sel_manoeuvre)]

# Severity filters
if not f.empty and severity != "Any":
    any_fatal   = (numcol(f, "fatalities") > 0)
    pc_fatal    = (numcol(f, "pilot_killed") + numcol(f, "crew_kill") > 0)
    pax_fatal   = (numcol(f, "pax_kill") > 0)
    crowd_fatal = (numcol(f, "spec_kill") + numcol(f, "pub_kill") > 0)

    if severity == "Fatal (any)":
        f = f[any_fatal]
    elif severity == "Non-fatal only":
        f = f[~any_fatal]
    elif severity == "Fatal: Pilot/Crew":
        f = f[pc_fatal]
    elif severity == "Fatal: Passengers":
        f = f[pax_fatal]
    elif severity == "Fatal: Spectators/Crowd":
        f = f[crowd_fatal]

# Accidents vs incidents
if not f.empty:
    if acc_on and not inc_on:
        f = f[f["casualties"] > 0]
    elif inc_on and not acc_on:
        f = f[f["casualties"] == 0]

# 5M mask
if not f.empty:
    selected_5m = [m_man, m_mach, m_med, m_mis, m_mgmt]
    n_selected = sum(selected_5m)

    # Only filter if the user has chosen a subset of 5M factors
    # 0 selected  -> no 5M filtering
    # 5 selected  -> no 5M filtering (show all, including 0-factor events)
    if 0 < n_selected < 5:
        mask_5m = pd.Series(False, index=f.index)
        if m_man:  mask_5m |= (f["man_factor"] == 1)
        if m_mach: mask_5m |= (f["machine_factor"] == 1)
        if m_med:  mask_5m |= (f["medium_factor"] == 1)
        if m_mis:  mask_5m |= (f["mission_factor"] == 1)
        if m_mgmt: mask_5m |= (f["management_factor"] == 1)
        f = f[mask_5m]
    # else: all or none selected → leave f unchanged


# Text search
if not f.empty and q:
    hay = (
        f.get("aircraft_type", "").astype(str) + " " +
        f.get("category", "").astype(str) + " " +
        f.get("manoeuvre", "").astype(str) + " " +
        f.get("event_name", "").astype(str) + " " +
        f.get("location", "").astype(str) + " " +
        f.get("country", "").astype(str) + " " +
        f.get("remarks", "").astype(str) + " " +
        f.get("contributing_factor", "").astype(str) + " " +
        f["date_parsed"].astype(str)
    ).str.lower()
    f = f[hay.str.contains(q.lower(), na=False)]

# --------- KPIs ----------
k1, k2, k3 = st.columns(3)
if not f.empty:
    k1.metric("Accidents/Incidents", int(f.shape[0]))
    k2.metric("Fatalities", int(f["fatalities"].sum()))
    k3.metric("Casualties", int(f["casualties"].sum()))
else:
    k1.metric("Accidents/Incidents", 0)
    k2.metric("Fatalities", 0)
    k3.metric("Casualties", 0)

# --------- Charts ----------
# 1. Filtered accidents per year
if not f.empty and "year" in f.columns:
    by_year = f.dropna(subset=["year"]).groupby("year").size().reset_index(name="count")
    fig1 = go.Figure()
    fig1.add_trace(
        go.Scatter(
            x=by_year["year"],
            y=by_year["count"],
            mode="lines+markers",
            name="Accidents/Incidents (filtered)",
        )
    )
    fig1.update_layout(height=400, margin=dict(l=12, r=12, t=28, b=12))
    st.plotly_chart(fig1, use_container_width=True)

st.divider()

# 2. Historical BAAR / AFR / ACR + AER
if hist.get("years"):
    years = hist["years"]
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=years, y=hist["BAAR"], mode="lines+markers",
        name="BAAR (per 10k)", line=dict(color="#e74c3c")
    ))
    fig2.add_trace(go.Scatter(
        x=years, y=hist["AFR"], mode="lines+markers",
        name="AFR (per 10k)"
    ))
    fig2.add_trace(go.Scatter(
        x=years, y=hist["ACR"], mode="lines+markers",
        name="ACR (per 10k)"
    ))
    fig2.add_trace(go.Scatter(
        x=years, y=hist["AER"], mode="lines+markers",
        name="AER (%)", yaxis="y2", line=dict(color="#2ecc71")
    ))
    fig2.update_layout(
        yaxis=dict(title="per 10k events"),
        yaxis2=dict(
            title="AER (%)",
            overlaying="y",
            side="right",
            range=[99.8, 100.0],
            tickformat=".3f",
            tick0=99.8,
            dtick=0.05,
        ),
        height=400,
        margin=dict(l=12, r=12, t=28, b=12),
    )
    st.plotly_chart(fig2, use_container_width=True)

st.divider()

# 3. 5M pie
if not f.empty:
    five = {
        "Man":        int((f["man_factor"] == 1).sum()),
        "Machine":    int((f["machine_factor"] == 1).sum()),
        "Medium":     int((f["medium_factor"] == 1).sum()),
        "Mission":    int((f["mission_factor"] == 1).sum()),
        "Management": int((f["management_factor"] == 1).sum()),
    }
    fig3 = go.Figure(data=[go.Pie(labels=list(five.keys()), values=list(five.values()))])
    fig3.update_traces(textinfo="percent+label", textposition="outside", automargin=True)
    fig3.update_layout(
        height=440,
        margin=dict(l=60, r=220, t=40, b=60),
        showlegend=True,
        legend=dict(orientation="v", x=1.02, y=0.5, xanchor="left", yanchor="middle"),
    )
    st.plotly_chart(fig3, use_container_width=True)

st.divider()

# 4. Aerobatic manoeuvre pie (including Split-S)
if not f.empty:
    keys = [
        ("cuban 8", "Cuban 8"), ("cuban eight", "Cuban 8"),
        ("loop", "Loop"), ("immelman", "Immelman"), ("immelmann", "Immelman"),
        ("split s", "Split-S"),
        ("barrel roll", "Barrel roll"), ("aileron roll", "Roll"), ("roll", "Roll"),
        ("spin", "Spin"),
        ("hammerhead", "Hammerhead"), ("stall turn", "Hammerhead"),
        ("tail slide", "Tailslide"), ("tailslide", "Tailslide"),
        ("snap roll", "Snap roll"), ("lomcevak", "Lomcevak"),
    ]
    txt = (
        f.get("manoeuvre", "").astype(str) + " " +
        f.get("remarks", "").astype(str) + " " +
        f.get("contributing_factor", "").astype(str)
    ).str.lower()
    counts = {}
    for pat, label in keys:
        counts[label] = counts.get(label, 0) + int(txt.str.contains(pat).sum())
    items = [(k, v) for k, v in counts.items() if v > 0]
    items.sort(key=lambda x: x[1], reverse=True)
    other = sum(v for _, v in items[10:])
    items = items[:10]
    if other > 0:
        items.append(("Other", other))
    if items:
        fig4 = go.Figure(data=[go.Pie(
            labels=[i[0] for i in items],
            values=[i[1] for i in items],
        )])
        fig4.update_traces(textinfo="percent+label", textposition="outside", automargin=True)
        fig4.update_layout(
            height=440,
            margin=dict(l=60, r=220, t=40, b=60),
            showlegend=True,
            legend=dict(orientation="v", x=1.02, y=0.5, xanchor="left", yanchor="middle"),
        )
        st.plotly_chart(fig4, use_container_width=True)

st.divider()

# --------- Table (newest first) ----------
if not f.empty:
    f_sorted = f.sort_values("date_parsed", ascending=False)
    show_cols = [c for c in [
        "date_parsed", "aircraft_type", "category",
        "country", "continent",
        "manoeuvre", "fit", "mac", "loc", "mechanical", "enviro",
        "fatalities", "casualties", "pilot_killed", "crew_kill", "pax_kill",
        "event_name", "location", "contributing_factor",
        "man_factor", "machine_factor", "medium_factor", "mission_factor", "management_factor",
        "remarks",
    ] if c in f_sorted.columns]
    disp = f_sorted[show_cols].copy()
    disp["date_parsed"] = pd.to_datetime(disp["date_parsed"], errors="coerce").dt.strftime("%Y-%m-%d")
    disp = disp.rename(columns={"date_parsed": "date"})
else:
    disp = pd.DataFrame()

st.dataframe(disp, use_container_width=True, hide_index=True)
