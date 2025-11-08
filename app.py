import streamlit as st
import pandas as pd
import numpy as np
import json
from pathlib import Path

# --- resilient plotly import ---
try:
    import plotly.graph_objects as go
except Exception:
    import sys, subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "plotly"])
    import plotly.graph_objects as go

st.set_page_config(page_title="Airshow Safety & Excellence Database", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_json(Path("data/airshow_accidents.json"))
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if "year" not in df.columns:
        df["year"] = df["date"].dt.year
    df["fatalities"] = pd.to_numeric(df.get("fatalities", 0), errors="coerce").fillna(0).astype(int)
    df["casualties"]  = pd.to_numeric(df.get("casualties", 0), errors="coerce").fillna(0).astype(int)
    for c in ["man_factor","machine_factor","medium_factor","mission_factor","management_factor"]:
        if c not in df.columns: df[c] = 0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    hist = json.loads(Path("data/historical_rates.json").read_text())
    return df, hist

df, hist = load_data()

st.title("Airshow Safety & Excellence Database")
st.markdown("#### 5M-aligned repository of airshow accidents (1908–2025). Use search and filters below. Charts update as you filter.")
st.markdown("### Barker Airshow Incident & Accident Database")

# --- Filters ---
q = st.text_input("Search", placeholder="e.g. engine, loop, MAC, Duxford")
ymin, ymax = int(df["year"].min()), int(df["year"].max())
y1, y2 = st.columns(2)
year_from = y1.number_input("Year from", value=ymin, min_value=ymin, max_value=ymax, step=1)
year_to   = y2.number_input("to",        value=ymax, min_value=ymin, max_value=ymax, step=1)

c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
acc_on = c1.checkbox("Accidents", True)
inc_on = c2.checkbox("Incidents", True)
m_man  = c3.checkbox("Man", True)
m_mach = c4.checkbox("Machine", True)
m_med  = c5.checkbox("Medium", True)
m_mis  = c6.checkbox("Mission", True)
m_mgmt = c7.checkbox("Management", True)

# --- Apply filters ---
f = df[(df["year"] >= year_from) & (df["year"] <= year_to)].copy()

# Event type by casualties
if acc_on and not inc_on:
    f = f[f["casualties"] > 0]
elif inc_on and not acc_on:
    f = f[f["casualties"] == 0]

# 5M mask (any selected)
if any([m_man, m_mach, m_med, m_mis, m_mgmt]):
    mask = pd.Series(False, index=f.index)
    if m_man:  mask |= (f["man_factor"] == 1)
    if m_mach: mask |= (f["machine_factor"] == 1)
    if m_med:  mask |= (f["medium_factor"] == 1)
    if m_mis:  mask |= (f["mission_factor"] == 1)
    if m_mgmt: mask |= (f["management_factor"] == 1)
    f = f[mask]

if q:
    hay = (
        f.get("aircraft_type","").astype(str) + " " +
        f.get("category","").astype(str) + " " +
        f.get("manoeuvre","").astype(str) + " " +
        f.get("event_name","").astype(str) + " " +
        f.get("location","").astype(str) + " " +
        f.get("country","").astype(str) + " " +
        f.get("remarks","").astype(str) + " " +
        f.get("contributing_factor","").astype(str) + " " +
        f["date"].astype(str)
    ).str.lower()
    f = f[hay.str.contains(q.lower(), na=False)]

# --- KPIs ---
k1, k2, k3 = st.columns(3)
k1.metric("Accidents/Incidents", int(f.shape[0]))
k2.metric("Fatalities", int(f["fatalities"].sum()))
k3.metric("Casualties", int(f["casualties"].sum()))

# --- Charts ---
by_year = f.dropna(subset=["year"]).groupby("year").size().reset_index(name="count")
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=by_year["year"], y=by_year["count"], mode="lines+markers", name="Accidents (filtered)"))
fig1.update_layout(height=360, margin=dict(l=10,r=10,t=10,b=10))
st.plotly_chart(fig1, use_container_width=True)

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=hist["years"], y=hist["BAAR"], mode="lines+markers", name="BAAR (per 10k)"))
fig2.add_trace(go.Scatter(x=hist["years"], y=hist["AFR"],  mode="lines+markers", name="AFR (per 10k)"))
fig2.add_trace(go.Scatter(x=hist["years"], y=hist["ACR"],  mode="lines+markers", name="ACR (per 10k)"))
fig2.add_trace(go.Scatter(x=hist["years"], y=hist["AER"],  mode="lines+markers", name="AER (%)", yaxis="y2"))
fig2.update_layout(
    yaxis=dict(title="per 10k events"),
    yaxis2=dict(
        title="AER (%)",
        overlaying="y",
        side="right",
        range=[99.8, 100],      # ← was [99, 100]
        tickformat=".3f",
        tick0=99.8,
        dtick=0.05
    ),
    height=360, margin=dict(l=10, r=10, t=10, b=10)
)
st.plotly_chart(fig2, use_container_width=True)

five = {
    "Man": int((f["man_factor"]==1).sum()),
    "Machine": int((f["machine_factor"]==1).sum()),
    "Medium": int((f["medium_factor"]==1).sum()),
    "Mission": int((f["mission_factor"]==1).sum()),
    "Management": int((f["management_factor"]==1).sum())
}
fig3 = go.Figure(data=[go.Pie(labels=list(five.keys()), values=list(five.values()))])
fig3.update_traces(textinfo="percent+label", hovertemplate="%{label}: %{value}")
fig3.update_layout(height=360, margin=dict(l=10,r=10,t=10,b=10), showlegend=True)
st.plotly_chart(fig3, use_container_width=True)

keys = [
    ("cuban 8","Cuban 8"), ("cuban eight","Cuban 8"),
    ("loop","Loop"), ("immelman","Immelman"), ("immelmann","Immelman"),
    ("split s","Split-S"),
    ("barrel roll","Barrel roll"), ("aileron roll","Roll"), ("roll","Roll"),
    ("spin","Spin"),
    ("hammerhead","Hammerhead"), ("stall turn","Hammerhead"),
    ("tail slide","Tailslide"), ("tailslide","Tailslide"),
    ("snap roll","Snap roll"), ("lomcevak","Lomcevak")
]
text = (
    f.get("manoeuvre","").astype(str) + " " +
    f.get("remarks","").astype(str) + " " +
    f.get("contributing_factor","").astype(str)
).str.lower()
counts = {}
for pat, label in keys:
    counts[label] = counts.get(label, 0) + text.str.contains(pat).sum()
items = sorted([(k,v) for k,v in counts.items() if v>0], key=lambda x: x[1], reverse=True)
other = sum(v for _,v in items[10:])
items = items[:10]
if other>0: items.append(("Other", other))
if items:
    fig4 = go.Figure(data=[go.Pie(labels=[i[0] for i in items], values=[i[1] for i in items])])
    fig4.update_traces(textinfo="percent+label", hovertemplate="%{label}: %{value}")
    fig4.update_layout(height=360, margin=dict(l=10,r=10,t=10,b=10), showlegend=True)
    st.plotly_chart(fig4, use_container_width=True)

# --- Table ---
show_cols = [c for c in [
    "date","aircraft_type","category","country","manoeuvre","fit","mac","loc","mechanical","enviro",
    "fatalities","casualties","event_name","location","contributing_factor",
    "man_factor","machine_factor","medium_factor","mission_factor","management_factor","remarks"
] if c in f.columns]
f_out = f[show_cols].sort_values("date", ascending=False).reset_index(drop=True)
st.dataframe(f_out, use_container_width=True, hide_index=True)

# --- Admin (local append; optional GitHub commit) ---
with st.expander("Admin: add incident/accident", expanded=False):
    ok = True
    if "ADMIN_PASSWORD" in st.secrets:
        ok = st.text_input("Admin password", type="password") == st.secrets["ADMIN_PASSWORD"]
    if ok:
        with st.form("add_record"):
            colA, colB, colC = st.columns(3)
            date = colA.date_input("Date")
            aircraft_type = colB.text_input("Aircraft type")
            category = colC.selectbox("Category", ["Accident","Incident"])
            col1, col2, col3 = st.columns(3)
            country = col1.text_input("Country")
            event_name = col2.text_input("Event name")
            location = col3.text_input("Location")
            manoeuvre = st.text_input("Manoeuvre")
            remarks = st.text_area("Remarks / Notes")
            contributing_factor = st.text_area("Contributing factor")
            st.write("Casualties")
            cf1, cf2, cf3 = st.columns(3)
            pilot_killed  = cf1.number_input("Pilot killed",  min_value=0, step=1)
            pilot_injured = cf1.number_input("Pilot injured", min_value=0, step=1)
            crew_kill     = cf2.number_input("Crew killed",   min_value=0, step=1)
            crew_inj      = cf2.number_input("Crew injured",  min_value=0, step=1)
            pax_kill      = cf3.number_input("Pax killed",    min_value=0, step=1)
            pax_inj       = cf3.number_input("Pax injured",   min_value=0, step=1)
            fx1, fx2, fx3, fx4, fx5 = st.columns(5)
            fit = fx1.checkbox("FIT")
            mac = fx2.checkbox("MAC")
            loc = fx3.checkbox("LOC")
            mechanical = fx4.checkbox("Mechanical/Structural")
            enviro = fx5.checkbox("Environmental")
            submit = st.form_submit_button("Add to database")

        if submit:
            new = {
                "date": pd.to_datetime(date).strftime("%Y-%m-%d"),
                "year": pd.to_datetime(date).year,
                "aircraft_type": aircraft_type,
                "category": category,
                "country": country,
                "event_name": event_name,
                "location": location,
                "manoeuvre": manoeuvre,
                "remarks": remarks,
                "contributing_factor": contributing_factor,
                "fit": int(fit), "mac": int(mac), "loc": int(loc),
                "mechanical": int(mechanical), "structural": 0,
                "enviro": int(enviro), "weather": 0, "bird_strike": 0,
                "pilot_killed": int(pilot_killed), "pilot_injured": int(pilot_injured),
                "crew_kill": int(crew_kill), "crew_inj": int(crew_inj),
                "pax_kill": int(pax_kill), "pax_inj": int(pax_inj),
            }
            man  = 1 if (new["loc"]==1 or new["fit"]==1) else 0
            mach = 1 if (new["mechanical"]==1 or new["structural"]==1) else 0
            med  = 1 if (new["enviro"]==1 or new["weather"]==1 or new["bird_strike"]==1) else 0
            mis  = 1 if (new["mac"]==1) else 0
            txt  = (new["remarks"] + " " + new["contributing_factor"]).lower()
            mgmt = 1 if any(k in txt for k in ["brief","oversight","management","organis","regulat","risk assess","fdd","air boss","safety officer","planning","arff","emergency response","procedure"]) else 0
            new.update({
                "man_factor":man,"machine_factor":mach,"medium_factor":med,"mission_factor":mis,"management_factor":mgmt
            })
            new["fatalities"] = new["pilot_killed"] + new["crew_kill"] + new["pax_kill"]
            new["casualties"] = new["fatalities"] + new["pilot_injured"] + new["crew_inj"] + new["pax_inj"]

            df2 = pd.read_json("data/airshow_accidents.json")
            df2 = pd.concat([df2, pd.DataFrame([new])], ignore_index=True)
            df2.to_json("data/airshow_accidents.json", orient="records")
            st.cache_data.clear()
            st.rerun()

