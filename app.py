
import streamlit as st
import pandas as pd, numpy as np, json
from pathlib import Path
import plotly.graph_objects as go

st.set_page_config(page_title="Airshow Safety & Excellence Database", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_json(Path("data/airshow_accidents.json"))
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    hist = json.loads(Path("data/historical_rates.json").read_text())
    return df, hist

df, hist = load_data()

# Sidebar filters
st.sidebar.header("Filters")
q = st.sidebar.text_input("Search (text)", placeholder="e.g. engine, Duxford, loop, MAC")
ymin, ymax = int(df['year'].min()), int(df['year'].max())
y1, y2 = st.sidebar.slider("Year range", min_value=ymin, max_value=ymax, value=(1908, ymax), step=1)

st.sidebar.subheader("Event type")
evt_acc = st.sidebar.checkbox("Accidents", True)
evt_inc = st.sidebar.checkbox("Incidents", True)

st.sidebar.subheader("5M factors (any match)")
m_man = st.sidebar.checkbox("Man", True)
m_machine = st.sidebar.checkbox("Machine", True)
m_medium = st.sidebar.checkbox("Medium", True)
m_mission = st.sidebar.checkbox("Mission", True)
m_mgmt = st.sidebar.checkbox("Management", True)

# Filter data
f = df[(df['year']>=y1) & (df['year']<=y2)].copy()
if evt_acc or evt_inc:
    cat = f['category'].astype(str).str.lower()
    keep = pd.Series(False, index=f.index)
    if evt_acc: keep |= cat.str.contains("accident", na=False)
    if evt_inc: keep |= cat.str.contains("incident", na=False)
    f = f[keep]

selected_5m = any([m_man, m_machine, m_medium, m_mission, m_mgmt])
if selected_5m:
    mask = False
    if m_man: mask |= (f['man_factor']==1)
    if m_machine: mask |= (f['machine_factor']==1)
    if m_medium: mask |= (f['medium_factor']==1)
    if m_mission: mask |= (f['mission_factor']==1)
    if m_mgmt: mask |= (f['management_factor']==1)
    f = f[mask]

if q:
    hay = (f['aircraft_type'].astype(str) + " " + f['category'].astype(str) + " " + 
           f['manoeuvre'].astype(str) + " " + f['event_name'].astype(str) + " " +
           f['location'].astype(str) + " " + f['country'].astype(str) + " " +
           f['remarks'].astype(str) + " " + f['contributing_factor'].astype(str) + " " +
           f['date'].astype(str))
    f = f[hay.str.lower().str.contains(q.lower())]

# KPI cards
col1, col2, col3 = st.columns(3)
col1.metric("Accidents/Incidents", f.shape[0])
col2.metric("Fatalities", int(f['fatalities'].sum()))
col3.metric("Casualties", int(f['casualties'].sum()))

# Charts
st.subheader("Accidents per year (filtered)")
by_year = f.dropna(subset=['year']).groupby('year').size().reset_index(name='count')
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=by_year['year'], y=by_year['count'], mode='lines+markers', name='Accidents (filtered)'))
fig1.update_layout(height=360, margin=dict(l=10,r=10,t=20,b=10))
st.plotly_chart(fig1, use_container_width=True)

st.subheader("Rates (2010–2024)")
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=hist['years'], y=hist['BAAR'], mode='lines+markers', name='BAAR (per 10k)'))
fig2.add_trace(go.Scatter(x=hist['years'], y=hist['AFR'], mode='lines+markers', name='AFR (per 10k)'))
fig2.add_trace(go.Scatter(x=hist['years'], y=hist['ACR'], mode='lines+markers', name='ACR (per 10k)'))
fig2.add_trace(go.Scatter(x=hist['years'], y=hist['AER'], mode='lines+markers', name='AER (%)', yaxis='y2'))
fig2.update_layout(
    yaxis=dict(title="per 10k events"),
    yaxis2=dict(title="AER (%)", overlaying='y', side='right', range=[99,100], tickformat=".3f"),
    height=360, margin=dict(l=10,r=10,t=20,b=10)
)
st.plotly_chart(fig2, use_container_width=True)

# 5M pie
st.subheader("5M distribution (filtered)")
five_counts = {
    "Man": int((f['man_factor']==1).sum()),
    "Machine": int((f['machine_factor']==1).sum()),
    "Medium": int((f['medium_factor']==1).sum()),
    "Mission": int((f['mission_factor']==1).sum()),
    "Management": int((f['management_factor']==1).sum())
}
fig3 = go.Figure(data=[go.Pie(labels=list(five_counts.keys()), values=list(five_counts.values()))])
fig3.update_traces(textinfo='percent+label', hovertemplate="%{label}: %{value}")
fig3.update_layout(height=360, margin=dict(l=10,r=10,t=20,b=10), showlegend=True)
st.plotly_chart(fig3, use_container_width=True)

# Manoeuvre pie
st.subheader("Aerobatic manoeuvres implicated (filtered)")
keys = [
    ('cuban 8','Cuban 8'), ('cuban eight','Cuban 8'),
    ('loop','Loop'), ('immelman','Immelman'), ('immelmann','Immelman'),
    ('split s','Split-S'),
    ('barrel roll','Barrel roll'), ('aileron roll','Roll'), ('roll','Roll'),
    ('spin','Spin'),
    ('hammerhead','Hammerhead'), ('stall turn','Hammerhead'),
    ('tail slide','Tailslide'), ('tailslide','Tailslide'),
    ('snap roll','Snap roll'),
    ('lomcevak','Lomcevak')
]
text = (f['manoeuvre'].astype(str) + " " + f['remarks'].astype(str) + " " + f['contributing_factor'].astype(str)).str.lower()
counts = {}
for pat, label in keys:
    counts[label] = counts.get(label, 0) + text.str.contains(pat).sum()
items = sorted([(k,v) for k,v in counts.items() if v>0], key=lambda x: x[1], reverse=True)
other = sum(v for _,v in items[10:])
items = items[:10]
if other>0: items.append(("Other", other))
if items:
    fig4 = go.Figure(data=[go.Pie(labels=[i[0] for i in items], values=[i[1] for i in items])])
    fig4.update_traces(textinfo='percent+label', hovertemplate="%{label}: %{value}")
    fig4.update_layout(height=360, margin=dict(l=10,r=10,t=20,b=10), showlegend=True)
    st.plotly_chart(fig4, use_container_width=True)
else:
    st.info("No manoeuvre keywords detected in the current filter.")

# Data table
st.subheader("Barker Airshow Incident & Accident Database")
show_cols = ['date','aircraft_type','category','country','manoeuvre','fit','mac','loc','mechanical','enviro','fatalities','casualties','event_name','location','contributing_factor','man_factor','machine_factor','medium_factor','mission_factor','management_factor','remarks']
f_out = f[show_cols].sort_values('date', ascending=False).reset_index(drop=True)
st.dataframe(f_out, use_container_width=True, hide_index=True)

# Download filtered CSV
csv = f_out.to_csv(index=False).encode('utf-8')
st.download_button("Download filtered CSV", data=csv, file_name="airshow_filtered.csv", mime="text/csv")

# ---------------- Admin ----------------
with st.expander("Admin: add incident/accident", expanded=False):
    pwd_ok = False
    if "ADMIN_PASSWORD" in st.secrets:
        pwd = st.text_input("Admin password", type="password")
        pwd_ok = (pwd == st.secrets["ADMIN_PASSWORD"])
    else:
        st.info("Tip: set `ADMIN_PASSWORD` in Streamlit secrets to protect this form. (Currently not set.)")
        pwd_ok = True

    if pwd_ok:
        with st.form("add_record"):
            st.write("**Minimal required fields**")
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

            st.write("**Casualties** (enter numbers)")
            colf1, colf2, colf3 = st.columns(3)
            pilot_killed = colf1.number_input("Pilot killed", min_value=0, step=1)
            pilot_injured = colf1.number_input("Pilot injured", min_value=0, step=1)
            crew_kill = colf2.number_input("Crew killed", min_value=0, step=1)
            crew_inj = colf2.number_input("Crew injured", min_value=0, step=1)
            pax_kill = colf3.number_input("Pax killed", min_value=0, step=1)
            pax_inj = colf3.number_input("Pax injured", min_value=0, step=1)

            colx1, colx2, colx3, colx4, colx5 = st.columns(5)
            fit = colx1.checkbox("FIT")
            mac = colx2.checkbox("MAC")
            loc = colx3.checkbox("LOC")
            mechanical = colx4.checkbox("Mechanical/Structural")
            enviro = colx5.checkbox("Environmental (weather/birds/obstacles)")

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
            def derive_5m(d):
                man = any([d.get('loc',0)==1, d.get('fit',0)==1])
                machine = any([d.get('mechanical',0)==1, d.get('structural',0)==1])
                medium = any([d.get('enviro',0)==1 or d.get('weather',0)==1 or d.get('bird_strike',0)==1])
                mission = any([d.get('mac',0)==1])
                mgmt = any([s in (d.get('contributing_factor','') or '').lower() for s in ['brief','oversight','management','organis','regulat','risk assess','fdd','air boss','safety officer','planning','arff','emergency response','procedure']])
                return int(man), int(machine), int(medium), int(mission), int(mgmt)
            man,machine,medium,mission,management = derive_5m(new)
            new.update({"man_factor":man,"machine_factor":machine,"medium_factor":medium,"mission_factor":mission,"management_factor":management})
            new['fatalities'] = new['pilot_killed'] + new['crew_kill'] + new['pax_kill']
            new['casualties'] = new['fatalities'] + new['pilot_injured'] + new['crew_inj'] + new['pax_inj']

            df2 = pd.read_json("data/airshow_accidents.json")
            df2 = pd.concat([df2, pd.DataFrame([new])], ignore_index=True)
            df2.to_json("data/airshow_accidents.json", orient="records")
            df2.to_csv("data/airshow_accidents_1908-2025.csv", index=False)
            st.success("Added locally.")

            repo = st.secrets.get("GITHUB_REPO", None)
            branch = st.secrets.get("GITHUB_BRANCH", "main")
            token = st.secrets.get("GITHUB_TOKEN", None)
            if repo and token:
                try:
                    import requests, base64
                    api = f"https://api.github.com/repos/{repo}/contents/data/airshow_accidents.json"
                    r = requests.get(api, headers={"Authorization": f"token {token}", "Accept":"application/vnd.github+json"}, params={"ref":branch})
                    sha = r.json().get("sha")
                    content_b64 = base64.b64encode(open('data/airshow_accidents.json','rb').read()).decode('utf-8')
                    msg = f"Add record via Streamlit admin: {new['date']} {new['aircraft_type']}"
                    put = requests.put(api, headers={"Authorization": f"token {token}", "Accept":"application/vnd.github+json"},
                                       json={"message": msg, "content": content_b64, "branch": branch, "sha": sha})
                    if put.status_code in (200,201):
                        st.success("Committed to GitHub successfully.")
                    else:
                        st.warning(f"GitHub commit failed: {put.status_code} {put.text[:200]}")
                except Exception as e:
                    st.warning(f"GitHub commit error: {e}")
            else:
                st.info("To persist across deployments: set secrets `GITHUB_REPO`, `GITHUB_BRANCH`, `GITHUB_TOKEN`.")
