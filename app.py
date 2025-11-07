import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Airshow Safety & Excellence Database", layout="wide")

# Hide chrome when embedded
params = st.experimental_get_query_params()
if "embed" in params:
    st.markdown("""<style>
    header {visibility: hidden;}
    footer {visibility: hidden;}
    div.block-container {padding-top: 0.6rem;}
</style>""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    df = pd.read_csv("data/airshow_accidents_1908-2025.csv", low_memory=False)
    # Try to ensure required cols exist for the demo; real CSV already has them
    for c in ['date','year']:
        if c not in df.columns:
            if 'Date' in df.columns: df['date']=df['Date']
            df['year'] = pd.to_datetime(df.get('date'), errors='coerce').dt.year
    rates = pd.read_csv("data/historical_rates_2010_2024.csv")
    return df, rates

df, rates = load_data()

st.title("Airshow Safety & Excellence Database")
st.caption("5M-aligned repository of airshow accidents (1908–2025). Use search and filters below. Charts update as you filter.")
st.subheader("Barker Airshow Incident & Accident Database")

# Fallback defaults to keep app robust if columns are missing
for col in ['aircraft_type','category','country','manoeuvre','fit','mac','loc','mechanical','enviro',
            'fatalities','casualties','event_name','location','contributing_factor',
            'man_factor','machine_factor','medium_factor','mission_factor','management_factor','remarks']:
    if col not in df.columns:
        df[col] = '' if df.get(col, pd.Series(dtype=object)).dtype=='O' else 0

df['date'] = pd.to_datetime(df['date'], errors='coerce')
df['year'] = pd.to_numeric(df['year'], errors='coerce')

with st.expander("Filters", expanded=True):
    c1, c2, c3, c4 = st.columns([2,2,3,5])
    with c1:
        year_min, year_max = int(df['year'].min()), int(df['year'].max())
        yr = st.slider("Year range", min_value=year_min, max_value=year_max, value=(1908, 2025))
    with c2:
        event_types = st.multiselect("Event type", ["Accidents","Incidents"], default=["Accidents","Incidents"])
    with c3:
        st.write("5M Filters (match any)")
        m1 = st.checkbox("Man", value=True)
        m2 = st.checkbox("Machine", value=True)
        m3 = st.checkbox("Medium", value=True)
        m4 = st.checkbox("Mission", value=True)
        m5 = st.checkbox("Management", value=True)
    with c4:
        q = st.text_input("Search", "", placeholder="e.g., MAC 2019, loop, Duxford")

mask = df['year'].between(yr[0], yr[1])
if event_types:
    cat = df['category'].astype(str).str.lower()
    wants_acc = "Accidents" in event_types
    wants_inc = "Incidents" in event_types
    mask &= ((wants_acc & cat.str.contains("accident")) | (wants_inc & cat.str.contains("incident")))

selected_5m = [m1,m2,m3,m4,m5]
if any(selected_5m):
    m_mask = False
    if m1: m_mask |= (df['man_factor']==1)
    if m2: m_mask |= (df['machine_factor']==1)
    if m3: m_mask |= (df['medium_factor']==1)
    if m4: m_mask |= (df['mission_factor']==1)
    if m5: m_mask |= (df['management_factor']==1)
    mask &= m_mask

if q:
    hay = (df['aircraft_type'].astype(str)+' '+df['category'].astype(str)+' '+df['manoeuvre'].astype(str)+' '+
           df['event_name'].astype(str)+' '+df['location'].astype(str)+' '+df['country'].astype(str)+' '+
           df['remarks'].astype(str)+' '+df['contributing_factor'].astype(str)).str.lower()
    mask &= hay.str.contains(q.lower())

filtered = df[mask].sort_values('date', ascending=False).copy()

k1,k2,k3 = st.columns(3)
k1.metric("Accidents / Incidents (filtered)", f"{len(filtered):,}")
k2.metric("Fatalities (filtered)", f"{int(pd.to_numeric(filtered['fatalities'], errors='coerce').fillna(0).sum()):,}")
k3.metric("Casualties (filtered)", f"{int(pd.to_numeric(filtered['casualties'], errors='coerce').fillna(0).sum()):,}")

c1,c2 = st.columns(2)
with c1:
    by_year = filtered.groupby('year').size().reset_index(name='count')
    fig1 = px.line(by_year, x='year', y='count', title='Accidents / Incidents per Year (filtered)')
    fig1.update_layout(height=380, margin=dict(l=20,r=20,t=60,b=20))
    st.plotly_chart(fig1, use_container_width=True, theme=None)

with c2:
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=rates['year'], y=rates['BAAR'], name='BAAR (per 10k)', yaxis='y'))
    fig2.add_trace(go.Scatter(x=rates['year'], y=rates['AFR'], name='AFR (per 10k)', yaxis='y'))
    fig2.add_trace(go.Scatter(x=rates['year'], y=rates['ACR'], name='ACR (per 10k)', yaxis='y'))
    fig2.add_trace(go.Scatter(x=rates['year'], y=rates['AER'], name='AER (%)', yaxis='y2'))
    fig2.update_layout(
        title='BAAR / AFR / ACR / AER (2010–2024)',
        xaxis=dict(title='Year'),
        yaxis=dict(title='per 10,000 events'),
        yaxis2=dict(title='AER (%)', overlaying='y', side='right', range=[99,100], tickformat='.3f'),
        height=380, margin=dict(l=20,r=20,t=60,b=20),
        legend=dict(orientation='h', y=1.15, x=0)
    )
    st.plotly_chart(fig2, use_container_width=True, theme=None)

c3,c4 = st.columns(2)
with c3:
    five = {
        'Man': int((filtered['man_factor']==1).sum()),
        'Machine': int((filtered['machine_factor']==1).sum()),
        'Medium': int((filtered['medium_factor']==1).sum()),
        'Mission': int((filtered['mission_factor']==1).sum()),
        'Management': int((filtered['management_factor']==1).sum())
    }
    five_df = pd.DataFrame({'Factor': list(five.keys()), 'Count': list(five.values())})
    fig3 = px.pie(five_df, names='Factor', values='Count', title='5M Distribution (filtered)')
    fig3.update_layout(height=360, margin=dict(l=0,r=0,t=50,b=0), showlegend=True)
    fig3.update_traces(textinfo='percent+label')
    st.plotly_chart(fig3, use_container_width=True, theme=None)

with c4:
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
    txt = (filtered['manoeuvre'].astype(str)+' '+filtered['remarks'].astype(str)+' '+filtered['contributing_factor'].astype(str)).str.lower()
    counts = {}
    for pat,label in keys:
        counts[label] = int(txt.str.contains(pat).sum()) + counts.get(label,0)
    items = sorted([(k,v) for k,v in counts.items() if v>0], key=lambda x:x[1], reverse=True)
    if not items:
        st.info('No aerobatic manoeuvres detected in the current filter.')
    else:
        top = items[:10]
        other = sum(v for _,v in items[10:])
        if other>0: top.append(('Other', other))
        mdf = pd.DataFrame(top, columns=['Manoeuvre','Count'])
        fig4 = px.pie(mdf, names='Manoeuvre', values='Count', title='Aerobatic Manoeuvres implicated (filtered)')
        fig4.update_layout(height=360, margin=dict(l=0,r=0,t=50,b=0))
        fig4.update_traces(textinfo='percent+label')
        st.plotly_chart(fig4, use_container_width=True, theme=None)

st.markdown('### Results')
display_cols = ['date','aircraft_type','category','country','manoeuvre','fit','mac','loc','mechanical','enviro',
                'fatalities','casualties','event_name','location','contributing_factor',
                'man_factor','machine_factor','medium_factor','mission_factor','management_factor','remarks']
present = [c for c in display_cols if c in filtered.columns]
st.dataframe(filtered[present], use_container_width=True, hide_index=True)

csv = filtered[present].to_csv(index=False).encode('utf-8')
st.download_button('Download filtered CSV', data=csv, file_name='airshow_filtered.csv', mime='text/csv')


# ---------------- Admin ----------------
with st.expander("Admin: add incident/accident", expanded=False):
    pwd_ok = False
    if "ADMIN_PASSWORD" in st.secrets:
        pwd = st.text_input("Admin password", type="password")
        pwd_ok = (pwd == st.secrets["ADMIN_PASSWORD"])
    else:
        st.info("Tip: set `ADMIN_PASSWORD` in Streamlit secrets to protect this form. (Currently not set.)")
        pwd_ok = True  # allow locally

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
            import pandas as pd, numpy as np, datetime, requests, base64
            # Build new row
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
            # Derive 5M
            def derive_5m(d):
                man = any([d.get('loc',0)==1, d.get('fit',0)==1])
                machine = any([d.get('mechanical',0)==1, d.get('structural',0)==1])
                medium = any([d.get('enviro',0)==1 or d.get('weather',0)==1 or d.get('bird_strike',0)==1])
                mission = any([d.get('mac',0)==1])
                mgmt = any([s in (d.get('contributing_factor','') or '').lower() for s in ['brief','oversight','management','organis','regulat','risk assess','fdd','air boss','safety officer','planning','arff','emergency response','procedure']])
                return int(man), int(machine), int(medium), int(mission), int(mgmt)
            man,machine,medium,mission,management = derive_5m(new)
            new.update({"man_factor":man,"machine_factor":machine,"medium_factor":medium,"mission_factor":mission,"management_factor":management})

            # Compute totals
            new['fatalities'] = new['pilot_killed'] + new['crew_kill'] + new['pax_kill']
            new['casualties'] = new['fatalities'] + new['pilot_injured'] + new['crew_inj'] + new['pax_inj']

            # 1) Append locally so charts refresh immediately
            df2 = pd.read_json("data/airshow_accidents.json")
            df2 = pd.concat([df2, pd.DataFrame([new])], ignore_index=True)
            df2.to_json("data/airshow_accidents.json", orient="records")
            df2.to_csv("data/airshow_accidents_1908-2025.csv", index=False)
            st.success("Added locally.")

            # 2) Optional: commit to GitHub if secrets provided
            repo = st.secrets.get("GITHUB_REPO", None)        # e.g. 'mkarachalios-1/airshow-safety-database'
            branch = st.secrets.get("GITHUB_BRANCH", "main")  # branch name
            token = st.secrets.get("GITHUB_TOKEN", None)      # a PAT with 'repo' scope
            if repo and token:
                try:
                    import requests, base64
                    # Fetch current file sha
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
