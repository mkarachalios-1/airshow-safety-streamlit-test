#!/usr/bin/env python3
import re, json
from pathlib import Path
import pandas as pd
import numpy as np

# -------- settings --------
MASTER_XLSX = Path("data/Airshow_Accidents_Cleaned_2025.xlsx")  # your file; keep this name
OUT_JSON    = Path("data/airshow_accidents.json")
OUT_XLSX    = Path("data/Airshow_Accidents_CLEAN.xlsx")
OUT_REPORT  = Path("data/cleaning_report.csv")

# -------- helpers --------
def norm_spaces_ser(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(r"\s+", " ", regex=True).str.strip()

def to_title_keep_caps(s: str) -> str:
    if not isinstance(s, str) or not s:
        return s
    parts = re.split(r"(\W+)", s)
    out = []
    for p in parts:
        if re.fullmatch(r"[A-Z0-9\-_/]+", p):
            out.append(p)           # keep acronyms/numbers/dashes
        else:
            out.append(p.title())
    return "".join(out)

def canon_aircraft(v: str) -> str:
    if not isinstance(v, str): v = str(v)
    s = v.strip()
    s = re.sub(r"[–—‑]", "-", s)          # normalize dash variants
    s = re.sub(r"\s+", " ", s)            # collapse spaces
    s = s.replace(" / ", "/")
    s_l = s.lower()
    patterns = [
        (r"^(mig)\s*-?\s*(\d+)(\w*)", lambda m: f"MiG-{m.group(2)}{m.group(3).upper()}"),
        (r"^(su)\s*-?\s*(\d+)(\w*)",  lambda m: f"Su-{m.group(2)}{m.group(3).upper()}"),
        (r"^(yak)\s*-?\s*(\d+)(\w*)", lambda m: f"Yak-{m.group(2)}{m.group(3).upper()}"),
        (r"^(an)\s*-?\s*(\d+)(\w*)",  lambda m: f"An-{m.group(2)}{m.group(3).upper()}"),
        (r"^(tu)\s*-?\s*(\d+)(\w*)",  lambda m: f"Tu-{m.group(2)}{m.group(3).upper()}"),
        (r"^(il)\s*-?\s*(\d+)(\w*)",  lambda m: f"Il-{m.group(2)}{m.group(3).upper()}"),
        (r"^(ka)\s*-?\s*(\d+)(\w*)",  lambda m: f"Ka-{m.group(2)}{m.group(3).upper()}"),
        (r"^(mi)\s*-?\s*(\d+)(\w*)",  lambda m: f"Mi-{m.group(2)}{m.group(3).upper()}"),
        (r"^([afbcptu]|kc|uh|ah|ch|sh|kh|mh|oh|th)\s*-?\s*(\d+)(\w*)",
         lambda m: f"{m.group(1).upper()}-{m.group(2)}{m.group(3).upper()}"),
        (r"^(l)\s*-?\s*(\d+)(\w*)",   lambda m: f"L-{m.group(2)}{m.group(3).upper()}"),
    ]
    for rgx, fmt in patterns:
        m = re.match(rgx, s_l)
        if m:
            s = fmt(m)
            break
    s = re.sub(r"\s*-\s*", "-", s)        # tighten hyphens
    s = re.sub(r"\s*/\s*", "/", s)        # tighten slashes
    tokens = re.split(r"(\s+)", s)
    out = []
    for t in tokens:
        if re.fullmatch(r"[A-Z]+-\d+\w*", t) or re.fullmatch(r"[A-Z]{2,}.*", t):
            out.append(t)                 # keep all-caps model codes
        else:
            out.append(to_title_keep_caps(t))
    s = "".join(out).strip()
    return s.replace("MIG-", "MiG-").replace("SU-", "Su-")

COUNTRY_MAP = {
    # Americas
    "usa":"United States","u.s.a.":"United States","us":"United States","u.s.":"United States",
    "united states of america":"United States","trinidad & tobago":"Trinidad and Tobago",
    # Europe
    "uk":"United Kingdom","u.k.":"United Kingdom","england":"United Kingdom","great britain":"United Kingdom",
    "britain":"United Kingdom","czech republic":"Czechia","russian federation":"Russia",
    "republic of ireland":"Ireland","bosnia and herzegovina":"Bosnia and Herzegovina",
    "north macedonia":"North Macedonia","moldova, republic of":"Moldova","serbia and montenegro":"Serbia",
    "macedonia":"North Macedonia","vatican":"Vatican City","holy see":"Vatican City",
    "cyprus":"Cyprus",  # explicit to avoid odd transforms
    # Middle East / Asia
    "uae":"United Arab Emirates","u.a.e.":"United Arab Emirates","turkiye":"Turkey","türkiye":"Turkey",
    "korea, south":"South Korea","south korea":"South Korea","korea, north":"North Korea",
    "iran, islamic republic of":"Iran","syrian arab republic":"Syria",
    "lao pdr":"Laos","lao people’s democratic republic":"Laos","viet nam":"Vietnam",
    "china, prc":"China","taiwan, province of china":"Taiwan",
    # Africa
    "côte d’ivoire":"Côte d’Ivoire","cote d’ivoire":"Côte d’Ivoire","cote d'ivoire":"Côte d’Ivoire",
    "drc":"DR Congo","democratic republic of the congo":"DR Congo",
    "congo, democratic republic of the":"DR Congo","swaziland":"Eswatini",
    # Oceania
    "new zealand":"New Zealand"
}
def canon_country(v: str) -> str:
    if not isinstance(v, str): v = str(v)
    s = v.strip()
    s_l = s.lower()
    if s_l in COUNTRY_MAP:
        return COUNTRY_MAP[s_l]
    special = {"and","of","the","&","de","la","el","al","da","do","di","der","des"}
    words = re.split(r"\s+", s_l)
    words = [w.capitalize() if w not in special else w for w in words]
    s2 = " ".join(words)
    return s2.replace("U.s.", "United States").replace("U.k.", "United Kingdom").strip()

def norm_manoeuvre(v: str) -> str:
    if not isinstance(v, str): v = str(v)
    s = v.strip().lower()
    s = re.sub(r"[–—‑]", "-", s)
    s = re.sub(r"\s+", " ", s)
    mapping = [
        (r"split[\s\-]?s\b", "Split-S"),
        (r"immelman+n?\b", "Immelman"),
        (r"\bcuban\s*8\b|\bcuban\s*eight\b", "Cuban 8"),
        (r"\bbarrel\s*roll\b", "Barrel roll"),
        (r"\baileron\s*roll\b|\bslow\s*roll\b|\broll\b", "Roll"),
        (r"\bhammerhead\b|\bstall\s*turn\b", "Hammerhead"),
        (r"\btail\s*slide\b|\btailslide\b", "Tailslide"),
        (r"\bsnap\s*roll\b", "Snap roll"),
        (r"\blomcevak\b", "Lomcevak"),
        (r"\bloop\b", "Loop"),
    ]
    for rgx, label in mapping:
        if re.search(rgx, s):
            return label
    return to_title_keep_caps(v.strip())

def derive_contrib_category(row):
    labs = []
    for flag,label in [("fit","FIT"),("mac","MAC"),("loc","LOC"),
                       ("mechanical","Mechanical"),("structural","Structural"),
                       ("enviro","Environmental"),("weather","Weather"),("bird_strike","Bird strike")]:
        if row.get(flag,0)==1: labs.append(label)
    for flag,label in [("man_factor","Man"),("machine_factor","Machine"),
                       ("medium_factor","Medium"),("mission_factor","Mission"),("management_factor","Management")]:
        if row.get(flag,0)==1: labs.append(label)
    txt = f"{row.get('remarks','')} {row.get('contributing_factor','')}".lower()
    if not labs:
        if "mid-air" in txt or "midair" in txt or "collision" in txt: labs.append("MAC")
        if "loss of control" in txt or " loc " in txt: labs.append("LOC")
        if "engine" in txt or "mechanical" in txt: labs.append("Mechanical")
        if "weather" in txt or "wind" in txt: labs.append("Weather")
    return "; ".join(sorted(set(labs))) if labs else ""

# -------- load --------
if not MASTER_XLSX.exists():
    raise SystemExit(f"Master Excel not found at {MASTER_XLSX}")

df = pd.read_excel(MASTER_XLSX)

# Ensure columns exist (we DO NOT drop rows)
for c in ["aircraft_type","country","manoeuvre","remarks","contributing_factor","event_name","location","date","year"]:
    if c not in df.columns: df[c] = ""

# Normalize fields
df["aircraft_type"] = norm_spaces_ser(df["aircraft_type"]).apply(canon_aircraft)
df["country"]       = norm_spaces_ser(df["country"]).apply(canon_country)
df["manoeuvre"]     = norm_spaces_ser(df["manoeuvre"]).apply(norm_manoeuvre)
df["remarks"]       = norm_spaces_ser(df["remarks"])
df["contributing_factor"] = norm_spaces_ser(df["contributing_factor"])
df["event_name"]    = norm_spaces_ser(df["event_name"]).apply(to_title_keep_caps)
df["location"]      = norm_spaces_ser(df["location"]).apply(to_title_keep_caps)

# Dates + Year (keep all rows)
dt = pd.to_datetime(df["date"], errors="coerce")
df["date"] = np.where(dt.notna(), dt.dt.strftime("%Y-%m-%d"), df["date"].astype(str))
df["year"] = pd.to_numeric(df["year"], errors="coerce")
df["year"] = df["year"].fillna(pd.to_datetime(df["date"], errors="coerce").dt.year)

# Casualty totals
for c in ["pilot_killed","crew_kill","pax_kill","pilot_injured","crew_inj","pax_inj"]:
    if c not in df.columns: df[c] = 0
    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
df["fatalities"] = (df["pilot_killed"] + df["crew_kill"] + df["pax_kill"]).astype(int)
df["casualties"] = (df["fatalities"] + df["pilot_injured"] + df["crew_inj"] + df["pax_inj"]).astype(int)

# 5M flags -> ints
for c in ["fit","mac","loc","mechanical","structural","enviro","weather","bird_strike",
          "man_factor","machine_factor","medium_factor","mission_factor","management_factor"]:
    if c not in df.columns: df[c] = 0
    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

# Derived standardized contributing category
df["contrib_category_std"] = df.apply(derive_contrib_category, axis=1)

# Column order
col_order = [
    "date","year","aircraft_type","category","manoeuvre",
    "fit","mac","loc","mechanical","structural","enviro","weather","bird_strike",
    "fatalities","casualties","pilot_killed","pilot_injured","crew_kill","crew_inj","pax_kill","pax_inj",
    "event_name","location","country","remarks","contributing_factor",
    "man_factor","machine_factor","medium_factor","mission_factor","management_factor","contrib_category_std"
]
rest = [c for c in df.columns if c not in col_order]
final_cols = [c for c in col_order if c in df.columns] + rest

# Save
df[final_cols].to_excel(OUT_XLSX, index=False)
OUT_JSON.write_text(df[final_cols].to_json(orient="records"), encoding="utf-8")

# Small sample report (present for audit; full mapping would be very large)
rep = pd.DataFrame({
    "aircraft_type_after": df["aircraft_type"],
    "country_after": df["country"],
    "manoeuvre_after": df["manoeuvre"]
}).head(1)
rep.to_csv(OUT_REPORT, index=False)

# Summary to stdout (visible in GitHub Actions logs)
print(json.dumps({
    "rows_output": int(len(df)),
    "aircraft_type_missing": int((df['aircraft_type'].astype(str).str.strip()=='').sum()),
    "unique_aircraft_types": int(df["aircraft_type"].nunique()),
    "unique_countries": int(df["country"].nunique()),
}, indent=2))
