"""
app.py - BikeFlow AI: interactive dashboard for "Beyond RMSE: A Cost-Sensitivity Analysis
of Bike-Share Demand Forecasting" (AST02.10, AIT).

All numbers come from model_bundle.joblib, built by build_model.py with the same protocol as the
final report. Nothing is retrained here: the app re-scores the stored held-out test forecasts live
for any cost ratio r, and adds the hour-of-day buffer (the r/(1+r) quantile of validation errors).

Run locally:  python app.py          (http://127.0.0.1:8000)
Production :  gunicorn --workers 2 --bind 127.0.0.1:8000 app:app
"""
import json
import math
import os

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template_string, request

HERE = os.path.dirname(os.path.abspath(__file__))
B = joblib.load(os.path.join(HERE, "model_bundle.joblib"))
DC, SEOUL = B["dc"], B["seoul"]
app = Flask(__name__)

MONTH_SEASON = {1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2, 7: 3, 8: 3, 9: 3, 10: 4, 11: 4, 12: 4}
PEAK_HOURS = [7, 8, 17, 18]
ARMB_RATIOS = sorted(DC["armb_test"].keys())

# Table 4.5 of the report (rolling-origin evaluation at r = 5); copied from the notebook output.
ROLLING_R5 = [
    {"window": "Mar-Apr 2012", "standard": 469.8, "refit": 472.0, "rerank": 469.8, "buffer": 318.6, "vs_std": -32, "vs_refit": -32},
    {"window": "May-Jun 2012", "standard": 290.0, "refit": 109.6, "rerank": 195.2, "buffer": 118.8, "vs_std": -59, "vs_refit": 8},
    {"window": "Jul-Aug 2012", "standard": 212.3, "refit": 185.6, "rerank": 212.3, "buffer": 122.3, "vs_std": -42, "vs_refit": -34},
    {"window": "Sep-Oct 2012", "standard": 192.2, "refit": 175.4, "rerank": 187.0, "buffer": 89.5, "vs_std": -53, "vs_refit": -49},
    {"window": "Nov-Dec 2012", "standard": 146.8, "refit": 129.4, "rerank": 146.8, "buffer": 88.5, "vs_std": -40, "vs_refit": -32},
]


# ---------------------------------------------------------------- helpers
def f64(a):
    return np.asarray(a, dtype=float)


def avg_cost(y, p, r):
    e = y - p
    return float(r * np.maximum(e, 0).mean() + np.maximum(-e, 0).mean())


def hourly_buffer(err, hr_val, r):
    """Buffer for each hour h = the r/(1+r) quantile of validation errors in hour h (report Sec. 4.6)."""
    a = r / (1 + r)
    return np.array([np.quantile(err[hr_val == h], a) for h in range(24)])


def crossover(y, p_xgb, p_naive):
    """Smallest r at which Seasonal Naive becomes cheaper than XGBoost (None = never)."""
    e1, e2 = y - p_xgb, y - p_naive
    u1, o1 = np.maximum(e1, 0).mean(), np.maximum(-e1, 0).mean()
    u2, o2 = np.maximum(e2, 0).mean(), np.maximum(-e2, 0).mean()
    if u1 <= u2:
        return None
    return float(max((o2 - o1) / (u1 - u2), 0.0))


def num(v, lo, hi, default):
    try:
        v = float(v)
        if math.isnan(v) or math.isinf(v):
            return default
        return min(max(v, lo), hi)
    except (TypeError, ValueError):
        return default


def r1(x):
    return None if x is None else round(float(x), 1)


# ---------------------------------------------------------------- cached arrays
CITY = {}
for key, src in (("dc", DC), ("seoul", SEOUL)):
    ts = pd.Series(src["ts_test"])
    CITY[key] = {
        "y": f64(src["y_test"]),
        "P": {k: f64(v) for k, v in src["preds_test"].items()},
        "err": {k: f64(v) for k, v in src["val_err_all"].items()},
        "hv": np.asarray(src["hr_val"]), "ht": np.asarray(src["hr_test"]),
        "date": ts.str[:10].values,
        "dates": sorted(ts.str[:10].unique().tolist()),
    }
CITY["dc"]["peak"] = (np.asarray(DC["workingday_test"]) == 1) & np.isin(CITY["dc"]["ht"], PEAK_HOURS)
CITY["dc"]["armb"] = {r: {k: f64(v) for k, v in d.items()} for r, d in DC["armb_test"].items()}

R_GRID = np.round(np.exp(np.linspace(np.log(0.25), np.log(10), 61)), 3)


def static_payload():
    out = {}
    for key, names in (("dc", ["Seasonal Naive", "Hour x Weekday Avg", "Linear Regression", "Random Forest", "XGBoost"]),
                       ("seoul", ["Seasonal Naive", "XGBoost"])):
        c = CITY[key]
        out[key] = {
            "r_grid": R_GRID.tolist(),
            "curves": {n: [round(avg_cost(c["y"], c["P"][n], r), 2) for r in R_GRID] for n in names},
            "rstar": crossover(c["y"], c["P"]["XGBoost"], c["P"]["Seasonal Naive"]),
            "dates": c["dates"],
            "rmse": {n: round(float(np.sqrt(np.mean((c["y"] - c["P"][n]) ** 2))), 1) for n in names},
            "test_period": (DC if key == "dc" else SEOUL)["test_period"],
        }
    imp = sorted(DC["importance"].items(), key=lambda kv: -kv[1])[:8]
    tot = sum(DC["importance"].values())
    out["importance"] = [[k, round(100 * v / tot, 1)] for k, v in imp]
    out["gbfs"] = B["gbfs"]
    out["rolling"] = ROLLING_R5
    out["armb_ratios"] = ARMB_RATIOS
    return out


STATIC = static_payload()


def whatif_features(hr, month, workingday, weathersit, temp_c, hum_pct, wind_kmh):
    """One row of the 13 Washington features for a hypothetical hour in 2012."""
    season = MONTH_SEASON[month]
    temp = temp_c / 41.0                           # UCI normalisation (max 41 C)
    a, b = DC["atemp_coef"]
    lag = DC["lag_med"].get((season, workingday, hr), 0.0)
    return {"season": season, "yr": 1, "mnth": month, "hr": hr, "holiday": 0,
            "weekday": 3 if workingday else 6, "workingday": workingday, "weathersit": weathersit,
            "temp": temp, "atemp": float(np.clip(a * temp + b, 0, 1)), "hum": hum_pct / 100.0,
            "windspeed": wind_kmh / 67.0, "lag_168h": float(lag)}


def evaluation(city, r, rule):
    """Re-score every plan on the whole held-out test block at cost ratio r."""
    c = CITY[city]; y = c["y"]; ht = c["ht"]
    use_buf = (r > 1) or not rule
    rows = []

    def add(name, p, group, note=""):
        row = {"name": name, "group": group, "note": note, "cost": round(avg_cost(y, p, r), 1),
               "cov": round(100 * float(np.mean(y <= p))), "rmse": None}
        if "peak" in c:
            m = c["peak"]
            row["peak_under"] = round(100 * float(np.mean(y[m] > p[m])))
        rows.append(row)

    def buffered(base_name):
        b = hourly_buffer(c["err"]["XGBoost" if base_name.startswith("XGBoost") else base_name], c["hv"], r)
        if not use_buf:
            b = np.zeros(24)
        return np.clip(c["P"][base_name] + b[ht], 0, None)

    if city == "dc":
        for n in ["Seasonal Naive", "Hour x Weekday Avg", "Linear Regression", "Random Forest"]:
            add(n, c["P"][n], "Arm A (re-scored)")
        add("XGBoost (RMSE winner)", c["P"]["XGBoost"], "Baseline")
        add("XGBoost refitted (train+val)", c["P"]["XGBoost refitted"], "Fair baseline")
        add("XGBoost + hourly buffer", buffered("XGBoost"), "Proposed")
        add("Refitted XGBoost + hourly buffer", buffered("XGBoost refitted"), "Recommended")
        rk = min(ARMB_RATIOS, key=lambda v: abs(v - r))
        if abs(rk - r) < 1e-9:
            add("Arm B (cost-trained XGBoost)", c["armb"][rk]["Arm B"], "Arm B")
            add("Arm B refitted (train+val)", c["armb"][rk]["Arm B refitted"], "Arm B")
        base = rows[4]["cost"]
    else:
        add("Seasonal Naive", c["P"]["Seasonal Naive"], "Arm A (re-scored)")
        add("Seasonal Naive + hourly buffer", buffered("Seasonal Naive"), "Buffer")
        add("XGBoost (RMSE winner)", c["P"]["XGBoost"], "Baseline")
        add("XGBoost + hourly buffer", buffered("XGBoost"), "Proposed")
        base = rows[2]["cost"]
    for row in rows:
        row["vs_base"] = round(100 * (row["cost"] / base - 1), 1) if base else None
    best = min(rows, key=lambda x: x["cost"])["name"]
    return rows, best


# ---------------------------------------------------------------- API
SEASON_MONTH = {1: 2, 2: 5, 3: 8, 4: 11}   # representative month for the season selector (what-if mode)


def table_by_name(rows):
    return {r["name"]: r for r in rows}


@app.route("/api/predict", methods=["POST"])
def api_predict():
    q = request.get_json(silent=True) or {}
    city = "seoul" if q.get("city") == "seoul" else "dc"
    c = CITY[city]
    r = num(q.get("r_ratio", q.get("r")), 0.25, 10.0, 5.0)
    rule = q.get("rule", True) not in (False, "false", 0, "0")
    base = "refit" if (q.get("base") == "refit" and city == "dc") else "train"
    mode = "replay" if (q.get("mode") == "replay" or city == "seoul") else "whatif"
    hr = int(num(q.get("hr"), 0, 23, 17))
    alpha = r / (1 + r)

    buf = hourly_buffer(c["err"]["XGBoost"], c["hv"], r)
    applied = (r > 1) or not rule
    buf_used = buf if applied else np.zeros(24)

    actual = [None] * 24
    date = None
    if mode == "replay":
        date = str(q.get("date") or "")
        if date not in c["dates"]:
            date = "2012-10-10" if city == "dc" else "2018-10-10"
        key = "XGBoost refitted" if base == "refit" else "XGBoost"
        base_pred = [None] * 24
        for i in np.where(c["date"] == date)[0]:
            h = int(c["ht"][i])
            base_pred[h] = float(c["P"][key][i]); actual[h] = float(c["y"][i])
    else:
        season = int(num(q.get("season"), 1, 4, 3))
        wd = 1 if str(q.get("workingday", 1)) in ("1", "True", "true") else 0
        ws = int(num(q.get("weathersit"), 1, 4, 1))
        rows = [whatif_features(h, SEASON_MONTH[season], wd, ws, num(q.get("temp_c"), 0, 41, 26),
                                num(q.get("hum_pct"), 0, 100, 50), num(q.get("wind_kmh"), 0, 67, 12)) for h in range(24)]
        model = DC["xgb_refit"] if base == "refit" else DC["xgb"]
        base_pred = np.clip(model.predict(pd.DataFrame(rows)[DC["feats"]]), 0, None).astype(float).tolist()

    plan = [None if b is None else max(0.0, b + float(buf_used[h])) for h, b in enumerate(base_pred)]
    day = {}
    have = [h for h in range(24) if base_pred[h] is not None and actual[h] is not None]
    if have:
        ya = np.array([actual[h] for h in have]); pb = np.array([base_pred[h] for h in have])
        pp = np.array([plan[h] for h in have])
        day = {"hours": len(have), "actual": round(float(ya.sum())),
               "cost_base": round(avg_cost(ya, pb, r), 1), "cost_plan": round(avg_cost(ya, pp, r), 1),
               "short_base": round(float(np.maximum(ya - pb, 0).sum())),
               "short_plan": round(float(np.maximum(ya - pp, 0).sum()))}

    rows, best = evaluation(city, r, rule)
    for row in rows:                      # RMSE for the plain forecasts
        n = row["name"].replace(" (RMSE winner)", "").replace(" (train+val)", "")
        key = {"XGBoost refitted": "XGBoost refitted"}.get(n, n)
        if key in c["P"]:
            row["rmse"] = round(float(np.sqrt(np.mean((c["y"] - c["P"][key]) ** 2))), 1)
    T = table_by_name(rows)
    xg = T["XGBoost (RMSE winner)"]
    prop = T["Refitted XGBoost + hourly buffer"] if base == "refit" else T["XGBoost + hourly buffer"]
    sel_b = base_pred[hr]
    return jsonify({
        # keys used by the original dashboard
        "hr": hr, "r": r, "alpha_pct": round(100 * alpha, 1),
        "base_pred": None if sel_b is None else round(sel_b), "buf_val": round(float(buf_used[hr])),
        "plan_pred": None if plan[hr] is None else round(plan[hr]),
        "plan_total": round(sum(v for v in plan if v is not None)),
        "base_preds": [None if v is None else round(v) for v in base_pred],
        "plan_preds": [None if v is None else round(v) for v in plan],
        "buffers": [round(float(v)) for v in buf_used],
        "cost_cut": round(100 * (1 - prop["cost"] / xg["cost"]), 1) if xg["cost"] else 0.0,
        "xgb_cost": xg["cost"], "buf_cost": prop["cost"],
        # new keys
        "city": city, "mode": mode, "date": date, "base": base, "rule_applied": applied,
        "buffers_raw": [round(float(v)) for v in buf],
        "actual": [None if v is None else round(v) for v in actual],
        "sel_actual": None if actual[hr] is None else round(actual[hr]),
        "plan_name": prop["name"], "day": day, "table": rows, "best": best,
    })


@app.route("/api/static")
def api_static():
    return jsonify(STATIC)


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "test_hours_dc": int(len(CITY["dc"]["y"])),
                    "test_hours_seoul": int(len(CITY["seoul"]["y"]))})


@app.route("/")
def index():
    return render_template_string(PAGE, static_json=json.dumps(STATIC))


PAGE = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>BikeFlow AI | Beyond RMSE Decision Dashboard</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500;600&family=Fira+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
<script src="https://cdn.tailwindcss.com"></script>
<script>if(window.tailwind)tailwind.config={theme:{extend:{fontFamily:{sans:['Fira Sans','system-ui','sans-serif'],mono:['Fira Code','ui-monospace','monospace']}}}}</script>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
:root{--bg:#0b1220;--card:#111a2b;--line:#1e293b;--line2:#273449;--txt:#e2e8f0;--mut:#94a3b8;--pri:#3b82f6;--plan:#10b981;--buf:#f59e0b;--act:#f43f5e}
body{background:var(--bg);color:var(--txt);font-family:'Fira Sans',system-ui,sans-serif}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px}
.num{font-family:'Fira Code',monospace;font-variant-numeric:tabular-nums}
.eyebrow{font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:var(--mut);font-weight:600}
.seg{display:inline-flex;background:#0b1220;border:1px solid var(--line2);border-radius:8px;padding:2px}
.seg button{font-size:12px;padding:6px 12px;border-radius:6px;color:#cbd5e1;white-space:nowrap;cursor:pointer;transition:background .2s,color .2s}
.seg button:hover{color:#fff}
.seg button.on{background:var(--pri);color:#fff;font-weight:600}
.chip{font-size:12px;padding:6px 12px;border-radius:999px;border:1px solid var(--line2);color:#cbd5e1;background:#0f172a;cursor:pointer;transition:border-color .2s,color .2s;white-space:nowrap}
.chip:hover{border-color:var(--pri);color:#fff}
.field{width:100%;background:#0b1220;border:1px solid var(--line2);border-radius:8px;padding:7px 10px;font-size:13px;color:var(--txt)}
input[type=range]{accent-color:var(--pri);width:100%;cursor:pointer}
input[type=date]{color-scheme:dark}
button:focus-visible,input:focus-visible,select:focus-visible{outline:2px solid var(--pri);outline-offset:2px}
.role{font-size:10.5px;padding:2px 8px;border-radius:999px;border:1px solid var(--line2);color:var(--mut);white-space:nowrap}
.role.base{border-color:#1d4ed8;color:#93c5fd}.role.prop{border-color:#047857;color:#6ee7b7}.role.rec{background:#064e3b;border-color:#059669;color:#a7f3d0}
tbody tr{border-top:1px solid var(--line)}
tbody tr:hover{background:#0f1a2e}
.secnum{width:22px;height:22px;border-radius:6px;background:#1e293b;color:#cbd5e1;font-size:12px;display:inline-flex;align-items:center;justify-content:center;font-weight:600}
@media (prefers-reduced-motion:reduce){*{transition:none!important;animation:none!important}}
</style>
</head><body class="min-h-screen">
<header class="border-b border-slate-800 bg-[#0b1220]/90 backdrop-blur sticky top-0 z-20">
<div class="max-w-7xl mx-auto px-6 py-3 flex flex-wrap items-center justify-between gap-3">
 <div class="flex items-center gap-3">
  <div class="w-9 h-9 rounded-lg bg-blue-600 flex items-center justify-center" aria-hidden="true">
   <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="5.5" cy="17.5" r="3.5"/><circle cx="18.5" cy="17.5" r="3.5"/><circle cx="15" cy="5" r="1"/><path d="M12 17.5V14l-3-3 4-3 2 3h2"/></svg></div>
  <div><h1 class="text-[15px] font-semibold text-white leading-tight">BikeFlow AI</h1>
   <p class="text-xs text-slate-400">Beyond RMSE: cost-aware hourly planning for bike-share operators</p></div></div>
 <div class="flex flex-wrap items-center gap-3">
  <div id="cityBtns" class="seg" role="group" aria-label="City"><button type="button" data-v="dc" class="on">Washington, D.C.</button><button type="button" data-v="seoul">Seoul</button></div>
  <span class="flex items-center gap-1.5 text-xs text-slate-400"><span class="w-2 h-2 rounded-full bg-emerald-500" aria-hidden="true"></span>Live on AWS EC2</span></div>
</div></header>

<main class="max-w-7xl mx-auto px-6 py-6 space-y-8">

<!-- key finding -->
<section class="card px-5 py-4 flex flex-wrap items-center justify-between gap-4" aria-label="Key finding">
 <div class="max-w-xl"><p class="eyebrow">Key finding</p>
  <p class="text-lg text-white font-medium mt-1">Keep the most accurate model, and plan higher, hour by hour.</p>
  <p class="text-sm text-slate-400 mt-1">Switching models by cost is unstable (r* moves between 0.5 and 4). An hour-of-day buffer, the r/(1+r) quantile of recent errors, cuts cost reliably with no retraining.</p></div>
 <dl class="grid grid-cols-3 gap-6 text-center">
  <div><dt class="text-xs text-slate-400">Cost at r = 5</dt><dd class="num text-2xl font-semibold text-emerald-400">−48.7%</dd><dd class="num text-xs text-slate-400">182.0 → 93.3</dd></div>
  <div><dt class="text-xs text-slate-400">Rolling windows</dt><dd class="num text-2xl font-semibold text-emerald-400">−32–59%</dd><dd class="text-xs text-slate-400">5 of 5 periods</dd></div>
  <div><dt class="text-xs text-slate-400">Seoul replication</dt><dd class="num text-2xl font-semibold text-emerald-400">−47%</dd><dd class="num text-xs text-slate-400">574.2 → 305.7</dd></div>
 </dl></section>

<!-- 1. plan -->
<section aria-labelledby="s1">
<div class="flex items-center gap-2 mb-3"><span class="secnum">1</span><h2 id="s1" class="text-base font-semibold text-white">Plan the next 24 hours</h2></div>
<div class="grid grid-cols-1 lg:grid-cols-12 gap-5">
 <div class="lg:col-span-4 card p-5 space-y-5">
  <div>
   <div class="flex items-baseline justify-between"><label for="r_ratio" class="text-sm font-medium text-white">Cost ratio r = C<sub>u</sub> / C<sub>o</sub></label><span id="rVal" class="num text-xl font-semibold text-blue-400">5.00</span></div>
   <input id="r_ratio" type="range" min="0.25" max="10" step="0.25" value="5" oninput="slide()" class="mt-2" aria-describedby="alphaInfo">
   <div class="flex justify-between text-[11px] text-slate-400 num"><span>0.25</span><span>1</span><span>5</span><span>10</span></div>
   <p id="alphaInfo" class="text-xs text-slate-300 mt-2"></p></div>
  <div><p class="eyebrow mb-2">Scenarios</p><div class="flex flex-wrap gap-2">
   <button type="button" class="chip" onclick="preset(3,17,1,1,26,50,12,5)">Rush hour · 17:00 · r=5</button>
   <button type="button" class="chip" onclick="preset(3,8,1,1,22,60,10,10)">High penalty · 08:00 · r=10</button>
   <button type="button" class="chip" onclick="preset(2,14,0,1,30,45,10,3)">Weekend · 14:00 · r=3</button>
   <button type="button" class="chip" onclick="preset(3,17,1,1,26,50,12,1)">Symmetric · r=1</button></div></div>
  <div class="grid grid-cols-1 gap-3">
   <div><p class="eyebrow mb-1.5">Input</p><div id="modeBtns" class="seg" role="group" aria-label="Input mode"><button type="button" data-v="whatif" class="on">What-if scenario</button><button type="button" data-v="replay">Replay a test day</button></div></div>
   <div id="baseBox"><p class="eyebrow mb-1.5">Base forecast</p><div id="baseBtns" class="seg" role="group" aria-label="Base forecast"><button type="button" data-v="train" class="on">XGBoost</button><button type="button" data-v="refit">XGBoost refitted</button></div></div></div>
  <div id="replayBox" class="hidden"><label for="date" class="eyebrow block mb-1.5">Held-out test day</label>
   <input id="date" type="date" onchange="run()" class="field num"><p class="text-[11px] text-slate-400 mt-1">Test period: <span id="period" class="num"></span></p></div>
  <div id="whatifBox" class="grid grid-cols-3 gap-2">
   <div><label for="season" class="eyebrow block mb-1.5">Season</label><select id="season" onchange="run()" class="field"><option value="1">Spring</option><option value="2">Summer</option><option value="3" selected>Fall</option><option value="4">Winter</option></select></div>
   <div><label for="workingday" class="eyebrow block mb-1.5">Day</label><select id="workingday" onchange="run()" class="field"><option value="1" selected>Working</option><option value="0">Weekend</option></select></div>
   <div><label for="weathersit" class="eyebrow block mb-1.5">Weather</label><select id="weathersit" onchange="run()" class="field"><option value="1" selected>Clear</option><option value="2">Mist</option><option value="3">Rain</option><option value="4">Storm</option></select></div></div>
  <div><div class="flex justify-between text-sm"><label for="hr">Hour of day</label><span id="hrVal" class="num font-medium text-white">17:00</span></div><input id="hr" type="range" min="0" max="23" value="17" oninput="slide()" class="mt-1"></div>
  <div id="weatherBox" class="space-y-3">
   <div><div class="flex justify-between text-sm"><label for="temp_c">Temperature</label><span id="tempVal" class="num text-white">26 °C</span></div><input id="temp_c" type="range" min="0" max="41" step="0.5" value="26" oninput="slide()" class="mt-1"></div>
   <div><div class="flex justify-between text-sm"><label for="hum_pct">Humidity</label><span id="humVal" class="num text-white">50%</span></div><input id="hum_pct" type="range" min="0" max="100" value="50" oninput="slide()" class="mt-1"></div>
   <div><div class="flex justify-between text-sm"><label for="wind_kmh">Wind speed</label><span id="windVal" class="num text-white">12 km/h</span></div><input id="wind_kmh" type="range" min="0" max="67" value="12" oninput="slide()" class="mt-1"></div>
   <p class="text-[11px] text-slate-400">What-if uses a typical one-week lag (median by season, day type and hour): an illustration, not a forecast for a real date.</p></div>
  <label class="flex items-start gap-2 text-xs text-slate-300 border-t border-slate-800 pt-4 cursor-pointer"><input id="rule" type="checkbox" checked onchange="run()" class="mt-0.5 accent-blue-500">
   <span>Operator rule: keep XGBoost and add the hourly buffer only when r &gt; 1 (no retraining).</span></label>
 </div>

 <div class="lg:col-span-8 flex flex-col gap-5">
  <div class="grid grid-cols-2 xl:grid-cols-4 gap-4">
   <div class="card p-4"><p id="lblBase" class="eyebrow">Forecast · 17:00</p><p class="mt-2"><span id="kBase" class="num text-3xl font-semibold text-blue-400">–</span> <span class="text-xs text-slate-400">rentals</span></p><p id="lblBaseSub" class="text-xs text-slate-400 mt-1">XGBoost mean forecast</p></div>
   <div class="card p-4"><p class="eyebrow">Hourly buffer</p><p class="mt-2"><span id="kBuf" class="num text-3xl font-semibold text-amber-400">–</span> <span class="text-xs text-slate-400">rentals</span></p><p id="lblAlpha" class="text-xs text-slate-400 mt-1">–</p></div>
   <div class="card p-4 border-emerald-700/60"><p class="eyebrow !text-emerald-300">Recommended plan</p><p class="mt-2"><span id="kPlan" class="num text-3xl font-semibold text-emerald-400">–</span> <span class="text-xs text-slate-400">rentals</span></p><p class="text-xs text-slate-400 mt-1">24-hour total <span id="kTot" class="num text-slate-200">–</span></p></div>
   <div class="card p-4"><p id="lblSav" class="eyebrow">Cost cut</p><p class="mt-2"><span id="kSav" class="num text-3xl font-semibold text-white">–</span></p><p id="lblCostSub" class="num text-xs text-slate-400 mt-1">–</p></div>
  </div>
  <div class="card p-5 flex-1 flex flex-col">
   <div class="flex flex-wrap items-start justify-between gap-2 mb-1"><div><h3 id="mainTitle" class="text-sm font-semibold text-white">Forecast vs. cost-optimal capacity plan</h3>
    <p class="text-xs text-slate-400">Shaded band = hour-of-day buffer above the forecast</p></div>
    <p id="dayStats" class="num text-xs text-slate-300"></p></div>
   <div class="relative w-full flex-1 min-h-[18rem]"><canvas id="cMain" aria-label="24-hour forecast, plan and actual demand" role="img"></canvas></div></div>
 </div></div></section>

<!-- 2. evaluate -->
<section aria-labelledby="s2">
<div class="flex items-center gap-2 mb-3"><span class="secnum">2</span><h2 id="s2" class="text-base font-semibold text-white">Evaluate on held-out test hours</h2></div>
<div class="grid grid-cols-1 lg:grid-cols-12 gap-5">
 <div class="lg:col-span-7 card p-5 overflow-x-auto">
  <div class="flex flex-wrap items-baseline justify-between gap-2 mb-3"><h3 class="text-sm font-semibold text-white">Asymmetric cost by plan <span id="tblR" class="num text-blue-400 font-medium"></span></h3><p id="tblSub" class="text-xs text-slate-400"></p></div>
  <table class="w-full text-sm"><thead><tr class="text-[11px] uppercase tracking-wide text-slate-400"><th class="text-left font-semibold pb-2 whitespace-nowrap">Model / plan</th><th class="text-left font-semibold pb-2 whitespace-nowrap">Role</th><th class="text-right font-semibold pb-2 pl-4 whitespace-nowrap">RMSE</th><th class="text-right font-semibold pb-2 pl-4 whitespace-nowrap">AvgCost(r)</th><th class="text-right font-semibold pb-2 pl-4 whitespace-nowrap">vs XGBoost</th><th class="text-right font-semibold pb-2 pl-4 whitespace-nowrap peakCol">Peak under</th><th class="text-right font-semibold pb-2 pl-4 whitespace-nowrap">Coverage</th></tr></thead>
  <tbody id="tbl"></tbody></table>
  <p class="text-[11px] text-slate-400 mt-3 leading-relaxed">AvgCost(r) = mean of r × shortfall + surplus, in rentals. Coverage = share of test hours with plan ≥ demand (target r/(1+r)). Peak under = share of working-day 07, 08, 17, 18 h under-predicted. Arm B rows appear at r = 1, 2, 3, 5 and 10.</p></div>
 <div class="lg:col-span-5 card p-5">
  <div class="flex items-start justify-between gap-2 mb-2"><div><h3 id="subTitle" class="text-sm font-semibold text-white">Hour-of-day buffer</h3><p id="subDesc" class="text-xs text-slate-400">Rentals added per hour from validation-error quantiles</p></div>
   <div id="subBtns" class="seg" role="group" aria-label="Chart"><button type="button" data-v="buf" class="on">Buffer</button><button type="button" data-v="imp">Features</button></div></div>
  <div class="relative w-full h-64"><canvas id="cSub" role="img" aria-label="Hourly buffer or feature importance"></canvas></div></div>
</div></section>

<!-- 3. evidence -->
<section aria-labelledby="s3">
<div class="flex items-center gap-2 mb-3"><span class="secnum">3</span><h2 id="s3" class="text-base font-semibold text-white">Why not just switch models?</h2></div>
<div class="grid grid-cols-1 lg:grid-cols-2 gap-5">
 <div class="card p-5"><h3 class="text-sm font-semibold text-white">Cost-sensitivity curve</h3><p id="curveNote" class="text-xs text-slate-400 mt-1 mb-3"></p>
  <div class="relative w-full h-64"><canvas id="cCurve" role="img" aria-label="Average cost by model across cost ratios"></canvas></div></div>
 <div class="card p-5"><h3 class="text-sm font-semibold text-white">When do real stations run empty?</h3>
  <p class="text-xs text-slate-400 mt-1 mb-3">Live GBFS feed, 877 stations, 14–22 Nov 2026. On weekdays empty stations peak near 10:00, after the commute peaks (shaded): the buffer sets how much capacity, not where.</p>
  <div class="relative w-full h-64"><canvas id="cGbfs" role="img" aria-label="Share of empty stations by hour"></canvas></div></div>
</div>
<div class="card p-5 mt-5 overflow-x-auto">
 <h3 class="text-sm font-semibold text-white">Robustness across five rolling test windows, r = 5</h3>
 <p class="text-xs text-slate-400 mt-1 mb-3">Each window is trained on earlier data and tuned on the two months before it. Lower is better.</p>
 <table class="w-full text-sm"><thead><tr class="text-[11px] uppercase tracking-wide text-slate-400"><th class="text-left font-semibold pb-2 whitespace-nowrap">Test window</th><th class="text-right font-semibold pb-2 pl-4 whitespace-nowrap">Standard</th><th class="text-right font-semibold pb-2 pl-4 whitespace-nowrap">Refitted</th><th class="text-right font-semibold pb-2 pl-4 whitespace-nowrap">Re-ranked</th><th class="text-right font-semibold pb-2 pl-4 whitespace-nowrap">Standard + buffer</th><th class="text-right font-semibold pb-2 pl-4 whitespace-nowrap">vs standard</th><th class="text-right font-semibold pb-2 pl-4 whitespace-nowrap">vs refitted</th></tr></thead>
 <tbody id="roll"></tbody></table>
 <p class="text-[11px] text-slate-400 mt-3">The buffer lowers cost in every window and beats the refitted model in four of five; in May–Jun, a period of fast growth, refitting alone is cheaper.</p></div>
</section>

<footer class="border-t border-slate-800 pt-4 pb-8 text-xs text-slate-400 flex flex-wrap justify-between gap-2">
 <p>Kobdej Manonai (st127172) &amp; Lwin Moe Aung (st127370) · AST02.10 Computer Programming for Data Science and AI · Asian Institute of Technology · Instructor: Dr. Chantri Polprasert</p>
 <p>Data: UCI Bike Sharing (2011–2012), UCI Seoul Bike Sharing (2017–2018), Capital Bikeshare GBFS. Cost is a proxy: data record rentals, not stockouts.</p></footer>
</main>

<script>
const S = {{ static_json|safe }};
const $=id=>document.getElementById(id), pad=n=>String(n).padStart(2,'0'), hrs=Array.from({length:24},(_,i)=>pad(i)+':00');
const C={pri:'#3b82f6',plan:'#10b981',buf:'#f59e0b',act:'#f43f5e',grid:'#1e293b',tick:'#94a3b8',txt:'#cbd5e1'};
Chart.defaults.font.family="'Fira Sans', system-ui, sans-serif";Chart.defaults.font.size=11;Chart.defaults.color=C.tick;Chart.defaults.borderColor=C.grid;
Chart.defaults.plugins.legend.labels.boxWidth=10;Chart.defaults.plugins.legend.labels.boxHeight=10;Chart.defaults.plugins.legend.labels.color=C.txt;
Chart.defaults.plugins.tooltip.backgroundColor='#0b1220';Chart.defaults.plugins.tooltip.borderColor='#334155';Chart.defaults.plugins.tooltip.borderWidth=1;
const st={city:'dc',mode:'whatif',base:'train'};
let tmr=null, sub='buf', curBuf=[], cSub=null, cCurve=null, last=null;
const fmt=v=>v===null||v===undefined?'–':Math.round(v).toLocaleString();
const cMain=new Chart($('cMain'),{type:'line',data:{labels:hrs,datasets:[
 {label:'Recommended plan',data:[],borderColor:C.plan,backgroundColor:'rgba(16,185,129,0.14)',borderWidth:2,borderDash:[6,4],fill:1,tension:0.3,pointRadius:0},
 {label:'XGBoost forecast',data:[],borderColor:C.pri,borderWidth:2.2,fill:false,tension:0.3,pointRadius:0},
 {label:'Actual demand',data:[],borderColor:C.buf,backgroundColor:C.buf,showLine:false,pointRadius:3.5,hidden:true},
 {label:'Selected hour',data:[],borderColor:C.act,backgroundColor:C.act,showLine:false,pointRadius:6,pointHoverRadius:7}
]},options:{responsive:true,maintainAspectRatio:false,interaction:{mode:'index',intersect:false},plugins:{legend:{position:'top',align:'end'}},
 scales:{x:{grid:{display:false},ticks:{maxRotation:0,autoSkip:true,maxTicksLimit:12}},y:{beginAtZero:true,title:{display:true,text:'rentals per hour'}}}}});
function group(id,cb){document.querySelectorAll('#'+id+' button').forEach(b=>b.onclick=()=>{document.querySelectorAll('#'+id+' button').forEach(x=>x.classList.remove('on'));b.classList.add('on');cb(b.dataset.v);});}
function setOn(id,v){document.querySelectorAll('#'+id+' button').forEach(b=>b.classList.toggle('on',b.dataset.v===v));}
function drawSub(){if(cSub)cSub.destroy();const hr=+$('hr').value, isB=sub==='buf';
 $('subTitle').innerText=isB?'Hour-of-day buffer':'XGBoost feature importance';$('subDesc').innerText=isB?(last&&!last.rule_applied?'Buffer the rule switches off at r ≤ 1 (shown in grey)':'Rentals added per hour from validation-error quantiles'):'Share of total gain, Washington model (%)';
 cSub=new Chart($('cSub'),{type:'bar',data:{labels:isB?hrs:S.importance.map(x=>x[0]),datasets:[{data:isB?curBuf:S.importance.map(x=>x[1]),
  backgroundColor:isB?hrs.map((_,i)=>i===hr?C.act:(last&&!last.rule_applied?'#475569':C.buf)):C.pri,borderRadius:3,maxBarThickness:28}]},
  options:{indexAxis:isB?'x':'y',responsive:true,maintainAspectRatio:false,animation:false,plugins:{legend:{display:false}},
  scales:{x:{grid:{display:!isB},ticks:isB?{maxRotation:0,autoSkip:true,maxTicksLimit:8}:{}},y:{grid:{display:isB},title:{display:isB,text:'rentals added'}}}}});}
function sync(){const r=parseFloat($('r_ratio').value);$('rVal').innerText=r.toFixed(2);$('hrVal').innerText=pad($('hr').value)+':00';$('tempVal').innerText=$('temp_c').value+' °C';$('humVal').innerText=$('hum_pct').value+'%';$('windVal').innerText=$('wind_kmh').value+' km/h';}
function slide(){sync();clearTimeout(tmr);tmr=setTimeout(run,80);}
function preset(s,h,w,we,t,hu,wi,r){$('season').value=s;$('hr').value=h;$('workingday').value=w;$('weathersit').value=we;$('temp_c').value=t;$('hum_pct').value=hu;$('wind_kmh').value=wi;$('r_ratio').value=r;
 if(st.mode==='replay'&&st.city==='dc')$('date').value=w===0?'2012-10-13':'2012-10-10';sync();run();}
function ui(){const c=S[st.city], seoul=st.city==='seoul';
 if(seoul){st.mode='replay';st.base='train';setOn('modeBtns','replay');setOn('baseBtns','train');}
 document.querySelector('#modeBtns [data-v=whatif]').classList.toggle('hidden',seoul);$('baseBox').classList.toggle('hidden',seoul);
 $('replayBox').classList.toggle('hidden',st.mode!=='replay');$('whatifBox').classList.toggle('hidden',st.mode==='replay');$('weatherBox').classList.toggle('hidden',st.mode==='replay');
 $('period').innerText=c.test_period[0].slice(0,10)+' to '+c.test_period[1].slice(0,10);$('date').min=c.dates[0];$('date').max=c.dates[c.dates.length-1];
 if(!c.dates.includes($('date').value))$('date').value=seoul?'2018-10-10':'2012-10-10';
 document.querySelectorAll('.peakCol').forEach(e=>e.classList.toggle('hidden',seoul));drawCurve();}
const roleCls={'Baseline':'base','Proposed':'prop','Recommended':'rec'};
async function run(){
 const p={city:st.city,mode:st.mode,base:st.base,date:$('date').value,rule:$('rule').checked,season:$('season').value,hr:$('hr').value,workingday:$('workingday').value,weathersit:$('weathersit').value,temp_c:$('temp_c').value,hum_pct:$('hum_pct').value,wind_kmh:$('wind_kmh').value,r_ratio:$('r_ratio').value};
 const d=await(await fetch('/api/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(p)})).json(); last=d;
 $('alphaInfo').innerHTML=d.rule_applied?`A missing rental costs <span class="num text-white">${d.r.toFixed(2)}×</span> an idle one, so plan at the <span class="num text-white">${d.alpha_pct}%</span> quantile.`:'r ≤ 1: the buffer is off and the mean forecast is used as it is.';
 $('lblBase').innerText=(d.base==='refit'?'Refitted forecast · ':'Forecast · ')+pad(d.hr)+':00';$('kBase').innerText=fmt(d.base_pred);
 $('lblBaseSub').innerHTML=d.sel_actual!==null?'Actual demand <span class="num text-amber-300">'+fmt(d.sel_actual)+'</span>':'XGBoost mean forecast';
 $('kBuf').innerText=(d.buf_val>=0?'+':'')+d.buf_val;$('lblAlpha').innerText=d.rule_applied?d.alpha_pct+'% quantile of validation errors':'Off (r ≤ 1)';
 $('kPlan').innerText=fmt(d.plan_pred);$('kTot').innerText=fmt(d.plan_total);
 $('lblSav').innerText='Cost cut · r = '+d.r.toFixed(2);$('kSav').innerText=(d.cost_cut>0?'−':'')+Math.abs(d.cost_cut).toFixed(1)+'%';
 $('kSav').className='num text-3xl font-semibold '+(d.cost_cut>0?'text-emerald-400':'text-white');
 $('lblCostSub').innerText=d.xgb_cost.toFixed(1)+' → '+d.buf_cost.toFixed(1)+(d.base==='refit'?' (refit + buffer)':' (XGBoost + buffer)');
 $('mainTitle').innerText=d.mode==='replay'?'Plan vs. actual demand · '+(d.city==='dc'?'Washington, D.C. · ':'Seoul · ')+d.date:'Forecast vs. cost-optimal capacity plan · what-if';
 const ds=d.day;$('dayStats').innerText=ds.hours?`Day total ${fmt(ds.actual)} · shortfall ${fmt(ds.short_base)} → ${fmt(ds.short_plan)} · AvgCost ${ds.cost_base} → ${ds.cost_plan}`:`Plan total ${fmt(d.plan_total)} rentals`;
 cMain.data.datasets[0].data=d.plan_preds;cMain.data.datasets[1].data=d.base_preds;cMain.data.datasets[1].label=d.base==='refit'?'Refitted XGBoost forecast':'XGBoost forecast';
 cMain.data.datasets[2].data=d.actual;cMain.data.datasets[2].hidden=d.mode!=='replay';cMain.data.datasets[3].data=hrs.map((_,i)=>i===d.hr?d.plan_preds[i]:null);cMain.update('none');
 $('tblR').innerText='· r = '+d.r.toFixed(2);$('tblSub').innerText=d.city==='dc'?'2,562 test hours · 12 Sep–31 Dec 2012':'1,204 test hours · 24 Sep–30 Nov 2018';
 $('tbl').innerHTML=d.table.map(x=>{const rc=roleCls[x.group]||'', hl=x.name===d.plan_name;
  const vs=x.vs_base===0?'—':(x.vs_base>0?'+':'−')+Math.abs(x.vs_base).toFixed(1)+'%';
  return `<tr class="${hl?'bg-emerald-950/40':''}"><td class="py-2 pr-2 ${hl?'text-emerald-300 font-semibold':'text-slate-200'}">${x.name}${x.name===d.best?' <span class="text-[10.5px] text-emerald-400 font-medium ml-1">lowest</span>':''}</td>
  <td class="py-2 pr-2"><span class="role ${rc}">${x.group}</span></td><td class="num text-right pl-4 text-slate-300">${x.rmse??'–'}</td>
  <td class="num text-right pl-4 ${hl?'text-emerald-300 font-semibold':'text-white'}">${x.cost.toFixed(1)}</td>
  <td class="num text-right pl-4 ${x.vs_base<0?'text-emerald-400':(x.vs_base>0?'text-rose-400':'text-slate-400')}">${vs}</td>
  <td class="num text-right pl-4 peakCol ${d.city==='seoul'?'hidden':''} text-slate-300">${x.peak_under===undefined?'–':x.peak_under+'%'}</td><td class="num text-right pl-4 text-slate-300">${x.cov}%</td></tr>`;}).join('');
 curBuf=d.rule_applied?d.buffers:d.buffers_raw;if(sub==='buf')drawSub();drawCurve();}
function drawCurve(){const c=S[st.city], r=+$('r_ratio').value, col={'Seasonal Naive':'#f97316','Hour x Weekday Avg':'#a78bfa','Linear Regression':'#94a3b8','Random Forest':'#22d3ee','XGBoost':C.pri};
 const sets=Object.entries(c.curves).map(([n,v])=>({label:n,data:v.map((y,i)=>({x:c.r_grid[i],y})),borderColor:col[n],backgroundColor:col[n],pointRadius:0,borderWidth:(n==='XGBoost'||n==='Seasonal Naive')?2.4:1.2,hidden:n==='Hour x Weekday Avg'}));
 const ymax=Math.max(...Object.entries(c.curves).filter(([n])=>n!=='Hour x Weekday Avg').map(([,v])=>Math.max(...v)));
 sets.push({label:'current r',data:[{x:r,y:0},{x:r,y:ymax}],borderColor:C.plan,backgroundColor:C.plan,borderDash:[4,4],pointRadius:0,borderWidth:1.5});
 if(c.rstar)sets.push({label:'r* ≈ '+c.rstar.toFixed(2),data:[{x:c.rstar,y:0},{x:c.rstar,y:ymax}],borderColor:C.act,backgroundColor:C.act,borderDash:[2,3],pointRadius:0,borderWidth:1.5});
 $('curveNote').innerText=c.rstar?`Re-scoring frozen models, Seasonal Naive becomes cheaper than XGBoost above r* ≈ ${c.rstar.toFixed(2)} on this test block, but r* moves between 0.5 and 4 across periods.`:'Seoul: no crossover at any r. XGBoost stays cheaper, and its lead narrows as r grows.';
 if(cCurve)cCurve.destroy();cCurve=new Chart($('cCurve'),{type:'line',data:{datasets:sets},options:{responsive:true,maintainAspectRatio:false,animation:false,parsing:false,
  scales:{x:{type:'logarithmic',min:0.25,max:10,grid:{display:false},title:{display:true,text:'cost ratio r (log scale)'},ticks:{callback:v=>[0.25,0.5,1,2,5,10].includes(+v)?v:''}},y:{beginAtZero:true,max:Math.ceil(ymax/50)*50,title:{display:true,text:'AvgCost(r)'}}}}});}
function drawStatic(){const g=S.gbfs, shade={id:'shade',beforeDraw(ch){const{ctx,chartArea:a,scales:{x}}=ch;ctx.save();ctx.fillStyle='rgba(148,163,184,.10)';[[7,9],[17,19]].forEach(([s,e])=>{const x0=x.getPixelForValue(s-0.5),x1=x.getPixelForValue(e-0.5);ctx.fillRect(x0,a.top,x1-x0,a.bottom-a.top);});ctx.restore();}};
 new Chart($('cGbfs'),{type:'line',plugins:[shade],data:{labels:[...Array(24).keys()],datasets:[
  {label:'Empty · weekday',data:g.Weekday.empty,borderColor:C.pri,backgroundColor:C.pri,pointRadius:2,borderWidth:2},{label:'Empty · weekend',data:g.Weekend.empty,borderColor:'#f97316',backgroundColor:'#f97316',pointRadius:2,borderWidth:2},
  {label:'Full · weekday',data:g.Weekday.full,borderColor:C.pri,borderDash:[4,3],pointRadius:0,hidden:true},{label:'Full · weekend',data:g.Weekend.full,borderColor:'#f97316',borderDash:[4,3],pointRadius:0,hidden:true}]},
  options:{responsive:true,maintainAspectRatio:false,scales:{x:{type:'linear',min:0,max:23,grid:{display:false},ticks:{stepSize:3},title:{display:true,text:'hour of day (Washington time)'}},y:{title:{display:true,text:'% of stations'}}}}});
 $('roll').innerHTML=S.rolling.map(w=>`<tr><td class="py-2 text-slate-200">${w.window.replace('-','–')}</td><td class="num text-right pl-4 text-slate-300">${w.standard.toFixed(1)}</td><td class="num text-right pl-4 text-slate-300">${w.refit.toFixed(1)}</td><td class="num text-right pl-4 text-slate-300">${w.rerank.toFixed(1)}</td>
  <td class="num text-right pl-4 text-emerald-300 font-semibold">${w.buffer.toFixed(1)}</td><td class="num text-right pl-4 text-emerald-400">−${Math.abs(w.vs_std)}%</td><td class="num text-right pl-4 ${w.vs_refit>0?'text-rose-400':'text-emerald-400'}">${w.vs_refit>0?'+':'−'}${Math.abs(w.vs_refit)}%</td></tr>`).join('');}
group('cityBtns',v=>{st.city=v;ui();run();});group('modeBtns',v=>{st.mode=v;ui();run();});group('baseBtns',v=>{st.base=v;run();});group('subBtns',v=>{sub=v;drawSub();});
sync();ui();drawStatic();run();
</script></body></html>
"""

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 8000)), debug=False)
