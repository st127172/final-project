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
<script src="https://cdn.tailwindcss.com"></script>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>body{font-family:system-ui,sans-serif;background:#0f172a;color:#f8fafc}input[type=range]{accent-color:#38bdf8}
.seg{font-size:11px;padding:4px 10px;border-radius:8px;border:1px solid #334155;background:#1e293b;color:#cbd5e1}
.seg.on{background:#0284c7;border-color:#0284c7;color:#fff;font-weight:600}
input[type=date]{color-scheme:dark}</style>
</head><body class="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-indigo-950 text-slate-100">
<header class="border-b border-slate-800 bg-slate-900/80 backdrop-blur sticky top-0 z-10">
<div class="max-w-7xl mx-auto px-5 py-3 flex flex-wrap items-center justify-between gap-2">
<div class="flex items-center gap-3">
<div class="w-9 h-9 rounded-xl bg-gradient-to-tr from-sky-500 to-emerald-500 flex items-center justify-center font-extrabold text-white">BF</div>
<div><h1 class="text-base font-bold text-white">BikeFlow AI <span class="text-emerald-400 text-xs">Beyond RMSE Cost-Sensitive Planner</span></h1>
<p class="text-xs text-slate-400">XGBoost + Hour-of-Day Quantile Buffer • Chronological 70/15/15 Split</p></div></div>
<div class="flex flex-wrap items-center gap-2">
<div id="cityBtns" class="flex gap-1"><button type="button" data-v="dc" class="seg on">Washington, D.C.</button><button type="button" data-v="seoul" class="seg">Seoul</button></div>
<span class="text-xs bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 px-3 py-1 rounded-full">● AWS EC2 Live</span></div>
</div></header>
<main class="max-w-7xl mx-auto px-5 py-4">
<div class="mb-4 flex flex-wrap items-center gap-2">
<span class="text-xs font-semibold text-slate-400 mr-1">Scenarios:</span>
<button type="button" onclick="preset(3,17,1,1,26,50,12,5)" class="text-xs bg-slate-800 hover:bg-sky-600 border border-slate-700 px-3 py-1.5 rounded-lg transition">Rush Hour (17:00, r=5)</button>
<button type="button" onclick="preset(3,8,1,1,22,60,10,10)" class="text-xs bg-slate-800 hover:bg-sky-600 border border-slate-700 px-3 py-1.5 rounded-lg transition">High Penalty (08:00, r=10)</button>
<button type="button" onclick="preset(2,14,0,1,30,45,10,3)" class="text-xs bg-slate-800 hover:bg-sky-600 border border-slate-700 px-3 py-1.5 rounded-lg transition">Sunny Weekend (14:00, r=3)</button>
<button type="button" onclick="preset(3,17,1,1,26,50,12,1)" class="text-xs bg-slate-800 hover:bg-sky-600 border border-slate-700 px-3 py-1.5 rounded-lg transition">Symmetric RMSE (r=1)</button>
</div>
<div class="grid grid-cols-1 lg:grid-cols-12 gap-5">
<div class="lg:col-span-4 bg-slate-900/80 border border-slate-800 rounded-2xl p-5 shadow-xl space-y-3">
<div class="p-3 rounded-xl bg-emerald-950/40 border border-emerald-500/30">
<div class="flex justify-between text-xs mb-1"><span class="font-bold text-emerald-300">Cost Ratio r (Cu / Co)</span><span id="rVal" class="font-extrabold text-emerald-400">r = 5.00</span></div>
<input id="r_ratio" type="range" min="0.25" max="10" step="0.25" value="5" oninput="slide()" class="w-full cursor-pointer">
<div id="alphaInfo" class="text-[11px] text-emerald-200/80 mt-1">Target Quantile α = r/(1+r) = 83.3% (Buffer Active)</div></div>
<div class="grid grid-cols-2 gap-2">
<div><label class="block text-[11px] text-slate-400 mb-1">Input mode</label><div id="modeBtns" class="flex gap-1">
<button type="button" data-v="whatif" class="seg on">What-if</button><button type="button" data-v="replay" class="seg">Replay test day</button></div></div>
<div id="baseBox"><label class="block text-[11px] text-slate-400 mb-1">Base forecast</label><div id="baseBtns" class="flex gap-1">
<button type="button" data-v="train" class="seg on">XGBoost</button><button type="button" data-v="refit" class="seg">Refitted</button></div></div></div>
<div id="replayBox" class="hidden"><label class="block text-[11px] text-slate-400 mb-1">Held-out test day (<span id="period"></span>)</label>
<input id="date" type="date" onchange="run()" class="w-full bg-slate-800 border border-slate-700 rounded-lg px-2 py-1.5 text-xs"></div>
<div id="whatifBox" class="space-y-3">
<div class="grid grid-cols-3 gap-2">
<div><label class="block text-[11px] text-slate-400 mb-1">Season</label>
<select id="season" onchange="run()" class="w-full bg-slate-800 border border-slate-700 rounded-lg px-2 py-1.5 text-xs">
<option value="1">Spring</option><option value="2">Summer</option><option value="3" selected>Fall</option><option value="4">Winter</option></select></div>
<div><label class="block text-[11px] text-slate-400 mb-1">Day Type</label>
<select id="workingday" onchange="run()" class="w-full bg-slate-800 border border-slate-700 rounded-lg px-2 py-1.5 text-xs">
<option value="1" selected>Working</option><option value="0">Weekend</option></select></div>
<div><label class="block text-[11px] text-slate-400 mb-1">Weather</label>
<select id="weathersit" onchange="run()" class="w-full bg-slate-800 border border-slate-700 rounded-lg px-2 py-1.5 text-xs">
<option value="1" selected>Clear</option><option value="2">Mist</option><option value="3">Rain</option><option value="4">Storm</option></select></div></div>
</div>
<div><div class="flex justify-between text-xs mb-0.5"><span>Hour of Day</span><span id="hrVal" class="font-bold text-sky-400">17:00</span></div>
<input id="hr" type="range" min="0" max="23" value="17" oninput="slide()" class="w-full cursor-pointer"></div>
<div id="weatherBox" class="space-y-3">
<div><div class="flex justify-between text-xs mb-0.5"><span>Temperature</span><span id="tempVal" class="font-bold text-sky-400">26 °C</span></div>
<input id="temp_c" type="range" min="0" max="41" step="0.5" value="26" oninput="slide()" class="w-full cursor-pointer"></div>
<div><div class="flex justify-between text-xs mb-0.5"><span>Humidity</span><span id="humVal" class="font-bold text-sky-400">50%</span></div>
<input id="hum_pct" type="range" min="0" max="100" value="50" oninput="slide()" class="w-full cursor-pointer"></div>
<div><div class="flex justify-between text-xs mb-0.5"><span>Windspeed</span><span id="windVal" class="font-bold text-sky-400">12 km/h</span></div>
<input id="wind_kmh" type="range" min="0" max="67" value="12" oninput="slide()" class="w-full cursor-pointer"></div>
<p class="text-[10px] text-slate-500">What-if uses a typical one-week lag (median by season, day type and hour), so it illustrates the plan rather than forecasting a real date.</p></div>
<label class="flex items-start gap-2 text-[11px] text-slate-400 bg-slate-800/70 p-2 rounded-lg border border-slate-700/60">
<input id="rule" type="checkbox" checked onchange="run()" class="mt-0.5">Operator Rule: Keep XGBoost; add hourly buffer when r &gt; 1 (no retraining).</label></div>
<div class="lg:col-span-8 flex flex-col gap-4">
<div class="grid grid-cols-2 sm:grid-cols-4 gap-3">
<div class="bg-slate-900/80 border border-slate-800 rounded-2xl p-3.5">
<div id="lblBase" class="text-[11px] text-slate-400 uppercase">XGBoost (17:00)</div>
<div class="mt-1 text-2xl font-extrabold text-sky-400"><span id="kBase">-</span> <span class="text-xs font-normal text-slate-400">rentals</span></div>
<div id="lblBaseSub" class="mt-1 text-[11px] text-slate-400">Mean forecast</div></div>
<div class="bg-slate-900/80 border border-slate-800 rounded-2xl p-3.5">
<div class="text-[11px] text-slate-400 uppercase">Hourly Buffer</div>
<div class="mt-1 text-2xl font-extrabold text-amber-400"><span id="kBuf">-</span> <span class="text-xs font-normal text-slate-400">rentals</span></div>
<div id="lblAlpha" class="mt-1 text-[11px] text-amber-300/80">α = 83.3% quantile</div></div>
<div class="bg-slate-900/80 border border-emerald-500/40 rounded-2xl p-3.5">
<div class="text-[11px] text-emerald-300 uppercase font-semibold">Recommended Plan</div>
<div class="mt-1 text-2xl font-extrabold text-emerald-400"><span id="kPlan">-</span> <span class="text-xs font-normal text-slate-400">rentals</span></div>
<div class="mt-1 text-[11px] text-slate-400">24h: <span id="kTot">-</span></div></div>
<div class="bg-slate-900/80 border border-slate-800 rounded-2xl p-3.5">
<div id="lblSav" class="text-[11px] text-slate-400 uppercase">Cost Cut (r=5.0)</div>
<div class="mt-1 text-2xl font-extrabold text-indigo-400"><span id="kSav">-</span>%</div>
<div id="lblCostSub" class="mt-1 text-[11px] text-slate-400">-</div></div></div>
<div class="bg-slate-900/80 border border-slate-800 rounded-2xl p-4 flex-1">
<div class="flex flex-wrap justify-between gap-2"><h3 id="mainTitle" class="text-sm font-bold text-white mb-0.5">24-Hour Forecast vs. Cost-Optimal Capacity Plan</h3>
<p id="dayStats" class="text-[11px] text-amber-300/90"></p></div>
<p class="text-xs text-slate-400 mb-2">Green band = Hour-of-day safety buffer protecting rush-hour stockouts</p>
<div class="relative w-full h-56"><canvas id="cMain"></canvas></div></div></div></div>
<div class="mt-4 grid grid-cols-1 lg:grid-cols-12 gap-5">
<div class="lg:col-span-7 bg-slate-900/80 border border-slate-800 rounded-2xl p-4 overflow-x-auto">
<h3 class="text-sm font-bold text-white mb-0.5">Beyond RMSE: Test Evaluation (<span id="tblR">r = 5.00</span>)</h3>
<p id="tblSub" class="text-xs text-slate-400 mb-2">2,562 held-out test hours • Live re-scored at selected r</p>
<table class="w-full text-xs text-left">
<thead><tr class="border-b border-slate-700 text-slate-400"><th class="py-1.5">Model / Plan</th><th>RMSE</th><th>AvgCost(r)</th><th>vs XGBoost</th><th class="peakCol">Peak Under%</th><th>Cov%</th></tr></thead>
<tbody id="tbl" class="divide-y divide-slate-800"></tbody></table>
<p class="text-[10px] text-slate-500 mt-2">AvgCost(r) = mean of r × shortfall + surplus (rentals). Cov% = test hours with plan ≥ demand (target r/(1+r)). Peak = working-day 07, 08, 17, 18 h. Arm B rows appear at r = 1, 2, 3, 5, 10 only.</p></div>
<div class="lg:col-span-5 bg-slate-900/80 border border-slate-800 rounded-2xl p-4">
<div class="flex items-center justify-between mb-2">
<div><h3 id="subTitle" class="text-sm font-bold text-white">Hour-of-Day Buffer Profile (Fig. 4.10)</h3>
<p id="subDesc" class="text-xs text-slate-400">Extra rentals added per hour from validation quantiles</p></div>
<button type="button" onclick="toggleSub()" id="subBtn" class="text-xs bg-slate-800 hover:bg-slate-700 border border-slate-700 px-2.5 py-1 rounded-lg">Feature Imp.</button></div>
<div class="relative w-full h-56"><canvas id="cSub"></canvas></div></div></div>
<div class="mt-4 grid grid-cols-1 lg:grid-cols-12 gap-5">
<div class="lg:col-span-6 bg-slate-900/80 border border-slate-800 rounded-2xl p-4">
<h3 class="text-sm font-bold text-white mb-0.5">Cost-Sensitivity Curve (Fig. 4.7)</h3>
<p id="curveNote" class="text-xs text-slate-400 mb-2"></p>
<div class="relative w-full h-56"><canvas id="cCurve"></canvas></div></div>
<div class="lg:col-span-6 bg-slate-900/80 border border-slate-800 rounded-2xl p-4">
<h3 class="text-sm font-bold text-white mb-0.5">When Do Real Stations Run Empty? (GBFS, Fig. 4.11)</h3>
<p class="text-xs text-slate-400 mb-2">877 Capital Bikeshare stations, 14–22 Nov 2026. Weekday empty share peaks near 10:00, after the commute peaks (shaded): the buffer sets how much capacity, not where.</p>
<div class="relative w-full h-56"><canvas id="cGbfs"></canvas></div></div></div>
<div class="mt-4 bg-slate-900/80 border border-slate-800 rounded-2xl p-4 overflow-x-auto">
<h3 class="text-sm font-bold text-white mb-0.5">Robustness: Five Rolling Test Windows at r = 5 (Table 4.5)</h3>
<table class="w-full text-xs text-left mt-2"><thead><tr class="border-b border-slate-700 text-slate-400"><th class="py-1.5">Test window</th><th>Standard</th><th>Refitted</th><th>Re-ranked</th><th>Standard + buffer</th><th>vs standard</th><th>vs refitted</th></tr></thead>
<tbody id="roll" class="divide-y divide-slate-800"></tbody></table>
<p class="text-[10px] text-slate-500 mt-2">The buffer lowers cost in every window (32–59%) and beats the refitted model in four of five; in May–Jun refitting alone is cheaper. Seoul: no crossover; the buffer cuts XGBoost's cost by 47% at r = 5.</p></div>
<footer class="text-[10px] text-slate-500 mt-4 pb-6 space-y-1">
<p>Kobdej Manonai (st127172) & Lwin Moe Aung (st127370) • AST02.10 Computer Programming for Data Science and AI, AIT • Instructor: Dr. Chantri Polprasert</p>
<p>Data: UCI Bike Sharing (Capital Bikeshare 2011–2012), UCI Seoul Bike Sharing Demand (2017–2018), Capital Bikeshare GBFS. The cost is a proxy: the data record rentals, not stockouts.</p></footer>
</main>
<script>
const S = {{ static_json|safe }};
const $=id=>document.getElementById(id), pad=n=>String(n).padStart(2,'0'), hrs=Array.from({length:24},(_,i)=>pad(i)+':00');
const st={city:'dc',mode:'whatif',base:'train'};
let tmr=null, mode='buf', curBuf=[], cSub=null, cCurve=null, last=null;
const impL=S.importance.map(x=>x[0]), impV=S.importance.map(x=>x[1]);
const tick={color:'#94a3b8',font:{size:10}};
const cMain=new Chart($('cMain'),{type:'line',data:{labels:hrs,datasets:[
{label:'Recommended Plan (+Buffer)',data:[],borderColor:'#10b981',backgroundColor:'rgba(16,185,129,0.16)',borderWidth:2.5,borderDash:[5,4],fill:1,tension:0.35,pointRadius:2},
{label:'XGBoost Forecast',data:[],borderColor:'#38bdf8',borderWidth:2.5,fill:false,tension:0.35,pointRadius:2},
{label:'Actual Demand',data:[],borderColor:'#fbbf24',backgroundColor:'#fbbf24',showLine:false,pointRadius:3.5,hidden:true}
]},options:{responsive:true,maintainAspectRatio:false,interaction:{mode:'index',intersect:false},plugins:{legend:{labels:{color:'#cbd5e1',boxWidth:12,font:{size:11}}}},scales:{x:{ticks:tick},y:{beginAtZero:true,ticks:tick}}}});
function group(id,cb){document.querySelectorAll('#'+id+' button').forEach(b=>b.onclick=()=>{document.querySelectorAll('#'+id+' button').forEach(x=>x.classList.remove('on'));b.classList.add('on');cb(b.dataset.v);});}
function setOn(id,v){document.querySelectorAll('#'+id+' button').forEach(b=>b.classList.toggle('on',b.dataset.v===v));}
function toggleSub(){mode=mode==='buf'?'imp':'buf';$('subBtn').innerText=mode==='buf'?'Feature Imp.':'Hourly Buffer';$('subTitle').innerText=mode==='buf'?'Hour-of-Day Buffer Profile (Fig. 4.10)':'XGBoost Feature Importance (%)';$('subDesc').innerText=mode==='buf'?'Extra rentals added per hour from validation quantiles':'Top predictors driving hourly demand (Washington, gain)';drawSub();}
function drawSub(){if(cSub)cSub.destroy();const hr=+$('hr').value;cSub=new Chart($('cSub'),{type:'bar',data:{labels:mode==='buf'?hrs:impL,datasets:[{data:mode==='buf'?curBuf:impV,backgroundColor:mode==='buf'?hrs.map((_,i)=>i===hr?'#f43f5e':(last&&!last.rule_applied?'#475569':'#f59e0b')):'#6366f1',borderRadius:4}]},options:{indexAxis:mode==='buf'?'x':'y',responsive:true,maintainAspectRatio:false,animation:false,plugins:{legend:{display:false}},scales:{x:{beginAtZero:true,ticks:tick},y:{beginAtZero:mode!=='buf',ticks:tick}}}});}
function sync(){const r=parseFloat($('r_ratio').value);$('rVal').innerText='r = '+r.toFixed(2);$('hrVal').innerText=pad($('hr').value)+':00';$('tempVal').innerText=$('temp_c').value+' °C';$('humVal').innerText=$('hum_pct').value+'%';$('windVal').innerText=$('wind_kmh').value+' km/h';}
function slide(){sync();clearTimeout(tmr);tmr=setTimeout(run,80);}
function preset(s,h,w,we,t,hu,wi,r){$('season').value=s;$('hr').value=h;$('workingday').value=w;$('weathersit').value=we;$('temp_c').value=t;$('hum_pct').value=hu;$('wind_kmh').value=wi;$('r_ratio').value=r;
 if(st.mode==='replay'&&st.city==='dc')$('date').value=w===0?'2012-10-13':'2012-10-10';sync();run();}
function ui(){const c=S[st.city], seoul=st.city==='seoul';
 if(seoul){st.mode='replay';st.base='train';setOn('modeBtns','replay');setOn('baseBtns','train');}
 document.querySelector('#modeBtns [data-v=whatif]').classList.toggle('hidden',seoul);$('baseBox').classList.toggle('invisible',seoul);
 $('replayBox').classList.toggle('hidden',st.mode!=='replay');$('whatifBox').classList.toggle('hidden',st.mode==='replay');$('weatherBox').classList.toggle('hidden',st.mode==='replay');
 $('period').innerText=c.test_period[0].slice(0,10)+' to '+c.test_period[1].slice(0,10);$('date').min=c.dates[0];$('date').max=c.dates[c.dates.length-1];
 if(!c.dates.includes($('date').value))$('date').value=seoul?'2018-10-10':'2012-10-10';
 document.querySelectorAll('.peakCol').forEach(e=>e.classList.toggle('hidden',seoul));drawCurve();}
async function run(){
 const p={city:st.city,mode:st.mode,base:st.base,date:$('date').value,rule:$('rule').checked,season:$('season').value,hr:$('hr').value,workingday:$('workingday').value,weathersit:$('weathersit').value,temp_c:$('temp_c').value,hum_pct:$('hum_pct').value,wind_kmh:$('wind_kmh').value,r_ratio:$('r_ratio').value};
 const d=await(await fetch('/api/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(p)})).json(); last=d;
 $('alphaInfo').innerText=!d.rule_applied?'r ≤ 1.0: Buffer Off (Use XGBoost Mean Forecast)':'Target Quantile α = r/(1+r) = '+d.alpha_pct+'% (Buffer Active)';
 $('lblBase').innerText=(d.base==='refit'?'Refitted XGBoost (':'XGBoost (')+pad(d.hr)+':00)';$('kBase').innerText=d.base_pred??'-';
 $('lblBaseSub').innerText=d.sel_actual!==null?'Actual demand: '+d.sel_actual.toLocaleString():'Mean forecast';
 $('kBuf').innerText=(d.buf_val>=0?'+':'')+d.buf_val;$('lblAlpha').innerText=!d.rule_applied?'Disabled (r ≤ 1)':'α = '+d.alpha_pct+'% quantile';
 $('kPlan').innerText=d.plan_pred??'-';$('kTot').innerText=d.plan_total.toLocaleString();
 $('lblSav').innerText='Cost Cut (r='+d.r.toFixed(1)+')';$('kSav').innerText=d.cost_cut;$('lblCostSub').innerText=d.xgb_cost.toFixed(1)+' → '+d.buf_cost.toFixed(1)+(d.base==='refit'?' (refit + buffer)':'');
 $('mainTitle').innerText=d.mode==='replay'?'24-Hour Plan vs. Actual Demand: '+(d.city==='dc'?'Washington, D.C., ':'Seoul, ')+d.date:'24-Hour Forecast vs. Cost-Optimal Capacity Plan';
 const ds=d.day;$('dayStats').innerText=ds.hours?`Day (${ds.hours} h): actual ${ds.actual.toLocaleString()} • shortfall ${ds.short_base.toLocaleString()} → ${ds.short_plan.toLocaleString()} • AvgCost ${ds.cost_base} → ${ds.cost_plan}`:'';
 cMain.data.datasets[0].data=d.plan_preds;cMain.data.datasets[1].data=d.base_preds;cMain.data.datasets[1].label=d.base==='refit'?'Refitted XGBoost Forecast':'XGBoost Forecast';
 cMain.data.datasets[2].data=d.actual;cMain.data.datasets[2].hidden=d.mode!=='replay';
 [0,1].forEach(k=>{cMain.data.datasets[k].pointRadius=hrs.map((_,i)=>i===d.hr?6:2);cMain.data.datasets[k].pointBackgroundColor=hrs.map((_,i)=>i===d.hr?'#f43f5e':(k===0?'#10b981':'#38bdf8'));});
 cMain.update('none');
 $('tblR').innerText='r = '+d.r.toFixed(2);$('tblSub').innerText=(d.city==='dc'?'2,562 held-out test hours (12 Sep–31 Dec 2012)':'1,204 held-out test hours (24 Sep–30 Nov 2018)')+' • Live re-scored at selected r';
 const prop=d.plan_name;
 $('tbl').innerHTML=d.table.map(x=>{const hl=x.name===prop, base=x.group==='Baseline';
  const nc=hl?'text-emerald-400 font-bold':(base?'text-sky-400 font-medium':'text-slate-300');
  const vs=x.vs_base===0?'—':(x.vs_base>0?'+':'')+x.vs_base.toFixed(1)+'%';
  return `<tr class="${hl?'bg-emerald-950/30':''}"><td class="py-1.5 ${nc}">${x.name}${x.name===d.best?' <span class="text-[10px] text-emerald-300">● lowest</span>':''}</td><td>${x.rmse??'-'}</td><td class="${hl?'font-bold text-emerald-400':''}">${x.cost.toFixed(1)}</td><td class="${x.vs_base<0?'text-emerald-400':(x.vs_base>0?'text-rose-400':'')}">${vs}</td><td class="peakCol ${d.city==='seoul'?'hidden':''} ${base?'text-rose-400':''}">${x.peak_under===undefined?'-':x.peak_under+'%'}</td><td>${x.cov}%</td></tr>`;}).join('');
 curBuf=d.rule_applied?d.buffers:d.buffers_raw;if(mode==='buf')drawSub();drawCurve();
}
function drawCurve(){const c=S[st.city], r=+$('r_ratio').value, col={'Seasonal Naive':'#f97316','Hour x Weekday Avg':'#a78bfa','Linear Regression':'#94a3b8','Random Forest':'#22d3ee','XGBoost':'#38bdf8'};
 const sets=Object.entries(c.curves).map(([n,v])=>({label:n,data:v.map((y,i)=>({x:c.r_grid[i],y})),borderColor:col[n],pointRadius:0,borderWidth:(n==='XGBoost'||n==='Seasonal Naive')?2.5:1.2,hidden:n==='Hour x Weekday Avg'}));
 const ymax=Math.max(...Object.entries(c.curves).filter(([n])=>n!=='Hour x Weekday Avg').map(([,v])=>Math.max(...v)));
 sets.push({label:'current r',data:[{x:r,y:0},{x:r,y:ymax}],borderColor:'#10b981',borderDash:[4,4],pointRadius:0,borderWidth:1.5});
 if(c.rstar)sets.push({label:'r* ≈ '+c.rstar.toFixed(2),data:[{x:c.rstar,y:0},{x:c.rstar,y:ymax}],borderColor:'#f43f5e',borderDash:[2,3],pointRadius:0,borderWidth:1.5});
 $('curveNote').innerText=c.rstar?`Re-scoring frozen models: Seasonal Naive becomes cheaper than XGBoost above r* ≈ ${c.rstar.toFixed(2)} on this test block, but r* is unstable (0.5–4 across windows).`:'Seoul: no crossover at any r. XGBoost stays cheaper; its lead narrows as r grows.';
 if(cCurve)cCurve.destroy();cCurve=new Chart($('cCurve'),{type:'line',data:{datasets:sets},options:{responsive:true,maintainAspectRatio:false,animation:false,parsing:false,
  scales:{x:{type:'logarithmic',min:0.25,max:10,title:{display:true,text:'cost ratio r (log scale)',color:'#94a3b8'},ticks:{...tick,callback:v=>[0.25,0.5,1,2,5,10].includes(+v)?v:''}},y:{beginAtZero:true,max:Math.ceil(ymax/50)*50,ticks:tick,title:{display:true,text:'AvgCost(r)',color:'#94a3b8'}}},
  plugins:{legend:{labels:{color:'#cbd5e1',boxWidth:10,font:{size:10}}}}}});}
function drawStatic(){const g=S.gbfs, shade={id:'shade',beforeDraw(ch){const{ctx,chartArea:a,scales:{x}}=ch;ctx.save();ctx.fillStyle='rgba(148,163,184,.12)';[[7,9],[17,19]].forEach(([s,e])=>{const x0=x.getPixelForValue(s-0.5),x1=x.getPixelForValue(e-0.5);ctx.fillRect(x0,a.top,x1-x0,a.bottom-a.top);});ctx.restore();}};
 new Chart($('cGbfs'),{type:'line',plugins:[shade],data:{labels:[...Array(24).keys()],datasets:[
  {label:'Empty, weekday',data:g.Weekday.empty,borderColor:'#38bdf8',pointRadius:2},{label:'Empty, weekend',data:g.Weekend.empty,borderColor:'#f97316',pointRadius:2},
  {label:'Full, weekday',data:g.Weekday.full,borderColor:'#38bdf8',borderDash:[4,3],pointRadius:0,hidden:true},{label:'Full, weekend',data:g.Weekend.full,borderColor:'#f97316',borderDash:[4,3],pointRadius:0,hidden:true}]},
  options:{responsive:true,maintainAspectRatio:false,scales:{x:{type:'linear',min:0,max:23,ticks:tick,title:{display:true,text:'hour of day (Washington time)',color:'#94a3b8'}},y:{ticks:tick,title:{display:true,text:'% of stations',color:'#94a3b8'}}},plugins:{legend:{labels:{color:'#cbd5e1',boxWidth:10,font:{size:10}}}}}});
 $('roll').innerHTML=S.rolling.map(w=>`<tr><td class="py-1.5 text-slate-300">${w.window}</td><td>${w.standard.toFixed(1)}</td><td>${w.refit.toFixed(1)}</td><td>${w.rerank.toFixed(1)}</td><td class="text-emerald-400 font-bold">${w.buffer.toFixed(1)}</td><td class="text-emerald-400">${w.vs_std}%</td><td class="${w.vs_refit>0?'text-rose-400':'text-emerald-400'}">${w.vs_refit>0?'+':''}${w.vs_refit}%</td></tr>`).join('');}
group('cityBtns',v=>{st.city=v;ui();run();});group('modeBtns',v=>{st.mode=v;ui();run();});group('baseBtns',v=>{st.base=v;run();});
sync();ui();drawStatic();run();
</script></body></html>
"""

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 8000)), debug=False)
