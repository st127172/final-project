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
@app.route("/api/plan", methods=["POST"])
def api_plan():
    q = request.get_json(silent=True) or {}
    city = "seoul" if q.get("city") == "seoul" else "dc"
    c = CITY[city]
    r = num(q.get("r"), 0.25, 10.0, 5.0)
    rule = bool(q.get("rule", True))
    base = "refit" if (q.get("base") == "refit" and city == "dc") else "train"
    mode = "whatif" if (q.get("mode") == "whatif" and city == "dc") else "replay"
    hr = int(num(q.get("hr"), 0, 23, 17))
    alpha = r / (1 + r)

    buf = hourly_buffer(c["err"]["XGBoost"], c["hv"], r)
    applied = (r > 1) or not rule
    buf_used = buf if applied else np.zeros(24)

    actual = [None] * 24
    if mode == "replay":
        date = str(q.get("date") or "")
        if date not in c["dates"]:
            date = "2012-10-10" if city == "dc" else "2018-10-10"
        idx = np.where(c["date"] == date)[0]
        key = "XGBoost refitted" if base == "refit" else "XGBoost"
        base_pred = [None] * 24
        for i in idx:
            h = int(c["ht"][i])
            base_pred[h] = float(c["P"][key][i]); actual[h] = float(c["y"][i])
        title = date
    else:
        date = None
        month = int(num(q.get("month"), 1, 12, 10))
        wd = 1 if q.get("workingday", 1) in (1, "1", True) else 0
        ws = int(num(q.get("weathersit"), 1, 4, 1))
        rows = [whatif_features(h, month, wd, ws, num(q.get("temp_c"), -8, 41, 22),
                                num(q.get("hum_pct"), 0, 100, 55), num(q.get("wind_kmh"), 0, 67, 12)) for h in range(24)]
        X = pd.DataFrame(rows)[DC["feats"]]
        model = DC["xgb_refit"] if base == "refit" else DC["xgb"]
        base_pred = np.clip(model.predict(X), 0, None).astype(float).tolist()
        title = "What-if scenario"

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
    return jsonify({
        "city": city, "mode": mode, "title": title, "date": date, "r": r, "alpha_pct": round(100 * alpha, 1),
        "rule_applied": applied, "base": base, "hr": hr,
        "base_preds": [None if v is None else round(v) for v in base_pred],
        "plan_preds": [None if v is None else round(v) for v in plan],
        "actual": [None if v is None else round(v) for v in actual],
        "buffers": [round(float(v), 1) for v in buf_used],
        "buffers_raw": [round(float(v), 1) for v in buf],
        "sel": {"base": None if base_pred[hr] is None else round(base_pred[hr]),
                "buffer": round(float(buf_used[hr])),
                "plan": None if plan[hr] is None else round(plan[hr]),
                "actual": None if actual[hr] is None else round(actual[hr])},
        "plan_total": round(sum(v for v in plan if v is not None)),
        "base_total": round(sum(v for v in base_pred if v is not None)),
        "day": day, "table": rows, "best": best,
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
<html lang="en" class="dark">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>BikeFlow AI | Beyond RMSE</title>
<script src="https://cdn.tailwindcss.com"></script>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
 body{background:#0b1120;font-family:Inter,ui-sans-serif,system-ui,sans-serif}
 .card{background:#111a2e;border:1px solid #1f2a44;border-radius:14px}
 .pill{border:1px solid #334155;border-radius:999px;padding:4px 12px;font-size:12px;color:#cbd5e1}
 .pill.on{background:#0ea5e9;border-color:#0ea5e9;color:#0b1120;font-weight:600}
 input[type=range]{accent-color:#10b981;width:100%}
 th{font-weight:500;color:#94a3b8}
 .best{background:rgba(16,185,129,.12)}
 select,input[type=date]{background:#0b1120;border:1px solid #334155;border-radius:8px;padding:4px 8px;color:#e2e8f0}
</style>
</head>
<body class="text-slate-200">
<div class="max-w-7xl mx-auto p-4 md:p-6 space-y-4">

<header class="flex flex-wrap items-end justify-between gap-3">
 <div>
  <h1 class="text-2xl font-semibold">BikeFlow <span class="text-emerald-400">AI</span></h1>
  <p class="text-sm text-slate-400">Beyond RMSE: cost-aware hourly planning on top of the most accurate forecast. Keep the model, plan higher, hour by hour.</p>
 </div>
 <div class="flex gap-2" id="cityBtns">
  <button class="pill on" data-city="dc">Washington, D.C. (Capital Bikeshare)</button>
  <button class="pill" data-city="seoul">Seoul</button>
 </div>
</header>

<div class="grid lg:grid-cols-4 gap-4">
 <!-- controls -->
 <aside class="card p-4 space-y-4 lg:col-span-1">
  <div>
   <div class="flex justify-between text-sm"><span>Cost ratio r = C<sub>u</sub>/C<sub>o</sub></span><b id="rVal" class="text-emerald-400"></b></div>
   <input id="r" type="range" min="0.25" max="10" step="0.25" value="5">
   <p class="text-xs text-slate-400">A missing rental counts <b id="rTxt"></b>x an idle one. Plan target: the <b id="aTxt"></b>% quantile.</p>
  </div>
  <div>
   <div class="flex justify-between text-sm"><span>Selected hour</span><b id="hrVal"></b></div>
   <input id="hr" type="range" min="0" max="23" step="1" value="17">
  </div>
  <div class="space-y-1">
   <p class="text-xs text-slate-400">Scenario presets</p>
   <div class="flex flex-wrap gap-2" id="presets">
    <button class="pill" data-p='{"hr":17,"r":5}'>Rush hour 17:00, r=5</button>
    <button class="pill" data-p='{"hr":8,"r":10}'>High penalty 08:00, r=10</button>
    <button class="pill" data-p='{"hr":14,"r":3,"workingday":0}'>Weekend 14:00, r=3</button>
    <button class="pill" data-p='{"hr":17,"r":1}'>Symmetric r=1</button>
   </div>
  </div>
  <div class="space-y-2" id="baseBox">
   <p class="text-xs text-slate-400">Base forecast</p>
   <div class="flex flex-wrap gap-2" id="baseBtns">
    <button class="pill on" data-base="train">XGBoost (train block)</button>
    <button class="pill" data-base="refit">Refitted XGBoost (train+val)</button>
   </div>
  </div>
  <label class="flex items-start gap-2 text-xs text-slate-300"><input id="rule" type="checkbox" checked class="mt-0.5">
   Operator rule: no buffer when r &le; 1 (report Sec. 5.3)</label>
  <div class="space-y-2">
   <p class="text-xs text-slate-400">Input mode</p>
   <div class="flex gap-2" id="modeBtns">
    <button class="pill on" data-mode="replay">Replay a test day</button>
    <button class="pill" data-mode="whatif">What-if</button>
   </div>
  </div>
  <div id="replayBox" class="space-y-1">
   <label class="text-xs text-slate-400">Held-out test day (<span id="period"></span>)</label>
   <input id="date" type="date" class="w-full">
  </div>
  <div id="whatifBox" class="space-y-3 hidden">
   <div class="grid grid-cols-2 gap-2 text-xs">
    <label>Month<select id="month" class="w-full">
     <option>1</option><option>2</option><option>3</option><option>4</option><option>5</option><option>6</option>
     <option>7</option><option>8</option><option>9</option><option selected>10</option><option>11</option><option>12</option></select></label>
    <label>Day type<select id="workingday" class="w-full"><option value="1">Working day</option><option value="0">Weekend / holiday</option></select></label>
   </div>
   <label class="text-xs block">Weather<select id="weathersit" class="w-full">
    <option value="1">1 Clear / partly cloudy</option><option value="2">2 Mist / cloudy</option>
    <option value="3">3 Light rain or snow</option><option value="4">4 Heavy rain / storm</option></select></label>
   <div><div class="flex justify-between text-xs"><span>Temperature</span><b id="tVal"></b></div><input id="temp_c" type="range" min="-8" max="40" value="22"></div>
   <div><div class="flex justify-between text-xs"><span>Humidity</span><b id="hVal"></b></div><input id="hum_pct" type="range" min="0" max="100" value="55"></div>
   <div><div class="flex justify-between text-xs"><span>Wind</span><b id="wVal"></b></div><input id="wind_kmh" type="range" min="0" max="60" value="12"></div>
   <p class="text-[11px] text-slate-500">What-if uses observed-style weather and a typical one-week lag (median by season, day type and hour), so it is an illustration, not a forecast for a real date.</p>
  </div>
 </aside>

 <!-- main -->
 <main class="lg:col-span-3 space-y-4">
  <div class="grid grid-cols-2 md:grid-cols-4 gap-3" id="kpis">
   <div class="card p-3"><p class="text-xs text-slate-400">XGBoost forecast <span id="kHr"></span></p><p class="text-2xl font-semibold text-sky-400" id="kBase">-</p><p class="text-xs text-slate-500">rentals</p></div>
   <div class="card p-3"><p class="text-xs text-slate-400">Hourly buffer</p><p class="text-2xl font-semibold text-amber-300" id="kBuf">-</p><p class="text-xs text-slate-500" id="kBufNote">rentals</p></div>
   <div class="card p-3"><p class="text-xs text-slate-400">Recommended plan</p><p class="text-2xl font-semibold text-emerald-400" id="kPlan">-</p><p class="text-xs text-slate-500" id="kAct">rentals</p></div>
   <div class="card p-3"><p class="text-xs text-slate-400">Test-block cost cut at this r</p><p class="text-2xl font-semibold" id="kCut">-</p><p class="text-xs text-slate-500" id="kCutNote"></p></div>
  </div>

  <div class="card p-4">
   <div class="flex flex-wrap justify-between gap-2 mb-2">
    <h2 class="font-medium" id="mainTitle">24-hour plan</h2>
    <p class="text-xs text-slate-400" id="dayStats"></p>
   </div>
   <div class="h-72"><canvas id="cMain"></canvas></div>
  </div>

  <div class="card p-4 overflow-x-auto">
   <div class="flex flex-wrap justify-between gap-2 mb-2">
    <h2 class="font-medium">Live evaluation on all held-out test hours</h2>
    <p class="text-xs text-slate-400" id="tblNote"></p>
   </div>
   <table class="w-full text-sm">
    <thead><tr class="text-left border-b border-slate-700"><th class="py-1">Plan</th><th>Role</th><th class="text-right">AvgCost(r)</th><th class="text-right">vs baseline</th><th class="text-right">Coverage</th><th class="text-right" id="thPeak">Peak hours under-predicted</th></tr></thead>
    <tbody id="tbl"></tbody>
   </table>
   <p class="text-[11px] text-slate-500 mt-2">AvgCost(r) = mean of r x shortfall + surplus (rentals). Coverage = share of test hours with plan &ge; actual demand (target = r/(1+r)). Peak hours = working-day 07, 08, 17, 18 h, defined in advance. Arm B rows appear only at r = 1, 2, 3, 5 and 10, the ratios it was trained for.</p>
  </div>

  <div class="grid md:grid-cols-2 gap-4">
   <div class="card p-4"><h2 class="font-medium mb-1">Cost-sensitivity curve (Fig. 4.7)</h2><p class="text-xs text-slate-400 mb-2" id="curveNote"></p><div class="h-64"><canvas id="cCurve"></canvas></div></div>
   <div class="card p-4">
    <div class="flex justify-between items-center mb-2"><h2 class="font-medium" id="subTitle">Hour-of-day buffer (Fig. 4.10)</h2>
     <div class="flex gap-2" id="subBtns"><button class="pill on" data-sub="buf">Buffer</button><button class="pill" data-sub="imp">Feature importance</button></div></div>
    <div class="h-64"><canvas id="cSub"></canvas></div>
   </div>
  </div>

  <div class="grid md:grid-cols-2 gap-4">
   <div class="card p-4"><h2 class="font-medium mb-1">When do real stations run empty? (GBFS, Fig. 4.11)</h2>
    <p class="text-xs text-slate-400 mb-2">Capital Bikeshare live feed, 14-22 Nov 2026, 192 snapshots of 877 stations. Weekday commute peaks shaded. Stockouts peak around 10:00 on weekdays, after the morning commute, so a city-wide buffer sets how much capacity to plan, not where to put it.</p>
    <div class="h-56"><canvas id="cGbfs"></canvas></div></div>
   <div class="card p-4 overflow-x-auto"><h2 class="font-medium mb-1">Robustness: five rolling test windows, r = 5 (Table 4.5)</h2>
    <table class="w-full text-sm"><thead><tr class="text-left border-b border-slate-700"><th class="py-1">Window</th><th class="text-right px-1">Standard</th><th class="text-right px-1">Refitted</th><th class="text-right px-1">Re-ranked</th><th class="text-right px-1">+ Buffer</th><th class="text-right px-1">vs std</th><th class="text-right px-1">vs refit</th></tr></thead><tbody id="roll"></tbody></table>
    <p class="text-[11px] text-slate-500 mt-2">The buffer lowers the cost in every window (32-59%) and beats the refitted model in four of five; in May-Jun (fast growth) refitting alone is cheaper. Re-ranking models by cost is unreliable. Seoul: no crossover, the buffer cuts XGBoost's cost by 47% at r = 5.</p>
   </div>
  </div>

  <footer class="text-[11px] text-slate-500 space-y-1 pb-6">
   <p>Kobdej Manonai (st127172) and Lwin Moe Aung (st127370). AST02.10 Computer Programming for Data Science and AI, Asian Institute of Technology. Instructor: Dr. Chantri Polprasert.</p>
   <p>Data: UCI Bike Sharing (Capital Bikeshare 2011-2012), UCI Seoul Bike Sharing Demand (2017-2018), Capital Bikeshare GBFS. Chronological 70/15/15 split; nothing is tuned on the test block. The cost is a proxy: the data record rentals, not stockouts, and forecasts are in rentals, not bikes.</p>
  </footer>
 </main>
</div>

<script>
const S = {{ static_json|safe }};
const st = {city:'dc', base:'train', mode:'replay', sub:'buf', workingday:1};
const $ = id => document.getElementById(id);
const fmt = v => (v===null||v===undefined) ? '-' : Math.round(v).toLocaleString();
const H = [...Array(24).keys()].map(h => String(h).padStart(2,'0')+':00');
Chart.defaults.color = '#94a3b8'; Chart.defaults.borderColor = '#1f2a44'; Chart.defaults.font.size = 11;
let cMain, cCurve, cSub, cGbfs, last = null, timer = null;

function group(id, attr, cb){ document.querySelectorAll('#'+id+' button').forEach(b => b.onclick = () => {
  document.querySelectorAll('#'+id+' button').forEach(x => x.classList.remove('on')); b.classList.add('on'); cb(b.dataset[attr]); }); }
function setOn(id, attr, val){ document.querySelectorAll('#'+id+' button').forEach(b => b.classList.toggle('on', b.dataset[attr]===val)); }

function labels(){
  const r = +$('r').value; $('rVal').textContent = r.toFixed(2); $('rTxt').textContent = r.toFixed(2);
  $('aTxt').textContent = (100*r/(1+r)).toFixed(1); $('hrVal').textContent = H[+$('hr').value];
  $('tVal').textContent = $('temp_c').value+' °C'; $('hVal').textContent = $('hum_pct').value+' %'; $('wVal').textContent = $('wind_kmh').value+' km/h';
}
function cityUI(){
  const c = S[st.city];
  $('period').textContent = c.test_period[0].slice(0,10)+' to '+c.test_period[1].slice(0,10);
  $('date').min = c.dates[0]; $('date').max = c.dates[c.dates.length-1];
  if (!c.dates.includes($('date').value)) $('date').value = st.city==='dc' ? '2012-10-10' : '2018-10-10';
  const seoul = st.city==='seoul';
  $('baseBox').classList.toggle('hidden', seoul);
  document.querySelector('#modeBtns [data-mode=whatif]').classList.toggle('hidden', seoul);
  if (seoul){ st.base='train'; st.mode='replay'; setOn('baseBtns','base','train'); setOn('modeBtns','mode','replay'); }
  $('replayBox').classList.toggle('hidden', st.mode!=='replay'); $('whatifBox').classList.toggle('hidden', st.mode!=='whatif');
  $('thPeak').classList.toggle('hidden', seoul);
}

function payload(){
  return {city:st.city, base:st.base, mode:st.mode, r:+$('r').value, hr:+$('hr').value, rule:$('rule').checked,
          date:$('date').value, month:+$('month').value, workingday:+$('workingday').value, weathersit:+$('weathersit').value,
          temp_c:+$('temp_c').value, hum_pct:+$('hum_pct').value, wind_kmh:+$('wind_kmh').value};
}
function onSlide(){ labels(); clearTimeout(timer); timer = setTimeout(refresh, 80); }
async function refresh(){
  const res = await fetch('/api/plan', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload())});
  last = await res.json(); render(last);
}

function render(d){
  // KPIs
  $('kHr').textContent = 'at '+H[d.hr]; $('kBase').textContent = fmt(d.sel.base);
  $('kBuf').textContent = (d.sel.buffer>=0?'+':'')+d.sel.buffer;
  $('kBufNote').textContent = d.rule_applied ? 'q'+d.alpha_pct+'% of validation errors' : 'off: r ≤ 1, forecast used as it is';
  $('kPlan').textContent = fmt(d.sel.plan);
  $('kAct').textContent = d.sel.actual!==null ? 'actual demand: '+fmt(d.sel.actual) : 'rentals';
  const T = d.table, base = T.find(x => x.group==='Baseline');
  const prop = T.find(x => x.group===(d.base==='refit'?'Recommended':'Proposed'));
  const cut = 100*(1-prop.cost/base.cost);
  $('kCut').textContent = (cut>=0?'-':'+')+Math.abs(cut).toFixed(1)+'%';
  $('kCut').className = 'text-2xl font-semibold '+(cut>0?'text-emerald-400':'text-rose-400');
  $('kCutNote').textContent = base.cost.toFixed(1)+' → '+prop.cost.toFixed(1)+' ('+prop.name+')';
  // main chart
  $('mainTitle').textContent = '24-hour plan: '+(d.mode==='replay' ? (d.city==='dc'?'Washington, D.C., ':'Seoul, ')+d.title : d.title);
  const ds = d.day;
  $('dayStats').textContent = ds.hours ? `Day (${ds.hours} h): actual ${fmt(ds.actual)} rentals | shortfall ${fmt(ds.short_base)} → ${fmt(ds.short_plan)} | AvgCost ${ds.cost_base} → ${ds.cost_plan}` :
     `Plan total ${fmt(d.plan_total)} vs forecast ${fmt(d.base_total)} rentals`;
  const sel = H.map((_,h) => h===d.hr ? d.plan_preds[h] : null);
  const data = {labels:H, datasets:[
    {label:'XGBoost forecast'+(d.base==='refit'?' (refitted)':''), data:d.base_preds, borderColor:'#38bdf8', backgroundColor:'#38bdf8', tension:.3, pointRadius:0, borderWidth:2, spanGaps:false},
    {label:'Recommended plan (+ buffer)', data:d.plan_preds, borderColor:'#10b981', backgroundColor:'rgba(16,185,129,.15)', borderDash:[6,4], tension:.3, pointRadius:0, borderWidth:2, fill:'-1'},
    {label:'Actual demand', data:d.actual, borderColor:'#fbbf24', backgroundColor:'#fbbf24', showLine:false, pointRadius:3, hidden:d.mode!=='replay'},
    {label:'Selected hour', data:sel, borderColor:'#f43f5e', backgroundColor:'#f43f5e', showLine:false, pointRadius:6}]};
  if (!cMain){ cMain = new Chart($('cMain'), {type:'line', data, options:{maintainAspectRatio:false, interaction:{mode:'index', intersect:false},
      scales:{y:{beginAtZero:true, title:{display:true, text:'Rentals per hour'}}}, plugins:{legend:{labels:{boxWidth:12}}}}}); }
  else { cMain.data = data; cMain.update('none'); }
  // table
  $('tblNote').textContent = (d.city==='dc'?'2,562 hours, 12 Sep-31 Dec 2012':'1,204 hours, 24 Sep-30 Nov 2018')+' | r = '+d.r.toFixed(2);
  $('tbl').innerHTML = T.map(x => `<tr class="border-b border-slate-800 ${x.name===d.best?'best':''}">
     <td class="py-1">${x.name}${x.name===d.best?' <span class="text-emerald-400 text-xs">lowest</span>':''}</td><td class="text-xs text-slate-400">${x.group}</td>
     <td class="text-right">${x.cost.toFixed(1)}</td><td class="text-right ${x.vs_base<0?'text-emerald-400':(x.vs_base>0?'text-rose-400':'')}">${x.vs_base>0?'+':''}${x.vs_base.toFixed(1)}%</td>
     <td class="text-right">${x.cov}%</td><td class="text-right ${d.city==='seoul'?'hidden':''}">${x.peak_under===undefined?'-':x.peak_under+'%'}</td></tr>`).join('');
  drawCurve(d.r); drawSub();
}

function drawCurve(r){
  const c = S[st.city], colors = {'Seasonal Naive':'#f97316','Hour x Weekday Avg':'#a78bfa','Linear Regression':'#94a3b8','Random Forest':'#22d3ee','XGBoost':'#38bdf8'};
  const sets = Object.entries(c.curves).map(([n,v]) => ({label:n, data:v.map((y,i)=>({x:c.r_grid[i], y})), borderColor:colors[n], pointRadius:0, borderWidth:n==='XGBoost'||n==='Seasonal Naive'?2.5:1.2, hidden:n==='Hour x Weekday Avg'}));
  const ymax = Math.max(...Object.entries(c.curves).filter(([n])=>n!=='Hour x Weekday Avg').map(([,v])=>Math.max(...v)));
  sets.push({label:'current r', data:[{x:r,y:0},{x:r,y:ymax}], borderColor:'#10b981', borderDash:[4,4], pointRadius:0, borderWidth:1.5});
  if (c.rstar) sets.push({label:'r* ≈ '+c.rstar.toFixed(2), data:[{x:c.rstar,y:0},{x:c.rstar,y:ymax}], borderColor:'#f43f5e', borderDash:[2,3], pointRadius:0, borderWidth:1.5});
  $('curveNote').textContent = c.rstar ? `Re-scoring frozen models: Seasonal Naive becomes cheaper than XGBoost above r* ≈ ${c.rstar.toFixed(2)} on this test block, but r* is unstable (0.5 to 4 across windows).`
                                       : 'Seoul: no crossover at any r. XGBoost stays cheaper; its lead narrows as r grows.';
  const opt = {maintainAspectRatio:false, animation:false, parsing:false, scales:{x:{type:'logarithmic', min:0.25, max:10, title:{display:true,text:'cost ratio r (log scale)'},
      ticks:{callback:v => [0.25,0.5,1,2,5,10].includes(+v) ? v : ''}}, y:{beginAtZero:true, max:Math.ceil(ymax/50)*50, title:{display:true,text:'AvgCost(r)'}}}, plugins:{legend:{labels:{boxWidth:10}}}};
  if (cCurve) cCurve.destroy(); cCurve = new Chart($('cCurve'), {type:'line', data:{datasets:sets}, options:opt});
}

function drawSub(){
  if (cSub) cSub.destroy();
  if (st.sub==='buf'){
    $('subTitle').textContent = 'Hour-of-day buffer (Fig. 4.10)';
    const b = last.rule_applied ? last.buffers : last.buffers_raw;
    cSub = new Chart($('cSub'), {type:'bar', data:{labels:H, datasets:[{label:last.rule_applied?'buffer at r = '+last.r.toFixed(2):'buffer the rule switches off (r ≤ 1)',
       data:b, backgroundColor:H.map((_,h)=>h===last.hr?'#f43f5e':(last.rule_applied?'#10b981':'#475569'))}]},
       options:{indexAxis:'x', maintainAspectRatio:false, animation:false, scales:{y:{title:{display:true,text:'rentals added'}}}}});
  } else {
    $('subTitle').textContent = 'XGBoost feature importance (gain, %)';
    cSub = new Chart($('cSub'), {type:'bar', data:{labels:S.importance.map(x=>x[0]), datasets:[{label:'Washington XGBoost', data:S.importance.map(x=>x[1]), backgroundColor:'#38bdf8'}]},
       options:{indexAxis:'y', maintainAspectRatio:false, animation:false}});
  }
}

function drawStatic(){
  const g = S.gbfs, shade = {id:'shade', beforeDraw(ch){ const {ctx, chartArea:a, scales:{x}} = ch; ctx.save(); ctx.fillStyle='rgba(148,163,184,.12)';
     [[7,9],[17,19]].forEach(([s,e]) => { const x0 = x.getPixelForValue(s-0.5), x1 = x.getPixelForValue(e-0.5); ctx.fillRect(x0, a.top, x1-x0, a.bottom-a.top); }); ctx.restore(); }};
  cGbfs = new Chart($('cGbfs'), {type:'line', plugins:[shade], data:{labels:[...Array(24).keys()], datasets:[
     {label:'Empty, weekday', data:g.Weekday.empty, borderColor:'#38bdf8', pointRadius:2},
     {label:'Empty, weekend', data:g.Weekend.empty, borderColor:'#f97316', pointRadius:2},
     {label:'Full, weekday', data:g.Weekday.full, borderColor:'#38bdf8', borderDash:[4,3], pointRadius:0, hidden:true},
     {label:'Full, weekend', data:g.Weekend.full, borderColor:'#f97316', borderDash:[4,3], pointRadius:0, hidden:true}]},
     options:{maintainAspectRatio:false, scales:{x:{type:'linear', min:0, max:23, title:{display:true,text:'hour of day (Washington time)'}}, y:{title:{display:true,text:'% of stations'}}}, plugins:{legend:{labels:{boxWidth:10}}}}});
  $('roll').innerHTML = S.rolling.map(w => `<tr class="border-b border-slate-800"><td class="py-1">${w.window}</td><td class="text-right">${w.standard.toFixed(1)}</td><td class="text-right">${w.refit.toFixed(1)}</td><td class="text-right">${w.rerank.toFixed(1)}</td>
     <td class="text-right text-emerald-400">${w.buffer.toFixed(1)}</td><td class="text-right">${w.vs_std}%</td><td class="text-right ${w.vs_refit>0?'text-rose-400':''}">${w.vs_refit>0?'+':''}${w.vs_refit}%</td></tr>`).join('');
}

group('cityBtns','city', v => { st.city=v; cityUI(); refresh(); });
group('baseBtns','base', v => { st.base=v; refresh(); });
group('modeBtns','mode', v => { st.mode=v; cityUI(); refresh(); });
group('subBtns','sub', v => { st.sub=v; drawSub(); });
document.querySelectorAll('#presets button').forEach(b => b.onclick = () => {
  const p = JSON.parse(b.dataset.p); $('hr').value = p.hr; $('r').value = p.r;
  if (p.workingday!==undefined) $('workingday').value = p.workingday; else $('workingday').value = 1;
  if (st.mode==='replay' && st.city==='dc') $('date').value = p.workingday===0 ? '2012-10-13' : '2012-10-10';
  onSlide(); });
['r','hr','temp_c','hum_pct','wind_kmh'].forEach(id => $(id).addEventListener('input', onSlide));
['date','month','workingday','weathersit','rule'].forEach(id => $(id).addEventListener('change', onSlide));
labels(); cityUI(); drawStatic(); refresh();
</script>
</body></html>
"""

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 8000)), debug=False)
