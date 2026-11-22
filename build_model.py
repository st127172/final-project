"""
build_model.py - offline training for the BikeFlow AI dashboard (Beyond RMSE project).

Reproduces the protocol of the final report / Beyond_RMSE_Code.ipynb:
  * lag_168h looked up by timestamp (not by shifting rows)
  * chronological 70/15/15 split, tuning on the validation block only
  * five Arm A models, refitted XGBoost, hour-of-day buffer, Arm B (cost-trained XGBoost)
  * Seoul replication (Seasonal Naive + XGBoost)

Everything the web app needs is saved to model_bundle.joblib, so app.py never retrains.
Usage:
    python build_model.py                  # full build (about 5-10 min on a laptop)
    python build_model.py --no-armb        # skip Arm B (faster, e.g. on a t3.micro)
    python build_model.py --data "Final presentation"
"""
import argparse
import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

FEATS = ["season", "yr", "mnth", "hr", "holiday", "weekday", "workingday", "weathersit",
         "temp", "atemp", "hum", "windspeed", "lag_168h"]
FEATS_SEOUL = ["hr", "temp", "hum", "wind", "vis", "dew", "solar", "rain", "snow",
               "season", "holiday", "weekday", "mnth", "lag_168h"]
RF_GRID = [dict(max_depth=md, min_samples_leaf=ml) for md in (None, 20) for ml in (1, 3)]
XG_GRID = [dict(n_estimators=ne, max_depth=md, learning_rate=lr)
           for ne in (300, 600) for md in (4, 6) for lr in (0.05, 0.1)]
ARMB_GRID = [dict(max_depth=md, learning_rate=lr, min_child_weight=mcw)
             for md in (4, 6) for lr in (0.05, 0.1) for mcw in (1, 10)]
ARMB_RATIOS = [1, 2, 3, 5, 10]


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - p) ** 2)))


def avg_cost(y, p, r):
    e = np.asarray(y) - np.asarray(p)
    return float(r * np.maximum(e, 0).mean() + np.maximum(-e, 0).mean())


def tune_rf(Xt, yt, Xv, yv):
    best = None
    for g in RF_GRID:
        m = RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=-1, **g).fit(Xt, yt)
        s = rmse(yv, np.clip(m.predict(Xv), 0, None))
        if best is None or s < best[0]:
            best = (s, g, m)
    return best[2], best[1]


def tune_xgb(Xt, yt, Xv, yv):
    best = None
    for g in XG_GRID:
        m = XGBRegressor(random_state=0, **g).fit(Xt, yt)
        s = rmse(yv, np.clip(m.predict(Xv), 0, None))
        if best is None or s < best[0]:
            best = (s, g, m)
    return best[2], best[1]


def build_dc(path, with_armb):
    df = pd.read_csv(path, parse_dates=["dteday"])
    df["ts"] = df["dteday"] + pd.to_timedelta(df["hr"], unit="h")
    df = df.sort_values("ts").reset_index(drop=True)
    count_at = df.set_index("ts")["cnt"]
    df["lag_168h"] = (df["ts"] - pd.Timedelta(hours=168)).map(count_at)
    d = df.dropna(subset=["lag_168h"]).reset_index(drop=True)

    n = len(d); i1, i2 = int(n * 0.70), int(n * 0.85)
    tr, va, te = d.iloc[:i1], d.iloc[i1:i2], d.iloc[i2:]
    Xtr, ytr = tr[FEATS], tr["cnt"]
    Xva, yva = va[FEATS], va["cnt"].values
    Xte, yte = te[FEATS], te["cnt"].values
    print(f"[DC] rows {len(df)} -> {n}; train {len(tr)}, val {len(va)}, test {len(te)}")

    t = time.time()
    rf, rf_best = tune_rf(Xtr, ytr, Xva, yva)
    xg, xg_best = tune_xgb(Xtr, ytr, Xva, yva)
    print(f"[DC] tuned in {time.time() - t:.0f}s  RF {rf_best}  XGB {xg_best}")

    hw = tr.groupby(["hr", "weekday"])["cnt"].mean(); gm = tr["cnt"].mean()
    lin = LinearRegression().fit(Xtr, ytr)

    def predict_all(X):
        p = {"Seasonal Naive": X["lag_168h"].values.astype(float),
             "Hour x Weekday Avg": np.array([hw.get((a, b), gm) for a, b in zip(X["hr"], X["weekday"])]),
             "Linear Regression": lin.predict(X),
             "Random Forest": rf.predict(X),
             "XGBoost": xg.predict(X)}
        return {k: np.clip(v, 0, None) for k, v in p.items()}

    P, PV = predict_all(Xte), predict_all(Xva)
    trva = pd.concat([tr, va])
    xg_refit = XGBRegressor(random_state=0, **xg_best).fit(trva[FEATS], trva["cnt"])
    P["XGBoost refitted"] = np.clip(xg_refit.predict(Xte), 0, None)

    armb = {}
    if with_armb:
        t = time.time()
        for r in ARMB_RATIOS:
            a = r / (1 + r); best = None
            for g in ARMB_GRID:
                m = XGBRegressor(objective="reg:quantileerror", quantile_alpha=a, random_state=0,
                                 n_estimators=3000, early_stopping_rounds=100, eval_metric="quantile", **g)
                m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
                s = avg_cost(yva, np.clip(m.predict(Xva), 0, None), r)
                if best is None or s < best[0]:
                    best = (s, g, m)
            g, m = best[1], best[2]
            mr = XGBRegressor(objective="reg:quantileerror", quantile_alpha=a, random_state=0,
                              n_estimators=m.best_iteration + 1, **g).fit(trva[FEATS], trva["cnt"])
            armb[r] = {"Arm B": np.clip(m.predict(Xte), 0, None).astype(np.float32),
                       "Arm B refitted": np.clip(mr.predict(Xte), 0, None).astype(np.float32)}
            print(f"[DC] Arm B r={r}: cost {avg_cost(yte, armb[r]['Arm B'], r):.1f}")
        print(f"[DC] Arm B done in {time.time() - t:.0f}s")

    # typical lag for what-if queries without a real date: median cnt by (season, workingday, hr)
    lag_med = trva.groupby(["season", "workingday", "hr"])["cnt"].median().round().to_dict()
    # feels-like temperature from temperature (normalized units), fitted on train + val
    a_coef = np.polyfit(trva["temp"], trva["atemp"], 1)
    imp = dict(zip(FEATS, xg.feature_importances_.round(4).tolist()))

    return {
        "feats": FEATS, "xgb": xg, "xgb_refit": xg_refit, "xgb_best": xg_best, "rf_best": rf_best,
        "y_test": yte.astype(np.float32),
        "preds_test": {k: v.astype(np.float32) for k, v in P.items()},
        "armb_test": armb,
        "val_err_xgb": (yva - PV["XGBoost"]).astype(np.float32),
        "val_err_all": {k: (yva - v).astype(np.float32) for k, v in PV.items()},
        "hr_val": va["hr"].values.astype(np.int8), "hr_test": te["hr"].values.astype(np.int8),
        "workingday_test": te["workingday"].values.astype(np.int8),
        "ts_test": te["ts"].dt.strftime("%Y-%m-%d %H:%M").values,
        "X_test": te[FEATS].reset_index(drop=True),
        "test_period": [str(te["ts"].min()), str(te["ts"].max())],
        "val_period": [str(va["ts"].min()), str(va["ts"].max())],
        "n": {"raw": len(df), "after_lag": n, "train": len(tr), "val": len(va), "test": len(te)},
        "lag_med": lag_med, "atemp_coef": a_coef.tolist(), "importance": imp,
    }


def build_seoul(path):
    s = pd.read_csv(path, encoding="latin1")
    s.columns = ["date", "cnt", "hr", "temp", "hum", "wind", "vis", "dew", "solar", "rain", "snow",
                 "season", "holiday", "functioning"]
    s["ts"] = pd.to_datetime(s["date"], dayfirst=True) + pd.to_timedelta(s["hr"], unit="h")
    s = s.sort_values("ts").reset_index(drop=True)
    s["weekday"] = s["ts"].dt.dayofweek
    s["mnth"] = s["ts"].dt.month
    s["season"] = s["season"].map({"Winter": 0, "Spring": 1, "Summer": 2, "Autumn": 3})
    s["holiday"] = (s["holiday"] == "Holiday").astype(int)
    open_h = s[s["functioning"] == "Yes"]
    s["lag_168h"] = (s["ts"] - pd.Timedelta(hours=168)).map(open_h.set_index("ts")["cnt"])
    sd = s[(s["functioning"] == "Yes") & s["lag_168h"].notna()].reset_index(drop=True)
    n = len(sd); j1, j2 = int(n * 0.70), int(n * 0.85)
    tr, va, te = sd.iloc[:j1], sd.iloc[j1:j2], sd.iloc[j2:]
    yv, yt = va["cnt"].values, te["cnt"].values
    xg, best = tune_xgb(tr[FEATS_SEOUL], tr["cnt"], va[FEATS_SEOUL], yv)
    pv = {"Seasonal Naive": va["lag_168h"].values.astype(float),
          "XGBoost": np.clip(xg.predict(va[FEATS_SEOUL]), 0, None)}
    pt = {"Seasonal Naive": te["lag_168h"].values.astype(float),
          "XGBoost": np.clip(xg.predict(te[FEATS_SEOUL]), 0, None)}
    print(f"[Seoul] rows used {n}; XGB {best}; RMSE {rmse(yt, pt['XGBoost']):.1f}")
    return {
        "xgb_best": best, "y_test": yt.astype(np.float32),
        "preds_test": {k: v.astype(np.float32) for k, v in pt.items()},
        "val_err_all": {k: (yv - v).astype(np.float32) for k, v in pv.items()},
        "val_err_xgb": (yv - pv["XGBoost"]).astype(np.float32),
        "hr_val": va["hr"].values.astype(np.int8), "hr_test": te["hr"].values.astype(np.int8),
        "ts_test": te["ts"].dt.strftime("%Y-%m-%d %H:%M").values,
        "test_period": [str(te["ts"].min()), str(te["ts"].max())],
        "n": {"raw": len(s), "used": n, "train": len(tr), "val": len(va), "test": len(te)},
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="Final presentation", help="folder with the three CSV files")
    ap.add_argument("--out", default="model_bundle.joblib")
    ap.add_argument("--no-armb", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    bundle = {"dc": build_dc(os.path.join(args.data, "hour.csv"), not args.no_armb),
              "seoul": build_seoul(os.path.join(args.data, "SeoulBikeData.csv"))}
    g = pd.read_csv(os.path.join(args.data, "gbfs_summary.csv"))
    bundle["gbfs"] = {day: {"empty": sub.sort_values("hour")["empty"].tolist(),
                            "full": sub.sort_values("hour")["full"].tolist()}
                      for day, sub in g.groupby("day_type")}
    joblib.dump(bundle, args.out, compress=3)
    print(f"saved {args.out} ({os.path.getsize(args.out) / 1e6:.1f} MB) in {time.time() - t0:.0f}s")
