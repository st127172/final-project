# Beyond RMSE - BikeFlow AI dashboard

Interactive dashboard for "Beyond RMSE: A Cost-Sensitivity Analysis of Bike-Share Demand Forecasting"
(AST02.10, AIT; Kobdej Manonai st127172 and Lwin Moe Aung st127370).

The app re-scores held-out test forecasts for any cost ratio r = Cu/Co and adds the hour-of-day buffer
(the r/(1+r) quantile of validation errors for each hour), exactly as in the final report.

## Files
- `build_model.py` trains all models offline (same protocol as `Beyond_RMSE_Code.ipynb`) and writes `model_bundle.joblib`.
- `app.py` Flask app + JSON API (`POST /api/plan`, `GET /api/static`, `GET /api/health`).
- `Final presentation/` must contain `hour.csv`, `SeoulBikeData.csv`, `gbfs_summary.csv`.

## Run
    pip install -r requirements.txt
    python build_model.py          # about 1.5 min on a laptop; add --no-armb to skip Arm B
    python app.py                  # http://127.0.0.1:8000

Production (EC2): `gunicorn --workers 2 --bind 127.0.0.1:8000 app:app` behind Nginx.

## Checked against the report (main split, r = 5)
XGBoost RMSE 73.8 / MAE 47.2; AvgCost: XGBoost 182.0, Seasonal Naive 174.0, refitted XGBoost 125.6,
XGBoost + hourly buffer 93.3 (coverage 77%), refitted XGBoost + buffer 92.0 (84%), Arm B 123.6, refitted Arm B 104.7;
Seoul XGBoost 574.2 -> 305.7 with the buffer. Crossover r* = 3.87.
