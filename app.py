from flask import Flask, jsonify
import pandas as pd
import numpy as np
import os
from src.change_point_detection import detect_change_points
from src.time_series_model import fit_arima

app = Flask(__name__)
df = pd.read_csv("../data/cleaned_brent_oil_prices.csv", parse_dates=["Date"])
df_arima = df.set_index("Date")


@app.route("/api/prices", methods=["GET"])
def get_prices():
    data = df.to_dict(orient="records")
    return jsonify(data)


@app.route("/api/change_points", methods=["GET"])
def get_change_points():
    trace = detect_change_points(df)
    tau_sorted = np.sort(trace["tau"].mean(axis=0))
    change_points = [
        {"date": df["Date"].iloc[int(tau)].isoformat(), "index": int(tau)}
        for tau in tau_sorted
    ]
    return jsonify(change_points)


@app.route("/api/arima", methods=["GET"])
def get_arima():
    results = fit_arima(df_arima)
    fitted = pd.Series(results.fittedvalues, index=df_arima.index).reset_index()
    fitted_data = fitted.rename(
        columns={"Date": "date", "Price": "fitted_price"}
    ).to_dict(orient="records")
    return jsonify(fitted_data)


if __name__ == "__main__":
    os.makedirs("../data", exist_ok=True)
    os.makedirs("../docs", exist_ok=True)
    app.run(debug=True)
