import pandas as pd
import numpy as np
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.statespace.sarimax import SARIMAX
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import os


def load_additional_data():
    # Placeholder: Replace with actual data (e.g., GDP, USD)
    data = pd.read_csv("../data/economic_indicators.csv", parse_dates=["Date"])
    return data.set_index("Date")


def fit_var(df):
    model = VAR(df)
    results = model.fit(maxlags=15, ic="aic")
    os.makedirs("../docs", exist_ok=True)
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df["OilPrice"], label="Actual Oil Price", alpha=0.5)
    plt.plot(
        df.index, results.fittedvalues["OilPrice"], label="VAR Fitted", color="orange"
    )
    plt.title("VAR Model Fit for Brent Oil Prices")
    plt.xlabel("Date")
    plt.ylabel("Price (USD/barrel)")
    plt.legend()
    plt.grid(True)
    plt.savefig("../docs/var_fit.png")
    plt.close()
    return results


def fit_markov_switching_arima(df):
    model = SARIMAX(df["Price"], order=(1, 1, 1), seasonal_order=(1, 1, 1, 2))
    results = model.fit()
    os.makedirs("../docs", exist_ok=True)
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df["Price"], label="Actual Price", alpha=0.5)
    plt.plot(
        df.index,
        results.fittedvalues,
        label="Markov-Switching ARIMA Fitted",
        color="orange",
    )
    plt.title("Markov-Switching ARIMA Fit")
    plt.xlabel("Date")
    plt.ylabel("Price (USD/barrel)")
    plt.legend()
    plt.grid(True)
    plt.savefig("../docs/markov_arima_fit.png")
    plt.close()
    return results


def fit_lstm(df, look_back=30):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_prices = scaler.fit_transform(df[["Price"]])
    X, y = [], []
    for i in range(len(scaled_prices) - look_back):
        X.append(scaled_prices[i : (i + look_back), 0])
        y.append(scaled_prices[i + look_back, 0])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    model = Sequential()
    model.add(LSTM(50, input_shape=(look_back, 1)))
    model.add(Dense(1))
    model.compile(loss="mean_squared_error", optimizer="adam")
    model.fit(X, y, epochs=10, batch_size=1, verbose=0)
    predictions = model.predict(X)
    predictions = scaler.inverse_transform(predictions)
    os.makedirs("../docs", exist_ok=True)
    plt.figure(figsize=(12, 6))
    plt.plot(
        df.index[look_back:], df["Price"][look_back:], label="Actual Price", alpha=0.5
    )
    plt.plot(df.index[look_back:], predictions, label="LSTM Predicted", color="orange")
    plt.title("LSTM Model Fit")
    plt.xlabel("Date")
    plt.ylabel("Price (USD/barrel)")
    plt.legend()
    plt.grid(True)
    plt.savefig("../docs/lstm_fit.png")
    plt.close()
    return model, predictions


if __name__ == "__main__":
    df = pd.read_csv("../data/cleaned_brent_oil_prices.csv", parse_dates=["Date"])
    df = df.set_index("Date")
    ms_arima_results = fit_markov_switching_arima(df)
    print("Markov-Switching ARIMA fitted. Plot saved to '../docs/markov_arima_fit.png'")
    lstm_model, lstm_preds = fit_lstm(df)
    print("LSTM model fitted. Plot saved to '../docs/lstm_fit.png'")
