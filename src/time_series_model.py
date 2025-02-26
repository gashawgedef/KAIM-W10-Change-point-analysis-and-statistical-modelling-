import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import os  # Added for directory handling


def fit_arima(df, order=(1, 1, 1)):
    # Ensure df['Price'] is used as the time series data
    model = ARIMA(df["Price"], order=order)
    results = model.fit()

    # Plot fitted values using the index (Date) instead of df['Date']
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df["Price"], label="Actual Price", alpha=0.5)
    plt.plot(df.index, results.fittedvalues, label="Fitted Price", color="orange")
    plt.title("ARIMA Model Fit")
    plt.xlabel("Date")
    plt.ylabel("Price (USD/barrel)")
    plt.legend()
    plt.grid(True)
    # Ensure the output directory exists
    os.makedirs("../docs", exist_ok=True)
    plt.savefig("../docs/arima_fit.png")
    plt.close()

    return results


if __name__ == "__main__":
    df = pd.read_csv("../data/cleaned_brent_oil_prices.csv", parse_dates=["Date"])
    df.set_index("Date", inplace=True)  # This makes 'Date' the index
    results = fit_arima(df)
    print("ARIMA model fitted. Summary:")
    print(results.summary())
    print("Plot saved to '../docs/arima_fit.png'")
