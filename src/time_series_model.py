import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

def fit_arima(df, order=(1, 1, 1)):
    model = ARIMA(df['Price'], order=order)
    results = model.fit()
    
    # Plot fitted values
    plt.figure(figsize=(12, 6))
    plt.plot(df['Date'], df['Price'], label='Actual Price', alpha=0.5)
    plt.plot(df['Date'], results.fittedvalues, label='Fitted Price', color='orange')
    plt.title('ARIMA Model Fit')
    plt.xlabel('Date')
    plt.ylabel('Price (USD/barrel)')
    plt.legend()
    plt.grid(True)
    plt.savefig('../docs/arima_fit.png')
    plt.close()
    
    return results

if __name__ == "__main__":
    df = pd.read_csv("../data/cleaned_brent_oil_prices.csv", parse_dates=['Date'])
    df.set_index('Date', inplace=True)
    results = fit_arima(df)
    print("ARIMA model fitted. Summary:")
    print(results.summary())
    print("Plot saved to '../docs/arima_fit.png'")