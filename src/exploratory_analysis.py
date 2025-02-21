# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# def plot_price_series(df, output_path):
#     plt.figure(figsize=(12, 6))
#     plt.plot(df['Date'], df['Price'], label='Brent Oil Price', color='blue')
#     plt.title('Brent Oil Prices (1987-2022)')
#     plt.xlabel('Date')
#     plt.ylabel('Price (USD/barrel)')
#     plt.legend()
#     plt.grid(True)
#     # plt.savefig(output_path)
#     plt.close()

# def plot_rolling_stats(df, window=30):
#     rolling_mean = df['Price'].rolling(window=window).mean()
#     rolling_std = df['Price'].rolling(window=window).std()
    
#     plt.figure(figsize=(12, 6))
#     plt.plot(df['Date'], df['Price'], label='Price', alpha=0.5)
#     plt.plot(df['Date'], rolling_mean, label=f'{window}-Day Rolling Mean', color='orange')
#     plt.plot(df['Date'], rolling_std, label=f'{window}-Day Rolling Std', color='red')
#     plt.title(f'Rolling Statistics (Window={window} Days)')
#     plt.xlabel('Date')
#     plt.ylabel('Price (USD/barrel)')
#     plt.legend()
#     plt.grid(True)
#     # plt.savefig('docs/rolling_stats.png')
#     plt.close()

# if __name__ == "__main__":
#     df = pd.read_csv("data/BrentOilPrices.csv", parse_dates=['Date'])
#     # plot_price_series(df, "docs/price_series.png")
#     # plot_rolling_stats(df)
#     print("EDA plots saved in '../docs/'")
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_price_series(df, output_path):
    plt.figure(figsize=(12, 6))
    plt.plot(df['Date'], df['Price'], label='Brent Oil Price', color='blue')
    plt.title('Brent Oil Prices (1987-2022)')
    plt.xlabel('Date')
    plt.ylabel('Price (USD/barrel)')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()

def plot_rolling_stats(df, window=30):
    rolling_mean = df['Price'].rolling(window=window).mean()
    rolling_std = df['Price'].rolling(window=window).std()
    
    plt.figure(figsize=(12, 6))
    plt.plot(df['Date'], df['Price'], label='Price', alpha=0.5)
    plt.plot(df['Date'], rolling_mean, label=f'{window}-Day Rolling Mean', color='orange')
    plt.plot(df['Date'], rolling_std, label=f'{window}-Day Rolling Std', color='red')
    plt.title(f'Rolling Statistics (Window={window} Days)')
    plt.xlabel('Date')
    plt.ylabel('Price (USD/barrel)')
    plt.legend()
    plt.grid(True)
    plt.savefig('../docs/rolling_stats.png')
    plt.close()

if __name__ == "__main__":
    df = pd.read_csv("../data/cleaned_brent_oil_prices.csv", parse_dates=['Date'])
    plot_price_series(df, "../docs/price_series.png")
    plot_rolling_stats(df)
    print("EDA plots saved in '../docs/'")