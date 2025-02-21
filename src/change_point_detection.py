# import pandas as pd
# import pymc as pm
# import numpy as np
# import matplotlib.pyplot as plt

# def detect_change_points(df, num_change_points=3):
#     prices = df['Price'].values
#     n = len(prices)
    
#     with pm.Model() as model:
#         # Define change point locations (uniform priors)
#         tau = pm.DiscreteUniform('tau', lower=0, upper=n, shape=num_change_points)
#         tau_sorted = pm.Deterministic('tau_sorted', pm.math.sort(tau))
        
#         # Define means for each segment
#         mu = pm.Normal('mu', mu=prices.mean(), sd=prices.std(), shape=num_change_points + 1)
        
#         # Assign means to each time point based on change points
#         idx = np.arange(n)
#         lambda_ = pm.Deterministic('lambda_', pm.math.switch(idx < tau_sorted[0], mu[0],
#                                                              pm.math.switch(idx < tau_sorted[1], mu[1],
#                                                                             pm.math.switch(idx < tau_sorted[2], mu[2], mu[3]))))
        
#         # Likelihood
#         observation = pm.Normal('obs', mu=lambda_, sd=prices.std()/2, observed=prices)
        
#         # Sampling
#         trace = pm.sample(1000, tune=1000, return_inferencedata=False)
    
#     return trace

# def plot_change_points(df, trace):
#     plt.figure(figsize=(12, 6))
#     plt.plot(df['Date'], df['Price'], label='Price', alpha=0.5)
#     for i, tau in enumerate(trace['tau_sorted'].mean(axis=0)):
#         plt.axvline(df['Date'].iloc[int(tau)], color='red', linestyle='--', label=f'Change Point {i+1}')
#     plt.title('Brent Oil Prices with Detected Change Points')
#     plt.xlabel('Date')
#     plt.ylabel('Price (USD/barrel)')
#     plt.legend()
#     plt.grid(True)
#     plt.savefig('../docs/change_points.png')
#     plt.close()

# if __name__ == "__main__":
#     df = pd.read_csv("../data/cleaned_brent_oil_prices.csv", parse_dates=['Date'])
#     trace = detect_change_points(df)
#     plot_change_points(df, trace)
#     print("Change point analysis completed. Plot saved to '../docs/change_points.png'")
import pandas as pd
import pymc3 as pm
import numpy as np
import matplotlib.pyplot as plt

def detect_change_points(df, num_change_points=3):
    prices = df['Price'].values
    n = len(prices)
    
    with pm.Model() as model:
        # Define change point locations (uniform priors)
        tau = pm.DiscreteUniform('tau', lower=0, upper=n, shape=num_change_points)
        tau_sorted = pm.Deterministic('tau_sorted', pm.math.sort(tau))
        
        # Define means for each segment
        mu = pm.Normal('mu', mu=prices.mean(), sd=prices.std(), shape=num_change_points + 1)
        
        # Assign means to each time point based on change points
        idx = np.arange(n)
        lambda_ = pm.Deterministic('lambda_', pm.math.switch(idx < tau_sorted[0], mu[0],
                                                             pm.math.switch(idx < tau_sorted[1], mu[1],
                                                                            pm.math.switch(idx < tau_sorted[2], mu[2], mu[3]))))
        
        # Likelihood
        observation = pm.Normal('obs', mu=lambda_, sd=prices.std()/2, observed=prices)
        
        # Sampling
        trace = pm.sample(1000, tune=1000, return_inferencedata=False)
    
    return trace

def plot_change_points(df, trace):
    plt.figure(figsize=(12, 6))
    plt.plot(df['Date'], df['Price'], label='Price', alpha=0.5)
    for i, tau in enumerate(trace['tau_sorted'].mean(axis=0)):
        plt.axvline(df['Date'].iloc[int(tau)], color='red', linestyle='--', label=f'Change Point {i+1}')
    plt.title('Brent Oil Prices with Detected Change Points')
    plt.xlabel('Date')
    plt.ylabel('Price (USD/barrel)')
    plt.legend()
    plt.grid(True)
    plt.savefig('../docs/change_points.png')
    plt.close()

if __name__ == "__main__":
    df = pd.read_csv("../data/cleaned_brent_oil_prices.csv", parse_dates=['Date'])
    trace = detect_change_points(df)
    plot_change_points(df, trace)
    print("Change point analysis completed. Plot saved to '../docs/change_points.png'")