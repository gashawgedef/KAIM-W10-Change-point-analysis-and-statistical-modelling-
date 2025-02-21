import pandas as pd

def load_and_clean_data(file_path):
    # Load the dataset
    df = pd.read_csv(file_path, parse_dates=['Date'], dayfirst=True)
    
    # Ensure correct column names and types
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%b-%y')
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
    
    # Sort by date and reset index
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Handle missing values (forward fill for simplicity)
    df['Price'] = df['Price'].fillna(method='ffill')
    
    # Check for duplicates
    df = df.drop_duplicates(subset=['Date'])
    
    return df

if __name__ == "__main__":
    df = load_and_clean_data("../data/brent_oil_prices.csv")
    df.to_csv("../data/cleaned_brent_oil_prices.csv", index=False)
    print("Data cleaned and saved to '../data/cleaned_brent_oil_prices.csv'")