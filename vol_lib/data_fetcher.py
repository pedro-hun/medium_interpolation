import yfinance as yf
import pandas as pd



def fetch_option_chain(ticker_symbol: str) -> pd.DataFrame:
    """
    Fetches the full option chain for a given ticker using yfinance.

    Args:
        ticker_symbol: The stock ticker symbol (e.g., "AAPL").

    Returns:
        A pandas DataFrame containing the option chain data, including spot price
        and fetch date, or an empty DataFrame if fetching fails or no options exist.
    """
    try:
        ticker = yf.Ticker(ticker_symbol)
        expiry_dates = ticker.options

        if not expiry_dates:
            print(f"No option expiry dates found for {ticker_symbol}.")
            return pd.DataFrame()

        all_options = []
        # Get current spot price efficiently
        hist = ticker.history(period='1d')
        if hist.empty or 'Close' not in hist.columns:
             print(f"Could not fetch current price for {ticker_symbol}.")
             # Fallback or error handling needed here - for now, return empty
             return pd.DataFrame()
        spot_price = hist['Close'].iloc[-1]
        fetch_date = pd.Timestamp.now().normalize() # Use date part only

        for expiry in expiry_dates:
            opts = ticker.option_chain(expiry)
            # Process calls
            calls = opts.calls
            if not calls.empty:
                calls = calls.copy() # Avoid SettingWithCopyWarning
                calls['Type'] = 'call'
                calls['Expiry'] = pd.to_datetime(expiry)
                all_options.append(calls)
            # Process puts
            puts = opts.puts
            if not puts.empty:
                puts = puts.copy() # Avoid SettingWithCopyWarning
                puts['Type'] = 'put'
                puts['Expiry'] = pd.to_datetime(expiry)
                all_options.append(puts)

        if not all_options:
            print(f"No options data found for {ticker_symbol} across all expiries.")
            return pd.DataFrame()

        option_chain_df = pd.concat(all_options, ignore_index=True)

        # Add spot price and fetch date
        option_chain_df['SpotPrice'] = spot_price
        option_chain_df['FetchDate'] = fetch_date

        # Basic data type conversions
        option_chain_df['strike'] = pd.to_numeric(option_chain_df['strike'], errors='coerce')
        option_chain_df['lastPrice'] = pd.to_numeric(option_chain_df['lastPrice'], errors='coerce')
        option_chain_df['bid'] = pd.to_numeric(option_chain_df['bid'], errors='coerce')
        option_chain_df['ask'] = pd.to_numeric(option_chain_df['ask'], errors='coerce')
        option_chain_df['volume'] = pd.to_numeric(option_chain_df['volume'].fillna(0), errors='coerce').fillna(0).astype(int)
        option_chain_df['openInterest'] = pd.to_numeric(option_chain_df['openInterest'].fillna(0), errors='coerce').fillna(0).astype(int)


    except Exception as e:
        print(f"Error fetching option data for {ticker_symbol}: {e}")
        return pd.DataFrame()

    return option_chain_df



# Example Usage (Optional - typically in a main script or notebook)
if __name__ == "__main__":
    ticker = 'SPY' # Example: S&P 500 ETF
    print(f"Fetching options data for {ticker}...")

    options_data = fetch_option_chain(
        ticker
    )

    if options_data is not None:
        print(f"\nSuccessfully fetched data for {ticker}.")
        print(f"Underlying Price: {options_data['SpotPrice']:.2f}")
        print("\nSample Options Data:")
        print(options_data.head())
        print("\nData Columns:")
        print(options_data.info())

        # Further check counts per expiration/type
        print("\nContracts per Expiration Date:")
        print(options_data['expirationDate'].dt.date.value_counts().sort_index())

        print("\nContracts by Type:")
        print(options_data['type'].value_counts())

    else:
        print(f"\nCould not retrieve or filter options data for {ticker}.")