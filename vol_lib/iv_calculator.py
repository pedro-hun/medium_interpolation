import datetime
import warnings
from typing import Callable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from dateutil.relativedelta import relativedelta
from scipy.optimize import brentq
from scipy.stats import norm

# --- Configuration ---
TICKER_SYMBOL: str = "AAPL" # Default ticker, can be overridden by environment variable
RISK_FREE_RATE: float = 0.15 # Example risk-free rate (annualized)
# Filter thresholds for option data cleaning
MIN_VOLUME: int = 1
MAX_REL_SPREAD: float = 0.10 # Maximum relative spread (Spread / MidPrice)
MIN_DAYS_TO_EXPIRY: int = 1 # Minimum days to expiry to include
MIN_IV: float = 0.01 # Minimum plausible IV
MAX_IV: float = 2.00 # Maximum plausible IV
# Interpolation Grid Resolution
N_STRIKES_GRID: int = 100
N_EXPIRIES_GRID: int = 100
INTERPOLATION_METHOD: str = 'cubic' # 'linear' or 'cubic'

# --- Black-Scholes Implementation ---
def black_scholes_price(
    S: float, K: float, T: float, r: float, sigma: float, option_type: str
) -> float:
    """
    Calculates the Black-Scholes price for a European option.

    Args:
        S: Current stock price.
        K: Option strike price.
        T: Time to expiration in years.
        r: Risk-free interest rate (annualized).
        sigma: Volatility of the underlying stock (annualized).
        option_type: 'call' or 'put'.

    Returns:
        The Black-Scholes price of the option. Returns NaN if inputs are invalid.
    """
    discount_factor = 1 / (1+r)**T
    if sigma <= 0 or T <= 0:
        # Return intrinsic value for zero time/volatility if needed, or NaN
        if option_type == "call":
            return max(0.0, S - K * discount_factor) # Approximation using discounted K
        elif option_type == "put":
            return max(0.0, K * discount_factor - S)
        else:
             return np.nan # Should not happen with valid type check
        # Or simply return NaN as BS is not well-defined
        # return np.nan

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    try:
        if option_type == "call":
            price = S * norm.cdf(d1) - K * discount_factor * norm.cdf(d2)
        elif option_type == "put":
            price = K * discount_factor * norm.cdf(-d2) - S * norm.cdf(-d1)
        else:
            # This case should ideally be handled before calling
            # raise ValueError("option_type must be 'call' or 'put'")
             return np.nan
    except OverflowError:
         return np.nan # Handle potential overflows with extreme inputs

    # Ensure price is not negative (can happen with numerical instability)
    return max(0.0, price)


def implied_volatility(
    S: float, K: float, T: float, r: float, market_price: float, option_type: str,
    low_vol: float = 1e-4, high_vol: float = 4.0, tol: float = 1e-6
) -> float:
    """
    Calculates the implied volatility using Brent's root-finding method.

    Args:
        S: Current stock price.
        K: Option strike price.
        T: Time to expiration in years.
        r: Risk-free interest rate (annualized).
        market_price: The observed market price of the option.
        option_type: 'call' or 'put'.
        low_vol: Lower bound for volatility search.
        high_vol: Upper bound for volatility search.
        tol: Tolerance for the root-finding algorithm.

    Returns:
        The implied volatility (annualized), or np.nan if calculation fails or
        market price violates arbitrage bounds.
    """
    if T <= 0 or market_price <= 0:
        return np.nan
    if option_type not in ['call', 'put']:
         return np.nan # Invalid option type

    discount_factor = 1 / (1+r)**T
    # Check for arbitrage violations (minimum price)
    if option_type == 'call' and market_price < max(0.0, S - K * discount_factor - tol):
         print(f"Price violation: Call price {market_price:.4f} < Intrinsic {max(0.0, S - K * discount_factor):.4f}")
         return np.nan
    if option_type == 'put' and market_price < max(0.0, K * discount_factor - S - tol):
         print(f"Price violation: Put price {market_price:.4f} < Intrinsic {max(0.0, K * discount_factor - S):.4f}")
         return np.nan

     # Check maximum price bounds
    if option_type == 'call' and market_price > S: # Call price cannot exceed stock price
        print(f"Price violation: Call price {market_price:.4f} > Stock Price {S:.4f}")
        return np.nan
    # Put price cannot exceed discounted strike (more accurately PV(K))
    if option_type == 'put' and market_price > K * discount_factor:
        #print(f"Price violation: Put price {market_price:.4f} > PV(K) {K * np.exp(-r*T):.4f}")
        return np.nan


    # Define the objective function for the root finder
    def objective_func(sigma: float) -> float:
        # Return large value for invalid sigma to guide solver
        if sigma <= 0:
            return 1e10
        try:
            model_price = black_scholes_price(S, K, T, r, sigma, option_type)
            # Check if model price is NaN (can happen from black_scholes_price)
            if np.isnan(model_price):
                 return 1e11 # Indicate error
            return model_price - market_price
        except (ValueError, OverflowError):
            return 1e12 # Indicate numerical error

    # Attempt to bracket the root
    try:
        f_low = objective_func(low_vol)
        f_high = objective_func(high_vol)

        # Check if objective func returned error indicators
        if f_low > 1e9 or f_high > 1e9:
            #print(f"Objective function error at bounds for K={K}, T={T:.4f}")
            return np.nan

        # Check if market price is outside the range achievable by BS model within vol bounds
        # Vega is positive, so BS price is monotonic with vol
        price_at_low_vol = black_scholes_price(S, K, T, r, low_vol, option_type)
        price_at_high_vol = black_scholes_price(S, K, T, r, high_vol, option_type)

        # Handle NaN results from BS pricing at bounds
        if np.isnan(price_at_low_vol) or np.isnan(price_at_high_vol):
            #print(f"BS price calculation failed at bounds for K={K}, T={T:.4f}")
            return np.nan

        # If market price is below the price at min vol (and above intrinsic), IV might be < low_vol
        if market_price < price_at_low_vol - tol:
            #print(f"Market price {market_price:.4f} below price at low vol bound {price_at_low_vol:.4f} for K={K}, T={T:.4f}")
            # Could potentially return low_vol or slightly less, but nan is safer
            return np.nan
        # If market price is above the price at max vol
        if market_price > price_at_high_vol + tol:
             #print(f"Market price {market_price:.4f} above price at high vol bound {price_at_high_vol:.4f} for K={K}, T={T:.4f}")
             # Could return high_vol, but nan is safer
             return np.nan


        # Check if signs are different (required for brentq)
        if np.sign(f_low) == np.sign(f_high):
            # Try adjusting bounds slightly if signs are same and price is within range
            # This can happen if market price is very close to price at bound
             if abs(f_low) < abs(f_high): # Market price closer to low vol price
                  high_vol_adj = high_vol * 1.5
                  f_high_adj = objective_func(high_vol_adj)
                  if np.sign(f_low) != np.sign(f_high_adj):
                      high_vol = high_vol_adj
                  else:
                       #print(f"Cannot bracket root (sign issue) K={K}, T={T:.4f}. f({low_vol:.4f})={f_low:.4e}, f({high_vol:.4f})={f_high:.4e}")
                       return np.nan

             else: # Market price closer to high vol price
                  low_vol_adj = low_vol * 0.5
                  f_low_adj = objective_func(low_vol_adj)
                  if np.sign(f_low_adj) != np.sign(f_high):
                      low_vol = low_vol_adj
                  else:
                      #print(f"Cannot bracket root (sign issue) K={K}, T={T:.4f}. f({low_vol:.4f})={f_low:.4e}, f({high_vol:.4f})={f_high:.4e}")
                      return np.nan
                  

    except (ValueError, OverflowError):
         #print(f"Numerical error during bound check K={K}, T={T:.4f}")
         return np.nan


    # Use Brent's method for root finding
    try:
        iv = brentq(objective_func, low_vol, high_vol, xtol=tol, rtol=tol)
    except ValueError:
        # This typically happens if f(a)*f(b) > 0 despite checks, or other issues
        print(f"Brentq ValueError K={K}, T={T:.4f}. Check bounds and objective function.")
        return np.nan
    except Exception as e:
        # Catch other potential numerical errors during optimization
        print(f"Unexpected error in brentq K={K}, T={T:.4f}: {e}")
        return np.nan

    # Final check: ensure calculated IV is within reasonable bounds
    if not (low_vol <= iv <= high_vol):
        # This might indicate convergence issues or extreme market conditions
        #print(f"Warning: Calculated IV {iv:.4f} outside search range [{low_vol:.4f}, {high_vol:.4f}] for K={K}, T={T:.4f}")
        # Returning NaN as the result might be unreliable
        return np.nan

    # Optional: Verify the calculated IV produces the market price approximately
    # final_price = black_scholes_price(S, K, T, r, iv, option_type)
    # if abs(final_price - market_price) > market_price * 0.01 + 0.01: # Check within 1% relative or $0.01 absolute diff
    #     print(f"Warning: IV {iv:.4f} yields price {final_price:.4f}, differs significantly from market {market_price:.4f} for K={K}, T={T:.4f}")
    #     return np.nan

    return iv


def calculate_iv_for_chain(
    option_chain_df: pd.DataFrame,
    r: float,
    min_volume: int = MIN_VOLUME,
    max_rel_spread: float = MAX_REL_SPREAD,
    min_days_expiry: int = MIN_DAYS_TO_EXPIRY
) -> pd.DataFrame:
    """
    Calculates Implied Volatility for a filtered option chain DataFrame.

    Args:
        option_chain_df: DataFrame from fetch_option_chain.
        r: Risk-free interest rate.
        min_volume: Minimum trading volume to include the option.
        max_rel_spread: Maximum relative bid-ask spread allowed.
        min_days_expiry: Minimum number of days to expiry.

    Returns:
        DataFrame with 'TimeToExpiry', 'Strike', 'ImpliedVolatility', 'Type'.
        Filters out options failing criteria or IV calculation.
    """
    df = option_chain_df.copy()

    # Ensure required columns exist
    required_cols = ['bid', 'ask', 'Strike', 'Expiry', 'SpotPrice', 'Type', 'volume']
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        print(f"Error: Missing required columns in input DataFrame: {missing}")
        # Return structure consistent with success, but empty
        return pd.DataFrame(columns=['TimeToExpiry', 'Strike', 'ImpliedVolatility', 'Type'])


    # Calculate Mid Price and Time to Expiry
    df['MidPrice'] = (df['bid'] + df['ask']) / 2.0
    # df['TimeToExpiry'] = (df['Expiry'] - df['FetchDate']).dt.total_seconds() / (365.25 * 24 * 60 * 60) # Accurate calculation including time

    # --- Filtering ---
    # 1. Price and Spread validity
    df = df[df['bid'] > 0]
    df = df[df['ask'] > 0]
    df = df[df['MidPrice'] > 0]
    df['Spread'] = df['ask'] - df['bid']
    df['RelativeSpread'] = df['Spread'] / df['MidPrice']
    df = df[df['RelativeSpread'] <= max_rel_spread]
    df = df[df['RelativeSpread'] >= 0] # Ensure spread is not negative

    # 2. Volume
    df = df[df['volume'] >= min_volume]

    # 3. Time to Expiry
    df = df[df['TimeToExpiry'] * 252 >= min_days_expiry]
    df = df[df['TimeToExpiry'] > 1e-6] # Avoid zero or negative time

    # 4. Drop rows with NaN in critical columns before IV calculation
    critical_cols_iv = ['SpotPrice', 'Strike', 'TimeToExpiry', 'MidPrice', 'Type']
    df.dropna(subset=critical_cols_iv, inplace=True)

    # Filter out options where mid-price is clearly below intrinsic value (arbitrage)
    df['IntrinsicValue'] = np.where(
        df['Type'] == 'call',
        np.maximum(0.0, df['SpotPrice'] - df['Strike'] / ((1+r) ** df['TimeToExpiry'])),
        np.maximum(0.0, df['Strike'] * ((1+r) ** df['TimeToExpiry']) - df['SpotPrice'])
    )
    # Allow a small tolerance for market friction / minor price discrepancies
    df = df[df['MidPrice'] >= df['IntrinsicValue'] - 0.005] # Price should not be significantly below intrinsic

    if df.empty:
         print("Warning: No valid options remaining after initial filtering.")
         return pd.DataFrame(columns=['TimeToExpiry', 'Strike', 'ImpliedVolatility', 'Type'])

    # Calculate IV row-wise (Vectorization is complex due to root finding)
    # Consider using joblib or multiprocessing for large chains if performance is critical
    iv_values = []
    for _, row in df.iterrows():
        iv = implied_volatility(
            S=row['SpotPrice'],
            K=row['Strike'],
            T=row['TimeToExpiry'],
            r=r,
            market_price=row['MidPrice'],
            option_type=row['Type']
        )
        iv_values.append(iv)

    df['ImpliedVolatility'] = iv_values

    # --- Post-IV Filtering ---
    # 1. Remove rows where IV calculation failed (returned NaN)
    df.dropna(subset=['ImpliedVolatility'], inplace=True)

    # 2. Filter based on plausible IV range
    df = df[(df['ImpliedVolatility'] >= MIN_IV) & (df['ImpliedVolatility'] <= MAX_IV)]

    # Select and rename final columns
    result_df = df[['TimeToExpiry', 'Strike', 'ImpliedVolatility', 'Type', 'Forward']].copy()


    return result_df

