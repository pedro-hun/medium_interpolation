# -*- coding: utf-8 -*-
"""
Section 3: Interpolating the Volatility Surface

This script demonstrates how to interpolate sparse implied volatility data
onto a regular grid suitable for visualization and analysis. It uses
scipy.interpolate.griddata and visualizes the result as a 3D surface plot.
"""

import datetime
import pandas as pd
import numpy as np
import yfinance as yf
from scipy.stats import norm
from scipy.optimize import brentq
from scipy.interpolate import griddata
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os # Used for environment variables if needed, although not for yfinance
import data_fetcher as DataFetcher
import iv_calculator as IVCalculator

data_fetcher = DataFetcher
iv_calculator = IVCalculator

# --- Configuration ---
# Set Matplotlib backend for compatibility if needed (e.g., in environments without a GUI)
# import matplotlib
# matplotlib.use('Agg')

TICKER_SYMBOL: str = "AAPL" # Default ticker, can be overridden by environment variable
RISK_FREE_RATE: float = 0.15 # Example risk-free rate (annualized)
# Filter thresholds for option data cleaning
MIN_VOLUME: int = 1
MAX_REL_SPREAD: float = 0.50 # Maximum relative spread (Spread / MidPrice)
MIN_DAYS_TO_EXPIRY: int = 1 # Minimum days to expiry to include
MIN_IV: float = 0.01 # Minimum plausible IV
MAX_IV: float = 2.00 # Maximum plausible IV
# Interpolation Grid Resolution
N_STRIKES_GRID: int = 100
N_EXPIRIES_GRID: int = 100
INTERPOLATION_METHOD: str = 'cubic' # 'linear' or 'cubic'



def interpolate_volatility_surface(
    iv_data: pd.DataFrame,
    method: str = INTERPOLATION_METHOD,
    n_strikes: int = N_STRIKES_GRID,
    n_expiries: int = N_EXPIRIES_GRID
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Interpolates the implied volatility surface onto a regular grid.

    Handles potential errors during interpolation and checks for sufficient data.

    Args:
        iv_data: DataFrame with 'TimeToExpiry', 'Strike', 'ImpliedVolatility'.
                 Should contain cleaned IV data.
        method: Interpolation method for griddata ('linear', 'cubic').
        n_strikes: Number of grid points for the strike dimension.
        n_expiries: Number of grid points for the time-to-expiry dimension.

    Returns:
        A tuple containing:
        - T_grid: Meshgrid for TimeToExpiry (empty if insufficient data/error).
        - K_grid: Meshgrid for Strike (empty if insufficient data/error).
        - iv_surface_interpolated: Interpolated IV values (empty/NaN filled if fails).
    """
    # Validate input DataFrame structure
    required_cols = ['TimeToExpiry', 'Strike', 'ImpliedVolatility']
    if not all(col in iv_data.columns for col in required_cols):
        print(f"Error: Input DataFrame missing required columns: {required_cols}")
        return np.array([]), np.array([]), np.array([])

    # Clean data: drop NaNs, duplicates (keep first by default)
    iv_data_clean = iv_data.dropna(subset=required_cols).drop_duplicates(subset=['TimeToExpiry', 'Strike'])

    # Check if sufficient unique points remain for 2D interpolation
    # griddata needs at least N+1 points for N dimensions (here N=2)
    # For 'cubic', more points are generally better/required for stability.
    if iv_data_clean.shape[0] < 4:
        print(f"Warning: Insufficient unique data points ({iv_data_clean.shape[0]}) for 2D interpolation.")
        # Return empty arrays matching the expected tuple structure
        return np.array([]), np.array([]), np.array([])

    points = iv_data_clean[['TimeToExpiry', 'Strike']].values
    values = iv_data_clean['ImpliedVolatility'].values

    # Define grid boundaries from the actual data range
    min_T, max_T = points[:, 0].min(), points[:, 0].max()
    min_K, max_K = points[:, 1].min(), points[:, 1].max()

    # Avoid creating a grid with zero range if all points fall on a line
    if max_T <= min_T: min_T, max_T = min_T - 0.01, max_T + 0.01 # Add small range if needed
    if max_K <= min_K: min_K, max_K = min_K - 1.0, max_K + 1.0 # Add small range if needed


    # Create the target grid
    T_lin = np.linspace(min_T, max_T, n_expiries)
    K_lin = np.linspace(min_K, max_K, n_strikes)
    T_grid, K_grid = np.meshgrid(T_lin, K_lin)

    # Perform interpolation
    iv_surface_interpolated: np.ndarray = np.array([]) # Initialize
    try:
        # Use fill_value=np.nan for points outside the convex hull of input data
        iv_surface_interpolated = griddata(
            points, values, (T_grid, K_grid), method=method, fill_value=np.nan
        )

        # Post-interpolation check: Replace potential negative IVs with NaN
        # These can arise especially with 'cubic' interpolation near boundaries
        if iv_surface_interpolated.size > 0:
            iv_surface_interpolated[iv_surface_interpolated < 0] = np.nan

    except Exception as e:
        print(f"Error during scipy.interpolate.griddata (method='{method}'): {e}")
        # Return empty arrays in case of a critical failure in griddata
        return np.array([]), np.array([]), np.array([])

    # Check quality of interpolation result (e.g., too many NaNs)
    if iv_surface_interpolated.size > 0:
        nan_percentage = np.isnan(iv_surface_interpolated).sum() / iv_surface_interpolated.size
        if nan_percentage > 0.8:
            print(f"Warning: Interpolation result (method='{method}') is {nan_percentage:.1%} NaN. "
                  "Consider checking input data or using 'linear' interpolation.")
    elif T_grid.size > 0: # If grid was created but interpolation array is empty
        print("Warning: Interpolation resulted in an empty array despite valid grid.")
        # Fill with NaNs to match grid shape if possible
        iv_surface_interpolated = np.full(T_grid.shape, np.nan)


    return T_grid, K_grid, iv_surface_interpolated

# def make_strictly_increasing(self, moneyness, iv):
#     """
#     Sort data by x and remove duplicates to ensure strictly increasing x values.
#     """
#     # Combine x and y data
#     combined = np.column_stack((moneyness, iv))

#     # Remove NaN rows
#     combined = combined[~np.isnan(combined).any(axis=1)]

#     # Sort by x values (first column)
#     sorted_indices = np.argsort(combined[:, 0])
#     combined = combined[sorted_indices]

#     # Remove duplicates (keep first occurrence)
#     unique_x, unique_indices = np.unique(combined[:, 0], return_index=True)
#     combined = combined[unique_indices]

#     return combined[:, 0], combined[:, 1]  # Return x, y

# def create_interpolation(self):
#     """
#     Create cubic spline interpolation from the loaded data using instance variables.

#     Returns:
#     --------
#     tuple
#         (cubic_spline_object, x_dense, y_dense)
#     """
#     if self.moneyness is None or self.iv is None:
#         raise ValueError("Data not loaded. Call get_data() first.")

#     # Clean and sort the data to ensure strictly increasing moneyness
#     clean_moneyness, clean_iv = self.make_strictly_increasing(self.moneyness, self.iv)

#     # Verify we have enough points for interpolation
#     if len(clean_moneyness) < 2:
#         raise ValueError("Need at least 2 unique data points for interpolation.")

#     self.cubic_spline = CubicSpline(clean_moneyness, clean_iv, bc_type="natural")

#     self.x_dense = np.linspace(clean_moneyness.min(), clean_moneyness.max(), self.num_points)
#     self.y_dense = self.cubic_spline(self.x_dense)

#     return self.cubic_spline, self.x_dense, self.y_dense

def plot_interpolated_surface(
    T_grid: np.ndarray,
    K_grid: np.ndarray,
    iv_surface: np.ndarray,
    filename: str = "plot.png",
    title: str = "Implied Volatility Surface",
    cmap: str = "viridis"
) -> None:
    """
    Plots the interpolated implied volatility surface as a 3D plot.

    Creates an empty plot file
    if the input data is invalid or empty, as per requirements.

    Args:
        T_grid: Meshgrid for TimeToExpiry from interpolation.
        K_grid: Meshgrid for Strike from interpolation.
        iv_surface: Interpolated Implied Volatility values on the grid.
        title: Title for the plot.
        cmap: Colormap for the surface plot (e.g., "viridis", "plasma").
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Check if data is valid for plotting
    plot_valid = (T_grid.size > 0 and
                  K_grid.size > 0 and
                  iv_surface.size > 0 and
                  T_grid.shape == K_grid.shape and
                  T_grid.shape == iv_surface.shape and
                  not np.all(np.isnan(iv_surface))) # Check if not entirely NaN

    if plot_valid:
        # Plot the surface - plot_surface handles NaNs internally by not drawing facets
        surf = ax.plot_surface(
            T_grid, K_grid, iv_surface * 100, # Plot IV as percentage
            cmap=cmap,
            edgecolor='none', # 'k' for black edges, 'none' for smooth
            rcount=100, ccount=100, # Control mesh density if needed
            antialiased=True
        )
        ax.set_title(title)
        ax.set_xlabel("Time to Expiry (Years)")
        ax.set_ylabel("Strike Price")
        ax.set_zlabel("Implied Volatility (%)")
        ax.set_zlim(0, MAX_IV * 100 * 1.1) # Set Z limit based on plausible IV * 1.1

        # Improve viewing angle
        ax.view_init(elev=25, azim=-65) # Adjust elevation and azimuth

        # Add a color bar which maps values to colors
        fig.colorbar(surf, shrink=0.6, aspect=10, pad=0.1, label='Implied Volatility (%)')

    else:
        # Create an empty plot with labels if data is invalid, as per instructions
        print(f"Warning: Input data for plot '{filename}' is invalid or empty. Generating empty plot.")
        ax.set_title(f"{title}\n(No valid data to plot)")
        ax.set_xlabel("Time to Expiry (Years)")
        ax.set_ylabel("Strike Price")
        ax.set_zlabel("Implied Volatility (%)")
        # Set some default limits to make the empty plot visible
        ax.set_xlim(0, 1)
        ax.set_ylim(K_grid.min() if K_grid.size > 0 else 0, K_grid.max() if K_grid.size > 0 else 100)
        ax.set_zlim(0, 100)

    try:
        plt.tight_layout()


    except Exception as e:
        print(f"Error saving plot {filename}: {e}")

    # plt.close(fig) # Close the figure explicitly to free memory


# --- Main Execution Logic ---
def main() -> None:
    """
    Main function to execute the workflow:
    1. Fetch option data.
    2. Calculate implied volatilities.
    3. Interpolate the volatility surface.
    4. Plot the interpolated surface.
    """
    print(f"--- Volatility Surface Interpolation for {TICKER_SYMBOL} ---")

    # 1. Fetch data
    print(f"\nFetching option chain data for {TICKER_SYMBOL}...")
    option_data = data_fetcher.fetch_option_chain(TICKER_SYMBOL)

    if option_data.empty:
        print(f"Could not retrieve option data for {TICKER_SYMBOL}. Exiting.")
        # Ensure plot file is created even on fetch failure
        plot_interpolated_surface(np.array([]), np.array([]), np.array([]), filename='plot.png', title=f"{TICKER_SYMBOL} IV Surface (Fetch Failed)")
        return

    print(f"Fetched {len(option_data)} option contracts.")
    print(f"Spot Price: {option_data['SpotPrice'].iloc[0]:.2f}")


    # 2. Calculate IV
    print("\nCalculating Implied Volatility (this may take a moment)...")
    iv_results_df = iv_calculator.calculate_iv_for_chain(option_data, r=RISK_FREE_RATE)

    if iv_results_df.empty or iv_results_df['ImpliedVolatility'].isnull().all():
        print("No valid Implied Volatility data could be calculated after filtering.")
        # Ensure plot file is created
        plot_interpolated_surface(np.array([]), np.array([]), np.array([]), filename='plot.png', title=f"{TICKER_SYMBOL} IV Surface (No Valid IV Data)")
        return

    valid_iv_count = iv_results_df['ImpliedVolatility'].notna().sum()
    print(f"Successfully calculated IV for {valid_iv_count} options after filtering.")
    print("Sample of calculated IV data:")
    print(iv_results_df.head())


    # 3. Interpolate the surface
    print(f"\nInterpolating the volatility surface using '{INTERPOLATION_METHOD}' method...")
    T_grid, K_grid, iv_surface_interpolated = interpolate_volatility_surface(
        iv_data=iv_results_df,
        method=INTERPOLATION_METHOD,
        n_strikes=N_STRIKES_GRID,
        n_expiries=N_EXPIRIES_GRID
    )

    # 4. Plot the interpolated surface
    if T_grid.size > 0: # Check if interpolation produced a grid
        print("Plotting the interpolated surface...")
        plot_title = (f"{TICKER_SYMBOL} Implied Volatility Surface\n"
                      f"({INTERPOLATION_METHOD.capitalize()} Interpolation, "
                      f"{N_EXPIRIES_GRID}x{N_STRIKES_GRID} Grid)")
        plot_interpolated_surface(
            T_grid,
            K_grid,
            iv_surface_interpolated,
            filename="plot.png",
            title=plot_title
        )
    else:
        print("Interpolation failed or yielded no usable grid. Cannot plot surface.")
        # Ensure plot file is created even if interpolation failed
        plot_interpolated_surface(np.array([]), np.array([]), np.array([]), filename='plot.png', title=f"{TICKER_SYMBOL} IV Surface (Interpolation Failed)")

    print("\n--- Script execution finished ---")


if __name__ == "__main__":
    # Optional: Override ticker from environment variable if set
    # TICKER_SYMBOL = os.environ.get("VOL_TICKER", TICKER_SYMBOL)
    main()