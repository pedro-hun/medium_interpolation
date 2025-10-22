from scipy.stats import norm
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, List, Tuple, Callable, Any

# Assume 'options_data_iv' is populated by the output of Section 2.
# It should be a dictionary where keys are time-to-maturity (TTM) in years,
# and values are pandas DataFrames. Each DataFrame must contain at least
# 'Strike', 'Implied Volatility', and 'Forward' columns for that TTM.
# Example structure (replace with actual data from Section 2):
"""
options_data_iv: Dict[float, pd.DataFrame] = {
    0.05: pd.DataFrame({
        'Strike': [90, 95, 100, 105, 110],
        'Implied Volatility': [0.35, 0.30, 0.28, 0.29, 0.32],
        'Forward': [100.5] * 5
    }),
    0.25: pd.DataFrame({
        'Strike': [85, 90, 95, 100, 105, 110, 115],
        'Implied Volatility': [0.32, 0.29, 0.27, 0.25, 0.26, 0.28, 0.31],
        'Forward': [101.0] * 7
    }),
    0.50: pd.DataFrame({
        'Strike': [80, 90, 100, 110, 120],
        'Implied Volatility': [0.30, 0.27, 0.24, 0.25, 0.27],
        'Forward': [101.8] * 5
    }),
    # Add more expiries as needed
}
"""
# --- Placeholder Data Generation (Remove if options_data_iv is loaded) ---
# This section creates placeholder data for demonstration if Section 2's output
# isn't available. Replace this with actual data loading/passing.
# np.random.seed(42) # for reproducibility
# options_data_iv: Dict[float, pd.DataFrame] = {}
# base_forward = 100.0
# ttms = [0.1, 0.25, 0.5, 1.0]
# strikes_range = np.linspace(80, 120, 15)
# base_iv = 0.25

# for ttm in ttms:
#     forward = base_forward * np.exp(0.02 * ttm) # Add slight drift
#     strikes = strikes_range
#     # Create a smile effect: higher IV for OTM/ITM strikes
#     moneyness = np.log(strikes / forward)
#     # Simple quadratic smile + noise
#     iv = base_iv + 0.5 * moneyness**2 - 0.1 * moneyness + np.random.normal(0, 0.015, len(strikes))
#     iv = np.clip(iv, 0.05, 0.8) # Ensure realistic IV bounds
#     df = pd.DataFrame({
#         'Strike': strikes,
#         'Implied Volatility': iv,
#         'Forward': [forward] * len(strikes)
#     })
#     # Add some filtering similar to real data preparation
#     df = df[df['Implied Volatility'] > 1e-5] # Remove zero IVs
#     options_data_iv[ttm] = df


def svi_raw(k: np.ndarray, params: Tuple[float, float, float, float, float]) -> np.ndarray:
    """
    Calculates the SVI total implied variance (w = IV^2 * T) using the raw parameterization.

    Args:
        k: Log-moneyness, k = log(K/F). Can be a scalar or NumPy array.
        params: Tuple containing SVI parameters (a, b, rho, m, sigma).

    Returns:
        Total implied variance (w).
    """
    a, b, rho, m, sigma = params
    # Apply constraints for numerical stability and basic viability
    b = max(b, 0.0)  # Ensure b is non-negative
    sigma = max(sigma, 1e-4) # Ensure sigma is strictly positive
    rho = max(min(rho, 1.0), -1.0) # Ensure rho is in [-1, 1]

    total_variance = a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma**2))
    # Ensure total variance is non-negative
    total_variance = np.maximum(total_variance, 1e-6)
    return total_variance

def svi_objective(params: Tuple[float, float, float, float, float],
                  k: np.ndarray,
                  market_w: np.ndarray,
                  weights: np.ndarray) -> float:
    """
    Objective function for SVI calibration. Calculates the weighted sum of
    squared errors between model total variance and market total variance.

    Args:
        params: Tuple containing SVI parameters (a, b, rho, m, sigma).
        k: Array of log-moneyness values for the current expiration.
        market_w: Array of market total variances (IV^2 * T) for the current expiration.
        weights: Array of weights for each observation (e.g., based on Vega).

    Returns:
        The weighted sum of squared errors.
    """
    model_w = svi_raw(k, params)
    error = np.sum(weights * (model_w - market_w)**2)

    # Penalty for parameters violating theoretical no-arbitrage constraints
    # (Basic constraints are in svi_raw, more complex ones can be added here)
    a, b, rho, m, sigma = params
    return error

def calculate_vega_forward(F: float, K: np.ndarray, T: float, sigma: np.ndarray) -> np.ndarray:
    """
    Calculate Vega using forward price (more appropriate for options data).
    
    Parameters:
    -----------
    F : float
        Forward price
    K : np.ndarray
        Array of strike prices
    T : float
        Time to expiration in years
    sigma : np.ndarray
        Array of implied volatilities
    
    Returns:
    --------
    np.ndarray
        Array of Vega values
    """
    # For forward-based calculation, d1 = (ln(F/K) + 0.5*sigma^2*T) / (sigma*sqrt(T))
    d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    
    # Vega = F * phi(d1) * sqrt(T) * exp(-r*T)
    # For simplicity, we'll use the discounted forward approach
    vega = F * norm.pdf(d1) * np.sqrt(T)
    
    return vega

def calibrate_svi_slice(k: np.ndarray,
                        market_w: np.ndarray,
                        ttm: float,
                        forward_price: float,
                        strikes: np.ndarray,
                        market_iv: np.ndarray,
                        initial_guess: List[float] = None,
                        bounds: List[Tuple[float, float]] = None,
                        use_vega_weights: bool = True) -> Tuple[np.ndarray, float]:
    """
    Calibrates SVI parameters for a single expiration slice using optimization.

    Args:
        k: Array of log-moneyness values.
        market_w: Array of market total variances (IV^2 * T).
        ttm: Time to maturity for this slice.
        initial_guess: Optional initial guess for SVI parameters [a, b, rho, m, sigma].
                       If None, a default guess is used.
        bounds: Optional bounds for the optimizer. If None, default bounds are used.
        weights: Optional weights for the optimization objective function.
                 Defaults to equal weights if None.
        use_vega_weights: If True, weights are based on option Vega.

    Returns:
        A tuple containing:
        - calibrated_params (np.ndarray): The optimized SVI parameters.
        - optimization_result (OptimizeResult): The full result object from scipy.optimize.minimize.
    """
    # Calculate weights
    if use_vega_weights:
        # Calculate Vega for each option
        vega_values = calculate_vega_forward(forward_price, strikes, ttm, market_iv)
        
        # Normalize Vega values to create weights
        # Higher Vega = higher sensitivity = higher weight in calibration
        weights = vega_values / np.sum(vega_values)
        

        weights = weights / np.sum(weights)  # Renormalize
        
        print(f"  Using Vega weights - Min: {weights.min():.4f}, Max: {weights.max():.4f}")
    else:
        weights = np.ones_like(market_w)
        weights = weights / np.sum(weights)  # Normalize weights



    # Default initial guess - often requires tuning based on market regime
    if initial_guess is None:
        # Rough estimation based on data properties
        w_atm = market_w[np.argmin(np.abs(k))] # Approx variance at the money
        min_w = np.min(market_w)
        max_w = np.max(market_w)
        # min_w = 0.2
        # max_w = 0.35
        k_min_w = k[np.argmin(market_w)] # Log-moneyness at min variance

        a_guess = min_w # 'a' roughly anchors the minimum variance level
        m_guess = k_min_w # 'm' is roughly the log-moneyness of min variance
        rho_guess = np.sign(market_w[-1] - market_w[0]) * 0.5 # Skew direction * guess
        rho_guess = np.clip(rho_guess, -0.95, 0.95)
        # b and sigma control the wings/curvature - harder to guess simply
        # Estimate sigma from the width of the smile
        k_range = np.max(k) - np.min(k)
        sigma_guess = max(k_range / 4.0, 0.05) # Heuristic
        # Estimate b based on the variance range
        b_guess = max((max_w - min_w) / (sigma_guess * 2.0), 1e-3) # Heuristic, avoid zero

        initial_guess = [a_guess, b_guess, rho_guess, m_guess, sigma_guess]

    # Default bounds - crucial for stability and meaningful parameters
    if bounds is None:
        max_market_w = np.max(market_w)
        k_min, k_max = np.min(k), np.max(k)
        bounds = [
            (1e-6, max_market_w * 1.5),  # a: level, must be positive, bound by observed max var
            (1e-6, None),                 # b: slope magnitude, must be non-negative
            (-0.999, 0.999),              # rho: correlation, strictly within (-1, 1)
            (k_min * 1.5 if k_min < 0 else k_min * 0.5, k_max * 1.5 if k_max > 0 else k_max * 0.5), # m: location of min, within/near observed k range
            (1e-4, None)                  # sigma: ATM curvature, must be positive
        ]

    result = minimize(svi_objective,
                      initial_guess,
                      args=(k, market_w, weights),
                      method='Nelder-Mead',
                      bounds=bounds,
                      options={'maxiter': 10000, 'ftol': 1e-8, 'gtol': 1e-7}) # Increased maxiter and adjusted tolerances

    if not result.success:
        print(f"Warning: Optimization for TTM={ttm:.4f} failed: {result.message}")
        # Fallback or handling for failed optimization could be added here
        # For now, we return the result as is.

    return result.x, result


# --- Main Calibration Loop ---
def calibration_loop(options_data_iv: Dict[float, pd.DataFrame],
                     use_vega_weights: bool = True) -> tuple[Dict[float, np.ndarray], Dict[float, Any]]:
    
    svi_params_calibrated: Dict[float, np.ndarray] = {}
    optimization_results: Dict[float, Any] = {}

    print("Starting SVI calibration for each expiration slice...")

    for ttm, df_slice in options_data_iv.items():
        if len(df_slice) < 5: # Need sufficient points to fit 5 parameters
            print(f"Skipping TTM={ttm*252:.4f} due to insufficient data points ({len(df_slice)}).")
            continue

        print(f"Calibrating for TTM = {ttm*252:.4f} years...")
        forward_price = df_slice['Forward'].iloc[0]
        strikes = df_slice['Strike'].values
        market_iv = df_slice['ImpliedVolatility'].values

        # Avoid issues with zero/infinite moneyness if F or K is zero
        valid_indices = (strikes > 1e-6) & (forward_price > 1e-6)
        if not np.any(valid_indices):
            print(f"Skipping TTM={ttm*252:.4f} due to invalid strike/forward prices.")
            continue

        strikes = strikes[valid_indices]
        market_iv = market_iv[valid_indices]

        if len(strikes) < 5: # Check again after filtering
            print(f"Skipping TTM={ttm*252:.4f} after filtering invalid prices ({len(strikes)} points left).")
            continue


        log_moneyness = np.log(strikes / forward_price)
        market_total_variance = market_iv**2 * ttm

        # Optional: Use Vega weighting (requires Black-Scholes Vega calculation - omitted for simplicity here)


        try:
            calibrated_params, optim_result = calibrate_svi_slice(
                log_moneyness, market_total_variance, ttm,
                forward_price=forward_price, strikes=strikes, market_iv=market_iv,
                use_vega_weights=use_vega_weights
            )
            svi_params_calibrated[ttm] = calibrated_params
            optimization_results[ttm] = optim_result
            print(f"  Success: {optim_result.success}, Params: {np.round(calibrated_params, 4)}")
        except Exception as e:
            print(f"  Error during optimization for TTM={ttm:.4f}: {e}")
        
    return svi_params_calibrated, optimization_results


print("\nSVI calibration finished.")


# --- Plot 4: Single Expiry Comparison ---
def single_expiry_comparison_plot(options_data_iv: Dict[float, pd.DataFrame],
                                  svi_params_calibrated: Dict[float, np.ndarray], n_expire: int) -> None:
        
    if svi_params_calibrated:
        # Select a TTM for plotting (e.g., the first one calibrated, or a middle one)
        plot_ttm = list(svi_params_calibrated.keys())[n_expire]
        print(f"\nGenerating comparison plot for TTM = {plot_ttm:.4f}...")

        df_plot = options_data_iv[plot_ttm]
        forward_plot = df_plot['Forward'].iloc[0]
        strikes_plot = df_plot['Strike'].values
        market_iv_plot = df_plot['ImpliedVolatility'].values

        valid_indices_plot = (strikes_plot > 1e-6) & (forward_plot > 1e-6)
        strikes_plot = strikes_plot[valid_indices_plot]
        market_iv_plot = market_iv_plot[valid_indices_plot]

        k_plot = np.log(strikes_plot / forward_plot)
        params_plot = svi_params_calibrated[plot_ttm]

        # Generate model IVs across a finer range of log-moneyness
        k_smooth = np.linspace(k_plot.min(), k_plot.max(), 100)
        model_w_smooth = svi_raw(k_smooth, params_plot)
        model_iv_smooth = np.sqrt(model_w_smooth / plot_ttm)

        plt.figure(figsize=(10, 6))
        plt.scatter(k_plot, market_iv_plot, label='Market IV', marker='o', color='blue')
        plt.plot(k_smooth, model_iv_smooth, label='SVI Model Fit', color='red', linestyle='-')
        plt.title(f'SVI Fit vs Market IV for TTM = {plot_ttm:.4f} years')
        plt.xlabel('Log-Moneyness (k = log(K/F))')
        plt.ylabel('ImpliedVolatility')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.show() # Uncomment to display plot interactively
        # plt.close()
    else:
        print("Skipping Plot 4 generation as no SVI parameters were successfully calibrated.")


    # --- Plot 5: 3D SVI Volatility Surface ---
    # if len(svi_params_calibrated) > 1: # Need at least two expiries to make a surface
    #     print("\nGenerating 3D SVI volatility surface plot...")

    #     calibrated_ttms = sorted(svi_params_calibrated.keys())
    #     all_params = np.array([svi_params_calibrated[t] for t in calibrated_ttms])

    #     # Create interpolation functions for each SVI parameter across TTMs
    #     # Using simple linear interpolation here; more advanced methods could be used
    #     interp_funcs: List[Callable] = []
    #     for i in range(all_params.shape[1]): # Iterate through parameters a, b, rho, m, sigma
    #         # Use linear interpolation, fill beyond bounds with nearest value
    #         interp_funcs.append(interp1d(calibrated_ttms, all_params[:, i], kind='linear', fill_value="extrapolate"))

    #     def get_svi_params_interp(ttm: float) -> Tuple[float, float, float, float, float]:
    #         """Interpolates SVI parameters for a given TTM."""
    #         # Ensure TTM is within reasonable bounds if extrapolating heavily
    #         ttm = np.clip(ttm, calibrated_ttms[0], calibrated_ttms[-1])
    #         return tuple(f(ttm) for f in interp_funcs) # type: ignore

    #     # Define grid for the surface plot
    #     min_k = min(np.log(df['Strike'].min() / df['Forward'].iloc[0]) for df in options_data_iv.values() if not df.empty)
    #     max_k = max(np.log(df['Strike'].max() / df['Forward'].iloc[0]) for df in options_data_iv.values() if not df.empty)
    #     k_grid_vals = np.linspace(min_k, max_k, 50)

    #     min_T = min(calibrated_ttms)
    #     max_T = max(calibrated_ttms)
    #     T_grid_vals = np.linspace(min_T, max_T, 40)

    #     K_grid, T_grid = np.meshgrid(k_grid_vals, T_grid_vals)
    #     IV_surface = np.full_like(K_grid, np.nan)

    #     # Calculate IV for each point on the grid using interpolated SVI params
    #     for i in range(T_grid.shape[0]):
    #         for j in range(K_grid.shape[1]):
    #             k_val = K_grid[i, j]
    #             T_val = T_grid[i, j]
    #             if T_val <= 1e-6: continue # Avoid division by zero TTM

    #             interp_params = get_svi_params_interp(T_val)
    #             model_w = svi_raw(k_val, interp_params)
    #             IV_surface[i, j] = np.sqrt(model_w / T_val)

    #     # Ensure IVs are within a reasonable range
    #     IV_surface = np.clip(IV_surface, 0.01, 1.5) # Clip extreme values

    #     # Create the 3D plot
    #     fig = plt.figure(figsize=(12, 8))
    #     ax = fig.add_subplot(111, projection='3d')
    #     surf = ax.plot_surface(K_grid, T_grid, IV_surface, cmap='viridis', # Use 'viridis' colormap
    #                         linewidth=0, antialiased=True, alpha=0.9)

    #     ax.set_xlabel('Log-Moneyness (k)')
    #     ax.set_ylabel('Time to Expiry (T)')
    #     ax.set_zlabel('Implied Volatility (SVI)')
    #     ax.set_title('Implied Volatility Surface from Calibrated SVI Model')
    #     fig.colorbar(surf, shrink=0.5, aspect=5, label='Implied Volatility')
    #     ax.view_init(elev=30, azim=-120) # Adjust view angle for better visualization
    #     plt.show() # Uncomment to display plot interactively
    #     # plt.close()

    # elif len(svi_params_calibrated) == 1:
    #     print("Skipping Plot 5 generation as only one expiry was calibrated. Need at least two for a surface.")
    # else:
    #     print("Skipping Plot 5 generation as no SVI parameters were successfully calibrated.")

    # print("\nSection 4 finished.")