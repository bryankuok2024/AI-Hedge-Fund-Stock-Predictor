# Placeholder for Quantitative Analyst Agent based on INVESTO-Stock-Predictor logic
import pandas as pd
import yfinance as yf
import pandas_ta as pta
# import ta as talib # Commenting out for now, using pandas_ta
# import statsmodels.api as sm # Commenting out for now, prediction models later
# Import other necessary models like XGBoost, LightGBM later if needed
# from xgboost import XGBRegressor
# from lightgbm import LGBMRegressor
# Import other necessary components from INVESTO structure if possible

# Placeholder function to be filled with logic from INVESTO
def run_quantitative_analysis(ticker: str, start_date: str, end_date: str) -> dict:
    """
    Fetches data, calculates a broader set of technical indicators, and prepares for predictive models
    based on INVESTO-Stock-Predictor logic.
    
    Now returns the historical data DataFrame with indicators included.

    Args:
        ticker: Stock ticker symbol.
        start_date: Start date for historical data (YYYY-MM-DD).
        end_date: End date for historical data (YYYY-MM-DD).

    Returns:
        A dictionary containing analysis results, including:
        - ticker: The stock ticker symbol.
        - technical_signals: Dictionary of the *latest* calculated indicator values.
        - historical_data: Pandas DataFrame containing OHLCV data and all calculated indicator series for the period.
        - prediction: Placeholder dictionary.
        - error: String containing error message if any occurred, else None.
        e.g.,
        {
            "ticker": ticker,
            "technical_signals": { ... latest values ... },
            "historical_data": DataFrame(...),
            "prediction": { "status": "Not implemented yet" },
            "error": None
        }
    """
    print(f"--- Running Quantitative Analysis for {ticker} ({start_date} to {end_date}) ---")
    results = {
        "ticker": ticker,
        "technical_signals": {},
        "historical_data": None, # Initialize new key
        "prediction": {"status": "Not implemented yet"},
        "error": None
    }
    df = pd.DataFrame() # Initialize original df
    df_simple = pd.DataFrame() # Initialize simplified df
    return_df = False # Flag to control df deletion in finally

    try:
        # 1. Fetch Data using yfinance
        print(f"Fetching data for {ticker} from {start_date} to {end_date}...")
        # Need enough historical data for indicators (e.g., SMA 200 needs > 200 periods)
        # Fetch more data than requested for calculations (adjust days if needed)
        calc_start_date = (pd.to_datetime(start_date) - pd.Timedelta(days=300)).strftime('%Y-%m-%d')
        df = yf.download(ticker, start=calc_start_date, end=end_date, progress=False)

        if df.empty:
            raise ValueError(f"Failed to download stock data for {ticker} or date range invalid.")

        print(f"Data fetched successfully. Shape: {df.shape}")
        # --- DEBUG --- Add prints to see raw columns and data head
        # print(f"DEBUG: Raw columns from yfinance for {ticker}: {df.columns}") # Commenting out debug print
        # print(f"DEBUG: Data head for {ticker}:\n{df.head()}") # Commenting out debug print
        # ------------- END DEBUG -------------

        # --- FIX for MultiIndex --- 
        # Check if columns are MultiIndex
        if isinstance(df.columns, pd.MultiIndex):
            print(f"Detected MultiIndex columns for {ticker}. Simplifying...")
            # Select columns for the current ticker and rename to standard lowercase
            try:
                # Select the standard OHLCV columns using the ticker symbol
                # Use .get() with default value None for columns that might be missing (like 'Adj Close')
                df_simple = pd.DataFrame({
                    'open': df[('Open', ticker)],
                    'high': df[('High', ticker)],
                    'low': df[('Low', ticker)],
                    'close': df[('Close', ticker)],
                    'volume': df[('Volume', ticker)],
                    # Include Adj Close if it exists in the MultiIndex, otherwise calculate later if needed
                    'adj_close': df.get(('Adj Close', ticker), None) 
                })
                # Drop adj_close if it was all None (meaning it didn't exist)
                if df_simple['adj_close'] is None or df_simple['adj_close'].isnull().all():
                    df_simple.drop(columns=['adj_close'], inplace=True)
                
                # If adj_close is still missing, use close as adj_close for indicator calculations
                if 'adj_close' not in df_simple.columns and 'close' in df_simple.columns:
                    df_simple['adj_close'] = df_simple['close'] 
                    
            except KeyError as e:
                raise ValueError(f"Missing expected column in MultiIndex for {ticker}: {e}. Columns: {df.columns}")
        else:
            # If not MultiIndex, assume standard columns and convert to lowercase
            print(f"Detected standard columns for {ticker}. Renaming to lowercase...")
            df.columns = [col.lower().replace(' ', '_') for col in df.columns]
            # Ensure adj_close exists, use close if necessary
            if 'adj_close' not in df.columns and 'close' in df.columns:
                df['adj_close'] = df['close']
            # Assign df to df_simple for consistent processing below
            df_simple = df 
            
        # --- Check required columns on df_simple --- 
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df_simple.columns for col in required_cols):
            missing_cols = [col for col in required_cols if col not in df_simple.columns]
            raise ValueError(f"Simplified data for {ticker} is missing required columns: {missing_cols}. Columns: {df_simple.columns}")
        # Ensure 'close' exists, as it's crucial for most indicators
        if 'close' not in df_simple.columns:
             raise ValueError(f"Simplified data for {ticker} lacks 'close' column required for indicators.")

        # 2. Calculate Technical Indicators using pandas_ta on the *simplified* DataFrame
        print(f"Calculating technical indicators for {ticker} using simplified data...")

        # Apply indicators to df_simple
        df_simple.ta.rsi(append=True)  # RSI_14
        df_simple.ta.macd(append=True) # MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9
        df_simple.ta.bbands(append=True) # BBL_20_2.0, BBM_20_2.0, BBU_20_2.0
        df_simple.ta.sma(length=50, append=True)  # SMA_50
        df_simple.ta.sma(length=200, append=True) # SMA_200
        df_simple.ta.ema(length=12, append=True)  # EMA_12
        df_simple.ta.ema(length=26, append=True)  # EMA_26
        df_simple.ta.atr(append=True)  # ATR_14
        df_simple.ta.stoch(append=True) # STOCHk_14_3_3, STOCHd_14_3_3

        # Identify all calculated indicator columns from df_simple
        indicator_cols = [col for col in df_simple.columns if col.startswith((
            'RSI', 'MACD', 'BBL', 'BBM', 'BBU',
            'SMA', 'EMA', 'ATR', 'STOCHk', 'STOCHd'
        ))]

        # Extract latest results for 'technical_signals' dictionary (keep this for summary)
        if df_simple.empty or not indicator_cols:
             print(f"Warning: No indicators calculated or DataFrame is empty for {ticker}.")
             results["technical_signals"] = {} # Ensure it's an empty dict
        elif df_simple[indicator_cols].iloc[-1].isnull().any():
             print(f"Warning: Some indicators for {ticker} contain NaN in the last row. May need more historical data or calculation failed.")
             latest_indicators = df_simple[indicator_cols].iloc[-1].to_dict()
             results["technical_signals"] = {k: (None if pd.isna(v) else v) for k, v in latest_indicators.items()}
        else:
            latest_indicators = df_simple[indicator_cols].iloc[-1].to_dict()
            results["technical_signals"] = latest_indicators

        print(f"Technical indicators calculated.")
        
        # --- Add the full DataFrame to results --- 
        # Return only the data within the originally requested date range if needed,
        # or the full df_simple which includes extra historical data for calculations.
        # Let's return the full df_simple for now, easier for charting context.
        if not df_simple.empty:
            results["historical_data"] = df_simple
            return_df = True # Set flag to prevent deletion
        # -----------------------------------------

        # 3. Run Predictive Models (ARIMA, XGBoost, etc.)
        # --- Placeholder ---
        print(f"Predictive modeling for {ticker} is not implemented yet.")
        # Populate results["prediction"] in the future
        # -----------------

        print(f"--- Quantitative Analysis Complete for {ticker} ---")

    except ValueError as ve:
        print(f"VALUE ERROR in Quantitative Analysis for {ticker}: {ve}")
        results["error"] = str(ve)
    except Exception as e:
        import traceback
        print(f"UNEXPECTED ERROR in Quantitative Analysis for {ticker}: {e}")
        print(traceback.format_exc()) # Print full traceback for unexpected errors
        results["error"] = f"An unexpected error occurred: {str(e)}"
    finally:
        # Optional: Clean up large dataframes if memory is a concern
        # Only delete if we are not returning them
        try:
            del df # Always delete the raw multi-index df
        except NameError:
            pass 
        if not return_df: # Only delete df_simple if not returning it
            try:
                if 'df_simple' in locals(): 
                     del df_simple
            except NameError:
                pass

    return results

if __name__ == '__main__':
    # Example usage for testing
    test_ticker = "GOOGL" # Try another ticker
    test_start = "2023-01-01" # Longer history for SMA 200
    test_end = "2024-03-31" # More recent end date
    print(f"\n--- Running Test for {test_ticker} ---")
    analysis_output = run_quantitative_analysis(test_ticker, test_start, test_end)
    print("\n--- Test Output ---")
    import json
    # Use default=str to handle potential non-serializable types like numpy floats
    print(json.dumps(analysis_output, indent=2, default=str)) 