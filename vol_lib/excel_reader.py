import xlwings as xw
import numpy as np
import pandas as pd
import math

class ExcelReader:
    def __init__(self, excel_file: str, sheet_name: str,
                 iv_col: str,
                 start_row: int, end_row: int, bid_col: str,
                 ask_col: str, last_price_col: str, strike_col: str,
                 open_interest_col: str, volume_col: str, ticker_col: str,
                 maturity_col: str, spot_price_col: str):
        self.excel_file = excel_file
        self.sheet_name = sheet_name
        self.iv_col = iv_col
        self.bid_col = bid_col
        self.ask_col = ask_col
        self.last_price_col = last_price_col
        self.strike_col = strike_col
        self.open_interest_col = open_interest_col
        self.volume_col = volume_col
        self.ticker_col = ticker_col
        self.maturity_col = maturity_col
        self.start_row = start_row
        self.end_row = end_row
        self.spot_price_col = spot_price_col
        self.wb = None
        self.sht = None
        self.moneyness = None
        self.iv = None
        self.data_df = None

    def read_sheet(self):
        """
        Open Excel workbook and return the specified sheet using instance variables.

        Returns:
        --------
        xlwings.Sheet
            The Excel sheet object
        """
        self.wb = xw.Book(self.excel_file)
        self.sht = self.wb.sheets[self.sheet_name]
        return self.sht

    def get_data(self):
        """
        Read moneyness and implied volatility data from Excel using instance variables.

        Returns:
        --------
        tuple
            (moneyness_array, iv_array, dataframe)
        """
        # Read sheet if not already open
        if self.sht is None or self.wb is None:
            self.read_sheet()

        iv = self.sht.range(f"{self.iv_col}{self.start_row} : {self.iv_col}{self.end_row}").value
        bid = self.sht.range(f"{self.bid_col}{self.start_row} : {self.bid_col}{self.end_row}").value
        ask = self.sht.range(f"{self.ask_col}{self.start_row} : {self.ask_col}{self.end_row}").value
        last_price = self.sht.range(f"{self.last_price_col}{self.start_row} : {self.last_price_col}{self.end_row}").value
        strike = self.sht.range(f"{self.strike_col}{self.start_row} : {self.strike_col}{self.end_row}").value
        open_interest = self.sht.range(f"{self.open_interest_col}{self.start_row} : {self.open_interest_col}{self.end_row}").value
        volume = self.sht.range(f"{self.volume_col}{self.start_row} : {self.volume_col}{self.end_row}").value
        ticker = self.sht.range(f"{self.ticker_col}{self.start_row} : {self.ticker_col}{self.end_row}").value
        maturity = self.sht.range(f"{self.maturity_col}{self.start_row} : {self.maturity_col}{self.end_row}").value
        spot_price = self.sht.range(f"{self.spot_price_col}{self.start_row} : {self.spot_price_col}{self.end_row}").value
        


        data_df = pd.DataFrame({"bid": bid, "ask": ask, "lastPrice": last_price, "Strike": strike,
                                "openInterest": open_interest, "volume": volume,
                                "ticker": ticker, "Expiry": maturity, "IV": iv, "SpotPrice": spot_price})

        return data_df

    def get_type(self):
        calls = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"]
        puts = ["M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X"]
        self.data_df = self.get_data()
        types = []
        for ticker in self.data_df["ticker"]:
            if ticker[4] in calls:
                types.append("call")
            elif ticker[4] in puts:
                types.append("put")
        self.data_df["Type"] = types
        return self.data_df
    
    
# Function to calculate time to expiration in business days
    def calculate_tte_business_days(self):
        """
        Calculate time to expiration in business days for option contracts
        
        Parameters:
        maturity_dates: Series or list of maturity dates
        """
        
        # Ensure data_df is loaded
        if self.data_df is None:
            self.data_df = self.get_type()
        
        # today = pd.Timestamp.now().date()
        today = pd.to_datetime('21/10/2025', format='%d/%m/%Y').date()

        # Convert to datetime if needed
        maturity_dates = pd.to_datetime(self.data_df["Expiry"], format='%d/%m/%Y')

        # Load holidays from CSV
        holidays_df = pd.read_csv('feriados_nacionais.csv')

        # Get holidays in the correct format for numpy busday_count
        holidays_for_numpy = pd.to_datetime(holidays_df["Data"], format='%m/%d/%Y').values.astype('datetime64[D]')

        # Calculate business days for each maturity using the properly formatted holidays
        tte_bdays = []
        bs_tte = []
        for maturity in maturity_dates:
            maturity_date = maturity.date()
            bdays = np.busday_count(today, maturity_date, holidays=holidays_for_numpy)
            bs_tte.append(bdays / 252)  # Assuming 252 trading days in a year
            tte_bdays.append(bdays)
        self.data_df["DaysToExpiry"] = tte_bdays
        self.data_df["TimeToExpiry"] = bs_tte
        return self.data_df
    
    def add_forward_price(self, risk_free_rate: float = 0.15):
        """
        Add forward price column to the dataframe using the spot price and time to expiration.
        
        Parameters:
        risk_free_rate: Annualized risk-free interest rate (default is 5%)
        """
        # Ensure data_df is loaded
        if self.data_df is None:
            self.data_df = self.calculate_tte_business_days()
        
        
        # Calculate forward price
        self.data_df["Forward"] = self.data_df["SpotPrice"] * ((1+risk_free_rate) ** self.data_df["TimeToExpiry"])
        return self.data_df
    
        
