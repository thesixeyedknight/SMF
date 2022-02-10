import nsepy as nse
import pandas as pd
import numpy as np
from datetime import date


nifty50=['ADANIPORTS','ASIANPAINT','AXISBANK','BAJAJ-AUTO','BAJAJFINSV','BAJFINANCE','BHARTIARTL','BPCL','BRITANNIA','CIPLA','COALINDIA','DIVISLAB','DRREDDY','EICHERMOT','GRASIM','HCLTECH','HDFC','HDFCBANK','HDFCLIFE','HEROMOTOCO','HINDALCO','HINDUNILVR','ICICIBANK','INDUSINDBK','INFY','IOC','ITC','JSWSTEEL','KOTAKBANK','LT','M&M','MARUTI','NESTLEIND','NTPC','ONGC','POWERGRID','RELIANCE','SBILIFE','SBIN','SHREECEM','SUNPHARMA','TATACONSUM','TATAMOTORS','TATASTEEL','TCS','TECHM','TITAN','ULTRACEMCO','UPL','WIPRO']
for x in nifty50:
    data=nse.get_history(symbol=x, start=date(2015,1,1),end=date(2021,12,31))
    file=x+'.csv'
    data.to_csv(file)