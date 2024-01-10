import yfinance as yf

def get_basf_data():
    return yf.download("BAS.DE")