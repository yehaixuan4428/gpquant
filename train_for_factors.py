import os
import rqdatac
import pandas as pd
from sklearn.model_selection import train_test_split
from factor_processors.data_loader import DataLoader


def load_data(start_date, end_date):
    loader = DataLoader(n_cores=os.cpu_count())
    stock_data = loader.get_stock_price_1d(start_date, end_date)
    stock_data = stock_data.swaplevel().sort_index()
