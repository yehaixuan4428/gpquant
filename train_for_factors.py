import os
import rqdatac
import pandas as pd
from sklearn.model_selection import train_test_split
from factor_processors.data_loader import DataLoader
from gpquant.SymbolicRegressor import SymbolicRegressor
import random


def load_data(start_date, end_date, columns=None):
    loader = DataLoader(n_cores=os.cpu_count())
    stock_data = loader.get_stock_price_1d(
        start_date,
        pd.to_datetime(rqdatac.get_next_trading_date(end_date, 6)),
        columns=columns,
    )
    stock_data = stock_data.swaplevel().sort_index()
    stock_data['label'] = (
        stock_data['open'].groupby(level=1).pct_change(5).groupby(level=1).shift(-6)
    )

    stock_data.dropna(inplace=True)
    dates = stock_data.index.get_level_values(0).unique()
    train_dates, test_dates = train_test_split(dates, test_size=0.2, shuffle=False)

    train_dataset = stock_data.loc[train_dates]
    test_dataset = stock_data.loc[test_dates]
    return train_dataset, test_dataset


def main(train_data, test_data):
    random.seed(10)
    sr = SymbolicRegressor(
        population_size=100,
        tournament_size=20,
        generations=3,
        stopping_criteria=2,
        p_crossover=0.6,
        p_subtree_mutate=0.2,
        p_hoist_mutate=0.1,
        p_point_mutate=0.05,
        init_depth=(2, 3),
        init_method="half and half",
        function_set=[],
        variable_set=[i for i in train_data.columns if i != 'label'],
        const_range=(1, 5),
        ts_const_range=(1, 5),
        build_preference=[0.75, 0.75],
        metric="sectional ic",
        transformer=None,
        transformer_kwargs=None,
        parsimony_coefficient=0.005,
    )
    sr.fit(train_data.drop(columns=['label']), train_data["label"])
    print(sr.score(test_data.drop(columns=['label']), test_data["label"]))
    return sr


if __name__ == "__main__":
    rqdatac.init(os.environ['RQ_URI'])
    start_date = pd.to_datetime('20240501')
    end_date = pd.to_datetime('20240601')

    variable_set = ['open', 'high', 'low', 'close', 'volume', 'num_trades']
    train_data, test_data = load_data(start_date, end_date, columns=variable_set)

    print("Data is loaded...")

    sr = main(train_data, test_data)
    print(sr)
