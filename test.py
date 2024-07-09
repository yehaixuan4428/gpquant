import pandas as pd
from gpquant.SymbolicRegressor import SymbolicRegressor
from gpquant.Function import function_map
from factor_processors.data_loader import DataLoader
import rqdatac
import os
import numpy as np


def original_test():
    file_path = "data.csv"
    df = pd.read_csv(file_path, parse_dates=["dt"])
    slippage = 0.001
    df["A"] = df["C"] * (1 + slippage)
    df["B"] = df["C"] * (1 - slippage)
    print(df)

    sr = SymbolicRegressor(
        population_size=2000,
        tournament_size=20,
        generations=50,
        stopping_criteria=2,
        p_crossover=0.6,
        p_subtree_mutate=0.2,
        p_hoist_mutate=0.1,
        p_point_mutate=0.05,
        init_depth=(6, 8),
        init_method="half and half",
        function_set=[],
        variable_set=["O", "H", "L", "C", "V"],
        const_range=(1, 20),
        ts_const_range=(1, 20),
        build_preference=[0.75, 0.75],
        metric="sharpe ratio",
        transformer="quantile",
        transformer_kwargs={
            "init_cash": 5000,
            "charge_ratio": 0.00002,
            "d": 15,
            "o_upper": 0.8,
            "c_upper": 0.6,
            "o_lower": 0.2,
            "c_lower": 0.4,
        },
        parsimony_coefficient=0.005,
    )

    sr.fit(df.iloc[:400], df["C"].iloc[:400])
    print(sr.score(df.iloc[400:800], df["C"].iloc[400:800]))


def get_data(online=False):

    if not online:
        loader = DataLoader(n_cores=4)
        data = (
            loader.get_stock_price_1d(
                pd.to_datetime("20240501"), pd.to_datetime("20240601")
            )
            .swaplevel()
            .sort_index()
        )
        data = data.groupby(level=0).tail(100)
    else:
        stocks = rqdatac.index_components("000016.XSHG", date=None)
        data = rqdatac.get_price(
            stocks,
            start_date="20240501",
            end_date="20240601",
            frequency="1d",
        )
        data = data.swaplevel().sort_index()
        data.index = data.index.set_names(["dt", "code"])
    return data


function_set = list(function_map.keys())[:20]


def test_fit(data):
    sr = SymbolicRegressor(
        population_size=100,
        tournament_size=20,
        generations=3,
        stopping_criteria=2,
        p_crossover=0.6,
        p_subtree_mutate=0.2,
        p_hoist_mutate=0.1,
        p_point_mutate=0.05,
        init_depth=(6, 8),
        init_method="half and half",
        function_set=function_set,
        variable_set=["open", "high", "low", "close", "volume"],
        const_range=(1, 20),
        ts_const_range=(1, 20),
        build_preference=[0.75, 0.75],
        metric="sectional ic",
        transformer=None,
        transformer_kwargs=None,
        parsimony_coefficient=0.005,
    )

    dates = data.index.get_level_values(0).unique()
    train_dates = dates[: int(0.8 * len(dates))]
    test_dates = dates[int(0.8 * len(dates)) :]
    train_data = data.loc[train_dates].sort_index()
    test_data = data.loc[test_dates].sort_index()

    sr.fit(train_data, train_data["close"])
    print(sr.score(test_data, test_data["close"]))


if __name__ == "__main__":
    # rqdatac.init()
    # data = get_data(online=True)
    # data.to_pickle("data.pkl")
    data = pd.read_pickle("data.pkl")
    # print(data["close"].rolling(10).cov(data["open"]))

    # print(
    #     data["close"]
    #     .groupby(level=1, group_keys=False)
    #     .apply(lambda x: x.rolling(10).apply(lambda y: y.ewm(alpha=0.1).mean()))
    # )
    x1 = data["close"]

    # volatility = volatility.groupby(level=1).rolling(d, int(d / 2)).sum()
    # print(data["close"].groupby(level=1).apply(lambda x: tb.KAMA(x, 10)))
    d = 10

    # print(data["close"].groupby(level=1).apply(lambda x: EMA(x, 10, 0.1)))
    print(x1.groupby(level=1).shift())

    # test_fit(data)
    # rqdatac.init(os.environ["RQ_URI"])
    # data = get_data(online = False)
    # print(data["close"].groupby(level=0).corr(data["open"]))
