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


def test_functions(data):
    from gpquant.Function import _clear_by_cond

    print(_clear_by_cond(data["close"], data["open"], data["high"]))

    from gpquant.Function import _if_then_else

    print(_if_then_else(data["close"], data["open"], data["high"]))

    from gpquant.Function import _if_cond_then_else

    print(_if_cond_then_else(data["close"], data["open"], data["high"], data["low"]))

    from gpquant.Function import _ts_delay

    print(_ts_delay(data["close"], 3))

    from gpquant.Function import _ts_delta

    print(_ts_delta(data["close"], 3))

    from gpquant.Function import _ts_pct_change

    print(_ts_pct_change(data["close"], 3))

    from gpquant.Function import _ts_mean_return

    print(_ts_mean_return(data["close"], 3))

    from gpquant.Function import _ts_max

    print(_ts_max(data["close"], 3))

    from gpquant.Function import _ts_min

    print(_ts_min(data["close"], 3))

    from gpquant.Function import _ts_sum

    print(_ts_sum(data["close"], 3))

    from gpquant.Function import _ts_product

    print(_ts_product(data["close"], 3))

    from gpquant.Function import _ts_mean

    print(_ts_mean(data["close"], 3))

    from gpquant.Function import _ts_std

    print(_ts_std(data["close"], 3))

    from gpquant.Function import _ts_median

    print(_ts_median(data["close"], 3))

    from gpquant.Function import _ts_midpoint

    print(_ts_midpoint(data["close"], 3))

    from gpquant.Function import _ts_skew

    print(_ts_skew(data["close"], 3))

    from gpquant.Function import _ts_kurt

    print(_ts_kurt(data["close"], 3))

    from gpquant.Function import _ts_inverse_cv

    print(_ts_inverse_cv(data["close"], 3))

    from gpquant.Function import _ts_cov

    print(_ts_cov(data["close"], data["open"], 3))

    from gpquant.Function import _ts_corr

    print(_ts_corr(data["close"], data["open"], 3))

    from gpquant.Function import _ts_autocorr

    print(_ts_autocorr(data["close"], 10, 1))

    from gpquant.Function import _ts_maxmin

    print(_ts_maxmin(data["close"], 3))

    from gpquant.Function import _ts_zscore

    print(_ts_zscore(data["close"], 3))

    from gpquant.Function import _ts_regression_beta

    print(_ts_regression_beta(data["close"], data["open"], 3))

    from gpquant.Function import _ts_linear_slope

    print(_ts_linear_slope(data["close"], 3))

    from gpquant.Function import _ts_linear_intercept

    print(_ts_linear_intercept(data["close"], 3))

    from gpquant.Function import _ts_argmax

    print(_ts_argmax(data["close"], 3))

    from gpquant.Function import _ts_argmin

    print(_ts_argmin(data["close"], 3))

    from gpquant.Function import _ts_argmaxmin

    print(_ts_argmaxmin(data["close"], 3))

    from gpquant.Function import _ts_rank

    print(_ts_rank(data["close"], 3))

    from gpquant.Function import _ts_ema

    print(_ts_ema(data["close"], 3))

    from gpquant.Function import _ts_dema

    print(_ts_dema(data["close"], 3))

    from gpquant.Function import _ts_kama

    print(_ts_kama(data["close"], 10, 5, 3))

    from gpquant.Function import _ts_AROONOSC

    print(_ts_AROONOSC(data["high"], data["low"], 10))

    from gpquant.Function import _ts_WR

    print(_ts_WR(data["high"], data["low"], data["close"], 10))

    from gpquant.Function import _ts_CCI

    print(_ts_CCI(data["high"], data["low"], data["close"], 10))

    from gpquant.Function import _ts_ATR

    print(_ts_ATR(data["high"], data["low"], data["close"], 10))

    from gpquant.Function import _ts_NATR

    print(_ts_NATR(data["high"], data["low"], data["close"], 10))

    from gpquant.Function import _ts_ADX

    print(_ts_ADX(data["high"], data["low"], data["close"], 10))

    from gpquant.Function import _ts_MFI

    print(_ts_MFI(data["high"], data["low"], data["close"], data["volume"], 10))


def test_fit(data):
    import random

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
        variable_set=["open", "high", "low", "close", "volume"],
        const_range=(1, 5),
        ts_const_range=(1, 5),
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
def test_tree():
    from gpquant.SyntaxTree import SyntaxTree
    function_set = list(function_map.values())
    variable_set=["open", "high", "low", "close", "volume"]
    tree = SyntaxTree(id = 0, init_dept = 3, init_method = 'half and half', function_set = function_set, variable_set=variable_set)


def test_tree():
    from gpquant.SyntaxTree import SyntaxTree

    tree = SyntaxTree(
        0,
        init_depth=(3, 3),
        init_method="half and half",
        function_set=list(function_map.values()),
        variable_set=["open", "high", "low", "close", "volume"],
        const_range=(1, 5),
        ts_const_range=(1, 5),
        build_preference=[0.75, 0.75],
        metric="sectional ic",
        transformer=None,
        transformer_kwargs=None,
        parsimony_coefficient=0.005,
    )
    print(tree)


if __name__ == "__main__":
    # rqdatac.init()
    # data = get_data(online=True)
    # data.to_pickle("data.pkl")
    # data = pd.read_pickle("data.pkl")

    # test_functions(data)
    # test_fit(data)

    test_tree()
