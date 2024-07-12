import os
import pandas as pd
import shutil
import hashlib
from joblib import Parallel, delayed
import numpy as np
import itertools
from glob import glob


def series_hash(s: pd.Series) -> str:
    return hashlib.md5(pd.util.hash_pandas_object(s).values).hexdigest()


def hash_arg(arg):
    if isinstance(arg, pd.Series):
        return series_hash(arg)
    elif isinstance(arg, (int, float, str, bool)):
        return str(arg)
    else:
        return hashlib.md5(str(arg).encode()).hexdigest()


class CacheManager:
    def __init__(self, cache_dir: str):
        """
        初始化缓存目录对象。

        该构造函数用于创建一个表示缓存目录的实例。它接受一个字符串参数来指定缓存目录的路径。

        参数:
            cache_dir (str): 缓存数据的目录路径。这个路径将被用于存储和检索缓存的文件。
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def clean_cache(self, cache_dir: str = None):
        """
        清理指定的缓存目录。

        如果未指定缓存目录，则使用类实例中定义的默认缓存目录。
        注意：此操作将永久删除指定的缓存目录及其所有内容。

        参数:
            cache_dir (str): 需要被清理的缓存目录路径。如果为None，则使用默认缓存目录。
        """
        # 检查是否提供了缓存目录参数，如果没有，则使用类的默认缓存目录
        if cache_dir is None:
            cache_dir = self.cache_dir
        # 完全删除指定的缓存目录
        shutil.rmtree(cache_dir)
        os.makedirs(cache_dir, exist_ok=True)

    def cache_all(
        self,
        data: pd.DataFrame,
        const_range: tuple,
        variable_names: list[str] = ["open", "high", "low", "close", "volume"],
        n_jobs=1,
    ):
        print("Cache ts basic...")
        self.cache_ts_basic(data, const_range, variable_names, n_jobs)
        print("Cache ts varies stats...")
        self.cache_ts_varies_stats(data, const_range, variable_names, n_jobs)
        print("Cache ema related...")
        self.cache_ema_related(data, const_range, variable_names, n_jobs)
        print("Cache corr related...")
        self.cache_corr_related(data, const_range, variable_names, n_jobs)
        print("Cache ts tech...")
        self.cache_ts_tech(data, const_range, n_jobs)

    def cache_ts_basic(
        self,
        data: pd.DataFrame,
        const_range: tuple,
        variable_names: list[str] = ["open", "high", "low", "close", "volume"],
        n_jobs=1,
    ):
        data = data[variable_names]

        def ts_delay(data, d):
            function_name = "_ts_delay"
            factor = data.groupby(level=1, group_keys=False).shift(d)
            for name in data.columns:
                args = (data[name], d)
                args_hash = "-".join(hash_arg(arg) for arg in args)
                cache_key = f"{function_name}_{args_hash}_"
                hash_key = hashlib.sha256(cache_key.encode()).hexdigest()
                cache_path = os.path.join(self.cache_dir, f"{hash_key}.pkl")
                factor[name].to_pickle(cache_path)

        def ts_delta(data, d):
            function_name = "_ts_delta"
            factor = data.groupby(level=1, group_keys=False).diff(d)
            for name in data.columns:
                args = (data[name], d)
                args_hash = "-".join(hash_arg(arg) for arg in args)
                cache_key = f"{function_name}_{args_hash}_"
                hash_key = hashlib.sha256(cache_key.encode()).hexdigest()
                cache_path = os.path.join(self.cache_dir, f"{hash_key}.pkl")
                factor[name].to_pickle(cache_path)

        def ts_pct_change(data, d):
            function_name = "_ts_pct_change"
            factor = data.groupby(level=1, group_keys=False).pct_change(
                d, fill_method=None
            )
            for name in data.columns:
                args = (data[name], d)
                args_hash = "-".join(hash_arg(arg) for arg in args)
                cache_key = f"{function_name}_{args_hash}_"
                hash_key = hashlib.sha256(cache_key.encode()).hexdigest()
                cache_path = os.path.join(self.cache_dir, f"{hash_key}.pkl")
                factor[name].to_pickle(cache_path)

        def ts_mean_return(data, d):
            function_name = "_ts_mean_return"
            factor = data.groupby(level=1, group_keys=False).pct_change(
                1, fill_method=None
            )
            factor = (
                data.groupby(level=1, group_keys=False)
                .rolling(d, min_periods=int(d / 2))
                .mean()
                .droplevel(0)
                .swaplevel()
                .sort_index()
            )
            for name in data.columns:
                args = (data[name], d)
                args_hash = "-".join(hash_arg(arg) for arg in args)
                cache_key = f"{function_name}_{args_hash}_"
                hash_key = hashlib.sha256(cache_key.encode()).hexdigest()
                cache_path = os.path.join(self.cache_dir, f"{hash_key}.pkl")
                factor[name].to_pickle(cache_path)

        from .Function import _log

        def ts_product(data, d):
            factor = np.exp(
                _log(data)
                .groupby(level=1, group_keys=False)
                .rolling(d, min_periods=int(d / 2))
                .sum()
                .droplevel(0)
                .swaplevel()
                .sort_index()
            )

            function_name = "product"
            for variable_name in data.columns:
                args = (data[variable_name], d)
                args_hash = "-".join(hash_arg(arg) for arg in args)
                cache_key = f"_ts_{function_name}_{args_hash}_"
                hash_key = hashlib.sha256(cache_key.encode()).hexdigest()
                cache_path = os.path.join(self.cache_dir, f"{hash_key}.pkl")
                factor[variable_name].to_pickle(cache_path)

        Parallel(n_jobs=n_jobs)(
            delayed(ts_delay)(data, d)
            for d in range(const_range[0], const_range[1] + 1)
        )
        Parallel(n_jobs=n_jobs)(
            delayed(ts_delta)(data, d)
            for d in range(const_range[0], const_range[1] + 1)
        )
        Parallel(n_jobs=n_jobs)(
            delayed(ts_pct_change)(data, d)
            for d in range(const_range[0], const_range[1] + 1)
        )
        Parallel(n_jobs=n_jobs)(
            delayed(ts_mean_return)(data, d)
            for d in range(const_range[0], const_range[1] + 1)
        )
        Parallel(n_jobs=n_jobs)(
            delayed(ts_product)(data, d)
            for d in range(const_range[0], const_range[1] + 1)
        )

    def _div(self, x1: pd.DataFrame, x2: pd.DataFrame):
        result = x1 / x2
        return result.mask(x2.abs() <= 0.001, 1.0)

    def cache_ts_varies_stats(
        self,
        data: pd.DataFrame,
        const_range: tuple,
        variable_names: list[str] = ["open", "high", "low", "close", "volume"],
        n_jobs=1,
    ):
        data = data[sorted(variable_names)]

        def func(data, d, variable_names):
            factor = (
                (
                    data.groupby(level=1, group_keys=False)
                    .rolling(d, min_periods=int(d / 2))
                    .agg(
                        [
                            "min",
                            "max",
                            "sum",
                            "mean",
                            "std",
                            "median",
                            "skew",
                            "kurt",
                            "rank",
                        ]
                    )
                )
                .swaplevel(axis=1)
                .sort_index(axis=1)
            )

            # midpoint
            tmp = pd.concat(
                [(factor["min"] + factor["max"]) / 2.0], keys=["midpoint"], axis=1
            )
            factor = pd.concat([factor, tmp], axis=1)
            # inverse_cv
            tmp = pd.concat(
                [self._div(factor["mean"], factor["std"])], keys=["inverse_cv"], axis=1
            )
            factor = pd.concat([factor, tmp], axis=1)
            factor = factor.droplevel(0).sort_index()

            # argmin
            tmp = (
                data.groupby(level=1)
                .rolling(d, min_periods=int(d / 2))
                .apply(np.argmin, raw=True, engine="numba")
                .droplevel(0)
                .sort_index()
            )
            tmp = pd.concat([tmp], keys=["argmin"], axis=1)
            factor = pd.concat([factor, tmp], axis=1)
            # argmax
            tmp = (
                data.groupby(level=1)
                .rolling(d, min_periods=int(d / 2))
                .apply(np.argmax, raw=True, engine="numba")
                .droplevel(0)
                .sort_index()
            )
            tmp = pd.concat([tmp], keys=["argmax"], axis=1)
            factor = pd.concat([factor, tmp], axis=1)

            # maxmin
            tmp = pd.concat(
                [self._div(data - factor["min"], factor["max"] - factor["min"])],
                keys=["maxmin"],
                axis=1,
            )
            factor = pd.concat([factor, tmp], axis=1)

            # zscore
            tmp = pd.concat(
                [self._div(data - factor["mean"], factor["std"])],
                keys=["zscore"],
                axis=1,
            )
            factor = pd.concat([factor, tmp], axis=1)

            # argmaxmin
            tmp = pd.concat(
                [factor["argmax"] - factor["argmin"]], keys=["argmaxmin"], axis=1
            )
            factor = pd.concat([factor, tmp], axis=1)

            for function_name in factor.columns.get_level_values(0).unique():
                for variable_name in variable_names:
                    args = (data[variable_name], d)
                    args_hash = "-".join(hash_arg(arg) for arg in args)
                    cache_key = f"_ts_{function_name}_{args_hash}_"
                    hash_key = hashlib.sha256(cache_key.encode()).hexdigest()
                    cache_path = os.path.join(self.cache_dir, f"{hash_key}.pkl")
                    factor[(function_name, variable_name)].to_pickle(cache_path)

        Parallel(n_jobs=n_jobs)(
            delayed(func)(data, d, variable_names)
            for d in range(const_range[0], const_range[1] + 1)
        )

    def cache_ema_related(
        self,
        data: pd.DataFrame,
        const_range: tuple,
        variable_names: list[str] = ["open", "high", "low", "close", "volume"],
        n_jobs=1,
    ):
        data = data[variable_names]
        const_range = range(const_range[0], const_range[1] + 1)
        conditions = itertools.product(variable_names, const_range)

        # cache ts_ema first
        from .Function import _ts_ema

        Parallel(n_jobs=n_jobs)(
            delayed(_ts_ema)(data[variable_name], d) for variable_name, d in conditions
        )

        # cache ts_dema
        from .Function import _ts_dema

        Parallel(n_jobs=n_jobs)(
            delayed(_ts_dema)(data[variable_name], d) for variable_name, d in conditions
        )

        from .Function import _ts_kama

        conditions = itertools.product(
            variable_names, const_range, const_range, const_range
        )
        Parallel(n_jobs=n_jobs)(
            delayed(_ts_kama)(data[variable_name], d1, d2, d3)
            for variable_name, d1, d2, d3 in conditions
        )

    def cache_ts_tech(
        self,
        data: pd.DataFrame,
        const_range: tuple,
        n_jobs=1,
    ):
        const_range = range(const_range[0], const_range[1] + 1)
        from .Function import _ts_CCI

        Parallel(n_jobs=n_jobs)(
            delayed(_ts_CCI)(data["high"], data["low"], data["close"], d)
            for d in const_range
        )

        from .Function import _ts_ATR

        Parallel(n_jobs=n_jobs)(
            delayed(_ts_ATR)(data["high"], data["low"], data["close"], d)
            for d in const_range
        )

        from .Function import _ts_ADX

        Parallel(n_jobs=n_jobs)(
            delayed(_ts_ADX)(data["high"], data["low"], data["close"], d)
            for d in const_range
        )

        from .Function import _ts_MFI

        Parallel(n_jobs=n_jobs)(
            delayed(_ts_MFI)(
                data["high"], data["low"], data["close"], data["volume"], d
            )
            for d in const_range
        )

    def cache_corr_related(
        self,
        data: pd.DataFrame,
        const_range: tuple,
        variable_names: list[str] = ["open", "high", "low", "close", "volume"],
        n_jobs=1,
    ):
        const_range = range(const_range[0], const_range[1] + 1)
        conditions = itertools.product(variable_names, variable_names, const_range)
        conditions = [
            i
            for i in conditions
            if variable_names.index(i[0]) < variable_names.index(i[1])
        ]

        from .Function import _ts_cov

        Parallel(n_jobs=n_jobs)(
            delayed(_ts_cov)(data[x1], data[x2], d) for x1, x2, d in conditions
        )

        from .Function import _ts_corr

        Parallel(n_jobs=n_jobs)(
            delayed(_ts_corr)(data[x1], data[x2], d) for x1, x2, d in conditions
        )

        from .Function import _ts_autocorr

        conditions = itertools.product(variable_names, const_range, const_range)
        Parallel(n_jobs=n_jobs)(
            delayed(_ts_autocorr)(data[x1], d, i) for x1, d, i in conditions
        )

    def test_cache(self, data: pd.DataFrame, d, cache_dir):
        ori_n_files = len(glob(os.path.join(cache_dir, "*.pkl")))

        from gpquant.Function import _ts_delay

        _ts_delay(data["open"], d)
        n_files = len(glob(os.path.join(cache_dir, "*.pkl")))
        assert n_files == ori_n_files, "ts_delay cache fail"

        from gpquant.Function import _ts_delta

        _ts_delta(data["open"], d)
        n_files = len(glob(os.path.join(cache_dir, "*.pkl")))
        assert n_files == ori_n_files, "ts_delta cache fail"

        from gpquant.Function import _ts_pct_change

        _ts_pct_change(data["open"], d)
        n_files = len(glob(os.path.join(cache_dir, "*.pkl")))
        assert n_files == ori_n_files, "ts_pct_change cache fail"

        from gpquant.Function import _ts_mean_return

        _ts_mean_return(data["open"], d)
        n_files = len(glob(os.path.join(cache_dir, "*.pkl")))
        assert n_files == ori_n_files, "ts_mean_return cache fail"

        from gpquant.Function import _ts_product

        _ts_product(data["open"], d)
        n_files = len(glob(os.path.join(cache_dir, "*.pkl")))
        assert n_files == ori_n_files, "ts_product cache fail"

        from gpquant.Function import _ts_min

        _ts_min(data["open"], d)
        n_files = len(glob(os.path.join(cache_dir, "*.pkl")))
        assert n_files == ori_n_files, "ts_min cache fail"

        from gpquant.Function import _ts_max

        _ts_max(data["open"], d)
        n_files = len(glob(os.path.join(cache_dir, "*.pkl")))
        assert n_files == ori_n_files, "ts_max cache fail"

        from gpquant.Function import _ts_sum

        _ts_sum(data["open"], d)
        n_files = len(glob(os.path.join(cache_dir, "*.pkl")))
        assert n_files == ori_n_files, "ts_sum cache fail"

        from gpquant.Function import _ts_mean

        _ts_mean(data["open"], d)
        n_files = len(glob(os.path.join(cache_dir, "*.pkl")))
        assert n_files == ori_n_files, "ts_mean cache fail"

        from gpquant.Function import _ts_std

        _ts_std(data["open"], d)
        n_files = len(glob(os.path.join(cache_dir, "*.pkl")))
        assert n_files == ori_n_files, "ts_std cache fail"

        from gpquant.Function import _ts_median

        _ts_median(data["open"], d)
        n_files = len(glob(os.path.join(cache_dir, "*.pkl")))
        assert n_files == ori_n_files, "ts_median cache fail"

        from gpquant.Function import _ts_skew

        _ts_skew(data["open"], d)
        n_files = len(glob(os.path.join(cache_dir, "*.pkl")))
        assert n_files == ori_n_files, "ts_skew cache fail"

        from gpquant.Function import _ts_kurt

        _ts_kurt(data["open"], d)
        n_files = len(glob(os.path.join(cache_dir, "*.pkl")))
        assert n_files == ori_n_files, "ts_kurt cache fail"

        from gpquant.Function import _ts_rank

        _ts_rank(data["open"], d)
        n_files = len(glob(os.path.join(cache_dir, "*.pkl")))
        assert n_files == ori_n_files, "ts_rank cache fail"

        from gpquant.Function import _ts_midpoint

        _ts_midpoint(data["open"], d)
        n_files = len(glob(os.path.join(cache_dir, "*.pkl")))
        assert n_files == ori_n_files, "ts_midpoint cache fail"

        from gpquant.Function import _ts_inverse_cv

        _ts_inverse_cv(data["open"], d)
        n_files = len(glob(os.path.join(cache_dir, "*.pkl")))
        assert n_files == ori_n_files, "ts_inverse_cv cache fail"

        from gpquant.Function import _ts_argmin

        _ts_argmin(data["open"], d)
        n_files = len(glob(os.path.join(cache_dir, "*.pkl")))
        assert n_files == ori_n_files, "ts_argmin cache fail"

        from gpquant.Function import _ts_argmax

        _ts_argmax(data["open"], d)
        n_files = len(glob(os.path.join(cache_dir, "*.pkl")))
        assert n_files == ori_n_files, "ts_argmax cache fail"

        from gpquant.Function import _ts_maxmin

        _ts_maxmin(data["open"], d)
        n_files = len(glob(os.path.join(cache_dir, "*.pkl")))
        assert n_files == ori_n_files, "ts_maxmin cache fail"

        from gpquant.Function import _ts_zscore

        _ts_zscore(data["open"], d)
        n_files = len(glob(os.path.join(cache_dir, "*.pkl")))
        assert n_files == ori_n_files, "ts_zscore cache fail"

        from gpquant.Function import _ts_argmaxmin

        _ts_argmaxmin(data["open"], d)
        n_files = len(glob(os.path.join(cache_dir, "*.pkl")))
        assert n_files == ori_n_files, "ts_argmaxmin cache fail"

        from gpquant.Function import _ts_ema

        _ts_ema(data["open"], d)
        n_files = len(glob(os.path.join(cache_dir, "*.pkl")))
        assert n_files == ori_n_files, "ts_ema cache fail"

        from gpquant.Function import _ts_dema

        _ts_dema(data["open"], d)
        n_files = len(glob(os.path.join(cache_dir, "*.pkl")))
        assert n_files == ori_n_files, "ts_dema cache fail"

        from gpquant.Function import _ts_kama

        _ts_kama(data["open"], d, d, d)
        n_files = len(glob(os.path.join(cache_dir, "*.pkl")))
        assert n_files == ori_n_files, "ts_kama cache fail"

        from gpquant.Function import _ts_CCI

        _ts_CCI(data["high"], data["low"], data["close"], d)
        n_files = len(glob(os.path.join(cache_dir, "*.pkl")))
        assert n_files == ori_n_files, "ts_CCI cache fail"

        from gpquant.Function import _ts_ATR

        _ts_ATR(data["high"], data["low"], data["close"], d)
        n_files = len(glob(os.path.join(cache_dir, "*.pkl")))
        assert n_files == ori_n_files, "ts_ATR cache fail"

        from gpquant.Function import _ts_ADX

        _ts_ADX(data["high"], data["low"], data["close"], d)
        n_files = len(glob(os.path.join(cache_dir, "*.pkl")))
        assert n_files == ori_n_files, "ts_ADX cache fail"

        from gpquant.Function import _ts_MFI

        _ts_MFI(data["high"], data["low"], data["close"], data["volume"], d)
        n_files = len(glob(os.path.join(cache_dir, "*.pkl")))
        assert n_files == ori_n_files, "ts_MFI cache fail"

        from gpquant.Function import _ts_cov

        _ts_cov(data["high"], data["low"], d)
        n_files = len(glob(os.path.join(cache_dir, "*.pkl")))
        assert n_files == ori_n_files, "ts_cov cache fail"

        from gpquant.Function import _ts_corr

        _ts_corr(data["high"], data["low"], d)
        n_files = len(glob(os.path.join(cache_dir, "*.pkl")))
        assert n_files == ori_n_files, "ts_corr cache fail"

        from gpquant.Function import _ts_autocorr

        _ts_autocorr(data["open"], d, d)
        n_files = len(glob(os.path.join(cache_dir, "*.pkl")))
        assert n_files == ori_n_files, "ts_autocorr cache fail"
