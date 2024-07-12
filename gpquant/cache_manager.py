import os
import pandas as pd
import shutil
import hashlib
from joblib import Parallel, delayed
import numpy as np
import itertools


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
        variable_names: list[str] = ['open', 'high', 'low', 'close', 'volume'],
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
        variable_names: list[str] = ['open', 'high', 'low', 'close', 'volume'],
        n_jobs=1,
    ):
        data = data[variable_names]

        def ts_delay(data, d):
            function_name = '_ts_delay'
            factor = data.groupby(level=1, group_keys=False).shift(d)
            for name in data.columns:
                args = (data[name], d)
                args_hash = '-'.join(hash_arg(arg) for arg in args)
                cache_key = f"{function_name}_{args_hash}_"
                hash_key = hashlib.sha256(cache_key.encode()).hexdigest()
                cache_path = os.path.join(self.cache_dir, f"{hash_key}.pkl")
                factor[name].to_pickle(cache_path)

        def ts_delta(data, d):
            function_name = '_ts_delta'
            factor = data.groupby(level=1, group_keys=False).diff(d)
            for name in data.columns:
                args = (data[name], d)
                args_hash = '-'.join(hash_arg(arg) for arg in args)
                cache_key = f"{function_name}_{args_hash}_"
                hash_key = hashlib.sha256(cache_key.encode()).hexdigest()
                cache_path = os.path.join(self.cache_dir, f"{hash_key}.pkl")
                factor[name].to_pickle(cache_path)

        def ts_pct_change(data, d):
            function_name = '_ts_pct_change'
            factor = data.groupby(level=1, group_keys=False).pct_change(
                d, fill_method=None
            )
            for name in data.columns:
                args = (data[name], d)
                args_hash = '-'.join(hash_arg(arg) for arg in args)
                cache_key = f"{function_name}_{args_hash}_"
                hash_key = hashlib.sha256(cache_key.encode()).hexdigest()
                cache_path = os.path.join(self.cache_dir, f"{hash_key}.pkl")
                factor[name].to_pickle(cache_path)

        def ts_mean_return(data, d):
            function_name = '_ts_mean_return'
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
                args_hash = '-'.join(hash_arg(arg) for arg in args)
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

            function_name = 'product'
            for variable_name in data.columns:
                args = (data[variable_name], d)
                args_hash = '-'.join(hash_arg(arg) for arg in args)
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
        variable_names: list[str] = ['open', 'high', 'low', 'close', 'volume'],
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
                            'min',
                            'max',
                            'sum',
                            'mean',
                            'std',
                            'median',
                            'skew',
                            'kurt',
                            'rank',
                        ]
                    )
                )
                .swaplevel(axis=1)
                .sort_index(axis=1)
            )

            # midpoint
            tmp = pd.concat(
                [(factor['min'] + factor['max']) / 2.0], keys=['midpoint'], axis=1
            )
            factor = pd.concat([factor, tmp], axis=1)
            # inverse_cv
            tmp = pd.concat(
                [self._div(factor['mean'], factor['std'])], keys=['inverse_cv'], axis=1
            )
            factor = pd.concat([factor, tmp], axis=1)
            factor = factor.droplevel(0).sort_index()

            # argmin
            tmp = (
                data.groupby(level=1)
                .rolling(d, min_periods=int(d / 2))
                .apply(np.argmin, raw=True, engine='numba')
                .droplevel(0)
                .sort_index()
            )
            tmp = pd.concat([tmp], keys=['argmin'], axis=1)
            factor = pd.concat([factor, tmp], axis=1)
            # argmax
            tmp = (
                data.groupby(level=1)
                .rolling(d, min_periods=int(d / 2))
                .apply(np.argmax, raw=True, engine='numba')
                .droplevel(0)
                .sort_index()
            )
            tmp = pd.concat([tmp], keys=['argmax'], axis=1)
            factor = pd.concat([factor, tmp], axis=1)

            # maxmin
            tmp = pd.concat(
                [self._div(data - factor['min'], factor['max'] - factor['min'])],
                keys=['maxmin'],
                axis=1,
            )
            factor = pd.concat([factor, tmp], axis=1)

            # zscore
            tmp = pd.concat(
                [self._div(data - factor['mean'], factor['std'])],
                keys=['zscore'],
                axis=1,
            )
            factor = pd.concat([factor, tmp], axis=1)

            # argmaxmin
            tmp = pd.concat(
                [factor['argmax'] - factor['argmin']], keys=['argmaxmin'], axis=1
            )
            factor = pd.concat([factor, tmp], axis=1)

            for function_name in factor.columns.get_level_values(0).unique():
                for variable_name in variable_names:
                    args = (data[variable_name], d)
                    args_hash = '-'.join(hash_arg(arg) for arg in args)
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
        variable_names: list[str] = ['open', 'high', 'low', 'close', 'volume'],
        n_jobs=1,
    ):
        data = data[variable_names]
        const_range = range(const_range[0], const_range[1] + 1)
        conditions = itertools.product(variable_names, const_range)

        # cache ts_ema first
        print("Cache ts_ema...")
        from .Function import _ts_ema

        Parallel(n_jobs=n_jobs)(
            delayed(_ts_ema)(data[variable_name], d) for variable_name, d in conditions
        )

        print("Cache ts_dema...")
        # cache ts_dema
        from .Function import _ts_dema

        Parallel(n_jobs=n_jobs)(
            delayed(_ts_dema)(data[variable_name], d) for variable_name, d in conditions
        )

        print("Cache ts_kama")
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
        print("Cache ts CCI..")
        from .Function import _ts_CCI

        Parallel(n_jobs=n_jobs)(
            delayed(_ts_CCI)(data['high'], data['low'], data['close'], d)
            for d in const_range
        )

        print("Cache ts ATR..")
        from .Function import _ts_ATR

        Parallel(n_jobs=n_jobs)(
            delayed(_ts_ATR)(data['high'], data['low'], data['close'], d)
            for d in const_range
        )

        print("Cache ts ADX..")
        from .Function import _ts_ADX

        Parallel(n_jobs=n_jobs)(
            delayed(_ts_ADX)(data['high'], data['low'], data['close'], d)
            for d in const_range
        )

        print("Cache ts MFI..")
        from .Function import _ts_MFI

        Parallel(n_jobs=n_jobs)(
            delayed(_ts_MFI)(
                data['high'], data['low'], data['close'], data['volume'], d
            )
            for d in const_range
        )

    def cache_corr_related(
        self,
        data: pd.DataFrame,
        const_range: tuple,
        variable_names: list[str] = ['open', 'high', 'low', 'close', 'volume'],
        n_jobs=1,
    ):
        const_range = range(const_range[0], const_range[1] + 1)
        conditions = itertools.product(variable_names, variable_names, const_range)
        conditions = [
            i
            for i in conditions
            if variable_names.index(i[0]) < variable_names.index(i[1])
        ]

        print("Cache ts cov")
        from .Function import _ts_cov

        Parallel(n_jobs=n_jobs)(
            delayed(_ts_cov)(data[x1], data[x2], d) for x1, x2, d in conditions
        )

        print("Cache ts corr")
        from .Function import _ts_corr

        Parallel(n_jobs=n_jobs)(
            delayed(_ts_corr)(data[x1], data[x2], d) for x1, x2, d in conditions
        )

        print("Cache ts autocorr")
        from .Function import _ts_autocorr

        conditions = itertools.product(variable_names, const_range, const_range)
        Parallel(n_jobs=n_jobs)(
            delayed(_ts_autocorr)(data[x1], d, i) for x1, d, i in conditions
        )
