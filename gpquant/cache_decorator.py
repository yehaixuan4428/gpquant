import functools
import os
import pickle
import pandas as pd
import numpy as np
from typing import Callable, Any
import hashlib
from multiprocessing import Lock, Pool


def series_hash(s: pd.Series) -> str:
    return hashlib.md5(pd.util.hash_pandas_object(s).values).hexdigest()


class FileLock:
    def __init__(self, file_path):
        self.lock = Lock()
        self.file_path = file_path

    def __enter__(self):
        self.lock.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.lock.release()


def cache_decorator():
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, cache_dir: str = './cache', **kwargs) -> Any:
            # this is called in the top layer. So cache_dir exists
            # os.makedirs(cache_dir, exist_ok=True)

            def hash_arg(arg):
                if isinstance(arg, pd.Series):
                    return series_hash(arg)
                elif isinstance(arg, (int, float, str, bool)):
                    return str(arg)
                else:
                    return hashlib.md5(str(arg).encode()).hexdigest()

            args_hash = '-'.join(hash_arg(arg) for arg in args)
            kwargs_hash = '-'.join(f"{k}:{hash_arg(v)}" for k, v in kwargs.items())

            cache_key = f"{func.__name__}_{args_hash}_{kwargs_hash}"
            hash_key = hashlib.sha256(cache_key.encode()).hexdigest()

            cache_path = os.path.join(cache_dir, f"{hash_key}.pkl")
            lock_path = os.path.join(cache_dir, f"{hash_key}.lock")

            with FileLock(lock_path):
                if os.path.exists(cache_path):
                    with open(cache_path, 'rb') as f:
                        return pickle.load(f)

                kwargs_without_cache_dir = {
                    k: v for k, v in kwargs.items() if k != 'cache_dir'
                }

                result = func(*args, **kwargs_without_cache_dir)

                if isinstance(result, pd.Series):
                    with open(cache_path, 'wb') as f:
                        pickle.dump(result, f)

                return result

        return wrapper

    return decorator


if __name__ == "__main__":

    @cache_decorator()
    def square(x1):
        return x1**2

    @cache_decorator()
    def add(x1, d):
        return x1 + d

    data = pd.read_pickle('./data.pkl')
    # print(add(data['close'], 2))
    print(add(data['close'], data['open'], cache_dir='./cache'))
    # print(add(3, 2))

    # with Pool(processes=7) as pool:
    #     results = pool.map(square, [data['close']] * 7)
    # print(results)
