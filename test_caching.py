from gpquant.cache_manager import CacheManager
import pandas as pd
from joblib import Parallel, delayed


if __name__ == "__main__":
    manager = CacheManager(cache_dir="./cache")
    manager.clean_cache()

    data = pd.read_pickle("./data.pkl")
    const_range = (5, 5)

    print("Cache all...")
    manager.cache_all(data, const_range=const_range, n_jobs=4)

    print("test  cache")
    manager.test_cache(data, d=5, cache_dir="~/researches/gpquant/cache")
