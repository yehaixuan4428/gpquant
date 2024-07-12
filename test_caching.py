from gpquant.cache_manager import CacheManager
import pandas as pd


if __name__ == "__main__":
    manager = CacheManager(cache_dir='./cache')
    manager.clean_cache()

    data = pd.read_pickle('./data.pkl')
    const_range = (5, 5)

    manager.cache_all(data, const_range=const_range, n_jobs=1)
