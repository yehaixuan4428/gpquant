import os
import pandas as pd
import numpy as np

if __name__ == "__main__":
    data = pd.read_pickle('./data.pkl')
    data = [data['high'], data['low'], data['open'], data['volume']]
    vars = np.int64([5, 6, 7])
    vars_float = np.float64([5.0, 6.0, 7.0])

    # one param functions

    from gpquant.Function import _square as func

    output = func(data[0])
    assert isinstance(output, pd.Series), type(output)
    np.testing.assert_array_equal(output.index, data[0].index)
    output = func(vars[0])
    assert isinstance(output, np.int64), type(output)
    output = func(vars_float[0])
    assert isinstance(output, np.float64), type(output)

    from gpquant.Function import _sqrt as func

    output = func(data[0])
    assert isinstance(output, pd.Series), type(output)
    np.testing.assert_array_equal(output.index, data[0].index)
    output = func(vars[0])
    assert isinstance(output, np.float64), type(output)
    output = func(vars_float[0])
    assert isinstance(output, np.float64), type(output)

    from gpquant.Function import _cube as func

    output = func(data[0])
    assert isinstance(output, pd.Series), type(output)
    np.testing.assert_array_equal(output.index, data[0].index)
    output = func(vars[0])
    assert isinstance(output, np.int64), type(output)
    output = func(vars_float[0])
    assert isinstance(output, np.float64), type(output)

    from gpquant.Function import _cbrt as func

    output = func(data[0])
    assert isinstance(output, pd.Series), type(output)
    np.testing.assert_array_equal(output.index, data[0].index)
    output = func(vars[0])
    assert isinstance(output, np.float64), type(output)
    output = func(vars_float[0])
    assert isinstance(output, np.float64), type(output)

    from gpquant.Function import _sign as func

    output = func(data[0])
    assert isinstance(output, pd.Series), type(output)
    np.testing.assert_array_equal(output.index, data[0].index)
    output = func(vars[0])
    assert isinstance(output, np.int64), type(output)
    output = func(vars_float[0])
    assert isinstance(output, np.float64), type(output)

    from gpquant.Function import _neg as func

    output = func(data[0])
    assert isinstance(output, pd.Series), type(output)
    np.testing.assert_array_equal(output.index, data[0].index)
    output = func(vars[0])
    assert isinstance(output, np.int64), type(output)
    output = func(vars_float[0])
    assert isinstance(output, np.float64), type(output)

    from gpquant.Function import _inv as func

    output = func(data[0])
    assert isinstance(output, pd.Series), type(output)
    np.testing.assert_array_equal(output.index, data[0].index)
    output = func(vars[0])
    assert isinstance(output, float) | isinstance(output, np.float64), type(output)
    output = func(vars_float[0])
    assert isinstance(output, float) | isinstance(output, np.float64), type(output)

    from gpquant.Function import _abs as func

    output = func(data[0])
    assert isinstance(output, pd.Series), type(output)
    np.testing.assert_array_equal(output.index, data[0].index)
    output = func(vars[0])
    assert isinstance(output, np.int64) | isinstance(output, np.int64), type(output)
    output = func(vars_float[0])
    assert isinstance(output, float) | isinstance(output, np.float64), type(output)

    from gpquant.Function import _sin as func

    output = func(data[0])
    assert isinstance(output, pd.Series), type(output)
    np.testing.assert_array_equal(output.index, data[0].index)
    output = func(vars[0])
    assert isinstance(output, float) | isinstance(output, np.float64), type(output)
    output = func(vars_float[0])
    assert isinstance(output, float) | isinstance(output, np.float64), type(output)

    from gpquant.Function import _cos as func

    output = func(data[0])
    assert isinstance(output, pd.Series), type(output)
    np.testing.assert_array_equal(output.index, data[0].index)
    output = func(vars[0])
    assert isinstance(output, float) | isinstance(output, np.float64), type(output)
    output = func(vars_float[0])
    assert isinstance(output, float) | isinstance(output, np.float64), type(output)

    from gpquant.Function import _tan as func

    output = func(data[0])
    assert isinstance(output, pd.Series), type(output)
    np.testing.assert_array_equal(output.index, data[0].index)
    output = func(vars[0])
    assert isinstance(output, float) | isinstance(output, np.float64), type(output)
    output = func(vars_float[0])
    assert isinstance(output, float) | isinstance(output, np.float64), type(output)

    from gpquant.Function import _log as func

    output = func(data[0])
    assert isinstance(output, pd.Series), type(output)
    np.testing.assert_array_equal(output.index, data[0].index)
    output = func(vars[0])
    assert isinstance(output, float) | isinstance(output, np.float64), type(output)
    output = func(vars_float[0])
    assert isinstance(output, float) | isinstance(output, np.float64), type(output)

    from gpquant.Function import _sig as func

    output = func(data[0])
    assert isinstance(output, pd.Series), type(output)
    np.testing.assert_array_equal(output.index, data[0].index)
    output = func(vars[0])
    assert isinstance(output, float) | isinstance(output, np.float64), type(output)
    output = func(vars_float[0])
    assert isinstance(output, float) | isinstance(output, np.float64), type(output)

    # two param functions
    from gpquant.Function import _add as func

    output = func(data[0], data[1])
    assert isinstance(output, pd.Series), type(output)
    np.testing.assert_array_equal(output.index, data[0].index)
    output = func(data[0], vars_float[0])
    assert isinstance(output, pd.Series), type(output)
    np.testing.assert_array_equal(output.index, data[0].index)
    output = func(vars_float[0], vars_float[1])
    assert isinstance(output, float) | isinstance(output, np.float64), type(output)

    from gpquant.Function import _sub as func

    output = func(data[0], data[1])
    assert isinstance(output, pd.Series), type(output)
    np.testing.assert_array_equal(output.index, data[0].index)
    output = func(data[0], vars_float[0])
    assert isinstance(output, pd.Series), type(output)
    np.testing.assert_array_equal(output.index, data[0].index)
    output = func(vars_float[0], data[0])
    assert isinstance(output, pd.Series), type(output)
    np.testing.assert_array_equal(output.index, data[0].index)
    output = func(vars_float[0], vars_float[1])
    assert isinstance(output, float) | isinstance(output, np.float64), type(output)

    from gpquant.Function import _mul as func

    output = func(data[0], data[1])
    assert isinstance(output, pd.Series), type(output)
    np.testing.assert_array_equal(output.index, data[0].index)
    output = func(data[0], vars_float[0])
    assert isinstance(output, pd.Series), type(output)
    np.testing.assert_array_equal(output.index, data[0].index)
    output = func(vars_float[0], data[0])
    assert isinstance(output, pd.Series), type(output)
    np.testing.assert_array_equal(output.index, data[0].index)
    output = func(vars_float[0], vars_float[1])
    assert isinstance(output, float) | isinstance(output, np.float64), type(output)

    from gpquant.Function import _div as func

    output = func(data[0], data[1])
    assert isinstance(output, pd.Series), type(output)
    np.testing.assert_array_equal(output.index, data[0].index)
    output = func(data[0], vars_float[0])
    assert isinstance(output, pd.Series), type(output)
    np.testing.assert_array_equal(output.index, data[0].index)
    output = func(vars_float[0], data[0])
    assert isinstance(output, pd.Series), type(output)
    np.testing.assert_array_equal(output.index, data[0].index)
    output = func(vars_float[0], vars_float[1])
    assert isinstance(output, float) | isinstance(output, np.float64), type(output)

    from gpquant.Function import _max as func

    output = func(data[0], data[1])
    assert isinstance(output, pd.Series), type(output)
    np.testing.assert_array_equal(output.index, data[0].index)
    output = func(data[0], vars_float[0])
    assert isinstance(output, pd.Series), type(output)
    np.testing.assert_array_equal(output.index, data[0].index)
    output = func(vars_float[0], data[0])
    assert isinstance(output, pd.Series), type(output)
    np.testing.assert_array_equal(output.index, data[0].index)
    output = func(vars_float[0], vars_float[1])
    assert isinstance(output, float) | isinstance(output, np.float64), type(output)

    from gpquant.Function import _max as func

    output = func(data[0], data[1])
    assert isinstance(output, pd.Series), type(output)
    np.testing.assert_array_equal(output.index, data[0].index)
    output = func(data[0], vars_float[0])
    assert isinstance(output, pd.Series), type(output)
    np.testing.assert_array_equal(output.index, data[0].index)
    output = func(vars_float[0], data[0])
    assert isinstance(output, pd.Series), type(output)
    np.testing.assert_array_equal(output.index, data[0].index)
    output = func(vars_float[0], vars_float[1])
    assert isinstance(output, float) | isinstance(output, np.float64), type(output)

    from gpquant.Function import _min as func

    output = func(data[0], data[1])
    assert isinstance(output, pd.Series), type(output)
    np.testing.assert_array_equal(output.index, data[0].index)
    output = func(data[0], vars_float[0])
    assert isinstance(output, pd.Series), type(output)
    np.testing.assert_array_equal(output.index, data[0].index)
    output = func(vars_float[0], data[0])
    assert isinstance(output, pd.Series), type(output)
    np.testing.assert_array_equal(output.index, data[0].index)
    output = func(vars_float[0], vars_float[1])
    assert isinstance(output, float) | isinstance(output, np.float64), type(output)

    from gpquant.Function import _mean as func

    output = func(data[0], data[1])
    assert isinstance(output, pd.Series), type(output)
    np.testing.assert_array_equal(output.index, data[0].index)
    output = func(data[0], vars_float[0])
    assert isinstance(output, pd.Series), type(output)
    np.testing.assert_array_equal(output.index, data[0].index)
    output = func(vars_float[0], data[0])
    assert isinstance(output, pd.Series), type(output)
    np.testing.assert_array_equal(output.index, data[0].index)
    output = func(vars_float[0], vars_float[1])
    assert isinstance(output, float) | isinstance(output, np.float64), type(output)

    # three params
    from gpquant.Function import _clear_by_cond as func

    output = func(data[0], data[1], data[2])
    assert isinstance(output, pd.Series), type(output)
    np.testing.assert_array_equal(output.index, data[0].index)

    output = func(data[0], vars_float[0], data[2])
    assert isinstance(output, pd.Series), type(output)
    np.testing.assert_array_equal(output.index, data[0].index)

    output = func(vars_float[0], data[0], data[2])
    assert isinstance(output, pd.Series), type(output)
    np.testing.assert_array_equal(output.index, data[0].index)

    output = func(data[0], vars_float[0], vars_float[1])
    assert isinstance(output, pd.Series), type(output)
    np.testing.assert_array_equal(output.index, data[0].index)

    output = func(vars_float[0], data[0], vars_float[1])
    assert isinstance(output, pd.Series), type(output)
    np.testing.assert_array_equal(output.index, data[0].index)

    output = func(vars_float[0], vars_float[1], data[0])
    assert np.isnan(output)

    output = func(vars_float[0], vars_float[1], vars_float[2])
    assert np.isnan(output)

    from gpquant.Function import _if_then_else as func

    output = func(data[0], data[1], data[2])
    assert isinstance(output, pd.Series), type(output)
    np.testing.assert_array_equal(output.index, data[0].index)

    output = func(data[0], vars_float[0], data[2])
    assert isinstance(output, pd.Series), type(output)
    np.testing.assert_array_equal(output.index, data[0].index)

    output = func(vars_float[0], data[0], data[2])
    assert np.isnan(output)

    output = func(data[0], vars_float[0], vars_float[1])
    assert isinstance(output, pd.Series), type(output)
    np.testing.assert_array_equal(output.index, data[0].index)

    output = func(vars_float[0], data[0], vars_float[1])
    assert np.isnan(output)

    output = func(vars_float[0], vars_float[1], data[0])
    assert np.isnan(output)

    output = func(vars_float[0], vars_float[1], vars_float[2])
    assert np.isnan(output)

    from gpquant.Function import _if_cond_then_else as func

    output = func(vars_float[0], vars_float[1], data[2], data[3])
    assert np.isnan(output)

    output = func(data[0], data[1], data[2], data[3])
    assert isinstance(output, pd.Series), type(output)
    np.testing.assert_array_equal(output.index, data[0].index)

    output = func(data[0], vars_float[0], data[2], data[3])
    assert isinstance(output, pd.Series), type(output)
    np.testing.assert_array_equal(output.index, data[0].index)

    output = func(vars_float[0], data[0], data[2], data[3])
    assert isinstance(output, pd.Series), type(output)
    np.testing.assert_array_equal(output.index, data[0].index)

    output = func(data[0], data[1], vars_float[0], data[2])
    assert isinstance(output, pd.Series), type(output)
    np.testing.assert_array_equal(output.index, data[0].index)

    output = func(data[0], data[1], data[2], vars_float[0])
    assert isinstance(output, pd.Series), type(output)
    np.testing.assert_array_equal(output.index, data[0].index)

    output = func(data[0], data[1], vars_float[0], vars_float[1])
    assert isinstance(output, pd.Series), type(output)
    np.testing.assert_array_equal(output.index, data[0].index)

    from gpquant.Function import _cs_count as func

    output = func(data[0])
    assert isinstance(output, pd.Series), type(output)
    np.testing.assert_array_equal(output.index, data[0].index)
    output = func(vars_float[0])
    assert np.isnan(output)

    from gpquant.Function import _ts_delay as func

    output = func(data[0], vars[0])
    assert isinstance(output, pd.Series), type(output)
    np.testing.assert_array_equal(output.index, data[0].index)

    output = func(data[0], vars_float[1])
    assert np.isnan(output)

    output = func(data[0], data[1])
    assert np.isnan(output)

    output = func(vars[0], data[0])
    assert np.isnan(output)

    output = func(vars[0], vars[1])
    assert np.isnan(output)

    from gpquant.Function import _ts_delta as func

    output = func(data[0], vars[0])
    assert isinstance(output, pd.Series), type(output)
    np.testing.assert_array_equal(output.index, data[0].index)

    output = func(data[0], vars_float[1])
    assert np.isnan(output)

    output = func(data[0], data[1])
    assert np.isnan(output)

    output = func(vars[0], data[0])
    assert np.isnan(output)

    output = func(vars[0], vars[1])
    assert np.isnan(output)

    from gpquant.Function import _ts_pct_change as func

    output = func(data[0], vars[0])
    assert isinstance(output, pd.Series), type(output)
    np.testing.assert_array_equal(output.index, data[0].index)

    output = func(data[0], vars_float[1])
    assert np.isnan(output)

    output = func(data[0], data[1])
    assert np.isnan(output)

    output = func(vars[0], data[0])
    assert np.isnan(output)

    output = func(vars[0], vars[1])
    assert np.isnan(output)

    from gpquant.Function import _ts_mean_return as func

    output = func(data[0], vars[0])
    assert isinstance(output, pd.Series), type(output)
    np.testing.assert_array_equal(output.index, data[0].index)

    output = func(data[0], vars_float[1])
    assert np.isnan(output)

    output = func(data[0], data[1])
    assert np.isnan(output)

    output = func(vars[0], data[0])
    assert np.isnan(output)

    output = func(vars[0], vars[1])
    assert np.isnan(output)

    from gpquant.Function import _ts_max as func

    output = func(data[0], vars[0])
    assert isinstance(output, pd.Series), type(output)
    np.testing.assert_array_equal(output.index, data[0].index)

    output = func(data[0], vars_float[1])
    assert np.isnan(output)

    output = func(data[0], data[1])
    assert np.isnan(output)

    output = func(vars[0], data[0])
    assert np.isnan(output)

    output = func(vars[0], vars[1])
    assert np.isnan(output)

    from gpquant.Function import _ts_min as func

    output = func(data[0], vars[0])
    assert isinstance(output, pd.Series), type(output)
    np.testing.assert_array_equal(output.index, data[0].index)

    output = func(data[0], vars_float[1])
    assert np.isnan(output)

    output = func(data[0], data[1])
    assert np.isnan(output)

    output = func(vars[0], data[0])
    assert np.isnan(output)

    output = func(vars[0], vars[1])
    assert np.isnan(output)

    from gpquant.Function import _ts_sum as func

    output = func(data[0], vars[0])
    assert isinstance(output, pd.Series), type(output)
    np.testing.assert_array_equal(output.index, data[0].index)

    output = func(data[0], vars_float[1])
    assert np.isnan(output)

    output = func(data[0], data[1])
    assert np.isnan(output)

    output = func(vars[0], data[0])
    assert np.isnan(output)

    output = func(vars[0], vars[1])
    assert np.isnan(output)

    from gpquant.Function import _ts_product as func

    output = func(data[0], vars[0])
    assert isinstance(output, pd.Series), type(output)
    np.testing.assert_array_equal(output.index, data[0].index)

    output = func(data[0], vars_float[1])
    assert np.isnan(output)

    output = func(data[0], data[1])
    assert np.isnan(output)

    output = func(vars[0], data[0])
    assert np.isnan(output)

    output = func(vars[0], vars[1])
    assert np.isnan(output)

    from gpquant.Function import _ts_mean as func

    output = func(data[0], vars[0])
    assert isinstance(output, pd.Series), type(output)
    np.testing.assert_array_equal(output.index, data[0].index)

    output = func(data[0], vars_float[1])
    assert np.isnan(output)

    output = func(data[0], data[1])
    assert np.isnan(output)

    output = func(vars[0], data[0])
    assert np.isnan(output)

    output = func(vars[0], vars[1])
    assert np.isnan(output)

    from gpquant.Function import _ts_std as func

    output = func(data[0], vars[0])
    assert isinstance(output, pd.Series), type(output)
    np.testing.assert_array_equal(output.index, data[0].index)

    output = func(data[0], vars_float[1])
    assert np.isnan(output)

    output = func(data[0], data[1])
    assert np.isnan(output)

    output = func(vars[0], data[0])
    assert np.isnan(output)

    output = func(vars[0], vars[1])
    assert np.isnan(output)

    from gpquant.Function import _ts_median as func

    output = func(data[0], vars[0])
    assert isinstance(output, pd.Series), type(output)
    np.testing.assert_array_equal(output.index, data[0].index)

    output = func(data[0], vars_float[1])
    assert np.isnan(output)

    output = func(data[0], data[1])
    assert np.isnan(output)

    output = func(vars[0], data[0])
    assert np.isnan(output)

    output = func(vars[0], vars[1])
    assert np.isnan(output)

    from gpquant.Function import _ts_midpoint as func

    output = func(data[0], vars[0])
    assert isinstance(output, pd.Series), type(output)
    np.testing.assert_array_equal(output.index, data[0].index)

    output = func(data[0], vars_float[1])
    assert np.isnan(output)

    output = func(data[0], data[1])
    assert np.isnan(output)

    output = func(vars[0], data[0])
    assert np.isnan(output)

    output = func(vars[0], vars[1])
    assert np.isnan(output)

    from gpquant.Function import _ts_skew as func

    output = func(data[0], vars[0])
    assert isinstance(output, pd.Series), type(output)
    np.testing.assert_array_equal(output.index, data[0].index)

    output = func(data[0], vars_float[1])
    assert np.isnan(output)

    output = func(data[0], data[1])
    assert np.isnan(output)

    output = func(vars[0], data[0])
    assert np.isnan(output)

    output = func(vars[0], vars[1])
    assert np.isnan(output)

    from gpquant.Function import _ts_kurt as func

    output = func(data[0], vars[0])
    assert isinstance(output, pd.Series), type(output)
    np.testing.assert_array_equal(output.index, data[0].index)

    output = func(data[0], vars_float[1])
    assert np.isnan(output)

    output = func(data[0], data[1])
    assert np.isnan(output)

    output = func(vars[0], data[0])
    assert np.isnan(output)

    output = func(vars[0], vars[1])
    assert np.isnan(output)

    from gpquant.Function import _ts_inverse_cv as func

    output = func(data[0], vars[0])
    assert isinstance(output, pd.Series), type(output)
    np.testing.assert_array_equal(output.index, data[0].index)

    output = func(data[0], vars_float[1])
    assert np.isnan(output)

    output = func(data[0], data[1])
    assert np.isnan(output)

    output = func(vars[0], data[0])
    assert np.isnan(output)

    output = func(vars[0], vars[1])
    assert np.isnan(output)

    from gpquant.Function import _ts_cov as func

    output = func(data[0], data[1], vars[0])
    assert isinstance(output, pd.Series), type(output)
    np.testing.assert_array_equal(output.index, data[0].index)

    output = func(data[0], data[1], vars_float[0])
    assert np.isnan(output)

    output = func(data[0], vars_float[0], vars[0])
    assert np.isnan(output)

    output = func(vars_float[0], data[0], vars[0])
    assert np.isnan(output)

    output = func(vars_float[0], vars_float[1], vars[0])
    assert np.isnan(output)

    from gpquant.Function import _ts_corr as func

    output = func(data[0], data[1], vars[0])
    assert isinstance(output, pd.Series), type(output)
    np.testing.assert_array_equal(output.index, data[0].index)

    output = func(data[0], data[1], vars_float[0])
    assert np.isnan(output)

    output = func(data[0], vars_float[0], vars[0])
    assert np.isnan(output)

    output = func(vars_float[0], data[0], vars[0])
    assert np.isnan(output)

    output = func(vars_float[0], vars_float[1], vars[0])
    assert np.isnan(output)

    from gpquant.Function import _ts_autocorr as func

    output = func(data[0], vars[0], vars[1])
    assert isinstance(output, pd.Series), type(output)
    np.testing.assert_array_equal(output.index, data[0].index)

    output = func(data[0], data[1], vars[1])
    assert np.isnan(output)

    output = func(data[0], vars[1], vars_float[1])
    assert np.isnan(output)

    output = func(data[0], vars_float[1], vars[1])
    assert np.isnan(output)

    from gpquant.Function import _ts_maxmin as func

    output = func(data[0], vars[0])
    assert isinstance(output, pd.Series), type(output)
    np.testing.assert_array_equal(output.index, data[0].index)

    output = func(data[0], vars_float[1])
    assert np.isnan(output)

    output = func(data[0], data[1])
    assert np.isnan(output)

    output = func(vars[0], data[0])
    assert np.isnan(output)

    output = func(vars[0], vars[1])
    assert np.isnan(output)

    from gpquant.Function import _ts_zscore as func

    output = func(data[0], vars[0])
    assert isinstance(output, pd.Series), type(output)
    np.testing.assert_array_equal(output.index, data[0].index)

    output = func(data[0], vars_float[1])
    assert np.isnan(output)

    output = func(data[0], data[1])
    assert np.isnan(output)

    output = func(vars[0], data[0])
    assert np.isnan(output)

    output = func(vars[0], vars[1])
    assert np.isnan(output)

    from gpquant.Function import _ts_regression_beta as func

    output = func(data[0], data[1], vars[0])
    assert isinstance(output, pd.Series), type(output)
    np.testing.assert_array_equal(output.index, data[0].index)

    output = func(data[0], data[1], vars_float[0])
    assert np.isnan(output)

    output = func(data[0], vars_float[0], vars[0])
    assert np.isnan(output)

    output = func(vars_float[0], data[0], vars[0])
    assert np.isnan(output)

    output = func(vars_float[0], vars_float[1], vars[0])
    assert np.isnan(output)

    from gpquant.Function import _ts_regression_alpha as func

    output = func(data[0], data[1], vars[0])
    assert isinstance(output, pd.Series), type(output)
    np.testing.assert_array_equal(output.index, data[0].index)

    output = func(data[0], data[1], vars_float[0])
    assert np.isnan(output)

    output = func(data[0], vars_float[0], vars[0])
    assert np.isnan(output)

    output = func(vars_float[0], data[0], vars[0])
    assert np.isnan(output)

    output = func(vars_float[0], vars_float[1], vars[0])
    assert np.isnan(output)

    from gpquant.Function import _ts_linear_slope as func

    output = func(data[0], vars[0])
    assert isinstance(output, pd.Series), type(output)
    np.testing.assert_array_equal(output.index, data[0].index)

    output = func(data[0], vars_float[1])
    assert np.isnan(output)

    output = func(data[0], data[1])
    assert np.isnan(output)

    output = func(vars[0], data[0])
    assert np.isnan(output)

    output = func(vars[0], vars[1])
    assert np.isnan(output)

    from gpquant.Function import _ts_linear_slope as func

    output = func(data[0], vars[0])
    assert isinstance(output, pd.Series), type(output)
    np.testing.assert_array_equal(output.index, data[0].index)

    output = func(data[0], vars_float[1])
    assert np.isnan(output)

    output = func(data[0], data[1])
    assert np.isnan(output)

    output = func(vars[0], data[0])
    assert np.isnan(output)

    output = func(vars[0], vars[1])
    assert np.isnan(output)

    from gpquant.Function import _ts_argmax as func

    output = func(data[0], vars[0])
    assert isinstance(output, pd.Series), type(output)
    np.testing.assert_array_equal(output.index, data[0].index)

    output = func(data[0], vars_float[1])
    assert np.isnan(output)

    output = func(data[0], data[1])
    assert np.isnan(output)

    output = func(vars[0], data[0])
    assert np.isnan(output)

    output = func(vars[0], vars[1])
    assert np.isnan(output)

    from gpquant.Function import _ts_argmin as func

    output = func(data[0], vars[0])
    assert isinstance(output, pd.Series), type(output)
    np.testing.assert_array_equal(output.index, data[0].index)

    output = func(data[0], vars_float[1])
    assert np.isnan(output)

    output = func(data[0], data[1])
    assert np.isnan(output)

    output = func(vars[0], data[0])
    assert np.isnan(output)

    output = func(vars[0], vars[1])
    assert np.isnan(output)

    from gpquant.Function import _ts_argmaxmin as func

    output = func(data[0], vars[0])
    assert isinstance(output, pd.Series), type(output)
    np.testing.assert_array_equal(output.index, data[0].index)

    output = func(data[0], vars_float[1])
    assert np.isnan(output)

    output = func(data[0], data[1])
    assert np.isnan(output)

    output = func(vars[0], data[0])
    assert np.isnan(output)

    output = func(vars[0], vars[1])
    assert np.isnan(output)

    from gpquant.Function import _ts_rank as func

    output = func(data[0], vars[0])
    assert isinstance(output, pd.Series), type(output)
    np.testing.assert_array_equal(output.index, data[0].index)

    output = func(data[0], vars_float[1])
    assert np.isnan(output)

    output = func(data[0], data[1])
    assert np.isnan(output)

    output = func(vars[0], data[0])
    assert np.isnan(output)

    output = func(vars[0], vars[1])
    assert np.isnan(output)

    from gpquant.Function import _ts_ema as func

    output = func(data[0], vars[0])
    assert isinstance(output, pd.Series), type(output)
    np.testing.assert_array_equal(output.index, data[0].index)

    output = func(data[0], vars_float[1])
    assert np.isnan(output)

    output = func(data[0], data[1])
    assert np.isnan(output)

    output = func(vars[0], data[0])
    assert np.isnan(output)

    output = func(vars[0], vars[1])
    assert np.isnan(output)

    from gpquant.Function import _ts_dema as func

    output = func(data[0], vars[0])
    assert isinstance(output, pd.Series), type(output)
    np.testing.assert_array_equal(output.index, data[0].index)

    output = func(data[0], vars_float[1])
    assert np.isnan(output)

    output = func(data[0], data[1])
    assert np.isnan(output)

    output = func(vars[0], data[0])
    assert np.isnan(output)

    output = func(vars[0], vars[1])
    assert np.isnan(output)

    from gpquant.Function import _ts_kama as func

    output = func(data[0], vars[0], vars[1], vars[2])
    assert isinstance(output, pd.Series), type(output)
    np.testing.assert_array_equal(output.index, data[0].index)

    output = func(vars[0], vars[0], vars[1], vars[2])
    assert np.isnan(output)

    output = func(data[0], vars_float[0], vars[1], vars[2])
    assert np.isnan(output)

    output = func(data[0], data[1], vars[1], vars[2])
    assert np.isnan(output)

    from gpquant.Function import _ts_AROONOSC as func

    output = func(data[0], data[1], vars[0])
    assert isinstance(output, pd.Series), type(output)
    np.testing.assert_array_equal(output.index, data[0].index)

    output = func(data[0], data[1], vars_float[0])
    assert np.isnan(output)

    output = func(data[0], data[1], data[2])
    assert np.isnan(output)

    from gpquant.Function import _ts_WR as func

    output = func(data[0], data[1], data[2], vars[0])
    assert isinstance(output, pd.Series), type(output)
    np.testing.assert_array_equal(output.index, data[0].index)

    output = func(data[0], data[1], data[2], vars_float[0])
    assert np.isnan(output)

    output = func(data[0], data[1], data[2], data[3])
    assert np.isnan(output)

    from gpquant.Function import _ts_CCI as func

    output = func(data[0], data[1], data[2], vars[0])
    assert isinstance(output, pd.Series), type(output)
    np.testing.assert_array_equal(output.index, data[0].index)

    output = func(data[0], data[1], data[2], vars_float[0])
    assert np.isnan(output)

    output = func(data[0], data[1], data[2], data[3])
    assert np.isnan(output)

    from gpquant.Function import _ts_ATR as func

    output = func(data[0], data[1], data[2], vars[0])
    assert isinstance(output, pd.Series), type(output)
    np.testing.assert_array_equal(output.index, data[0].index)

    output = func(data[0], data[1], data[2], vars_float[0])
    assert np.isnan(output)

    output = func(data[0], data[1], data[2], data[3])
    assert np.isnan(output)

    from gpquant.Function import _ts_NATR as func

    output = func(data[0], data[1], data[2], vars[0])
    assert isinstance(output, pd.Series), type(output)
    np.testing.assert_array_equal(output.index, data[0].index)

    output = func(data[0], data[1], data[2], vars_float[0])
    assert np.isnan(output)

    output = func(data[0], data[1], data[2], data[3])
    assert np.isnan(output)

    from gpquant.Function import _ts_ADX as func

    output = func(data[0], data[1], data[2], vars[0])
    assert isinstance(output, pd.Series), type(output)
    np.testing.assert_array_equal(output.index, data[0].index)

    output = func(data[0], data[1], data[2], vars_float[0])
    assert np.isnan(output)

    output = func(data[0], data[1], data[2], data[3])
    assert np.isnan(output)

    from gpquant.Function import _ts_MFI as func

    output = func(data[0], data[1], data[2], data[3], vars[0])
    assert isinstance(output, pd.Series), type(output)
    np.testing.assert_array_equal(output.index, data[0].index)

    output = func(data[0], data[1], data[2], data[3], vars_float[0])
    assert np.isnan(output)

    output = func(data[0], data[1], data[2], data[3], data[3])
    assert np.isnan(output)
