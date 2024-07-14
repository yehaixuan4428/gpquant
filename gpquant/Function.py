import warnings

import numpy as np
import pandas as pd
import talib as ta
from .cache_decorator import cache_decorator

warnings.filterwarnings("ignore")  # prevent reporting 'All-NaN slice encountered'


class Function:
    def __init__(
        self, function, name: str, arity: int, is_ts: int = 0, fixed_params: list = None
    ) -> None:
        self.function = function  # function
        self.name = name  # function name
        self.arity = arity  # number of function arguments
        # number of parameters forced to be constants
        self.is_ts = is_ts  # 0: basis function, >0: time-series function
        # arguments forced to be certain variables
        self.fixed_params = [] if fixed_params is None else fixed_params

    def __call__(self, *args, cache_dir: str = './cache'):
        try:
            return self.function(*args, cache_dir=cache_dir)
        except TypeError:
            return self.function(*args)


def __rolling(x1: pd.Series, d: int, function=None, **kwargs) -> np.ndarray:
    """auxiliary function, rolling more effectively than apply"""
    try:
        incomplete_w = np.lib.stride_tricks.sliding_window_view(x1.values, d)[..., ::-1]
        window = pd.DataFrame(
            np.vstack((np.full([d - 1, d], np.nan), incomplete_w)), index=x1.index
        )
        try:
            result = function(
                window, **kwargs
            )  # add other arguments needed in function
            # for i in range(d - 1):
            #     result[i] = np.nan
            result[: d - 1] = np.nan
            return pd.Series(result, index=x1.index)
        except:
            return window
    except:
        return np.nan


def __scalar_ema(window: pd.DataFrame, alpha: float) -> np.ndarray:
    """auxiliary function, calculating the ema of the last value in time-series"""
    try:
        if isinstance(alpha == alpha, bool):  # alpha is a scalar
            alpha_vec = np.broadcast_to(alpha, (window.shape[0], 1))
        else:  # alpha is a vector
            alpha_vec = np.broadcast_to(alpha, (1, len(alpha))).T
        window *= alpha_vec.dot(
            np.broadcast_to(np.arange(window.shape[1]), (1, window.shape[1]))
        )
        return np.nansum(window, axis=1) * alpha
    except:
        return np.nan


def _square(x1):
    return x1**2


def _sqrt(x1):
    """sign-protected"""
    return np.sqrt(np.abs(x1)) * np.sign(x1)


def _cube(x1):
    return x1**3


def _cbrt(x1):
    return np.cbrt(x1)


def _sign(x1):
    return np.sign(x1)


def _neg(x1):
    return -x1


def _inv(x1):
    """closure of inverse for zero arguments"""
    with np.errstate(divide="ignore", invalid="ignore"):
        try:
            return (1.0 / x1).mask(x1.abs() <= 0.001, 0.0)
        except:
            if isinstance(x1, np.ndarray):
                x1 = x1[0]
            if np.abs(x1) > 0.001:
                return 1.0 / x1
            else:
                return 0.0


def _abs(x1):
    return np.abs(x1)


def _sin(x1):
    return np.sin(x1)


def _cos(x1):
    return np.cos(x1)


def _tan(x1):
    return np.tan(x1)


def _log(x1):
    """closure of log for zero arguments, sign-protected"""
    with np.errstate(divide="ignore", invalid="ignore"):
        try:
            x1 = x1.mask(x1.abs() <= 0.001, 1.0)
            return np.log(x1.abs()).mask(x1 < -1, np.log(x1.abs()) * np.sign(x1))
        except:
            try:
                x1 = x1[0]
            except:
                pass
            if np.abs(x1) <= 0.001:
                return 0.0
            else:
                if x1 < -1:
                    return np.log(np.abs(x1)) * np.sign(x1)
                else:
                    return np.log(np.abs(x1))


def _sig(x1):
    """logistic function"""
    return 1 / (1 + np.exp(-x1))


def _add(x1, x2):
    return x1 + x2


def _sub(x1, x2):
    return x1 - x2


def _mul(x1, x2):
    return x1 * x2


def _div(x1, x2):
    """closure of division (x1/x2) for zero denominator"""
    with np.errstate(divide="ignore", invalid="ignore"):
        values = np.where(np.abs(x2) > 0.001, np.divide(x1, x2), 1.0)
        if isinstance(x1, pd.Series):
            return pd.Series(values, index=x1.index)
        if isinstance(x2, pd.Series):
            return pd.Series(values, index=x2.index)

        return values.flatten()[0]


def _max(x1, x2):
    return np.maximum(x1, x2)


def _min(x1, x2):
    return np.minimum(x1, x2)


def _mean(x1, x2):
    return (x1 + x2) / 2


def _clear_by_cond(x1, x2, x3):
    """if x1 < x2 (keep NaN if and only if both x1 and x2 are NaN), then 0, else x3"""
    try:
        values = np.where(x1 < x2, 0, np.where(~np.isnan(x1) | ~np.isnan(x2), x3, np.nan))
        if isinstance(x1, pd.Series):
            return pd.Series(values, index=x1.index)
        if isinstance(x2, pd.Series):
            return pd.Series(values, index=x2.index)
        if isinstance(x3, pd.Series):
            return pd.Series(values, index=x3.index)
        return np.nan
    except:
        return np.nan


def _if_then_else(x1, x2, x3):
    try:
        """if x1 is nonzero (keep NaN), then x2, else x3"""
        values = np.where(x1, x2, np.where(~np.isnan(x1), x3, np.nan))
        if isinstance(x1, pd.Series):
            return pd.Series(values, index=x1.index)
        if isinstance(x2, pd.Series):
            return pd.Series(values, index=x2.index)
        if isinstance(x3, pd.Series):
            return pd.Series(values, index=x3.index)

        '''disable invalid operation if all variables are float'''
        return np.nan
    except:
        return np.nan


def _if_cond_then_else(x1, x2, x3, x4):
    try:
        """if x1 < x2 (keep NaN if and only if both x1 and x2 are NaN), then x3, else x4"""
        values = np.where(x1 < x2, x3, np.where(~np.isnan(x1) | ~np.isnan(x2), x4, np.nan))
        if isinstance(x1, pd.Series):
            return pd.Series(values, index=x1.index)
        if isinstance(x2, pd.Series):
            return pd.Series(values, index=x2.index)
        if isinstance(x3, pd.Series):
            return pd.Series(values, index=x3.index)
        if isinstance(x4, pd.Series):
            return pd.Series(values, index=x4.index)

        '''disable invalid operation if all variables are float'''
        return np.nan
    except:
        return np.nan


def _cs_count(x):
    if isinstance(x, pd.Series):
        return x.groupby(level=0).transform('size')
    return np.nan


@cache_decorator()
def _ts_delay(x1, d: int):
    """x1 d datetimes ago"""
    # return pd.Series(x1).shift(d).values
    try:
        return x1.groupby(level=1, group_keys=False).shift(d)
    except:
        return np.nan


@cache_decorator()
def _ts_delta(x1, d: int):
    """difference between x1 and x1 d datetimes ago"""
    # return x1 - pd.Series(x1).shift(d).values
    try:
        return x1.groupby(level=1, group_keys=False).diff(d)
    except:
        return np.nan


@cache_decorator()
def _ts_pct_change(x1, d: int):
    """percentage change of x1 in the last d datetimes"""
    # return _div(_ts_delta(x1, d), x1) * np.sign(x1)
    try:
        return x1.groupby(level=1, group_keys=False).pct_change(d, fill_method=None)
    except:
        return np.nan


@cache_decorator()
def _ts_mean_return(x1, d: int):
    """moving average of percentage change of x1 with one lag"""
    return _ts_mean(_ts_pct_change(x1, 1), d)


@cache_decorator()
def _ts_max(x1, d: int):
    """maximum x1 in the last d datetimes"""
    try:
        return x1.groupby(level=1, group_keys=False).rolling(d, int(d/2)).max().droplevel(0).swaplevel().sort_index()
    except:
        return np.nan


@cache_decorator()
def _ts_min(x1, d: int):
    """minimum x1 in the last d datetimes"""
    try:
        return x1.groupby(level=1, group_keys=False).rolling(d, int(d/2)).min().droplevel(0).swaplevel().sort_index()
    except:
        return np.nan


@cache_decorator()
def _ts_sum(x1, d: int):
    """moving sum"""
    try:
        return x1.groupby(level=1, group_keys=False).rolling(d, int(d/2)).sum().droplevel(0).swaplevel().sort_index()
    except:
        return np.nan


@cache_decorator()
def _ts_product(x1, d: int):
    """moving product"""
    try:
        x1 = _log(x1)
        return np.exp(x1.groupby(level=1, group_keys=False).rolling(d, int(d/2)).sum().droplevel(0).swaplevel().sort_index())
    except:
        return np.nan


@cache_decorator()
def _ts_mean(x1, d: int):
    """moving average"""
    try:
        return x1.groupby(level=1, group_keys=False).rolling(d, int(d/2)).mean().droplevel(0).swaplevel().sort_index()
    except:
        return np.nan


@cache_decorator()
def _ts_std(x1, d: int):
    """moving standard deviation"""
    try:
        return x1.groupby(level=1, group_keys=False).rolling(d, int(d/2)).std().droplevel(0).swaplevel().sort_index()
    except:
        return np.nan


@cache_decorator()
def _ts_median(x1, d: int):
    """moving median"""
    try:
        return x1.groupby(level=1, group_keys=False).rolling(d, int(d/2)).median().droplevel(0).swaplevel().sort_index()
    except:
        return np.nan


@cache_decorator()
def _ts_midpoint(x1, d: int):
    """moving midpoint: (ts_max + ts_min) / 2"""
    return (_ts_max(x1, d) + _ts_min(x1, d)) / 2.0


@cache_decorator()
def _ts_skew(x1, d: int):
    """moving skewness"""
    try:
        return x1.groupby(level=1, group_keys=False).rolling(d, int(d/2)).skew().droplevel(0).swaplevel().sort_index()
    except:
        return np.nan


@cache_decorator()
def _ts_kurt(x1, d: int):
    """moving kurtosis"""
    try:
        return x1.groupby(level=1, group_keys=False).rolling(d, int(d/2)).kurt().droplevel(0).swaplevel().sort_index()
    except:
        return np.nan


@cache_decorator()
def _ts_inverse_cv(x1, d: int):
    """moving inverse of coefficient of variance"""
    return _div(_ts_mean(x1, d), _ts_std(x1, d))


@cache_decorator()
def _ts_cov(x1, x2, d: int):
    """moving covariance of x1 and x2"""
    if ~isinstance(x1, pd.Series) or ~isinstance(x2, pd.Series):
        return np.nan

    def func(x, d):
        return x["x1"].rolling(d, min_periods=int(d / 2)).cov(x["x2"])

    x1 = x1.to_frame("x1")
    x1["x2"] = x2
    return (
        x1.groupby(level=1, group_keys=False).apply(lambda x: func(x, d)).sort_index()
    )


@cache_decorator()
def _ts_corr(x1, x2, d: int):
    """moving correlation coefficient of x1 and x2"""
    if ~isinstance(x1, pd.Series) or ~isinstance(x2, pd.Series):
        return np.nan

    # return pd.Series(x1).rolling(d, min_periods=int(d / 2)).corr(pd.Series(x2)).values
    def func(x, d):
        return x["x1"].rolling(d, min_periods=int(d / 2)).corr(x["x2"])

    x1 = pd.Series(x1).to_frame("x1")
    x1["x2"] = pd.Series(x2)
    return (
        x1.groupby(level=1, group_keys=False).apply(lambda x: func(x, d)).sort_index()
    )


@cache_decorator()
def _ts_autocorr(x1, d: int, i: int):
    """moving autocorrelation coefficient between x and x lag i period"""
    x2 = _ts_delay(x1, i)
    return _ts_corr(x1, x2, d)


@cache_decorator()
def _ts_maxmin(x1, d: int):
    """moving maxmin normalization"""
    ts_max, ts_min = _ts_max(x1, d), _ts_min(x1, d)
    return _div(x1 - ts_min, ts_max - ts_min)


@cache_decorator()
def _ts_zscore(x1, d: int):
    """moving zscore standardization"""
    return _div(x1 - _ts_mean(x1, d), _ts_std(x1, d))


@cache_decorator()
def _ts_regression_beta(x1, x2, d: int):
    """slope of regression x1 onto x2 in the last d datetimes"""
    return _div(_ts_cov(x1, x2, d), _ts_std(x2, d) ** 2)


@cache_decorator()
def _ts_regression_alpha(x1, x2, d: int):
    """slope of regression x1 onto x2 in the last d datetimes"""
    return _ts_mean(x1, d) - _ts_mean(x2, d) * _ts_regression_beta(x1, x2, d)


@cache_decorator()
def _ts_linear_slope(x1, d: int):
    """slope of regression x1 in the last d datetimes onto (1, 2, ..., d)"""
    try:
        x2 = pd.Series(np.arange(len(x1)) + 1, index=x1.index)
        return _div(_ts_cov(x1, x2, d), _ts_std(x2, d) ** 2)
    except:
        return np.nan


@cache_decorator()
def _ts_linear_intercept(x1, d: int):
    """intercept of regression x1 in the last d datetimes onto (1, 2, ..., d)"""
    return _ts_mean(x1, d) - (1 + d) / 2 * _ts_linear_slope(x1, d)


@cache_decorator()
def _ts_argmax(x1, d: int):
    """position of maximum x1 in the last d datetimes"""
    # return pd.Series(x1).rolling(d).apply(np.argmax, engine="numba", raw=True).values
    try:
        return (
            x1.groupby(level=1)
            .rolling(d, min_periods=int(d / 2))
            .apply(np.argmax, raw=True)
            .droplevel(0)
            .sort_index()
        )
    except:
        return np.nan


@cache_decorator()
def _ts_argmin(x1, d: int):
    """position of minimum x1 in the last d datetimes"""
    # return pd.Series(x1).rolling(d).apply(np.argmin, engine="numba", raw=True).values
    try:
        return (
            x1.groupby(level=1)
            .rolling(d, min_periods=int(d / 2))
            .apply(np.argmin, raw=True)
            .droplevel(0)
            .sort_index()
        )
    except:
        return np.nan


@cache_decorator()
def _ts_argmaxmin(x1, d: int):
    """relative position of maximum x1 to minimum x1 in the last d datetimes"""
    return _ts_argmax(x1, d) - _ts_argmin(x1, d)


@cache_decorator()
def _ts_rank(x1, d: int):
    """moving quantile of current x1"""
    try:
        return (
            x1.groupby(level=1)
            .rolling(d, min_periods=int(d / 2))
            .rank()
            .droplevel(0)
            .sort_index()
        )
    except:
        return np.nan


@cache_decorator()
def _ts_ema(x1, d: int):
    """exponential moving average (EMA)"""
    try:
        alpha = 2 / (d + 1)
        # return __rolling(pd.Series(x1), d, function=__scalar_ema, alpha=alpha)
        return (
            x1.groupby(level=1)
            .apply(
                lambda x: __rolling(
                    x.droplevel(1), d, function=__scalar_ema, alpha=alpha
                )
            )
            .swaplevel()
            .sort_index()
        )
    except:
        return np.nan


@cache_decorator()
def _ts_dema(x1, d: int):
    """double exponential moving average (DEMA): 2 * EMA(x1) - EMA(EMA(x1))"""
    ema = _ts_ema(x1, d)
    return 2 * ema - _ts_ema(ema, d)


@cache_decorator()
def _ts_kama(x1, d1: int, d2: int, d3: int):
    """Kaufman's adaptive moving average (KAMA):
    1) KAMA is an exponential moving average with an adaptive alpha SC
    KAMA_{t} = SC_{f,s} * x1 + (1 - SC{f,s}) * KAMA_{t-1}
    2) SC (smoothing constant) is a weighted average of f and s with weight ER adaptive
    SC_{f,s} = [ER_{t,d} * f + (1 - ER_{t,d}) * s]^2
    3) ER (efficiency ratio) is the price change adjusted for the daily volatility
    ER_{t,d} = Change_{t,d} / Volatility_{t,d}
    Change_{t,d} = abs(x1 - x1 d datetimes ago)
    Volatility_{t,d} = sum(abs(x1 - x1 1 datetimes ago)
    4) d is lag period, f is fastest smoothing constant, s is slowest smoothing constant
    d = d1, f = 1 / (1 + d2), s = 1 / (1 + d3)"""
    try:

        def func(x1, d1, d2, d3):
            d, f, s = (
                d1,
                1 / (1 + min(d2, d3)),
                1 / (1 + max(d2, d3)),
            )  # f should greater than s
            change = np.abs(x1 - x1.shift(d))
            volatility = (x1 - x1.shift()).abs().rolling(d, int(d / 2)).sum()
            ER = _div(change, volatility)
            SC = (ER * f + (1 - ER) * s) ** 2
            return __rolling(x1, d, function=__scalar_ema, alpha=SC)

        return (
            x1.groupby(level=1)
            .apply(lambda x: func(x.droplevel(1), d1, d2, d3))
            .swaplevel()
            .sort_index()
        )
    except:
        return np.nan


def _ts_AROONOSC(high, low, d: int):
    """Aroon Oscillator: Aroon-up - Aroon-down
    Aroon-Up = (d - HH) * d (HH: number of datetimes ago the highest price occurred)
    Aroon-Down = (d - LL) * d (LL: number of datetimes ago the lowest price occurred)"""
    return (_ts_argmax(high, d) - _ts_argmin(low, d)) / d


def _ts_WR(high, low, close, d: int):
    """Williams %R: (H_{d} - C) / (H_{d} - L_{d})"""
    return (_ts_max(high, d) - close) / (_ts_max(high, d) - _ts_min(low, d))


@cache_decorator()
def _ts_CCI(high, low, close, d: int):
    """Commodity Channel Index: (TP - MA) / (0.015 * MD)
    TP (Typical Price) = (High + Low + Close) / 3
    MA (Moving Average) = sum(TP) / d
    MD (Mean Deviation) = sum(abs(TP - MA)) / d"""
    TP = (high + low + close) / 3
    MA = _ts_mean(TP, d)
    MD = _ts_mean((TP - MA).abs(), d)
    return _div(TP - MA, 0.015 * MD)


@cache_decorator()
def _ts_ATR(high, low, close, d: int):
    """Average True Range: ts_mean(TR, d)
    TR (True Range) = max(High - Low, abs(High - previous Close), abs(Low - previous Close))
    """
    close_shift = close.groupby(level=1).shift()
    TR = np.maximum(high - low, high - close_shift, low - close_shift)
    return _ts_mean(TR, d)


def _ts_NATR(high, low, close, d: int):
    """Normalized Average True Range: ATR / Close * 100"""
    return _ts_ATR(high, low, close, d) / close * 100


@cache_decorator()
def _ts_ADX(high, low, close, d: int):
    """Average Directional Index: ts_mean(DX, d)
    DX = abs(+DI - -DI) / abs(+DI + -DI) * 100
    +DI (Directional Index) = ts_mean(+DM, d) / ATR * 100
    -DI (Directional Index) = ts_mean(-DM, d) / ATR * 100
    +DM (Directional Movement) = High - previous High
    -DM (Directional Movement) = Low - previous Low"""
    ATR = _ts_ATR(high, low, close, d)
    pDI = _div(_ts_mean(high - high.groupby(level=1).shift(), d), ATR)
    nDI = _div(_ts_mean(low - low.groupby(level=1).shift(), d), ATR)
    DX = _div((pDI - nDI).abs(), (pDI + nDI).abs()) * 100.0

    return _ts_mean(DX, d)


@cache_decorator()
def _ts_MFI(high, low, close, volume, d: int):
    """Money Flow Index: 100 - (100 / (1 + MFR))
    MFR (Money Flow Ratio) = sum(PMF) / sum(NMF)
    PMF (Positive Money Flow) = RMF where TP > previous TP
    NMF (Negative Money Flow) = RMF where TP < previous TP
    RMF (Raw Money Flow) = TP * Volume
    TP (Typical Price) = (High + Low + Close) / 3"""
    TP = (high + low + close) / 3
    pn = TP - TP.groupby(level=1).shift()
    RMF = TP * volume
    PMF = _clear_by_cond(pn, 0, RMF)
    NMF = _clear_by_cond(0, pn, RMF)
    MFR = _div(_ts_sum(PMF, d), _ts_sum(NMF, d))
    return 100.0 - _div(100.0, 1.0 + MFR)


# 1. basic functions (scalar arguments, vectorized computation)
# 1.1. single variable
square1 = Function(function=_square, name="square", arity=1)
sqrt1 = Function(function=_sqrt, name="sqrt", arity=1)
cube1 = Function(function=_cube, name="cube", arity=1)
cbrt1 = Function(function=_cbrt, name="cbrt", arity=1)
sign1 = Function(function=_sign, name="sign", arity=1)
neg1 = Function(function=_neg, name="neg", arity=1)
inv1 = Function(function=_inv, name="inv", arity=1)
abs1 = Function(function=_abs, name="abs", arity=1)
sin1 = Function(function=_sin, name="sin", arity=1)
cos1 = Function(function=_cos, name="cos", arity=1)
tan1 = Function(function=_tan, name="tan", arity=1)
log1 = Function(function=_log, name="log", arity=1)
sig1 = Function(function=_sig, name="sig", arity=1)
# 1.2. double variables
add2 = Function(function=_add, name="add", arity=2)
sub2 = Function(function=_sub, name="sub", arity=2)
mul2 = Function(function=_mul, name="mul", arity=2)
div2 = Function(function=_div, name="div", arity=2)
max2 = Function(function=_max, name="max", arity=2)
min2 = Function(function=_min, name="min", arity=2)
mean2 = Function(function=_mean, name="mean", arity=2)

# 2. conditional functions (scalar arguments, vectorized computation)
# 2.1. three variables
clear_by_cond3 = Function(function=_clear_by_cond, name="clear_by_cond", arity=3)
if_then_else3 = Function(function=_if_then_else, name="if_then_else", arity=3)
# 2.2. four variables
if_cond_then_else4 = Function(
    function=_if_cond_then_else, name="if_cond_then_else", arity=4
)

cs_count1 = Function(function=_cs_count, name='cs_count', arity=1)

# 3. time-series functions (time series arguments with time window, vectorized computation)
# 3.1. difference
ts_delay2 = Function(function=_ts_delay, name="ts_delay", arity=2, is_ts=1)
ts_delta2 = Function(function=_ts_delta, name="ts_delta", arity=2, is_ts=1)
ts_pct_change2 = Function(
    function=_ts_pct_change, name="ts_pct_change", arity=2, is_ts=1
)
ts_mean_return2 = Function(
    function=_ts_mean_return, name="ts_mean_return", arity=2, is_ts=1
)
# 3.2. statistics
ts_max2 = Function(function=_ts_max, name="ts_max", arity=2, is_ts=1)
ts_min2 = Function(function=_ts_min, name="ts_min", arity=2, is_ts=1)
ts_sum2 = Function(function=_ts_sum, name="ts_sum", arity=2, is_ts=1)
ts_product2 = Function(function=_ts_product, name="ts_product", arity=2, is_ts=1)
ts_mean2 = Function(function=_ts_mean, name="ts_mean", arity=2, is_ts=1)
ts_std2 = Function(function=_ts_std, name="ts_std", arity=2, is_ts=1)
ts_median2 = Function(function=_ts_median, name="ts_median", arity=2, is_ts=1)
ts_midpoint2 = Function(function=_ts_midpoint, name="ts_midpoint", arity=2, is_ts=1)
ts_skew2 = Function(function=_ts_skew, name="ts_skew", arity=2, is_ts=1)
ts_kurt2 = Function(function=_ts_kurt, name="ts_kurt", arity=2, is_ts=1)
ts_inverse_cv2 = Function(
    function=_ts_inverse_cv, name="ts_inverse_cv", arity=2, is_ts=1
)
ts_cov3 = Function(function=_ts_cov, name="ts_cov", arity=3, is_ts=1)
ts_corr3 = Function(function=_ts_corr, name="ts_corr", arity=3, is_ts=1)
ts_autocorr3 = Function(function=_ts_autocorr, name="ts_autocorr", arity=3, is_ts=2)
ts_maxmin2 = Function(function=_ts_maxmin, name="ts_maxmin", arity=2, is_ts=1)
ts_zscore2 = Function(function=_ts_zscore, name="ts_zscore", arity=2, is_ts=1)
# 3.3. regression
ts_regression_beta3 = Function(
    function=_ts_regression_beta, name="ts_regression_beta", arity=3, is_ts=1
)

ts_regression_alpha3 = Function(
    function=_ts_regression_alpha, name="ts_regression_alpha", arity=3, is_ts=1
)

ts_linear_slope2 = Function(
    function=_ts_linear_slope, name="ts_linear_slope", arity=2, is_ts=1
)
ts_linear_intercept2 = Function(
    function=_ts_linear_intercept, name="ts_linear_intercept", arity=2, is_ts=1
)
# 3.4. relevant position
ts_argmax2 = Function(function=_ts_argmax, name="ts_argmax", arity=2, is_ts=1)
ts_argmin2 = Function(function=_ts_argmin, name="ts_argmin", arity=2, is_ts=1)
ts_argmaxmin2 = Function(function=_ts_argmaxmin, name="ts_argmaxmin", arity=2, is_ts=1)
ts_rank2 = Function(function=_ts_rank, name="ts_rank", arity=2, is_ts=1)
# 3.5. technical indicator
ts_ema2 = Function(function=_ts_ema, name="ts_ema", arity=2, is_ts=1)
ts_dema2 = Function(function=_ts_dema, name="ts_dema", arity=2, is_ts=1)
ts_kama4 = Function(function=_ts_kama, name="ts_kama", arity=4, is_ts=3)
ts_AROONOSC3 = Function(
    function=_ts_AROONOSC,
    name="ts_AROONOSC",
    arity=3,
    is_ts=1,
    fixed_params=["high", "low"],
)
ts_WR4 = Function(
    function=_ts_WR,
    name="ts_WR",
    arity=4,
    is_ts=1,
    fixed_params=["high", "low", "close"],
)
ts_CCI4 = Function(
    function=_ts_CCI,
    name="ts_CCI",
    arity=4,
    is_ts=1,
    fixed_params=["high", "low", "close"],
)
ts_ATR4 = Function(
    function=_ts_ATR,
    name="ts_ATR",
    arity=4,
    is_ts=1,
    fixed_params=["high", "low", "close"],
)
ts_NATR4 = Function(
    function=_ts_NATR,
    name="ts_NATR",
    arity=4,
    is_ts=1,
    fixed_params=["high", "low", "close"],
)
ts_ADX4 = Function(
    function=_ts_ADX,
    name="ts_ADX",
    arity=4,
    is_ts=1,
    fixed_params=["high", "low", "close"],
)
ts_MFI5 = Function(
    function=_ts_MFI,
    name="ts_MFI",
    arity=5,
    is_ts=1,
    fixed_params=["high", "low", "close", "volume"],
)


function_map = {
    "square": square1,
    "sqrt": sqrt1,
    "cube": cube1,
    "cbrt": cbrt1,
    "sign": sign1,
    "neg": neg1,
    "inv": inv1,
    "abs": abs1,
    "sin": sin1,
    "cos": cos1,
    "tan": tan1,
    "log": log1,
    "sig": sig1,
    "add": add2,
    "sub": sub2,
    "mul": mul2,
    "div": div2,
    "max": max2,
    "min": min2,
    "mean": mean2,
    "clear by cond": clear_by_cond3,
    "if then else": if_then_else3,
    "if cond then else": if_cond_then_else4,
    "cs count": cs_count1,
    "ts delay": ts_delay2,
    "ts delta": ts_delta2,
    "ts pct change": ts_pct_change2,
    "ts mean return": ts_mean_return2,
    "ts max": ts_max2,
    "ts min": ts_min2,
    "ts sum": ts_sum2,
    "ts product": ts_product2,
    "ts mean": ts_mean2,
    "ts std": ts_std2,
    "ts median": ts_median2,
    "ts midpoint": ts_midpoint2,
    "ts skew": ts_skew2,
    "ts kurt": ts_kurt2,
    "ts inverse cv": ts_inverse_cv2,
    "ts cov": ts_cov3,
    "ts corr": ts_corr3,
    "ts autocorr": ts_autocorr3,
    "ts maxmin": ts_maxmin2,
    "ts zscore": ts_zscore2,
    "ts regression beta": ts_regression_beta3,
    "ts regression alpha": ts_regression_alpha3,
    "ts linear slope": ts_linear_slope2,
    "ts linear intercept": ts_linear_intercept2,
    "ts argmax": ts_argmax2,
    "ts argmin": ts_argmin2,
    "ts argmaxmin": ts_argmaxmin2,
    "ts rank": ts_rank2,
    "ts ema": ts_ema2,
    "ts dema": ts_dema2,
    "ts kama": ts_kama4,
    "ts AROONOSC": ts_AROONOSC3,
    "ts WR": ts_WR4,
    "ts CCI": ts_CCI4,
    "ts ATR": ts_ATR4,
    "ts NATR": ts_NATR4,
    "ts ADX": ts_ADX4,
    "ts MFI": ts_MFI5,
}
