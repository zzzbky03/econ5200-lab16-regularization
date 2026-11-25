"""High-dimensional estimation that will not hand you a causal number by accident (ECON 5200 Lab 16).

The design rule: lasso_support returns a support, never coefficients, because the most common
failure in applied high-dimensional work is reporting a shrunk LASSO coefficient as an effect.
Causal estimates come from double_selection, the only function here that returns an interval.
"""
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LassoCV

RANDOM_STATE = 42          # the seed the lab uses, and the default of every function here


def lasso_support(X, y, seed: int = RANDOM_STATE) -> np.ndarray:
    """The positions of the columns LASSO keeps. Deliberately not the coefficients."""
    fit = LassoCV(cv=5, random_state=seed, max_iter=20_000).fit(X, y)
    return np.flatnonzero(np.abs(fit.coef_) > 1e-8)


def coverage_study(make_fn, estimator, truth: float, trials: int = 200,
                   first_seed: int = 4000) -> dict:
    """How often estimator's 95% interval covers truth over trials samples, and its MC SE."""
    hits, widths = 0, []
    for i in range(trials):
        lo, hi = estimator(*make_fn(first_seed + i))["ci"]
        hits += lo <= truth <= hi
        widths.append(hi - lo)
    coverage = hits / trials
    return {"coverage": coverage, "mc_se": (coverage * (1 - coverage) / trials) ** 0.5,
            "mean_width": float(np.mean(widths)), "trials": trials}


# Paste your double_selection function from Diagnosis 2 below this line, unchanged.
def double_selection(Z, d, y, seed=RANDOM_STATE):
    A = set(lasso_support(Z, y, seed))
    B = set(lasso_support(Z, d, seed))
    union = sorted(A | B)
    fit = sm.OLS(y, sm.add_constant(np.column_stack([d, Z[:, union]]))).fit()
    lo, hi = fit.conf_int()[1]
    return {"estimate": fit.params[1], "se": fit.bse[1], "ci": (lo, hi), "n_union": len(union)}
