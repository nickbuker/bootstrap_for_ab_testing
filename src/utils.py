# standard library imports
from typing import Dict
# third party imports
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import resample
from statsmodels.stats.weightstats import ttest_ind
from statsmodels.stats.api import CompareMeans, DescrStatsW
# make it pretty
plt.style.use('ggplot')


def make_hist(
    x1: np.ndarray,
    x2: np.ndarray,
) -> None:
    """Plots the distributions of two samples.

    Args:
        x1: Array of data from group 1.
        x2: Array of data from group 2.

    Returns:
        None
    """
    plt.figure(figsize=(10, 6))
    plt.hist(x1, bins=30, alpha=0.5, color='#E24A33', label='x1')
    plt.hist(x2, bins=30, alpha=0.5, color='#348ABD', label='x2')
    plt.legend()
    return


def t_test(
    x1: np.ndarray,
    x2: np.ndarray,
) -> Dict[str, float]:
    """Conducts Welch's t-test on two samples.

    Args:
        x1: Array of data from group 1.
        x2: Array of data from group 2.

    Returns:
        Dictionary containing the t-statistic, p-value, degrees of freedom, difference in means between the groups,
        confidence interval lower bound, and confidence interval upper bounds.
    """
    t_stat, p_value, df = ttest_ind(
        x1=x1,
        x2=x2,
        usevar='unequal',
    )
    diff_means = x2.mean() - x1.mean()
    cm = CompareMeans(DescrStatsW(x2), DescrStatsW(x1))
    ci_lower, ci_upper = cm.tconfint_diff(usevar='unequal')
    results = {
        't_stat': t_stat, 
        'p_value': p_value, 
        'df': df,
        'diff_means': diff_means,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
    }
    return results


def bootstrap_test(
    x1: np.ndarray,
    x2: np.ndarray,
    boots: int = 10000,
) -> np.ndarray:
    """Uses bootstrap resampling to generate a distribution of differences in means.

    Args:
        x1: Array of data from group 1.
        x2: Array of data from group 2.
        boots: Number of boostrap samples to compare during the test. Defaults to 10000.

    Returns:
        Array of bootstrap resampled differences in means.
    """
    mean_diffs = []
    for i in range(boots):
        x1_tmp_mean = resample(x1, replace=True, random_state=i).mean()
        x2_tmp_mean = resample(x2, replace=True, random_state=i).mean()
        mean_diffs.append(x2_tmp_mean - x1_tmp_mean)
    return np.array(mean_diffs)


def plot_bootstrap(mean_diffs: np.ndarray) -> None:
    """Plots distribution of bootstrapped differences in means with 95% confidence intervals.

    Args:
        mean_diffs: Array of bootstrap resampled differences in means from `bootstrap_test()`.

    Returns:
        None
    """
    plt.figure(figsize=(10, 6))
    plt.hist(mean_diffs, bins=30, color='#8EBA42', alpha=0.7)
    diffs_mean = mean_diffs.mean()
    plt.axvline(
        diffs_mean, 
        color='k',
        label=f'mean diff: {diffs_mean}',
    )
    ci_lower = np.percentile(mean_diffs, q=2.5)
    plt.axvline(
        ci_lower, 
        color='k', 
        linestyle='--',
        label=f'lower CI: {ci_lower}',
    )
    ci_upper = np.percentile(mean_diffs, q=97.5)
    plt.axvline(
        ci_upper, 
        color='k', 
        linestyle='--',
        label=f'upper CI: {ci_upper}',
    )
    plt.legend()
    return
