# Copyright (c) 2013-2021 pgmpy
# Modified from pgmpy.
# License: MIT

import logging
from functools import reduce
from typing import Optional, Set, Tuple

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy import stats

from dodiscover.ci.base import BaseConditionalIndependenceTest
from dodiscover.typing import Column


def power_divergence(
    X: ArrayLike, Y: ArrayLike, Z: ArrayLike, data: pd.DataFrame, lambda_: str = "cressie-read"
) -> Tuple[float, float, int]:
    """
    Computes the Cressie-Read power divergence statistic [1]. The null hypothesis
    for the test is X is independent of Y given Z. A lot of the frequency comparison
    based statistics (eg. chi-square, G-test etc) belong to power divergence family,
    and are special cases of this test.

    Parameters
    ----------
    X: int, string, hashable object
        A variable name contained in the data set

    Y: int, string, hashable object
        A variable name contained in the data set, different from X

    Z: list, array-like
        A list of variable names contained in the data set, different from X and Y.
        This is the separating set that (potentially) makes X and Y independent.
        Default: []

    data: pandas.DataFrame
        The dataset on which to test the independence condition.

    lambda_: float or string
        The lambda parameter for the power_divergence statistic. Some values of
        lambda_ results in other well known tests:
            "pearson"             1          "Chi-squared test"
            "log-likelihood"      0          "G-test or log-likelihood"
            "freeman-tukey"     -1/2        "freeman-tukey Statistic"
            "mod-log-likelihood"  -1         "Modified Log-likelihood"
            "neyman"              -2         "Neyman's statistic"
            "cressie-read"        2/3        "The value recommended in the paper
                                             :footcite:`cressieread1984`"

    Returns
    -------
    CI Test Results: tuple
        Returns a tuple (chi, p_value, dof). `chi` is the
        chi-squared test statistic. The `p_value` for the test, i.e. the
        probability of observing the computed chi-square statistic (or an even
        higher value), given the null hypothesis that X \u27c2 Y | Zs is True.
        If boolean = True, returns True if the p_value of the test is greater
        than `significance_level` else returns False.

    See Also
    --------
    scipy.stats.power_divergence

    References
    ----------
    .. footbibliography::

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> data = pd.DataFrame(np.random.randint(0, 2, size=(50000, 4)), columns=list('ABCD'))
    >>> data['E'] = data['A'] + data['B'] + data['C']
    >>> chi_square(X='A', Y='C', Z=[], data=data, boolean=True, significance_level=0.05)
    True
    >>> chi_square(X='A', Y='B', Z=['D'], data=data, boolean=True, significance_level=0.05)
    True
    >>> chi_square(X='A', Y='B', Z=['D', 'E'], data=data, boolean=True, significance_level=0.05)
    False
    """

    # Step 1: Check if the arguments are valid and type conversions.
    if isinstance(Z, str):
        Z = [Z]
    if (X in Z) or (Y in Z):
        raise ValueError(f"The variables X or Y can't be in Z. Found {X if X in Z else Y} in Z.")

    # Step 2: Do a simple contingency test if there are no conditional variables.
    if len(Z) == 0:
        chi, p_value, dof, expected = stats.chi2_contingency(
            data.groupby([X, Y]).size().unstack(Y, fill_value=0), lambda_=lambda_
        )

    # Step 3: If there are conditionals variables, iterate over unique states and do
    #         the contingency test.
    else:
        chi = 0
        dof = 0
        for z_state, df in data.groupby(Z):
            try:
                # Note: The fill value is set to 1e-7 to avoid the following error:
                # where there are not enough samples in the data, which results in a nan pvalue
                sub_table_z = df.groupby([X, Y]).size().unstack(Y, fill_value=1e-7)
                c, _, d, _ = stats.chi2_contingency(sub_table_z, lambda_=lambda_)
                chi += c
                dof += d
            except ValueError:
                # If one of the values is 0 in the 2x2 table.
                if isinstance(z_state, str):
                    logging.info(
                        f"Skipping the test {X} \u27c2 {Y} | {Z[0]}={z_state}. Not enough samples"
                    )
                else:
                    z_str = ", ".join([f"{var}={state}" for var, state in zip(Z, z_state)])
                    logging.info(f"Skipping the test {X} \u27c2 {Y} | {z_str}. Not enough samples")

            if np.isnan(c):
                raise RuntimeError(
                    f"The resulting chi square test statistic is NaN, which occurs "
                    f"when there are not enough samples in your data "
                    f"{df.shape}, {sub_table_z}."
                )

        p_value = 1 - stats.chi2.cdf(chi, df=dof)

    # Step 4: Return the values
    return chi, p_value, dof


class CategoricalCITest(BaseConditionalIndependenceTest):
    def __init__(self, lambda_="cressie-read") -> None:
        """CI test for categorical data.

        Uses the power-divergence class of test statistics to test categorical data
        for (conditional) independences.

        Parameters
        ----------
        lambda_ : str, optional
            The lambda parameter for the power_divergence statistic, by default 'cressie-read'.
            Some values of lambda_ results in other well known tests:
                "pearson"             1          "Chi-squared test"
                "log-likelihood"      0          "G-test or log-likelihood"
                "freeman-tukey"     -1/2        "freeman-tukey Statistic"
                "mod-log-likelihood"  -1         "Modified Log-likelihood"
                "neyman"              -2         "Neyman's statistic"
                "cressie-read"        2/3        "The value recommended in the paper
                                                 :footcite:`cressieread1984`"

        See Also
        --------
        scipy.stats.power_divergence

        References
        ----------
        .. footbibliography::
        """
        self.lambda_ = lambda_

    def test(
        self,
        df: pd.DataFrame,
        x_vars: Set[Column],
        y_vars: Set[Column],
        z_covariates: Optional[Set[Column]] = None,
    ) -> Tuple[float, float]:
        x_vars = reduce(lambda x: x, x_vars)  # type: ignore
        y_vars = reduce(lambda x: x, y_vars)  # type: ignore
        stat, pvalue, dof = power_divergence(
            x_vars, y_vars, z_covariates, data=df, lambda_=self.lambda_
        )
        self.dof_ = dof
        return stat, pvalue


class CausalLearnCITest(BaseConditionalIndependenceTest):
    def __init__(self, method_name="gsq") -> None:
        self.method_name = method_name

    def test(
        self,
        df: pd.DataFrame,
        x_vars: Set[Column],
        y_vars: Set[Column],
        z_covariates: Optional[Set[Column]] = None,
    ) -> Tuple[float, float]:
        import numpy as np
        from causallearn.utils.cit import Chisq_or_Gsq

        data = df.to_numpy()
        x_vars = reduce(lambda x: x, x_vars)  # type: ignore
        y_vars = reduce(lambda x: x, y_vars)  # type: ignore
        x = np.argwhere(df.columns == x_vars)
        y = np.argwhere(df.columns == y_vars)
        z = []
        if z_covariates is not None:
            for _z in z_covariates:
                z.append(np.argwhere(df.columns == _z).squeeze())
        z = np.array(z)

        tester = Chisq_or_Gsq(data, method_name=self.method_name)
        return np.nan, tester(x, y, z)
