from typing import Set, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from numpy.typing import ArrayLike

from dodiscover.ci.utils import _default_regularization, _estimate_propensity_scores, compute_kernel
from dodiscover.typing import Column

from .base import BaseConditionalDiscrepancyTest


class KernelCDTest(BaseConditionalDiscrepancyTest):
    """Kernel conditional discrepancy test among conditional distributions.

    Tests the equality of conditional distributions using a kernel approach
    outlined in :footcite:`Park2021conditional`.

    Parameters
    ----------
    distance_metric : str, optional
        _description_, by default "euclidean"
    metric : str, optional
        _description_, by default "rbf"
    l2 : float | tuple of float, optional
        The l2 regularization to apply for inverting the kernel matrices of 'x' and 'y'
        respectively, by default None. If a single number, then the same l2 regularization
        will be applied to inverting both matrices. If ``None``, then a default
        regularization will be computed that chooses the value that minimizes the upper bound
        on the mean squared prediction error.
    kwidth_x : float, optional
        Kernel width among X variables, by default None, which we will then estimate
        using the median l2 distance between the X variables.
    kwidth_y : float, optional
        Kernel width among Y variables, by default None, which we will then estimate
        using the median l2 distance between the Y variables.
    null_reps : int, optional
        Number of times to sample the null distribution, by default 1000.
    n_jobs : int, optional
        Number of jobs to run computations in parallel, by default None.
    random_state : int, optional
        Random seed, by default None.

    References
    ----------
    .. footbibliography::
    """

    def __init__(
        self,
        distance_metric="euclidean",
        metric="rbf",
        l2=None,
        kwidth_x=None,
        kwidth_y=None,
        null_reps: int = 1000,
        n_jobs=None,
        random_state=None,
    ) -> None:
        self.l2 = l2
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.null_reps = null_reps

        self.kwidth_x = kwidth_x
        self.kwidth_y = kwidth_y
        self.metric = metric
        self.distance_metric = distance_metric

    def test(
        self,
        df: pd.DataFrame,
        x_vars: Set[Column],
        y_vars: Set[Column],
        group_col: Column,
    ) -> Tuple[float, float]:
        """Compute k-sample test statistic and pvalue.

        Tests the null hypothesis::

            H_0: P(Y|X) = P'(Y|X)

        where the different distributions arise from the different datasets
        collected denoted by the ``group_col`` parameter.

        Parameters
        ----------
        df : pd.DataFrame
            The dataset containing the columns denoted by ``x_vars``, ``y_vars``,
            and the ``group_col``.
        x_vars : Set[Column]
            Set of X variables.
        y_vars : Set[Column]
            Set of Y variables.
        group_col : Column
            The column denoting, which group (i.e. environment) each sample belongs to.

        Returns
        -------
        stat : float
            The computed test statistic.
        pvalue : float
            The computed p-value.
        """
        x_cols = list(x_vars)
        y_cols = list(y_vars)

        # check test input
        self._check_test_input(df, x_vars, y_vars, group_col)

        group_ind = df[group_col].to_numpy()
        if set(np.unique(group_ind)) != {0, 1}:
            raise RuntimeError(f"Group indications in {group_col} column should be all 1 or 0.")

        # compute kernel for the X and Y data
        X = df[x_cols].to_numpy()
        Y = df[y_cols].to_numpy()
        K, sigma_x = compute_kernel(
            X,
            distance_metric=self.distance_metric,
            metric=self.metric,
            kwidth=self.kwidth_x,
            n_jobs=self.n_jobs,
        )
        L, sigma_y = compute_kernel(
            Y,
            distance_metric=self.distance_metric,
            metric=self.metric,
            kwidth=self.kwidth_y,
            n_jobs=self.n_jobs,
        )

        # store fitted attributes
        self.kwidth_x_ = sigma_x
        self.kwidth_y_ = sigma_y

        # compute the statistic
        stat = self._statistic(K, L, group_ind)

        # compute propensity scores
        self.propensity_penalty_ = _default_regularization(K)
        e_hat = _estimate_propensity_scores(
            K,
            group_ind,
            penalty=self.propensity_penalty_,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
        )

        # now compute null distribution
        null_dist = self.compute_null(
            e_hat, K, L, null_reps=self.null_reps, random_state=self.random_state
        )
        self.null_dist_ = null_dist

        # compute the pvalue
        pvalue = (1 + np.sum(null_dist >= stat)) / (1 + self.null_reps)
        return stat, pvalue

    def _statistic(self, K: ArrayLike, L: ArrayLike, group_ind: ArrayLike) -> float:
        n_samples = len(K)

        # compute W matrices from K and z
        W0, W1 = self._compute_inverse_kernel(K, group_ind)

        # compute L kernels
        first_mask = np.array(1 - group_ind, dtype=bool)
        second_mask = np.array(group_ind, dtype=bool)
        L0 = L[np.ix_(first_mask, first_mask)]
        L1 = L[np.ix_(second_mask, second_mask)]
        L01 = L[np.ix_(first_mask, second_mask)]

        # compute the final test statistic
        K0 = K[:, first_mask]
        K1 = K[:, second_mask]
        KW0 = K0 @ W0
        KW1 = K1 @ W1

        # compute the three terms in Lemma 4.4
        first_term = np.trace(KW0.T @ KW0 @ L0)
        second_term = np.trace(KW1.T @ KW0 @ L01)
        third_term = np.trace(KW1.T @ KW1 @ L1)

        # compute final statistic
        stat = (first_term - 2 * second_term + third_term) / n_samples
        return stat

    def _compute_inverse_kernel(self, K, z) -> Tuple[ArrayLike, ArrayLike]:
        """Compute W matrices as done in KCD test.

        Parameters
        ----------
        K : ArrayLike of shape (n_samples, n_samples)
            The kernel matrix.
        z : ArrayLike of shape (n_samples)
            The indicator variable of 1's and 0's for which samples belong
            to which group.

        Returns
        -------
        W0 : ArrayLike of shape (n_samples_i, n_samples_i)
            The inverse of the kernel matrix from the first group.
        W1 : NDArraArrayLike of shape (n_samples_j, n_samples_j)
            The inverse of the kernel matrix from the second group.

        Notes
        -----
        Compute the W matrix for the estimated conditional average in
        the KCD test :footcite:`Park2021conditional`.

        References
        ----------
        .. footbibliography::
        """
        # compute kernel matrices
        first_mask = np.array(1 - z, dtype=bool)
        second_mask = np.array(z, dtype=bool)

        # TODO: CHECK THAT THIS IS CORRECT
        K0 = K[np.ix_(first_mask, first_mask)]
        K1 = K[np.ix_(second_mask, second_mask)]

        # compute regularization factors
        self._get_regs(K0, K1)

        # compute the number of samples in each
        n0 = int(np.sum(1 - z))
        n1 = int(np.sum(z))

        W0 = np.linalg.inv(K0 + self.regs_[0] * np.identity(n0))
        W1 = np.linalg.inv(K1 + self.regs_[1] * np.identity(n1))
        return W0, W1

    def _get_regs(self, K0: ArrayLike, K1: ArrayLike):
        """Compute regularization factors."""
        if isinstance(self.l2, (int, float)):
            l0 = self.l2
            l1 = self.l2
            self.regs_ = (l0, l1)
        elif self.l2 is None:
            self.regs_ = (_default_regularization(K0), _default_regularization(K1))
        else:
            if len(self.l2) != 2:
                raise RuntimeError(f"l2 regularization {self.l2} must be a 2-tuple, or a number.")
            self.regs_ = self.l2

    def compute_null(self, e_hat, K, L, null_reps=1000, random_state=None):
        rng = np.random.default_rng(random_state)

        # compute the test statistic on the conditionally permuted
        # dataset, where each group label is resampled for each sample
        # according to its propensity score
        null_dist = Parallel(n_jobs=self.n_jobs)(
            [delayed(self._statistic)(K, L, rng.binomial(1, e_hat)) for _ in range(null_reps)]
        )
        return null_dist
