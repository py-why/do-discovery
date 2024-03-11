# Adapted from: https://github.com/juangamella/ges/
# BSD 3-Clause License

from typing import Callable, Union, Dict

import numpy as np
import pandas as pd


class ScoreFunction:
    def __init__(self, score: Union[Callable, str]='bic') -> None:
        self._cache: Dict = dict()

        if score == 'bic':
            self.score_func = bic_score

    def local_score(self, data: pd.DataFrame, source, source_parents) -> float:
        """Compute the local score of an edge.

        Parameters
        ----------
        data : pd.DataFrame
            The dataset.
        source : Node
            The origin node.
        source_parents : list of Node
            The parents of the source.

        Returns
        -------
        float
            The score.
        """
        # key is a tuple of the form (source, sorted(source_parents))
        key = (source, tuple(sorted(source_parents)))

        try:
            score = self._cache[key]
        except KeyError:
            score = self.score_func(data, source, source_parents)
            self._cache[key] = score
        return score

    # XXX: this is only for Likelihood score for Gaussian data
    def full_score(self, A):
        """Full score of a DAG.

        Given a DAG adjacency A, return the l0-penalized log-likelihood of
        a sample from a single environment, by finding the maximum
        likelihood estimates of the corresponding connectivity matrix
        (weights) and noise term variances.

        Parameters
        ----------
        A : np.array
            The adjacency matrix of a DAG, where A[i,j] != 0 => i -> j.

        Returns
        -------
        score : float
            the penalized log-likelihood score.

        """
        # Compute MLE
        B, omegas = self._mle_full(A)
        # Compute log-likelihood (without log(2π) term)
        K = np.diag(1 / omegas)
        I_B = np.eye(self.p) - B.T
        log_term = self.n * np.log(omegas.prod())
        if self.method == "scatter":
            # likelihood = 0.5 * self.n * (np.log(det_K) - np.trace(K @ I_B @ self._scatter @ I_B.T))
            likelihood = log_term + self.n * np.trace(K @ I_B @ self._scatter @ I_B.T)
        else:
            # Center the data, exclude the intercept column
            inv_cov = I_B.T @ K @ I_B
            cov_term = 0
            for i, x in enumerate(self._centered):
                cov_term += x @ inv_cov @ x
            likelihood = log_term + cov_term
        #   Note: the number of parameters is the number of edges + the p marginal variances
        l0_term = self.lmbda * (np.sum(A != 0) + 1 * self.p)
        score = -0.5 * likelihood - l0_term
        return score


def _mle(data, source, source_parents):
    """Compute the maximum likelihood estimates of the parameters of a linear Gaussian model.

    Parameters
    ----------
    data : pd.DataFrame
        The dataset.
    source : Node
        The origin node.
    source_parents : list of Node
        The parents of the source.

    Returns
    -------
    beta : np.array
        The MLE of the coefficients.
    sigma : float
        The MLE of the noise term variance.
    """
    _, n_features = data.shape
    beta = np.zeros(n_features)

    # compute the MLE of the coefficients
    # using leaset squares regression
    Y = data[source].to_numpy()

    if len(source_parents) > 0:
        X = data[source_parents].to_numpy()
        parents_coef = np.linalg.lstsq(X, Y, rcond=None)[0]
        parents_idx = [data.columns.get_loc(p) for p in source_parents]
        beta[parents_idx] = parents_coef

        # compute the estimate of the noise-term variance
        sigma = np.var(Y - X @ parents_coef)
    
    # XXX: it is possible to compute things using the empirical covariance matrix
    # beta = (\Sigma_{source, Pa(source)} @ \Sigma_{Pa(source), Pa(source)})^{-1}

    if sigma < 0:
        sigma = 1e-5

    return beta, sigma


def bic_score(data, source, source_parents):
    """Compute the Bayesian Information Criterion (BIC) score of an edge.

    Implements the BIC score described in :footcite:`koller2009probabilistic`.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset.
    source : Node
        Variable to score.
    source_parents : list of Node
        The parents of the source.
    """
    n_samples = len(data)

    # compute MLE
    _, sigma = _mle(data, source, source_parents)

    # compute log-likelihood
    likelihood = -0.5 * n_samples * (1 + np.log(sigma))

    # penalty term
    l0_term = 0.5 * np.log(n_samples) * (len(source_parents) + 1)
    return likelihood - l0_term


def bdeu_score(data, source, source_parents):
    """Compute the Bayesian Dirichlet equivalent uniform (BDeu) score of an edge.

    Implements the BDeu score described in :footcite:`koller2009probabilistic`
    and :footcite:`heckerman2013learning`.

    Parameters
    ----------
    data : _type_
        _description_
    source : _type_
        _description_
    source_parents : _type_
        _description_
    """
    pass


def bds_score(data, source, source_parents):
    """Compute the Bayesian Dirichlet sparse (BDs) score of an edge.

    Implements the score described in :footcite:`koller2009probabilistic`.

    Parameters
    ----------
    data : _type_
        _description_
    source : _type_
        _description_
    source_parents : _type_
        _description_
    """
    pass


def k2_score(data, source, source_parents):
    """Compute the K2 score of an edge.

    Implements the score described in :footcite:`scutari2016empirical`.

    Parameters
    ----------
    data : _type_
        _description_
    source : _type_
        _description_
    source_parents : _type_
        _description_
    """
    pass
