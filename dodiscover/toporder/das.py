import numpy as np
from numpy.typing import NDArray
from scipy.stats import ttest_ind

from dodiscover.toporder._base import SteinMixin
from dodiscover.toporder.score import SCORE
from dodiscover.toporder.utils import fullAdj2Order


class DAS(SCORE):
    """The Discovery At Scale (DAS) algorithm :footcite:`Montagna2023a` for causal discovery.

    The method infer the topological ordering using SCORE :footcite:`rolland2022`.
    Then it prunes the fully connected DAG by inspection of the non diagonal entries of the Hessian of the log likelihood.
    A final, computationally cheap, pruning step is performed with CAM pruning :footcite:`Buhlmann2013`.
    The method assumes Additive Noise Model and Gaussianity of the noise terms.
    DAS is a highly scalable method, allowing to run causal discovery on thousands of nodes.
    It reduces the computational complexity of the pruning method of an order of magnitude with respect to SCORE.

    Parameters
    ----------
    eta_G: float
        Regularization parameter for Stein gradient estimator
    eta_H : float
        Regularization parameter for Stein Hessian estimator
    cam_cutoff : float
        alpha value for independence testing for CAM edge pruning
    das_cutoff : float
        alpha value for hypothesis testing in preliminary DAS pruning
    n_splines : int
        Number of splines to use for feature selection with GAM (Generalized Additive Model) fitting.
        Automatically decreased in case of insufficient samples
    splines_degree: int
        Order of splines for feature selection with GAM (Generalized Additive Model) fitting.
    min_parents : int
        Minimum number of edges retained by DAS preliminary pruning step.
        min_parents < 5 doesn't significantly affects execution time, while increasing the accuracy.
    max_parents : int
        Maximum number of parents allowed for a single node.
        Given that CAM pruning is inefficient for > ~20 nodes, larger values are not advised.
        The value of max_parents should be decrease under the assumption of sparse graphs.
    """

    def __init__(
        self,
        eta_G: float = 0.001,
        eta_H: float = 0.001,
        cam_cutoff: float = 0.001,
        das_cutoff: float = 0.01,
        n_splines: int = 10,
        splines_degree: int = 3,
        min_parents: int = 5,
        max_parents: int = 20,
    ):
        super().__init__(
            eta_G, eta_H, cam_cutoff, n_splines, splines_degree, estimate_variance=True, pns=False
        )
        self.min_parents = min_parents
        self.max_parents = max_parents
        self.das_cutoff = das_cutoff

    def prune(self, X: NDArray, A_dense: NDArray) -> NDArray:
        """
        DAS preliminary pruning of A_dense matrix representation of a fully connected graph.

        Parameters
        ----------
        X : np.ndarray
            n x d matrix of the data
        A_dense : np.ndarray
            fully connected matrix corresponding to a topological ordering

        Return
        ------
        np.ndarray
            Sparse adjacency matrix representing the pruned DAG.
        """
        _, d = X.shape
        order = fullAdj2Order(A_dense)
        stein = SteinMixin()
        max_parents = self.max_parents + 1  # +1 to account for A[l, l] = 1
        remaining_nodes = list(range(d))
        A_das = np.zeros((d, d))

        hess = stein.hessian(X, eta_G=self.eta_G, eta_H=self.eta_H)
        for i in range(d - 1):
            l = order[::-1][i]
            hess_l = hess[:, l, :][:, remaining_nodes]
            hess_m = np.abs(np.median(hess_l * self.var[l], axis=0))
            max_parents = min(max_parents, len(remaining_nodes))

            # Find index of the reference for the hypothesis testing
            topk_indices = np.argsort(hess_m)[::-1][:max_parents]
            topk_values = hess_m[topk_indices]  # largest
            argmin = topk_indices[np.argmin(topk_values)]

            # Edges selection step
            parents = []
            hess_l = np.abs(hess_l)
            l_index = remaining_nodes.index(
                l
            )  # leaf index in the remaining nodes (from 0 to len(remaining_nodes)-1)
            for j in range(max_parents):
                node = topk_indices[j]
                if node != l_index:  # enforce diagonal elements = 0
                    if j < self.min_parents:  # do not filter minimum number of parents
                        parents.append(remaining_nodes[node])
                    else:  # filter potential parents with hp testing
                        # Use hess_l[:, argmin] as sample from a zero mean population (implicit assumption: argmin corresponding to zero mean of the hessian entry)
                        _, pval = ttest_ind(
                            hess_l[:, node],
                            hess_l[:, argmin],
                            alternative="greater",
                            equal_var=False,
                        )
                        if pval < self.das_cutoff:
                            parents.append(remaining_nodes[node])

            A_das[parents, l] = 1
            del remaining_nodes[l_index]

        return super().prune(X, A_das)
