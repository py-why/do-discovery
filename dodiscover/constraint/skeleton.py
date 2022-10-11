import logging
from collections import defaultdict
from itertools import chain, combinations
from typing import Iterable, Optional, Set, SupportsFloat, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd

from dodiscover.ci import BaseConditionalIndependenceTest
from dodiscover.constraint.config import SkeletonMethods
from dodiscover.typing import Column, SeparatingSet

from ..context import Context
from ..context_builder import make_context

logger = logging.getLogger()


def _iter_conditioning_set(
    possible_variables: Iterable,
    x_var: Union[SupportsFloat, str],
    y_var: Union[SupportsFloat, str],
    size_cond_set: int,
) -> Iterable[Set]:
    """Iterate function to generate the conditioning set.

    Parameters
    ----------
    possible_variables : iterable
        A set/list/dict of possible variables to consider for the conditioning set.
        This can be for example, the current adjacencies.
    x_var : node
        The node for the 'x' variable.
    y_var : node
        The node for the 'y' variable.
    size_cond_set : int
        The size of the conditioning set to consider. If there are
        less adjacent variables than this number, then all variables will be in the
        conditioning set.

    Yields
    ------
    Z : set
        The set of variables for the conditioning set.
    """
    exclusion_set = {x_var, y_var}

    all_adj_excl_current = [p for p in possible_variables if p not in exclusion_set]

    # loop through all possible combinations of the conditioning set size
    for cond in combinations(all_adj_excl_current, size_cond_set):
        cond_set = set(cond)
        yield cond_set


def _find_neighbors_along_path(G: nx.Graph, start, end) -> Set:
    """Find neighbors that are along a path from start to end.

    Parameters
    ----------
    G : nx.Graph
        The graph.
    start : Node
        The starting node.
    end : Node
        The ending node.

    Returns
    -------
    nbrs : Set
        The set of neighbors that are also along a path towards
        the 'end' node.
    """

    def _assign_weight(u, v, edge_attr):
        if u == node or v == node:
            return np.inf
        else:
            return 1

    nbrs = set()
    for node in G.neighbors(start):
        if not G.has_edge(start, node):
            raise RuntimeError(f"{start} and {node} are not connected, but they are assumed to be.")

        # find a path from start node to end
        path = nx.shortest_path(G, source=node, target=end, weight=_assign_weight)
        if len(path) > 0:
            if start in path:
                raise RuntimeError("There is an error with the input. This is not possible.")
            nbrs.add(node)
    return nbrs


class LearnSkeleton:
    """Learn a skeleton graph from a Markovian causal model.

    Parameters
    ----------
    ci_estimator : BaseConditionalIndependenceTest
        The conditional independence test function.
    sep_set : dictionary of dictionary of list of set
        Mapping node to other nodes to separating sets of variables.
        If ``None``, then an empty dictionary of dictionary of list of sets
        will be initialized.
    alpha : float, optional
        The significance level for the conditional independence test, by default 0.05.
    min_cond_set_size : int
        The minimum size of the conditioning set, by default 0. The number of variables
        used in the conditioning set.
    max_cond_set_size : int, optional
        Maximum size of the conditioning set, by default None. Used to limit
        the computation spent on the algorithm.
    max_combinations : int, optional
        The maximum number of conditional independence tests to run from the set
        of possible conditioning sets. By default None, which means the algorithm will
        check all possible conditioning sets. If ``max_combinations=n`` is set, then
        for every conditioning set size, 'p', there will be at most 'n' CI tests run
        before the conditioning set size 'p' is incremented. For controlling the size
        of 'p', see ``min_cond_set_size`` and ``max_cond_set_size``. This can be used
        in conjunction with ``keep_sorted`` parameter to only test the "strongest"
        dependences.
    skeleton_method : SkeletonMethods
        The method to use for testing conditional independence. Must be one of
        ('complete', 'neighbors', 'neighbors_path'). See Notes for more details.
    keep_sorted : bool
        Whether or not to keep the considered conditioning set variables in sorted
        dependency order. If True (default) will sort the existing dependencies of each variable
        by its dependencies from strongest to weakest (i.e. largest CI test statistic value
        to lowest). This can be used in conjunction with ``max_combinations`` parameter
        to only test the "strongest" dependences.
    ci_estimator_kwargs : dict
        Keyword arguments for the ``ci_estimator`` function.

    Attributes
    ----------
    adj_graph_ : nx.Graph
        The discovered graph from data. Stored using an undirected
        graph. The graph contains edge attributes for the smallest value of the
        test statistic encountered (key name 'test_stat'), the largest pvalue seen in
        testing 'x' || 'y' given some conditioning set (key name 'pvalue').
    sep_set_ : dictionary of dictionary of list of set
        Mapping node to other nodes to separating sets of variables.

    Notes
    -----
    Proceed by testing neighboring nodes, while keeping track of test
    statistic values (these are the ones that are
    the "most dependent"). Remember we are testing the null hypothesis

    .. math::
        H_0: X \\perp Y | Z

    where the alternative hypothesis is that they are dependent and hence
    require a causal edge linking the two variables.

    Overview of learning causal skeleton from data:

        This algorithm consists of four general loops through the data.

        - "infinite" loop through size of the conditioning set, 'size_cond_set'. The
          minimum size is set by ``min_cond_set_size``, whereas the maximum is controlled
          by ``max_cond_set_size`` hyperparameter.
        - loop through nodes of the graph, 'x_var'
        - loop through variables adjacent to selected node, 'y_var'
        - loop through combinations of the conditioning set of size p, 'cond_set'.
          The ``max_combinations`` parameter allows one to limit the fourth loop through
          combinations of the conditioning set.

        At each iteration of the outer infinite loop, the edges that were deemed
        independent for a specific 'size_cond_set' are removed and 'size_cond_set' is incremented.

        Furthermore, the maximum pvalue is stored for existing
        dependencies among variables (i.e. any two nodes with an edge still).
        The ``keep_sorted`` hyperparameter keeps the considered neighbors in
        a sorted order.

        The stopping condition is when the size of the conditioning variables for all (X, Y)
        pairs is less than the size of 'size_cond_set', or if the 'max_cond_set_size' is
        reached.

    Different methods for learning the skeleton:

        There are different ways to learn the skeleton that are valid under various
        assumptions. The value of ``skeleton_method`` completely defines how one
        selects the conditioning set.

        - 'complete': This exhaustively conditions on all combinations of variables in
          the graph. This essentially refers to the SGS algorithm :footcite:`Spirtes1993`
        - 'neighbors': This only conditions on adjacent variables to that of 'x_var' and 'y_var'.
          This refers to the traditional PC algorithm :footcite:`Meek1995`
        - 'neighbors_path': This is 'neighbors', but restricts to variables with an adjacency path
          from 'x_var' to 'y_var'. This is a variant from the RFCI paper :footcite:`Colombo2012`
    """

    adj_graph_: nx.Graph
    sep_set_: SeparatingSet
    remove_edges: Set
    context: Context
    min_cond_set_size_: int
    max_cond_set_size_: int
    max_combinations_: int

    def __init__(
        self,
        ci_estimator: BaseConditionalIndependenceTest,
        sep_set: Optional[SeparatingSet] = None,
        alpha: float = 0.05,
        min_cond_set_size: int = 0,
        max_cond_set_size: Optional[int] = None,
        max_combinations: Optional[int] = None,
        skeleton_method: SkeletonMethods = SkeletonMethods.NBRS,
        keep_sorted: bool = False,
        **ci_estimator_kwargs,
    ) -> None:
        self.ci_estimator = ci_estimator
        self.sep_set = sep_set
        self.alpha = alpha
        self.ci_estimator_kwargs = ci_estimator_kwargs
        self.skeleton_method = skeleton_method

        # control of the conditioning set
        self.min_cond_set_size = min_cond_set_size
        self.max_cond_set_size = max_cond_set_size
        self.max_combinations = max_combinations

        # for tracking strength of dependencies
        self.keep_sorted = keep_sorted

        # debugging mode
        self.n_ci_tests = 0

    def _initialize_params(self) -> None:
        """Initialize parameters for learning skeleton.

        Parameters
        ----------
        nodes : list of nodes
            The list of nodes that will be present in the learned skeleton graph.
        """
        # error checks of passed in arguments
        if self.max_combinations is not None and self.max_combinations <= 0:
            raise RuntimeError(f"Max combinations must be at least 1, not {self.max_combinations}")

        if self.skeleton_method not in SkeletonMethods:
            raise ValueError(
                f"Skeleton method must be one of {SkeletonMethods}, not {self.skeleton_method}."
            )

        if self.sep_set is None:
            # keep track of separating sets
            self.sep_set_ = defaultdict(lambda: defaultdict(list))
        else:
            self.sep_set_ = self.sep_set

        # control of the conditioning set
        if self.max_cond_set_size is None:
            self.max_cond_set_size_ = np.inf
        else:
            self.max_cond_set_size_ = self.max_cond_set_size
        if self.min_cond_set_size is None:
            self.min_cond_set_size_ = 0
        else:
            self.min_cond_set_size_ = self.min_cond_set_size
        if self.max_combinations is None:
            self.max_combinations_ = np.inf
        else:
            self.max_combinations_ = self.max_combinations

    def evaluate_edge(
        self, data: pd.DataFrame, X: Column, Y: Column, Z: Optional[Set[Column]] = None
    ) -> Tuple[float, float]:
        """Test any specific edge for X || Y | Z.

        Parameters
        ----------
        data : pd.DataFrame
            The dataset
        X : column
            A column in ``data``.
        Y : column
            A column in ``data``.
        Z : set, optional
            A list of columns in ``data``, by default None.

        Returns
        -------
        test_stat : float
            Test statistic.
        pvalue : float
            The pvalue.
        """
        if Z is None:
            Z = set()
        test_stat, pvalue = self.ci_estimator.test(data, {X}, {Y}, Z, **self.ci_estimator_kwargs)
        self.n_ci_tests += 1
        return test_stat, pvalue

    def fit(self, data: pd.DataFrame, context: Context) -> None:
        """Run structure learning to learn the skeleton of the causal graph.

        Parameters
        ----------
        data : pd.DataFrame
            The data to learn the causal graph from.
        context : Context
            A context object.
        """
        self.context = make_context(context).build()

        # get the initialized graph
        adj_graph = self.context.init_graph
        X = data

        # track progress of the algorithm for which edges to remove to ensure stability
        self.remove_edges = set()

        # initialize learning parameters
        self._initialize_params()

        # the size of the conditioning set will start off at the minimum
        size_cond_set = self.min_cond_set_size_

        edge_attrs = set(chain.from_iterable(d.keys() for *_, d in adj_graph.edges(data=True)))
        if "test_stat" in edge_attrs or "pvalue" in edge_attrs:
            raise RuntimeError(
                "Running skeleton discovery with adjacency graph "
                "with 'test_stat' or 'pvalue' is not supported yet."
            )

        # store the absolute value of test-statistic values and pvalue for
        # every single candidate parent-child edge (X -> Y)
        nx.set_edge_attributes(adj_graph, np.inf, "test_stat")
        nx.set_edge_attributes(adj_graph, -1e-5, "pvalue")

        logger.info(
            f"\n\nRunning skeleton phase with: \n"
            f"max_combinations: {self.max_combinations_},\n"
            f"min_cond_set_size: {self.min_cond_set_size_},\n"
            f"max_cond_set_size: {self.max_cond_set_size_},\n"
        )

        # Outer loop: iterate over 'size_cond_set' until stopping criterion is met
        # - 'size_cond_set' > 'max_cond_set_size' or
        # - All (X, Y) pairs have candidate conditioning sets of size < 'size_cond_set'
        while 1:
            cont = False
            # initialize set of edges to remove at the end of every loop
            self.remove_edges = set()

            # loop through every node
            for x_var in adj_graph.nodes:
                possible_adjacencies = set(adj_graph.neighbors(x_var))

                logger.info(f"Considering node {x_var}...\n\n")

                for y_var in possible_adjacencies:
                    # a node cannot be a parent to itself in DAGs
                    if y_var == x_var:
                        continue

                    # ignore fixed edges
                    if (x_var, y_var) in self.context.included_edges.edges:
                        continue

                    # compute the possible variables used in the conditioning set
                    possible_variables = self._compute_candidate_conditioning_sets(
                        adj_graph,
                        x_var,
                        y_var,
                        skeleton_method=self.skeleton_method,
                    )

                    logger.debug(
                        f"Adj({x_var}) without {y_var} with size={len(possible_adjacencies) - 1} "
                        f"with p={size_cond_set}. The possible variables to condition on are: "
                        f"{possible_variables}."
                    )

                    # check that number of adjacencies is greater then the
                    # cardinality of the conditioning set
                    if len(possible_variables) < size_cond_set:
                        logger.debug(
                            f"\n\nBreaking for {x_var}, {y_var}, {len(possible_adjacencies)}, "
                            f"{size_cond_set}, {possible_variables}"
                        )
                        continue
                    else:
                        cont = True

                    # generate iterator through the conditioning sets
                    conditioning_sets = _iter_conditioning_set(
                        possible_variables=possible_variables,
                        x_var=x_var,
                        y_var=y_var,
                        size_cond_set=size_cond_set,
                    )

                    # now iterate through the possible parents
                    for comb_idx, cond_set in enumerate(conditioning_sets):
                        # check the number of combinations of possible parents we have tried
                        # to use as a separating set
                        if (
                            self.max_combinations_ is not None
                            and comb_idx >= self.max_combinations_
                        ):
                            break

                        # compute conditional independence test
                        test_stat, pvalue = self.evaluate_edge(X, x_var, y_var, set(cond_set))

                        # if any "independence" is found through inability to reject
                        # the null hypothesis, then we will break the loop comparing X and Y
                        # and say X and Y are conditionally independent given 'cond_set'
                        if pvalue > self.alpha:
                            break

                    # post-process the CI test results
                    removed_edge = self._postprocess_ci_test(
                        adj_graph, x_var, y_var, cond_set, test_stat, pvalue
                    )

                    # summarize the comparison of XY
                    self._summarize_xy_comparison(x_var, y_var, removed_edge, pvalue)

            # finally remove edges after performing
            # conditional independence tests
            logger.info(f"For p = {size_cond_set}, removing all edges: {self.remove_edges}")

            # Remove non-significant links
            # Note: Removing edges at the end ensures "stability" of the algorithm
            # with respect to the randomness choice of pairs of edges considered in the inner loop
            adj_graph.remove_edges_from(self.remove_edges)

            # increment the conditioning set size
            size_cond_set += 1

            # only allow conditioning set sizes up to maximum set number
            if size_cond_set > self.max_cond_set_size_ or cont is False:
                break

        self.adj_graph_ = adj_graph

    def _summarize_xy_comparison(
        self, x_var: Column, y_var: Column, removed_edge: bool, pvalue: float
    ) -> None:
        # exit loop if we have found an independency and removed the edge
        if removed_edge:
            remove_edge_str = "Removing edge"
        else:
            remove_edge_str = "Did not remove edge"

        logger.info(
            f"{remove_edge_str} between {x_var} and {y_var}... \n"
            f"Statistical summary:\n"
            f"- PValue={pvalue} at alpha={self.alpha}"
        )

    def _compute_candidate_conditioning_sets(
        self, adj_graph: nx.Graph, x_var: Column, y_var: Column, skeleton_method: SkeletonMethods
    ) -> Set[Column]:
        """Compute candidate conditioning sets.

        Parameters
        ----------
        adj_graph : nx.Graph
            The current adjacency graph.
        x_var : node
            The 'X' node.
        y_var : node
            The 'Y' node.
        skeleton_method : SkeletonMethods
            The skeleton method, which dictates how we choose the corresponding
            conditioning sets.

        Returns
        -------
        possible_variables : Set
            The set of nodes in 'adj_graph' that are candidates for the
            conditioning set.
        """
        if skeleton_method == SkeletonMethods.COMPLETE:
            possible_variables = set(adj_graph.nodes)
        elif skeleton_method == SkeletonMethods.NBRS:
            possible_variables = set(adj_graph.neighbors(x_var))
            # possible_adjacencies.copy()
        elif skeleton_method == SkeletonMethods.NBRS_PATH:
            # constrain adjacency set to ones with a path from x_var to y_var
            possible_variables = _find_neighbors_along_path(adj_graph, start=x_var, end=y_var)

        if self.keep_sorted:
            # Note it is assumed in public API that 'test_stat' is set
            # inside the adj_graph
            possible_variables = sorted(
                possible_variables,
                key=lambda n: adj_graph.edges[x_var, n]["test_stat"],
                reverse=True,
            )  # type: ignore

        if x_var in possible_variables:
            possible_variables.remove(x_var)
        if y_var in possible_variables:
            possible_variables.remove(y_var)

        return possible_variables

    def _postprocess_ci_test(
        self,
        adj_graph: nx.Graph,
        x_var: Column,
        y_var: Column,
        cond_set: Set[Column],
        test_stat: float,
        pvalue: float,
    ) -> bool:
        # keep track of the smallest test statistic, meaning the highest pvalue
        # meaning the "most" independent. keep track of the maximum pvalue as well
        if pvalue > adj_graph.edges[x_var, y_var]["pvalue"]:
            adj_graph.edges[x_var, y_var]["pvalue"] = pvalue
        if test_stat < adj_graph.edges[x_var, y_var]["test_stat"]:
            adj_graph.edges[x_var, y_var]["test_stat"] = test_stat

        # two variables found to be independent given a separating set
        if pvalue > self.alpha:
            self.remove_edges.add((x_var, y_var))
            self.sep_set_[x_var][y_var].append(set(cond_set))
            self.sep_set_[y_var][x_var].append(set(cond_set))
            return True
        return False


class LearnSemiMarkovianSkeleton(LearnSkeleton):
    """Learning a skeleton from a semi-markovian causal model.

    This proceeds by learning a skeleton by testing edges with candidate
    separating sets from the "possibly d-separating" sets (PDS), or PDS
    sets that lie on a path between two nodes :footcite:`Spirtes1993`.
    This algorithm requires the input of a collider-oriented PAG, which
    provides the necessary information to compute the PDS set for any
    given nodes. See Notes for more details.

    Parameters
    ----------
    ci_estimator : BaseConditionalIndependenceTest
        The conditional independence test function.
    sep_set : dictionary of dictionary of list of set
        Mapping node to other nodes to separating sets of variables.
        If ``None``, then an empty dictionary of dictionary of list of sets
        will be initialized.
    alpha : float, optional
        The significance level for the conditional independence test, by default 0.05.
    min_cond_set_size : int
        The minimum size of the conditioning set, by default 0. The number of variables
        used in the conditioning set.
    max_cond_set_size : int, optional
        Maximum size of the conditioning set, by default None. Used to limit
        the computation spent on the algorithm.
    max_combinations : int, optional
        The maximum number of conditional independence tests to run from the set
        of possible conditioning sets. By default None, which means the algorithm will
        check all possible conditioning sets. If ``max_combinations=n`` is set, then
        for every conditioning set size, 'p', there will be at most 'n' CI tests run
        before the conditioning set size 'p' is incremented. For controlling the size
        of 'p', see ``min_cond_set_size`` and ``max_cond_set_size``. This can be used
        in conjunction with ``keep_sorted`` parameter to only test the "strongest"
        dependences.
    skeleton_method : SkeletonMethods
        The method to use for testing conditional independence. Must be one of
        ('pds', 'pds_path'). See Notes for more details.
    keep_sorted : bool
        Whether or not to keep the considered conditioning set variables in sorted
        dependency order. If True (default) will sort the existing dependencies of each variable
        by its dependencies from strongest to weakest (i.e. largest CI test statistic value
        to lowest). This can be used in conjunction with ``max_combinations`` parameter
        to only test the "strongest" dependences.
    max_path_length : int, optional
        The maximum length of any discriminating path, or None if unlimited.
    ci_estimator_kwargs : dict
        Keyword arguments for the ``ci_estimator`` function.

    Notes
    -----
    To learn the skeleton of a Semi-Markovian causal model, one approach is to consider
    the possibly d-separating (PDS) set, which is a superset of the d-separating sets in
    the true causal model. Knowing the PDS set requires knowledge of the skeleton and orientation
    of certain edges. Therefore, we first learn an initial skeleton by checking conditional
    independences with respect to node neighbors. From this, one can orient certain colliders.
    The resulting PAG can now be used to enumerate the PDS sets for each node, which
    are now conditioning candidates to check for conditional independence.

    For visual examples, see Figures 16, 17 and 18 in :footcite:`Spirtes1993`. Also,
    see the RFCI paper for other examples :footcite:`Colombo2012`.

    Different methods for learning the skeleton:

        There are different ways to learn the skeleton that are valid under various
        assumptions. The value of ``skeleton_method`` completely defines how one
        selects the conditioning set.

        - 'pds': This conditions on the PDS set of 'x_var'. Note, this definition does
          not rely on 'y_var'. See :footcite:`Spirtes1993`.
        - 'pds_path': This is 'pds', but restricts to variables with a possibly directed path
          from 'x_var' to 'y_var'. This is a variant from the RFCI paper :footcite:`Colombo2012`.

    References
    ----------
    .. footbibliography::
    """

    def __init__(
        self,
        ci_estimator: BaseConditionalIndependenceTest,
        sep_set: Optional[SeparatingSet] = None,
        alpha: float = 0.05,
        min_cond_set_size: int = 0,
        max_cond_set_size: Optional[int] = None,
        max_combinations: Optional[int] = None,
        skeleton_method: SkeletonMethods = SkeletonMethods.PDS,
        keep_sorted: bool = False,
        max_path_length: Optional[int] = None,
        **ci_estimator_kwargs,
    ) -> None:
        super().__init__(
            ci_estimator,
            sep_set,
            alpha,
            min_cond_set_size,
            max_cond_set_size,
            max_combinations,
            skeleton_method,
            keep_sorted,
            **ci_estimator_kwargs,
        )
        if max_path_length is None:
            max_path_length = np.inf
        self.max_path_length = max_path_length

    def _compute_candidate_conditioning_sets(
        self, adj_graph: nx.Graph, x_var: Column, y_var: Column, skeleton_method: SkeletonMethods
    ) -> Set[Column]:
        import pywhy_graphs as pgraph

        # get PAG from the context object
        pag = self.context.state_variable("PAG")

        if skeleton_method == SkeletonMethods.PDS:
            # determine how we want to construct the candidates for separating nodes
            # perform conditioning independence testing on all combinations
            possible_variables = pgraph.pds(
                pag, x_var, y_var, max_path_length=self.max_path_length  # type: ignore
            )
        elif skeleton_method == SkeletonMethods.PDS_PATH:
            # determine how we want to construct the candidates for separating nodes
            # perform conditioning independence testing on all combinations
            possible_variables = pgraph.pds_path(
                pag, x_var, y_var, max_path_length=self.max_path_length  # type: ignore
            )

        if self.keep_sorted:
            # Note it is assumed in public API that 'test_stat' is set
            # inside the adj_graph
            possible_variables = sorted(
                possible_variables,
                key=lambda n: adj_graph.edges[x_var, n]["test_stat"],
                reverse=True,
            )  # type: ignore

        if x_var in possible_variables:
            possible_variables.remove(x_var)
        if y_var in possible_variables:
            possible_variables.remove(y_var)

        return possible_variables

    def fit(self, data: pd.DataFrame, context: Context) -> None:
        return super().fit(data, context)
