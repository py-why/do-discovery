from typing import Any, Dict, Optional, Set, Union

import networkx as nx
import pandas as pd

from ._protocol import Graph
from .context import Context
from .typing import Column


class ContextBuilder:
    """A builder class for creating Context objects ergonomically.

    The context builder provides a way to capture assumptions, domain knowledge, and data.
    """

    _graph: Optional[Graph] = None
    _included_edges: Optional[Union[nx.Graph, nx.DiGraph]] = None
    _excluded_edges: Optional[Union[nx.Graph, nx.DiGraph]] = None
    _observed_variables: Optional[Set[Column]] = None
    _latent_variables: Optional[Set[Column]] = None
    _state_variables: Optional[Dict[str, Any]] = None

    def graph(self, graph: Graph) -> "ContextBuilder":
        """Set the partial graph to start with.

        Parameters
        ----------
        graph : Graph - the new graph instance


        Returns
        -------
        ContextBuilder
            The builder instance
        """
        self._graph = graph
        return self

    def edges(
        self,
        include: Optional[Union[nx.Graph, nx.DiGraph]] = None,
        exclude: Optional[Union[nx.Graph, nx.DiGraph]] = None,
    ) -> "ContextBuilder":
        """Set edge constraints to apply in discovery.

        Parameters
        ----------
        included : Optional[Union[nx.Graph, nx.DiGraph]]
            Edges that should be included in the resultant graph
        excluded : Optional[Union[nx.Graph, nx.DiGraph]]
            Edges that must be excluded in the resultant graph

        Returns
        -------
        ContextBuilder
            The builder instance
        """
        self._included_edges = include
        self._excluded_edges = exclude
        return self

    def variables(
        self,
        observed: Optional[Set[Column]] = None,
        latents: Optional[Set[Column]] = None,
        data: Optional[pd.DataFrame] = None,
    ) -> "ContextBuilder":
        """Set variable-list information to utilize in discovery.

        Parameters
        ----------
        observed : Optional[Set[Column]]
            Set of observed variables, by default None. If neither ``latents``,
            nor ``variables`` is set, then it is presumed that ``variables`` consists
            of the columns of ``data`` and ``latents`` is the empty set.
        latents : Optional[Set[Column]] - variables that are latent
            Set of latent "unobserved" variables, by default None. If neither ``latents``,
            nor ``variables`` is set, then it is presumed that ``variables`` consists
            of the columns of ``data`` and ``latents`` is the empty set.
        data : Optional[pd.DataFrame] - the data to use for variable inference.

        Returns
        -------
        ContextBuilder
            The builder instance
        """
        if data is not None:
            # initialize and parse the set of variables, latents and others
            columns = set(data.columns)
            if observed is not None and latents is not None:
                if columns - set(observed) != set(latents):
                    raise ValueError(
                        "If observed and latents are set, then they must be "
                        "include all columns in data."
                    )
            elif observed is None and latents is not None:
                observed = columns - set(latents)
            elif latents is None and observed is not None:
                latents = columns - set(observed)
            elif observed is None and latents is None:
                # when neither observed, nor latents is set, it is assumed
                # that the data is all "not latent"
                observed = columns
                latents = set()

            observed = set(observed)  # type: ignore
            latents = set(latents)  # type: ignore

        self._observed_variables = observed
        self._latent_variables = latents
        if (self._observed_variables is None) or (self._latent_variables is None):
            raise ValueError("Could not infer variables from data or given arguments.")
        return self

    def state_variables(self, state_variables: Dict[str, Any]) -> "ContextBuilder":
        """Set the state variables to use in discovery.

        Parameters
        ----------
        state_variables : Dict[str, Any]
            The state variables to use in discovery.

        Returns
        -------
        ContextBuilder
            The builder instance
        """
        self._state_variables = state_variables
        return self

    def build(self) -> Context:
        """Build the Context object.

        Returns
        -------
        Context
            The populated Context object
        """
        if (self._observed_variables is None) or (self._latent_variables is None):
            raise ValueError("Must set variables() before building Context.")

        graph = self._graph
        variables = self._observed_variables
        latents = self._latent_variables
        included_edges = self._included_edges
        excluded_edges = self._excluded_edges
        state_variables = self._state_variables

        # initialize the starting graph
        if graph is None:
            graph = nx.complete_graph(variables, create_using=nx.Graph)
        else:
            if set(graph.nodes) != set(variables):
                raise ValueError(
                    f"The nodes within the initial graph, {graph.nodes}, "
                    f"do not match the nodes in the passed in data, {variables}."
                )

        # initialize set of fixed and included edges
        if included_edges is None:
            included_edges = nx.empty_graph(variables, create_using=nx.Graph)
        if excluded_edges is None:
            excluded_edges = nx.empty_graph(variables, create_using=nx.Graph)

        if state_variables is None:
            state_variables = dict()

        return Context(
            init_graph=graph,
            included_edges=included_edges,
            excluded_edges=excluded_edges,
            variables=variables,
            latents=latents,
            state_variables=state_variables,
        )


def make_context() -> ContextBuilder:
    """Create a new ContextBuilder instance.

    Returns
    -------
    ContextBuilder
        The new ContextBuilder instance
    """
    return ContextBuilder()
