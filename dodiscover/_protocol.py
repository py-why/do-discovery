from typing import Dict, FrozenSet, Iterable, List, Optional, Protocol

import networkx as nx

from .typing import Column


class Graph(Protocol):
    """Protocol for graphs to work with dodiscover algorithms."""

    @property
    def nodes(self) -> Iterable:
        """Return an iterable over nodes in graph."""
        pass

    def edges(self, data=None) -> Iterable:
        """Return an iterable over edge tuples in graph."""
        pass

    def has_edge(self, u, v, edge_type="any") -> bool:
        """Check if graph has an edge for a specific edge type."""
        pass

    def add_node(self, node_for_adding, **attr) -> None:
        """Add a node to the graph."""
        pass

    def remove_node(self, u) -> None:
        """Remove a node from the graph."""
        pass

    def remove_edges_from(self, edges: List) -> None:
        """Remove a set of edges from the graph."""
        pass

    def add_edge(self, u, v, edge_type="all") -> None:
        """Add an edge to the graph."""
        pass

    def remove_edge(self, u, v, edge_type="all") -> None:
        """Remove an edge from the graph."""
        pass

    def neighbors(self, node) -> Iterable:
        """Iterate over all nodes that have any edge connection with 'node'."""
        pass

    def to_undirected(self) -> nx.Graph:
        """Convert a graph to a fully undirected networkx graph.

        All nodes are connected by an undirected edge if there are any
        edges between the two.
        """
        pass

    def subgraph(self, nodes):
        """Get subgraph based on nodes."""
        pass

    def copy(self):
        """Create a copy of the graph."""
        pass


class TimeSeriesGraph(Graph, Protocol):
    """A protocol for time-series graphs."""

    @property
    def max_lag(self) -> int:
        pass

    def lagged_neighbors(self, u: Column) -> List[Column]:
        """Return neighbors of u that are in a previous time point."""
        pass

    def contemporaneous_neighbors(self, u: Column) -> List[Column]:
        """Return neighbors of u that are in the same time point."""
        pass

    # TODO: refactor to
    # 1. remove_forward_homologous_edges(self, u, v)
    # 2. remove_backward_homologous_edges(self, u, v)
    def set_auto_removal(self, auto: Optional[str]) -> None:
        """Specify how to auto-remove homologous edges."""
        pass

    def nodes_at(self, t):
        """Nodes at specific time point."""
        pass


class EquivalenceClass(Graph, Protocol):
    """Protocol for equivalence class of graphs."""

    @property
    def excluded_triples(self) -> Dict[FrozenSet, None]:
        """A set of triples that are excluded from orientation."""
        pass

    @property
    def directed_edge_name(self) -> str:
        """Name of the directed edges."""
        pass

    @property
    def undirected_edge_name(self) -> str:
        """Name of the undirected edges."""
        pass

    @property
    def circle_edge_name(self) -> str:
        """Name of the circle edge endpoints."""
        pass

    @property
    def bidirected_edge_name(self) -> str:
        """Name of the bidirected edges."""
        pass

    def get_graphs(self, graph_name: Optional[str] = None):
        """Get edge sub-graph."""
        pass

    def orient_uncertain_edge(self, u, v) -> None:
        """Orients an uncertain edge in the equivalence class to directed ``'u'*->'v'``."""
        pass

    def mark_unfaithful_triple(self, v_i, u, v_j) -> None:
        """Mark a triple as unfaithful, and put it in the excluded triple set."""
        pass

    def predecessors(self, node) -> Iterable:
        """Nodes with directed edges pointing to 'node'."""
        pass

    def successors(self, node) -> Iterable:
        """Nodes with directed edges pointing from 'node'."""
        pass
