from typing import Callable, Optional, Union

import networkx as nx
import pandas as pd
from pywhy_graphs.array.export import clearn_arr_to_graph

from dodiscover.context import Context


# XXX: see https://github.com/juangamella/ges
class GES:
    graph_: Optional[nx.DiGraph]

    def __init__(
        self,
        scoring_method: Union[Callable, str] = "bic",
        max_indegree: int = None,
        **scoring_method_kwargs,
    ) -> None:
        self.scoring_method = scoring_method
        self.max_indegree = max_indegree
        self.scoring_method_kwargs = scoring_method_kwargs

        self.graph_ = None

    def fit(self, df: pd.DataFrame, ctx: Context):
        from causallearn.search.ScoreBased.GES import ges

        X = df.to_numpy()

        # run causal-learn
        ges_record = ges(
            X, score_func=self.scoring_method, maxP=self.max_indegree, **self.scoring_method_kwargs
        )

        causal_learn_graph = ges_record["G"]
        names = [n.name for n in causal_learn_graph.nodes]
        adjmat = causal_learn_graph.graph

        self.causal_learn_graph_ = causal_learn_graph
        self.score_ = ges_record["score"]
        self.graph_ = clearn_arr_to_graph(adjmat, arr_idx=names, graph_type="DiGraph")
        return self
