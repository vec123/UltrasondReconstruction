"""GraphBuilder strategies (INSTRUCTIONS.md T4).

Strategy: points -> (graph, supergraph|None), 
selected by which class the caller instantiates.
 it calls the ``get_graphs_from_vertices`` / ``build_super_graph`` 
 primitives directly rather than inventing new code.
"""

from abc import ABC, abstractmethod

import torch

from src.graphs.graphs import get_graphs_from_vertices, build_super_graph


class GraphBuilder(ABC):
    """Strategy: geometry -> (graph, supergraph|None)."""

    @abstractmethod
    def build(self, vertices, mask, rng, areas=None, normals=None):
        """Returns (graph, supergraph). `supergraph` may be None.

        `rng` is passed in, never stored: same rng state + same spec => same graph.
        """
        ...


class RadiusGraphBuilder(GraphBuilder):
    """Radius-ball graph + optional supernode aggregation -- today's only strategy.

    Moves (does not rewrite) ``build_training_graph``'s logic behind the
    ``GraphBuilder`` interface: same two calls (``get_graphs_from_vertices`` then,
    if ``spec.use_supernodes``, ``build_super_graph``), same node-feature fallback.

    Unlike the pre-T4 loader, this DOES thread ``spec.noise_std`` through to
    ``get_graphs_from_vertices`` -- ``build_training_graph`` could never do that
    because it hardcoded ``noise_std=0.0`` and took no such parameter (see
    ``GraphSpec``'s docstring). At the default ``noise_std=0.0`` this is a no-op.
    """

    # build_training_graph's fixed defaults for the two bipartite-aggregation knobs
    # that never made it onto GraphSpec (T3): the pre-T4 loader never forwarded
    # them either, so they've always been fixed at these values, not configurable.
    _BIPARTITE_SEED = None
    _BIPARTITE_MAX_NEIGHBORS = 1024

    def __init__(self, spec):
        self.spec = spec

    def build(self, vertices, mask, rng, areas=None, normals=None):
        spec = self.spec.resolve(rng)

        graph = get_graphs_from_vertices(
            vertices, masks=mask, r_max=spec.r_max, dropout_rate=spec.dropout_rate,
            noise_std=spec.noise_std, key=rng, sampling_mode=spec.sampling_mode_graph,
            areas=areas, normals=normals,
            recompute_area=spec.recompute_area, area_k=spec.area_k,
        )

        if spec.use_supernodes:
            supergraph = build_super_graph(
                vertices, mask, graph,
                num_samples=spec.n_supernodes, r_max=spec.r_supergraph,
                mode=spec.sampling_mode_supernodes,
                seed=self._BIPARTITE_SEED,
                max_num_neighbors=self._BIPARTITE_MAX_NEIGHBORS,
            )
        else:
            supergraph = None

        # get_graphs_from_vertices leaves `graph.x` unset; the encoder consumes it as
        # the constant 1x0e node feature (mirrors helpers.py:168-171).
        if not hasattr(graph, 'x') or graph.x is None:
            graph.x = torch.ones(graph.num_nodes, 1)
        if supergraph is not None and (not hasattr(supergraph, 'x') or supergraph.x is None):
            supergraph.x = torch.ones(supergraph.num_nodes, 1)

        return graph, supergraph


class KNNGraphBuilder(GraphBuilder):
    """Stub: k-nearest-neighbour graph construction is not implemented.

    A second GraphBuilder is what proves the Strategy abstraction is real -- this
    stays a stub, honestly, until something actually needs it.
    """

    def __init__(self, spec):
        self.spec = spec

    def build(self, vertices, mask, rng, areas=None, normals=None):
        raise NotImplementedError(
            "KNNGraphBuilder is not implemented yet. Use RadiusGraphBuilder, or "
            "implement kNN graph construction here (model it after RadiusGraphBuilder "
            "and src/graphs/graphs.py's build_radius_graph)."
        )