import torch
from dataclasses import dataclass

class CohortLoader:
    """Load cohort shapes with graphs, mini-batched over sources.

    Converts cohort Shape objects → graph/super_graph/shape tuples, with optional
    mini-batching over cohort members. Designed to feed both CohortRegistration (T27+)
    and encoder integration (T29+).

    Yields: (graph, super_graph, source_shape_or_list, template_shape)
    where source is a single Shape if batch_size=1, or list of Shapes if batch_size>1.
    The graph is precomputed for encoder use.
    """
    def __init__(self, cohort_shapes, template, builder,
                 rng=None, batch_size=None):
        """Initialize cohort loader.

        Args:
            cohort_shapes: list of Shape objects (cohort members)
            template: Shape object (target template)
            builder: GraphBuilder instance for graph construction
            rng: torch.Generator for reproducible mini-batching (optional)
            batch_size: mini-batch size over cohort, or None for full batch each step
        """
        self.cohort_shapes = cohort_shapes
        self.template = template
        self.builder = builder
        self.rng = rng
        self.batch_size = batch_size
        self.n_shapes = len(cohort_shapes)

    def _pick_indices(self):
        """Draw a fresh random subset of cohort indices, or None for full batch."""
        n = self.n_shapes
        if self.batch_size is None or self.batch_size >= n:
            return None
        return torch.randperm(n, generator=self.rng)[:self.batch_size]

    def __iter__(self):
        while True:
            indices = self._pick_indices()

            # Select cohort members
            if indices is None:
                selected_shapes = self.cohort_shapes
            else:
                selected_shapes = [self.cohort_shapes[i] for i in indices]

            # Concatenate vertices/mask/areas/normals from selected shapes for graph construction
            vertices = torch.cat([s.points for s in selected_shapes], dim=0)
            mask = torch.ones(vertices.shape[0], dtype=torch.bool)

            areas = None
            if selected_shapes[0].weights is not None:
                areas = torch.cat([s.weights for s in selected_shapes], dim=0)

            normals = None
            if selected_shapes[0].normals is not None:
                normals = torch.cat([s.normals for s in selected_shapes], dim=0)

            # Build graphs over concatenated vertices
            graph, super_graph = self.builder.build(vertices, mask, self.rng,
                                                     areas=areas, normals=normals)

            # Yield graph + original Shape objects (which have correct faces/weights/normals)
            if len(selected_shapes) == 1:
                yield (graph, super_graph, selected_shapes[0], self.template)
            else:
                yield (graph, super_graph, selected_shapes, self.template)


class OneBatchLoader:
    """Yields the same prebuilt (graph, true_verts, mask) batch each step."""
    def __init__(self, batch):
        self.batch = batch

    def __iter__(self):
        while True:
            yield self.batch


class ResamplingGraphLoader:
    """Rebuilds the encoder graph from raw geometry each step with a fresh rng.

    Graph construction is delegated to a ``GraphBuilder`` strategy
    (src/learning/data/builders.py): the loader calls
    ``builder.build(verts, mask, rng, areas=..., normals=...)`` and knows nothing
    about radii, dropout, or supernodes -- that policy lives on the ``GraphSpec``
    the builder was constructed with. Swapping radius-graph for kNN construction is
    now "pass a different ``GraphBuilder``", not a change to this class.

    ``vertices`` / ``mask`` / ``areas`` / ``normals`` are direct arguments: they are
    the loader's fixed per-shape geometry (parallel arrays, indexed identically), not a
    construction choice, so they don't belong on the builder/spec.

    The ``rng`` generator's state advances as it is consumed, giving a different graph
    each step while staying reproducible from the seed.

    ``two_view`` makes the contrastive path OPTIONAL: when False (default) each step
    yields the ordinary single-view ``(graph, super, true_verts, mask)`` four-tuple; when
    True it draws TWO independent graphs (different vertex dropout / sampling) of the same
    shapes and yields ``(graph_a, super_a, true_verts, mask, graph_b, super_b)`` -- view A
    in the first four slots so the logger (which reads ``batch[:4]``) is unaffected. The
    six-tuple is what ``TrainingStepper.train_step`` turns into a "same shape -> same
    encoding" alignment loss.

    ``batch_size`` selects mini-batching over SHAPES (dim 0 of ``vertices``): when set and
    smaller than the dataset, each step draws a fresh random subset of that many shapes and
    builds the graph from only those. The yielded ``true_verts`` / ``mask`` are sliced to
    the SAME subset so the decoder's reconstruction target matches the encoder's input; in
    ``two_view`` mode both views use the same subset. ``None`` (default) keeps the original
    full-batch behaviour (every shape, every step).
    """
    def __init__(self, vertices, mask, builder,
                 rng=None, two_view=False, batch_size=None,
                 areas=None, normals=None):

        self.vertices = vertices
        self.mask = mask
        self.builder = builder
        self.rng = rng
        self.two_view = two_view
        self.batch_size = batch_size
        self.areas = areas
        self.normals = normals

    def _pick_indices(self):
        """Draw a fresh random subset of shape indices, or None for the full batch."""
        n = self.vertices.shape[0]
        if self.batch_size is None or self.batch_size >= n:
            return None
        return torch.randperm(n, generator=self.rng)[:self.batch_size]

    def _draw(self, idx=None):
        """Build one fresh view (new dropout / node sampling) from the fixed geometry.

        ``idx`` (if given) restricts the graph to that subset of shapes; the matching
        sliced ``(vertices, mask)`` are returned alongside so the caller can use them as
        the reconstruction target.
        """
        verts = self.vertices if idx is None else self.vertices[idx]
        mask = self.mask if idx is None else self.mask[idx]
        areas = self.areas if (idx is None or self.areas is None) else self.areas[idx]
        normals = self.normals if (idx is None or self.normals is None) else self.normals[idx]

        graph, super_graph = self.builder.build(verts, mask, self.rng,
                                                  areas=areas, normals=normals)
        return graph, super_graph, verts, mask

    def __iter__(self):
        while True:
            idx = self._pick_indices()
            graph_a, super_a, verts, mask = self._draw(idx)
            if self.two_view:
                # Same shape subset, fresh sampling -> "same shape -> same encoding".
                graph_b, super_b, _, _ = self._draw(idx)
                yield (graph_a, super_a, verts, mask, graph_b, super_b)
            else:
                yield (graph_a, super_a, verts, mask)


@dataclass
class CohortBatch:
    """Batch of shapes with points, shape_ids, optional weights, and per-shape faces."""
    points: torch.Tensor
    shape_ids: torch.Tensor
    weights: torch.Tensor = None
    faces: list = None  # List of [F, 3] face tensors, one per shape in batch


class CohortBatchLoader:
    """Yields batches of cohort shapes for CohortRegistration training.

    Mini-batches shapes over cohort members, yielding (template_batch, sample_batch) tuples
    where template_batch is the canonical reference [1, M, 3] and sample_batch is a CohortBatch
    with shape_ids [B, N, 3] containing the cohort members. Shapes are kept separate in the
    batch dimension, not concatenated.
    """
    def __init__(self, cohort_shapes, template, batch_size=None, rng=None):
        """Initialize cohort batch loader.

        Args:
            cohort_shapes: list of Shape objects (cohort members to augment and encode)
            template: Shape object (canonical reference, where flow starts)
            batch_size: mini-batch size over cohort, or None for full batch
            rng: torch.Generator for reproducible mini-batching (optional)
        """
        self.cohort_shapes = cohort_shapes
        self.template = template
        self.batch_size = batch_size or len(cohort_shapes)
        self.rng = rng
        self.n_shapes = len(cohort_shapes)

    def _pick_indices(self):
        """Draw a fresh random subset of cohort indices."""
        n = self.n_shapes
        if self.batch_size >= n:
            return torch.arange(n)
        return torch.randperm(n, generator=self.rng)[:self.batch_size]

    def __iter__(self):
        while True:
            indices = self._pick_indices()
            selected_shapes = [self.cohort_shapes[i] for i in indices]

            # Stack points [B, N, 3] and create shape_ids [B]
            points = torch.stack([s.points.squeeze(0) for s in selected_shapes], dim=0)  # [B, N, 3]
            shape_ids = torch.arange(len(selected_shapes), dtype=torch.long)  # [B] = [0, 1, 2, ...]

            # Optionally stack weights
            weights = None
            if selected_shapes[0].weights is not None:
                weights = torch.stack([s.weights.squeeze(0) for s in selected_shapes], dim=0)

            # Collect per-shape faces (one per shape in batch)
            sample_faces = [s.faces if hasattr(s, 'faces') else None for s in selected_shapes]

            # Create template batch (canonical reference) with single faces object
            template_batch = CohortBatch(
                points=self.template.points,  # [1, M, 3]
                shape_ids=torch.tensor([0], dtype=torch.long),  # [1]
                weights=self.template.weights,
                faces=[self.template.faces if hasattr(self.template, 'faces') else None]
            )

            sample_batch = CohortBatch(points=points, shape_ids=shape_ids, weights=weights, faces=sample_faces)
            yield template_batch, sample_batch