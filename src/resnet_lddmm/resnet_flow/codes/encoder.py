"""Encoder-based codes via GroupEncoder (STEPS T31 T32).

All heavy imports (e3nn, torch_geometric, GroupEncoder) live in this module.
The lazy Registry guarantees that configs using auto_decoder work even without
these dependencies installed.
"""

import torch
from src.resnet_lddmm.codes.base import ShapeCode


class EncoderCodes(ShapeCode):
    """Amortised codes via equivariant graph encoding.

    Builds a graph from point cloud, passes it through GroupEncoder,
    and extracts latent codes. The encoder output also carries pose
    information (rotation/translation) for optional pre-alignment (T32).

    All heavy imports are deferred until instantiation: when this class
    is imported, torch_geometric, e3nn, and GroupEncoder are not loaded.
    """

    def __init__(self, graph_builder, encoder, n_z=None):
        """Initialize encoder-based code source.

        Args:
            graph_builder: GraphBuilder instance (from Registry.create)
            encoder: GroupEncoder instance (from Registry.create)
            n_z: latent dimension (optional, inferred from encoder if not provided)

        Determinism and SE(3) invariance are the ENCODER's business, configured by
        its supernode_samples/supernode_seed members -- nothing is overridden here.
        """
        super().__init__()
        self.graph_builder = graph_builder
        self.encoder = encoder
        self.n_z = n_z or encoder.latent_dim
        self._last = None  # Stores last EncoderOutput for pose access (T32)
        self._last_graph = None  # Stores last graph for logging/visualization
        self._last_supergraph = None  # Stores last supergraph for logging

    def forward(self, batch):
        """Extract codes via graph encoding.

        Args:
            batch: CohortBatch with .points [B, N, 3], .shape_ids, optional .weights/.faces

        Returns:
            [B, n_z] codes sampled from encoder's latent distribution
        """
        points = batch.points  # [B, N, 3]
        B = points.shape[0]
        N = points.shape[1]

        # Create 2D boolean mask for graph builder (all valid nodes)
        mask = torch.ones((B, N), dtype=torch.bool, device=points.device)

        # Get RNG for graph building
        rng = self._get_rng()

        # Extract weights and normals if available
        weights = None
        if hasattr(batch, 'weights') and batch.weights is not None:
            weights = batch.weights

        normals = None
        if hasattr(batch, 'normals') and batch.normals is not None:
            normals = batch.normals

        # Build graph (vertices should be [B, N, 3], mask [B, N])
        graph, supergraph = self.graph_builder.build(
            points, mask, rng,
            areas=weights, normals=normals
        )

        # Cache graph for logging/visualization
        self._last_graph = graph
        self._last_supergraph = supergraph

        # Move graph to same device as points
        graph = graph.to(points.device)
        if supergraph is not None:
            supergraph = supergraph.to(points.device)

        # Forward through encoder: returns EncoderOutput with latent (or mu) and pose
        encoder_out = self.encoder(graph, supergraph)
        self._last = encoder_out

        # Sample from latent (deterministic=True uses mu if VAE, else latent)
        z = encoder_out.sample(deterministic=True)  # [B, n_z]

        return z

    def penalty(self):
        """No regularisation for encoder codes (unlike AutoDecoderCodes embedding table).

        Returns:
            None
        """
        return None

    def _get_rng(self):
        """Get PRNG for graph building.

        Returns:
            None to use default PyTorch RNG
        """
        # Return None to use default RNG; can be seeded via torch.manual_seed if needed
        return None

    def train(self):
        """Set encoder to training mode."""
        super().train()
        self.encoder.train()

    def eval(self):
        """Set encoder to evaluation mode."""
        super().eval()
        self.encoder.eval()

    def get_pose(self):
        """Get encoder's last predicted pose (rotation, translation), in PIPELINE convention.

        The rotation is TRANSPOSED relative to what GroupEncoder returns, and that
        transpose is load-bearing rather than cosmetic.

        GroupEncoder builds its rotation by Gram-Schmidt from type-1 (vector)
        features, so it is equivariant as ``R(x·Q) = Qᵀ·R(x)`` — identically, for
        every weight setting, since this is a property of the e3nn construction and
        not something training selects. But poses are applied downstream as
        ``points @ R`` (mapping_error._apply_encoder_pose, matching SE3_transform),
        which needs the opposite handedness: ``P(x·Q) = P(x)·Q``.

        Those two cannot be reconciled by learning. Requiring both gives
        ``Qᵀ A = A Q`` for all Q, i.e. ``Qᵀ = A Q A⁻¹`` — inversion is an
        ANTI-automorphism while conjugation is an automorphism, and on a non-abelian
        group such as SO(3) no such A exists. Left as-is, the pose objective is not
        merely hard to optimise, it is unsatisfiable, and the pose head sits at
        chance forever.

        Transposing fixes it exactly: ``P := Rᵀ`` gives
        ``P(x·Q) = (Qᵀ R(x))ᵀ = R(x)ᵀ Q = P(x)·Q``.

        A side effect worth having: the returned rotation is now directly comparable
        with the element an augmentation drew (SE3_transform uses the same ``x @ R``
        convention), which is what makes pose supervision meaningful.

        Returns:
            (rotation, translation) tuple where each is [B, ...] or None if no last output
        """
        if self._last is None:
            return None, None
        rotation = self._last.rotation
        if rotation is not None:
            rotation = rotation.transpose(-2, -1)
        return rotation, self._last.translation

    def with_pose_transform(self, base_transform):
        """Create a pose-aware FrameTransform by folding encoder pose into base transform.

        Args:
            base_transform: FrameTransform with center/scale

        Returns:
            New FrameTransform with encoder's rotation/translation folded in, or base_transform if no pose
        """
        rotation, translation = self.get_pose()   # one source of truth for the convention
        if rotation is None and translation is None:
            return base_transform

        from src.resnet_lddmm.io import FrameTransform
        return FrameTransform(
            center=base_transform.center,
            scale=base_transform.scale,
            rotation=rotation,
            translation=translation
        )
