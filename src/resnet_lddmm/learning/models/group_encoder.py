from dataclasses import dataclass, replace
from typing import Optional

import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool, global_add_pool
from torch_geometric.utils import softmax as scatter_softmax
from e3nn import o3
from src.learning.layers.equivariant.Self_Spatial_layer import EquiLayer
from src.learning.modules.equivariant.interaction import (
    BipartiteSpatialConvolution, MonteCarloBipartiteSpatialConvolution)

from src.learning.modules.equivariant.transformer import build_equivariant_transformer
from src.learning.modules.equivariant.irreps_utils import scalar_features, vector_features

from src.learning.registry import Registry


@dataclass
class _TokenSet:
    """One point set the encoder pools over -- features plus the geometry they live on.

    Two of these flow through ``forward``: the NODES (the full graph, after message
    passing) and the POOLED set the graph-level latent is read out from (the
    supernodes, or the nodes again when no supergraph is given). Bundling them is what
    keeps the two branches from becoming parallel ``*_nodes`` copies of every local.
    """

    feat: torch.Tensor                    # [n, irreps.dim]
    pos: torch.Tensor                     # [n, 3]
    batch: torch.Tensor                   # [n] -- which shape each token belongs to
    area: Optional[torch.Tensor] = None   # [n] surface measure; None unless area_pool

    @property
    def area_col(self):
        """``area`` as an [n, 1] column, ready to broadcast against per-token logits."""
        if self.area is None or self.area.dim() != 1:
            return self.area
        return self.area.reshape(-1, 1)


class GroupEncoder(nn.Module):
    def __init__(self, layers_cfg,
                 latent_dim: int = 5,
                 output_irreps: str = None,
                 readout: str = "mean",
                 readout_heads: int = 1,
                 supernode_sh_lmax: int = 4,
                 transformer_type: str = "se3",
                 transformer_cfg: dict = None,
                 area_pool: bool = True,
                 latent_mode: str = "gaussian",
                 pose_mode: str = "first_moment",
                 supernode_samples: int = None,
                 supernode_seed: int = 1,
                 verbose: bool = False):
        
        """``layers_cfg``: a non-empty list of per-layer dicts, one EquiLayer each,
        threaded in order (layer i's output feeds layer i+1's input):

            {"in_irreps": ..., "target_irreps": ..., "spatial_sh_lmax": ...,
             "interaction_sh_lmax": ...}   # interaction_sh_lmax optional, default 4

        ``spatial_sh_lmax`` -- every layer's caller
        must state it explicitly, no default, no silent inherit of an encoder-wide
        value

        ``latent_mode`` selects the LatentHead strategy from the registry:
        ``"gaussian"`` (VAE: mu/logvar) or ``"deterministic"`` (auto-encoder: a
        plain latent). Both emit [B, latent_dim], so nothing downstream branches.

        ``supernode_samples`` / ``supernode_seed`` govern how each supernode
        aggregates its neighbourhood, and together decide whether this encoder is
        deterministic and exactly SE(3)-invariant:

          None (default)     -- aggregate EVERY neighbour. Exact, repeatable, and
                                exactly invariant. ``supernode_seed`` is unused.
          int + seed=int     -- Monte-Carlo sample that many neighbours from a
                                pinned draw: repeatable call to call, but NOT
                                rotation-invariant, because the sampler keys edges
                                POSITIONALLY (interaction.py:274) and a radius graph
                                orders the same edge set differently once rotated.
          int + seed=None    -- fresh draw per call: sampling acts as a regulariser,
                                at the cost of repeatability AND exact invariance.

        ``pose_mode`` selects how the frame is built from the 1o vectors. Two
        independent things can go wrong, and the three modes address them separately:

          "first_moment"   sum_i w_i v_i, then Gram-Schmidt. The default, and weak on
                           BOTH counts -- see the two entries below.
          "second_moment"  eigenvectors of sum_i w_i v_i v_i^T. Fixes the POOLING: the
                           first moment nearly cancels over a closed surface, leaving
                           the frame built from the residual. Its axes come out exactly
                           orthonormal, so the orthogonalisation below is a no-op and
                           there is no "second_moment + polar" worth having; what remains
                           is eigh's 1/(lambda_i - lambda_j) gradient.
          "polar"          first-moment pooling, but ORTHOGONALISED symmetrically
                           (polar factor) instead of by Gram-Schmidt. Fixes the
                           ORTHOGONALISATION: Gram-Schmidt pins axis 1 to v1 exactly and
                           forces the entire correction into v2, so v1's error passes
                           straight into the frame. See ``_polar_frame``.

        All three emit [B, 3, 3], so nothing downstream branches.
        """

        super().__init__()
        self.latent_dim = latent_dim
        self.readout = readout
        self.area_pool = area_pool
        self.verbose = verbose

        if pose_mode not in ("first_moment", "second_moment", "polar"):
            raise ValueError(
                f"pose_mode must be 'first_moment', 'second_moment' or 'polar', "
                f"got {pose_mode!r}."
            )
        self.pose_mode = pose_mode
        self.supernode_samples = supernode_samples
        self.supernode_seed = supernode_seed

        if not layers_cfg:
            raise ValueError("GroupEncoder requires at least one entry in layers_cfg.")

        self.in_irreps_str = layers_cfg[0]["in_irreps"]
        self.intermediate_irreps_str = layers_cfg[-1]["target_irreps"]
        self.output_irreps_str = output_irreps or f"{latent_dim}x0e + 2x1o"

        # Each layer's output must match the next layer's input -- otherwise the
        # tensor-product chain inside EquiLayer fails with an opaque shape
        # mismatch deep in the stack instead of a clear message here.
        for i in range(len(layers_cfg) - 1):
            out_ir = o3.Irreps(layers_cfg[i]["target_irreps"])
            next_in_ir = o3.Irreps(layers_cfg[i + 1]["in_irreps"])
            if out_ir != next_in_ir:
                raise ValueError(
                    f"layers_cfg[{i}]['target_irreps'] ({layers_cfg[i]['target_irreps']!r}) "
                    f"must equal layers_cfg[{i + 1}]['in_irreps'] "
                    f"({layers_cfg[i + 1]['in_irreps']!r})."
                )

        self.layers = nn.ModuleList([
            EquiLayer(in_irreps=l["in_irreps"],
                        target_irreps=l["target_irreps"],
                        spatial_sh_lmax=l["spatial_sh_lmax"],
                        interaction_sh_lmax=l.get("interaction_sh_lmax", 4),
                        verbose=verbose)
            for l in layers_cfg
        ])

        # Optional supernode aggregation: an equivariant bipartite conv maps the
        # full-node features onto the supernodes (in_irreps == out_irreps, so the
        # scalar count is unchanged). When enabled, the readout below pools over
        # supernodes instead of nodes.
        # Attention: if supergraph is None these are dead weights 
        self.supernode_in_irreps = o3.Irreps(self.intermediate_irreps_str)   
        self.supernode_out_irreps = o3.Irreps(self.intermediate_irreps_str)   
        self.supernode_conv = MonteCarloBipartiteSpatialConvolution(
                in_irreps=self.supernode_in_irreps, target_irreps=self.supernode_out_irreps,
                sh_lmax=supernode_sh_lmax, verbose=verbose,
            )
                
        # Optional equivariant transformer refinement of the pooled features, applied
        # after supernode aggregation and before the scalars are filtered out. Selectable
        # backend ('se3' | 'equiformer'); in-place on out_irreps so nothing downstream
        # changes. Disable with transformer_type=None.
       
        self.equi_transformer_irreps = o3.Irreps(self.intermediate_irreps_str)    
        self.equi_transformer = build_equivariant_transformer(
            transformer_type, self.equi_transformer_irreps, transformer_cfg, verbose=verbose,
        )

       
        self.out_irreps = o3.Irreps(self.output_irreps_str)
        self.final_linear = o3.Linear(self.intermediate_irreps_str, self.output_irreps_str)
         
        # Latent head: owns the readout AND the distribution the
        # pooled scalars parameterize. Resolved through the lazy Registry, 
        # adding a third mode is a registration line, not an edit here.
        #
        # Constructed HERE, between final_linear and weight_net, and internally in
        # the order readout_pool -> mu_net -> var_net: 
        # nn.Linear draws from the global RNG at construction
        self.latent_mode = latent_mode
        self.latent_head = Registry.create(
            "latent_head", latent_mode,
            latent_dim=latent_dim, readout=readout, readout_heads=readout_heads,
        )

        # Two SEPARATE per-token attention gates, one per branch (see _latent / _pose).
        # They pool different token sets (supernodes vs nodes) under different
        # normalizations (area-weighted vs not), so a single shared nn.Linear would have
        # to satisfy both at once -- the optimal logits for one are roughly the other's
        # minus log(area). Both stay on the encoder rather than on the latent head,
        # which owns only the readout and the distribution.
        self.weight_net = nn.Linear(latent_dim, 1)
        # The pose gate additionally sees the 2 vector norms -- invariant, and the part
        # of the input that actually varies across nodes.
        self.vector_weight_net = nn.Linear(latent_dim + 2, 1)

        # NOT used by forward; kept so existing checkpoints still load.
        self.mu_bn = nn.BatchNorm1d(latent_dim)
        self.mu_ln = nn.LayerNorm(latent_dim)

        # Both readouts ("mean" and "attention") collapse to one token per shape
        # (see forward, below). Exposed so a factory can check encoder/decoder
        # compatibility on the constructed objects, multi-token not yet supported.
        self.n_tokens = 1


    def forward(self, graph, supergraph, monte_carlo_reg=None):
        """(graph, supergraph|None) -> EncoderOutput carrying the latent AND the pose.

        Two token sets, deliberately different ones: the graph-level LATENT is pooled
        from the supernodes -- a coarse summary is enough for a shape descriptor --
        while the POSE frame is pooled from the nodes, whose fine-grained local
        geometry is what the 1o vectors need. Without a supergraph the two coincide.
        """
        nodes = self._message_passing(graph)
        pooled = self._pool_set(nodes, supergraph, monte_carlo_reg)

        # Pass size=num_graphs to every pool below, so a non-contiguous batch vector
        # can't silently drop a shape.
        num_graphs = int(pooled.batch.max().item()) + 1

        # Without a supergraph both branches hold the SAME tokens (_pool_set returns the
        # node set itself), so refine and project once instead of twice.
        no_supergraph = pooled is nodes

        # The transformer refines the FULL irreps; final_linear then reduces them to the
        # encoding set (latent_dim scalars + 2 vectors). Keep the pooled branch FIRST:
        # autograd accumulates gradients in graph-construction order, so swapping these
        # two perturbs every shared parameter's gradient in the last float digits.
        pooled = self._transformer_refine(pooled)
        nodes = pooled if no_supergraph else nodes #self._transformer_refine(nodes)

        pooled_feat = self.final_linear(pooled.feat)
        node_feat = pooled_feat if no_supergraph else self.final_linear(nodes.feat)

        latent_out = self._latent(pooled_feat, pooled, num_graphs)
        rotation, translation, pose_aux = self._pose(node_feat, nodes, num_graphs)

        # Node features BEFORE final_linear: keeping the full irreps structure (1o
        # components included) is what makes them potentially usable as geometric descriptors
        aux = {**(latent_out.aux or {}), **pose_aux, 'node_features_full': nodes.feat}

        # Attach the pose to whatever latent fields the head produced, without this
        # method needing to know which kind of head it holds.
        return replace(latent_out, rotation=rotation, translation=translation, aux=aux)

    def _message_passing(self, graph):
        """Run the EquiLayer stack over the full graph -> the node-level token set."""
        x = graph.x
        node_normal = getattr(graph, 'normal', None)
        if node_normal is not None:
            x = torch.cat([x, node_normal], dim=-1)

        # Area weighting turns sums over a point set into surface integrals; off unless
        # area_pool AND an 'area' attribute are present -> falls back to uniform.
        # See Graph Kernel Operators for GNNs as MC-approx, of (surface) integrals.
        node_area = getattr(graph, 'area', None) if self.area_pool else None

        for i, layer in enumerate(self.layers):
            if self.verbose:
                print(f"---------Layer {i} with: "
                      f" x: {x.shape}, "
                      f" pos: {graph.pos.shape}, "
                      f" edge index: {graph.edge_index.shape}")

            x = layer(x, graph.pos, graph.edge_index, area=node_area)
            if self.verbose:
                print(f"Layer {i} output shape: {x.shape}")

        return _TokenSet(feat=x, pos=graph.pos, batch=graph.batch, area=node_area)

    def _supernode_sampling(self, monte_carlo_reg):
        """Resolve (num_samples, seed) for the supernode aggregation.

        The ENCODER owns this policy via supernode_samples/supernode_seed;
        ``monte_carlo_reg`` is a per-call override kept for existing callers.

        Args:
            monte_carlo_reg: None defers to the encoder's own settings; True forces
                a fresh sampled draw; False forces the exact, unsampled sum

        Returns:
            (num_samples, seed) for MonteCarloBipartiteSpatialConvolution.forward
        """
        if monte_carlo_reg is None:
            return self.supernode_samples, self.supernode_seed
        if monte_carlo_reg:
            return (self.supernode_samples or 30), None
        return None, None

    def _pool_set(self, nodes, supergraph, monte_carlo_reg):
        """The token set the graph-level latent is read out from.

        With a supergraph: an equivariant bipartite conv carries the node features onto
        the supernodes (in_irreps == out_irreps, so nothing downstream changes shape).
        Without one: the node set ITSELF is returned -- that identity is what lets
        ``forward`` share the refinement between the two branches.
        """
        if supergraph is None:
            return nodes

        super_pos, super_batch = supergraph.pos, supergraph.batch
        super_edge_index = supergraph.edge_index
        if super_pos is None or super_batch is None or super_edge_index is None:
            raise ValueError(
                "use_supernodes=True requires super_pos, super_batch and super_edge_index."
            )

        # Sampling policy comes from the encoder's own members (see __init__).
        # num_samples=None aggregates ALL neighbours, which is the only setting that
        # is exactly SE(3)-invariant: pinning the seed is not enough, since the
        # sampler keys edges positionally and a radius graph reorders the same edge
        # set once the input is rotated.
        num_samples, seed = self._supernode_sampling(monte_carlo_reg)
        feat = self.supernode_conv(nodes.feat, nodes.pos, super_pos, super_edge_index,
                                   area_src=nodes.area,
                                   num_samples=num_samples,
                                   seed=seed)                    # [S, out_irreps.dim]
        super_area = getattr(supergraph, 'area', None) if self.area_pool else None
        return _TokenSet(feat=feat, pos=super_pos, batch=super_batch, area=super_area)

    def _transformer_refine(self, tokens):
        """Equivariant transformer refinement over a token set.

        In-place on the irreps, applied BEFORE the invariant scalars are filtered out.
        No-op when the encoder was built with ``transformer_type=None``.
        """
        if self.equi_transformer is None:
            return tokens
        return replace(tokens,
                       feat=self.equi_transformer(tokens.feat, tokens.pos, tokens.batch))

    def _attention_weights(self, feats, batch, net, area=None):
        """Per-token pooling weights [n, 1], from ``net`` applied to invariant features.

        ``net`` is passed in because the latent and the pose own SEPARATE gates (see
        __init__); only the normalization below is shared. ``feats`` must be invariant
        (0e) -- scalars, or scalars plus vector norms -- so the weights stay invariant
        and anything equivariant they scale stays equivariant.

        The softmax is taken WITHIN each shape (grouped by ``batch``), so the weights
        sum to 1 per shape and don't leak across shapes. Use with global_add_pool so
        the softmax is the single normalization -- a following global_mean_pool would
        divide again and crush the per-shape result toward 0.

        Area-weighted when areas are given: softmax(logit + log a) = a·e^logit / Σ a·e^logit,
        so a denser sampling no longer over-counts (a per-shape scale of a cancels).
        """
        logits = net(feats)
        if area is not None:
            logits = logits + torch.log(area.clamp_min(1e-12))
        return scatter_softmax(logits, batch)

    def _latent(self, feat, tokens, num_graphs):
        """Graph-level latent, pooled from the token set's INVARIANT scalars.

        The readout AND the distribution live in the latent head, which returns an
        EncoderOutput carrying only the latent fields -- gaussian -> mu/logvar,
        deterministic -> latent -- and ``forward`` adds the pose to it.
        """
        scalars = scalar_features(feat, self.out_irreps)              # [n_pool, #0e]
        assert scalars.shape[1] == self.latent_dim, (
            f"output_irreps must carry exactly latent_dim ({self.latent_dim}) scalars, "
            f"got {scalars.shape[1]}"
        )
        weights = self._attention_weights(scalars, tokens.batch, self.weight_net,
                                          area=tokens.area_col)
        return self.latent_head(scalars, weights, tokens.batch, num_graphs)

    def _pose(self, feat, tokens, num_graphs):
        """Equivariant pose from a token set -> (rotation [B,3,3], translation [B,3], aux).

        ``aux`` carries the raw pose vectors and the per-token scalars for logging and
        for descriptor-based matching downstream.
        """
        scalars = scalar_features(feat, self.out_irreps)              # [n, #0e]
        vectors = vector_features(feat, self.out_irreps, '1o')        # [n, n_vec, 3]
        n_vec = vectors.shape[1]
        assert n_vec == 2, f"pose needs exactly 2 1o vectors in output_irreps, got {n_vec}"

        # The pose gets its OWN gate (vector_weight_net)
        #
        # Its extra input is the per-channel vector NORMS. They are rotation invariants
        # (0e), so the gate stays invariant and the pooled vector stays equivariant --
        # and unlike the scalars they actually vary across nodes, which is what lets the
        # gate be selective at all.
        #
        # No area term, deliberately: area weighting makes that quadrature MORE faithful
        # and drives the residual further toward zero.
        gate_in = torch.cat([scalars, vectors.norm(dim=-1)], dim=-1)  # [n, #0e + n_vec]
        weights = self._attention_weights(gate_in, tokens.batch,
                                          self.vector_weight_net)     # [n, 1]

        if self.pose_mode == "second_moment":
            v1, v2 = self._second_moment_axes(vectors, weights, tokens.batch, num_graphs)
        else:
            # Weighted sum over tokens (weights sum to 1 per shape) -> [B, n_vec, 3].
            vec_weighted = (weights.unsqueeze(-1) * vectors).reshape(vectors.shape[0], -1)
            vec_graph = global_add_pool(vec_weighted, tokens.batch,
                                        size=num_graphs).reshape(-1, n_vec, 3)
            v1, v2 = vec_graph[:, 0, :], vec_graph[:, 1, :]

        # Translation: the token centroid. Under SO(3)-only training it carries no
        # gradient, which is expected -- only the rotation is used there.
        transl = global_mean_pool(tokens.pos, tokens.batch, size=num_graphs)

        aux = {'v1': v1, 'v2': v2, 'node_features_scalars': scalars}
        # Orthogonalisation is a separate choice from pooling. second_moment already
        # returns orthonormal axes, so Gram-Schmidt is exact there and only the
        # first-moment path benefits from the symmetric alternative.
        rotation = (self._polar_frame(v1, v2) if self.pose_mode == "polar"
                    else self._Gram_Schmidt_frame(v1, v2))
        return rotation, transl, aux

    @staticmethod
    def _polar_frame(v1, v2, eps=1e-8):
        """Orthonormal frame from two vectors by SYMMETRIC (polar) orthogonalisation.

        Gram-Schmidt is asymmetric: it keeps ``v1`` exactly and pushes the whole
        correction onto ``v2``. An angular error in v1 therefore lands in the frame
        untouched, while the same error in v2 is partly repaired -- the frame is more
        accurate in one of its inputs than the other, for no reason the geometry
        justifies. The polar factor of ``[v1|v2|v1 x v2]`` splits the discrepancy
        evenly instead: the classical Lowdin symmetric orthogonalisation.

        With unit inputs at angle ``theta`` the Gram matrix is
        ``A^T A = [[1,c,0],[c,1,0],[0,0,1]]``, ``c = cos theta``, so
        ``R = A (A^T A)^{-1/2}`` leaves the third axis alone and rotates each of the
        first two by ``(90 - theta)/2`` -- half of what Gram-Schmidt applies to v2
        alone. That makes the answer available in CLOSED FORM, which is what is
        computed here.

        Not via ``torch.linalg.svd``, deliberately. The polar factor is U V^T, but at
        theta = 90 degrees A is already orthogonal, all three singular values are 1,
        and U and V are then individually arbitrary. Their product is still correct in
        exact arithmetic; in float32 it is noisy, and measurably so -- the SVD route
        amplified input perturbation ~27x MORE than Gram-Schmidt at 90 degrees, which
        is numerical noise rather than genuine sensitivity. The closed form has no such
        failure: it reduces to the identity map exactly where SVD is worst.

        Columns, matching ``_Gram_Schmidt_frame``, so the encoder's
        LEFT-equivariance convention (R -> QR under x -> Qx) is unchanged and the
        transpose in ``EncoderCodes.get_pose`` still applies -- ``cross(Qv1, Qv2) =
        Q cross(v1, v2)`` for det Q = 1, so every axis carries Q out front.

        This does NOT rescue exactly parallel inputs: the frame is genuinely undefined
        there and every method diverges. It changes how error is ALLOCATED, not
        whether the degeneracy exists.

        Args:
            v1: [B, 3] first pose vector
            v2: [B, 3] second pose vector
            eps: guards the normalisations

        Returns:
            [B, 3, 3] proper rotations (det = +1)
        """
        u1 = v1 / (v1.norm(dim=-1, keepdim=True) + eps)
        u2 = v2 / (v2.norm(dim=-1, keepdim=True) + eps)

        # In-plane orthonormal pair: the bisector of u1,u2 and the in-plane direction
        # perpendicular to it. u1 = b cos(t/2) - p sin(t/2), u2 = b cos(t/2) + p sin(t/2),
        # so b and p are orthogonal for any theta and the symmetric answer is the pair
        # at +-45 degrees about the bisector.
        b = u1 + u2
        p = u2 - u1
        b = b / (b.norm(dim=-1, keepdim=True) + eps)
        p = p / (p.norm(dim=-1, keepdim=True) + eps)

        inv_sqrt2 = 0.7071067811865476
        a1 = (b - p) * inv_sqrt2
        a2 = (b + p) * inv_sqrt2
        a3 = torch.cross(a1, a2, dim=-1)          # right-handed by construction

        return torch.stack([a1, a2, a3], dim=-1)  # [B, 3, 3], columns

    def _second_moment_axes(self, vectors, weights, batch, num_graphs):
        """Frame axes from the pooled SECOND moment M = sum_i w_i sum_c v_ic v_ic^T.

        Why a second moment: M is PSD, so nothing cancels. The first moment
        sum_i w_i v_i is a quadrature of an integral that very nearly vanishes over a
        closed surface (cf. int_S n dA = 0), which leaves the frame to be built out of
        the cancellation residual -- measured at ~1% of the per-node vector norms, and
        *smaller* the more faithfully the surface is integrated. A second moment has no
        such symmetry, so a uniform gate is no longer fatal.

        Equivariance: M -> R M R^T under x -> Rx, so its eigenvectors carry over as
        u -> ±Ru. ``eigh`` picks that SIGN arbitrarily, which would break equivariance
        outright, so it is fixed here by the third moment of the projections,
        sum_i w_i (u . v_i)^3: each (u . v_i) is invariant, and flipping u flips the
        statistic, so requiring it >= 0 is a canonical and equivariant choice. (It is
        the one thing that can still go ambiguous -- on a shape whose projections are
        symmetric the statistic vanishes, exactly the configurations that have no
        canonical frame to begin with.)

        Returns two ORTHONORMAL axes (the two leading eigenvectors), so the Gram-Schmidt
        downstream is well-conditioned and the near-zero-norm guard cannot trip. The
        cost is ``eigh``, whose gradient carries 1/(lambda_i - lambda_j) terms: nearly
        degenerate eigenvalues (a shape with no distinguished axis) make it stiff.
        """
        n = vectors.shape[0]
        # [n, n_vec, 3] -> per-node outer products summed over the vector channels.
        outer = (vectors.unsqueeze(-1) * vectors.unsqueeze(-2)).sum(1)     # [n, 3, 3]
        M = global_add_pool((weights.unsqueeze(-1) * outer).reshape(n, 9),
                            batch, size=num_graphs).reshape(-1, 3, 3)      # [B, 3, 3]
        M = 0.5 * (M + M.transpose(-1, -2))          # kill float asymmetry before eigh

        # eigh returns ascending eigenvalues; eigenvectors are the COLUMNS.
        _, evecs = torch.linalg.eigh(M)
        axes = [evecs[..., -1], evecs[..., -2]]                            # [B, 3] each

        signed = []
        for u in axes:
            proj = (vectors * u[batch].unsqueeze(1)).sum(-1)               # [n, n_vec]
            skew = global_add_pool((weights * proj.pow(3)).sum(-1, keepdim=True),
                                   batch, size=num_graphs)                 # [B, 1]
            # sign(0) would zero the axis, so fall back to +1 on an exact tie.
            signed.append(u * torch.where(skew < 0, -torch.ones_like(skew),
                                          torch.ones_like(skew)))
        return signed[0], signed[1]

    @staticmethod
    def _Gram_Schmidt_frame(v1, v2):
        """Compute rotation matrix from two vectors using Gram-Schmidt orthogonalization.

        Static, like ``_polar_frame``: the two are alternatives selected by
        ``pose_mode`` and neither reads any state off the encoder. The internal
        ``self._Gram_Schmidt_frame(...)`` call sites are unaffected.

        IMPORTANT: Encoder must learn non-zero pose vectors. If vectors remain near-zero,
        this will raise an error to alert the user to fix the encoder initialization.

        Args:
            v1: [B, 3] first vector
            v2: [B, 3] second vector

        Returns:
            [B, 3, 3] rotation matrix with orthonormal columns

        Raises:
            RuntimeError: if vectors remain near-zero after multiple steps (encoder not learning)
        """
        B = v1.shape[0]
        device = v1.device
        dtype = v1.dtype

        # Compute vector norms
        v1_norm = torch.norm(v1, dim=-1, keepdim=True)  # [B, 1]
        v2_norm = torch.norm(v2, dim=-1, keepdim=True)  # [B, 1]

        # Check if vectors are too small (encoder hasn't learned pose yet)
        # Threshold: 1e-4 is quite generous
        min_norm_threshold = 1e-4
        is_dead = (v1_norm < min_norm_threshold) | (v2_norm < min_norm_threshold)  # [B, 1]

        if is_dead.any():
            n_dead = is_dead.sum().item()
            print(f"[ROT_ERROR] {n_dead}/{B} samples have near-zero pose vectors!")
            print(f"  v1_norm: {v1_norm.view(-1).tolist()}")
            print(f"  v2_norm: {v2_norm.view(-1).tolist()}")
            raise RuntimeError(
                f"Encoder pose vectors are too small ({n_dead}/{B} samples). "
                f"The encoder is not learning the pose. This usually means:\n"
                f"  1. Encoder irreps must have 1o (pseudo-vector) components for pose\n"
                f"  2. Check that readout is configured correctly\n"
                f"  3. Try different initialization or learning rate\n"
                f"  4. Verify that pose is being used in the loss computation"
            )

        # Normalize with epsilon to avoid division by zero
        eps = 1e-8
        u = v1 / (v1_norm + eps)  # [B, 3], normalized first vector

        # Gram-Schmidt orthogonalization
        dot = torch.sum(u * v2, dim=-1, keepdim=True)  # [B, 1]
        w_raw = v2 - dot * u  # [B, 3]
        w_norm = torch.norm(w_raw, dim=-1, keepdim=True)  # [B, 1]
        w = w_raw / (w_norm + eps)  # [B, 3], normalized orthogonal vector

        # Third basis vector via cross product
        v3 = torch.cross(u, w, dim=-1)  # [B, 3]

        # Stack into rotation matrix
        R = torch.stack([u, w, v3], dim=-1)  # [B, 3, 3]

        return R