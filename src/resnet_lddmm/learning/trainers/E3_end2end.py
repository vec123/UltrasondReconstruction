import torch
import torch.optim as optim

from src.learning.losses.losses import (
    combined_surface_loss,
    contrastive_alignment_loss,
    frobenius_latent_loss,
)
from src.learning.losses.composer import LossComposer, LossTerm
from src.learning.models.encoder_output import EncoderOutput
from src.learning.callbacks.base import TrainingContext


def _resolve_device(device):
    if device is not None:
        return torch.device(device)
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TrainingStepper:
    """Runs a single optimization step:
    encode -> reparameterize -> decode -> loss.

    Encoder and decoder are injected, so swapping model variants 
    (equivariant vs not, supernodes vs full, different decoders) needs no change to this class.

    Contrastive objective ("same shape, different vertex sampling -> same encoding") is
    OPTIONAL and driven by the batch shape: hand ``train_step`` a SIX-tuple
    ``(graph_a, super_a, true_verts, mask, graph_b, super_b)`` and it encodes BOTH views,
    reconstructs each against the shared target, and makes an alignment loss between the
    two views' latents available as the ``contrastive`` term. A plain FOUR-tuple runs the
    ordinary single-view step, where that term is simply absent.

    Which terms actually count, and at what weight, is the ``composer``'s business
    -- this class computes every value it can and sums nothing itself.
    """

    def __init__(self, encoder, decoder, learning_rate=1e-3, composer=None,
                 device=None, verbose=False):
        self.device = _resolve_device(device)
        self.encoder = encoder.to(self.device)
        self.decoder = decoder.to(self.device)
        # Loss POLICY: which terms, their weights, and any
        # per-term kwargs all live on the composer. 
        # The default is reconstruction alon
        self.composer = composer if composer is not None else LossComposer([LossTerm("recon", 1.0)])
        self.verbose = verbose
        self.optimizer = optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=learning_rate,
        )

    def encode(self, graph, supergraph):
        """Adapt the current GroupEncoder output into the standard EncoderOutput."""
        out = self.encoder(graph, supergraph)
        assert isinstance(out, EncoderOutput)
        return out

    def _encode_decode(self, graph, super_graph, true_verts, padding_mask, deterministic):
        """One view: encode -> latent -> decode -> recon.

        Returns ``(pred, recon_loss, enc)``. The whole ``EncoderOutput`` comes back
        so the caller can ask it for whatever the configured terms need, without
        this method knowing which terms exist -- or which KIND of encoder ran.
        Does NOT touch the optimizer; the caller COMPOSES the total loss and decides
        whether to backprop."""
        graph = graph.to(self.device)
        super_graph = super_graph.to(self.device) if super_graph is not None else None
        true_verts = true_verts.to(self.device)
        padding_mask = padding_mask.to(self.device)

        enc = self.encode(graph, super_graph)
        # THE latent seam. VAE -> a reparameterized sample (or mu when deterministic);
        # auto-encoder -> its plain latent. One call, no branch on encoder kind, and
        # the only surviving reparameterize in the codebase (EncoderOutput.sample).
        latent = enc.sample(deterministic=deterministic)
        pred = self.decoder(latent)

        # The encoder's batch dim comes from graph.batch, which can desync from the
        # target if a shape lost all its nodes to dropout. Catch it loudly here instead
        # of letting torch.cdist broadcast (B_enc=1 vs B_target=N) into a wrong loss.
        if pred.shape[0] != true_verts.shape[0]:
            raise ValueError(
                f"batch mismatch: encoder produced {pred.shape[0]} shapes but target has "
                f"{true_verts.shape[0]}. A shape likely dropped all nodes during graph build."
            )

        recon_loss = combined_surface_loss(pred, true_verts, padding_mask)
        return pred, recon_loss, enc

    def _loss_values(self, recon, enc_a, enc_b=None):
        """Assemble ``{term name: Tensor | None}`` for the composer.

        Every term this trainer CAN produce is built here; the composer drops the
        ones a run didn't configure, and ``None`` means "not available in this
        mode" -- examples:
        ``kl`` without a posterior (auto-encoder), 
        ``contrastive`` outside a two-view step. 
        latent_mode becomes a pure config switch.

        The latents used by ``contrastive``/``frobenius`` are always
        DETERMINISTIC encodings: those terms describe where a shape lands in
        latent space, which shouldn't jitter with the reparameterization noise
        that only the decoder path wants.
        """
        z_a = enc_a.sample(deterministic=True)
        kl_a = enc_a.kl()                     # None whenever there is no posterior

        if enc_b is None:
            return {
                "recon": recon,
                "kl": kl_a,
                "contrastive": None,          # training-only, two-view-only
                "frobenius": frobenius_latent_loss(z_a),
            }

        z_b = enc_b.sample(deterministic=True)
        assert z_a.shape == z_b.shape, (
            f"two views produced different latent shapes {z_a.shape} vs {z_b.shape} "
            f"(a shape likely dropped out of one view); lower the dropout rate."
        )
        kl_b = enc_b.kl()
        return {
            "recon": recon,
            "kl": None if kl_a is None else 0.5 * (kl_a + kl_b),
            # var_weight and friends ride along on the term itself, not on self.
            "contrastive": contrastive_alignment_loss(
                z_a, z_b, **self.composer.kwargs_for("contrastive")),
            "frobenius": 0.5 * (frobenius_latent_loss(z_a) + frobenius_latent_loss(z_b)),
        }

    def train_step(self, *batch):
        """Single- or two-view step. Returns ``(pred, loss, breakdown)`` where
        ``loss`` is a python float and ``breakdown`` maps each CONTRIBUTING term
        name to its raw (unweighted) float value."""
        self.optimizer.zero_grad()

        if len(batch) == 6:
            graph_a, super_a, true_verts, mask, graph_b, super_b = batch
            pred, recon_a, enc_a = self._encode_decode(
                graph_a, super_a, true_verts, mask, deterministic=False)
            _,    recon_b, enc_b = self._encode_decode(
                graph_b, super_b, true_verts, mask, deterministic=False)
            values = self._loss_values(0.5 * (recon_a + recon_b), enc_a, enc_b)
        elif len(batch) == 4:
            graph, super_graph, true_verts, mask = batch
            pred, recon, enc = self._encode_decode(
                graph, super_graph, true_verts, mask, deterministic=False)
            values = self._loss_values(recon, enc)
        else:
            raise ValueError(
                f"train_step expected a 4-tuple (single view) or 6-tuple (two views), "
                f"got {len(batch)} elements."
            )

        # The composer keeps the non-finite guard that used to live in _total_loss:
        # a NaN would otherwise train silently to all-NaN weights and write NaN VTPs.
        loss, breakdown = self.composer.compute(values)
        loss.backward()
        self.optimizer.step()
        return pred, loss.item(), breakdown

    @torch.no_grad()
    def eval_step(self, graph, super_graph, true_verts, padding_mask):
        """Validation forward pass: single-view recon, no optimizer update.
        Returns ``(pred, loss, breakdown)`` -- the SAME shape as ``train_step``,
        which is what enables logging ``val/<term>`` against ``train/<term>``.

        ``deterministic=True`` is an INTENTIONAL behavior change:
        validation used to reparameterize with fresh random noise under no_grad,
        so identical weights on identical data scored differently run to run.
        Set ``encoder``/``decoder`` to eval mode around this (the orchestrator does)."""
        pred, recon, enc = self._encode_decode(
            graph, super_graph, true_verts, padding_mask, deterministic=True)
        loss, breakdown = self.composer.compute(self._loss_values(recon, enc))
        return pred, loss.item(), breakdown


class TrainingOrchestrator:
    """Drives the training loop and nothing else: fetch a batch, step, fire hooks.

    The dataloader yields ``(graph, super_graph, true_verts, padding_mask)`` batches,
    which are forwarded to ``stepper.train_step(*batch)``.

    Everything that happens *around* the step -- logging, plotting,
    checkpointing, VTP export, validation -- is a ``Callback``, each with
    its own cadence. 
    """

    def __init__(self, stepper, dataloader, callbacks=(), log_dir="logs"):
        self.stepper = stepper
        self.dataloader = dataloader
        self.callbacks = list(callbacks)
        self.log_dir = log_dir

    def run(self, num_steps, start_step=0):
        """Get a batch, take a step, fire the hooks. Nothing else lives here.

         Adding early stopping, an LR schedule, or a new
        artifact means adding a callback, not editing this loop.

        ``start_step`` offsets the step NUMBER, not the amount of work:
        ``num_steps`` steps are taken either way. A run resumed from a checkpoint
        passes the step it was written at, so its exports and checkpoints carry
        on numbering instead of overwriting the ones it resumed from -- and every
        callback cadence, which is ``step % every_n_steps``, stays aligned with
        the original run's.
        """
        ctx = TrainingContext(stepper=self.stepper, log_dir=self.log_dir,
                              num_steps=num_steps, start_step=start_step,
                              callbacks=self.callbacks)

        for callback in self.callbacks:
            callback.on_train_start(ctx)

        data_iter = iter(self.dataloader)
        for step in range(start_step, start_step + num_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.dataloader)
                batch = next(data_iter)

            pred, loss, breakdown = self.stepper.train_step(*batch)
            # The composer's per-term split plus the total, under one key each.
            # Splits are applied by MetricsRecorder, not here -- this loop doesn't
            # know what "train" and "val" mean.
            metrics = {"loss": loss, **breakdown}

            for callback in self.callbacks:
                callback.on_step_end(ctx, step, metrics, batch, pred)

            # Any callback may have asked to stop (see TrainingContext.stop).
            if ctx.should_stop:
                break

        for callback in self.callbacks:
            callback.on_train_end(ctx)

        return ctx