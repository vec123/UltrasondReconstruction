"""Cohort registration stepper for Milestone B (multi-shape training). STEPS T27 T29."""

import torch
import torch.nn as nn
from src.resnet_lddmm.learning.losses.composer import LossComposer
from src.resnet_lddmm.resnet_flow.losses.terms import LossContext
from src.resnet_lddmm.resnet_flow.losses import BidirectionalMappingError


class CohortRegistration:
    """Multi-shape registration with shared flow Θ and per-shape codes Z.

    Registers a cohort of shapes to a shared template using a learned flow
    and per-shape latent codes. Uses bidirectional mapping error and two
    optimizer parameter groups: flow Θ with weight decay, codes Z without
    (since code regularization flows through the composer for visibility).
    Implements the four-method protocol (state_dict, load_state_dict, train, eval).
    """

    def __init__(self, flow, code_source, data_term, mapping_error, composer, optimizer, iso_loss=None, augmentation=None, use_encoder_pose=False, loss_terms=None):
        """Initialize the cohort registration stepper.

        Args:
            flow: NeuralODEFlow instance
            code_source: ShapeCode instance (must be AutoDecoderCodes for T27)
            data_term: DataTerm instance
            mapping_error: MappingError strategy (BidirectionalMappingError)
            composer: LossComposer instance
            optimizer: torch optimizer with param groups [flow_params, code_params]
            iso_loss: IsometryLoss instance (optional)
            augmentation: Augmentation instance (optional; defaults to NoAugmentation)
            use_encoder_pose: bool, whether to extract and apply encoder pose from code_source
            loss_terms: dict of {name: term module} from cfg.loss.terms (optional)
        """
        self.flow = flow
        self.code_source = code_source
        self.data_term = data_term
        self.mapping_error = mapping_error
        self.composer = composer
        self.optimizer = optimizer
        self.iso_loss = iso_loss
        self.augmentation = augmentation
        self.use_encoder_pose = use_encoder_pose
        self.loss_terms = loss_terms or {}
        if self.augmentation is None:
            from src.resnet_lddmm.resnet_flow.augmentation.none import NoAugmentation
            self.augmentation = NoAugmentation()
        self.is_bidirectional = isinstance(mapping_error, BidirectionalMappingError)
        self.backward_traj = None
        self.augmented_sample_batch = None  # Store augmented sample batch for logger access

    def _values(self, template, sample_batch):
        """Compute trajectory and per-term loss values.

        Args:
            template: batch-like with .points [B,N,3] (canonical reference)
            sample_batch: batch-like with .points [B,M,3] and .shape_ids [B] (input to augment and encode)

        Returns:
            (fwd_traj, values_dict) where values_dict has keys for data, kinetic, code_reg, isometry
        """
        # Apply augmentation to sample points (random SO(3) or SE(3) transformation)
        augmented_points = self.augmentation(sample_batch.points)

        # Create augmented batch with transformed points, preserving other fields
        augmented_sample_batch = type(sample_batch)(
            points=augmented_points,
            shape_ids=sample_batch.shape_ids,
            weights=sample_batch.weights,
            faces=sample_batch.faces,
        )

        # Store augmented sample batch for logger access
        self.augmented_sample_batch = augmented_sample_batch

        code = self.code_source(augmented_sample_batch)

        # Extract encoder pose if enabled
        encoder_pose = None
        if self.use_encoder_pose and hasattr(self.code_source, 'get_pose'):
            encoder_pose = self.code_source.get_pose()

        # Use mapping error strategy: flow deforms template to match augmented samples
        data, kinetic = self.mapping_error(self.flow, self.data_term, template, augmented_sample_batch, code, encoder_pose=encoder_pose)

        # Use trajectory computed by mapping_error (avoids double computation with subsampling)
        fwd_traj = self.mapping_error.last_fwd_traj

        # For bidirectional mode, use backward trajectory from mapping_error
        if self.is_bidirectional:
            self.backward_traj = self.mapping_error.last_bwd_traj
        else:
            self.backward_traj = None

        values = {
            "data": data,
            "kinetic": kinetic,
            "code_reg": self.code_source.penalty(),
        }

        # Add isometry loss if enabled
        if self.iso_loss is not None:
            # The code goes with it: a conditioned field cannot be evaluated
            # without one, and it must be the code the trajectory was flowed with.
            values["isometry"] = self.iso_loss(fwd_traj, self.flow.field, code)

        # Configured extra terms (loss.terms). Each returns a scalar or None; the
        # composer skips None, so mode-dependent terms need no branch here. Adding
        # a term is a registry line plus config -- this block never changes.
        if self.loss_terms:
            ctx = LossContext(
                flow=self.flow,
                code=code,
                template_points=self.mapping_error.last_template_points,
                sample_points=self.mapping_error.last_sample_points,
                fwd_traj=fwd_traj,
                bwd_traj=self.backward_traj,
                pred=None,
                encoder_pose=self.mapping_error.last_effective_pose,
                augmentation_pose=self.augmentation.last_element(),
                data_term=self.data_term,
                code_source=self.code_source,
                template=template,
                sample=augmented_sample_batch,
            )
            values.update({name: term(ctx) for name, term in self.loss_terms.items()})

        return fwd_traj, values

    def train_step(self, template, sample_batch, debug_gradients=False):
        """One gradient step: forward, loss, backward, optimizer step.

        Args:
            template: template (canonical reference)
            sample_batch: sample shape batch with shape_ids (input to augment and encode)
            debug_gradients: if True, print detailed gradient statistics

        Returns:
            (trajectory, loss_float, breakdown_dict)
        """
        self.optimizer.zero_grad()
        traj, values = self._values(template, sample_batch)
        loss, breakdown = self.composer.compute(values)

        loss.backward()

        if debug_gradients:
            self._log_gradient_stats()

        self.optimizer.step()
        return traj, loss.item(), breakdown

    def _log_gradient_stats(self):
        """Log detailed gradient statistics for debugging."""
        print("\n" + "="*60)
        print("GRADIENT DEBUG STATS")
        print("="*60)

        # Encoder gradients
        print("\n[ENCODER]")
        encoder_stats = self._compute_param_stats(self.code_source.encoder.parameters())
        self._print_stats("encoder", encoder_stats)

        # Flow gradients
        print("\n[FLOW]")
        flow_stats = self._compute_param_stats(self.flow.parameters())
        self._print_stats("flow", flow_stats)

        # Code source (embeddings) gradients
        print("\n[CODE SOURCE]")
        code_stats = self._compute_param_stats(self.code_source.parameters())
        self._print_stats("code_source", code_stats)

        print("="*60 + "\n")

    def _compute_param_stats(self, parameters):
        """Compute gradient statistics for a parameter group."""
        grad_norms = []
        grad_means = []
        param_count = 0
        has_nan = False
        has_inf = False
        no_grad_count = 0

        for param in parameters:
            param_count += 1
            if param.grad is None:
                no_grad_count += 1
            else:
                grad = param.grad.data
                if torch.isnan(grad).any():
                    has_nan = True
                if torch.isinf(grad).any():
                    has_inf = True
                grad_norms.append(grad.norm().item())
                grad_means.append(grad.mean().item())

        return {
            'grad_norms': grad_norms,
            'grad_means': grad_means,
            'param_count': param_count,
            'no_grad_count': no_grad_count,
            'has_nan': has_nan,
            'has_inf': has_inf,
        }

    def _print_stats(self, name, stats):
        """Print gradient statistics."""
        if stats['no_grad_count'] == stats['param_count']:
            print(f"  [{name}] ❌ NO GRADIENTS ({stats['param_count']} params)")
            return

        norms = stats['grad_norms']
        means = stats['grad_means']

        if norms:
            print(f"  [{name}] ✓ {len(norms)}/{stats['param_count']} params have gradients")
            print(f"    norm:  min={min(norms):.2e}, max={max(norms):.2e}, mean={sum(norms)/len(norms):.2e}")
            print(f"    mean:  min={min(means):.2e}, max={max(means):.2e}, mean={sum(means)/len(means):.2e}")
            if stats['has_nan']:
                print(f"    ⚠️  NaN detected in {name} gradients!")
            if stats['has_inf']:
                print(f"    ⚠️  Inf detected in {name} gradients!")

    @torch.no_grad()
    def eval_step(self, template, sample_batch):
        """Evaluate loss without gradient accumulation.

        Args:
            template: template (canonical reference)
            sample_batch: sample shape batch with shape_ids (input to augment and encode)

        Returns:
            (trajectory, loss_float, breakdown_dict)
        """
        traj, values = self._values(template, sample_batch)
        loss, breakdown = self.composer.compute(values)
        return traj, loss.item(), breakdown

    def state_dict(self):
        """Serialize flow, codes, and optimizer state.

        Returns:
            Dict with keys "flow", "codes", "optimizer"
        """
        return {
            "flow": self.flow.state_dict(),
            "codes": self.code_source.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

    def load_state_dict(self, state):
        """Restore whichever of flow, codes and optimizer ``state`` carries.

        Absent keys are skipped rather than an error, which is what makes a
        PARTIAL restore possible: a warm start hands in ``{"flow": ...}`` alone
        so the new run keeps its own fresh optimizer. WHICH parts to restore is
        the caller's policy (see config.InitCfg.parts) -- this class only owns
        what its state consists of, and validating the parts here would put that
        policy in two places.

        Args:
            state: dict with any subset of "flow", "codes", "optimizer"
        """
        if "flow" in state:
            self.flow.load_state_dict(state["flow"])
        if "codes" in state:
            self.code_source.load_state_dict(state["codes"])
        if "optimizer" in state:
            self.optimizer.load_state_dict(state["optimizer"])

    def train(self):
        """Set flow and code source to training mode."""
        self.flow.train()
        self.code_source.train()

    def eval(self):
        """Set flow and code source to evaluation mode."""
        self.flow.eval()
        self.code_source.eval()

    def infer(self, new_shape, target, steps_adam, steps_lbfgs=0):
        """Optimize code for a new shape with frozen flow (STEPS T29, T32 semi-amortised init).

        Args:
            new_shape: Shape object with .points [1, N, 3] and optional weights/faces
            target: target template Shape with .points [1, M, 3]
            steps_adam: number of Adam optimization steps
            steps_lbfgs: number of L-BFGS steps after Adam (optional, default 0)

        Returns:
            optimized_code: Tensor [1, n_z], the inferred code for the new shape

        This implements the DeepSDF auto-decoder test-time protocol: freeze Θ,
        optimize only a fresh z. If code_source is EncoderCodes, initializes z from
        encoder prediction (semi-amortised); otherwise initializes from N(0, 2/N_z·I).
        Asserts flow is bit-identical before/after (frozen-decoder guarantee).
        """
        from src.resnet_lddmm.learning.loader.loaders import CohortBatch

        # Snapshot flow state to verify it stays frozen
        flow_state_before = {
            name: param.data.clone() for name, param in self.flow.named_parameters()
        }

        # Freeze flow: disable gradients
        for param in self.flow.parameters():
            param.requires_grad = False

        # Initialize code: try encoder first (T32 semi-amortised), fall back to random
        n_z = self.code_source.n_z
        device = next(self.flow.parameters()).device

        # Create batch for new shape (needed for encoder or baseline)
        source_batch = CohortBatch(
            points=new_shape.points.to(device),
            shape_ids=torch.tensor([0], dtype=torch.long, device=device),
            weights=new_shape.weights.to(device) if hasattr(new_shape, "weights") and new_shape.weights is not None else None,
            faces=[new_shape.faces] if hasattr(new_shape, "faces") and new_shape.faces is not None else [None],
        )

        # Check if code_source is EncoderCodes and can provide amortised init
        try:
            from src.resnet_lddmm.resnet_flow.codes.encoder import EncoderCodes
            if isinstance(self.code_source, EncoderCodes):
                # Use encoder to get initial code from new shape (semi-amortised)
                z_init = self.code_source(source_batch)  # [1, n_z]
            else:
                # Fall back to random initialization
                z_init = torch.randn(1, n_z, device=device) * (2.0 / n_z) ** 0.5
        except (ImportError, AttributeError):
            # EncoderCodes not available or different error; use random init
            z_init = torch.randn(1, n_z, device=device) * (2.0 / n_z) ** 0.5

        z_new = z_init.clone().detach()
        z_new.requires_grad = True

        # Create target batch (single point)
        target_batch = CohortBatch(
            points=target.points.to(device),
            shape_ids=torch.tensor([0], dtype=torch.long, device=device),
            weights=target.weights.to(device) if hasattr(target, "weights") and target.weights is not None else None,
            faces=[target.faces] if hasattr(target, "faces") and target.faces is not None else [None],
        )

        # Use optimizer learning rate (shared across param groups)
        lr = self.optimizer.param_groups[0]["lr"]
        optimizer = torch.optim.Adam([z_new], lr=lr)

        # Set to eval mode for inference (no dropout, stable BN)
        self.flow.eval()
        self.code_source.eval()

        # Adam optimization loop
        for _ in range(steps_adam):
            optimizer.zero_grad()

            # Forward pass: compute loss with frozen flow, variable z
            with torch.enable_grad():
                # Forward trajectory
                fwd_traj = self.flow(source_batch.points, z_new)

                # Backward trajectory (single template point broadcast)
                tgt_broadcast = target_batch.points.expand(1, -1, -1)
                bwd_traj = self.flow.inverse(tgt_broadcast, z_new)

                # Data term: bidirectional (source→target and target→source)
                data_fwd = self.data_term(fwd_traj.end, target_batch.points)
                data_bwd = self.data_term(bwd_traj.end, source_batch.points)
                data = data_fwd + data_bwd

                # Kinetic energy: sum both directions
                kinetic = fwd_traj.kinetic_energy() + bwd_traj.kinetic_energy()

                # Code regularization: direct L2 on z (not through embedding table)
                code_reg = (z_new ** 2).mean()

                # Compose loss with same weights as training
                values = {
                    "data": data,
                    "kinetic": kinetic,
                    "code_reg": code_reg,
                }
                loss, _ = self.composer.compute(values)

            loss.backward()
            optimizer.step()

        # Optional L-BFGS refinement (max_iter controls steps)
        if steps_lbfgs > 0:
            optimizer_lbfgs = torch.optim.LBFGS([z_new], max_iter=steps_lbfgs)

            def closure():
                optimizer_lbfgs.zero_grad()
                fwd_traj = self.flow(source_batch.points, z_new)
                tgt_broadcast = target_batch.points.expand(1, -1, -1)
                bwd_traj = self.flow.inverse(tgt_broadcast, z_new)
                data_fwd = self.data_term(fwd_traj.end, target_batch.points)
                data_bwd = self.data_term(bwd_traj.end, source_batch.points)
                data = data_fwd + data_bwd
                kinetic = fwd_traj.kinetic_energy() + bwd_traj.kinetic_energy()
                code_reg = (z_new ** 2).mean()
                values = {"data": data, "kinetic": kinetic, "code_reg": code_reg}
                loss, _ = self.composer.compute(values)
                loss.backward()
                return loss

            optimizer_lbfgs.step(closure)

        # Verify flow is frozen: bit-identical before/after
        for name, param in self.flow.named_parameters():
            assert torch.allclose(
                param.data, flow_state_before[name], atol=1e-7
            ), f"Flow param '{name}' was modified during infer()"

        # Re-enable gradients on flow for future training
        for param in self.flow.parameters():
            param.requires_grad = True

        return z_new.detach()
