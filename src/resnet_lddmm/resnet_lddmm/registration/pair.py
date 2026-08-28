"""Pair registration stepper for Milestone A (per-pair training)."""

import torch
from src.learning.losses.composer import LossComposer
from src.resnet_lddmm.losses.terms import LossContext
from src.resnet_lddmm.losses import BidirectionalMappingError
from src.resnet_lddmm.io import Shape


class PairRegistration:
    """One-to-one shape registration with optional code regularisation.

    Uses a mapping error strategy (unidirectional or bidirectional) to compute
    data and kinetic terms. Optionally includes code regularization and isometry loss.
    Implements both the training loop (train_step) and the four-method protocol
    (state_dict, load_state_dict, train, eval) for reused callbacks.
    """

    def __init__(self, flow, code_source, data_term, mapping_error, composer, optimizer, iso_loss=None, augmentation=None, use_encoder_pose=False, loss_terms=None):
        """Initialize the registration stepper.

        Args:
            flow: NeuralODEFlow instance
            code_source: ShapeCode instance (e.g., NoCode or AutoDecoderCodes)
            data_term: DataTerm instance (e.g., CDData or L2Data)
            mapping_error: MappingError strategy (UnidirectionalMappingError or BidirectionalMappingError)
            composer: LossComposer instance
            optimizer: torch optimizer instance
            iso_loss: IsometryLoss instance (optional, created by runner if enabled)
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
            from src.resnet_lddmm.augmentation.none import NoAugmentation
            self.augmentation = NoAugmentation()
        self.is_bidirectional = isinstance(mapping_error, BidirectionalMappingError)
        self.backward_traj = None  # Stored when bidirectional
        self.augmented_sample = None  # Store augmented sample for logger access

    def _values(self, template, sample):
        """Compute trajectory and per-term loss values.

        Args:
            template: batch-like with .points [B,N,3] (canonical reference, where flow starts)
            sample: batch-like with .points [B,M,3] (input to be augmented and encoded)

        Returns:
            (fwd_traj, values_dict) where values_dict has keys for data, kinetic, code_reg, isometry
        """
        # Set template on field if it's a contextual field
        if hasattr(self.flow.field, 'set_template'):
            self.flow.field.set_template(template.points)

        # Apply augmentation to sample points (random SO(3) or SE(3) transformation)
        augmented_points = self.augmentation(sample.points)

        # Create augmented batch with transformed points, preserving other fields
        augmented_sample = Shape(
            points=augmented_points,
            faces=sample.faces,
            weights=sample.weights,
            normals=getattr(sample, 'normals', None),
        )

        # Store augmented sample for logger access
        self.augmented_sample = augmented_sample

        code = self.code_source(augmented_sample)

        # Extract encoder pose if enabled
        encoder_pose = None
        if self.use_encoder_pose and hasattr(self.code_source, 'get_pose'):
            encoder_pose = self.code_source.get_pose()

        # Use mapping error strategy: flow deforms template to match augmented sample
        # template is the canonical reference (fixed), sample is what we compare against
        data, kinetic = self.mapping_error(self.flow, self.data_term, template, augmented_sample, code, encoder_pose=encoder_pose)

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
                sample=augmented_sample,
            )
            values.update({name: term(ctx) for name, term in self.loss_terms.items()})

        return fwd_traj, values

    def train_step(self, template, sample):
        """One gradient step: forward, loss, backward, optimizer step.

        Args:
            template: template (canonical reference)
            sample: sample shape (input to encode and augment)

        Returns:
            (trajectory, loss_float, breakdown_dict)
        """
        self.optimizer.zero_grad()
        traj, values = self._values(template, sample)
        loss, breakdown = self.composer.compute(values)
        loss.backward()
        self.optimizer.step()
        return traj, loss.item(), breakdown

    @torch.no_grad()
    def eval_step(self, template, sample):
        """Evaluate loss without gradient accumulation.

        Args:
            template: template (canonical reference)
            sample: sample shape (input to encode and augment)

        Returns:
            (trajectory, loss_float, breakdown_dict)
        """
        traj, values = self._values(template, sample)
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
