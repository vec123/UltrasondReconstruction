"""LossComposer (INSTRUCTIONS.md T8): weighted sum of named terms as one thing.

``config.config_fields.LossTermConfig`` 
is what a config layer converts into ``LossTerm``
 below before handing terms to this class.
"""

from typing import NamedTuple, Optional

import torch


class LossTerm(NamedTuple):
    """One term of a LossComposer's weighted sum..

    A NamedTuple (not a dataclass): LossComposer.compute unpacks terms as plain
    (name, weight, kwargs) tuples 
    -- a NamedTuple supports that AND attribute access (``term.name``).
    ``kwargs`` defaults to None, not {}, to avoid a NamedTuple mutable-default
    footgun (a shared dict instance across every LossTerm that doesn't set one).
    """
    name: str
    weight: float = 1.0
    kwargs: Optional[dict] = None


class LossComposer:
    """Composite: weighted sum of named terms -> (total, per-term breakdown).

    Adding a term is a new entry in ``terms``, not a new constructor argument, a
    new ``if``, and a new return-tuple slot (E3_end2end.py).
    """

    def __init__(self, terms):
        """`terms`: sequence of (name, weight, kwargs) -- LossTerm or any 3-tuple
        with those fields. No config import here."""
        self.terms = list(terms)

    def names(self):
        """Configured term names, in order."""
        return [name for name, _, _ in self.terms]

    def kwargs_for(self, name):
        """Extra keyword arguments configured for ``name`` (``{}`` if unconfigured).

        The composer never calls loss functions itself -- it sums values that are
        already computed -- so these are for the CALLER that computes the term,
        e.g. the trainer passing ``var_weight`` into ``contrastive_alignment_loss``.
        Keeping such per-term hyperparameters on the term is what stops them
        becoming extra TrainingStepper constructor arguments (T10 step 6).
        """
        for term_name, _, kwargs in self.terms:
            if term_name == name:
                return dict(kwargs or {})
        return {}

    def compute(self, values):
        """`values`: {name: Tensor | None}. Returns (total_scalar, {name: float}).

        A term whose value is None is SKIPPED -- not an error, and not zero-filled
        into the breakdown either. This is how `kl` vanishes when it isn't a
        configured term at all, and how `contrastive` vanishes during validation. 
        Only terms that actually contributed appear in ``breakdown``, keyed by the same name
        in both train and val, so ``val/<term>`` lines up with ``train/<term>``
        wherever both are present.
        """
        total = None
        breakdown = {}

        for name, weight, kwargs in self.terms:
            value = values.get(name)
            if value is None:
                continue
            weighted = weight * value
            total = weighted if total is None else total + weighted
            breakdown[name] = value.item()

        if total is None:
            # Every configured term was skipped this call -- still return a valid
            # scalar zero, not None, so callers can always call .backward() on it.
            total = torch.zeros(())

        # The deleted draft seeded its accumulator with torch.zeros(1) -- shape
        # [1], not scalar -- and broadcasting silently kept it that way forever.
        # Catch that class of bug loudly instead of letting a non-scalar total
        # propagate into .backward().
        assert total.dim() == 0, f"composed loss must be scalar, got shape {tuple(total.shape)}"

        if not torch.isfinite(total):
            raise FloatingPointError(f"non-finite loss ({total.item()}); aborting.")

        return total, breakdown