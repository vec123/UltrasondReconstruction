"""Config for scripts/upsample_atlas.py."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


# --------------------------------------------------------------------------
# scripts/upsample_atlas.py
# --------------------------------------------------------------------------

@dataclass
class UpsampleAtlasConfig:
    """Resamples a Deformetrica atlas's template to roughly ``scale`` x its
    vertex count, and writes the result as a new, independent atlas directory.

    Only the template surface changes resolution. The control points and every
    subject's momenta are copied through unchanged, because they do not depend
    on how densely the carrier mesh is sampled: :func:`src.model.deformetrica.shoot`
    takes the deformation kernel and *any* set of points, so a denser template
    is shot by exactly the code path the original resolution already uses -
    this is not an approximation layered on top of the model, it is the same
    model evaluated at more points. Reusing that path is also what makes the
    output a normal atlas directory: point ``shape_model`` / ``shape_model_mat``
    at it in any other config and every loader treats it like any other.
    """

    source: Path            # the original atlas directory
    output_dir: Path        # where the new, denser atlas is written
    scale: float = 2.0      # target vertex-count multiplier
    # Shoots the whole population at the new resolution and PCAs it once,
    # right after writing, so the expensive part (see time_points below) is
    # already cached beside the new atlas - src.model.deformetrica caches by
    # kernel width and time_points, so this has to match whatever another
    # config's model.atlas.time_points will ask for to actually save that run
    # its ~90s-and-up shoot. False skips it: the first real use pays that cost
    # instead, exactly like a freshly-decomposed original atlas would.
    warm_cache: bool = True
    warm_cache_time_points: int = 11
    verbose: bool = True

    def __post_init__(self):
        if self.scale <= 1.0:
            raise ValueError(f"scale must be greater than 1.0, got {self.scale}")
        if self.warm_cache_time_points < 2:
            raise ValueError(
                f"warm_cache_time_points must be at least 2, got {self.warm_cache_time_points}"
            )
