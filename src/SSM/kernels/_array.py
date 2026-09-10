"""Array-namespace dispatch, so one kernel implementation serves two backends.

The kernel algebra is numpy: a model is built once, offline, and cached, and
nothing in that path needs a gradient. But learning kernel *hyperparameters*
(see ``src/SSM/hyper/``) does need one, and the quantity being differentiated is
the same radial expression the numpy path evaluates - ``scale * exp(-d^2/2s^2)``
and its relatives. Writing those twice would guarantee they drift.

So the stationary kernels are written against whatever namespace their input
array came from. Feed them numpy and they are numpy; feed them torch tensors
that require grad and the same lines build a differentiable graph. Nothing here
imports torch: the module is identified from the array's own type, so this file
- and therefore ``src.SSM.kernels`` - stays importable in an environment with no
torch at all.

Only the handful of operations the stationary kernels actually use is covered.
That is deliberate: a general array-API shim is a large surface with subtle
differences (``clip``/``clamp``, ``concatenate``/``cat``, reduction keyword
names), and every entry added here is another place the two backends can
disagree. Add to :class:`_Torch` only when a kernel genuinely needs it.
"""

from __future__ import annotations

import numpy as np


def is_torch(x) -> bool:
    """True when ``x`` is a torch tensor, without importing torch to find out."""
    return type(x).__module__.split(".")[0] == "torch"


class _Torch:
    """The subset of the numpy API the stationary kernels use, on torch.

    Held as a class of static methods rather than the ``torch`` module itself
    because a few names differ (``clamp`` for ``clip``) and a few numpy
    signatures accept arguments torch's do not.
    """

    def __init__(self, torch_module):
        self._t = torch_module

    def exp(self, x):
        return self._t.exp(x)

    def sqrt(self, x):
        return self._t.sqrt(x)

    def maximum(self, x, y):
        return self._t.clamp(x, min=y) if np.isscalar(y) else self._t.maximum(x, y)

    def where(self, cond, x, y):
        return self._t.where(cond, x, y)

    def zeros_like(self, x):
        return self._t.zeros_like(x)

    def ones_like(self, x):
        return self._t.ones_like(x)

    def asarray(self, x, like=None):
        if is_torch(x):
            return x
        return self._t.as_tensor(
            np.asarray(x),
            dtype=None if like is None else like.dtype,
            device=None if like is None else like.device,
        )


def array_namespace(*arrays):
    """The namespace to evaluate in, chosen from the arrays passed in.

    numpy unless one of ``arrays`` is a torch tensor, in which case a shim over
    the live ``torch`` module. The import happens here, on the branch that
    already holds a tensor, so a numpy-only caller never triggers it.
    """
    for array in arrays:
        if is_torch(array):
            import torch  # noqa: PLC0415 - deliberate: only reached when a tensor is present

            return _Torch(torch)
    return np
