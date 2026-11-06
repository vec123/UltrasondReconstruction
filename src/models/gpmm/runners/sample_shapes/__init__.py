"""Interactive explorer for the shape space: one slider per mode.

The mean shape and the heatmaps from ``mean_shape.py`` describe the model in
the abstract - this walks it. Each slider sets one coefficient, so what the
window shows at any moment is

.. math:: x(\\alpha) = \\bar{x} + \\sum_k \\alpha_k \\phi_k

with the slider reading in *standard deviations* rather than raw coefficients:
:math:`\\alpha_k = s_k \\sqrt{\\lambda_k}`. That is the only scale on which the
modes are comparable - the raw coefficients differ by orders of magnitude
between the leading and the trailing modes, and by another few between the two
models - so slider 0 at +2 and slider 5 at +2 mean the same thing, namely two
standard deviations of a population that is Gaussian by construction. It also
makes the plausibility of a sample readable straight off the sliders: the sum
of their squares is the Mahalanobis distance the priors in the fitting code
penalise.

A model has far more modes than fit on screen (183 on the fetus model, 163 on
the baby), so the panel scrolls: a fixed pool of slider widgets is re-bound to
a sliding window of modes, which keeps the widget count - and the cost of
scrolling - independent of how many modes there are.

Which model opens is a config key, but it is not the end of it: the window
carries a dropdown listing every model the config declares under ``models``
*and* every archive found under ``library.root``, and picking one reloads the
window in place. That is what makes a folder of built GPMMs explorable - the
twenty-six kernels under ``configs/gpmm/build`` are a click apart
instead of an edit and a restart - and it is why nothing below is specific to
any one model: they all arrive through
:func:`src.models.gpmm.model.model_memory_io.load_model_file` as the same
mean/eigenfunctions/eigenvalues triple.

The info line under the dropdown reports the two numbers that decide whether a
model is worth looking at: how many modes it has, and how much of its own
covariance those modes carry. The second is below 100% for every GPMM built by
the ssm_build stage, because the decomposition is truncated and the remainder is
booked as ``noise`` - see :meth:`SampleShapesRunner._set_variance_explained`.

The runner is one class made of mixins, one module per part of the window:
runner.py (the model, the state, export), scene.py (the surface and its colour
modes), widgets.py (sliders, buttons, labels, theme), dropdown.py (the model
dropdown and switching models), events.py (scrolling, mouse and keys) and
style.py (colours, themes and the panel geometry).
"""

from src.models.gpmm.runners.sample_shapes.runner import SampleShapesResult, SampleShapesRunner

__all__ = ["SampleShapesResult", "SampleShapesRunner"]
