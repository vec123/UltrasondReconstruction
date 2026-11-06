"""Runner for the morphing pipeline.

Step 1: landmark-driven similarity alignment (1a) + landmark fit of the SSM (1b).
Step 2: correspondence + MAP-regularised NICP fits (gamma sweep).
Step 3: regional Laplacian-constrained deformation.

Each stage registers its fitted shape in ``self.shapes`` / ``self.vertices``
under a key; later stages address earlier results by that key:

    refined          step 1 result (mean shape if step 1 is disabled)
    gamma_<value>    one per entry in nicp.gammas
    <variant name>   one per entry in nicp.variants

Stages are switched on and off with the ``steps`` config block. Step 3 can
therefore run straight off ``refined`` when step 2 is disabled.

The runner is split by step, into mixins of one class: runner.py (the plan, the
inputs and the order of the steps), alignment.py (steps 0b and 1), fitting.py
(steps 2 and 3) and results.py (PipelineResult, result.json).
"""

from src.models.gpmm.runners.pipeline.results import REFINED_KEY, PipelineResult, write_fit_results
from src.models.gpmm.runners.pipeline.runner import PipelineRunner

__all__ = ["REFINED_KEY", "PipelineResult", "PipelineRunner", "write_fit_results"]
