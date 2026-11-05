"""Fits of a frozen shape model (src/shared/shape_model.py) to observations.

    code_fit.py      the fit loop: Adam over the code, the decoder fixed
    data_terms.py    chamfer, l2 and varifold data terms
    stage.py         the code_fit stage: one target scan per run
    registrations.py the stage, by name (joined by scripts/pipeline/registry.py)
"""
