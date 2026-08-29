import numpy as np
from scripts.shape_model.align.src.correspondence.correspondence import closest_points
from scripts.shape_model.align.src.fitting.fitting import fit_once

def icp_fit(model,
            noisy_mesh_vertices,
            alpha0=None,
            n_iterations=20,
            probability=0.95):

    if alpha0 is None:
        alpha = np.zeros(model.n_components)
    else:
        alpha = alpha0.copy()

    history = []

    for _ in range(n_iterations):

        current_vertices = (
            model.reconstruct(alpha)
            .reshape(-1, 3)
        )

        targets = closest_points(
            current_vertices,
            noisy_mesh_vertices
        )

        alpha = fit_once(
            model,
            alpha,
            targets,
            probability
        )

        history.append(alpha.copy())

    return alpha, history

