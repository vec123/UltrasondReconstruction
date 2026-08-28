from scripts.shape_model.align.src.plausibility.plausibility import project_to_plausible_space


def parameter_update(phi,
                     current_vertices,
                     target_vertices):

    x_current = current_vertices.reshape(-1)
    x_target = target_vertices.reshape(-1)

    displacement = x_target - x_current

    delta_alpha = phi.T @ displacement

    return delta_alpha


def fit_once(model,
             alpha,
             target_vertices,
             probability=0.95):

    current_vertices = (
        model.reconstruct(alpha)
        .reshape(-1, 3)
    )

    delta_alpha = parameter_update(
        model.eigenfunctions,
        current_vertices,
        target_vertices
    )

    alpha_new = alpha + delta_alpha

    alpha_new = project_to_plausible_space(
        alpha_new,
        model.eigenvalues,
        probability
    )

    return alpha_new