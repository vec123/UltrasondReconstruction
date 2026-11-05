"""How good a template is as the atlas of a cohort: central, compact, specific.

The functions take plain arrays, so any model's atlas can be scored the same way
(as src/evaluation/metrics.py does for distances): the members' reconstructions
in vertex correspondence with the template, [M, N, 3] in mm, their codes
[M, n_z], their velocities, their Jacobian determinants. Running a model to get
these arrays is the caller's job.

Centrality: an atlas is central when the members' displacements from it cancel
out (bias near 0) and no other shape needs less total deformation (Fréchet test).
Compactness, specificity and generalization are the three standard measures of a
shape model (Styner et al. 2003, Davies et al. 2002).
"""

import numpy as np


def bias(template, recons, vertex_weights=None):
    """The mean displacement from the template and how much of the displacement it is.

    The bias ratio |mean displacement| / mean |displacement| is 0 when the members
    move in all directions from the template (a central template) and 1 when they
    all move the same way.

    Args:
        template: [N, 3] template points
        recons: [M, N, 3] the members' reconstructions, in vertex correspondence
        vertex_weights: [N] weights of the overall ratio, e.g. the coverage; None: all 1

    Returns:
        (mean displacement [N, 3], bias ratio per vertex [N], overall bias ratio)
    """
    displacement = recons - template
    mean = displacement.mean(axis=0)
    mean_norm = np.linalg.norm(mean, axis=1)
    norm_mean = np.linalg.norm(displacement, axis=2).mean(axis=0)
    w = np.ones(len(template)) if vertex_weights is None else vertex_weights
    per_vertex = mean_norm / np.maximum(norm_mean, 1e-12)
    return mean, per_vertex, float(np.sum(w * mean_norm) / np.sum(w * norm_mean))


def coverage(distances, threshold):
    """[N] the share of members whose scan is nearer than `threshold` to each vertex.

    Args:
        distances: [M, N] distance of each member's reconstructed vertex to its scan
        threshold: the distance below which a vertex counts as seen
    """
    return (distances < threshold).mean(axis=0)


def _squared_distances(a, b):
    """[len(a), len(b)] mean squared vertex distance between shapes [A, N, 3] and [B, N, 3]."""
    a, b = a.reshape(len(a), -1), b.reshape(len(b), -1)
    n = a.shape[1] // 3
    gram = (a ** 2).sum(1)[:, None] + (b ** 2).sum(1)[None] - 2 * a @ b.T
    return np.maximum(gram, 0.0) / n


def frechet_test(template, recons, n_random=10, seed=0):
    """Total squared deformation from the template, against other candidate atlases.

    The total is sum_i mean_v |X_i(v) - C(v)|^2 for a candidate C. The members' mean
    shape has the smallest total of all shapes, and any C exceeds it by
    M * mean_v |C - mean|^2: the template's excess is its squared bias. A central
    template is near the mean and below the medoid (the member with the smallest
    total to all others) and random members.

    Returns:
        {"mean", "template", "medoid", "medoid_index", "random"}: totals, "random"
        a list of n_random totals
    """
    recons_sq = _squared_distances(recons, recons)
    totals = recons_sq.sum(axis=0)
    medoid = int(np.argmin(totals))
    picks = np.random.default_rng(seed).choice(len(recons), size=min(n_random, len(recons)), replace=False)
    return {
        "mean": float(_squared_distances(recons.mean(axis=0)[None], recons).sum()),
        "template": float(_squared_distances(template[None], recons).sum()),
        "medoid": float(totals[medoid]),
        "medoid_index": medoid,
        "random": [float(totals[i]) for i in picks],
    }


def kinetic_energy(velocities, dt):
    """½ ∫ mean_n ||v_n(t)||² dt per member, as Trajectory.kinetic_energy reduces one member.

    Args:
        velocities: [..., K, N, 3] the velocity of each point at each of K steps
        dt: the step length
    """
    return 0.5 * (velocities ** 2).sum(-1).mean(-1).sum(-1) * dt


def mean_code_difference(template, recons, mean_code_recon):
    """How far the shape of the mean code is from the mean shape of the members.

    The two are equal when the decoder is linear in the code; otherwise the
    difference is of second order in the code spread.

    Returns:
        {"mean", "p95", "max"} of the per-vertex difference, "relative" (its mean over
        the mean |mean displacement|) and "cosine" between the two displacement fields
    """
    mean_displacement = recons.mean(axis=0) - template
    code_displacement = mean_code_recon - template
    gap = np.linalg.norm(code_displacement - mean_displacement, axis=1)
    cosine = np.sum(code_displacement * mean_displacement) / (
        np.linalg.norm(code_displacement) * np.linalg.norm(mean_displacement))
    return {"mean": float(gap.mean()), "p95": float(np.percentile(gap, 95)), "max": float(gap.max()),
            "relative": float(gap.mean() / np.linalg.norm(mean_displacement, axis=1).mean()),
            "cosine": float(cosine)}


def compactness(samples):
    """[min(M, D)] cumulative share of the variance in the first k principal components.

    A compact model needs few components for most of the variance. `samples` is
    [M, ...]: shapes [M, N, 3] or codes [M, n_z].
    """
    flat = samples.reshape(len(samples), -1)
    s = np.linalg.svd(flat - flat.mean(axis=0), compute_uv=False)
    return np.cumsum(s ** 2) / np.sum(s ** 2)


def specificity(samples, recons):
    """[S] RMS vertex distance from each sampled shape to its nearest member.

    Low values mean that shapes drawn from the model look like real members.

    Args:
        samples: [S, N, 3] shapes decoded from random codes
        recons: [M, N, 3] the members' reconstructions
    """
    return np.sqrt(_squared_distances(samples, recons).min(axis=1))


def generalization(errors):
    """Summary of held-out errors: a model that generalizes reconstructs new cases well.

    Args:
        errors: one array per held-out case, e.g. its scan-to-reconstruction distances

    Returns:
        {"mean", "median", "p95"} over the cases of each case's mean error
    """
    per_case = np.array([np.mean(e) for e in errors])
    return {"mean": float(per_case.mean()), "median": float(np.median(per_case)),
            "p95": float(np.percentile(per_case, 95))}


def triangle_area_ratios(template, recons, faces):
    """[M, F] each triangle's area in each reconstruction over its area in the template.

    The surface Jacobian determinant: 1 keeps the area, <= 0 would be a collapsed
    or flipped triangle.

    Args:
        faces: [F, 3] vertex indices of the template's triangles
    """
    def areas(points):
        a, b, c = (points[..., faces[:, k], :] for k in range(3))
        return 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=-1)
    return areas(recons) / np.maximum(areas(template), 1e-12)


def determinant_stats(determinants):
    """{"min", "p1", "median", "p99", "max", "share_nonpositive"} of Jacobian determinants."""
    d = np.asarray(determinants).ravel()
    return {"min": float(d.min()), "p1": float(np.percentile(d, 1)), "median": float(np.median(d)),
            "p99": float(np.percentile(d, 99)), "max": float(d.max()),
            "share_nonpositive": float((d <= 0).mean())}
