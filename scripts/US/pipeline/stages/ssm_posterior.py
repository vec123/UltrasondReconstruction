"""Stage: conditions an SSM on the trusted vertices of a registration - a GP posterior.

The input is a registration that deformed the SSM's own mean shape, vertex for
vertex (resnet_lddmm_apply's world/deformed.vtp on the baby mean), and a
per-vertex error for it (recon_target_distance's recon_w_distance.vtp, point
field recon_dist_2_target). `noise` turns that error into an observation std
(see src.SSM.model.inference.observations): `cutoff` observes only the vertices under a
threshold, `logistic` (the default) observes all of them but trusts the far
ones less and less. With `error_direction: target_to_recon` the error is read
from the scan's side instead: every target point picks its nearest
reconstruction vertex, so only vertices some scan point lands on are observed
at all - a vertex close to the scan but not on it (past its border, say) is
not. The observed position is always the registration's own vertex; only
which vertices count, and how much, changes. `observation_spacing` then keeps
about one observation per that many mm: neighbouring vertices share their
error, and the thinned set is closer to the independent measurements the
posterior assumes. The model conditioned on those observations
(src.SSM.model.inference.posterior) keeps the prior's uncertainty where the scan showed
nothing, and reports it per vertex.

Frames. The model lives in its own frame (~0.013 units across for the baby
models), the registration in mm. `template_vtp` is the model mean in the
registration's frame (resnet_lddmm_apply's world/template_points.vtp); a
Procrustes fit of the model mean onto it, vertex for vertex, gives the
similarity between the two, and the stage refuses to run when that fit leaves
a residual (the mean is then not the template). Observations go into the model
frame through it and results come back out, so every mesh written is in
observed_vtp's frame - recon_target_distance can score posterior_mean.vtp with
the same settings as observed_vtp. posterior_model.npz stays in the model
frame, so ssm_fit can use it as `shape_model_mat`.

Pose. A registration with a pose of its own (resnet_lddmm's
pose_application: post) leaves the observations rotated and rescaled against
the template, which the model can only absorb with extreme shape modes.
`pose: similarity` (the default; `rigid` without scale, `none` to skip) removes
that pose before conditioning and refits it `pose_iterations` times against the
latest posterior mean - see src.SSM.model.inference.posterior.posterior_with_pose. The
written meshes carry the pose again, so they still sit on observed_vtp;
posterior_model.npz is the pose-free shape posterior.

Outputs, in output_dir:
    posterior_mean.vtp       template_vtp's point fields, plus (lengths in mm):
                             posterior_std_mm / prior_std_mm (sqrt trace of each
                             vertex's covariance), observed (0/1),
                             informative (0/1), observation_std_mm (NaN where
                             unobserved), the error (error_array, or
                             target_to_recon_error; NaN where no target point
                             picked the vertex), posterior_shift_mm (distance
                             from observed_vtp)
    posterior_model.npz      the posterior as a GPMM archive, model frame
    samples/sample_XX.vtp    n_samples draws from the posterior
    posterior_summary.json   counts, frame fit, log marginal likelihood, statistics (mm)

Name output_dir after variant_name(cfg), so the folder says what it holds.
"""

import dataclasses
import json
import os

import numpy as np
import vtk
from scipy.spatial import cKDTree
from vtk.util import numpy_support
from vtk.util.numpy_support import vtk_to_numpy

from src.SSM.config import SsmPosteriorConfig as Config
from src.SSM.model.model_memory_io import load_model_file
from src.SSM.model.inference.observations import Observations, build_noise_model
from src.SSM.model.inference.posterior import posterior, posterior_with_pose, save_posterior
from src.shared.transforms.similarity import SimilarityTransform
from src.shared.vtk.create import create_polydata
from src.shared.vtk.fields import add_point_field, add_point_fields
from src.shared.vtk.io import load_vtp, save_vtp


def mesh_points(polydata, n_vertices, path):
    points = vtk_to_numpy(polydata.GetPoints().GetData()).astype(np.float64)
    if len(points) != n_vertices:
        raise ValueError(f"{path} has {len(points)} points, the model has {n_vertices}")
    return points


def point_array(polydata, name, path):
    array = polydata.GetPointData().GetArray(name)
    if array is None:
        raise ValueError(f"{path} has no point field '{name}'")
    return vtk_to_numpy(array).astype(np.float64).reshape(-1)


POSE_TOKENS = {"similarity": "pose", "rigid": "rigid", "none": "nopose"}
DIRECTION_TOKENS = {"recon_to_target": "r2t", "target_to_recon": "t2r"}


def variant_name(cfg) -> str:
    """Folder name spelling out every setting that changes the result:

        posterior_<model>_<pose>_<distance>_<noise><threshold>mm[_spacing<s>mm]_sigma<sigma>mm

    e.g. ``posterior_baby-pca_pose_t2r_cutoff1.5mm_spacing3mm_sigma3mm``. A noise model's
    other parameters, a changed ``pose_iterations`` and ``include_model_noise:
    false`` are appended only when they differ from their defaults, so the
    common names stay short and two different settings never share one.
    """
    noise = build_noise_model(cfg.noise.name, cfg.noise.kwargs)
    params = {f.name: getattr(noise, f.name) for f in dataclasses.fields(noise)}
    defaults = {f.name: f.default for f in dataclasses.fields(noise)}
    noise_token = cfg.noise.name
    if "threshold" in params:
        noise_token += f"{params['threshold']:g}mm"
    noise_token += "".join(f"-{key}{value:g}" for key, value in params.items()
                           if key not in ("sigma", "threshold") and value != defaults[key])

    model = os.path.splitext(os.path.basename(cfg.shape_model_mat))[0].replace("_", "-")
    parts = ["posterior", model, POSE_TOKENS[cfg.pose], DIRECTION_TOKENS[cfg.error_direction], noise_token]
    if cfg.observation_spacing > 0:
        parts.append(f"spacing{cfg.observation_spacing:g}mm")
    parts.append(f"sigma{params['sigma']:g}mm")
    if cfg.pose != "none" and cfg.pose_iterations != Config.pose_iterations:
        parts.append(f"poseit{cfg.pose_iterations}")
    if not cfg.include_model_noise:
        parts.append("nomodelnoise")
    return "_".join(parts)


def target_to_recon_errors(recon_points, target_points):
    """(N,) per recon vertex: the smallest distance of the target points whose
    nearest recon vertex it is; inf where no target point picked it."""
    distances, nearest = cKDTree(recon_points).query(target_points)
    errors = np.full(len(recon_points), np.inf)
    np.minimum.at(errors, nearest, distances)
    return errors


def vertex_errors(cfg, n_vertices):
    """(N,) per-vertex error and the point field name it is saved under."""
    error_mesh = load_vtp(cfg.error_vtp)
    if cfg.error_direction == "recon_to_target":
        errors = point_array(error_mesh, cfg.error_array, cfg.error_vtp)
        if len(errors) != n_vertices:
            raise ValueError(f"{cfg.error_vtp} has {len(errors)} points, the model has {n_vertices}")
        return errors, cfg.error_array
    if cfg.error_direction == "target_to_recon":
        target = load_vtp(cfg.target_vtp)
        target_points = vtk_to_numpy(target.GetPoints().GetData()).astype(np.float64)
        recon_points = mesh_points(error_mesh, n_vertices, cfg.error_vtp)
        return target_to_recon_errors(recon_points, target_points), "target_to_recon_error"
    raise ValueError(f"error_direction must be 'recon_to_target' or 'target_to_recon', "
                     f"got '{cfg.error_direction}'")


def fit_frame(model_mean, template_points, tolerance, template_path):
    """Similarity from template_vtp's frame into the model's, and its residual.

    Fitted model -> template (the direction a Procrustes fit is best posed in,
    with the template as the fixed shape), then inverted.
    """
    to_template = SimilarityTransform.from_correspondences(
        model_mean, template_points, with_scale=True, stage="ssm_posterior")
    residual = np.linalg.norm(to_template.apply_points(model_mean) - template_points, axis=1)
    diagonal = np.linalg.norm(np.ptp(template_points, axis=0))
    if residual.max() > tolerance * diagonal:
        raise ValueError(
            f"{template_path} is not the model mean up to a similarity: Procrustes residual "
            f"{residual.max():.4g} > {tolerance} x bbox diagonal {diagonal:.4g}"
        )
    return to_template.inverse(), residual


def with_points(template, points):
    """A copy of template (faces + point fields) at new point positions."""
    mesh = vtk.vtkPolyData()
    mesh.DeepCopy(template)
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(numpy_support.numpy_to_vtk(np.ascontiguousarray(points, dtype=np.float64), deep=True))
    mesh.SetPoints(vtk_points)
    return mesh


def run(cfg: Config):
    os.makedirs(cfg.output_dir, exist_ok=True)

    model = load_model_file(cfg.shape_model_mat)
    n_vertices = model.n_vertices
    template = load_vtp(cfg.template_vtp)
    template_points = mesh_points(template, n_vertices, cfg.template_vtp)
    observed_points = mesh_points(load_vtp(cfg.observed_vtp), n_vertices, cfg.observed_vtp)
    errors, error_name = vertex_errors(cfg, n_vertices)

    to_model, frame_residual = fit_frame(model.mean.reshape(-1, 3), template_points,
                                         cfg.frame_tolerance, cfg.template_vtp)
    to_observed = to_model.inverse()
    mm = 1.0 / to_model.scale  # observed_vtp units per model unit

    noise = build_noise_model(cfg.noise.name, cfg.noise.kwargs)
    observations = Observations.from_errors(observed_points, errors, noise)
    n_selected = observations.n
    observations = observations.thinned(cfg.observation_spacing)
    print(f"ssm_posterior: {noise.describe()} -> {observations.describe()} "
          f"of {n_vertices} vertices ({n_selected} before thinning to {cfg.observation_spacing:g} mm)")
    if not observations.n:
        raise ValueError("no vertex is observed - every noise std is inf; loosen the noise model")

    in_model = observations.transformed(to_model)
    if cfg.pose == "none":
        result = posterior(model, in_model, include_model_noise=cfg.include_model_noise)
    elif cfg.pose in ("rigid", "similarity"):
        result = posterior_with_pose(model, in_model, with_scale=cfg.pose == "similarity",
                                     iterations=cfg.pose_iterations,
                                     include_model_noise=cfg.include_model_noise)
    else:
        raise ValueError(f"pose must be 'none', 'rigid' or 'similarity', got '{cfg.pose}'")
    # A model-frame length times this is mm on observed_vtp, the pose's scale included.
    length = mm * (result.pose.scale if result.pose is not None else 1.0)

    def to_observed_points(shape):
        return to_observed.apply_points(result.posed(shape).reshape(-1, 3))

    observed = observations.mask(n_vertices)
    observation_std = np.full(n_vertices, np.nan)
    observation_std[observations.indices] = observations.stds
    mean_points = to_observed_points(result.mean)

    mesh = with_points(template, mean_points)
    add_point_fields(mesh, {
        "prior_std_mm": result.prior_std() * length,
        "observed": observed.astype(np.uint8),
        "informative": result.informative_mask().astype(np.uint8),
        "observation_std_mm": observation_std,
        error_name: np.where(np.isfinite(errors), errors, np.nan),
        "posterior_shift_mm": np.linalg.norm(mean_points - observed_points, axis=1),
    })
    add_point_field(mesh, result.vertex_std() * length, "posterior_std_mm")
    mean_path = os.path.join(cfg.output_dir, "posterior_mean.vtp")
    save_vtp(mesh, mean_path)

    model_path = os.path.join(cfg.output_dir, "posterior_model.npz")
    save_posterior(model_path, result, provenance={
        "prior": cfg.shape_model_mat,
        "observed_vtp": cfg.observed_vtp,
        "error_vtp": cfg.error_vtp,
        "error_array": cfg.error_array,
        "error_direction": cfg.error_direction,
        "noise": {"name": cfg.noise.name, "kwargs": dict(cfg.noise.kwargs)},
        "include_model_noise": cfg.include_model_noise,
        "pose": cfg.pose,
        "pose_iterations": cfg.pose_iterations,
        "observation_spacing": cfg.observation_spacing,
    })

    if cfg.n_samples > 0:
        sample_dir = os.path.join(cfg.output_dir, "samples")
        os.makedirs(sample_dir, exist_ok=True)
        for k, shape in enumerate(result.sample(cfg.n_samples, rng=cfg.seed)):
            save_vtp(create_polydata(to_observed_points(shape), faces=model.faces),
                     os.path.join(sample_dir, f"sample_{k:02d}.vtp"))

    summary = {
        "units": "observed_vtp units (mm)",
        "shape_model_mat": cfg.shape_model_mat,
        "observed_vtp": cfg.observed_vtp,
        "error_vtp": cfg.error_vtp,
        "error_array": cfg.error_array,
        "error_direction": cfg.error_direction,
        "noise": {"name": cfg.noise.name, "kwargs": dict(cfg.noise.kwargs)},
        "include_model_noise": cfg.include_model_noise,
        "observation_spacing_mm": cfg.observation_spacing,
        "n_selected_before_thinning": int(n_selected),
        "frame": {"scale_mm_per_model_unit": float(mm),
                  "procrustes_residual_max": float(frame_residual.max())},
        "posterior_mean": mean_path,
        "posterior_model": model_path,
        **result.summary(length_scale=length),
    }
    if result.pose is not None:
        summary["pose"] = {**summary["pose"], "mode": cfg.pose, "iterations": cfg.pose_iterations,
                           "translation_mm": float(np.linalg.norm(result.pose.translation) * mm)}
    summary_path = os.path.join(cfg.output_dir, "posterior_summary.json")
    with open(summary_path, "w") as handle:
        json.dump(summary, handle, indent=2)

    print(
        f"ssm_posterior: {summary['n_informative']}/{n_vertices} informative, posterior std median "
        f"{summary['posterior_std_informative'].get('median', float('nan')):.3f} mm informative / "
        f"{summary['posterior_std_uninformative'].get('median', float('nan')):.3f} mm elsewhere "
        f"-> {cfg.output_dir}"
    )
