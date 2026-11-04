"""Fit only the code of a frozen SSM-trained flow to the partial observations of posterior_ssm_tests.py.

posterior_ssm_tests.py draws ground truths from the shape model, poses them randomly,
adds noise and observes only part of each (face_center, right_half, lower_face,
upper_face, sparse_face), then fits the SSM posterior. This script takes the SAME tests
(re-drawn with the same seed and checked against the saved ground_truth.vtp /
observed.vtp) and fits instead the code z of a cohort auto-decoder trained on samples of
that model, with its flow frozen, by minimising

    E(z) = sum_obs |phi_z(T)_i - y_i|^2 / (2 sigma^2)  +  w * 1/2 (z - mu)^T S^-1 (z - mu)

over the observed vertices i (correspondence is known: the observations are the
template's vertices), with mu, S the mean and covariance of the run's training codes.
The variants (--variants) set sigma (mm) and the prior weight w: "data" is the data term
alone (w = 0), "gauss<sigma>" the MAP estimate under the Gaussian code prior (w = 1), the
counterpart of the posterior fit with the same sigma.

The fit is given the TRUE pose (the posterior estimates its own): the observations are
mapped back into the model frame with it, and every fit is posed with it again for the
comparison with the ground truth.

Writes <output>/summary.json (per test and variant: median error on the observed and the
free vertices, 95th percentile on the free ones, the code's squared Mahalanobis distance;
the posterior's numbers next to them) and <output>/<test>.png: ground truth | observations
| each code fit | the posterior with the largest sigma, from the front, left and right, plus
a frontal row of error maps.

Run with:
    python scripts/inspection/code_fit_partial_tests.py \\
        --run outputs/baby_ssm_cohort/resnet_lddmm_cohort_autodecoder_sobol1000_global_l2_mlp6x256_kin200
"""

import argparse
import json
import os
import sys

# KeOps compiles its CUDA kernels against the CUDA headers and libraries that PyTorch's pip
# packages ship, linked under the standard names into <env>/cuda_keops (include/, lib/).
_KEOPS_CUDA = os.path.join(sys.prefix, "cuda_keops")
if os.path.isdir(_KEOPS_CUDA):
    os.environ.setdefault("CUDA_PATH", _KEOPS_CUDA)
    os.environ["LD_LIBRARY_PATH"] = os.path.join(_KEOPS_CUDA, "lib") + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")

import numpy as np
import torch
from vtk.util.numpy_support import vtk_to_numpy

from scripts.inspection import render_frontal_views as views  # noqa: E402
import scripts.inspection.render_ssm_cohort_views  # noqa: E402,F401  (RGB panels: colour None draws the point scalars)
from scripts.inspection.eval_ssm_cohort import load_run  # noqa: E402
from src.shared.core.config import load_yaml  # noqa: E402
from src.models.lddmm.resnet_flow.config import ExperimentCfg  # noqa: E402
from src.models.lddmm.resnet_flow.runner import _normalization_domain  # noqa: E402
from scripts.inspection.posterior_ssm_tests import MM_PER_UNIT, TESTS, random_pose, region  # noqa: E402
from scripts.inspection.render_ssm_posterior_views import ramp_colors, sub_surface, with_colors  # noqa: E402
from src.models.gpmm.model.model_memory_io import load_model_file  # noqa: E402
from src.shared.vtk.create import create_polydata  # noqa: E402
from src.shared.vtk.io import load_vtp  # noqa: E402

ERROR_MAX_MM = 3.0


def redraw_tests(model, seed, noise_mm, tests_dir):
    """Per test: (name, posed truth [N,3] mm, observations [N,3] mm, mask [N], pose matrix [4,4]),
    re-drawn exactly as posterior_ssm_tests.py draws them and checked against its files."""
    rng = np.random.default_rng(seed)
    lam = np.clip(model.eigenvalues, 0.0, None)
    names, idx = model.landmark_names(), model.landmark_indices()
    out = []
    for t, name in enumerate(TESTS):
        alpha = rng.standard_normal(model.n_components) * np.sqrt(lam)
        truth = (model.mean + model.eigenfunctions @ alpha).reshape(-1, 3) * MM_PER_UNIT
        pose, _, _ = random_pose(rng, truth.mean(axis=0))
        posed = pose.apply_points(truth)
        observed = posed + rng.normal(scale=noise_mm, size=truth.shape)
        lmk = {n: posed[i] for n, i in zip(names, idx)}
        mask = region(name, posed, lmk, rng)
        test = f"test_{t:02d}_{name}"
        saved = vtk_to_numpy(load_vtp(os.path.join(tests_dir, test, "ground_truth.vtp")).GetPoints().GetData())
        saved_obs = load_vtp(os.path.join(tests_dir, test, "observed.vtp"))
        saved_mask = vtk_to_numpy(saved_obs.GetPointData().GetArray("observed")).astype(bool)
        if np.abs(saved - posed).max() > 1e-3 or not np.array_equal(saved_mask, mask):
            raise ValueError(f"{test}: the re-drawn test does not match {tests_dir}; "
                             f"check --seed / --noise-mm against the posterior run")
        obs_pts = vtk_to_numpy(saved_obs.GetPoints().GetData()).astype(np.float64)
        out.append((test, posed, obs_pts, mask, pose.matrix, lmk))
    return out


def flow_decoder(flow, template, center, scale, mm_per_unit):
    """decoder(idx [T, M], z [T, n_z]) -> [T, M, 3] mm: template vertices idx flowed with z."""
    def decode(idx, z):
        return (flow(template[idx], z).points[-1] * scale + center) * mm_per_unit
    return decode


def ssm_decoder(model, mm_per_unit, device):
    """decoder(idx [T, M], s [T, K]) -> [T, M, 3] mm: mean + Phi sqrt(lambda) s, s in sigma units."""
    mean = torch.as_tensor(model.mean.reshape(-1, 3), dtype=torch.float32, device=device)
    basis = torch.as_tensor((model.eigenfunctions * np.sqrt(np.clip(model.eigenvalues, 0, None)))
                            .reshape(-1, 3, model.n_components), dtype=torch.float32, device=device)

    def decode(idx, s):
        return (mean[idx] + torch.einsum("tmck,tk->tmc", basis[idx], s)) * mm_per_unit
    return decode


def triangle_geometry(tris):
    """[..., F, 3, 3] triangle vertices -> (centres, unit normals, areas), as fshape.face_geometry."""
    v0, v1, v2 = tris.unbind(-2)
    cross = torch.cross(v1 - v0, v2 - v0, dim=-1)
    norm = cross.norm(dim=-1, keepdim=True)
    return (v0 + v1 + v2) / 3.0, cross / norm.clamp_min(1e-12), 0.5 * norm[..., 0]


def laplacian_smooth(points, faces, iterations, lam=0.5):
    """[N, 3] points after `iterations` uniform Laplacian steps over the edges of `faces`.

    Vertices in no face keep their position. For the varifold's observations: vertex noise
    comparable to the edge length turns the triangle normals the varifold compares
    into noise, which a few steps remove (at the cost of a slight shrinkage).
    """
    if iterations <= 0:
        return points
    edges = torch.cat([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges = torch.cat([edges, edges.flip(1)])
    degree = torch.zeros(len(points), device=points.device).index_add_(0, edges[:, 0], torch.ones(len(edges), device=points.device))
    has = degree > 0
    for _ in range(iterations):
        mean = torch.zeros_like(points).index_add_(0, edges[:, 0], points[edges[:, 1]])
        mean[has] = mean[has] / degree[has, None]
        points = torch.where(has[:, None], (1 - lam) * points + lam * mean, points)
    return points


def varifold_setup(template_mm, template_faces, obs_triangles, n_faces, seed=0):
    """Fixed, area-proportional triangle subsets (drawn once, without replacement), as fshape.py.

    A fixed subset keeps the varifold a deterministic, smooth function of the vertex
    positions; re-drawing it every step made the objective noisy, and drawing with
    replacement put duplicate pairs into <fit, fit>. Each subset's masses are scaled by
    total area / subset area, so it stands for the whole surface.

    Returns:
        dict with the template subset (faces [F, 3], its vertices [U] and the faces
        re-indexed into them [F, 3], the area scale) and per target the observed subset
        (centres, normals, masses).
    """
    gen = torch.Generator(device=template_mm.device).manual_seed(seed)
    tmpl_area = triangle_geometry(template_mm[template_faces])[2]
    fi = torch.multinomial(tmpl_area, min(n_faces, len(tmpl_area)), replacement=False, generator=gen)
    faces = template_faces[fi]
    verts, local = torch.unique(faces, return_inverse=True)
    obs = []
    for tris in obs_triangles:
        c, n, a = triangle_geometry(tris)
        gi = torch.multinomial(a, min(n_faces, len(a)), replacement=False, generator=gen)
        m = a[gi] * a.sum() / a[gi].sum()
        pad = n_faces - len(gi)                  # fewer observed triangles: zero-mass padding
        obs.append(tuple(torch.cat([x, x.new_zeros((pad,) + x.shape[1:])]) for x in (c[gi], n[gi], m)))
    return {"verts": verts, "local": local, "area_scale": tmpl_area.sum() / tmpl_area[fi].sum(), "obs": obs}


def varifold_data(vf, decoder, z, args, gen=None):
    """[T] varifold distance of each target's fit to its observations, in mm^2.

    Unoriented varifold (fshape.py's kernel without a signal) at the widths
    --varifold-sigmas, each normalised by <obs, obs> and multiplied by its width^2, so
    a small misfit d gives about d^2. Both surfaces are the fixed subsets of
    varifold_setup, so the term is a smooth function of the fit's vertex positions.
    Each fit triangle is weighted by its distance d to the observed triangles, held
    fixed within the step (a mask): 1 up to --trunc-mm, then exp(-((d - trunc) / trunc)^2),
    so the surface the observations do not cover is not pulled onto them - the
    varifold's counterpart of the truncated chamfer.
    """
    n_t = z.shape[0]
    pts = decoder(vf["verts"][None].expand(n_t, -1), z)                 # [T, U, 3]
    cf, nf, af = triangle_geometry(pts[:, vf["local"]])                  # [T, F, ...]
    mf = af * vf["area_scale"]
    co, no, mo = (torch.stack(x) for x in zip(*vf["obs"]))               # [T, G, ...]
    with torch.no_grad():
        d = torch.cdist(cf, co).min(2).values
        w = torch.exp(-((d - args.trunc_mm).clamp_min(0) / args.trunc_mm) ** 2)
    mf = mf * w
    sq_ff, sq_fo, sq_oo = torch.cdist(cf, cf) ** 2, torch.cdist(cf, co) ** 2, torch.cdist(co, co) ** 2
    n_ff, n_fo, n_oo = (nf @ nf.transpose(1, 2)) ** 2, (nf @ no.transpose(1, 2)) ** 2, (no @ no.transpose(1, 2)) ** 2
    total = 0.0
    for width in args.varifold_sigmas:
        def inner(m1, sq, nn, m2):
            return torch.einsum("tf,tfg,tg->t", m1, torch.exp(-sq / width ** 2) * nn, m2)
        oo = inner(mo, sq_oo, n_oo, mo)
        total = total + (inner(mf, sq_ff, n_ff, mf) - 2 * inner(mf, sq_fo, n_fo, mo) + oo) / oo * width ** 2
    return total / len(args.varifold_sigmas)


def keops_varifold_one(cf, nf, mf, co, no, mo, widths):
    """Varifold distance (mm^2, as varifold_data) between ONE fit and ONE observation, with KeOps.

    No kernel matrix is stored, so both surfaces can be large (tens of thousands of
    triangles each). Inputs: centres, unit normals [F, 3] / [G, 3] and masses [F] / [G].
    """
    from pykeops.torch import LazyTensor

    def inner(c1, n1, m1, c2, n2, m2, width):
        k = (-(LazyTensor(c1[:, None, :]) - LazyTensor(c2[None, :, :])).sqnorm2() / width ** 2).exp()
        k = k * (LazyTensor(n1[:, None, :]) | LazyTensor(n2[None, :, :])) ** 2
        return (m1[:, None] * (k * LazyTensor(m2[None, :, None])).sum(dim=1)).sum()

    total = 0.0
    for w in widths:
        oo = inner(co, no, mo, co, no, mo, w)
        total = total + (inner(cf, nf, mf, cf, nf, mf, w) - 2 * inner(cf, nf, mf, co, no, mo, w) + oo) / oo * w ** 2
    return total / len(widths)


def keops_varifold_setup(template_mm, template_faces, obs_triangles, n_faces, seed=0):
    """Fit side: a fixed, area-proportional subset of the template's triangles; observations: all of theirs."""
    gen = torch.Generator(device=template_mm.device).manual_seed(seed)
    tmpl_area = triangle_geometry(template_mm[template_faces])[2]
    fi = torch.multinomial(tmpl_area, min(n_faces, len(tmpl_area)), replacement=False, generator=gen)
    verts, local = torch.unique(template_faces[fi], return_inverse=True)
    return {"verts": verts, "local": local, "area_scale": tmpl_area.sum() / tmpl_area[fi].sum(),
            "obs": [triangle_geometry(t) for t in obs_triangles]}


def keops_varifold_step(vf, decoder, z, args, scale):
    """Per target: varifold data term * scale[t], backpropagated at once; returns the [T] distances."""
    from pykeops.torch import LazyTensor
    out = []
    for t in range(z.shape[0]):
        pts = decoder(vf["verts"][None], z[t:t + 1])[0]
        cf, nf, af = triangle_geometry(pts[vf["local"]])
        co, no, mo = vf["obs"][t]
        with torch.no_grad():
            d = (LazyTensor(cf[:, None, :]) - LazyTensor(co[None, :, :])).sqnorm2().min(dim=1).sqrt()[:, 0]
            w = torch.exp(-((d - args.trunc_mm).clamp_min(0) / args.trunc_mm) ** 2)
        dist = keops_varifold_one(cf, nf, af * vf["area_scale"] * w, co, no, mo, args.varifold_sigmas)
        (dist * scale[t]).backward()
        out.append(dist.detach())
    return torch.stack(out)


def load_run_config(run):
    """The ExperimentCfg a run was trained with (its config.json)."""
    data = load_yaml(os.path.join(run, "config.json"))
    data.pop("mode", None)
    return ExperimentCfg.from_dict(data)


def code_parametrization(kind, train_cfg, train_codes, template_norm, domain, args, device):
    """(wrap, theta mean [P], prior precision [P, P], label): how the fitted code z is parametrised.

    gauss: theta = z, prior N(mean, covariance) of the training codes (needs fewer code
        entries than training codes, else the covariance is singular).
    pca: theta = s, z = mean + U sqrt(lambda) s over the principal components of the training
        codes holding --pca-var of their variance; prior 1/2 |s|^2. Statistical only.
    pca_residual: theta = (s, r), z = mean + U sqrt(lambda) s + sqrt(v) r, v the training codes'
        variance per entry (so s and r are both in sigma units); prior 1/2 |s|^2 +
        1/2 sum_e w_e r_e^2. On a direct
        grid, w_e comes from each node's distance to the template surface, in grid cells
        (as code.grid_reg kind distance, independent of how the run was trained): 0 up to
        --residual-near, 1 from --residual-far on, linear in between, mapped onto
        --residual-w LOW HIGH. A low w lets the data deform that node's region beyond the
        statistical part, a high w holds it at the statistical part.
    wrap(decoder) turns a decoder of z into a decoder of theta.
    """
    z = train_codes.double().numpy()
    mean = z.mean(0)
    if kind == "gauss":
        prec = np.linalg.inv(np.cov(z, rowvar=False))
        return (lambda dec: dec), torch.as_tensor(mean, dtype=torch.float32, device=device), \
            torch.as_tensor(prec, dtype=torch.float32, device=device), "gauss"
    _, sv, vt = np.linalg.svd(z - mean, full_matrices=False)
    var = sv ** 2 / (len(z) - 1)
    k = int(np.searchsorted(np.cumsum(var) / var.sum(), args.pca_var) + 1)
    basis = torch.as_tensor((vt[:k].T * np.sqrt(var[:k])), dtype=torch.float32, device=device)   # [D, K]
    mean_t = torch.as_tensor(mean, dtype=torch.float32, device=device)
    d = z.shape[1]
    if kind == "pca":
        def wrap(dec):
            return lambda idx, theta: dec(idx, mean_t + theta @ basis.T)
        return wrap, torch.zeros(k, device=device), torch.eye(k, device=device), f"pca{k}"
    if kind != "pca_residual":
        raise ValueError(f"unknown --code-prior {kind!r}")
    import dataclasses
    from src.models.lddmm.resnet_flow.conditioning.grid_regularization import grid_penalty_weights
    from src.models.lddmm.resnet_flow.config import GridRegCfg
    if not train_cfg.code.grid_direct:
        raise ValueError("--code-prior pca_residual needs a direct-grid run (code.grid_direct)")
    t = template_norm.points[0].to(device)
    spatial = dataclasses.replace(train_cfg.code, grid_reg=GridRegCfg.from_dict(
        {"kind": "distance", "low": 0.0, "high": 1.0, "near": args.residual_near, "far": args.residual_far}))
    reg = grid_penalty_weights(spatial, [t], t, None, domain)[0].to(device).float()   # [D], 0..1
    low, high = args.residual_w
    w = low + (high - low) * reg
    v = torch.as_tensor(z.var(0), dtype=torch.float32, device=device).clamp_min(1e-12)
    prec = torch.zeros(k + d, k + d, device=device)
    prec[:k, :k] = torch.eye(k, device=device)
    prec[k:, k:] = torch.diag(w)
    sd = v.sqrt()

    def wrap(dec):
        return lambda idx, theta: dec(idx, mean_t + theta[:, :k] @ basis.T + theta[:, k:] * sd)
    return wrap, torch.zeros(k + d, device=device), prec, f"pca{k}_residual"


def fit_codes(decoder, n_template, targets, candidates, mu, s_inv, args, device, lr=None,
              template_mm=None, template_faces=None, obs_triangles=None, free=None):
    """{variant: [T, D] codes} minimising E(z) for each target, the decoder fixed.

    Args:
        decoder: decoder(idx [T, M], z [T, D]) -> [T, M, 3] mm, the shape's template
            vertices idx for code z (flow_decoder: a frozen flow; ssm_decoder: a linear SSM)
        n_template: number of template vertices
        targets: per target, its observations [N_i, 3] in mm, in the unposed model frame
        candidates: per target, the indices into it to draw observations from (for --data
            l2 also the template vertices they correspond to)
        mu, s_inv: the prior's mean and inverse covariance (the training codes', or 0 and I
            for SSM coefficients in sigma units)
        args: --variants, --steps, --lr, --points, --data, --fit-points, --trunc-mm
        lr: Adam step size, default args.lr (the codes' scale; SSM sigmas need a larger one)
        template_mm: [N, 3] the template (model mean) in mm, unposed (--data varifold)
        template_faces: [F, 3] long, the template's triangles (--data varifold)
        obs_triangles: per target, its observed triangles [G_i, 3, 3] in mm, unposed (--data varifold)
        free: flow parameters fitted with the codes, shared by the targets: an object with
            .params, .lr, .reset() (at each variant's start), .penalty(z, data_weight, gen)
            (added to the energy) and .save(variant) (at its end); see
            code_fit_fused_targets.FreeNorms
    """
    if free is not None and args.data == "varifold" and args.keops:
        raise ValueError("free flow parameters are not supported with --keops")
    n_obs = torch.tensor([float(len(i)) for i in candidates], device=device)
    gen = torch.Generator(device=device).manual_seed(0)
    if args.data == "varifold":
        vf = (keops_varifold_setup if args.keops else varifold_setup)(template_mm, template_faces, obs_triangles,
                                                                        args.fit_faces)
    codes = {}
    for v in args.variants:
        sigma, weight = parse_variant(v)
        z = mu.expand(len(targets), -1).clone().requires_grad_(True)
        groups = [{"params": [z]}]
        if free is not None:
            free.reset()
            groups.append({"params": free.params, "lr": free.lr})
        opt = torch.optim.Adam(groups, lr=lr or args.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps) if args.lr_decay else None
        for step in range(args.steps):
            picks = [i[torch.randint(len(i), (args.points,), device=device, generator=gen)] for i in candidates]
            y = torch.stack([t[p] for t, p in zip(targets, picks)])          # [T, M, 3] observations
            if args.data == "l2":
                # Corresponded: the template vertices the observations were taken from.
                pred_mm = decoder(torch.stack(picks), z)
                sq = ((pred_mm - y) ** 2).sum(-1).mean(1)
                data = sq * n_obs / (2 * sigma ** 2)
            elif args.data == "varifold" and args.keops:
                # Each target's data term is backpropagated inside, one target at a time.
                opt.zero_grad()
                sq = keops_varifold_step(vf, decoder, z, args, n_obs / (2 * sigma ** 2))
                data = None
            elif args.data == "varifold":
                sq = varifold_data(vf, decoder, z, args, gen)
                data = sq * n_obs / (2 * sigma ** 2)
            else:
                # Chamfer, no correspondence: observations -> fit, plus fit -> observations
                # truncated at --trunc-mm, so the free region is not pulled onto the partial data.
                fit_idx = torch.randint(n_template, (args.fit_points,), device=device, generator=gen)
                pred_mm = decoder(fit_idx[None].expand(len(targets), -1), z)
                dist = torch.cdist(y, pred_mm)                                   # [T, M, P]
                sq = dist.min(2).values.pow(2).mean(1)
                back = dist.min(1).values.pow(2).clamp(max=args.trunc_mm ** 2).mean(1)
                data = (sq + back) * n_obs / (2 * sigma ** 2)
            d = z - mu
            prior = weight * 0.5 * torch.einsum("ji,ik,jk->j", d, s_inv, d)
            if data is None:                     # data gradients are already in z.grad
                prior.sum().backward()
            else:
                opt.zero_grad()
                loss = (data + prior).sum()
                if free is not None:
                    loss = loss + free.penalty(z, (n_obs / (2 * sigma ** 2)).mean(), gen)
                loss.backward()
            opt.step()
            if sched is not None:
                sched.step()
            if step % 250 == 0 or step == args.steps - 1:
                print(f"{v} step {step:5d}  rms obs -> fit (mm) " + " ".join(f"{np.sqrt(s):.2f}" for s in sq.tolist()))
        codes[v] = z.detach()
        if free is not None:
            free.save(v)
    return codes


def parse_variant(name):
    """'data' -> (sigma 1 mm, prior weight 0); 'gauss<sigma>' -> (sigma, 1)."""
    if name == "data":
        return 1.0, 0.0
    if name.startswith("gauss"):
        return float(name[len("gauss"):]), 1.0
    raise ValueError(f"unknown variant {name!r}: 'data' or 'gauss<sigma_mm>'")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run", required=True, help="the SSM-trained cohort run")
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--tests-dir", default="outputs/posterior_SSM_tests")
    parser.add_argument("--model", default="assets/shape_models/baby/baby_pca.npz")
    parser.add_argument("--seed", type=int, default=0, help="the seed posterior_ssm_tests.py ran with")
    parser.add_argument("--noise-mm", type=float, default=0.3, help="the noise posterior_ssm_tests.py ran with")
    parser.add_argument("--variants", nargs="+", default=["data", "gauss0.5", "gauss3"])
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--points", type=int, default=2000, help="observed vertices per code and step")
    parser.add_argument("--data", choices=("chamfer", "varifold", "l2"), default="chamfer",
                        help="chamfer or varifold (no correspondence), or l2 (the observations' own vertex indices)")
    parser.add_argument("--fit-faces", type=int, default=2000, help="varifold: fixed triangle subset per surface")
    parser.add_argument("--varifold-sigmas", type=float, nargs="+", default=[3.0, 6.0, 12.0],
                        help="varifold: kernel widths (mm)")
    parser.add_argument("--lr-decay", action="store_true", help="cosine decay of the step size to 0 over --steps")
    parser.add_argument("--code-prior", choices=("gauss", "pca", "pca_residual"), default="gauss",
                        help="code parametrisation and prior, see code_parametrization")
    parser.add_argument("--pca-var", type=float, default=0.99, help="pca*: variance share of the kept components")
    parser.add_argument("--theta-lr", type=float, default=0.05, help="pca*: Adam step size (parameters in sigma units)")
    parser.add_argument("--residual-w", type=float, nargs=2, default=[1.0, 100.0],
                        help="pca_residual: residual weight near the face and far from it")
    parser.add_argument("--residual-near", type=float, default=0.5,
                        help="pca_residual: grid cells from the template up to which a node gets the low weight")
    parser.add_argument("--residual-far", type=float, default=1.0,
                        help="pca_residual: grid cells from the template from which a node gets the high weight")
    parser.add_argument("--keops", action="store_true",
                        help="varifold with KeOps: all observed triangles, --fit-faces fixed template triangles")
    parser.add_argument("--smooth-obs", type=int, default=0,
                        help="varifold: Laplacian steps on the observed vertices before their triangles are taken")
    parser.add_argument("--fit-points", type=int, default=6000, help="chamfer: template vertices flowed per step")
    parser.add_argument("--trunc-mm", type=float, default=2.0, help="chamfer: cap of the fit -> observation distance")
    parser.add_argument("--no-ssm", dest="ssm", action="store_false",
                        help="skip the SSM fit under the same energy (default: fit it too)")
    parser.add_argument("--ssm-lr", type=float, default=0.05, help="Adam step size of the SSM coefficients (sigma units)")
    parser.add_argument("--output", default=None, help="default: <tests-dir>/code_fit_<run name>_<data>")
    parser.add_argument("--panel-px", type=int, default=300)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)
    out_dir = args.output or os.path.join(args.tests_dir, f"code_fit_{os.path.basename(os.path.normpath(args.run))}_{args.data}"
                                          + ("" if args.code_prior == "gauss" else f"_{args.code_prior}"))
    os.makedirs(out_dir, exist_ok=True)

    model = load_model_file(args.model)
    faces = model.faces
    tests = redraw_tests(model, args.seed, args.noise_mm, args.tests_dir)
    _, flow, train_codes, _, template_norm, transform = load_run(args.run, args.step, device)
    for p in flow.parameters():
        p.requires_grad_(False)
    z_train = train_codes.double().numpy()
    train_cfg = load_run_config(args.run)
    wrap, mu, s_inv, prior_label = code_parametrization(args.code_prior, train_cfg, train_codes, template_norm,
                                                        _normalization_domain(train_cfg), args, device)
    print(f"code prior: {prior_label}")
    center, scale = transform.center.to(device), transform.scale.to(device)
    template = template_norm.points[0].to(device)                       # [N, 3] normalised

    # Targets in the unposed model frame (mm). One code per test, the tests of a variant optimised
    # together, the variants one after the other (a chamfer step flows many template vertices).
    targets, obs_idx = [], []
    for test, posed, obs, mask, matrix, _ in tests:
        inv = np.linalg.inv(matrix)
        targets.append(torch.as_tensor(obs @ inv[:3, :3].T + inv[:3, 3], dtype=torch.float32, device=device))
        obs_idx.append(torch.as_tensor(np.flatnonzero(mask), device=device))
    # Varifold: the observed surface is the template triangles with all three vertices observed.
    vf_inputs = {}
    if args.data == "varifold":
        faces_t = torch.as_tensor(faces, dtype=torch.long, device=device)
        obs_tris = []
        for (test, posed, obs, mask, matrix, _), target in zip(tests, targets):
            keep = torch.as_tensor(mask[faces].all(1), device=device)
            obs_tris.append(laplacian_smooth(target, faces_t[keep], args.smooth_obs)[faces_t[keep]])
        vf_inputs = {"template_mm": torch.as_tensor(model.mean.reshape(-1, 3) * MM_PER_UNIT, dtype=torch.float32,
                                                    device=device),
                     "template_faces": faces_t, "obs_triangles": obs_tris}
    # The flow's code, and (--ssm) the SSM's coefficients under the same energy, data term and pose.
    n_vert = len(template)
    methods = {"code fit": (wrap(flow_decoder(flow, template, center, scale, MM_PER_UNIT)), mu, s_inv,
                             args.lr if args.code_prior == "gauss" else args.theta_lr)}
    if args.ssm:
        k = model.n_components
        methods["SSM fit"] = (ssm_decoder(model, MM_PER_UNIT, device), torch.zeros(k, device=device),
                              torch.eye(k, device=device), args.ssm_lr)
    fits = {}                                                            # (method, variant) -> [T, N, 3] mm, d2 [T]
    for method, (decoder, m, si, lr) in methods.items():
        print(f"== {method}")
        codes = fit_codes(decoder, n_vert, targets, obs_idx, m, si, args, device, lr=lr, **vf_inputs)
        with torch.no_grad():
            for v, z in codes.items():
                full = torch.arange(n_vert, device=device)[None].expand(len(tests), -1)
                fits[(method, v)] = (decoder(full, z).cpu().numpy(),
                                     torch.einsum("ji,ik,jk->j", z - m, si, z - m).cpu().numpy())

    # Full-mesh evaluation, posed with the true pose.
    summary = {"run": args.run, "step": args.step, "tests_dir": args.tests_dir, "variants": args.variants,
               "steps": args.steps, "data": args.data, "trunc_mm": args.trunc_mm,
               "pose": "true pose for the code and SSM fits (the posterior estimates its own)",
               "posterior": "ssm_posterior stage: corresponded observations", "tests": {}}
    posterior_summary = json.load(open(os.path.join(args.tests_dir, "summary.json")))["tests"]
    for ti, (test, posed, obs, mask, matrix, lmk) in enumerate(tests):
        entry = {"n_observed": int(mask.sum()), "posterior": {
            s: {k: f[k] for k in ("error_observed_median_mm", "error_free_median_mm", "error_free_p95_mm",
                                  "mahalanobis_sq")} for s, f in posterior_summary[test]["fits"].items()}}
        polys = {}
        for (method, v), (pts, d2) in fits.items():
            fit = pts[ti] @ matrix[:3, :3].T + matrix[:3, 3]
            err = np.linalg.norm(fit - posed, axis=1)
            entry.setdefault(method, {})[v] = {"error_observed_median_mm": float(np.median(err[mask])),
                                               "error_free_median_mm": float(np.median(err[~mask])),
                                               "error_free_p95_mm": float(np.percentile(err[~mask], 95)),
                                               "mahalanobis_sq": float(d2[ti])}
            polys[f"{method} {v}"] = (create_polydata(fit, faces), err)
        summary["tests"][test] = entry
        print(f"{test} ({mask.sum()} obs):")
        for method in list(methods) + ["posterior"]:
            print(f"    {method:<10}" + "  ".join(
                f"{v}: {r['error_observed_median_mm']:.2f}/{r['error_free_median_mm']:.2f}/"
                f"{r['error_free_p95_mm']:.2f} d2 {r['mahalanobis_sq']:.0f}" for v, r in entry[method].items()))

        # Views: truth | observations | code fits | SSM fits | posterior (largest sigma); then error maps.
        truth_mesh = create_polydata(posed, faces)
        observed_mesh = sub_surface(load_vtp(os.path.join(args.tests_dir, test, "observed.vtp")), mask)
        post_dir = sorted(d for d in os.listdir(os.path.join(args.tests_dir, test)) if d.startswith("posterior_"))
        post_sigmas = {float(d.rsplit("sigma", 1)[1].rstrip("mm")): d for d in post_dir}
        s_max = max(post_sigmas)
        post = load_vtp(os.path.join(args.tests_dir, test, post_sigmas[s_max], "posterior_mean.vtp"))
        post_err = np.linalg.norm(vtk_to_numpy(post.GetPoints().GetData()).astype(float) - posed, axis=1)
        panels = [("Ground truth", truth_mesh), (f"Observations {mask.sum()}/{len(mask)}", observed_mesh)]
        panels += [(label, poly) for label, (poly, _) in polys.items()]
        panels += [(f"SSM posterior (corresp.) s{s_max:g}", post)]
        cams = {view: views.view_camera(lmk, [truth_mesh], view) for view in ("frontal", "left", "right")}
        rows = [(cams[view], [(t + ("" if view == "frontal" else f" ({view})"), p,
                               views.SCAN_COLOR if t.startswith("Obs") else views.MESH_COLOR, [])
                              for t, p in panels]) for view in cams]
        rgb = np.where(mask[:, None], views.SCAN_COLOR, views.MESH_COLOR) * 255
        err_row = [("Ground truth: observed (blue) / free", with_colors(truth_mesh, rgb.round().astype(np.uint8)), None, []), None]
        err_row += [(f"{label}: free med {np.median(e[~mask]):.2f} mm", with_colors(p, ramp_colors(e, ERROR_MAX_MM)), None, [])
                    for label, (p, e) in polys.items()]
        err_row += [(f"posterior: free med {np.median(post_err[~mask]):.2f} mm",
                     with_colors(post, ramp_colors(post_err, ERROR_MAX_MM)), None, [])]
        rows.append((cams["frontal"], err_row))
        header = (f"{test}: {args.data} fits, true pose: code of the frozen flow "
                  f"({os.path.basename(os.path.normpath(args.run))}) and SSM coefficients; SSM posterior on corresponded "
                  f"observations for reference; error light 0 -> dark {ERROR_MAX_MM:g} mm")
        views.write_grid(rows, len(panels), header, os.path.join(out_dir, f"{test}.png"), args.panel_px)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
