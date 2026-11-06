"""Fit the code of a frozen shape model to observations: Adam over the code, the decoder fixed.

For each target the energy is

    E(theta) = data(decoder(theta), observations) * n_obs / (2 sigma^2)
               + w * 1/2 (theta - mean)^T precision (theta - mean)

with n_obs the target's number of observation points, so sigma (mm) weighs the data
against the prior, and w switches the prior on (1) or off (0). The data terms are in
data_terms.py. Moved from scripts/studies/code_fit_partial_tests.py, whose
fit_codes now calls this one; the code_fit stage (stage.py) fits one target per run.
"""

import numpy as np
import torch

from src.shared.fitting.data_terms import (
    keops_varifold_setup,
    keops_varifold_step,
    varifold_data,
    varifold_setup,
)

DATA_TERMS = ("chamfer", "l2", "varifold")


def parse_variant(name):
    """'data' -> (sigma 1 mm, prior weight 0); 'gauss<sigma>' -> (sigma, 1)."""
    if name == "data":
        return 1.0, 0.0
    if name.startswith("gauss"):
        return float(name[len("gauss"):]), 1.0
    raise ValueError(f"unknown variant {name!r}: 'data' or 'gauss<sigma_mm>'")


def fit_codes(decoder, n_template, targets, candidates, mu, s_inv, variants, steps, lr, points, device,
              data="chamfer", fit_points=6000, trunc_mm=2.0, lr_decay=False, template_mm=None,
              template_faces=None, obs_triangles=None, fit_faces=2000, varifold_sigmas=(3.0, 6.0, 12.0),
              keops=False, free=None, seed=0, log_every=250):
    """{variant: [T, D] codes} minimising E for each target, the decoder fixed.

    Args:
        decoder: decoder(idx [T, M], z [T, D]) -> [T, M, 3] mm, the shape's template
            vertices idx for code z (a frozen flow, a linear SSM, a ShapeModel)
        n_template: number of template vertices
        targets: per target, its observations [N_i, 3] in mm, in the unposed model frame
        candidates: per target, the indices into it to draw observations from (for l2
            also the template vertices they correspond to)
        mu, s_inv: the prior's mean and precision; the fit starts at mu
        variants: {label: (sigma_mm, prior weight)}, one fit each (see parse_variant)
        steps, lr: Adam steps and step size; lr_decay: cosine decay to 0 over the steps
        points: observations drawn per target and step
        data: one of DATA_TERMS
        fit_points: chamfer: template vertices decoded per step
        trunc_mm: chamfer: cap of the fit -> observation distance; varifold: the mask's width
        template_mm, template_faces, obs_triangles: varifold: the template (unposed, mm), its
            triangles [F, 3] long, and per target its observed triangles [G_i, 3, 3]
        fit_faces: varifold: the fixed triangle subset per surface
        varifold_sigmas: varifold: the kernel widths (mm)
        keops: varifold with KeOps, all observed triangles
        free: parameters fitted with the codes, shared by the targets: an object with
            .params, .lr, .reset() (at each variant's start), .penalty(z, data_weight, gen)
            (added to the energy) and .save(variant) (at its end); see
            scripts/studies/code_fit_fused_targets.FreeNorms
        seed: of the draws of observations and template vertices
        log_every: steps between two printed rms distances
    """
    if data not in DATA_TERMS:
        raise ValueError(f"unknown data term {data!r}; known: {list(DATA_TERMS)}")
    if free is not None and data == "varifold" and keops:
        raise ValueError("free flow parameters are not supported with keops")
    n_obs = torch.tensor([float(len(i)) for i in candidates], device=device)
    gen = torch.Generator(device=device).manual_seed(seed)
    if data == "varifold":
        vf = (keops_varifold_setup if keops else varifold_setup)(template_mm, template_faces, obs_triangles, fit_faces)
    codes = {}
    for v, (sigma, weight) in variants.items():
        z = mu.expand(len(targets), -1).clone().requires_grad_(True)
        groups = [{"params": [z]}]
        if free is not None:
            free.reset()
            groups.append({"params": free.params, "lr": free.lr})
        opt = torch.optim.Adam(groups, lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps) if lr_decay else None
        for step in range(steps):
            picks = [i[torch.randint(len(i), (points,), device=device, generator=gen)] for i in candidates]
            y = torch.stack([t[p] for t, p in zip(targets, picks)])          # [T, M, 3] observations
            if data == "l2":
                # Corresponded: the template vertices the observations were taken from.
                pred_mm = decoder(torch.stack(picks), z)
                sq = ((pred_mm - y) ** 2).sum(-1).mean(1)
                data_term = sq * n_obs / (2 * sigma ** 2)
            elif data == "varifold" and keops:
                # Each target's data term is backpropagated inside, one target at a time.
                opt.zero_grad()
                sq = keops_varifold_step(vf, decoder, z, trunc_mm, varifold_sigmas, n_obs / (2 * sigma ** 2))
                data_term = None
            elif data == "varifold":
                sq = varifold_data(vf, decoder, z, trunc_mm, varifold_sigmas)
                data_term = sq * n_obs / (2 * sigma ** 2)
            else:
                # Chamfer, no correspondence: observations -> fit, plus fit -> observations
                # truncated at trunc_mm, so the free region is not pulled onto the partial data.
                fit_idx = torch.randint(n_template, (fit_points,), device=device, generator=gen)
                pred_mm = decoder(fit_idx[None].expand(len(targets), -1), z)
                dist = torch.cdist(y, pred_mm)                                   # [T, M, P]
                sq = dist.min(2).values.pow(2).mean(1)
                back = dist.min(1).values.pow(2).clamp(max=trunc_mm ** 2).mean(1)
                data_term = (sq + back) * n_obs / (2 * sigma ** 2)
            d = z - mu
            prior = weight * 0.5 * torch.einsum("ji,ik,jk->j", d, s_inv, d)
            if data_term is None:                # data gradients are already in z.grad
                prior.sum().backward()
            else:
                opt.zero_grad()
                loss = (data_term + prior).sum()
                if free is not None:
                    loss = loss + free.penalty(z, (n_obs / (2 * sigma ** 2)).mean(), gen)
                loss.backward()
            opt.step()
            if sched is not None:
                sched.step()
            if step % log_every == 0 or step == steps - 1:
                print(f"{v} step {step:5d}  rms obs -> fit (mm) " + " ".join(f"{np.sqrt(s):.2f}" for s in sq.tolist()))
        codes[v] = z.detach()
        if free is not None:
            free.save(v)
    return codes
