"""The data terms of a code fit (code_fit.py): how far a decoded shape is from its observations.

    chamfer   observations -> fit, quadratic, plus fit -> observations truncated at trunc_mm,
              so the surface the observations do not cover is not pulled onto them
    l2        corresponded: the template vertices the observations were taken from
    varifold  unoriented varifold of fixed triangle subsets at several widths (mm), the fit's
              triangles weighted down beyond trunc_mm from the observations; also with KeOps

All in mm^2, in the unposed model frame scaled to mm. Moved from
scripts/studies/code_fit_partial_tests.py.
"""

import torch


def triangle_geometry(tris):
    """[..., F, 3, 3] triangle vertices -> (centres, unit normals, areas), as fshape.face_geometry."""
    v0, v1, v2 = tris.unbind(-2)
    cross = torch.cross(v1 - v0, v2 - v0, dim=-1)
    norm = cross.norm(dim=-1, keepdim=True)
    return (v0 + v1 + v2) / 3.0, cross / norm.clamp_min(1e-12), 0.5 * norm[..., 0]


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


def varifold_data(vf, decoder, z, trunc_mm, widths):
    """[T] varifold distance of each target's fit to its observations, in mm^2.

    Unoriented varifold (fshape.py's kernel without a signal) at the ``widths``, each
    normalised by <obs, obs> and multiplied by its width^2, so a small misfit d gives
    about d^2. Both surfaces are the fixed subsets of varifold_setup, so the term is a
    smooth function of the fit's vertex positions. Each fit triangle is weighted by its
    distance d to the observed triangles, held fixed within the step (a mask): 1 up to
    trunc_mm, then exp(-((d - trunc) / trunc)^2), so the surface the observations do not
    cover is not pulled onto them - the varifold's counterpart of the truncated chamfer.
    """
    n_t = z.shape[0]
    pts = decoder(vf["verts"][None].expand(n_t, -1), z)                 # [T, U, 3]
    cf, nf, af = triangle_geometry(pts[:, vf["local"]])                  # [T, F, ...]
    mf = af * vf["area_scale"]
    co, no, mo = (torch.stack(x) for x in zip(*vf["obs"]))               # [T, G, ...]
    with torch.no_grad():
        d = torch.cdist(cf, co).min(2).values
        w = torch.exp(-((d - trunc_mm).clamp_min(0) / trunc_mm) ** 2)
    mf = mf * w
    sq_ff, sq_fo, sq_oo = torch.cdist(cf, cf) ** 2, torch.cdist(cf, co) ** 2, torch.cdist(co, co) ** 2
    n_ff, n_fo, n_oo = (nf @ nf.transpose(1, 2)) ** 2, (nf @ no.transpose(1, 2)) ** 2, (no @ no.transpose(1, 2)) ** 2
    total = 0.0
    for width in widths:
        def inner(m1, sq, nn, m2):
            return torch.einsum("tf,tfg,tg->t", m1, torch.exp(-sq / width ** 2) * nn, m2)
        oo = inner(mo, sq_oo, n_oo, mo)
        total = total + (inner(mf, sq_ff, n_ff, mf) - 2 * inner(mf, sq_fo, n_fo, mo) + oo) / oo * width ** 2
    return total / len(widths)


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


def keops_varifold_step(vf, decoder, z, trunc_mm, widths, scale):
    """Per target: varifold data term * scale[t], backpropagated at once; returns the [T] distances."""
    from pykeops.torch import LazyTensor
    out = []
    for t in range(z.shape[0]):
        pts = decoder(vf["verts"][None], z[t:t + 1])[0]
        cf, nf, af = triangle_geometry(pts[vf["local"]])
        co, no, mo = vf["obs"][t]
        with torch.no_grad():
            d = (LazyTensor(cf[:, None, :]) - LazyTensor(co[None, :, :])).sqnorm2().min(dim=1).sqrt()[:, 0]
            w = torch.exp(-((d - trunc_mm).clamp_min(0) / trunc_mm) ** 2)
        dist = keops_varifold_one(cf, nf, af * vf["area_scale"] * w, co, no, mo, widths)
        (dist * scale[t]).backward()
        out.append(dist.detach())
    return torch.stack(out)
