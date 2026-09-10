# GPMM build configs

One matched family of model builds per domain: `baby/` and `fetus/` hold the
same thirteen kernels, and only the source and the hyperparameters differ. Build
any of them with

```
python -m scripts.SSM.build_model configs/ssm/build_configs/fetus/pca_times_matern52.yaml
```

Models land in `src/SSM/saved_models/<domain>/<domain>_<name>.npz`.

## Naming

A file is named after its kernel tree, read **left to right**, with `sum` for
`+` and `times` for `*`:

| file                               | kernel                |
| ---------------------------------- | --------------------- |
| `pca`                              | `pca`                 |
| `matern52`                         | `matern52` (prior only, no empirical term) |
| `pca_sum_matern52`                 | `pca + matern52`      |
| `pca_times_matern52`               | `pca * matern52`      |
| `pca_sum_gaussian_times_gaussian`  | `(pca + gaussian) * gaussian` |
| `pca_times_gaussian_sum_gaussian`  | `(pca * gaussian) + gaussian` |

## Why the numbers differ between the folders

The two models are not in the same units - see the "Units" note in
`src/SSM/kernels/stationary.py`.

|                                  | `baby/`                            | `fetus/`                        |
| -------------------------------- | ---------------------------------- | ------------------------------- |
| source                           | `src/SSM/BabyFM/BabyFaceModel.mat` | `src/SSM/FetusModel_x4` (atlas) |
| vertices / PCA modes             | 31027 / 163                        | 5743 / 183 (184 subjects)       |
| bounding-box diagonal            | 0.0199                             | 101.1 (mm)                      |
| mean edge length                 | 9.65e-5                            | 1.05                            |
| mean per-coordinate PCA variance | 4.16e-8                            | 1.449                           |
| per-vertex RMS deviation         | 3.34e-4 (1.68% of diagonal)        | 1.93 (1.91% of diagonal)        |

The two populations are almost equally variable relative to their own size
(1.68% vs 1.91%), so a hyperparameter transfers by pure rescaling: **lengths by
5077** (the diagonal ratio), **variances by 3.48e7** (the PCA-variance ratio,
consistent with 5077^2 up to the shape difference).

## The two length scales

Both are quoted as a fraction of the domain's bounding-box diagonal, so the same
choice means the same thing in both folders:

| role                                                    | fraction | baby  | fetus |
| ------------------------------------------------------- | -------- | ----- | ----- |
| **local** - the envelope in a `pca_times_*`, i.e. how far a localized PCA mode reaches | 15%      | 0.003 | 15.0  |
| **smooth** - the extra term in a `pca_sum_*`, i.e. deformation outside the PCA subspace | 25%      | 0.005 | 25.0  |

The local scale is the one to tune first. Its floor for the fetus is the atlas's
own deformation-kernel width: `params.json` gives `kwd: 7.0`, and Deformetrica's
`exp(-d^2/sigma^2)` convention makes that `7/sqrt(2) = 4.95` in this codebase's
`exp(-d^2/2 sigma^2)` convention. Below roughly that there is no PCA detail left
to localize.

### Matching a length scale across kernel families

`gaussian(sigma=s)`, `matern52(rho=s)` and `wendland(radius=s)` are *not* the
same width - the parameters mean different things. Comparing `pca_times_gaussian`
against `pca_times_matern52` is only meaningful if the two reach the same
distance, so every config's length scale is set from its **practical range**: the
distance at which correlation falls to 0.05.

| kernel     | practical range | to match `gaussian(sigma)` |
| ---------- | --------------- | -------------------------- |
| `gaussian` | 2.4477 sigma    | -                          |
| `matern12` | 2.9957 rho      | `rho = 0.8171 sigma`       |
| `matern32` | 2.7389 rho      | `rho = 0.8937 sigma`       |
| `matern52` | 2.6469 rho      | `rho = 0.9248 sigma`       |
| `wendland` | 1.0000 radius (hard cutoff) | `radius = 2.4477 sigma` |

Applied to the two scales above:

|              | gaussian | matern32 | matern52 | wendland |
| ------------ | -------- | -------- | -------- | -------- |
| baby local   | 0.003    | 0.00268  | 0.00277  | 0.00734  |
| baby smooth  | 0.005    | 0.00447  | 0.00462  | 0.01224  |
| fetus local  | 15.0     | 13.4     | 13.9     | 36.7     |
| fetus smooth | 25.0     | 22.3     | 23.1     | 61.2     |

## The variance scale

`scale` means different things in the two combinators, which is the easiest
thing to get wrong:

* in a **product** the scalar kernel block-scales each 3x3 PCA block, so
  `scale: 1.0` is the neutral choice - it modulates correlation and leaves the
  per-vertex variance alone. A variance does not belong here.
* in a **sum** the kernel is lifted isotropically and its `scale` *is* a
  per-coordinate variance, in the domain's squared coordinate units.

Every sum term in this family is set to **1x the domain's own mean
per-coordinate PCA variance** (`4.16e-8` baby, `1.449` fetus), so the added
prior is comparable to the population rather than dominating it, and so the four
`pca_sum_*` models differ only in kernel smoothness. Raise it - 10x is the
permissive end - to let the prior lead instead.
