"""Unsupervised point clustering from point-data FEATURE fields already
attached to a mesh (e.g. echotexture's echo_* fields) - an exploratory first
pass at separating tissue types (face, placenta, hand/limb, noise, ...) with
NO labeled training data. This clusters rather than classifies: a cluster id
is data-driven and arbitrary (not tied to a fixed tissue vocabulary, and not
guaranteed stable across reruns unless `random_state` is fixed) - it has to
be interpreted after the fact, e.g. by comparing a cluster's per-feature
means (see `describe_clusters`) against what's expected for a known tissue,
or by visual inspection.

Pure numpy/sklearn - no VTK here, so it's testable without building meshes.
"""

import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import StandardScaler


def fit_predict_clusters(X, n_clusters, algorithm="minibatch_kmeans", random_state=0,
                         n_init=10, batch_size=10_000):
    """Standardizes `X` (each feature independently, across ALL rows) then
    fits one clustering model, returning `(labels [N], centers [n_clusters, F],
    scaler)`.

    `centers` is in STANDARDIZED units; pass through `scaler.inverse_transform`
    to read them back in each feature's own original units (see
    `describe_clusters`).

    Fit ONE model over every subject's points POOLED together (concatenate
    before calling this) rather than one model per subject - fitting
    independently per subject gives arbitrary, incomparable label ids from
    one mesh to the next (cluster "0" in subject A has no reason to mean the
    same thing as cluster "0" in subject B).

    `algorithm="minibatch_kmeans"` (default) trades a little cluster quality
    for much better scaling to the millions of pooled points a multi-subject
    ultrasound mesh ensemble can reach; `"kmeans"` runs full Lloyd's KMeans
    for a smaller/more precise run.
    """
    X = np.asarray(X, dtype=np.float64)
    if not np.all(np.isfinite(X)):
        raise ValueError(
            "Feature matrix contains NaN/Inf - check that every input mesh actually "
            "carries every requested feature field (e.g. a merge stage that only "
            "backfills common fields across subjects can introduce NaNs)."
        )

    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    if algorithm == "kmeans":
        model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init)
    elif algorithm == "minibatch_kmeans":
        model = MiniBatchKMeans(
            n_clusters=n_clusters, random_state=random_state, n_init=n_init, batch_size=batch_size
        )
    else:
        raise ValueError(f"Unknown algorithm {algorithm!r}; expected 'kmeans' or 'minibatch_kmeans'.")

    labels = model.fit_predict(X_std)
    return labels, model.cluster_centers_, scaler


def describe_clusters(labels, centers_std, scaler, feature_names):
    """Returns a list of one human-readable summary line per cluster: its
    size (count and share of all points) and its per-feature mean, converted
    back to each feature's OWN original units - the quickest way to guess
    which cluster is which real tissue (e.g. high echo_planarity + mid
    echo_intensity ~ skin/face; high echo_linearity ~ cord/limb; high
    echo_sphericity + low echo_energy ~ noise or a flat, featureless patch)."""
    centers_original = scaler.inverse_transform(centers_std)
    n_pts = len(labels)

    lines = []
    for k in range(centers_std.shape[0]):
        size = int((labels == k).sum())
        means = ", ".join(f"{name}={val:.3g}" for name, val in zip(feature_names, centers_original[k]))
        share = size / n_pts if n_pts else 0.0
        lines.append(f"cluster {k}: {size:,} points ({share:.1%}) - {means}")
    return lines
