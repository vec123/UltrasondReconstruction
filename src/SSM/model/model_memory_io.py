"""Persistence and cache keys for a built GPMM.

Two related jobs: write a built model to one ``.npz``, and decide when a
stored one may be reused instead of rebuilt. See
``docs/gpmm_architecture.md`` section 12 - this module is that section, under
a name that says what it actually stores: the model's *memory* of a build, not
merely bytes on disk.

    from src.SSM.model.model_memory_io import ModelCacheSpec, save_model, load_model, cache_path

    spec = ModelCacheSpec(kernel_spec=resolved_kernel_spec,      # from builder.resolve_kernel_spec
                          approximation={"name": "pivoted_cholesky", "kwargs": {"max_rank": 300}},
                          truncation=Truncation(variance_threshold=0.99))
    path = cache_path(cache_dir, spec_hash(model.domain, spec.kernel_spec,
                                          spec.approximation, spec.truncation))
    save_model(path, model, spec)
    ...
    cached = load_model(path, expect=spec, domain=model.domain)   # None if stale or missing

### Why this exists - builds are expensive and specs are cheap to mistype

Measured: 3.2 s for a baby ``pca``, 14.7 s for ``pca + gaussian`` at rank 202,
~90 s to shoot the fetus population. A tuning session changes one ``sigma`` and
rebuilds; without a cache that is the whole loop, every time.

### The hard part is the key, not the format

The storage format follows the existing shooting cache
(``FetusModel_x4/ssm_shot_sigma7_tp11.npz``, holding ``mean``,
``eigenfunctions``, ``eigenvalues`` and a few scalars) - that convention is
kept, just extended with the kernel and domain that produced them. **Getting
the key wrong** is what gives the worst failure mode available: a stale model
that loads without complaint and is quietly answering the wrong question.
:func:`spec_hash` is everything that determines the result, and nothing that
does not:

    spec_hash = H( domain.hash()            # points, faces, metric, mass
                 , kernel_spec              # names + kwargs, recursively,
                                            # with every injected array
                                            # (pca basis, LB eigenpairs,
                                            # weight fields) reduced to its
                                            # own content digest
                 , approximation_spec       # method + kwargs + seed
                 , truncation )

Three decisions inside that, each recorded because it is the one that is easy
to get wrong by looking reasonable:

* **Hash the injected arrays' bytes, not their source path.** A path plus
  mtime goes stale in both directions - an edited file that keeps its
  timestamp, a touched file that did not change. The baby basis is 121 MB,
  which hashes in about a tenth of a second against a 3-15 s build; cheap
  enough that the safe option is also the fast one. See :func:`array_digest`.
* **Store the spec inside the file, not only in its name.** :func:`load_model`
  then *verifies* against a freshly stated spec rather than trusting the
  filename - a hash collision or a hand-renamed file is caught rather than
  silently accepted.
* **``inference`` is stored but excluded from the key.** Preparing
  ``kernel_regression`` costs seconds against a decomposition that can cost
  minutes, so a changed inference strategy does not need to invalidate the
  expensive part - :func:`load_model` rebuilds it fresh from whatever is on
  file, or from a differently-requested spec, either way inexpensively.

### What is deliberately not here

No implicit build-or-load. This module only ever reads and writes what is
already decided; :mod:`src.SSM.model.builder` is what decides whether to call
it (``cache_dir`` on :func:`~src.SSM.model.builder.build_model`). Keeping
``build_model`` pure otherwise - a spec and the files it names go in, a model
comes out - is what lets a hyperparameter search run it hundreds of times with
``cache_dir=None`` and never touch disk.

No entry cap enforced automatically. A hyperparameter search generates
hundreds of distinct specs in one run and would otherwise fill the disk with
models nobody asks for twice; call :func:`prune_cache` after a sweep, or pass
``cache_dir=None`` for a search that must not cache at all.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from src.SSM.kernels.base import Kernel, build_kernel
from src.SSM.model.continuous_inference import build_inference
from src.SSM.model.mean_domain import MeanDomain
from src.SSM.model.model import StatisticalShapeModel

#: Bumped when the on-disk layout changes incompatibly. A file from an older
#: version is treated as a cache miss (rebuilt and overwritten) rather than
#: read with a possibly wrong assumption about its fields.
CACHE_VERSION = 1


# --------------------------------------------------------------------------
# content hashing
# --------------------------------------------------------------------------

def array_digest(array) -> str:
    """Content hash of an array's bytes.

    What makes "the same population, unchanged" checkable regardless of what
    the file holding it is called or when it was last touched - see the
    module docstring's first bullet.
    """
    a = np.ascontiguousarray(np.asarray(array))
    return hashlib.blake2b(a.tobytes(), digest_size=16).hexdigest()


def _canonicalize(node):
    """A spec tree, JSON-safe: dicts get sorted keys, arrays become a digest.

    An array is replaced by ``{"__array__": digest, "shape": ..., "dtype":
    ...}`` rather than embedded literally - a 93081 x 163 basis has no
    business inside a JSON string, and the digest is exactly the fact that
    matters for "did the input to this kernel change".
    """
    if isinstance(node, dict):
        return {str(key): _canonicalize(value) for key, value in sorted(node.items())}
    if isinstance(node, (list, tuple)):
        return [_canonicalize(value) for value in node]
    if isinstance(node, np.ndarray):
        return {"__array__": array_digest(node), "shape": list(node.shape), "dtype": str(node.dtype)}
    if isinstance(node, Path):
        return str(node)
    if isinstance(node, Kernel):
        raise TypeError(
            "spec_hash takes a spec (name/kwargs/children), not a built Kernel - pass "
            "what src.SSM.model.builder.resolve_kernel_spec returns, before build_kernel"
        )
    if is_dataclass(node) and not isinstance(node, type):
        return _canonicalize(asdict(node))
    if node is None or isinstance(node, (str, int, float, bool)):
        return node
    raise TypeError(f"cannot canonicalise {type(node).__name__} into a cache key")


def _component_dict(spec) -> Dict[str, Any]:
    """``{"name": ..., "kwargs": {...}}`` from a string, a mapping, or None."""
    if spec is None:
        return {"name": None, "kwargs": {}}
    if isinstance(spec, str):
        return {"name": spec, "kwargs": {}}
    name = spec.get("name") if hasattr(spec, "get") else getattr(spec, "name", None)
    kwargs = spec.get("kwargs") if hasattr(spec, "get") else getattr(spec, "kwargs", None)
    return {"name": name, "kwargs": dict(kwargs or {})}


def _truncation_dict(spec) -> Dict[str, Any]:
    if spec is None:
        return {}
    if is_dataclass(spec) and not isinstance(spec, type):
        return asdict(spec)
    return dict(spec)


def _stable_json(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def spec_hash(domain: MeanDomain, kernel_spec, approximation=None, truncation=None) -> str:
    """Everything that determines the built model, folded into one key.

    ``kernel_spec`` is the *resolved* tree -
    :func:`src.SSM.model.builder.resolve_kernel_spec`'s output, with the
    ``pca``/``laplace_beltrami``/``localized`` arrays already present as
    ``np.ndarray`` values rather than as ``source:``/``weight_field:``
    references. Passing the unresolved spec would key the cache on "read the
    basis from this path" rather than on the basis itself, defeating the
    point of hashing bytes at all.
    """
    payload = {
        "version": CACHE_VERSION,
        "domain": domain.hash(),
        "kernel": _canonicalize(kernel_spec),
        "approximation": _canonicalize(_component_dict(approximation)),
        "truncation": _canonicalize(_truncation_dict(truncation)),
    }
    return hashlib.blake2b(_stable_json(payload).encode("utf-8"), digest_size=16).hexdigest()


def cache_path(root, key: str) -> Path:
    """Deterministic on-disk location for a cache key.

    Flat, not sharded into subdirectories - the cache is expected to hold
    hundreds of entries across a tuning session (see :func:`prune_cache`), not
    millions.
    """
    return Path(root) / f"{key}.npz"


# --------------------------------------------------------------------------
# extracting / reinstating the arrays inside a resolved kernel spec
# --------------------------------------------------------------------------

def _extract_arrays(node, arrays: Dict[str, np.ndarray], prefix: str = "kernel"):
    """A JSON-safe skeleton of ``node``, with every array pulled into ``arrays``.

    Each array is replaced in place by ``{"__ref__": key}``; the key is the
    dotted path to where it was found, which is unique because a spec tree has
    no repeated attachment points - a disambiguating suffix guards the one
    case that could collide (two kernels of the same shape at the same depth
    inside a list, before their own kwargs are added).
    """
    if isinstance(node, dict):
        return {key: _extract_arrays(value, arrays, f"{prefix}.{key}") for key, value in node.items()}
    if isinstance(node, list):
        return [_extract_arrays(value, arrays, f"{prefix}.{i}") for i, value in enumerate(node)]
    if isinstance(node, np.ndarray):
        key = prefix if prefix not in arrays else f"{prefix}#{len(arrays)}"
        arrays[key] = np.ascontiguousarray(node)
        return {"__ref__": key}
    return node


def _inject_arrays(node, arrays: Dict[str, np.ndarray]):
    """The inverse of :func:`_extract_arrays`: skeleton + arrays -> the resolved spec."""
    if isinstance(node, dict):
        if set(node.keys()) == {"__ref__"}:
            return arrays[node["__ref__"]]
        return {key: _inject_arrays(value, arrays) for key, value in node.items()}
    if isinstance(node, list):
        return [_inject_arrays(value, arrays) for value in node]
    return node


# --------------------------------------------------------------------------
# the spec bundle save/load agree on
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelCacheSpec:
    """What :func:`save_model` writes and :func:`load_model` verifies against.

    ``kernel_spec`` must be *resolved* (see :func:`spec_hash`). ``inference``
    is carried along for provenance and is used to rebuild a
    :class:`~src.SSM.model.continuous_inference.ContinuousInference` on load,
    but - see the module docstring's third bullet - it is not part of the hash.
    """

    kernel_spec: Any
    approximation: Any = None
    truncation: Any = None
    inference: Any = None

    def hash(self, domain: MeanDomain) -> str:
        return spec_hash(domain, self.kernel_spec, self.approximation, self.truncation)


# --------------------------------------------------------------------------
# save / load
# --------------------------------------------------------------------------

def save_model(path, model: StatisticalShapeModel, spec: ModelCacheSpec) -> None:
    """Writes one ``.npz`` holding the model and the spec that produced it.

    A model with no ``domain`` cannot be cached coherently - there would be
    nothing for a later call to hash against - so this raises rather than
    guessing a domain from ``mean``/``faces`` behind the caller's back.
    """
    if model.domain is None:
        raise ValueError(
            "cannot cache a model with no domain; build it through "
            "src.SSM.model.builder.build_model, which always attaches one"
        )

    key = spec.hash(model.domain)
    arrays: Dict[str, np.ndarray] = {}
    skeleton = _extract_arrays(spec.kernel_spec, arrays)

    landmarks = model.landmarks or {}
    payload = {
        "version": CACHE_VERSION,
        "key": key,
        "created": time.time(),
        "kernel_skeleton": skeleton,
        "approximation": _component_dict(spec.approximation),
        "truncation": _truncation_dict(spec.truncation),
        "inference": _component_dict(spec.inference),
        "inner_product": model.inner_product,
        "domain_metric": model.domain.metric,
        "landmark_names": [str(name) for name in landmarks.get("names ", [])],
    }

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    verts = np.asarray(landmarks.get("verts ", []), dtype=np.int64).reshape(-1)
    faces = model.faces
    tmp = path.with_suffix(path.suffix + ".tmp")
    # Pass an open handle so numpy does not append another ``.npz`` suffix to
    # the temporary name. That would leave ``tmp`` absent and make the rename
    # fail on every save.
    with tmp.open("wb") as handle:
        np.savez_compressed(
            handle,
            spec_json=np.array(_stable_json(payload)),
            mean=model.mean,
            eigenfunctions=model.eigenfunctions,
            eigenvalues=model.eigenvalues,
            noise=np.asarray(float(model.noise), dtype=np.float64),
            domain_points=model.domain.points,
            domain_mass=model.domain.mass,
            has_faces=np.asarray(faces is not None),
            faces=np.zeros((0, 3), dtype=np.int64) if faces is None else faces,
            landmark_verts=verts,
            **{f"array::{name}": array for name, array in arrays.items()},
        )
    # Rename rather than write in place: a build killed mid-write must not
    # leave a truncated file that a later load reads as a legitimate miss's
    # replacement and half-trusts.
    tmp.replace(path)


def annotate_landmarks(model: StatisticalShapeModel, landmarks_json) -> StatisticalShapeModel:
    """Gives a model that carries no landmarks the ones named in a JSON file.

    A GPMM built on a Deformetrica atlas inherits the atlas's lack of landmark
    annotation (see :mod:`src.SSM.model.landmarks`), and the archive keeps the
    source's vertex ordering unchanged - so the very file that annotates the
    atlas annotates every model built from it, with no rebuild. A model that
    brought its own landmarks along (anything descended from the BabyFM .mat)
    keeps them: this only ever fills a gap, it never overrides.
    """
    if landmarks_json is None or len(model.landmarks.get("names ", [])):
        return model

    from src.SSM.model.landmarks import read_landmark_map

    mapping = read_landmark_map(landmarks_json)
    if not mapping:
        return model

    verts = np.asarray(list(mapping.values()), dtype=np.int64)
    if verts.max(initial=-1) >= model.n_vertices:
        raise ValueError(
            f"{landmarks_json} names vertex {verts.max()}, but this model has only "
            f"{model.n_vertices} vertices - it annotates a different mesh"
        )
    model.landmarks = {"names ": list(mapping), "verts ": verts}
    return model


def load_model_file(path, normalize: bool = False, atlas_options=None,
                    verbose: bool = False) -> StatisticalShapeModel:
    """Load a serialized GPMM or a legacy model source into one model object.

    ``.npz`` files are the archives written by :func:`save_model`. Legacy
    ``.mat`` files and atlas directories are converted to the same
    :class:`StatisticalShapeModel` contract without changing their source
    files. This keeps callers independent of the on-disk representation.

    ``atlas_options`` reaches a ``.npz`` only through its ``landmarks`` entry
    (see :func:`annotate_landmarks`); everything else in it describes how to
    turn an atlas into a linear model, which the archive already is.
    """
    path = Path(path)
    if path.suffix.lower() == ".npz":
        model = load_model(path)
        if model is None:
            raise ValueError(f"could not load serialized GPMM model from {path}")
        return annotate_landmarks(model, dict(atlas_options or {}).get("landmarks"))

    from src.SSM.model.load import load_shape_model, load_shape_model_triangulation

    mean, eigenfunctions, eigenvalues, landmarks = load_shape_model(
        path, normalize=normalize, atlas_options=atlas_options, verbose=verbose
    )
    faces = load_shape_model_triangulation(path)
    domain = MeanDomain.from_mean(mean, faces)
    return StatisticalShapeModel(
        mean=np.asarray(mean, dtype=np.float64).reshape(-1),
        eigenfunctions=np.asarray(eigenfunctions, dtype=np.float64),
        eigenvalues=np.asarray(eigenvalues, dtype=np.float64).reshape(-1),
        landmarks=landmarks or {},
        faces=faces,
        domain=domain,
    )


def load_model(path, expect: Optional[ModelCacheSpec] = None,
               domain: Optional[MeanDomain] = None) -> Optional[StatisticalShapeModel]:
    """Reads a cached model back, or ``None`` on any reason not to trust it.

    ``None`` covers three cases uniformly - file absent, wrong schema version,
    and (when ``expect`` is given) a spec that no longer matches - because a
    caller's response to all three is the same: rebuild. ``domain`` lets the
    caller supply the domain it already has (from freshly re-reading the model
    source) rather than trusting the one stored in the file for the
    verification step; omitted, verification trusts the stored geometry, which
    only checks that the *kernel and method* still match, not that the source
    mesh does.
    """
    path = Path(path)
    if not path.exists():
        return None

    try:
        with np.load(path, allow_pickle=False) as data:
            payload = json.loads(str(data["spec_json"]))
            if payload.get("version") != CACHE_VERSION:
                return None

            arrays = {name[len("array::"):]: data[name] for name in data.files
                     if name.startswith("array::")}
            kernel_spec = _inject_arrays(payload["kernel_skeleton"], arrays)

            mean = np.array(data["mean"])
            eigenfunctions = np.array(data["eigenfunctions"])
            eigenvalues = np.array(data["eigenvalues"])
            noise = float(data["noise"])
            domain_points = np.array(data["domain_points"])
            domain_mass = np.array(data["domain_mass"])
            faces = np.array(data["faces"]) if bool(data["has_faces"]) else None
            landmark_verts = np.array(data["landmark_verts"])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None

    try:
        stored_domain = MeanDomain(points=domain_points, faces=faces, mass=domain_mass,
                                   metric=payload["domain_metric"])

        if expect is not None:
            reference = domain if domain is not None else stored_domain
            if expect.hash(reference) != payload["key"]:
                return None

        kernel = build_kernel(kernel_spec)
        inference_spec = payload["inference"]
        inference_name = inference_spec["name"] or "none"
        inference = build_inference(inference_name, inference_spec["kwargs"]).prepare(
            stored_domain, kernel, eigenfunctions
        )

        return StatisticalShapeModel(
            mean=mean,
            eigenfunctions=eigenfunctions,
            eigenvalues=eigenvalues,
            landmarks={"verts ": landmark_verts, "names ": payload["landmark_names"]},
            faces=faces,
            domain=stored_domain,
            kernel=kernel,
            inference=inference,
            noise=noise,
            inner_product=payload["inner_product"],
        )
    except (KeyError, TypeError, ValueError, RuntimeError):
        return None


def prune_cache(root, max_entries: int = 200) -> int:
    """Deletes the oldest cache files past ``max_entries``. Returns how many.

    A hyperparameter search (``docs/gpmm_architecture.md`` section 16)
    generates hundreds of distinct specs in one run; without a cap, one
    overnight sweep fills the disk with models nobody will ask for twice. Not
    called automatically - a search invokes it between rounds, or a script
    passes ``cache_dir=None`` and never writes at all.
    """
    root = Path(root)
    if not root.exists():
        return 0
    files = sorted(root.glob("*.npz"), key=lambda p: p.stat().st_mtime)
    doomed = files[:max(0, len(files) - max_entries)]
    for stale in doomed:
        stale.unlink()
    return len(doomed)
