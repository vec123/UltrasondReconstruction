"""Config for one NICP fit: inputs, compute settings and outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .model import ModelConfig


@dataclass
class ToModelSpace:
    """How the target mesh gets into model space, which is where the fit happens.

    Everything measured in model units - correspondence ``dist_thresh``, chamfer
    ``max_dist``, the prior weights - is meaningless until this is right. The
    model spans about +/-0.01 with an RMS radius of 5.6e-3 while the raw meshes
    span about +/-1, a factor of ~154.

    Two ways to get there, and at most one may be set:

    template_vtp   solve it. The template is vertex-wise corresponding with the
                   model (both 31027 points in the same order), so the
                   similarity transform is a closed-form Procrustes fit over
                   every vertex - no landmarks. This is the pipeline's step 0b,
                   direction to_model, and it makes the run self-contained.
    transform_npz  replay a solved one, e.g. the outputs/gmm_2_template.npz the
                   pipeline writes.

    Both null means mesh_vtp is already in model space.
    """

    template_vtp: Optional[Path] = None
    transform_npz: Optional[Path] = None
    inverse: bool = False           # transform_npz only: apply its inverse
    allow_scale: bool = True        # template_vtp only: similarity vs rigid-only
    save_transform: bool = True     # template_vtp only: write the solved transform out

    def __post_init__(self):
        if self.template_vtp is not None and self.transform_npz is not None:
            raise ValueError(
                "set at most one of inputs.to_model_space.template_vtp / .transform_npz"
            )


@dataclass
class NicpInputs:
    # BabyFaceModel .mat statistical shape model, or a Deformetrica atlas
    # directory (see AtlasConfig).
    shape_model_mat: Path
    mesh_vtp: Path                   # target surface to fit (the ultrasound scan)
    to_model_space: ToModelSpace = field(default_factory=ToModelSpace)
    # Optional starting point. init_alpha_npz reads an 'alpha' array (as written
    # by an earlier run); init_vtp projects a mesh onto the model basis instead.
    # Both unset starts from the mean shape.
    init_vtp: Optional[Path] = None
    init_alpha_npz: Optional[Path] = None

    def __post_init__(self):
        if self.init_vtp is not None and self.init_alpha_npz is not None:
            raise ValueError("set at most one of inputs.init_vtp / inputs.init_alpha_npz")


@dataclass
class NicpComputeConfig:
    device: str = "auto"             # auto | cpu | cuda
    # float32 halves the memory traffic and, on a consumer GPU, runs ~32x faster
    # than float64. The K x K system inside the closed-form solver is promoted to
    # float64 regardless, because at 163 x 163 that costs nothing.
    dtype: str = "float32"
    seed: Optional[int] = 0          # seeds the loss subsampling
    # Attach the model's triangulation to every instance. This changes matching:
    # normals then come from the surface and are consistently oriented, where a
    # point cloud gets them from a local SVD whose sign is arbitrary. False
    # treats the model as a point cloud, which is what the pipeline does.
    use_model_faces: bool = True


@dataclass
class NicpOutputFiles:
    # Per run, written under outputs.dir / run_dir. Results appear in both
    # frames: the fit happens in model space, but the scan and everything
    # downstream of it live in world space, so each result is also mapped back
    # through the inverse of the to_model_space transform.
    fit: str = "model_space/fit.vtp"
    fit_world: str = "world/fit.vtp"
    alpha: str = "alpha.npz"
    matching_field: str = "model_space/matches.vtp"
    iteration: str = "model_space/iterations/iter_{iteration:04d}.vtp"
    iteration_world: str = "world/iterations/iter_{iteration:04d}.vtp"
    # shared, written once under outputs.dir. The inputs are saved in both
    # frames because the fit happens in model space while the data lives in
    # world space, and comparing them is the fastest way to see a bad transform.
    world_mesh: str = "world/mesh.vtp"
    world_template: str = "world/template.vtp"
    model_mesh: str = "model_space/mesh.vtp"
    model_template: str = "model_space/template.vtp"
    model_init: str = "model_space/init_shape.vtp"
    model_mean: str = "model_space/mean.vtp"
    transform: str = "gmm_2_template.npz"        # written when solved from the template
    metrics: str = "metrics.csv"
    summary: str = "summary.json"


@dataclass
class NicpOutputs:
    dir: Path
    run_dir: str = "{run}"             # one subfolder per run; "" puts them all in dir
    files: NicpOutputFiles = field(default_factory=NicpOutputFiles)
    save_alpha: bool = True            # lets a later step start where this one stopped
    save_matching_field: bool = True   # skipped by correspondence-free runs
    save_history: bool = True          # per-iteration objective, into summary.json
    # Also write every result mapped back to the frame the inputs arrived in.
    # Has no effect when to_model_space is off, since the two frames coincide.
    save_world: bool = True
    # One mesh per iteration, so the fit can be played back. ~370 KB each, so a
    # 300-iteration gradient run is ~110 MB - raise the stride to thin it out.
    save_iterations: bool = False
    iteration_stride: int = 1

    def __post_init__(self):
        if self.iteration_stride < 1:
            raise ValueError(f"iteration_stride must be >= 1, got {self.iteration_stride}")

    def path(self, key: str, **fmt) -> Path:
        """A shared output path; parent directories are created."""
        path = self.dir / getattr(self.files, key).format(**fmt)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def run_path(self, key: str, run: str, **fmt) -> Path:
        """An output path belonging to one run; parent directories are created."""
        folder = self.dir / self.run_dir.format(run=run) if self.run_dir else self.dir
        path = folder / getattr(self.files, key).format(run=run, **fmt)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path


@dataclass
class NicpFitConfig:
    inputs: NicpInputs
    outputs: NicpOutputs
    model: ModelConfig = field(default_factory=ModelConfig)
    compute: NicpComputeConfig = field(default_factory=NicpComputeConfig)
    # Raw mappings: 'runs' entries are partial overrides of 'defaults', so they
    # are merged before being validated into NicpFitSpec.
    defaults: Dict[str, Any] = field(default_factory=dict)
    runs: List[Dict[str, Any]] = field(default_factory=list)
    verbose: bool = True
