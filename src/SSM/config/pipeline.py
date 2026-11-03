"""Config for scripts/pipeline.py: the whole SSM build, from GMM template to the Laplacian."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .model import ModelConfig
from .nicp import NicpConfig
from src.SSM.outputs import PipelineOutputFiles  # noqa: F401  (part of this module's API)


# --------------------------------------------------------------------------
# scripts/pipeline.py
# --------------------------------------------------------------------------

@dataclass
class PipelineInputs:
    # BabyFaceModel .mat statistical shape model, or a Deformetrica atlas
    # directory (see AtlasConfig). The atlas carries no landmarks, so
    # steps.align and steps.landmark_fit have to be off when one is used.
    shape_model_mat: Path
    mesh_vtp: Path               # target surface (ultrasound) to fit
    # Target landmark positions, name -> [x, y, z]. Only read by steps.align
    # and steps.landmark_fit; may be omitted when both are off.
    landmarks_json: Optional[Path] = None
    # The model's own mean-shape mesh (e.g. assets/templates/baby_mean.vtp), vertex-wise
    # corresponding with shape_model_mat - same vertex count, same ordering. The
    # model format (.mat or Deformetrica atlas) carries geometry and landmarks
    # but no arbitrary point-data fields, so every mesh this pipeline writes is
    # bare; if this is set, PipelineRunner._save_points copies template_vtp's
    # point-data arrays (e.g. heatmap_<label>, which resnet_lddmm's landmark
    # loss reads off the source mesh) onto any output whose point count matches
    # it. Optional - leave unset if nothing downstream needs those fields.
    template_vtp: Optional[Path] = None


@dataclass
class StepsConfig:
    """Which stages run. Every stage writes its shape into the result registry
    under a key, and later stages address earlier results by that key."""

    gmm_2_template: bool = False  # 0b: solve the GMM <-> template similarity transform
    align: bool = True            # 1a: similarity transform of the target surface into model space
    landmark_fit: bool = True     # 1b: fit SSM coefficients to the landmarks
    nicp: bool = True             # 2:  correspondence + MAP-regularised surface fit
    laplacian: bool = True        # 3:  spectral operators + regional deformation


@dataclass
class GmmToTemplateConfig:
    """Similarity transform between GMM (model) space and template space.

    template_vtp must be vertex-wise corresponding with the model, i.e. carry
    exactly n_vertices points in the model's ordering, which lets the transform
    be solved in closed form over every vertex.
    """

    template_vtp: Path
    # to_model:    map the template (and the target surface) into GMM space.
    #              PREFERRED - the SSM is untouched, so eigenfunctions and
    #              eigenvalues stay meaningful for sampling, and gamma/chi_clamp
    #              keep the meaning they were tuned with. Distance thresholds
    #              must then be given in model units.
    # to_template: rescale the SSM itself into template space (mean and
    #              eigenfunctions by s, eigenvalues by s^2). A valid change of
    #              variables in theory, but it invalidates two things here:
    #                * clamp_coefficients() hardcodes beta_sq = 10000**2 rather
    #                  than deriving it from the eigenvalues, so chi_clamp
    #                  silently changes meaning;
    #                * l2_closed_form_step solves (Phi.T @ Phi + gamma*I), and
    #                  Phi.T @ Phi scales by s^2, so nicp.gammas must be
    #                  rescaled by s^2 to mean the same thing.
    direction: str = "to_model"
    allow_scale: bool = True
    apply_to_target: bool = True   # to_model only: also map inputs.mesh_vtp + landmarks

    def __post_init__(self):
        directions = ("to_model", "to_template")
        if self.direction not in directions:
            raise ValueError(
                f"gmm_2_template.direction must be one of {list(directions)}, "
                f"got {self.direction!r}"
            )


@dataclass
class RigidConfig:
    iterations: int = 1                        # rigid align <-> landmark fit rounds
    landmark_fit_learning_rate: float = 0.01   # ShapeInstance.fit_landmarks step size
    allow_scale: bool = True                   # similarity (True) vs rigid-only (False)


@dataclass
class SpectralConfig:
    path: Path                        # .npz holding L, M, eigenvalues, eigenvectors
    compute: bool = False             # False = load 'path', True = compute from the NICP result
    save: bool = True                 # write the operators to 'path' after computing
    source_key: Optional[str] = None  # result to build the operators on (default: laplacian.template_key)
    k_eig: int = 20
    downsampling: bool = False        # must stay False: the Laplacian has to keep every vertex
    test_spectrum: bool = True


@dataclass
class RegionConfig:
    landmark: str = "prn"   # landmark the deformation region is centred on
    radius: float = 0.3     # region radius in model units


@dataclass
class LaplacianConfig:
    # Keys address the result registry: "refined" (step 1), "gamma_<value>"
    # (the NICP sweep) or a NICP variant name.
    template_key: str = "gamma_80000"   # result used as the deformation template
    target_key: str = "gamma_10000"     # result the region is pulled towards
    lambda_reg: float = 1.0             # weight of the Laplacian (shape preserving) term
    region: RegionConfig = field(default_factory=RegionConfig)


@dataclass
class PipelineOutputs:
    dir: Path
    files: PipelineOutputFiles = field(default_factory=PipelineOutputFiles)

    def path(self, key: str, **fmt) -> Path:
        return self.dir / getattr(self.files, key).format(**fmt)


@dataclass
class PipelineConfig:
    inputs: PipelineInputs
    outputs: PipelineOutputs
    steps: StepsConfig = field(default_factory=StepsConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    gmm_2_template: Optional[GmmToTemplateConfig] = None
    rigid: RigidConfig = field(default_factory=RigidConfig)
    nicp: NicpConfig = field(default_factory=NicpConfig)
    spectral: SpectralConfig = field(default_factory=lambda: SpectralConfig(path=Path("spectral_data.npz")))
    laplacian: LaplacianConfig = field(default_factory=LaplacianConfig)
    verbose: bool = True
