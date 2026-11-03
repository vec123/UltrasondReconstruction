
The problem:

US processing (1 pipeline)  ──►  targets (fused scan meshes, bounding-box cuts, landmark CSVs)
                                     │
          ┌──────────────────────────┴──────────────────────────┐
  GPMM (src/SSM: model, kernels, nicp)                Neural LDDMM (src/resnet_lddmm)
  correspondences computed DURING the fit             correspondences computed AFTER the fit
          └──────────────────────────┬──────────────────────────┘
                     recon + target meshes with point fields
                                     │
               metrics (distances, histograms, landmarks) + correspondence fields
                                     │
                       comparison GPMM vs LDDMM, downstream data




Structure:


src/
  core/            registry, config base (one interpolate), device, stage framework (from scripts/pipeline)
  geometry/        one numpy + one torch implementation (the twin delegates where it can)
  vtk/             mesh/field IO and VTK adapters (no copies in geometry.py)
  transforms/      SimilarityTransform (+ from_json; apply_polydata rotates normals)
  ultrasound/      the US stages (from scripts/US/pipeline/stages): targets for both models
  models/
    gpmm/          from src/SSM (model, kernels, lowrank, nicp, runners)
    lddmm/         from src/resnet_lddmm
  correspondence/  from src/SSM/correspondence: model-neutral (cloud, base, methods, compute, writers)
  evaluation/      metrics, ReconSource / TargetSource (scan mm), case table, results store, compare
scripts/           thin command lines only: parse arguments, call one src function


Import rules:

src never imports scripts.
scripts never imports scripts.
models/gpmm and models/lddmm never import each other.
Both models and evaluation import only the shared packages: core, geometry, vtk, transforms, correspondence