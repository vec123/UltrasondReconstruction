"""Registrations for resnet_lddmm components.

Each task adds its component's line here when created.
The registry resolves (category, name) pairs to lazy-loaded classes.
"""

from src.resnet_lddmm.learning.registry import Registry

# Each implemented component goes here.
# Format: Registry.register("category", "name", "module.path:ClassName")

# STEPS T4: Conditioning ABC
Registry.register("conditioning", "none", "src.resnet_lddmm.resnet_flow.conditioning.base:NoConditioning")

# STEPS T22: Concat + FiLM + PositionAware conditioning
Registry.register("conditioning", "concat", "src.resnet_lddmm.resnet_flow.conditioning.film:ConcatConditioning")
Registry.register("conditioning", "film", "src.resnet_lddmm.resnet_flow.conditioning.film:FiLMConditioning")
Registry.register("conditioning", "position_aware", "src.resnet_lddmm.resnet_flow.conditioning.position_aware:PositionAware")

# STEPS T6: VelocityField + TimeVaryingField + StationaryField
Registry.register("field", "time_varying", "src.resnet_lddmm.resnet_flow.fields.time_varying:TimeVaryingField")
Registry.register("field", "time_varying_fourier", "src.resnet_lddmm.resnet_flow.fields.fourier:FourierTimeVaryingField")
Registry.register("field", "time_varying_local", "src.resnet_lddmm.resnet_flow.fields.local:LocalRefinementField")
Registry.register("field", "stationary", "src.resnet_lddmm.resnet_flow.fields.time_varying:StationaryField")

# SO(3)-equivariant stationary field using e3nn SelfInteraction
Registry.register("field", "equivariant_stationary", "src.resnet_lddmm.resnet_flow.fields.equivariant:EquivariantStationaryField")

# NOT registered, because the modules do not exist (they never have -- no commit
# in this repo's history contains them):
#
#   equivariant_contextual         fields.equivariant_with_context:EquivariantContextualField
#   equivariant_contextual_simple  fields.equivariant_contextual_simple:EquivariantContextualFieldSimple
#   equivariant_template           fields.equivariant_template_field:EquivariantTemplateField
#
# Registering a target that cannot be imported turns `Registry.create` into a
# ModuleNotFoundError instead of the registry's own "unknown kind, valid names
# are ..." -- the registry advertised three fields nobody could build. runner.build
# still has a branch for each; those now fail on the registry's error, which names
# the kinds that DO exist. Re-add a line here when the module lands.

# STEPS T9: DataTerm + CDData + L2Data + EMDData
Registry.register("data_term", "chamfer", "src.resnet_lddmm.resnet_flow.losses.data_terms:CDData")
Registry.register("data_term", "l2", "src.resnet_lddmm.resnet_flow.losses.data_terms:L2Data")
Registry.register("data_term", "emd", "src.resnet_lddmm.resnet_flow.losses.data_terms:EMDData")

# Explicit opt-out: always zero, no gradient. For landmark-only runs (loss.
# landmark_weight as the sole positional supervision) -- see NullData docstring.
Registry.register("data_term", "none", "src.resnet_lddmm.resnet_flow.losses.data_terms:NullData")

# STEPS T33: WeightedCD / PCD / NCD data terms
Registry.register("data_term", "weighted_chamfer", "src.resnet_lddmm.resnet_flow.losses.data_terms:WeightedCDData")

# Per-direction (weight, base_sigma, point-field) chamfer -- see config.ChamferSideCfg
Registry.register("data_term", "field_weighted_chamfer", "src.resnet_lddmm.resnet_flow.losses.data_terms:FieldWeightedCDData")
# Same per-direction (weight, base_sigma, field) config shape as
# field_weighted_chamfer, but the residual is measured to the target's tangent
# plane. Needs a 'normal' point array on the target -- see
# scripts/write_surface_normals.py and runner.FIELD_WEIGHTED_DATA_NAMES.
Registry.register("data_term", "field_weighted_point2plane_chamfer", "src.resnet_lddmm.resnet_flow.losses.data_terms:FieldWeightedP2PlaneData")
Registry.register("data_term", "pcd", "src.resnet_lddmm.resnet_flow.losses.data_terms:PCDData")
Registry.register("data_term", "ncd", "src.resnet_lddmm.resnet_flow.losses.data_terms:NCDData")

# STEPS T34: SinkhornData (lazy-loaded geomloss)
Registry.register("data_term", "sinkhorn", "src.resnet_lddmm.resnet_flow.losses.data_terms:SinkhornData")

# STEPS T10: ShapeCode + NoCode
Registry.register("code", "none", "src.resnet_lddmm.resnet_flow.codes.none:NoCode")

# STEPS T27: AutoDecoderCodes (per-shape latent codes)
Registry.register("code", "auto_decoder", "src.resnet_lddmm.resnet_flow.codes.auto_decoder:AutoDecoderCodes")

# STEPS T31: EncoderCodes (amortised codes via equivariant encoder)
Registry.register("code", "encoder", "src.resnet_lddmm.resnet_flow.codes.encoder:EncoderCodes")

# Rigid pose sources (code.pose in the YAML). Free parameters, no network -- the
# auto-decoding pattern applied to SE(3). Injected into ANY code source, so
# `code: {kind: none, pose: {kind: se3}}` is rigid-only registration while
# `code: {kind: auto_decoder, pose: {kind: se3}}` fits deformation and pose together.
Registry.register("pose", "so3", "src.resnet_lddmm.resnet_flow.poses.free:FreeSO3")
Registry.register("pose", "se3", "src.resnet_lddmm.resnet_flow.poses.free:FreeSE3")

# IsometryLoss (shape-preserving regularization)
Registry.register("iso_loss", "isometry", "src.resnet_lddmm.resnet_flow.losses.data_terms:IsometryLoss")

# Config-selected loss terms (loss.terms in the YAML). Each takes a LossContext
# and returns a scalar or None; see losses/terms.py.
Registry.register("loss_term", "equivariant_deformation_loss", "src.resnet_lddmm.resnet_flow.losses.terms:EquivariantDeformationLoss")
Registry.register("loss_term", "pose_supervision_loss", "src.resnet_lddmm.resnet_flow.losses.terms:PoseSupervisionLoss")
Registry.register("loss_term", "flow_rotation_penalty", "src.resnet_lddmm.resnet_flow.losses.terms:FlowRotationPenalty")
Registry.register("loss_term", "landmark_guidance", "src.resnet_lddmm.resnet_flow.losses.terms:LandmarkGuidanceLoss")
Registry.register("loss_term", "point_weighted_kinetic", "src.resnet_lddmm.resnet_flow.losses.kinetic:PointWeightedKineticLoss")
Registry.register("loss_term", "fshape_data", "src.resnet_lddmm.resnet_flow.losses.fshape:FshapeDataTerm")

# Augmentation: random group transformations (SE(3), SO(3), or none)
Registry.register("augmentation", "none", "src.resnet_lddmm.resnet_flow.augmentation.none:NoAugmentation")
Registry.register("augmentation", "so3", "src.resnet_lddmm.resnet_flow.augmentation.so3:SO3Augmentation")
Registry.register("augmentation", "se3", "src.resnet_lddmm.resnet_flow.augmentation.se3:SE3Augmentation")
