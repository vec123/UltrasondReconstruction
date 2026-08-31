"""Entry points shared by run_stage.py and run_pipeline.py."""

import os

from .config import from_dict, interpolate, load_yaml, project_root
from .registry import STAGE_REGISTRY


def _bind_stage_config(stage_name, raw):
    """Binds an already-interpolated raw dict to its registered stage Config.

    Uses the stage's own binder (`StageSpec.from_dict`) when it has one -
    SSM and resnet-lddmm each already validate their YAML more strictly than
    the generic nested-dataclass binder below could (cross-field checks,
    aliasing, nested unions) - falling back to the generic binder otherwise.
    """
    spec = STAGE_REGISTRY.get(stage_name)
    binder = spec.from_dict or (lambda d: from_dict(spec.config_cls, d))
    return spec, binder(raw)


def _no_pipeline_context(reference):
    raise ValueError(
        f"'{reference}' looks like a cross-stage reference, but a standalone stage run "
        f"(run_stage.py) has no other stages to reference. Use a literal path, or run this "
        f"stage as part of a pipeline (run_pipeline.py) instead."
    )


def stage_main(config_path, extra_vars=None):
    """Loads and runs a single stage from its own YAML config file."""
    variables = {"PROJECT_ROOT": project_root(), **(extra_vars or {})}

    raw = load_yaml(config_path)
    stage_name = raw.pop("stage", None)
    if stage_name is None:
        raise ValueError(f"'{config_path}' is missing a top-level 'stage: <name>' field.")

    raw = interpolate(raw, variables, resolve_ref=_no_pipeline_context)
    spec, cfg = _bind_stage_config(stage_name, raw)

    print(f"Running stage '{stage_name}' from '{config_path}'")
    spec.run(cfg)
    return cfg


def pipeline_main(pipeline_config_path, extra_vars=None):
    """Runs every stage listed in a pipeline YAML, in order.

    `stages:` only lists execution order and each stage's `enabled` flag; a
    stage's actual fields live in the top-level `paths:` and
    `hyperparameters:` sections, keyed by stage id - so every path across
    the whole run is grouped together, and every hyperparameter is grouped
    together, rather than being inlined stage-by-stage.

    Each entry has a `name` (which STAGE_REGISTRY implementation runs) and
    an optional `id` (defaults to `name`) that keys its `paths:` /
    `hyperparameters:` and is what other stages reference. Giving an entry
    its own `id` lets the same implementation run more than once in one
    pipeline - e.g. two `name: roughness_filter` entries, `id: roughness_filter`
    and `id: roughness_filter_final`, each with their own paths/hyperparameters.

    A field may reference an earlier stage's own (already-resolved) field
    directly, e.g. `vtp_shapes_dir: nrrd_to_vtp.output_dir`, or build a path
    out of one, e.g. `transforms_dir: align_to_reference.output_dir + /transforms`
    - see `pipeline/config.py::interpolate`. References use the target's `id`
    and may only point at stages listed earlier in `stages:`. `name` may be
    any stage registered in `STAGE_REGISTRY`, from any pipeline - a preprocess
    stage, `ssm_fit`, or `resnet_lddmm_train` - so one file can wire all three
    together.
    """
    pipeline_config_path = os.path.abspath(pipeline_config_path)
    pipeline_cfg = load_yaml(pipeline_config_path)
    variables = {"PROJECT_ROOT": project_root(), **(extra_vars or {})}

    stages = pipeline_cfg.get("stages", [])
    if not stages:
        raise ValueError(f"'{pipeline_config_path}' has no stages.")

    all_paths = pipeline_cfg.get("paths", {})
    all_hyperparameters = pipeline_cfg.get("hyperparameters", {})

    resolved_stages = {}  # stage id -> its bound Config, for cross-stage field references
    resolved_stage_fields = {}  # stage id -> set of field names its own 'paths:'/'hyperparameters:' explicitly set

    def resolve_ref(reference):
        stage_id, *attr_path = reference.split(".")
        if stage_id not in resolved_stages:
            raise ValueError(
                f"'{reference}' (in '{pipeline_config_path}') references stage '{stage_id}', "
                f"which hasn't run yet. Cross-stage references only see stages listed earlier "
                f"under 'stages:'."
            )
        if attr_path[0] not in resolved_stage_fields[stage_id]:
            raise ValueError(
                f"'{reference}' (in '{pipeline_config_path}') references '{stage_id}.{attr_path[0]}', "
                f"which is not explicitly set under stage '{stage_id}' in this file's 'paths:' or "
                f"'hyperparameters:'. A cross-stage reference may only name a field the target stage's "
                f"own entry actually sets - relying on that stage's run()-time default-filling instead "
                f"would leave this dependency invisible in the config. Add '{attr_path[0]}:' explicitly "
                f"under '{stage_id}' (see e.g. align_to_template's out_shapes_vtp_dir)."
            )
        target = resolved_stages[stage_id]
        for attr in attr_path:
            if not hasattr(target, attr):
                raise ValueError(f"Stage '{stage_id}' has no field '{attr}' (from reference '{reference}').")
            target = getattr(target, attr)
        return target

    for stage_entry in stages:
        stage_name = stage_entry.get("name")
        if stage_name is None:
            raise ValueError(f"A stage entry in '{pipeline_config_path}' is missing 'name'.")
        stage_id = stage_entry.get("id", stage_name)
        if stage_id in resolved_stages:
            raise ValueError(
                f"Duplicate stage id '{stage_id}' in '{pipeline_config_path}'. Running the same "
                f"stage implementation more than once requires each entry to have its own 'id'."
            )
        enabled = stage_entry.get("enabled", True)

        stage_paths = all_paths.get(stage_id, {})
        stage_hyperparameters = all_hyperparameters.get(stage_id, {})
        overlap = set(stage_paths) & set(stage_hyperparameters)
        if overlap:
            raise ValueError(
                f"Field(s) {sorted(overlap)} for stage '{stage_id}' appear in both 'paths' and "
                f"'hyperparameters' in '{pipeline_config_path}'."
            )
        raw = {**stage_paths, **stage_hyperparameters}

        raw = interpolate(raw, variables, resolve_ref=resolve_ref)
        spec, cfg = _bind_stage_config(stage_name, raw)

        # Bind and resolve the config either way, so later stages can still
        # reference this one's fields (e.g. `nrrd_to_vtp.output_dir`) even
        # when it's disabled - only the actual run() call is skipped.
        label = stage_id if stage_id == stage_name else f"{stage_id} ({stage_name})"
        if enabled:
            print(f"\n{'=' * 60}\nRunning stage: {label}\n{'=' * 60}")
            spec.run(cfg)
        else:
            print(f"\n{'=' * 60}\nSkipping stage: {label} (disabled)\n{'=' * 60}")

        resolved_stages[stage_id] = cfg
        resolved_stage_fields[stage_id] = set(stage_paths) | set(stage_hyperparameters)

    print("\nPipeline finished successfully.")
