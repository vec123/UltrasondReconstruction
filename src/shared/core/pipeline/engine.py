"""The stage-pipeline engine: binds stage YAMLs to their Config and runs them.

The engine knows no stage. Each caller passes the registry of the stages it
may run (scripts/pipeline/registry.py joins those of every layer).

A YAML is one stage (``stage: <name>``) or a pipeline (``stages: [...]``);
run_main runs either. Both may define the ``${NAME}`` variables they use in a
top-level ``variables:`` block, so the YAML alone says what a run does, and
a ``targets:`` block runs it once per case (targets.py, batch.py). Each stage
that finishes gets an entry in its output folder's manifest.json (manifest.py).

A stage file may split its fields into ``paths:`` and ``hyperparameters:``,
as a pipeline does. A pipeline entry can then name it (``config:``, relative
to configs/, or a pipeline-wide ``stage_configs: {<stage name>: <file>}``) and
takes over its hyperparameters - never its paths - so each stage's settings
are written once, in its domain folder. A stage file can ``extends:`` another
one of the same stage and hold only what differs from it.
"""

import os
import time

from src.shared.core.config import from_dict, interpolate_paths, load_yaml, path_variables
from src.shared.core.paths import get_project_root
from src.shared.core.pipeline import batch, manifest
from src.shared.core.pipeline.cases import with_case_key
from src.shared.core.pipeline.targets import find_cases

CONFIGS_DIR = "configs"
SOURCES_FILE = "sources.yaml"  # the layout variables of every run, in CONFIGS_DIR
SECTIONS = ("paths", "hyperparameters")
RUN_KEYS = ("targets", "batch")  # which cases run, and how (run_main); not fields of a stage


def _bind_stage_config(registry, stage_name, raw, source):
    """Binds an already-interpolated raw dict to its registered stage Config.

    Uses the stage's own binder (`StageSpec.from_dict`) when it has one -
    resnet-lddmm's configs check across fields and accept aliases - and the
    strict generic binder (src.shared.core.config.from_dict) otherwise. `source`
    names the YAML in the error messages.
    """
    spec = registry.get(stage_name)
    binder = spec.from_dict or (lambda d: from_dict(spec.config_cls, d, source=source))
    return spec, binder(raw)


def _own_variables(yaml_variables, base, source):
    """The ``variables:`` block of ``source`` as strings; it may not redefine a name of ``base`` (.env)."""
    if yaml_variables is None:
        yaml_variables = {}
    if not isinstance(yaml_variables, dict):
        raise ValueError(f"'variables:' in '{source}' must be a mapping of NAME: value, "
                         f"got {type(yaml_variables).__name__}.")
    own = {str(name): str(value) for name, value in yaml_variables.items()}
    clash = sorted(set(own) & set(base))
    if clash:
        raise ValueError(f"'variables:' in '{source}' redefines {clash}, which the repo's .env sets; "
                         f"remove them there (every path of the run would silently move).")
    return own


def _source_variables(base):
    """The layout variables of configs/sources.yaml (none if the file is absent)."""
    path = os.path.join(get_project_root(), CONFIGS_DIR, SOURCES_FILE)
    if not os.path.isfile(path):
        return {}
    return _own_variables(load_yaml(path).get("variables"), base, path)


def _variables(yaml_variables, extra_vars, source):
    """The ``${NAME}`` variables of a run: PROJECT_ROOT/DATA_ROOT (.env), then the layout
    of configs/sources.yaml, then the YAML's own ``variables:`` block, then
    ``extra_vars`` (a case of targets:, --set), each overriding the ones before,
    and CASE_KEY (cases.py) when there is a SUBJECT_ID. A variable may use any
    other variable; it is filled in where a path uses it."""
    base = path_variables()
    own = _own_variables(yaml_variables, base, source)
    return with_case_key({**base, **_source_variables(base), **own, **(extra_vars or {})})


def _deep_merge(base, override):
    """``override`` on top of ``base``: nested mappings merge key by key, any other value replaces."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _section(raw, name, source):
    section = raw.get(name)
    if section is None:
        return {}
    if not isinstance(section, dict):
        raise ValueError(f"'{name}:' in '{source}' must be a mapping, got {type(section).__name__}.")
    return section


def _join_sections(raw, source):
    """A stage file's ``paths:`` and ``hyperparameters:`` as one flat field mapping."""
    extra = sorted(set(raw) - set(SECTIONS))
    if extra:
        raise ValueError(f"'{source}' has 'paths:'/'hyperparameters:' sections, so its fields belong in them; "
                         f"move {extra} into one of the two.")
    paths, hyperparameters = _section(raw, "paths", source), _section(raw, "hyperparameters", source)
    overlap = sorted(set(paths) & set(hyperparameters))
    if overlap:
        raise ValueError(f"Field(s) {overlap} appear in both 'paths' and 'hyperparameters' in '{source}'.")
    return {**paths, **hyperparameters}


def _extended(raw, stage_name, source, seen):
    """``raw``'s own ``hyperparameters:`` on top of those of the file it ``extends:``, if any."""
    own = _section(raw, "hyperparameters", source)
    base = raw.get("extends")
    if base is None:
        return own
    return _deep_merge(_included_hyperparameters(base, stage_name, f"{source} (extends)", seen), own)


def _included_hyperparameters(config, stage_name, source, seen=()):
    """The ``hyperparameters:`` of the stage file that a pipeline entry names with ``config:``
    (or that another stage file ``extends:``), with the files it extends below them."""
    if config in seen:
        raise ValueError(f"{source}: 'extends:' loops: {' -> '.join(seen + (config,))}.")
    path = os.path.join(get_project_root(), CONFIGS_DIR, config)
    raw = load_yaml(path)
    if raw.get("stage") != stage_name:
        raise ValueError(f"{source}: '{config}' is a '{raw.get('stage')}' stage file, "
                         f"but the entry runs '{stage_name}'.")
    if not isinstance(raw.get("hyperparameters"), dict):
        raise ValueError(f"{source}: '{config}' has no 'hyperparameters:' section. A pipeline takes "
                         f"over only that section, never a stage file's paths.")
    return _extended(raw, stage_name, config, seen + (config,))


def _runner(raw, config_path):
    """pipeline_main or stage_main, as the YAML says."""
    if "stages" in raw:
        return pipeline_main
    if "stage" in raw:
        return stage_main
    raise ValueError(
        f"'{config_path}' has neither 'stage: <name>' (one stage) nor 'stages:' (a pipeline), "
        f"so no registered code can run it."
    )


def run_main(config_path, registry, extra_vars=None, case_command=None):
    """Runs ``config_path`` as one stage (``stage:``) or as a pipeline (``stages:``).

    With a ``targets:`` block (targets.py), it runs once per case: in this
    process, or with ``batch:`` (batch.py) each case in its own process,
    ``case_command(config_path, case variables)`` its command line. A case's
    variables override the YAML's own; ``extra_vars`` override both.
    """
    raw = load_yaml(config_path)
    run = _runner(raw, config_path)
    if "targets" not in raw:
        return run(config_path, registry, extra_vars)

    def case_extra(case):
        return {**case.values, **(extra_vars or {})}

    def case_variables(case):
        return _variables(raw.get("variables"), case_extra(case), config_path)

    run_variables = _variables(raw.get("variables"), extra_vars, config_path)
    cases = find_cases(raw["targets"], run_variables, config_path)
    if raw.get("batch") is None:
        failed = batch.run_in_process(cases, lambda case: run(config_path, registry, case_extra(case)))
    else:
        if case_command is None:
            raise ValueError(f"'batch:' in '{config_path}' runs each case in its own process, "
                             f"but this caller gives no command for it.")
        failed = batch.run_in_processes(
            cases, raw["batch"], run_variables, case_variables,
            lambda case: case_command(os.path.abspath(config_path), case_extra(case)), config_path)
    if failed:
        raise batch.CasesFailed(f"{len(failed)} of {len(cases)} cases failed: {failed}")
    print(f"\nAll {len(cases)} cases finished.")


def run_case(config_path, registry, case):
    """Runs ``config_path`` once, for one case of its ``targets:`` block (the variables ``case``)."""
    return _runner(load_yaml(config_path), config_path)(config_path, registry, case)


def no_pipeline_context(reference):
    raise ValueError(
        f"'{reference}' looks like a cross-stage reference, but a standalone stage run "
        f"(run_stage.py) has no other stages to reference. Use a literal path, or run this "
        f"stage as part of a pipeline (run_pipeline.py) instead."
    )


def stage_main(config_path, registry, extra_vars=None):
    """Loads and runs a single stage of ``registry`` from its own YAML config file."""
    raw = load_yaml(config_path)
    stage_name = raw.pop("stage", None)
    if stage_name is None:
        raise ValueError(f"'{config_path}' is missing a top-level 'stage: <name>' field.")
    variables = _variables(raw.pop("variables", None), extra_vars, config_path)
    for key in RUN_KEYS:
        raw.pop(key, None)
    if "hyperparameters" in raw:
        hyperparameters = _extended(raw, stage_name, config_path, ())
        raw.pop("extends", None)
        raw = _join_sections({**raw, "hyperparameters": hyperparameters}, config_path)

    raw = interpolate_paths(raw, variables, resolve_ref=no_pipeline_context)
    spec, cfg = _bind_stage_config(registry, stage_name, raw, config_path)

    print(f"Running stage '{stage_name}' from '{config_path}'")
    started = time.time()
    spec.run(cfg)
    manifest.record(spec, cfg, stage_name, stage_name, config_path, variables, started)
    return cfg


def pipeline_main(pipeline_config_path, registry, extra_vars=None):
    """Runs every stage listed in a pipeline YAML, in order, from ``registry``.

    `stages:` only lists execution order and each stage's `enabled` flag; a
    stage's actual fields live in the top-level `paths:` and
    `hyperparameters:` sections, keyed by stage id - so every path across
    the whole run is grouped together, and every hyperparameter is grouped
    together, rather than being inlined stage-by-stage.

    Each entry has a `name` (which implementation of ``registry`` runs) and
    an optional `id` (defaults to `name`) that keys its `paths:` /
    `hyperparameters:` and is what other stages reference. Giving an entry
    its own `id` lets the same implementation run more than once in one
    pipeline - e.g. two `name: roughness_filter` entries, `id: roughness_filter`
    and `id: roughness_filter_final`, each with their own paths/hyperparameters.

    An entry may also name its stage's own file with `config:` (relative to
    configs/, e.g. `config: ultrasound/stages/01_nrrd_to_vtp.yaml`), or the
    top-level `stage_configs: {<stage name>: <file>}` names one for every
    entry of that stage that has no `config:` of its own. The file's
    `hyperparameters:` section (with the files it `extends:` below it) is the
    base, and this file's `hyperparameters:` block for the entry's own `id`
    overrides single keys of it; a `name`-keyed block does not apply to such
    an entry. The stage file's paths are never used: they are only for a run
    of it alone.

    `paths:` is looked up by `id` only - paths are inherently per-instance
    (each run needs its own output_dir). `hyperparameters:`, however, falls
    back to a `name`-keyed block when no `id`-keyed one exists: several ids
    sharing one implementation (e.g. `resnet_lddmm_apply_fetus_mean` and
    `resnet_lddmm_apply_fetus_ssm`, both `name: resnet_lddmm_apply`) share the
    `hyperparameters: resnet_lddmm_apply:` block unless one of them defines
    its own `hyperparameters: <id>:` block, which then fully replaces the
    shared one for that id (no merging) - see `baby_ssm_fit` overriding
    `ssm_fit`'s block below.

    A field may reference an earlier stage's own (already-resolved) field
    directly, e.g. `vtp_shapes_dir: nrrd_to_vtp.output_dir`, or build a path
    out of one, e.g. `transforms_dir: align_to_reference.output_dir + /transforms`
    - see `src/shared/core/config/variables.py::interpolate_paths`. References use the target's `id`
    and may only point at stages listed earlier in `stages:`. `name` may be
    any stage of ``registry``, from any pipeline - a preprocess
    stage, `ssm_fit`, or `resnet_lddmm_train` - so one file can wire all three
    together.
    """
    pipeline_config_path = os.path.abspath(pipeline_config_path)
    pipeline_cfg = load_yaml(pipeline_config_path)
    variables = _variables(pipeline_cfg.get("variables"), extra_vars, pipeline_config_path)

    stages = pipeline_cfg.get("stages", [])
    if not stages:
        raise ValueError(f"'{pipeline_config_path}' has no stages.")

    all_paths = _section(pipeline_cfg, "paths", pipeline_config_path)
    all_hyperparameters = _section(pipeline_cfg, "hyperparameters", pipeline_config_path)
    stage_configs = _section(pipeline_cfg, "stage_configs", pipeline_config_path)
    shadowed = sorted(set(stage_configs) & set(all_hyperparameters))
    if shadowed:
        raise ValueError(
            f"'{pipeline_config_path}' maps {shadowed} to a stage file under 'stage_configs:' and also has a "
            f"name-keyed 'hyperparameters:' block for them, which would not apply; move it into the file."
        )

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
        # Falls back to the base `name`'s block when this id has none of its own,
        # so ids sharing one implementation share its hyperparameters by default -
        # see the docstring above.
        stage_hyperparameters = all_hyperparameters.get(
            stage_id, all_hyperparameters.get(stage_name, {})
        )
        # `config:` (or stage_configs:) names the stage's own file; then only
        # this entry's own id block overrides single keys of it (nested ones
        # too). The shared name block does not apply: the stage file has that role.
        config = stage_entry.get("config", stage_configs.get(stage_name))
        if config is not None:
            stage_hyperparameters = _deep_merge(
                _included_hyperparameters(config, stage_name, f"{pipeline_config_path} ({stage_id})"),
                all_hyperparameters.get(stage_id, {}),
            )
        overlap = set(stage_paths) & set(stage_hyperparameters)
        if overlap:
            raise ValueError(
                f"Field(s) {sorted(overlap)} for stage '{stage_id}' appear in both 'paths' and "
                f"'hyperparameters' in '{pipeline_config_path}'."
            )
        raw = {**stage_paths, **stage_hyperparameters}

        raw = interpolate_paths(raw, variables, resolve_ref=resolve_ref)
        spec, cfg = _bind_stage_config(registry, stage_name, raw, f"{pipeline_config_path} ({stage_id})")

        # Bind and resolve the config either way, so later stages can still
        # reference this one's fields (e.g. `nrrd_to_vtp.output_dir`) even
        # when it's disabled - only the actual run() call is skipped.
        label = stage_id if stage_id == stage_name else f"{stage_id} ({stage_name})"
        if enabled:
            print(f"\n{'=' * 60}\nRunning stage: {label}\n{'=' * 60}")
            started = time.time()
            spec.run(cfg)
            manifest.record(spec, cfg, stage_id, stage_name, pipeline_config_path, variables, started)
        else:
            print(f"\n{'=' * 60}\nSkipping stage: {label} (disabled)\n{'=' * 60}")

        resolved_stages[stage_id] = cfg
        resolved_stage_fields[stage_id] = set(stage_paths) | set(stage_hyperparameters)

    print("\nPipeline finished successfully.")
