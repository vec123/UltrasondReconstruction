"""Write a trained run's decomposition as .vtp files, in world units (mm), and summarise it.

For a run trained with ``field.modes``. Two ways in, same files:
``export_decomposition`` reads a finished run's config.json and a checkpoint
(nothing is trained; scripts/LDDMM/decompose_modes.py), and
``write_decomposition`` takes a flow already in memory (DecompositionExporter,
at the end of training). Both write to ``<run>/mode_decomposition/``:

    target.vtp                  the run's target, in the same frame
    start.vtp                   the run's start shape (undeformed source), with disp_total /
                                disp_network / disp_<mode> (vectors, mm) and their magnitudes
                                *_mm: the actual flow split exactly, the parts add up to disp_total
    deformed_full.vtp           the trained flow (what apply produces)
    deformed_network_only.vtp   the network alone
    deformed_<mode>_only.vtp    each mode alone
    deformed_modes_only.vtp     all modes together, without the network
    trajectories/<flow>.pvd     every Euler step of each of those flows (full, network_only,
    trajectories/<flow>/        <mode>_only, modes_only) as step_XXXX.vtp meshes - open the
                                .pvd in ParaView for a time series (time 0 .. 1)

The deformed_* files carry the same fields as start.vtp, so they colour alike in
ParaView. Each trajectory step carries ``velocity`` (the velocity that moved it
there, mm per unit time; zero at step 0) and ``disp_total`` (mm since the
start); where several parts are on (full, modes_only), also ``velocity_<part>``
and ``disp_<part>``: each part's share so far, adding up to ``disp_total``.
"""

import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from src.deformation_modes.landmarks import landmark_vertices
from src.resnet_lddmm.resnet_flow.callbacks.deformation.modes.parts import (
    NETWORK, PartsTrajectory, decompose, flow_with, modal_field,
)
from src.resnet_lddmm.resnet_flow.callbacks.checkpoints.files import load_checkpoint, resolve_checkpoint
from src.resnet_lddmm.resnet_flow.config import ExperimentCfg
from src.resnet_lddmm.resnet_flow.io import export_shape

OUT_DIR = "mode_decomposition"
TRAJECTORY_DIR = "trajectories"


def load_run(run: Path, checkpoint: Optional[Path] = None):
    """(config, flow with the checkpoint's weights, checkpoint dict) of a finished run.

    Raises:
        ValueError: the run has no designed modes, or uses a per-shape code.
    """
    # Local import: the runner is the composition root and imports the callbacks.
    from src.resnet_lddmm.resnet_flow.runner import build_flow, set_default_dtype

    data = json.loads((Path(run) / "config.json").read_text())
    data.pop("mode", None)                     # provenance stamp, not a config field (see apply/config.py)
    cfg = ExperimentCfg.from_dict(data)
    if cfg.field.modes is None:
        raise ValueError(f"{run} was trained without field.modes - there is nothing to decompose")
    if cfg.code.kind != "none":
        raise ValueError(f"code.kind={cfg.code.kind!r}: only pair runs without a code are supported")
    set_default_dtype(cfg)
    flow = build_flow(cfg)
    state = load_checkpoint(resolve_checkpoint(checkpoint or Path(run) / "checkpoints"))
    flow.load_state_dict(state["flow"])
    flow.eval()
    return cfg, flow, state


def summary_text(parts_mm: Dict[str, np.ndarray], total_mm: np.ndarray, reach: np.ndarray,
                 landmarks: Dict[str, int], start_mm: np.ndarray) -> str:
    """A table of what each part moved - over the face, within the modes' reach, at the alae."""
    names = list(parts_mm)
    rows = [("part", "face RMS", "face max", "reach RMS", "reach max", "alar width")]
    have_alae = {"aL", "aR"} <= landmarks.keys()
    if have_alae:
        aL, aR = landmarks["aL"], landmarks["aR"]
        lateral = start_mm[aL] - start_mm[aR]
        lateral /= np.linalg.norm(lateral)

    def row(label, d):
        m = np.linalg.norm(d, axis=1)
        width = f"{(d[aL] - d[aR]) @ lateral:+.2f} mm" if have_alae else "-"
        return (label, f"{np.sqrt((m ** 2).mean()):.2f}", f"{m.max():.2f}",
                f"{np.sqrt((m[reach] ** 2).mean()):.2f}", f"{m[reach].max():.2f}", width)

    for name in names:
        rows.append(row(name, parts_mm[name]))
    rows.append(row("sum of modes", sum(parts_mm[n] for n in names if n != NETWORK)))
    rows.append(row("total", total_mm))
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    lines = ["displacement in mm (face = every template vertex; reach = where a mode is > 10% of its peak):"]
    lines += ["  " + "   ".join(c.rjust(w) if i else c.ljust(w) for i, (c, w) in enumerate(zip(r, widths)))
              for r in rows]

    modes = sum(np.linalg.norm(parts_mm[n][reach], axis=1).sum() for n in names if n != NETWORK)
    network = np.linalg.norm(parts_mm[NETWORK][reach], axis=1).sum()
    lines.append(f"within the modes' reach, the modes did {modes / (modes + network):.0%} of the motion "
                 f"(sum of |displacement| per part)")
    if have_alae:
        end_mm = start_mm + total_mm
        before = np.linalg.norm(start_mm[aL] - start_mm[aR])
        after = np.linalg.norm(end_mm[aL] - end_mm[aR])
        lines.append(f"alar width aL-aR: {before:.2f} -> {after:.2f} mm ({after - before:+.2f} mm)")
    return "\n".join(lines)


def _write_pvd(path: Path, entries: List[Tuple[float, str]]) -> None:
    """A ParaView collection: the listed files as one time series."""
    lines = ['<?xml version="1.0"?>', '<VTKFile type="Collection" version="0.1">', "  <Collection>"]
    lines += [f'    <DataSet timestep="{time:g}" file="{file}"/>' for time, file in entries]
    lines += ["  </Collection>", "</VTKFile>", ""]
    path.write_text("\n".join(lines))


def write_trajectory(folder: Path, name: str, trajectory: PartsTrajectory, faces, transform) -> Path:
    """Every step of ``trajectory`` as ``folder/name/step_XXXX.vtp``, plus ``folder/name.pvd``."""
    scale = float(transform.scale)
    mm = lambda t: (t.detach().cpu().double().numpy() * scale).astype(np.float32)
    steps_dir = folder / name
    steps_dir.mkdir(parents=True, exist_ok=True)
    zeros = torch.zeros_like(trajectory.start)
    several = len(trajectory.velocities) > 1
    entries = []
    for k in range(trajectory.steps + 1):
        velocity = {part: (v[k - 1] if k else zeros) for part, v in trajectory.velocities.items()}
        fields = {"velocity": mm(sum(velocity.values())),
                  "disp_total": mm(trajectory.points[k] - trajectory.start)}
        if several:
            for part in trajectory.velocities:
                fields[f"velocity_{part}"] = mm(velocity[part])
                fields[f"disp_{part}"] = mm(trajectory.displacement(part, k))
        for key in list(fields):
            fields[f"{key}_mm"] = np.linalg.norm(fields[key], axis=1).astype(np.float32)
        file = f"step_{k:04d}.vtp"
        export_shape(trajectory.points[k], faces, transform, str(steps_dir / file),
                     label=f"{name} step {k}", point_fields=fields)
        entries.append((k * trajectory.dt, f"{name}/{file}"))
    pvd = folder / f"{name}.pvd"
    _write_pvd(pvd, entries)
    return pvd


def check_frame(transform) -> None:
    """Raise ValueError when the run's frame has a rotation: the displacements would need rotating too."""
    if transform.rotation is not None:
        raise ValueError("the run's frame has a rotation; displacements would need rotating too")


def export_decomposition(run: Path, checkpoint: Optional[Path] = None, out: Optional[Path] = None,
                         trajectories: bool = True, log: Callable[[str], None] = print) -> Path:
    """Decompose run ``run`` (its final checkpoint, or ``checkpoint``) and write the .vtp files.

    See :func:`write_decomposition`.

    Returns:
        The folder written (``out``, default ``<run>/mode_decomposition``).

    Raises:
        ValueError: see :func:`load_run` and :func:`check_frame`.
    """
    # Local import, as in load_run.
    from src.resnet_lddmm.resnet_flow.runner import load_pair

    run = Path(run)
    cfg, flow, state = load_run(run, checkpoint)
    source, target, transform = load_pair(cfg)
    log(f"checkpoint step {state.get('step')} ({state.get('path')})")
    return write_decomposition(flow, source, target, transform, out if out is not None else run / OUT_DIR,
                               trajectories=trajectories, log=log)


def write_decomposition(flow, source, target, transform, out: Path, trajectories: bool = True,
                        log: Callable[[str], None] = print) -> Path:
    """Decompose the trained ``flow`` on ``source`` and write the .vtp files to ``out``.

    ``source`` / ``target`` are the run's pair as runner.load_pair returns it, in
    the flow's frame; the source's landmarks give the alar width in the summary.
    ``trajectories`` also writes every Euler step of every flow (about 150 MB for
    a 31k-vertex template, 10 steps and two modes); False writes the end shapes only.

    Returns:
        ``out``.

    Raises:
        ValueError: the flow has no designed modes, or see :func:`check_frame`.
    """
    check_frame(transform)
    scale = float(transform.scale)
    field = modal_field(flow)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    log(f"modes {field.names}")
    amplitudes = field.amplitude.detach()
    for m, name in enumerate(field.names):
        log(f"  {name}: amplitude per step (mm/unit time) "
            + " ".join(f"{a:+.2f}" for a in amplitudes[:, m].tolist())
            + f"   mean {amplitudes[:, m].mean():+.2f}")

    points = source.points[0]
    parts = decompose(flow, points)
    to_mm = lambda t: t.detach().cpu().double().numpy() * scale
    parts_mm = {name: to_mm(d) for name, d in parts.parts.items()}
    total_mm = to_mm(parts.total)
    closure = np.abs(sum(parts_mm.values()) - total_mm).max()
    log(f"split check: parts add up to the total to {closure:.1e} mm")

    with torch.no_grad():
        speed = field.basis(points[None].to(field.amplitude.dtype))[0].norm(dim=-1)   # [N, M]
    reach = (speed >= 0.1 * speed.max(dim=0).values).any(dim=1).cpu().numpy()
    landmarks = landmark_vertices(source.point_fields or {})
    start_mm = points.detach().cpu().double().numpy() * scale
    log("\n" + summary_text(parts_mm, total_mm, reach, landmarks, start_mm))

    fields = {"disp_total": total_mm.astype(np.float32),
              "disp_total_mm": np.linalg.norm(total_mm, axis=1).astype(np.float32),
              "mode_reach": reach.astype(np.float32)}
    for name, d in parts_mm.items():
        fields[f"disp_{name}"] = d.astype(np.float32)
        fields[f"disp_{name}_mm"] = np.linalg.norm(d, axis=1).astype(np.float32)

    def write(name, end, label):
        export_shape(end, source.faces, transform, str(out / f"{name}.vtp"), label=label, point_fields=fields)

    export_shape(target.points[0], target.faces, transform, str(out / "target.vtp"), label="target",
                 point_fields=target.point_fields)
    write("start", points, "start shape with per-part displacements")
    flows = {"full": parts, "network_only": flow_with(flow, points, [NETWORK])}
    flows.update({f"{name}_only": flow_with(flow, points, [name]) for name in field.names})
    flows["modes_only"] = flow_with(flow, points, field.names)
    for name, trajectory in flows.items():
        write(f"deformed_{name}", trajectory.end, name.replace("_", " "))
    if trajectories:
        for name, trajectory in flows.items():
            write_trajectory(out / TRAJECTORY_DIR, name, trajectory, source.faces, transform)
        log(f"trajectories: {len(flows)} flows x {parts.steps + 1} steps in {out / TRAJECTORY_DIR} "
            f"(open the .pvd files in ParaView)")
    log(f"\nwrote {out}")
    return out
