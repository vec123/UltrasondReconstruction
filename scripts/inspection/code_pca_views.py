"""Plot the first two PCA components of a cohort auto-decoder's codes, with views of
the reconstructions at chosen scatter points to inspect what the codes mean.

The chosen points are the extremes along PC1 and PC2 (far apart), the closest pairs
in the PC1/PC2 plane (close together) and the point nearest the centre. Each is shown
as its deformed template (apply/<member>/world/deformed.vtp), seen from the front
and from both sides with one common camera, so size and pose differences stay visible.

Run with:
    python scripts/inspection/code_pca_views.py --run outputs/Casos_Mar/resnet_lddmm_cohort_autodecoder_fieldweighted_nopos
"""

import argparse
import glob
import json
import os
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.shared.vtk.render import MESH_COLOR, SCAN_COLOR, read_landmarks, view_camera, write_grid  # noqa: E402
from src.shared.transforms.similarity import SimilarityTransform  # noqa: E402
from src.shared.vtk.io import load_vtp, save_vtp  # noqa: E402

from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy  # noqa: E402

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
# The training targets are point clouds resampled from each member's fused scan mesh,
# in the same frame; the path is relative to the batch root (the run's parent folder).
FUSED_SCAN = "fetus_ssm/{case}/{week}/end_to_end/fused_scan_ssm/fused_scan.vtp"


def load_codes(run):
    """(member names, [M, n_z] codes) in the embedding's row order."""
    codes = torch.load(os.path.join(run, "checkpoints", "final.pt"), map_location="cpu",
                       weights_only=False)["codes"]["codes.weight"].numpy()
    names = [None] * len(codes)
    for path in glob.glob(os.path.join(run, "apply", "*", "apply.json")):
        member = json.load(open(path))["cohort_member"]
        names[member["index"]] = member["name"]
    return names, codes


def pca(codes):
    """Scores on the first two components and the variance ratio of each."""
    centred = codes - codes.mean(axis=0)
    _, s, vt = np.linalg.svd(centred, full_matrices=False)
    return centred @ vt[:2].T, s[:2] ** 2 / np.sum(s ** 2)


def choose_points(scores, n_pairs):
    """{index: tag}: PC1/PC2 extremes, the n_pairs closest pairs and the centre point."""
    chosen = {}
    for axis in (0, 1):
        chosen.setdefault(int(np.argmin(scores[:, axis])), f"PC{axis + 1} min")
        chosen.setdefault(int(np.argmax(scores[:, axis])), f"PC{axis + 1} max")
    dist = np.linalg.norm(scores[:, None] - scores[None], axis=-1)
    dist[np.tril_indices(len(scores))] = np.inf
    pairs = 0
    for flat in np.argsort(dist, axis=None):
        i, j = np.unravel_index(flat, dist.shape)
        if i in chosen or j in chosen:
            continue
        pairs += 1
        chosen[int(i)] = chosen[int(j)] = f"close pair {pairs}"
        if pairs == n_pairs:
            break
    centre = int(np.argmin(np.linalg.norm(scores, axis=1)))
    chosen.setdefault(centre, "centre")
    return chosen


def similarity_map(path):
    """points [N, 3] -> the same points under a similarity.json's x' = s R x + t, or as they are.

    For runs whose world frame is not mm, e.g. a shape model's frame: the similarity.json
    scripts/SSM/target_to_ssm_frame.py writes maps that frame back to the scans' mm frame,
    where the cameras and the mm colour scales of these views are meant to work.

    Args:
        path: a similarity.json ({scale, rotation, translation}), or None for the identity
    """
    if path is None:
        return lambda points: points
    return SimilarityTransform.from_json(path).apply_points


def template_landmarks(run, member, to_frame=None, model=None):
    """{label: point} on the template, in the deformed templates' (world) frame.

    The eval stage moves the recon into its case's reference-scan frame, so its
    landmarks are carried back through their nearest recon vertex: deformed.vtp and
    recon_w_distance.vtp share the template's vertex order. ``to_frame`` (see
    similarity_map) maps the world points first. Without an eval (no recon_w_distance.vtp
    / recon_landmarks.vtp), ``model`` (a model_memory_io.load_model_file model, whose
    landmark vertex indices address the SAME template - the run's apply is a per-vertex
    flow decode of it) is used instead.
    """
    member_dir = os.path.join(run, "apply", member)
    world_points = vtk_to_numpy(load_vtp(os.path.join(member_dir, "world", "deformed.vtp")).GetPoints().GetData())
    if to_frame is not None:
        world_points = to_frame(world_points)
    eval_dir = os.path.join(member_dir, "eval")
    if os.path.isfile(os.path.join(eval_dir, "recon_w_distance.vtp")):
        eval_points = vtk_to_numpy(load_vtp(os.path.join(eval_dir, "recon_w_distance.vtp")).GetPoints().GetData())
        lmk = read_landmarks(os.path.join(eval_dir, "recon_landmarks.vtp"))
        return {label: world_points[np.argmin(np.linalg.norm(eval_points - p, axis=1))] for label, p in lmk.items()}
    if model is None:
        raise FileNotFoundError(f"no eval at {eval_dir}, and no model given for its landmark fallback")
    return {name: world_points[i] for name, i in zip(model.landmark_names(), model.landmark_indices())}


def target_path(run, member, to_mm=None):
    """The member's fused scan mesh, in mm, with its original faces where available: the one
    its eval scored (eval/distance_summary.json, already mm), else the Casos_Mar layout beside
    the run (members named <case>_<week>_semanas, also already mm), else, for a training source
    built by scripts/SSM/resample_fixed_points.py (whose resampling drops the faces for
    CohortBatchLoader's equal-count stacking - see its TODO), the manifest resample_fixed_points
    writes next to the run's own training config's source dir (``original_scans.json``:
    member name -> its pre-resample fused_scan.vtp, with faces, already mm), else that
    resampled, faceless source itself - in the run's OWN (world) frame, so ``to_mm`` (see
    similarity_map) is applied and the result written to a temp file, whose path is returned."""
    summary = os.path.join(run, "apply", member, "eval", "distance_summary.json")
    if os.path.isfile(summary):
        with open(summary) as f:
            return json.load(f)["target_path"]
    if "_" in member:
        case, week = member.split("_", 1)
        legacy = os.path.join(os.path.dirname(os.path.normpath(run)),
                              FUSED_SCAN.format(case=case, week=week.replace("_", " ")))
        if os.path.isfile(legacy):
            return legacy
    with open(os.path.join(run, "config.json")) as f:
        source = json.load(f)["source"]
    # Either beside the training source itself (resample_fixed_points.py's own --output-dir),
    # or one level up (a frame conversion / train-test split sits between the two).
    for manifest_dir in (source, os.path.dirname(os.path.normpath(source))):
        manifest_path = os.path.join(manifest_dir, "original_scans.json")
        if os.path.isfile(manifest_path):
            with open(manifest_path) as f:
                manifest = json.load(f)
            if member in manifest:
                return manifest[member]
    path = os.path.join(source, f"{member}.vtp")
    if to_mm is None:
        return path
    scan = load_vtp(path)
    scan.GetPoints().SetData(numpy_to_vtk(to_mm(vtk_to_numpy(scan.GetPoints().GetData())).astype(np.float32),
                                         deep=True))
    out = os.path.join(tempfile.mkdtemp(), f"{member}.vtp")
    save_vtp(scan, out)
    return out


def render_thumbnails(run, names, chosen, panel_px, to_mm=None):
    """{index: RGB image}: frontal, left and right views (rows) of each chosen
    deformed template (left column) and its target (right column). ``to_mm`` (see
    similarity_map) maps the deformed templates into the targets' mm frame first."""
    lmk = template_landmarks(run, names[next(iter(chosen))], to_mm)
    meshes = {i: load_vtp(os.path.join(run, "apply", names[i], "world", "deformed.vtp")) for i in chosen}
    if to_mm is not None:
        for mesh in meshes.values():
            mesh.GetPoints().SetData(numpy_to_vtk(np.ascontiguousarray(
                to_mm(vtk_to_numpy(mesh.GetPoints().GetData())), dtype=np.float32), deep=True))
    targets = {i: load_vtp(target_path(run, names[i], to_mm)) for i in chosen}
    cameras = [view_camera(lmk, list(meshes.values()), view) for view in ("frontal", "left", "right")]
    images = {}
    with tempfile.TemporaryDirectory() as tmp:
        for i, mesh in meshes.items():
            path = os.path.join(tmp, f"{i}.png")
            pair = [("Recon", mesh, MESH_COLOR, []), ("Target", targets[i], SCAN_COLOR, [])]
            write_grid([(camera, pair) for camera in cameras], 2, "", path, panel_px)
            images[i] = plt.imread(path)[40:]  # drop write_grid's empty header
    return images


def week_of(name):
    return int(name.split("_")[1])


def plot(names, scores, ratio, chosen, images, title, out_path):
    """Scatter of the codes, chosen points lettered, their views in a grid below."""
    weeks = np.array([week_of(n) for n in names])
    order = sorted(chosen, key=lambda i: (chosen[i], scores[i, 0]))
    columns = 4
    tile_rows = int(np.ceil(len(order) / columns))
    fig = plt.figure(figsize=(4.4 * columns, 8 + tile_rows * 6.6), dpi=110)
    grid = fig.add_gridspec(1 + tile_rows, columns, height_ratios=[8] + [6.6] * tile_rows)
    ax = fig.add_subplot(grid[0, :])

    points = ax.scatter(scores[:, 0], scores[:, 1], c=weeks, cmap="Blues", vmin=weeks.min() - 8,
                        vmax=weeks.max(), s=40, edgecolors="white", linewidths=1, zorder=2)
    for i, name in enumerate(names):
        if i not in chosen:
            ax.annotate(name.split("_")[0], scores[i], fontsize=7, color="0.45", xytext=(3, 3),
                        textcoords="offset points")
    for i in order:
        ax.scatter(*scores[i], s=110, facecolors="none", edgecolors="0.1", linewidths=1.2, zorder=3)
    colorbar = fig.colorbar(points, ax=ax, shrink=0.8, pad=0.01)
    colorbar.set_label("Gestational week")
    ax.set_xlabel(f"PC1 ({100 * ratio[0]:.1f} % of code variance)")
    ax.set_ylabel(f"PC2 ({100 * ratio[1]:.1f} % of code variance)")
    ax.set_title(title)
    ax.grid(color="0.92", zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    for k, i in enumerate(order):
        view = fig.add_subplot(grid[1 + k // columns, k % columns])
        view.imshow(images[i])
        view.set_axis_off()
        view.set_title(f"{LETTERS[k]}  {names[i].replace('_semanas', ' wk')}\n{chosen[i]}", fontsize=10)

    # Letters are placed once the layout is final, since they are placed in pixels.
    fig.tight_layout()
    fig.canvas.draw()
    place_letters(ax, scores, [i for i in range(len(names)) if i not in chosen], order)
    fig.savefig(out_path)
    plt.close(fig)


def place_letters(ax, scores, labelled, order):
    """Letter each point in `order`, each where it is farthest from the other points,
    the case-number labels of `labelled` and the letters placed before it, with a
    leader line when it had to move away from its point."""
    to_px = ax.transData.transform
    points = to_px(scores)
    obstacles = list(points) + [points[i] + np.array([8.0, 6.0]) for i in labelled]
    directions = [np.array([np.cos(a), np.sin(a)]) for a in np.linspace(0, 2 * np.pi, 16, endpoint=False)]
    for k, i in enumerate(order):
        best, best_clearance = None, -1.0
        for radius in (14.0, 24.0, 36.0, 50.0):
            for d in directions:
                spot = points[i] + radius * d
                clearance = min(np.linalg.norm(o - spot) for o in obstacles if o is not points[i])
                if clearance > best_clearance:
                    best, best_clearance = spot, clearance
            if best_clearance >= 14.0:
                break
        obstacles.append(best)
        offset = best - points[i]
        ax.annotate(LETTERS[k], scores[i], xytext=offset, textcoords="offset pixels",
                    ha="center", va="center", fontsize=11, fontweight="bold", color="0.1", zorder=5,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85),
                    arrowprops=dict(arrowstyle="-", color="0.3", lw=0.7, shrinkA=0, shrinkB=5)
                    if np.linalg.norm(offset) > 15 else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="Cohort auto-decoder run folder (checkpoints/, apply/).")
    parser.add_argument("--output", default=None, help="PNG to write (default: <run>/code_pca.png).")
    parser.add_argument("--pairs", type=int, default=3, help="How many closest pairs to show.")
    parser.add_argument("--panel-px", type=int, default=260, help="Width and height of each view in pixels.")
    parser.add_argument("--similarity", default=None,
                        help="similarity.json mapping the run's world frame to mm (a shape-model-frame run, "
                             "see scripts/SSM/target_to_ssm_frame.py); default: world is mm already.")
    args = parser.parse_args()

    names, codes = load_codes(args.run)
    scores, ratio = pca(codes)
    chosen = choose_points(scores, args.pairs)
    to_mm = similarity_map(args.similarity) if args.similarity else None
    images = render_thumbnails(args.run, names, chosen, args.panel_px, to_mm)
    out_path = args.output or os.path.join(args.run, "code_pca.png")
    plot(names, scores, ratio, chosen, images,
         f"{os.path.basename(os.path.normpath(args.run))}: auto-decoder codes ({len(names)} members)", out_path)
    with open(out_path[:-4] + ".csv", "w") as f:
        f.write("member,pc1,pc2,chosen\n")
        f.writelines(f"{n},{scores[i, 0]:.5f},{scores[i, 1]:.5f},{chosen.get(i, '')}\n" for i, n in enumerate(names))
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
