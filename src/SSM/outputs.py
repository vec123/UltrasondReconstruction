"""The files a GPMM fit writes: the output contract of src/SSM.

Code outside the model (src/evaluation) reads a fit's folder through these
names, so this module imports nothing but the standard library.
"""

from dataclasses import dataclass


@dataclass
class PipelineOutputFiles:
    mean: str = "mean.vtp"
    init_shape: str = "init_shape.vtp"
    surface: str = "surface.vtp"
    rigid_alignment: str = "rigid_alignment.vtp"
    refined_vertices: str = "refined_vertices.vtp"
    original_landmarks: str = "orig_langmarks.vtp"
    transformed_landmarks: str = "new_target_landmarks.vtp"
    matching_field: str = "matching_field.vtp"
    template_aligned: str = "template_aligned.vtp"
    gmm_2_template_transform: str = "gmm_2_template.npz"
    # One family for every fit, keyed by the fit's own name. A desugared gamma
    # sweep names its fits "gamma_5", so it still writes
    # nicp_refined_vertices_gamma_5.vtp - the filenames never moved, and no
    # gamma-aware code is needed to keep them.
    nicp_fit: str = "nicp_refined_vertices_{name}.vtp"
    nicp_fit_matching: str = "nicp_matching_field_{name}.vtp"
    # Copies of nicp_fit / rigid_alignment with the correspondence field
    # (save_correspondence_field): correspondence_id, colored with one range so
    # that equal colors mark a correspondence, and correspondence_weight, the
    # mask. One pair per fit, since the matches depend on which fit produced them.
    nicp_fit_colors: str = "nicp_refined_vertices_{name}_colors.vtp"
    nicp_fit_target_colors: str = "rigid_alignment_{name}_colors.vtp"
    nicp_fit_iteration: str = "nicp_refined_vertices_{name}_iter_{iteration:04d}.vtp"
    region_vertices: str = "n_region_verts.vtp"
    deformed_vertices: str = "deformed_vertices.vtp"
