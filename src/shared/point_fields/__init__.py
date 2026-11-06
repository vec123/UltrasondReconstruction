"""Point fields added to a surface mesh, one kind at a time.

    kinds.py         the field kinds (normals, edge, ...): settings + apply(polydata)
    stage.py         the point_fields stage: one mesh, a list of kinds, in place or to a new file
    registrations.py the stage, by name (joined by scripts/pipeline/registry.py)
"""
