"""Interactive VTK windows.

``core`` holds the pieces every window shares - window, themes, mesh view,
colouring, text overlay, markers, slider panel, input routing - and knows
nothing about shape models or landmarks. Each sibling package is one tool
composed from them (``set_landmarks``: the landmark picker behind
``scripts/SSM/place_landmarks.py``).
"""
