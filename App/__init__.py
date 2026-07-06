"""
App package layout
==================

Each subpackage is self-contained and runnable alone (``python -m App.<pkg>``):

- ``ui`` — shell: ribbon split + dock grid + empty panel hosts
- ``carto`` — study I/O (unchanged)
- ``table_pkg`` — tkinter editable table widget
- ``viewer3d`` — OpenGL mesh viewer panel
- ``plotting`` — signal figure layout + trace rendering style
- ``utility`` / ``ml`` — signal processing and ML inference helpers

``main_app.py`` is the organizer: mounts panels via ``mediators``, wires
callbacks, and holds app logic (navigation, delta, ML predictions).
"""
