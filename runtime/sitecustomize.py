"""Loaded at start-up by every Python process whose PYTHONPATH contains this folder
(the cocrystal array jobs set it), including PyXtal's spawn-started workers. Applies the
CellNet runtime fixes to PyXtal; see cellnet/pyxtal_patches.py."""
try:
    import importlib.util as _u
    import os as _os
    _spec = _u.spec_from_file_location(
        "cellnet_pyxtal_patches",
        _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "cellnet", "pyxtal_patches.py"))
    _m = _u.module_from_spec(_spec)
    _spec.loader.exec_module(_m)
    _m.apply()
except Exception:  # never stop an interpreter from starting
    pass
