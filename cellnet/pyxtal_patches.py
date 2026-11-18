"""Runtime fixes for the shared PyXtal install (the env's site-packages is not edited).

representation.to_pyxtal shadowing bug
--------------------------------------
``representation.to_pyxtal`` builds the sites component by component. Inside the per-site
loop it looks for an existing molecule type with

    for mol_id, molecule in enumerate(struc.molecules):

which overwrites the outer ``molecule`` (the component's own molecule). When a component
after the first has two or more sites (cocrystal 'A.B.B' with B on '4e,4e'), its second
site is built with the previous component's molecule: a nitromethane site carrying CL-20's
36 atoms. QRS then swaps in a nitromethane conformer without updating the atom list, and
the contact check fails with "cannot reshape array of size ... into shape (36,0,36)".

``apply()`` recompiles the method with the loop variable renamed. It is a no-op when the
pattern is absent (fixed upstream). It is applied when cellnet.lattice_conf_pipeline is
imported (main process and the fork-started QRS grid workers) and, for processes started
with "spawn" (PyXtal's optimisation pool), by runtime/sitecustomize.py on PYTHONPATH.
"""
from __future__ import annotations

import inspect
import textwrap

_BAD = (
    "for mol_id, molecule in enumerate(struc.molecules):\n"
    "{ind}    if str(site.molecule) == str(molecule):"
)
_GOOD = (
    "for mol_id, _known_mol in enumerate(struc.molecules):\n"
    "{ind}    if str(site.molecule) == str(_known_mol):"
)

_applied = False


def apply() -> bool:
    """Patch pyxtal.representation.representation.to_pyxtal in this process.
    Returns True when the fix is active (patched now or earlier)."""
    global _applied
    if _applied:
        return True
    import importlib

    rep_mod = importlib.import_module("pyxtal.representation")  # the module, not the class

    cls = rep_mod.representation
    fn = cls.to_pyxtal
    if getattr(fn, "_cellnet_fixed", False):
        _applied = True
        return True
    src = textwrap.dedent(inspect.getsource(fn))
    # indentation of the 'if' line relative to the 'for' line inside the dedented source
    line = next((ln for ln in src.splitlines() if "for mol_id, molecule in enumerate(struc.molecules):" in ln), None)
    if line is None:
        return False  # pattern gone: nothing to fix
    ind = line[: len(line) - len(line.lstrip())]
    bad, good = _BAD.format(ind=ind), _GOOD.format(ind=ind)
    if bad not in src:
        return False
    ns: dict = {}
    exec(compile(src.replace(bad, good), f"<cellnet patched {rep_mod.__file__}>", "exec"), rep_mod.__dict__, ns)
    new = ns["to_pyxtal"]
    new._cellnet_fixed = True
    cls.to_pyxtal = new
    _applied = True
    return True
