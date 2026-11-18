"""Relax the experimental crystal with a run's own force field (CHARMM + Sage 2.1 and the
parameters.xml the run generated), box free and box frozen, and write the results as
reference CIFs for scoring. A search that finds these has done its job; whether they
are within 0.3 of experiment is a property of the force field.

    python EM/relax_ref.py CODE --db EM/CODE/CODE.db --runs EM_blind/CODE --out EM_blind/CODE/ref_sage
Writes <out>/exp_relaxed_box_free.cif, exp_relaxed_box_frozen.cif and summary.json
(max displacement of each from experiment)."""
import argparse, json, os, sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from pymatgen.analysis.structure_matcher import StructureMatcher  # noqa: E402
from cellnet.lattice_conf_pipeline import _load_pyxtal_from_db  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("code"); ap.add_argument("--db", required=True)
    ap.add_argument("--runs", required=True, help="folder holding the run(s); first parameters.xml found is used")
    ap.add_argument("--out", required=True)
    ap.add_argument("--style", default="openff", choices=["openff", "gaff"],
                    help="force-field style (default openff = Sage 2.1, as the search; gaff for comparison)")
    a = ap.parse_args()
    from pyocse.parameters import ForceFieldParameters
    from pyxtal.optimize.common import optimizer
    os.environ.setdefault("PYOCSE_OPENFF", "2.1")
    exp = _load_pyxtal_from_db(a.code, db_path=a.db)                    # the setting the search relaxes in
    ref = _load_pyxtal_from_db(a.code, db_path=a.db, keep_special=True).to_pymatgen(); ref.remove_species(["H"])
    smi = [m.smile for m in exp.molecules]
    xmls = sorted(Path(a.runs).glob("**/conf_qrs_*/parameters.xml")) if a.style == "openff" else []
    out = Path(a.out); (out / "calc").mkdir(parents=True, exist_ok=True)
    params = ForceFieldParameters(smi, style=a.style, ncpu=1)
    p0 = params.load_parameters(str(xmls[0]))[0] if xmls else params.params_init.copy()
    cwd = os.getcwd(); os.chdir(out)
    ff = params.get_ase_charmm(p0); ff.write_charmmfiles(base="calc/pyxtal"); info = ff.get_atom_info()
    os.chdir(cwd)
    sm = StructureMatcher(ltol=0.3, stol=0.3, angle_tol=5); sml = StructureMatcher(ltol=0.5, stol=0.7, angle_tol=15)
    # full-cell retry: a relaxed crystal that lost exact centring no longer reduces to the same primitive cell
    smf = StructureMatcher(ltol=0.5, stol=0.7, angle_tol=15, primitive_cell=False)
    summary = {"style": a.style, "parameters_from": str(xmls[0]) if xmls else "default"}
    for mode, opt_lat in (("box_free", True), ("box_frozen", False)):
        res = optimizer(exp.copy(), info, str(out / "calc"), tag=f"exp_{mode}", opt_lat=opt_lat, skip_mlp=True)
        if res is None:
            summary[mode] = "relaxation failed"; continue
        res["xtal"].to_file(str(out / f"exp_relaxed_{mode}.cif"))
        s = res["xtal"].to_pymatgen(); s.remove_species(["H"])
        r = sm.get_rms_dist(ref, s) or sml.get_rms_dist(ref, s) or smf.get_rms_dist(ref, s)
        summary[mode] = {"max_disp_from_exp": round(r[1], 3) if r else None}
        print(f"{a.code} {mode}: Sage-relaxed experiment is {summary[mode]['max_disp_from_exp']} from experiment", flush=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
