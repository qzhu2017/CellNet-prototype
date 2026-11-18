#!/usr/bin/env python3
"""Blind cocrystal CSP with CellNet: predicted space groups, per-draw densities, and every
way to place each component on general positions or special sites.

Input is the formula unit as a dotted SMILES with the stoichiometry written out
('HMX.CL20.CL20' = 1 HMX : 2 CL-20). Nothing comes from the experimental structure; the
reference CIF is used only by the scoring job that runs after all searches have ended.

Rules (fixed in advance, same spirit as run_blind.py):
  * space groups: the auxiliary head's p(Hall | formula-unit graph), summed over settings,
    one Hall setting per space group, until 90% coverage (at most --max-groups); plus the
    --special-groups most probable further groups that have a special site some
    component fits;
  * Z′ is counted in formula units (FU) per asymmetric unit: ½ and 1;
  * each component goes either on general positions or on special sites of one site
    symmetry its conformers can adopt (SiteFit, SMILES only; inversion centres use the
    built centrosymmetric conformers); an inversion centre holds one molecule;
  * arrangements equivalent under the normalizer are searched once;
  * the all-general Z′ = 1 arrangement of a group gets boxes in proportion to the group's
    probability (--total-boxes, at least --min-boxes); every other arrangement gets
    --special-boxes, at most --max-special-runs per group (fewest molecules first).

Jobs: one Slurm array per system (cc-CODE). Each run is sliced into tasks of at most
--chunk-boxes boxes (run_pipeline --conf-slice); each task scores itself against the
experimental CIF after its search has ended (sg_*/rematch_exp_0.7/summary.json).
EM/cocrystal_status.py collects the results so far.

  python scripts/run_cocrystal.py --targets EM_cocrystal/targets.json --plan-only
  python scripts/run_cocrystal.py --targets EM_cocrystal/targets.json --codes QIQWIZ --submit
"""
from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from run_blind import AUX, FLOW, PROTOCOL, SiteFit, _normalizer_letter_maps, make_tasks  # noqa: E402


def component_options(g, n_mol, fits, max_sites):
    """Ways to put n_mol molecules of one component in the cell: general-position copies,
    or special sites of one site symmetry (an inversion centre holds one molecule; an axis
    or plane can hold several). Returns [(labels, site_symmetry)]."""
    n_gp = len(g[0])
    opts = []
    if n_mol % n_gp == 0 and 0 < n_mol // n_gp <= max_sites:
        opts.append(([g[0].get_label()] * (n_mol // n_gp), "1"))
    by_symm = {}
    for wp in g:
        if wp.multiplicity < n_gp and fits(wp):
            by_symm.setdefault(wp.get_site_symmetry_object().name, []).append(wp)
    for symm, wps in by_symm.items():
        wps = sorted(wps, key=lambda w: w.letter)

        def extend(start, left, chosen):
            if left == 0:
                yield list(chosen)
                return
            if len(chosen) >= max_sites:
                return
            for i in range(start, len(wps)):
                w = wps[i]
                if w.multiplicity > left:
                    continue
                chosen.append(w)
                yield from extend(i if w.get_dof() > 0 else i + 1, left - w.multiplicity, chosen)
                chosen.pop()

        for combo in extend(0, n_mol, []):
            opts.append(([w.get_label() for w in combo], symm))
    return opts


def cocrystal_arrangements(hall, counts, fits_list, zfus=(0.5, 1.0), max_records=4):
    """All site arrangements of a cocrystal in one Hall setting.
    Returns [(per-component labels, zfu, per-component site symmetry)], fewest molecules first."""
    from pyxtal.symmetry import Group

    g = Group(hall, use_hall=True)
    n_gp = len(g[0])
    by_label = {wp.get_label(): wp for wp in g}
    maps = _normalizer_letter_maps(g)
    out, seen = [], set()
    for zfu in zfus:
        n_fu = zfu * n_gp
        if abs(n_fu - round(n_fu)) > 1e-6:
            continue
        n_fu = int(round(n_fu))
        per = [component_options(g, s * n_fu, fits, max_records) for s, fits in zip(counts, fits_list)]
        for choice in itertools.product(*per):
            labels = [c[0] for c in choice]
            if sum(len(l) for l in labels) > max_records:
                continue
            fixed = [l for ls in labels for l in ls if by_label[l].get_dof() == 0]
            if len(fixed) != len(set(fixed)):  # two molecules on one inversion centre
                continue
            letters = [[by_label[l].letter for l in ls] for ls in labels]
            key = min(tuple(tuple(sorted(m[x] for x in ls)) for ls in letters) for m in maps)
            if (zfu, key) in seen:
                continue
            seen.add((zfu, key))
            out.append((labels, zfu, [c[1] for c in choice]))
    out.sort(key=lambda a: (sum(len(l) for l in a[0]), a[1], any(s != "1" for s in a[2])))
    return out


def plan_target(code, fu_smiles, args):
    from pyxtal.symmetry import Group

    from cellnet.aux_inference import predict_space_groups
    from cellnet.lattice_conf_pipeline import split_components

    comps, counts = split_components(fu_smiles)
    fits_list = [SiteFit(c) for c in comps]
    probs = predict_space_groups(args.aux_checkpoint, fu_smiles)
    p_num, best_hall, p_hall = {}, {}, {}
    for hall, _zp, p in probs:
        p_hall[hall] = p_hall.get(hall, 0.0) + p
    for h, p in p_hall.items():
        n = Group(h, use_hall=True).number
        p_num[n] = p_num.get(n, 0.0) + p
        if n not in best_hall or p > p_hall[best_hall[n]]:
            best_hall[n] = h
    ranked = sorted(p_num, key=lambda n: -p_num[n])
    planned, cum = [], 0.0
    for n in ranked:
        if len(planned) >= args.max_groups or (planned and cum >= args.sg_top_p):
            break
        planned.append(n)
        cum += p_num[n]
    extra = []
    for n in ranked:
        if len(extra) >= args.special_groups:
            break
        if n in planned:
            continue
        g = Group(best_hall[n], use_hall=True)
        if any(w.multiplicity < len(g[0]) and any(f(w) for f in fits_list) for w in g):
            extra.append(n)
    psum = sum(p_num[n] for n in planned)
    runs = []
    for n in planned + extra:
        h = best_hall[n]
        arr = cocrystal_arrangements(h, counts, fits_list, max_records=args.max_records)
        general = [a for a in arr if all(s == "1" for s in a[2]) and a[1] == 1.0]
        others = [a for a in arr if a not in general]
        # extra groups were added for their special sites only
        for labels, zfu, symms in (general if n in planned else []):
            boxes = max(args.min_boxes, round(args.total_boxes * p_num[n] / psum))
            runs.append((h, n, labels, zfu, symms, boxes))
        for labels, zfu, symms in others[: args.max_special_runs]:
            runs.append((h, n, labels, zfu, symms, args.special_boxes))
    groups = []
    for h, n, labels, zfu, symms, boxes in runs:
        spec = ";".join(",".join(l) for l in labels)
        tag = f"sg_{h}_" + "_".join("".join(l) for l in labels)
        groups.append({"hall": h, "number": n, "symbol": Group(h, use_hall=True).symbol, "zprime_fu": zfu,
                       "prob": round(p_num[n], 4), "boxes": boxes, "wyckoff": spec, "site_symmetry": symms,
                       "tag": tag})
    note = {c: sorted(k for k, v in f.cache.items() if v) for c, f in zip(comps, fits_list)}
    return {"code": code, "smiles": fu_smiles, "components": comps, "stoichiometry": counts,
            "total_probability": cum, "groups": groups, "site_fits": note,
            "top10_groups": [[n, round(p_num[n], 4)] for n in ranked[:10]]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--targets", required=True, help="JSON {code: {smiles: FU SMILES, ref_cif: path}}")
    ap.add_argument("--codes", default=None, help="comma list (default: all targets)")
    ap.add_argument("--out", default=str(ROOT / "EM_cocrystal"))
    ap.add_argument("--checkpoint", default=str(FLOW))
    ap.add_argument("--aux-checkpoint", default=str(AUX))
    ap.add_argument("--sg-top-p", type=float, default=0.9)
    ap.add_argument("--max-groups", type=int, default=10)
    ap.add_argument("--special-groups", type=int, default=3)
    ap.add_argument("--max-records", type=int, default=4, help="most molecules in the asymmetric unit")
    ap.add_argument("--total-boxes", type=int, default=128)
    ap.add_argument("--min-boxes", type=int, default=12)
    ap.add_argument("--special-boxes", type=int, default=12)
    ap.add_argument("--max-special-runs", type=int, default=3)
    ap.add_argument("--chunk-boxes", type=int, default=6, help="boxes per array task (runs are sliced)")
    ap.add_argument("--partition", default="Orion",
                    help="Nebula has only older Intel nodes (Skylake/Cascade Lake); Apus is shared with labmates")
    ap.add_argument("--constraint", default="genoa|turin",
                    help="node types: AMD Genoa/Turin ran a 6-box task 1.5x faster than Cascade Lake (2.1 vs 3.2 h); '' = any")
    ap.add_argument("--cpus", type=int, default=24)
    ap.add_argument("--mem", default="40G", help="measured peak ~20 GB at 24 CPUs")
    ap.add_argument("--throttle", type=int, default=2, help="array tasks running at once per system")
    ap.add_argument("--nice", type=int, default=200, help="below the user's other jobs: cocrystals fill leftover CPUs")
    ap.add_argument("--k", type=int, default=192)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--sage", default="2.1")
    ap.add_argument("--time", default="2-00:00:00", help="tasks take 2-7 h; <= 2 days keeps Nebula eligible")
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--submit", action="store_true")
    args = ap.parse_args()

    targets = json.loads(Path(args.targets).read_text())
    codes = args.codes.split(",") if args.codes else list(targets)
    for code in codes:
        t = targets[code]
        pl = plan_target(code, t["smiles"], args)
        print(f"\n{code}: {' + '.join(f'{n}x {c}' for c, n in zip(pl['components'], pl['stoichiometry']))}")
        print(f"  site symmetries each component fits: {pl['site_fits']}")
        print(f"  groups (p summed over settings): {pl['top10_groups'][:6]} ... coverage {pl['total_probability']:.2f}")
        for g in pl["groups"]:
            print(f"  {g['symbol']:<12} Hall {g['hall']:<4} Z'(FU)={g['zprime_fu']:g}  p={g['prob']:.3f}  "
                  f"boxes={g['boxes']:<3} sites {g['wyckoff']}  ({'/'.join(g['site_symmetry'])})")
        print(f"  {len(pl['groups'])} runs, {sum(g['boxes'] for g in pl['groups'])} boxes")
        if args.plan_only:
            continue
        out = Path(args.out).resolve() / code
        out.mkdir(parents=True, exist_ok=True)
        tasks = make_tasks(pl["groups"], args.chunk_boxes)
        pl["tasks"] = tasks
        tdir = out / "tasks"
        tdir.mkdir(exist_ok=True)
        (out / "logs").mkdir(exist_ok=True)
        ref_cif = str(Path(t["ref_cif"]).resolve())
        for k, tk in enumerate(tasks):
            g = pl["groups"][tk["run"]]
            cmd = ["python", "scripts/run_pipeline.py", "--checkpoint", args.checkpoint,
                   "--aux-checkpoint", args.aux_checkpoint, "--random-template",
                   "--hall", str(g["hall"]), "--zprime", str(g["zprime_fu"]), "--wyckoff", f"'{g['wyckoff']}'",
                   "--smiles", f"'{t['smiles']}'", "--output-dir", str(out), "--tag", tk["tag"],
                   "--sage", args.sage, "--k", str(args.k), "--seed", str(args.seed),
                   "--max-conf-runs", str(g["boxes"])]
            if tk["n_slices"] > 1:
                cmd += ["--conf-slice", f"{tk['slice']}/{tk['n_slices']}"]
            cmd += PROTOCOL
            (tdir / f"task_{k}.sh").write_text(
                f"# {code} task {k}: {g['symbol']} Hall {g['hall']} Z'(FU)={g['zprime_fu']:g} sites {g['wyckoff']} "
                f"slice {tk['slice']}/{tk['n_slices']} of {g['boxes']} boxes\n"
                # PyXtal runtime fix for every process incl. spawn workers (cellnet/pyxtal_patches.py)
                f"export PYTHONPATH={ROOT / 'runtime'}${{PYTHONPATH:+:$PYTHONPATH}}\n"
                f"mkdir -p {out / tk['tag']}\n"
                + " ".join(cmd)
                + " --lattice-qrs-nproc $((NPROC/2)) --conf-nproc $NPROC"
                + f" 2>&1 | tee {out / tk['tag']}/pipeline.log\n"
                # post-hoc scoring of this task only (the search above never sees the reference)
                + f"python EM/rematch.py {tk['tag']} --dir {out} --nproc $NPROC --cutoff 0.7 "
                  f"--ref-cif {ref_cif} --label exp || true\n")
        job = out / "run_array.sh"
        job.write_text(
            "#!/bin/bash\n"
            f"#SBATCH --partition={args.partition}\n#SBATCH --nodes=1\n#SBATCH --ntasks=1\n"
            f"#SBATCH --cpus-per-task={args.cpus}\n#SBATCH --mem={args.mem}\n"
            + (f"#SBATCH --constraint={args.constraint}\n" if args.constraint else "")
            + f"#SBATCH --time={args.time}\n#SBATCH --job-name=cc-{code}\n#SBATCH --nice={args.nice}\n"
            f"#SBATCH --array=0-{len(tasks) - 1}%{args.throttle}\n"
            f"#SBATCH --output={out}/logs/%A_%a.out\n"
            "set -o pipefail\n"
            f"cd {ROOT}\n"
            'source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate cellnet\n'
            'export PATH="$HOME/CHARMM/bin:$PATH" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1\n'
            f"NPROC=${{SLURM_CPUS_PER_TASK:-{args.cpus}}}\n"
            f'source {tdir}/task_${{SLURM_ARRAY_TASK_ID}}.sh\n')
        print(f"  {len(tasks)} array tasks ({args.chunk_boxes} boxes each at most), "
              f"{args.throttle} at a time, {args.cpus} CPUs each")
        if args.submit:
            pl["array_job"] = subprocess.run(["sbatch", "--parsable", str(job)], capture_output=True, text=True,
                                             check=True).stdout.strip()
            print(f"  submitted array {pl['array_job']}")
        (out / "cocrystal_plan.json").write_text(json.dumps(pl, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))

if __name__ == "__main__":
    main()
