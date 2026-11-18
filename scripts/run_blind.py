#!/usr/bin/env python3
"""Blind CSP: predict the space group(s), then run CellNet in each, with per-draw densities.

The auxiliary model (scripts/train_aux.py) gives p(Hall, Z′ | molecule). The most probable
(Hall, Z′) classes are taken until their summed probability reaches --sg-top-p (at most
--max-groups), and the box budget (--total-boxes) is split between them in proportion to
probability (at least --min-boxes each). All runs go into one Slurm array (bl-CODE): each
run is sliced into tasks of at most --chunk-boxes boxes (run_pipeline --conf-slice), each
task scores itself against the CSD entry when its search ends, and a scoring job runs after
the whole array (also against the Sage-relaxed experiment). In every
job the flow draws get their own density from the density mixture (--aux-checkpoint), the
packing template is a random placement in that class (not the CSD structure), and the CSD
entry, if given, is used only to check matches.

  # plan only
  python scripts/run_blind.py YIFCAU01 --db EM_z12/YIFCAU01/YIFCAU01.db --out EM_blind
  # submit one Slurm array (24-CPU tasks on Orion/Nebula, low priority) + its scoring job
  python scripts/run_blind.py YIFCAU01 --db EM_z12/YIFCAU01/YIFCAU01.db --out EM_blind --submit

Results: <out>/<CODE>/sg_<Hall>_z<Z′>[_c<slice>]/ ; plan (with the task list) in
<out>/<CODE>/blind_plan.json; task scripts in tasks/, Slurm logs in logs/
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def _first_existing(*paths):
    return next((p for p in paths if p.exists()), paths[0])


# shipped checkpoints first; outputs/ holds them after a local (re)training run
FLOW = _first_existing(ROOT / "checkpoints/cellnet_flow_z12/best.pt",
                       ROOT / "outputs/joint_z12/flow_scratch/given_hz_conditional_lattice_flow_gnn/best.pt")
AUX = _first_existing(ROOT / "checkpoints/aux_sg_density/best.pt", ROOT / "outputs/aux_sg_density/best.pt")

# Same search as benchmark/run_code.sh (kept in sync by hand; run_code.sh is not edited
# while jobs read it), but strictly blind: --strict-posthoc-search keeps the experimental
# structure out of the search entirely, so no box stops early on a match and the
# frozen-box pass (--adaptive-relax) always runs. Matching is done afterwards by a
# separate scoring job (EM/rematch.py) that starts when all group jobs have finished.
PROTOCOL = [
    "--strict-posthoc-search", "--qrs-stages", "12", "--qrs-samples-per-stage", "1024",
    "--max-lattice-qrs-lambda-ratio", "8.0", "--conf-ngen", "20", "--conf-npop", "96",
    "--conf-selection", "multichannel", "--conf-channel-lambda", "6", "--conf-channel-qrs-loss", "6",
    "--conf-channel-density", "4", "--conf-channel-disagreement", "4",
    "--conf-channel-axis-alternates", "12", "--axis-alternate-top", "6", "--conf-qrs-origin-only",
    "--conf-restarts", "1", "--conf-seed-stride", "100000", "--cell-blind", "--relax-lattice",
    "--adaptive-relax", "--no-conf-check-stable",
]


def plan(probs, top_p, max_halls, total_boxes, min_boxes, zprimes=(1.0, 2.0), z2_share=0.4):
    """Rank Hall settings by p(Hall) = sum over Z′ (the space-group head ranks the right
    setting well, but its Z′ choice is weak: AUC 0.71 on the test set), take the top
    settings until their probability reaches top_p (at most max_halls), and run each with
    every Z′ in ``zprimes``. Boxes go to settings in proportion to p(Hall); within a setting
    Z′ = 2 gets ``z2_share`` of them."""
    p_hall = {}
    for hall, _zp, p in probs:
        p_hall[hall] = p_hall.get(hall, 0.0) + p
    halls, cum = [], 0.0
    for hall in sorted(p_hall, key=lambda h: -p_hall[h]):
        if len(halls) >= max_halls or (halls and cum >= top_p):
            break
        halls.append(hall)
        cum += p_hall[hall]
    psum = sum(p_hall[h] for h in halls)
    groups = []
    for h in halls:
        n_h = total_boxes * p_hall[h] / psum
        for z in zprimes:
            share = (z2_share if z == 2.0 else 1.0 - z2_share) if len(zprimes) > 1 else 1.0
            groups.append({"hall": h, "zprime": z, "prob": round(p_hall[h], 4),
                           "boxes": max(min_boxes, round(n_h * share))})
    return groups, cum


def _normalizer_letter_maps(g):
    """Wyckoff-letter permutations under the group's Euclidean normalizer (origin shifts
    and other symmetry-preserving moves). Two site combinations related by one of these
    are the same crystal in a different origin, so only one is searched."""
    letters = [wp.letter for wp in g][::-1]  # a, b, c, ...
    maps = [{l: l for l in letters}]
    try:
        for row in g.get_alternatives()["Transformed WP"]:
            images = row.split()
            if len(images) == len(letters):
                maps.append(dict(zip(letters, images)))
    except Exception:  # no table for this setting: keep every combination
        pass
    return maps


class SiteFit:
    """Can the molecule sit on a site with this site symmetry? Decided from the SMILES alone:
    the search's own conformer generator must give at least one conformer whose symmetry
    contains the site symmetry (e.g. a centrosymmetric conformer for an inversion centre,
    a C2 conformer for a 2-fold axis). Cached per site point group."""

    def __init__(self, smiles):
        self.smiles = smiles.strip()
        self.cache = {}

    def __call__(self, wp):
        key = wp.get_site_symmetry_object().name.replace(".", "")
        if key not in self.cache:
            from cellnet.lattice_conf_pipeline import _pregen_component_pool

            try:
                pool, _ = _pregen_component_pool(self.smiles, [wp])
                self.cache[key] = bool(pool)
            except Exception as exc:  # conformer generation failure: treat as no fit
                print(f"  site-symmetry check {key} failed ({exc}); treated as no fit")
                self.cache[key] = False
        return self.cache[key]


def special_site_runs(hall, zprimes, fits, max_sites=4):
    """Every way to fill one Hall setting with molecules on special sites, for each Z′.

    Z′ fixes the molecules per cell, Z = Z′ x (general-position multiplicity). Each run puts
    every molecule on a special site of one site symmetry that the molecule can adopt
    (``fits``): e.g. P2_1/c Z′ = 1/2 -> 2a; Z′ = 1 -> 2a+2b, 2a+2c, 2a+2d; Fdd2 Z′ = 1/2 -> 8a;
    Z′ = 1 -> 8a+8a (two molecules on 2-fold axes). A site without free parameters (an
    inversion centre) holds at most one molecule; a site with free parameters (an axis or a
    plane) can hold several. Combinations equivalent under the normalizer are searched once.
    At most ``max_sites`` molecules in the asymmetric unit.
    Returns [(labels, zprime, site_symmetry)], fewest molecules first.
    """
    from pyxtal.symmetry import Group

    g = Group(hall, use_hall=True)
    n_gp = len(g[0])
    maps = _normalizer_letter_maps(g)
    by_symm = {}
    for wp in g:
        if wp.multiplicity < n_gp and fits(wp):
            by_symm.setdefault(wp.get_site_symmetry_object().name, []).append(wp)
    runs, seen = [], set()
    for zp in sorted(zprimes):
        z_mols = zp * n_gp
        if abs(z_mols - round(z_mols)) > 1e-6:
            continue
        z_mols = int(round(z_mols))
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
                    # an inversion centre (no free parameter) takes one molecule only
                    nxt = i if w.get_dof() > 0 else i + 1
                    chosen.append(w)
                    yield from extend(nxt, left - w.multiplicity, chosen)
                    chosen.pop()

            for combo in extend(0, z_mols, []):
                letters = [w.letter for w in combo]
                key = min(tuple(sorted(m[l] for l in letters)) for m in maps)
                if (zp, key) in seen:
                    continue
                seen.add((zp, key))
                runs.append(([w.get_label() for w in combo], zp, symm))
    return runs


def make_tasks(groups, chunk, special_chunk=None):
    """Split every run into array tasks of at most `chunk` boxes (`special_chunk` for runs
    with molecules on special sites, whose DFS boxes are slower) and order them so the most
    probable group finishes first: groups in plan order; inside a Hall setting, slice 0 of
    every run, then slice 1, ... Returns [{run, slice, n_slices, tag}]."""
    by_hall = {}
    for r, g in enumerate(groups):
        by_hall.setdefault(g["hall"], []).append(r)
    tasks = []
    for runs in by_hall.values():
        n_sl = {}
        for r in runs:
            c = special_chunk if (special_chunk and groups[r].get("wyckoff")) else chunk
            n_sl[r] = max(1, -(-groups[r]["boxes"] // c))
        for i in range(max(n_sl.values())):
            for r in runs:
                if i < n_sl[r]:
                    tag = groups[r]["tag"] + (f"_c{i}" if n_sl[r] > 1 else "")
                    tasks.append({"run": r, "slice": i, "n_slices": n_sl[r], "tag": tag})
    return tasks


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("code", help="CSD code (for matching and folder name); with --smiles any name")
    ap.add_argument("--db", default=None, help="PyXtal db holding the CSD entry (matching only)")
    ap.add_argument("--smiles", default=None, help="SMILES (default: from --db)")
    ap.add_argument("--out", default=str(ROOT / "EM_blind"))
    ap.add_argument("--checkpoint", default=str(FLOW))
    ap.add_argument("--aux-checkpoint", default=str(AUX))
    ap.add_argument("--sg-top-p", type=float, default=0.9)
    ap.add_argument("--max-halls", type=int, default=4, help="most Hall settings to run")
    ap.add_argument("--zprimes", default="1,2", help="Z′ values to run in each Hall setting")
    ap.add_argument("--z2-share", type=float, default=0.4, help="share of a setting's boxes for Z′ = 2")
    ap.add_argument("--groups", default=None, metavar="HALL:ZP,...",
                    help="skip the space-group prediction and run these (Hall, Z′), e.g. 143:2 "
                    "(tests the per-draw density alone)")
    ap.add_argument("--special", choices=["auto", "off"], default="auto",
                    help="special-site branch: also search arrangements with molecules on special "
                    "sites (inversion centres, 2-fold axes, mirrors, ...) whose site symmetry some "
                    "conformer of the molecule can adopt; decided from the SMILES alone")
    ap.add_argument("--only-special", action="store_true",
                    help="submit only the special-site runs (general-position runs already submitted); "
                    "plan goes to blind_plan_special.json")
    ap.add_argument("--special-groups", type=int, default=3,
                    help="besides the planned Hall settings, the most probable settings that have a "
                    "special site the molecule fits")
    ap.add_argument("--special-zprimes", default="0.5,1,2", help="Z′ values for special-site arrangements")
    ap.add_argument("--special-max-sites", type=int, default=4, help="most molecules in the asymmetric unit")
    ap.add_argument("--max-special-runs", type=int, default=8,
                    help="most special-site runs per Hall setting (fewest molecules per cell first)")
    ap.add_argument("--special-boxes", type=int, default=12, help="boxes per special-site run")
    ap.add_argument("--plan-only", action="store_true", help="print the plan; skip the job scripts")
    ap.add_argument("--total-boxes", type=int, default=128)
    ap.add_argument("--min-boxes", type=int, default=8)
    ap.add_argument("--k", type=int, default=192)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--sage", default="2.1")
    ap.add_argument("--submit", action="store_true", help="submit one Slurm job array (and its scoring job)")
    ap.add_argument("--chunk-boxes", type=int, default=6, help="boxes per array task (runs are sliced)")
    ap.add_argument("--special-chunk-boxes", type=int, default=2,
                    help="boxes per task for special-site (DFS) runs, whose boxes are slower")
    ap.add_argument("--partition", default="Orion",
                    help="Nebula has only older Intel nodes (Skylake/Cascade Lake); Apus is shared with labmates")
    ap.add_argument("--constraint", default="genoa|turin",
                    help="node types: AMD Genoa/Turin ran a 6-box task 1.5x faster than Cascade Lake (2.1 vs 3.2 h); '' = any")
    ap.add_argument("--cpus", type=int, default=24, help="measured: 24-CPU tasks keep ~64%% of CPUs busy, 48 only ~38%%")
    ap.add_argument("--mem", default="40G", help="measured peak ~20 GB at 24 CPUs, ~35 GB at 48")
    ap.add_argument("--time", default="2-00:00:00", help="<= 2 days keeps Nebula eligible")
    ap.add_argument("--throttle", type=int, default=4, help="array tasks running at once")
    ap.add_argument("--nice", type=int, default=200, help="below the user's other jobs: fills leftover CPUs")
    args = ap.parse_args()

    from cellnet.aux_inference import predict_space_groups

    smiles = args.smiles
    if smiles is None:
        if not args.db:
            sys.exit("give --smiles or --db")
        from cellnet.lattice_conf_pipeline import load_crystal_from_db

        smiles = load_crystal_from_db(args.db, args.code).smiles
    probs = predict_space_groups(args.aux_checkpoint, smiles)
    if args.groups:
        manual = [(int(g.split(":")[0]), float(g.split(":")[1])) for g in args.groups.split(",")]
        n = max(args.min_boxes, args.total_boxes // len(manual))
        groups = [{"hall": h, "zprime": z, "prob": None, "boxes": n} for h, z in manual]
        cum = float("nan")
    else:
        zps = tuple(float(z) for z in args.zprimes.split(","))
        groups, cum = plan(probs, args.sg_top_p, args.max_halls, args.total_boxes, args.min_boxes,
                           zprimes=zps, z2_share=args.z2_share)

    from pyxtal.symmetry import Group

    p_hall = {}
    for hall, _zp, p in probs:
        p_hall[hall] = p_hall.get(hall, 0.0) + p
    # Special-site branch: from the molecule (which site symmetries a conformer can adopt),
    # the Z′ values and the space groups only; never from the CSD entry.
    special_note = "off"
    special = []
    if args.special != "off":
        fits = SiteFit(smiles)
        sp_zps = [float(z) for z in args.special_zprimes.split(",")]
        # One Hall setting per space group: P2_1/c, P2_1/n and P2_1/a give the same crystals
        # in different cells, so the special-site runs go to the most probable setting only,
        # ranked by the group's probability summed over its settings.
        num = {h: Group(h, use_hall=True).number for h in p_hall}
        p_num, best_hall = {}, {}
        for h, p in p_hall.items():
            p_num[num[h]] = p_num.get(num[h], 0.0) + p
            if num[h] not in best_hall or p > p_hall[best_hall[num[h]]]:
                best_hall[num[h]] = h
        if args.groups:
            halls = list(dict.fromkeys(g["hall"] for g in groups))
        else:
            planned = list(dict.fromkeys(num[g["hall"]] for g in groups))
            halls = [best_hall[n] for n in planned]
        extra = 0
        for n in ([] if args.groups else sorted(p_num, key=lambda n: -p_num[n])):
            if extra >= args.special_groups:
                break
            h = best_hall[n]
            if h not in halls and any(w.multiplicity < len(Group(h, use_hall=True)[0]) and fits(w)
                                      for w in Group(h, use_hall=True)):
                halls.append(h)
                extra += 1
        for h in halls:
            runs = special_site_runs(h, sp_zps, fits, args.special_max_sites)
            for labels, zp, symm in runs[: args.max_special_runs]:
                special.append({"hall": h, "zprime": zp, "prob": round(p_num.get(num.get(h), 0.0), 4),
                                "boxes": args.special_boxes, "wyckoff": ",".join(labels),
                                "site_symmetry": symm, "n_arrangements": len(runs),
                                "tag": f"sg_{h}_site_{''.join(labels)}"})
        ok = sorted(k for k, v in fits.cache.items() if v)
        special_note = (f"molecule fits site symmetries {ok}" if ok else
                        f"no conformer fits any special site (tested {sorted(fits.cache)})")
    groups = special if args.only_special else groups + special
    if args.only_special:
        cum = float("nan")

    out = Path(args.out).resolve() / args.code
    plan_name = "blind_plan_special.json" if args.only_special else "blind_plan.json"
    out.mkdir(parents=True, exist_ok=True)
    print(f"{args.code}: {smiles}")
    print("space groups given (--groups)" if args.groups else
          f"predicted Hall settings (total p(Hall) {cum:.2f}), each run with Z′ = {args.zprimes}:")
    for g in groups:
        g["symbol"] = Group(g["hall"], use_hall=True).symbol
        g.setdefault("tag", f"sg_{g['hall']}_z{g['zprime']:g}")
        ptxt = "given" if g["prob"] is None else f"p(Hall)={g['prob']:.3f}"
        site = (f"  on special sites {g['wyckoff']} (site symmetry {g['site_symmetry']}, "
                f"{g['n_arrangements']} arrangements in this group)" if g.get("wyckoff") else "")
        print(f"  {g['symbol']:<12} Hall {g['hall']:<4} Z'={g['zprime']:g}  {ptxt}  boxes={g['boxes']}{site}")
    print(f"special-site branch: {special_note}")
    if args.plan_only:
        return
    (out / plan_name).write_text(json.dumps(
        {"code": args.code, "smiles": smiles, "groups": groups, "total_probability": cum,
         "top10": [[h, z, round(p, 4)] for h, z, p in probs[:10]]}, indent=1))

    tasks = make_tasks(groups, args.chunk_boxes, args.special_chunk_boxes)
    tdir = out / "tasks"
    tdir.mkdir(exist_ok=True)
    (out / "logs").mkdir(exist_ok=True)
    for k, tk in enumerate(tasks):
        g = groups[tk["run"]]
        cmd = [
            "python", "scripts/run_pipeline.py", "--checkpoint", args.checkpoint,
            "--aux-checkpoint", args.aux_checkpoint, "--random-template",
            "--hall", str(g["hall"]), "--zprime", str(g["zprime"]),
            "--output-dir", str(out), "--tag", tk["tag"], "--sage", args.sage,
            "--k", str(args.k), "--seed", str(args.seed), "--max-conf-runs", str(g["boxes"]),
        ]
        if tk["n_slices"] > 1:
            cmd += ["--conf-slice", f"{tk['slice']}/{tk['n_slices']}"]
        if g.get("wyckoff"):
            cmd += ["--wyckoff", g["wyckoff"]]
        elif abs(float(g["zprime"]) - 2.0) < 1e-6:
            # Z′ = 2: give the two copies their own conformers, alternating mirror-image
            # pairs and same-family pairs over the boxes (PyXtal QRS alone puts one
            # conformer on both copies).
            cmd += ["--z2-pairing", "alternate"]
        if args.db:
            cmd += ["--csd-code", args.code, "--db", str(Path(args.db).resolve())]
        else:
            cmd += ["--smiles", f"'{smiles}'"]
        cmd += PROTOCOL
        body = (
            f"# {args.code} task {k}: {g['symbol']} Hall {g['hall']} Z'={g['zprime']:g}"
            + (f" sites {g['wyckoff']}" if g.get("wyckoff") else "")
            + f" slice {tk['slice']}/{tk['n_slices']} of {g['boxes']} boxes\n"
            # PyXtal runtime fix for every process incl. spawn workers (cellnet/pyxtal_patches.py)
            + f"export PYTHONPATH={ROOT / 'runtime'}${{PYTHONPATH:+:$PYTHONPATH}}\n"
            + f"mkdir -p {out / tk['tag']}\n"
            + " ".join(cmd)
            + " --lattice-qrs-nproc $((NPROC/2)) --conf-nproc $NPROC"
            + f" 2>&1 | tee {out / tk['tag']}/pipeline.log\n")
        if args.db:
            # post-hoc scoring of this task only, after its search has ended (early results);
            # the final scoring job redoes every task and adds the Sage-relaxed references
            body += (f"python EM/rematch.py {tk['tag']} --dir {out} --nproc $NPROC --cutoff 0.7 "
                     f"--ref {args.code} --ref-db {Path(args.db).resolve()} || true\n")
        (tdir / f"task_{k}.sh").write_text(body)
    job = out / "run_array.sh"
    job.write_text(
        "#!/bin/bash\n"
        f"#SBATCH --partition={args.partition}\n#SBATCH --nodes=1\n#SBATCH --ntasks=1\n"
        f"#SBATCH --cpus-per-task={args.cpus}\n#SBATCH --mem={args.mem}\n"
            + (f"#SBATCH --constraint={args.constraint}\n" if args.constraint else "")
            + f"#SBATCH --time={args.time}\n#SBATCH --job-name=bl-{args.code}\n#SBATCH --nice={args.nice}\n"
        f"#SBATCH --array=0-{len(tasks) - 1}%{args.throttle}\n"
        f"#SBATCH --output={out}/logs/%A_%a.out\n"
        "set -o pipefail\n"
        f"cd {ROOT}\n"
        'source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate cellnet\n'
        'export PATH="$HOME/CHARMM/bin:$PATH" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1\n'
        f"NPROC=${{SLURM_CPUS_PER_TASK:-{args.cpus}}}\n"
        f"source {tdir}/task_${{SLURM_ARRAY_TASK_ID}}.sh\n")
    print(f"  {len(tasks)} array tasks (at most {args.chunk_boxes} boxes, {args.special_chunk_boxes} for "
          f"special sites), {args.throttle} at a time, {args.cpus} CPUs / {args.mem} each, "
          f"partition {args.partition}")
    plan_out = {"code": args.code, "smiles": smiles, "groups": groups, "tasks": tasks,
                "total_probability": cum, "top10": [[h, z, round(p, 4)] for h, z, p in probs[:10]]}
    if args.submit:
        plan_out["array_job"] = subprocess.run(["sbatch", "--parsable", str(job)], capture_output=True,
                                               text=True, check=True).stdout.strip()
        print(f"  submitted array {plan_out['array_job']} (bl-{args.code})")
        if args.db:
            # Post-hoc scoring against the CSD entry and the Sage-relaxed experiment, after every task ended.
            plan_out["score_job"] = subprocess.run(
                ["sbatch", "--parsable", f"--dependency=afterany:{plan_out['array_job']}",
                 f"--job-name=score-{args.code}", f"--nice={args.nice}", f"--partition={args.partition}",
                 *([f"--constraint={args.constraint}"] if args.constraint else []),
                 "--time=1-00:00:00", f"--output={out}/score_%j.out",
                 str(ROOT / "EM/score_blind.sh"), str(out), args.code, str(Path(args.db).resolve()),
                 *[tk["tag"] for tk in tasks]],
                capture_output=True, text=True, check=True, cwd=ROOT).stdout.strip()
            print(f"  scoring job {plan_out['score_job']} (after the array): {out}/sg_*/rematch_{args.code}_0.7/ "
                  f"vs experiment, rematch_sagefree_0.7/ and rematch_sagefrozen_0.7/ vs the Sage-relaxed experiment")
    (out / plan_name).write_text(json.dumps(plan_out, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))

if __name__ == "__main__":
    main()
