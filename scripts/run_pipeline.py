#!/usr/bin/env python3
"""
Full CSP pipeline: flow K-samples → lattice QRS → dedup → PyXtal conf QRS.

Example (any molecule; steps 1–3 only for a quick test):
  python scripts/run_pipeline.py \\
    --smiles "CC(=O)Oc1ccccc1C(=O)O" --hall 81 --zprime 1 \\
    --k 96 --skip-conf-qrs

Example (benchmark code, looked up in datasets/test.db or the test.db bundled with PyXtal):
  python scripts/run_pipeline.py --csd-code OBEQIX --sage 2.0
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

# Before cellnet/torch/e3nn imports (also set in lattice_conf_pipeline for spawn workers)
warnings.filterwarnings(
    "ignore",
    message=r"You are using `torch.load` with `weights_only=False`",
    category=FutureWarning,
)
warnings.filterwarnings("ignore", category=UserWarning, module=r"torch_geometric\.typing")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPADE_TEST_CSV = ROOT / "datasets/spade-csp/spade_test.csv"
DEFAULT_SPADE_TRAIN_CSV = ROOT / "datasets/spade-csp/spade_train.csv"
DEFAULT_SPADE_TRAIN_PRECOMPUTED_CSV = ROOT / "datasets/spade-csp/spade_train_precomputed.csv"
DEFAULT_SPADE_TEST_PRECOMPUTED_CSV = ROOT / "datasets/spade-csp/spade_test_precomputed.csv"
sys.path.insert(0, str(ROOT))

from cellnet.inference import describe_flow_checkpoint
from cellnet.lattice_conf_pipeline import (
    ConfQRSRecord,
    PipelineResult,
    _load_pyxtal_from_db,
    conf_restart_schedule,
    conf_sweep_all_errored,
    conf_sweep_had_hit,
    deduplicate_cellpars,
    flow_npz_path,
    load_crystal_from_db,
    load_flow_batch,
    print_top_lattice_qrs_cellpar_matches,
    reference_match_score,
    run_conf_qrs_one,
    lattice_records_from_selling,
    qrs_origin_unique_cells,
    append_disagreeing_selling_cells,
    unique_cell_is_qrs_origin,
    unique_cell_is_sell_only,
    posterior_effective_mode_count,
    dynamic_conf_run_count,
    scaled_conf_elite,
    select_conf_unique_cells,
    run_lattice_qrs_batch,
    sample_flow_batch,
    save_flow_batch,
    save_pipeline_artifacts,
    summarize_adaptive_sweeps,
    select_diverse_unique_cells,
    lattice_records_from_unique_axis_alternates,
    select_multichannel_unique_cells,
    unique_cell_is_axis_alternate,
    select_target_free_quota_unique_cells,
)
from cellnet.hybrid_qrs import target_volume_from_density
from cellnet.qrs import QRSConfig
from cellnet.spade import hall_number_from_spg



def _default_test_db(root: Path) -> Path:
    """``datasets/test.db`` if present, else the copy PyXtal ships in ``pyxtal/database``."""
    local = root / "datasets" / "test.db"
    if local.is_file():
        return local
    try:
        import pyxtal
    except ImportError:
        return local
    return Path(pyxtal.__file__).resolve().parent / "database" / "test.db"


DEFAULT_TEST_DB = _default_test_db(ROOT)

def _lookup_spade_csv_row(csv_path: Path, csd_code: str) -> dict[str, str] | None:
    """Scan a SPaDe CSV for ``refcode == csd_code`` without loading the full table."""
    import csv as csv_mod

    if not csv_path.is_file():
        return None
    code = str(csd_code).strip().upper()
    with csv_path.open(newline="") as handle:
        reader = csv_mod.DictReader(handle)
        if not reader.fieldnames or "refcode" not in reader.fieldnames:
            return None
        for row in reader:
            if str(row.get("refcode", "")).strip().upper() == code:
                return row
    return None


def _sample_from_spade_row(row: dict[str, str], csv_path: Path):
    cellpar = np.array(
        [
            float(row["a"]),
            float(row["b"]),
            float(row["c"]),
            float(row["alpha"]),
            float(row["beta"]),
            float(row["gamma"]),
        ],
        dtype=np.float64,
    )
    sg_number = int(row["sg_number"])
    hall_number = int(row.get("hall_number") or hall_number_from_spg(sg_number))
    print(f"Loaded {row['refcode']} from {csv_path}")
    return (
        str(row["smiles"]),
        hall_number,
        float(row["zprime"]),
        cellpar,
        None,
        False,
    )


def _try_load_from_db(db_path: Path, csd_code: str, *, label: str):
    try:
        db_info = load_crystal_from_db(db_path, csd_code)
    except ValueError:
        return None
    print(f"Loaded {csd_code} from {db_info.db_path} (SG #{db_info.space_group_number}){label}")
    return db_info


def _resolve_crystal_metadata(
    csd_code: str,
    *,
    csv_path: str,
    db_path: Path | None,
    smiles_override: str | None,
    hall_override: int | None,
    zprime_override: float | None,
) -> tuple[str, int, float, np.ndarray | None, Path | None, bool]:
    """
    Resolve SMILES, Hall, Z′, and true cellpar.

    Lookup order:
      1. explicit ``--db``
      2. ``datasets/test.db``
      3. ``--csv`` (default: spade_test.csv)
      4. spade_train.csv / precomputed train+test CSVs

    Returns (smiles, hall, zprime, true_cellpar, effective_db_path, from_test_db).
    """
    def _from_db(db_info, *, from_test_db: bool, path: Path):
        return (
            smiles_override or db_info.smiles,
            hall_override if hall_override is not None else db_info.hall_number,
            zprime_override if zprime_override is not None else db_info.zprime,
            db_info.cellpar,
            path,
            from_test_db,
        )

    def _from_row(row: dict[str, str], path: Path):
        smiles, hall, zprime, cellpar, _, _ = _sample_from_spade_row(row, path)
        return (
            smiles_override or smiles,
            hall_override if hall_override is not None else hall,
            zprime_override if zprime_override is not None else zprime,
            cellpar,
            None,
            False,
        )

    if db_path is not None:
        db_info = load_crystal_from_db(db_path, csd_code)
        print(f"Loaded {csd_code} from {db_info.db_path} (SG #{db_info.space_group_number})")
        return _from_db(
            db_info,
            from_test_db=db_path.resolve() == DEFAULT_TEST_DB.resolve(),
            path=db_path,
        )

    # 1) Prefer test.db when present (benchmark / reference structures).
    if DEFAULT_TEST_DB.is_file():
        db_info = _try_load_from_db(DEFAULT_TEST_DB, csd_code, label="")
        if db_info is not None:
            return _from_db(db_info, from_test_db=True, path=DEFAULT_TEST_DB.resolve())

    # 2) User --csv, then train/test SPaDe tables (including precomputed).
    csv_candidates: list[Path] = []
    primary = Path(csv_path)
    csv_candidates.append(primary)
    for extra in (
        DEFAULT_SPADE_TEST_CSV,
        DEFAULT_SPADE_TRAIN_CSV,
        DEFAULT_SPADE_TEST_PRECOMPUTED_CSV,
        DEFAULT_SPADE_TRAIN_PRECOMPUTED_CSV,
    ):
        if extra.resolve() != primary.resolve():
            csv_candidates.append(extra)

    searched: list[str] = []
    for path in csv_candidates:
        searched.append(str(path))
        row = _lookup_spade_csv_row(path, csd_code)
        if row is not None:
            return _from_row(row, path)

    raise ValueError(
        f"CSD code not found for {csd_code}. Searched test.db"
        + (f" ({DEFAULT_TEST_DB})" if DEFAULT_TEST_DB.is_file() else " (missing)")
        + " and CSVs:\n  - "
        + "\n  - ".join(searched)
    )


def _load_ref_pmg(csd_code: str, db_path: str | Path | None = None, remove_h: bool = True):
    ref = _load_pyxtal_from_db(csd_code, db_path=db_path)
    if ref is None:
        return None
    ref_pmg = ref.to_pymatgen()
    if remove_h:
        ref_pmg.remove_species(["H"])
    return ref_pmg


def validate_cell_blind_args(args: argparse.Namespace) -> None:
    """Reject options that would feed reference-cell information into selection."""
    if getattr(args, "cell_blind", False) and args.conf_selection == "reference":
        raise ValueError("--cell-blind cannot be used with --conf-selection reference")


def main() -> None:
    parser = argparse.ArgumentParser(description="Lattice-flow + lattice QRS + conf QRS pipeline")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(ROOT / "checkpoints/cellnet_flow" / "best.pt"),
        help="Flow checkpoint (default: the shipped V9 model)",
    )
    parser.add_argument(
        "--db",
        type=str,
        default=None,
        help="PyXtal ASE database for metadata lookup (e.g. datasets/test.db)",
    )
    parser.add_argument(
        "--ff-name",
        "--sage",
        dest="ff_name",
        type=str,
        default=None,
        metavar="VERSION|OFFXML",
        help=(
            "OpenFF (Sage) force field for the CHARMM search: a version such as 2.0 or 2.1, "
            "or a full file name such as openff-2.0.0.offxml. Sets PYOCSE_OPENFF for this "
            "process and its QRS workers (pyocse >= 0.1.6 resolves versions); default: "
            "pyocse's default (Sage 2.1 unless PYOCSE_OPENFF is set)."
        ),
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=str(ROOT / "datasets/spade-csp/spade_test.csv"),
        help="Primary SPaDe CSV after test.db (train CSV searched if missing)",
    )
    parser.add_argument("--csd-code", type=str, default="DATSIC")
    parser.add_argument("--smiles", type=str, default=None)
    parser.add_argument("--hall", type=int, default=None)
    parser.add_argument("--zprime", type=float, default=None)
    parser.add_argument("--k", type=int, default=100, help="Flow samples and lattice QRS cells")
    parser.add_argument("--output-dir", type=str, default=str(ROOT / "outputs/pipeline"))
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for flow sampling and QRS")
    parser.add_argument(
        "--flow-npz",
        type=str,
        default=None,
        help="Reuse saved flow draws (skip step 1; --checkpoint not required)",
    )
    parser.add_argument("--use-selling", action="store_true", default=True, help="Include Selling in lattice QRS (default: on)")
    parser.add_argument("--no-selling", action="store_false", dest="use_selling", help="Disable Selling in lattice QRS")
    parser.add_argument("--skip-lattice-qrs", action="store_true", help="Stop after flow sampling")
    parser.add_argument("--skip-conf-qrs", action="store_true", help="Stop after deduplication")
    parser.add_argument("--max-conf-runs", type=int, default=None, help="Limit conf QRS to first N unique cells (ignored when --conf-budget auto)")
    parser.add_argument(
        "--conf-budget",
        choices=["fixed", "auto"],
        default="fixed",
        help=(
            "fixed: use --max-conf-runs (default 24). auto: set the QRS-origin "
            "conf count from the flow posterior spread (tight → fewer, spread → more)"
        ),
    )
    parser.add_argument("--conf-runs-min", type=int, default=12, help="Lower bound for --conf-budget auto")
    parser.add_argument("--conf-runs-max", type=int, default=48, help="Upper bound for --conf-budget auto")
    parser.add_argument(
        "--conf-selection",
        choices=["diverse", "multichannel", "target-free-quota", "qrs", "reference", "first"],
        default="diverse",
        help="How to choose cells when --max-conf-runs truncates the unique set",
    )
    parser.add_argument(
        "--cell-blind",
        action="store_true",
        help=(
            "Suppress true-cell output/oracle ranking and prohibit reference-based "
            "lattice selection; CSD templates and the conf reference matcher remain available"
        ),
    )
    parser.add_argument(
        "--conf-channel-lambda",
        type=int,
        default=6,
        help="Multichannel quota for λ + reciprocal-λ consistency (default: 6)",
    )
    parser.add_argument(
        "--conf-channel-qrs-loss",
        type=int,
        default=6,
        help="Multichannel quota for globally low finite lattice-QRS loss (default: 6)",
    )
    parser.add_argument(
        "--conf-channel-density",
        type=int,
        default=4,
        help="Multichannel quota for density/target-volume consistency (default: 4)",
    )
    parser.add_argument(
        "--conf-channel-disagreement",
        type=int,
        default=4,
        help="Multichannel quota for QRS-versus-Selling disagreement (default: 4)",
    )
    parser.add_argument(
        "--conf-channel-axis-alternates",
        type=int,
        default=12,
        help=(
            "Multichannel quota for monoclinic unique-axis alternates: the most "
            "λ-consistent QRS cells relabeled so b sits on each other edge "
            "(default: 12, v11 protocol; 0 restores the v9/v10 behaviour of b on λ₁ only)"
        ),
    )
    parser.add_argument(
        "--axis-alternate-top",
        type=int,
        default=6,
        help="How many top λ-consistent QRS cells get unique-axis alternates (default: 6)",
    )
    parser.add_argument(
        "--conf-qrs-origin-only",
        action="store_true",
        help="Restrict multichannel selection to QRS-origin cells",
    )
    parser.add_argument(
        "--conf-selling-quota",
        type=int,
        default=24,
        help=(
            "Selling-only quota for --conf-selection target-free-quota "
            "(default: 24, giving 48 QRS + 24 Selling cells for 72 runs)"
        ),
    )
    parser.add_argument(
        "--selling-density-ratio-factor",
        type=float,
        default=None,
        help=(
            "Symmetric predicted-density gate for Selling cells; 1.25 accepts "
            "cell density/predicted density and cell volume/target volume in [0.8, 1.25]. "
            "Unset preserves the v6 gate"
        ),
    )
    parser.add_argument(
        "--conf-elite",
        type=int,
        default=6,
        help=(
            "Diverse-mode elite size (target-free): lowest-λ-MSE QRS cells, "
            "QRS cells whose angles agree with the same draw's Selling vector, "
            "and one Selling reconstruction most consistent with that draw's λ"
        ),
    )
    parser.add_argument(
        "--conf-clusters",
        type=int,
        default=12,
        help=(
            "Selling-space farthest-point clusters whose QRS cells fill diversity slots "
            "(0=off; default 12, restricted to the 32 lowest-λ-MSE draws)"
        ),
    )
    parser.add_argument(
        "--conf-disagree",
        type=int,
        default=12,
        help=(
            "Max QRS–Selling disagreement pairs to add even if the QRS draw was not selected "
            "(0=off; default 12; extras still require angle MAE ≥ 3°)"
        ),
    )
    parser.add_argument("--qrs-stages", type=int, default=12)
    parser.add_argument("--qrs-samples-per-stage", type=int, default=1024)
    parser.add_argument(
        "--lattice-qrs-nproc",
        type=int,
        default=1,
        help="Parallel lattice QRS workers (one flow draw per worker; inner QRS stays serial)",
    )
    parser.add_argument(
        "--max-lattice-qrs-lambda-ratio",
        type=float,
        default=8.0,
        help=(
            "Skip flow draws with λmax/λmin above this value before lattice QRS "
            "(default: 8; use 0 to disable)"
        ),
    )
    parser.add_argument("--conf-ngen", type=int, default=20)
    parser.add_argument("--conf-npop", type=int, default=48)
    parser.add_argument("--conf-nproc", type=int, default=1)
    parser.add_argument(
        "--conf-restarts",
        type=int,
        default=1,
        help="Fixed conformational-QRS restarts per selected cell (default: 1)",
    )
    parser.add_argument(
        "--conf-seed-stride",
        type=int,
        default=100_000,
        help="Seed offset between conformational-QRS restarts (default: 100000)",
    )
    parser.add_argument(
        "--relax-lattice",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Allow final conformational-QRS CHARMM runs to relax atomic "
            "coordinates and the symmetry-constrained lattice "
            "(default: fixed lattice)"
        ),
    )
    parser.add_argument(
        "--stop-after-first-hit",
        action="store_true",
        help=(
            "End a conf-QRS sweep as soon as one cell matches the reference "
            "(coverage benchmarks only; needs reference matching). In the v9 suite "
            "the median first hit was the 2nd of 72 cells and 87%% of conf-QRS time "
            "was spent after the first hit."
        ),
    )
    parser.add_argument(
        "--adaptive-relax",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "If the entire conformational-QRS sweep finds no match, repeat it "
            "once with the opposite --relax-lattice setting. The second sweep "
            "costs nothing on structures that already succeed "
            "(default: disabled, v9 behaviour)"
        ),
    )
    parser.add_argument(
        "--conf-check-stable",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Run post-relaxation stability perturbation sweeps during "
            "conformational QRS (default: enabled)"
        ),
    )
    parser.add_argument("--match-ref", action="store_true", help="Match conf QRS against reference structure")
    parser.add_argument(
        "--no-match-ref",
        action="store_true",
        help="Disable reference matching even when structure is loaded from test.db",
    )
    parser.add_argument(
        "--strict-posthoc-search",
        action="store_true",
        help=(
            "Do not load or pass a reference structure into conformational QRS. "
            "The CSD-derived molecular/Wyckoff template is still used; evaluate artifacts post-hoc"
        ),
    )
    args = parser.parse_args()
    if args.ff_name:
        # Must be set before pyocse is imported anywhere in this process; spawned
        # QRS workers inherit the environment.
        os.environ["PYOCSE_OPENFF"] = args.ff_name
    try:
        validate_cell_blind_args(args)
    except ValueError as exc:
        parser.error(str(exc))
    if args.flow_npz is None and args.checkpoint is None:
        parser.error("Provide --checkpoint or --flow-npz")
    if args.conf_restarts < 1:
        parser.error("--conf-restarts must be >= 1")
    if args.stop_after_first_hit and args.strict_posthoc_search:
        parser.error("--stop-after-first-hit needs live reference matching (incompatible with --strict-posthoc-search)")
    if args.conf_seed_stride < 1:
        parser.error("--conf-seed-stride must be >= 1")
    if args.conf_selling_quota < 0:
        parser.error("--conf-selling-quota must be >= 0")
    if args.selling_density_ratio_factor is not None and not (
        np.isfinite(args.selling_density_ratio_factor)
        and 1.0 <= args.selling_density_ratio_factor <= 4.0
    ):
        parser.error("--selling-density-ratio-factor must be finite and in [1, 4]")
    if args.relax_lattice:
        import pyxtal

        print(f"PyXtal source for lattice relaxation: {Path(pyxtal.__file__).resolve()}")

    db_path: Path | None = Path(args.db).resolve() if args.db else None
    true_cellpar = None
    match_ref = args.match_ref and not args.no_match_ref
    if args.csd_code:
        tag = args.csd_code
        smiles, hall, zprime, true_cellpar, resolved_db, from_test_db = _resolve_crystal_metadata(
            args.csd_code,
            csv_path=args.csv,
            db_path=db_path,
            smiles_override=args.smiles,
            hall_override=args.hall,
            zprime_override=args.zprime,
        )
        if resolved_db is not None:
            db_path = resolved_db
        if from_test_db and not args.no_match_ref and not args.match_ref:
            match_ref = True
            print("  Auto-enabled reference matching (structure from datasets/test.db)")
    else:
        if not args.smiles or args.hall is None or args.zprime is None:
            parser.error("Provide --csd-code or (--smiles --hall --zprime)")
        smiles = args.smiles
        hall = args.hall
        zprime = args.zprime
        tag = "custom"

    out_dir = Path(args.output_dir) / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Pipeline: {tag} ===")
    if args.checkpoint and not args.flow_npz:
        ckpt_info = describe_flow_checkpoint(args.checkpoint)
        epoch = ckpt_info["epoch"]
        epoch_str = f"  epoch={epoch}" if epoch is not None else ""
        print(f"Checkpoint: {ckpt_info['path']}")
        print(f"Model:      {ckpt_info['model_name']}{epoch_str}")
    elif args.flow_npz:
        print(f"Flow NPZ:   {Path(args.flow_npz).resolve()}  (checkpoint skipped)")
    print(f"SMILES: {smiles}")
    print(f"Hall:   {hall}   Z′: {zprime}")
    try:
        import pyocse.forcefield as _pf

        print(f"OpenFF: {_pf.DEFAULT_OPENFF}" + ("  (--ff-name)" if args.ff_name else "  (pyocse default)"))
    except Exception:
        print(f"OpenFF: {os.environ.get('PYOCSE_OPENFF', 'pyocse default')}")
    if true_cellpar is not None and not args.cell_blind:
        cp = true_cellpar
        print(
            f"True L: a={cp[0]:.3f} b={cp[1]:.3f} c={cp[2]:.3f} "
            f"α={cp[3]:.1f} β={cp[4]:.1f} γ={cp[5]:.1f}"
        )
    print()

    print(f"=== Step 1: Flow K={args.k} samples ===")
    flow_npz = Path(args.flow_npz).resolve() if args.flow_npz else flow_npz_path(out_dir, tag, args.k)
    if args.flow_npz:
        flow = load_flow_batch(flow_npz)
        if flow.k != args.k:
            raise ValueError(f"--k={args.k} but {flow_npz} contains K={flow.k}")
        print(f"  loaded flow draws from {flow_npz}")
    else:
        print(f"  sampling with seed={args.seed}")
        flow = sample_flow_batch(
            args.checkpoint,
            smiles,
            hall,
            zprime,
            k=args.k,
            seed=args.seed,
        )
        save_flow_batch(flow, flow_npz, seed=args.seed)
        print(f"  saved flow draws → {flow_npz}")
    print(f"  predicted ρ = {flow.target_rho:.4f} g/cm³")
    if flow.shape_probabilities is not None:
        if flow.neural_shape_probabilities is not None:
            neural = ", ".join(
                f"q{i}={prob:.3f}"
                for i, prob in enumerate(flow.neural_shape_probabilities)
            )
            print(f"  neural shape probabilities: {neural}")
        if flow.retrieval_shape_probabilities is not None:
            retrieval = ", ".join(
                f"q{i}={prob:.3f}"
                for i, prob in enumerate(flow.retrieval_shape_probabilities)
            )
            print(
                f"  retrieval shape prior (confidence={flow.retrieval_confidence:.3f}): "
                f"{retrieval}"
            )
        probs = ", ".join(
            f"q{i}={prob:.3f}" for i, prob in enumerate(flow.shape_probabilities)
        )
        counts = np.bincount(
            flow.sampled_shape_bins,
            minlength=len(flow.shape_probabilities),
        )
        print(f"  blended shape probabilities: {probs}")
        print(f"  sampled shape bins: {counts.tolist()}")
    lam = np.exp(flow.all_log_lambda)
    print(
        f"  λ range (Å): [{lam.min():.2f}, {lam.max():.2f}]  "
        f"mean draw #0: {lam[0].tolist()}"
    )

    result = PipelineResult(
        tag=tag,
        smiles=smiles,
        hall_number=hall,
        zprime=zprime,
        true_cellpar=None if args.cell_blind else true_cellpar,
        flow=flow,
    )

    if args.skip_lattice_qrs:
        save_pipeline_artifacts(result, out_dir, flow_seed=args.seed)
        print(f"\nSaved step-1 artifacts under {out_dir}")
        return

    print(
        f"\n=== Step 2: Lattice QRS × {args.k} "
        f"(λ + λ* + ρ, w_S={2 if args.use_selling else 0}, "
        f"nproc={args.lattice_qrs_nproc}, "
        f"max-ratio={args.max_lattice_qrs_lambda_ratio:g}) ==="
    )
    qrs_cfg = QRSConfig(
        n_stages=args.qrs_stages,
        samples_per_stage=args.qrs_samples_per_stage,
        w_selling=2.0 if args.use_selling else 0.0,
        w_lambda=2.0,
        w_lambda_recip=2.0,
        w_density=2.0,
        seed=args.seed,
        n_workers=1,
    )
    result.lattice_qrs = run_lattice_qrs_batch(
        flow,
        qrs_config=qrs_cfg,
        use_selling=args.use_selling,
        qrs_seed=args.seed,
        n_proc=args.lattice_qrs_nproc,
        max_lambda_ratio=args.max_lattice_qrs_lambda_ratio,
        reference_cellpar=None,
        verbose=True,
    )
    selling_recs = lattice_records_from_selling(
        flow,
        density_ratio_factor=args.selling_density_ratio_factor,
    )
    if selling_recs:
        result.lattice_qrs.extend(selling_recs)
        print(
            f"  stored {len(selling_recs)} Selling reconstructions "
            "(added to conf only if they disagree with a selected QRS draw)"
        )
    if args.conf_channel_axis_alternates > 0 and args.conf_selection in (
        "multichannel",
        "target-free-quota",
    ):
        axis_alt_recs = lattice_records_from_unique_axis_alternates(
            flow,
            result.lattice_qrs,
            n_top=args.axis_alternate_top,
        )
        if axis_alt_recs:
            result.lattice_qrs.extend(axis_alt_recs)
            print(
                f"  stored {len(axis_alt_recs)} unique-axis alternates of the "
                f"{args.axis_alternate_top} most λ-consistent QRS cells "
                "(monoclinic b moved onto each other edge; the (λ, λ*, ρ) objective "
                "cannot rank the three settings)"
            )

    if true_cellpar is not None and not args.cell_blind:
        result.top_lattice_matches = print_top_lattice_qrs_cellpar_matches(
            result.lattice_qrs,
            true_cellpar,
            hall,
            top_n=5,
        )

    print(f"\n=== Step 3: Deduplicate lattice cells ===")
    result.unique_cells = deduplicate_cellpars(result.lattice_qrs, hall)
    n_alt_u = sum(1 for u in result.unique_cells if unique_cell_is_axis_alternate(u))
    n_qrs_u = sum(1 for u in result.unique_cells if unique_cell_is_qrs_origin(u)) - n_alt_u
    n_sell_u = sum(1 for u in result.unique_cells if unique_cell_is_sell_only(u))
    print(
        f"  {len(result.lattice_qrs)} lattice cells → {len(result.unique_cells)} unique "
        f"({n_qrs_u} QRS-origin, {n_sell_u} Selling-only"
        + (f", {n_alt_u} unique-axis alternates" if n_alt_u else "")
        + ")"
    )
    save_pipeline_artifacts(result, out_dir, flow_seed=args.seed)
    print(f"  Saved step 2–3 artifacts under {out_dir}", flush=True)

    if args.skip_conf_qrs:
        print(f"\nSaved through step 3 under {out_dir}")
        return

    ref_pmg = (
        _load_ref_pmg(tag, db_path=db_path)
        if match_ref and not args.strict_posthoc_search
        else None
    )
    if match_ref and not args.strict_posthoc_search and ref_pmg is None:
        ref_src = db_path if db_path is not None else "known PyXtal databases"
        print(f"  Warning: --match-ref requested but no reference found in {ref_src}")

    targets = result.unique_cells
    selection_provenance: dict[int, str] = {}
    n_conf = args.max_conf_runs
    if n_conf is None:
        # Every selection mode gets a cap; without one the sweep ran conf QRS on
        # every unique cell (often >100) under the default "diverse" mode, while
        # --conf-budget's help text promised 24.
        n_conf = 72 if args.conf_selection == "target-free-quota" else 24
    n_elite = args.conf_elite
    if args.conf_budget == "auto":
        n_eff = posterior_effective_mode_count(result.unique_cells, result.lattice_qrs)
        n_conf = dynamic_conf_run_count(
            n_eff,
            n_min=args.conf_runs_min,
            n_max=args.conf_runs_max,
        )
        n_elite = scaled_conf_elite(args.conf_elite, n_conf)
        print(
            f"  conf budget auto: n_eff={n_eff:.1f} → {n_conf} QRS-origin cells "
            f"(min={args.conf_runs_min}, max={args.conf_runs_max}); "
            f"elite {args.conf_elite} → {n_elite}",
            flush=True,
        )
    if n_conf is not None:
        if args.conf_selection in ("diverse", "multichannel", "target-free-quota"):
            qrs_recs = [
                rec
                for rec in result.lattice_qrs
                if getattr(rec, "source", "qrs") != "selling"
            ]
            selling_only = [
                rec
                for rec in result.lattice_qrs
                if getattr(rec, "source", "qrs") == "selling"
            ]
            target_volume = target_volume_from_density(
                flow.target_rho, smiles, zprime, hall
            )
            if args.conf_selection == "multichannel":
                targets, selection_provenance = select_multichannel_unique_cells(
                    result.unique_cells,
                    qrs_recs,
                    selling_only,
                    n_conf,
                    hall,
                    target_rho=flow.target_rho,
                    target_volume=target_volume,
                    lambda_quota=args.conf_channel_lambda,
                    qrs_loss_quota=args.conf_channel_qrs_loss,
                    density_quota=args.conf_channel_density,
                    disagreement_quota=args.conf_channel_disagreement,
                    qrs_origin_only=args.conf_qrs_origin_only,
                    density_ratio_factor=args.selling_density_ratio_factor,
                    axis_alternate_quota=args.conf_channel_axis_alternates,
                )
            elif args.conf_selection == "target-free-quota":
                targets, selection_provenance = select_target_free_quota_unique_cells(
                    result.unique_cells,
                    qrs_recs,
                    selling_only,
                    n_conf,
                    hall,
                    selling_quota=args.conf_selling_quota,
                    target_rho=flow.target_rho,
                    target_volume=target_volume,
                    density_ratio_factor=args.selling_density_ratio_factor,
                    lambda_quota=args.conf_channel_lambda,
                    qrs_loss_quota=args.conf_channel_qrs_loss,
                    density_quota=args.conf_channel_density,
                    disagreement_quota=args.conf_channel_disagreement,
                    axis_alternate_quota=args.conf_channel_axis_alternates,
                )
            else:
                targets = select_conf_unique_cells(
                    result.unique_cells,
                    qrs_recs,
                    selling_only,
                    n_conf,
                    hall,
                    n_elite=n_elite,
                    all_selling=flow.all_selling,
                    n_clusters=args.conf_clusters,
                    n_disagree=args.conf_disagree,
                    target_volume=target_volume,
                    density_ratio_factor=args.selling_density_ratio_factor,
                )
        elif args.conf_selection == "reference":
            if true_cellpar is None:
                raise SystemExit("--conf-selection reference requires a known reference cellpar")
            targets = sorted(
                result.unique_cells,
                key=lambda cell: reference_match_score(cell.cellpar, true_cellpar, hall)[2],
            )[:n_conf]
        elif args.conf_selection == "qrs":
            qrs_by_flow = {record.flow_idx: record.qrs_loss for record in result.lattice_qrs}
            targets = sorted(
                result.unique_cells,
                key=lambda cell: min(
                    (qrs_by_flow.get(idx, float("inf")) for idx in cell.source_flow_indices),
                    default=float("inf"),
                ),
            )[:n_conf]
        else:
            targets = targets[:n_conf]
        sel_note = {
            "diverse": (
                "QRS-origin λ-MSE + Selling-agree elite (one per draw), "
                "Selling k-center diversity, then disagreement extras "
                "(Selling of unselected draws if QRS/Selling angles disagree)"
            ),
            "multichannel": (
                "fixed target-free λ-consistency, finite-QRS-loss, density-volume, "
                "QRS/Selling-disagreement"
                + (
                    f", and unique-axis-alternate ({args.conf_channel_axis_alternates}) "
                    if args.conf_channel_axis_alternates > 0
                    else " "
                )
                + "quotas; shape k-centers fill the remainder"
            ),
            "target-free-quota": (
                "separate QRS-origin multichannel and Selling-only disagreement, "
                "shape-diversity, parent-λ, and deterministic-fallback quotas"
            ),
            "reference": "ORACLE ranked vs known cell (not for fair benchmarks)",
            "qrs": "lowest QRS loss",
            "first": "first N unique",
        }.get(args.conf_selection, args.conf_selection)
        print(
            f"  conf selection: {args.conf_selection} "
            f"({len(targets)} cells, elite={n_elite}, clusters={args.conf_clusters}, "
            f"disagree={args.conf_disagree}, {sel_note})"
        )

    relax_mode = "coordinates + lattice" if args.relax_lattice else "coordinates only"
    stability_mode = "stability checks on" if args.conf_check_stable else "stability checks off"
    print(
        f"\n=== Step 4: Conformational QRS (QRS_conf) on {len(targets)} "
        f"unique cells × {args.conf_restarts} fixed restart(s) "
        f"({relax_mode}, {stability_mode}) ==="
    )
    component_pool_cache: dict = {}
    forcefield_info_cache: dict = {}

    def _run_conf_sweep(relax_lattice: bool, schedule_root: Path) -> list[ConfQRSRecord]:
        """One full conformational-QRS pass over every selected cell."""
        produced: list[ConfQRSRecord] = []
        for u in targets:
            cp = u.cellpar
            channel = selection_provenance.get(u.dedup_idx, args.conf_selection)
            schedule = conf_restart_schedule(
                schedule_root,
                u.dedup_idx,
                args.conf_restarts,
                args.seed,
                args.conf_seed_stride,
            )
            for restart, conf_seed, workdir in schedule:
                print(
                    f"  conf QRS #{u.dedup_idx} restart {restart}/{args.conf_restarts}: "
                    f"a={cp[0]:.3f} b={cp[1]:.3f} c={cp[2]:.3f} "
                    f"(channel={channel}, seed={conf_seed}, "
                    f"from {len(u.source_flow_indices)} flow draws)",
                    flush=True,
                )
                try:
                    rec = run_conf_qrs_one(
                        u.dedup_idx,
                        smiles,
                        hall,
                        zprime,
                        cp,
                        workdir,
                        csd_code=tag if args.csd_code else None,
                        db_path=db_path,
                        ref_pmg=ref_pmg,
                        ngen=args.conf_ngen,
                        npop=args.conf_npop,
                        nproc=args.conf_nproc,
                        seed=conf_seed,
                        restart=restart,
                        selection_channel=channel,
                        component_pool_cache=component_pool_cache,
                        forcefield_info_cache=forcefield_info_cache,
                        relax_lattice=relax_lattice,
                        check_stable=args.conf_check_stable,
                        enable_reference_matching=not args.strict_posthoc_search,
                    )
                    print(
                        f"    done: success_rate={rec.success_rate}  "
                        f"time={rec.time_min:.1f} min  conformers={rec.n_conformers}",
                        flush=True,
                    )
                except Exception as exc:
                    rec = ConfQRSRecord(
                        dedup_idx=u.dedup_idx,
                        cellpar=cp.tolist(),
                        success_rate=None,
                        time_min=0.0,
                        workdir=str(workdir),
                        n_conformers=0,
                        error=str(exc),
                        restart=restart,
                        seed=conf_seed,
                        selection_channel=channel,
                        reference_matching_enabled=not args.strict_posthoc_search,
                        relax_lattice=relax_lattice,
                    )
                    print(f"    FAILED: {exc}", flush=True)
                produced.append(rec)
                result.conf_qrs.append(rec)
                save_pipeline_artifacts(result, out_dir, flow_seed=args.seed)
                if args.stop_after_first_hit and conf_sweep_had_hit([rec]):
                    n_left = len(targets) - targets.index(u) - 1
                    print(
                        f"  first hit at cell #{u.dedup_idx} (sweep position "
                        f"{targets.index(u) + 1}/{len(targets)}); stopping the sweep, "
                        f"{n_left} cell(s) skipped (--stop-after-first-hit)",
                        flush=True,
                    )
                    return produced
        return produced

    primary_records = _run_conf_sweep(args.relax_lattice, out_dir)

    if args.adaptive_relax:
        fallback_relax = not args.relax_lattice
        fallback_mode = "coordinates + lattice" if fallback_relax else "coordinates only"
        fallback_root = out_dir / (
            "adaptive_relaxed_lattice" if fallback_relax else "adaptive_fixed_lattice"
        )
        fallback_records: list[ConfQRSRecord] | None = None
        skipped_reason: str | None = None

        if conf_sweep_had_hit(primary_records):
            skipped_reason = "primary sweep matched"
        elif not primary_records:
            skipped_reason = "no primary runs"
        elif conf_sweep_all_errored(primary_records):
            # Every run raised. Freezing the lattice cannot fix a broken
            # force field or CHARMM install; do not burn a second sweep on it.
            skipped_reason = "every primary run errored"
            print(
                f"\n=== Step 4b: adaptive fallback skipped -- all {len(primary_records)} "
                f"primary run(s) errored; a second sweep would fail the same way ===",
                flush=True,
            )
        else:
            print(
                f"\n=== Step 4b: adaptive fallback -- {len(primary_records)} run(s) over "
                f"{len(targets)} cell(s) produced no match; repeating the sweep with "
                f"{fallback_mode} ===",
                flush=True,
            )
            fallback_records = _run_conf_sweep(fallback_relax, fallback_root)
            if conf_sweep_had_hit(fallback_records):
                print(f"  adaptive fallback recovered a match ({fallback_mode})", flush=True)
            else:
                print("  adaptive fallback found no match either", flush=True)

        # Separate accounting for the two passes, so consumers of the summary
        # never have to infer which records belong to which protocol.
        result.adaptive_relax = summarize_adaptive_sweeps(
            enabled=True,
            primary_relax_lattice=args.relax_lattice,
            primary=primary_records,
            fallback=fallback_records,
            fallback_root=str(fallback_root) if fallback_records is not None else None,
            skipped_reason=skipped_reason,
        )

    save_pipeline_artifacts(result, out_dir, flow_seed=args.seed)
    print(f"\nPipeline complete. Artifacts: {out_dir}")


if __name__ == "__main__":
    main()
