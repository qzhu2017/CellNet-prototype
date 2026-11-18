# CellNet

CellNet predicts unit cells for molecular crystal structure prediction (CSP). Given a
molecule (SMILES), a Hall setting and Z′, a conditional flow model samples basis-invariant
lattice descriptors; quasi-random search (QRS) turns each sample into a Hall-compatible
cell, and a symmetry-aware packing search in [PyXtal](https://github.com/MaterSim/PyXtal)
places and relaxes the molecules in the proposed cells.

This repository contains the code, the trained model and the training tables for

> Q. Zhu and Y. Weng, *Molecular Crystal Structure Prediction from Conditional Flow on the
> Unit Cells* (manuscript, 2026).

On the 84 single-component systems with Z′ ≤ 1 of the Zhu & Hattori benchmark, the
reported protocol (64 cells per system) reproduces the experimental structure for
**83/84 (98.8 %)**: 80 with the lattice free to relax and 3 more through a frozen-lattice
fallback; one of those three (XAFPAY) rests on a single marginal match. Per-code results are
in [`benchmark/results.csv`](benchmark/results.csv).

Since then, CellNet also runs [blind](#blind-prediction): a flow trained on Z′ = 1 and Z′ = 2
crystals, a model that predicts the space group and the density from the molecule, molecules on
special positions (inversion centres, 2-fold axes, mirror planes) and multi-component crystals.

## How it works

1. **Conditional flow** (`cellnet/models.py`). A GNN encodes the molecular graph; learned
   Hall and Z′ embeddings complete the condition. The flow samples the 12-dimensional vector
   [6 Delaunay–Selling scalars, log λ, log λ\*] (λ, λ\* = direct and reciprocal successive
   minima, sampled after the Selling scalars), and a separate head predicts the density.
2. **Lattice QRS** (`cellnet/qrs.py`). Each draw is inverted to cell parameters of the given
   Hall setting by a 12-stage Sobol search that matches λ, λ\*, the Selling scalars and the
   density.
3. **Cell selection** (`cellnet/lattice_conf_pipeline.py`). Duplicates are removed and 64 cells
   are chosen without reference to any experimental structure: fixed quotas by λ
   consistency, QRS loss, density, Selling disagreement and (monoclinic only) alternative
   unique-axis assignments, then shape diversity.
4. **Packing search**. For each cell, PyXtal's QRS samples molecular positions and
   orientations, and CHARMM relaxes every candidate with an OpenFF (Sage) force field built by
   [pyocse](https://github.com/MaterSim/pyocse).

## Installation

```bash
conda create -n cellnet python=3.11
conda activate cellnet
conda install -c conda-forge --override-channels \
  openff-toolkit=0.14.3 openff-interchange=0.3.18 openff-units=0.2.0 openff-models=0.0.4 \
  openmm=8.0.0 parmed=4.0.0 "pydantic<2" ambertools openbabel
pip install -r requirements.txt
```

Steps 1–3 need only the Python packages. The packing search (step 4) needs the conda-forge
packages: pyocse builds the OpenFF force field through Interchange internals, so keep those
versions pinned, and PyXtal uses openbabel to rebuild molecules after each CHARMM
relaxation. It also needs a `charmm` executable on `PATH`. The paper used OpenFF
**Sage 2.0.0**, the default of pyocse 0.1.3; the pipeline selects it explicitly with
`--sage 2.0`.

## Repository layout

| Path | Contents |
|------|----------|
| `cellnet/` | Python package: invariants, flow model, lattice QRS, cell selection, packing pipeline |
| `scripts/` | Command-line entry points (below) |
| `checkpoints/cellnet_flow/` | Trained flow (`best.pt`), normalization and Hall vocabulary (`stats.json`), training history |
| `checkpoints/cellnet_flow_z12/` | Flow trained on Z′ = 1 + 2 (used by the blind scripts) |
| `checkpoints/aux_sg_density/` | Space-group and density model (used by the blind scripts) |
| `datasets/spade-csp/` | Training and test tables derived from SPaDe-CSP ([details](datasets/README.md)) |
| `datasets/csd-z2/` | Z′ = 2 training and test tables from the CSD ([details](datasets/README.md)) |
| `EM/` | Post-hoc scoring used by the blind jobs: `rematch.py`, `relax_ref.py`, `score_blind.sh`, `cocrystal_status.py` |
| `training/` | Slurm recipes that trained the two new models |
| `runtime/` | Start-up hook that applies `cellnet/pyxtal_patches.py` in PyXtal's worker processes |
| `benchmark/` | Benchmark code list, run and summary scripts, per-code results ([details](benchmark/README.md)) |
| `tests/` | Unit tests |

| Script | Purpose |
|--------|---------|
| `scripts/run_pipeline.py` | Full pipeline (flow → lattice QRS → selection → packing) for one molecule or benchmark code |
| `scripts/run_blind.py` | Blind CSP from a SMILES: predicted space groups, Z′, densities and special sites, as one Slurm array |
| `scripts/run_cocrystal.py` | The same for multi-component crystals (formula-unit SMILES) |
| `scripts/train_aux.py` | Train and evaluate the space-group and density model |
| `scripts/build_joint_dataset.py` | Leak-free joint Z′ = 1 + 2 tables |
| `scripts/collect_csd_zprime.py` | Collect Z′ = 2 crystals from the CSD (needs the CSD Python API) |
| `scripts/validate_flow.py` | K-sample validation of the flow on the test split |
| `scripts/train_flow.py` | Train the conditional flow |
| `scripts/build_dataset.py` | Rebuild the training tables from the SPaDe-CSP release |
| `scripts/precompute_lattice_cache.py` | Precompute lattice targets and molecular graphs for training |
| `scripts/shard_graph_sidecar.py` | Split the graph cache into shards for RAM-bounded training |

## Quick start

Propose cells for a molecule (steps 1–3; minutes on a workstation):

```bash
python scripts/run_pipeline.py \
  --smiles "CC(=O)Oc1ccccc1C(=O)O" --hall 81 --zprime 1 \
  --k 96 --skip-conf-qrs --tag aspirin
```

The flow draws (`*_flow_k96.npz`), the lattice-QRS cells (`*_lattice_qrs.csv`) and the
deduplicated cells (`*_unique_cells.csv`) are written under `outputs/pipeline/aspirin/`
(`--output-dir` changes the parent folder).

Run the packing search as well, with the paper's cell budget and search settings:

```bash
python scripts/run_pipeline.py \
  --smiles "CC(=O)Oc1ccccc1C(=O)O" --hall 81 --zprime 1 --sage 2.0 \
  --k 96 --conf-selection multichannel --max-conf-runs 64 --cell-blind \
  --conf-qrs-origin-only --conf-npop 96 --no-conf-check-stable \
  --relax-lattice --conf-nproc 48 --tag aspirin
```

Relaxed structures and their force-field energies are written to each cell's `conf_qrs_*` folder.
Without a reference structure there is nothing to trigger the frozen-lattice fallback
(`--adaptive-relax`); for a prospective search, run the same command once more without
`--relax-lattice` to cover both passes.

A benchmark code can be given instead of a molecule (`--csd-code OBEQUJ`); its SMILES,
Hall setting and Z′ are then read from `test.db`, and each relaxed structure is matched
against the experimental one. [`benchmark/run_code.sh`](benchmark/run_code.sh) holds the
complete benchmark protocol.

## Reproducing the benchmark

```bash
bash benchmark/run_code.sh OBEQUJ                     # one code
sbatch --array=0-83 benchmark/submit_slurm.sh         # all 84 codes on a Slurm cluster
python benchmark/summarize.py --root outputs/benchmark --compare benchmark/results.csv
```

The experimental structures come from `test.db`, which ships with PyXtal
(`pyxtal/database/test.db`); the scripts use it automatically. See
[`benchmark/README.md`](benchmark/README.md) for the protocol, cost and caveats.

## Blind prediction

The benchmark protocol is given the experimental space group and Z′. The blind scripts start
from the SMILES alone:

- **Space group and density.** `checkpoints/aux_sg_density/` (`cellnet/aux_model.py`) gives
  p(Hall, Z′ | molecule) and a three-Gaussian density distribution for a given molecule, Hall
  setting and Z′. On the test molecules the right setting is among the top 1 / 3 / 5 / 10 for
  31.4 / 66.7 / 79.4 / 90.8 % (always guessing the most common settings: 21.6 / 54.0 / 76.2 /
  88.8 %). The mean density is off by 1.7 %, and the central 95 % interval holds 94.5 % of the test
  crystals (`checkpoints/aux_sg_density/test_metrics.json`). Each flow draw gets its own density
  from this distribution (`run_pipeline.py --aux-checkpoint`).
- **Z′ = 2.** `checkpoints/cellnet_flow_z12/` was trained from scratch on the Z′ = 1 tables plus
  8,234 Z′ = 2 crystals, with the test-target molecules removed ([details](datasets/README.md)). On
  1,000 Z′ = 2 test molecules (K = 100, `scripts/validate_flow.py`), its best-of-100 error on λ
  (normalised MSE) is 0.036, against 0.337 for the Z′ = 1 model given the same crystals as Z′ = 1;
  on the Z′ = 1 test the two are the same (0.028 and 0.030). For Z′ = 2, `--z2-pairing` sets how the two copies get their
  conformers (the blind runs use `alternate`).
- **Special positions.** Molecules can sit on inversion centres, 2-fold axes or mirror planes
  (`run_pipeline.py --wyckoff`, `--keep-special-sites`). A site symmetry is used only if
  conformers of the molecule can have it; for inversion centres exact centrosymmetric
  conformers are built. These runs use PyXtal's DFS search, since its QRS cannot place molecules
  on special sites.
- **No experimental template.** The packing template is a random placement in the chosen
  setting (`--random-template`), and the search never reads the reference
  (`--strict-posthoc-search`).

```bash
# plan only: the settings taken, their probabilities and boxes
python scripts/run_blind.py MYMOL --smiles "<SMILES>" --plan-only \
  --sg-top-p 0.9 --max-halls 10 --total-boxes 192 --min-boxes 12
# write the task scripts and submit one Slurm array
python scripts/run_blind.py MYMOL --smiles "<SMILES>" --submit --partition <partition> --constraint "" \
  --sg-top-p 0.9 --max-halls 10 --total-boxes 192 --min-boxes 12
```

The most probable Hall settings are taken until they cover 90 % of p(Hall). The boxes are split
between them in proportion to p, with Z′ = 1 and 2 in each, and special-site runs are added
(`--special auto`). Each run is cut into array tasks of at most `--chunk-boxes` boxes. The
`--partition`, `--constraint`, `--cpus` and `--mem` defaults are for our cluster. Results go to
`EM_blind/<CODE>/sg_*/` (`--out`).

To score a blind run against a known structure, give `--db` (a PyXtal database holding the CSD
entry) instead of `--smiles`. Each task then compares its structures with the experiment once its
search has ended. After the whole array, `EM/score_blind.sh` compares them again, both with the
experiment and with the experiment relaxed by the same force field (`EM/relax_ref.py`), which
separates search misses from force-field drift.

Cocrystals and salts use `scripts/run_cocrystal.py --targets targets.json`, a JSON object
`{"NAME": {"smiles": "<A>.<B>.<B>", "ref_cif": "<path>"}, ...}`: the formula unit with its
stoichiometry written out, and a reference CIF that is read only for scoring. Each component goes on general
positions or on special sites of one site symmetry. `python EM/cocrystal_status.py --dir <out>`
summarises the results. `cellnet/pyxtal_patches.py` fixes a PyXtal bug that gave a later
component the previous component's molecule. It is applied on import and, for PyXtal's spawned
workers, through `runtime/sitecustomize.py`, which the generated jobs put on `PYTHONPATH`.

## Retraining

```bash
# 1. Precompute lattice targets and molecular graphs (slow; use several workers)
python scripts/precompute_lattice_cache.py --input datasets/spade-csp/spade_train.csv --workers 8
python scripts/precompute_lattice_cache.py --input datasets/spade-csp/spade_test.csv --workers 8

# 2. Train (defaults match the shipped checkpoint's settings)
python scripts/train_flow.py --precomputed --output-dir outputs/my_flow
```

The shipped checkpoint is the lowest-validation-loss epoch (130) of a 400-epoch run, fine-tuned
from an earlier flow trained on a smaller split, so training from scratch gives a comparable
but not identical model. `checkpoints/cellnet_flow/best.pt` stores the argument list it was
trained with (`torch.load(...)["args"]`); entries such as `shape_bins` or
`shape_retrieval_blend` belong to an experimental model that is not part of this release and
were inactive for the shipped one. To rebuild the tables themselves from the SPaDe-CSP release, see
[`datasets/README.md`](datasets/README.md).

The two blind-mode models:

```bash
python scripts/build_joint_dataset.py   # datasets/joint-z12/{train,test}.csv and build_report.txt
bash training/joint_z12/submit.sh       # precompute the caches (CPU), then train the flow (GPU)
sbatch training/aux/train_aux.sh        # space-group and density model
```

Both write to `outputs/`. The blind scripts use the shipped checkpoints unless given
`--checkpoint` and `--aux-checkpoint`.

## Tests

```bash
python -m pytest tests/ -q
```

## License

The code and the trained model are released under the [MIT License](LICENSE). The training
tables derive from SPaDe-CSP and the CSD; see [`datasets/README.md`](datasets/README.md) for
their provenance.

## Acknowledgments

Supported by the NSF (DMR-2410178); computing resources from ACCESS (TG-MAT230046).
Training data derive from SPaDe-CSP (Taniguchi & Fukasawa, *Digital Discovery* 2025).
