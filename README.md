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
reported protocol reproduces the experimental structure for **82/84 (97.6 %)**: 80 with the
lattice free to relax and 2 more through a frozen-lattice fallback
(per-code results in [`benchmark/results.csv`](benchmark/results.csv)).

## How it works

1. **Conditional flow** (`cellnet/models.py`). A GNN encodes the molecular graph; learned
   Hall and Z′ embeddings complete the condition. The flow samples the 12-dimensional vector
   [6 Delaunay–Selling scalars, log λ, log λ\*] (λ, λ\* = direct and reciprocal successive
   minima, sampled after the Selling scalars), and a separate head predicts the density.
2. **Lattice QRS** (`cellnet/qrs.py`). Each draw is inverted to cell parameters of the given
   Hall setting by a 12-stage Sobol search that matches λ, λ\*, the Selling scalars and the
   density.
3. **Cell selection** (`cellnet/lattice_conf_pipeline.py`). Duplicates are removed and 72 cells
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
pip install -r requirements.txt
```

Steps 1–3 need only the Python packages. The packing search (step 4) also needs a `charmm`
executable on `PATH`. The paper used OpenFF **Sage 2.0.0**, the default of pyocse 0.1.3;
the pipeline selects it explicitly with `--sage 2.0`.

## Repository layout

| Path | Contents |
|------|----------|
| `cellnet/` | Python package: invariants, flow model, lattice QRS, cell selection, packing pipeline |
| `scripts/` | Command-line entry points (below) |
| `checkpoints/cellnet_flow_v9/` | Trained flow (`best.pt`), normalization and Hall vocabulary (`stats.json`), training history |
| `datasets/spade-csp/` | Training and test tables derived from SPaDe-CSP ([details](datasets/README.md)) |
| `benchmark/` | Benchmark code list, run and summary scripts, per-code results ([details](benchmark/README.md)) |
| `tests/` | Unit tests |

| Script | Purpose |
|--------|---------|
| `scripts/run_pipeline.py` | Full pipeline (flow → lattice QRS → selection → packing) for one molecule or benchmark code |
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
  --k 96 --skip-conf-qrs --output-dir outputs/aspirin
```

The flow draws (`*_flow_k96.npz`), the lattice-QRS cells (`*_lattice_qrs.csv`) and the
deduplicated cells (`*_unique_cells.csv`) are written under `outputs/aspirin/`.

Run the packing search as well, with the paper's cell budget and search settings:

```bash
python scripts/run_pipeline.py \
  --smiles "CC(=O)Oc1ccccc1C(=O)O" --hall 81 --zprime 1 --sage 2.0 \
  --k 96 --conf-selection multichannel --max-conf-runs 72 --cell-blind \
  --conf-qrs-origin-only --conf-npop 96 --no-conf-check-stable \
  --relax-lattice --conf-nproc 48
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

## Retraining

```bash
# 1. Precompute lattice targets and molecular graphs (slow; use several workers)
python scripts/precompute_lattice_cache.py --input datasets/spade-csp/spade_train.csv --workers 8
python scripts/precompute_lattice_cache.py --input datasets/spade-csp/spade_test.csv --workers 8

# 2. Train (defaults match the shipped checkpoint's settings)
python scripts/train_flow.py --precomputed --output-dir outputs/my_flow
```

The shipped checkpoint (epoch 130 of a 400-epoch run with early stopping) was fine-tuned
from an earlier flow trained on a smaller split, so training from scratch gives a comparable
but not identical model. `checkpoints/cellnet_flow_v9/best.pt` stores the full argument list
(`torch.load(...)["args"]`). To rebuild the tables themselves from the SPaDe-CSP release, see
[`datasets/README.md`](datasets/README.md).

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
