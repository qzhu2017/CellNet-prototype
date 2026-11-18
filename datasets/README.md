# Training data

`spade-csp/spade_train.csv` and `spade-csp/spade_test.csv` are the tables the shipped flow
was trained and tested on. They derive from the organic crystal structures compiled by
Taniguchi and Fukasawa for SPaDe-CSP (*Digital Discovery*, 2025,
[doi:10.1039/d5dd00304k](https://doi.org/10.1039/d5dd00304k);
[github.com/takuyhaa/SPaDe-CSP](https://github.com/takuyhaa/SPaDe-CSP)): non-polymeric,
solvent-free CSD entries with R < 10 % and Z′ = 1.

| | |
|---|---:|
| Structures | 170,278 |
| Train / test | 135,837 / 34,441 (80 / 20, seed 42) |
| Hall settings | 105 |
| Z′ | 1 |

Columns: `refcode`, `sg_symbol`, `sg_number`, `hall_number`, `smiles`, `zprime`,
`a`, `b`, `c`, `alpha`, `beta`, `gamma` (Å, degrees), `density` (g/cm³).

Compared with the original SPaDe-CSP split, all space groups of the filtered set are kept
(not only the 32 most common), each structure gets a Hall number from its CSD space-group
symbol (so that, e.g., P2₁/c, P2₁/n and P2₁/a are distinct settings), and rhombohedral cells
are stored in the conventional hexagonal setting. The five most common Hall settings account
for 80.8 % of the structures and the 15 most common for 95.9 %.

The tables contain CSD refcodes, SMILES and cell parameters only, no atomic coordinates.
Crystal structures themselves are available from the CSD under the CCDC licence.

## Rebuilding the tables

Download `crystal-info_CSD_filtered.csv` from the SPaDe-CSP repository into
`datasets/spade-csp/`, then:

```bash
python scripts/build_dataset.py \
  --crystal-info datasets/spade-csp/crystal-info_CSD_filtered.csv \
  --out-dir datasets/spade-csp --test-fraction 0.2 --seed 42
```

This writes the split (`split_train.csv`, `split_test.csv`) and the tables above.

## Training caches

Training reads precomputed targets and molecular graphs rather than the raw tables:

```bash
python scripts/precompute_lattice_cache.py --input datasets/spade-csp/spade_train.csv --workers 8
python scripts/precompute_lattice_cache.py --input datasets/spade-csp/spade_test.csv --workers 8
# optional, for GPU nodes with limited RAM:
python scripts/shard_graph_sidecar.py --input datasets/spade-csp/spade_train_precomputed_graphs.pt --shard-size 10000
```

Each run writes `*_precomputed.csv` (signed-log Selling scalars, log λ, log λ\* and log
density) and a graph sidecar `*_precomputed_graphs.pt` (about
800 MB for the training split). `--resume` continues an interrupted run; `--max-samples N`
runs on a subset. Training detects shards through `*_graphs_shards.json`.

## Z′ = 2 tables and the joint training set

`csd-z2/spade_train.csv` and `csd-z2/spade_test.csv` (8,234 / 2,164 crystals) hold CSD crystals
with Z′ = 2 whose two independent molecules are the same compound. They were selected with the
SPaDe-CSP filters above (organic, not polymeric, no solvent, R < 10 %; elements C H N O S P F
Cl Br I) by `scripts/collect_csd_zprime.py`, which needs the CSD Python API, and split 80 / 20 by
`scripts/build_dataset.py` (seed 42). They have the same columns as above, with `zprime` = 2.

The joint flow (`checkpoints/cellnet_flow_z12/`) and the space-group and density model
(`checkpoints/aux_sg_density/`) were trained on the joint tables:

```bash
python scripts/build_joint_dataset.py
```

The script compares molecules by InChIKey, ignoring stereo, and

1. removes the molecules in `benchmark_exclude_inchikeys.txt` (40 CSP test targets) from all
   training rows (90 Z′ = 1 rows);
2. removes the Z′ = 2 training rows whose molecule is in the Z′ = 1 test split (185 rows; the
   shipped `csd-z2/spade_train.csv` is already cleaned);
3. leaves both test splits unchanged.

This gives `joint-z12/train.csv`, 143,981 rows (Z′ = 1 135,747 + Z′ = 2 8,234), and
`joint-z12/test.csv`, 36,605 rows. `joint-z12/build_report.txt` lists the removed rows and the
final leak check. The caches are built as above (`training/joint_z12/precompute.sh`).

The original Z′ = 1 checkpoint (`checkpoints/cellnet_flow/`) was trained before this cleaning:
its training rows include 90 crystals of the test targets.
