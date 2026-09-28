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

Each run writes `*_precomputed.csv` (normalized Selling, log λ, log λ\*, log density and
symmetry-reduced free parameters) and a graph sidecar `*_precomputed_graphs.pt` (about
800 MB for the training split). `--resume` continues an interrupted run; `--max-samples N`
runs on a subset. Training detects shards through `*_graphs_shards.json`.
