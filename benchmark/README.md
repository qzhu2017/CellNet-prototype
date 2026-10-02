# Benchmark: Z′ ≤ 1 organic crystals

The benchmark set is the 100 organic crystals of Zhu & Hattori (*Digital Discovery*, 2025),
stored in `test.db`, which ships with PyXtal (`pyxtal/database/test.db`). A system is
eligible if it is a single component, its Hall setting is in the model vocabulary, and its
Z′ stays ≤ 1 after the special-position step. 84 systems qualify (`codes.txt`); two
single-component Z′ ≤ 1 entries become Z′ = 2 in the pipeline and are excluded
(`excluded.txt`). `list_eligible_codes.py` reproduces the selection.

## Protocol

`run_code.sh` runs one code:

- 96 flow draws (seed 42) → lattice QRS (12 Sobol stages × 1024 points) → 64 cells chosen by
  the cell-blind multichannel selection, including 12 slots for alternative unique-axis
  assignments of monoclinic cells;
- one packing search per cell (PyXtal QRS, 20 generations × 96 structures), each structure
  relaxed with CHARMM and OpenFF Sage 2.0.0 with the lattice free;
- if none of the 64 cells matches the experimental structure, the same cells are searched
  again with the lattice frozen.

A structure matches if pymatgen's `StructureMatcher` (H removed; ltol = stol = 0.3, angle
tolerance 5°) maps it onto the experimental structure and the largest site displacement is
below 0.3 Å. A system is **covered** if at least one cell yields a match. The success rate
(SR) of a cell is the percentage of its relaxed structures that match.

The experimental structure is used only for evaluation and, in this benchmark, for two
controls: a cell's search stops once it has three matches, and the frozen-lattice pass
runs only when the lattice-free pass has no match. Cell generation, selection and energy
ranking never see it.

## Results

`results.csv` lists the per-code outcome of the runs reported in the paper:

| | Systems |
|---|---:|
| Eligible | 84 |
| Covered, lattice free | 80 |
| Covered, frozen pass only (OBEQUJ, XAFPAY, XULDUD01) | 3 |
| Not covered (QAXMEH53) | 1 |
| **Coverage** | **83/84 (98.8 %)** |

Columns: `n_hit_cells` of `n_cells` lattice-free cells matched; `max_sr_percent` is the best
cell's SR; `first_hit_cell` is the position of the first matching cell in the sweep;
`n_fallback_*` refer to the frozen-lattice pass. Because a cell's search stops after three
matches, SR depends on that rule and is not comparable between systems.

For QAXMEH53, XULDUD01 and OBEQUJ, relaxing the experimental structure itself with the same
force field (lattice free) moves it outside the matching criteria, which is why the
frozen-lattice pass exists. It recovers OBEQUJ (4 cells), XULDUD01 (5 cells) and XAFPAY;
QAXMEH53 stays unmatched.

XAFPAY is marginal: a single structure in one cell matches (SR 0.05 %, largest displacement
0.24 Å against the 0.3 Å cap), and the same cell and seed gave no match in an earlier run with
older PyXtal and library versions. Fixed seeds make a run reproducible within one software
environment, not across environments: across the 5,493 cell searches shared by the two runs,
the hit/no-hit outcome agreed in 94 %. A rerun with other versions should reproduce the
coverage of the clear cases, but per-code hit counts can differ by a few cells, and single
low-SR matches such as XAFPAY's may come and go.

The paper's earlier 72-cell protocol (`MAX_CONF_RUNS=72`) gave 82/84, with XAFPAY a miss;
reducing the budget to 64 cells changed no other outcome.

## Running

```bash
bash benchmark/run_code.sh UREAXX02                   # one code -> outputs/benchmark/UREAXX02/
sbatch --array=0-83 benchmark/submit_slurm.sh         # all codes (edit the #SBATCH lines first)
python benchmark/summarize.py --root outputs/benchmark --out my_results.csv --compare benchmark/results.csv
```

Cost is dominated by the packing search and depends on the molecule and on the CPU type.
Median time per cell on 48 CPU cores, ranging over benchmark runs on different nodes:

| Case | Example | Time per cell |
|------|---------|--------------:|
| Small rigid | UREAXX02 | 1–2 min |
| Large rigid PAH | QUATER10 | 11–15 min |
| Flexible, 16 conformers | MIVDEC | 15–16 min |
| Large conformer pool | YOKBIK | 10–20 min |
| Large and flexible | QQQCIG04 | 12–21 min |

The same search runs up to about twice as fast on AMD EPYC (Genoa, Turin) nodes as on Intel
Xeon (Cascade Lake) nodes; on Slurm, `--constraint` can request a node type. A 64-cell sweep
therefore takes from about an hour to more than a day per code.
