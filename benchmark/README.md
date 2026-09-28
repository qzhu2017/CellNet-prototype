# Benchmark: Z′ ≤ 1 organic crystals

The benchmark set is the 100 organic crystals of Zhu & Hattori (*Digital Discovery*, 2025),
stored in `test.db`, which ships with PyXtal (`pyxtal/database/test.db`). A system is
eligible if it is a single component, its Hall setting is in the model vocabulary, and its
Z′ stays ≤ 1 after the special-position step. 84 systems qualify (`codes.txt`); two
single-component Z′ ≤ 1 entries become Z′ = 2 in the pipeline and are excluded
(`excluded.txt`). `list_eligible_codes.py` reproduces the selection.

## Protocol

`run_code.sh` runs one code:

- 96 flow draws (seed 42) → lattice QRS (12 Sobol stages × 1024 points) → 72 cells chosen by
  the cell-blind multichannel selection, including 12 slots for alternative unique-axis
  assignments of monoclinic cells;
- one packing search per cell (PyXtal QRS, 20 generations × 96 structures), each structure
  relaxed with CHARMM and OpenFF Sage 2.0.0 with the lattice free;
- if none of the 72 cells matches the experimental structure, the same cells are searched
  again with the lattice frozen.

A structure matches if pymatgen's `StructureMatcher` (H removed; ltol = stol = 0.3, angle
tolerance 5°) maps it onto the experimental structure and the largest site displacement is
below 0.3 Å. A system is **covered** if at least one cell yields a match. The success rate
(SR) of a cell is the percentage of its relaxed structures that match.

The experimental structure is used only for evaluation and, in this benchmark, for two
controls: a cell's search stops after more than three matches, and the frozen-lattice pass
runs only when the lattice-free pass has no match. Cell generation, selection and energy
ranking never see it.

## Results

`results.csv` lists the per-code outcome of the runs reported in the paper:

| | Systems |
|---|---:|
| Eligible | 84 |
| Covered, lattice free | 80 |
| Covered, frozen pass only (OBEQUJ, XULDUD01) | 2 |
| Not covered (QAXMEH53, XAFPAY) | 2 |
| **Coverage** | **82/84 (97.6 %)** |

Columns: `n_hit_cells` of `n_cells` lattice-free cells matched; `max_sr_percent` is the best
cell's SR; `first_hit_cell` is the position of the first matching cell in the sweep;
`n_fallback_*` refer to the frozen-lattice pass. Because a cell's search stops after three
matches, SR depends on that rule and is not comparable between systems.

For QAXMEH53, XULDUD01 and OBEQUJ, relaxing the experimental structure itself with the same
force field (lattice free) moves it outside the matching criteria, which is why the
frozen-lattice pass exists. XAFPAY matches if the displacement cap is raised from 0.30 to
0.35 Å.

A rerun reproduces coverage, but per-code hit counts can differ by a few cells: flow
sampling and multiprocess relaxations are not bit-for-bit deterministic across hardware and
library versions.

## Running

```bash
bash benchmark/run_code.sh UREAXX02                   # one code -> outputs/benchmark/UREAXX02/
sbatch --array=0-83 benchmark/submit_slurm.sh         # all codes (edit the #SBATCH lines first)
python benchmark/summarize.py --root outputs/benchmark --out my_results.csv --compare benchmark/results.csv
```

Cost is dominated by the packing search and depends on the molecule. Approximate time per
cell on 48 CPU cores:

| Case | Example | Time per cell |
|------|---------|--------------:|
| Small rigid | UREAXX02 | ~1 min |
| Large rigid PAH | QUATER10 | ~12 min |
| Flexible, 16 conformers | MIVDEC | ~12 min |
| Large conformer pool | YOKBIK | ~15 min |
| Large and flexible | QQQCIG04 | ~28 min |

A 72-cell sweep therefore takes from about an hour to more than a day per code.
