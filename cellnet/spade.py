"""SPaDe-CSP dataset helpers: extract, Hall lookup, and loading."""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import spglib

from cellnet.data import CrystalSample
from cellnet.reciprocal import cellpar_to_reciprocal_metric
from cellnet.symmetry import apply_cellpar_constraints, crystal_system_from_hall, crystal_system_from_spg

SPADE_EXPORT_COLUMNS = [
    "refcode",
    "sg_symbol",
    "sg_number",
    "hall_number",
    "smiles",
    "zprime",
    "a",
    "b",
    "c",
    "alpha",
    "beta",
    "gamma",
    "density",
]


@lru_cache(maxsize=1)
def spg_to_default_hall() -> dict[int, int]:
    """Map international space-group number → default Hall number (first spglib setting)."""
    mapping: dict[int, int] = {}
    for hall in range(1, 531):
        try:
            sg = spglib.get_spacegroup_type(hall)
        except Exception:
            continue
        mapping.setdefault(int(sg.number), int(hall))
    return mapping


def hall_number_from_spg(sg_number: int) -> int:
    """Return default Hall number for an international space-group number."""
    mapping = spg_to_default_hall()
    sg_number = int(sg_number)
    if sg_number not in mapping:
        raise KeyError(f"No Hall number found for space group {sg_number}")
    return mapping[sg_number]


# Known CSD symbol aliases → (sg_number, canonical PyXtal/CSD symbol).
CSD_SG_SYMBOL_ALIASES: dict[tuple[int, str], str] = {
    (61, "pcab"): "pbca",
    # Monoclinic origin / setting variants (CSD vs PyXtal compact symbols).
    (4, "p1121"): "p21",
    (4, "p2111"): "p21",
    (5, "b112"): "c2",
    (9, "b11b"): "c11b",
    (9, "a11a"): "c11a",
    (14, "p1121/b"): "p21/c",
    (14, "p1121/n"): "p21/n",
    (14, "p1121/a"): "p21/a",
    (14, "b21/c"): "p21/c",
    (14, "b21/a"): "p21/a",
    (14, "p21/c11"): "p21/c",
    (14, "p21/b11"): "p21/c",
    (15, "b112/b"): "c2/c",
    (15, "a112/a"): "c2/c",
    (15, "i112/a"): "i2/a",
    (18, "p22121"): "p21212",
    (18, "p21221"): "p21212",
    (29, "pbc21"): "pca21",
    (29, "pc21b"): "pca21",
    (29, "p21ab"): "pca21",
    (29, "p21ca"): "pca21",
    (29, "pb21a"): "pca21",
    (33, "pn21a"): "pna21",
    (33, "p21cn"): "pna21",
    (33, "p21nb"): "pna21",
    (33, "pc21n"): "pna21",
    (33, "pbn21"): "pna21",
    (41, "c2cb"): "cc2",
    (43, "f2dd"): "fdd2",
    (45, "i2cb"): "ib2",
    (56, "pnaa"): "pccn",
    (56, "pbnb"): "pccn",
    (60, "pnab"): "pbcn",
    (60, "pbna"): "pbcn",
    (60, "pnca"): "pbcn",
    (60, "pcan"): "pbcn",
    (60, "pcnb"): "pbcn",
    (152, "p3121"): "p312",
    (154, "p3221"): "p322",
    (158, "p3c1"): "p31c",
}

# PyXtal Hall numbers for rhombohedral (:R) settings → hexagonal (:H) partners.
RHOMBOHEDRAL_TO_HEX_HALL: dict[int, int] = {
    434: 433,  # R3
    437: 436,  # R-3
    451: 450,  # R3m
    453: 452,  # R3c
    459: 458,  # R-3m
    461: 460,  # R-3c
}


def hall_symbol_to_csd(hall_symbol: str) -> str:
    """Convert a PyXtal/spglib Hall symbol to CSD-style compact notation."""
    s = hall_symbol.strip()
    if ":" in s:
        s = s.split(":")[0]
    parts = s.split()
    if len(parts) >= 2 and parts[-1] == "1" and not (len(parts) == 2 and parts[0] == "P"):
        parts = parts[:-1]
    if len(parts) >= 3 and parts[1] == "1":
        parts = [parts[0]] + parts[2:]
    return "".join(parts)


@lru_cache(maxsize=512)
def _hall_settings_for_spg(sg_number: int) -> tuple[tuple[int, str, str], ...]:
    """Return (hall_number, csd_symbol, full_hall_symbol) for an international SG."""
    from pyxtal.symmetry import Hall

    h = Hall(int(sg_number))
    return tuple(
        (int(num), hall_symbol_to_csd(sym), sym)
        for num, sym in zip(h.hall_numbers, h.hall_symbols)
    )


def hall_number_from_sg_symbol(sg_number: int, sg_symbol: str) -> tuple[int, str, str] | None:
    """
    Map (international SG number, CSD sg_symbol) → (Hall number, CSD Hall symbol, full Hall symbol).

    Returns None when no PyXtal Hall setting matches the symbol.
    """
    resolved = resolve_hall_from_sg_symbol(sg_number, sg_symbol, allow_fallback=False)
    if resolved is None:
        return None
    hall_number, hall_csd, full_sym, _source = resolved
    return hall_number, hall_csd, full_sym


def _normalize_csd_symbol(sg_symbol: str) -> str:
    return str(sg_symbol).strip().replace(" ", "").lower()


def resolve_hall_from_sg_symbol(
    sg_number: int,
    sg_symbol: str,
    *,
    allow_fallback: bool = False,
) -> tuple[int, str, str, str] | None:
    """
    Resolve Hall number from CSD space-group metadata.

    Returns ``(hall_number, csd_symbol, full_hall_symbol, source)`` where
    ``source`` is ``symbol``, ``alias``, or ``fallback``.
    """
    sg_number = int(sg_number)
    sym_key = _normalize_csd_symbol(sg_symbol)
    alias = CSD_SG_SYMBOL_ALIASES.get((sg_number, sym_key))
    lookup_key = alias if alias is not None else sym_key
    for hall_num, csd_sym, full_sym in _hall_settings_for_spg(sg_number):
        if csd_sym.lower() == lookup_key:
            source = "alias" if alias is not None else "symbol"
            return int(hall_num), csd_sym, full_sym, source
    if not allow_fallback:
        return None
    hall_number = hall_number_from_spg(sg_number)
    for hall_num, csd_sym, full_sym in _hall_settings_for_spg(sg_number):
        if int(hall_num) == hall_number:
            return hall_number, csd_sym, full_sym, "fallback"
    return hall_number, str(sg_symbol), "", "fallback"


def hall_number_to_hex_setting(hall_number: int) -> int:
    """Map a rhombohedral (:R) Hall number to its hexagonal (:H) partner when known."""
    hall_number = int(hall_number)
    return RHOMBOHEDRAL_TO_HEX_HALL.get(hall_number, hall_number)


def is_rhombohedral_cellpar(cellpar: np.ndarray, hall_number: int | None = None) -> bool:
    """
    True when cell parameters use a rhombohedral description (a≈b≈c, α≈β≈γ, not hex).

    Also true when ``hall_number`` is a known rhombohedral (:R) Hall setting.
    """
    if hall_number is not None:
        hall_number = int(hall_number)
        if hall_number in RHOMBOHEDRAL_TO_HEX_HALL:
            return True
        sg_type = spglib.get_spacegroup_type(hall_number)
        if not str(sg_type.international_short).startswith("R"):
            return False
    cp = np.asarray(cellpar, dtype=np.float64)
    lengths = cp[:3]
    angles = cp[3:6]
    if np.ptp(lengths) / max(float(lengths.mean()), 1e-8) > 0.02:
        return False
    if np.ptp(angles) > 1.0:
        return False
    if abs(float(angles[2]) - 120.0) < 0.75 and abs(float(angles[0]) - 90.0) < 0.75:
        return False
    return True


def convert_cellpar_to_hexagonal_setting(
    cellpar: np.ndarray,
    hall_number: int,
) -> tuple[np.ndarray, int]:
    """
    Convert rhombohedral lattice descriptions to hexagonal setting.

    A primitive rhombohedral cell becomes the conventional R-centered
    hexagonal cell, whose volume is three times larger. Returns
    ``(cellpar_hex, hall_number_hex)``.
    """
    hall_number = hall_number_to_hex_setting(int(hall_number))
    cp = np.array(cellpar, dtype=np.float64, copy=True)
    if not is_rhombohedral_cellpar(cp, hall_number):
        return apply_cellpar_constraints(cp, hall_number), hall_number

    from pyxtal.lattice import Lattice

    system = crystal_system_from_hall(hall_number)
    lat = Lattice.from_para(*cp, ltype=system)
    cell = (lat.matrix, np.zeros((1, 3)), np.ones(1, dtype=int))
    std = spglib.standardize_cell(cell, to_primitive=False, no_idealize=False)
    if std is None:
        return apply_cellpar_constraints(cp, hall_number), hall_number

    lat_hex = Lattice.from_matrix(std[0])
    cp_hex = np.array(lat_hex.get_para(degree=True), dtype=np.float64)
    cp_hex[0] = cp_hex[1] = 0.5 * (cp_hex[0] + cp_hex[1])
    cp_hex[3] = 90.0
    cp_hex[4] = 90.0
    cp_hex[5] = 120.0
    return apply_cellpar_constraints(cp_hex, hall_number), hall_number


@dataclass(frozen=True)
class ExtractStats:
    written: int
    skipped_unmapped_hall: int
    skipped_missing_smiles: int

    @property
    def skipped(self) -> int:
        return self.skipped_unmapped_hall + self.skipped_missing_smiles


@dataclass(frozen=True)
class ExtractStatsV9(ExtractStats):
    alias_mapped: int = 0
    fallback_mapped: int = 0
    rhombo_converted: int = 0


def _row_to_export_v9(row: dict, *, convert_rhombo: bool = True) -> tuple[dict, str] | None:
    smiles = str(row.get("SMILES", "")).strip()
    if not smiles:
        return None
    sg_number = int(float(row["sg_number"]))
    sg_symbol = str(row.get("sg_symbol", "")).strip()
    resolved = resolve_hall_from_sg_symbol(sg_number, sg_symbol, allow_fallback=True)
    if resolved is None:
        return None
    hall_number, _hall_csd, _full, source = resolved
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
    converted = False
    if convert_rhombo and (
        hall_number in RHOMBOHEDRAL_TO_HEX_HALL or is_rhombohedral_cellpar(cellpar, hall_number)
    ):
        cellpar, hall_number = convert_cellpar_to_hexagonal_setting(cellpar, hall_number)
        converted = True
    return (
        {
            "refcode": str(row.get("refcode", "")).strip(),
            "sg_symbol": sg_symbol,
            "sg_number": sg_number,
            "hall_number": hall_number,
            "smiles": smiles,
            "zprime": float(row["Z-prime"]),
            "a": float(cellpar[0]),
            "b": float(cellpar[1]),
            "c": float(cellpar[2]),
            "alpha": float(cellpar[3]),
            "beta": float(cellpar[4]),
            "gamma": float(cellpar[5]),
            "density": float(row["density"]),
        },
        ("rhombo_hex" if converted else source),
    )


def extract_v9_csv(
    input_path: str | Path,
    output_path: str | Path,
    *,
    convert_rhombo: bool = True,
) -> ExtractStatsV9:
    """
    Extract all rows from ``crystal-info_CSD_filtered.csv`` (170,278 release).

    Uses expanded CSD symbol aliases, default-Hall fallback for residual symbols,
    and rhombohedral→hexagonal cell conversion for R-family settings.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_written = 0
    skipped_unmapped_hall = 0
    skipped_missing_smiles = 0
    alias_mapped = 0
    fallback_mapped = 0
    rhombo_converted = 0

    with input_path.open(newline="") as fin, output_path.open("w", newline="") as fout:
        reader = csv.DictReader(fin)
        writer = csv.DictWriter(fout, fieldnames=SPADE_EXPORT_COLUMNS)
        writer.writeheader()
        for row in reader:
            if not str(row.get("SMILES", "")).strip():
                skipped_missing_smiles += 1
                continue
            sg_number = int(float(row["sg_number"]))
            sg_symbol = str(row.get("sg_symbol", "")).strip()
            if resolve_hall_from_sg_symbol(sg_number, sg_symbol, allow_fallback=False) is None:
                if resolve_hall_from_sg_symbol(sg_number, sg_symbol, allow_fallback=True) is None:
                    skipped_unmapped_hall += 1
                    continue
            exported = _row_to_export_v9(row, convert_rhombo=convert_rhombo)
            if exported is None:
                skipped_missing_smiles += 1
                continue
            out, source = exported
            if source == "alias":
                alias_mapped += 1
            elif source == "fallback":
                fallback_mapped += 1
            elif source == "rhombo_hex":
                rhombo_converted += 1
            writer.writerow(out)
            n_written += 1

    return ExtractStatsV9(
        written=n_written,
        skipped_unmapped_hall=skipped_unmapped_hall,
        skipped_missing_smiles=skipped_missing_smiles,
        alias_mapped=alias_mapped,
        fallback_mapped=fallback_mapped,
        rhombo_converted=rhombo_converted,
    )


def split_crystal_info_csv(
    input_path: str | Path,
    train_path: str | Path,
    test_path: str | Path,
    *,
    test_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[int, int]:
    """
    Deterministic refcode split of ``crystal-info_CSD_filtered.csv`` into train/test CSVs.

    Uses the same column layout as the SPaDe release files (without ``Mol``).
    """
    input_path = Path(input_path)
    train_path = Path(train_path)
    test_path = Path(test_path)
    train_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open(newline="") as fin:
        reader = csv.DictReader(fin)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    def _is_test(refcode: str) -> bool:
        digest = hashlib.sha256(f"{seed}:{refcode}".encode()).hexdigest()
        bucket = int(digest[:8], 16) / 0xFFFFFFFF
        return bucket < test_fraction

    n_train = n_test = 0
    with train_path.open("w", newline="") as ftrain, test_path.open("w", newline="") as ftest:
        train_writer = csv.DictWriter(ftrain, fieldnames=fieldnames)
        test_writer = csv.DictWriter(ftest, fieldnames=fieldnames)
        train_writer.writeheader()
        test_writer.writeheader()
        for row in rows:
            if _is_test(str(row.get("refcode", ""))):
                test_writer.writerow(row)
                n_test += 1
            else:
                train_writer.writerow(row)
                n_train += 1
    return n_train, n_test


def _row_to_export(row: dict) -> dict | None:
    smiles = str(row.get("SMILES", "")).strip()
    if not smiles:
        return None
    sg_number = int(float(row["sg_number"]))
    sg_symbol = str(row.get("sg_symbol", "")).strip()
    hall = hall_number_from_sg_symbol(sg_number, sg_symbol)
    if hall is None:
        return None
    hall_number, _hall_csd, _full = hall
    return {
        "refcode": str(row.get("refcode", "")).strip(),
        "sg_symbol": sg_symbol,
        "sg_number": sg_number,
        "hall_number": hall_number,
        "smiles": smiles,
        "zprime": float(row["Z-prime"]),
        "a": float(row["a"]),
        "b": float(row["b"]),
        "c": float(row["c"]),
        "alpha": float(row["alpha"]),
        "beta": float(row["beta"]),
        "gamma": float(row["gamma"]),
        "density": float(row["density"]),
    }


def extract_spade_csv(input_path: str | Path, output_path: str | Path) -> ExtractStats:
    """Extract normalized columns from a SPaDe train/test CSV."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_written = 0
    skipped_unmapped_hall = 0
    skipped_missing_smiles = 0
    with input_path.open(newline="") as fin, output_path.open("w", newline="") as fout:
        reader = csv.DictReader(fin)
        writer = csv.DictWriter(fout, fieldnames=SPADE_EXPORT_COLUMNS)
        writer.writeheader()
        for row in reader:
            if not str(row.get("SMILES", "")).strip():
                skipped_missing_smiles += 1
                continue
            sg_number = int(float(row["sg_number"]))
            sg_symbol = str(row.get("sg_symbol", "")).strip()
            if hall_number_from_sg_symbol(sg_number, sg_symbol) is None:
                skipped_unmapped_hall += 1
                continue
            out = _row_to_export(row)
            assert out is not None
            writer.writerow(out)
            n_written += 1
    return ExtractStats(
        written=n_written,
        skipped_unmapped_hall=skipped_unmapped_hall,
        skipped_missing_smiles=skipped_missing_smiles,
    )


def load_spade_csv(
    csv_path: str | Path,
    max_samples: int | None = None,
    verbose: bool = True,
) -> list[CrystalSample]:
    """Load extracted SPaDe CSV into CrystalSample records."""
    csv_path = Path(csv_path)
    samples: list[CrystalSample] = []

    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for i, row in enumerate(reader):
            if max_samples is not None and i >= max_samples:
                break

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
            rec_metric = cellpar_to_reciprocal_metric(cellpar)

            samples.append(
                CrystalSample(
                    id=i,
                    csd_code=str(row.get("refcode", f"SPADE_{i}")),
                    smiles=str(row["smiles"]),
                    hall_number=hall_number,
                    zprime=float(row["zprime"]),
                    spg_num=sg_number,
                    space_group=str(row.get("sg_symbol", "")),
                    l_type="",
                    cellpar=cellpar,
                    reciprocal_metric=rec_metric,
                )
            )

    if verbose:
        print(f"Loaded {len(samples)} SPaDe samples from {csv_path}")
    return samples


def load_structure_csv(
    csv_path: str | Path,
    max_samples: int | None = None,
    verbose: bool = True,
) -> list[CrystalSample]:
    """Load HEM or extracted SPaDe CSV by header inspection."""
    from cellnet.data import load_hem_csv

    csv_path = Path(csv_path)
    with csv_path.open(newline="") as handle:
        header = handle.readline()
    if "cell_parameters" in header:
        return load_hem_csv(csv_path, max_samples=max_samples, verbose=verbose)
    return load_spade_csv(csv_path, max_samples=max_samples, verbose=verbose)
