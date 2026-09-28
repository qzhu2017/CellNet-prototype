"""
Reciprocal-space lattice utilities.

Following arXiv:2601.17981, cell-parameter changes primarily affect the Cartesian
coordinates of reciprocal lattice vectors. We represent the unit cell via
reciprocal cell parameters (a*, b*, c*, alpha*, beta*, gamma*) which have a
smooth, bijective relationship with direct cell parameters, then reconstruct
direct-space cell parameters analytically.

ASE convention: cell.array rows are direct lattice vectors a1, a2, a3.
"""

from __future__ import annotations

import numpy as np


def direct_matrix_from_cellpar(cellpar: np.ndarray) -> np.ndarray:
    """Build 3x3 direct lattice matrix from [a, b, c, alpha, beta, gamma] in degrees."""
    a, b, c, alpha, beta, gamma = cellpar
    alpha, beta, gamma = np.radians([alpha, beta, gamma])
    cos_a, cos_b, cos_g = np.cos([alpha, beta, gamma])
    sin_g = np.sin(gamma)

    ax, ay, az = a, 0.0, 0.0
    bx = b * cos_g
    by = b * sin_g
    bz = 0.0
    cx = c * cos_b
    cy = c * (cos_a - cos_b * cos_g) / sin_g
    cz = c * np.sqrt(max(0.0, 1.0 - cos_a**2 - cos_b**2 - cos_g**2 + 2 * cos_a * cos_b * cos_g)) / sin_g
    return np.array([[ax, ay, az], [bx, by, bz], [cx, cy, cz]], dtype=np.float64)


def cellpar_from_direct_matrix(matrix: np.ndarray) -> np.ndarray:
    """Recover [a, b, c, alpha, beta, gamma] from a 3x3 direct lattice matrix."""
    a = np.linalg.norm(matrix[0])
    b = np.linalg.norm(matrix[1])
    c = np.linalg.norm(matrix[2])
    alpha = np.degrees(np.arccos(np.clip(np.dot(matrix[1], matrix[2]) / (b * c), -1, 1)))
    beta = np.degrees(np.arccos(np.clip(np.dot(matrix[0], matrix[2]) / (a * c), -1, 1)))
    gamma = np.degrees(np.arccos(np.clip(np.dot(matrix[0], matrix[1]) / (a * b), -1, 1)))
    return np.array([a, b, c, alpha, beta, gamma], dtype=np.float64)


def reciprocal_matrix(direct_matrix: np.ndarray) -> np.ndarray:
    """Return reciprocal lattice matrix B (rows = b1, b2, b3)."""
    return 2.0 * np.pi * np.linalg.inv(direct_matrix).T


def direct_matrix_from_reciprocal(reciprocal_matrix: np.ndarray) -> np.ndarray:
    """Recover direct lattice matrix from reciprocal lattice matrix B."""
    return 2.0 * np.pi * np.linalg.inv(reciprocal_matrix).T


def metric_tensor(matrix: np.ndarray) -> np.ndarray:
    """Symmetric metric tensor G = M @ M.T for a lattice matrix M (rows = basis vectors)."""
    return matrix @ matrix.T


def metric_to_independent(g: np.ndarray) -> np.ndarray:
    """Pack symmetric 3x3 metric tensor into 6 independent components."""
    return np.array([g[0, 0], g[1, 1], g[2, 2], g[0, 1], g[0, 2], g[1, 2]], dtype=np.float64)


def independent_to_metric(x: np.ndarray) -> np.ndarray:
    """Unpack 6 components into a symmetric 3x3 metric tensor."""
    g11, g22, g33, g12, g13, g23 = x
    return np.array(
        [[g11, g12, g13], [g12, g22, g23], [g13, g23, g33]],
        dtype=np.float64,
    )


def _cell_volume(a: float, b: float, c: float, alpha: float, beta: float, gamma: float) -> float:
    """Unit cell volume from direct cell parameters (angles in degrees)."""
    ar, br, gr = np.radians([alpha, beta, gamma])
    cos_a, cos_b, cos_g = np.cos([ar, br, gr])
    term = 1.0 - cos_a**2 - cos_b**2 - cos_g**2 + 2.0 * cos_a * cos_b * cos_g
    return a * b * c * np.sqrt(max(0.0, term))


def cellpar_to_reciprocal_star(cellpar: np.ndarray) -> np.ndarray:
    """
    Convert direct cell parameters to reciprocal cell parameters.

    Returns [a*, b*, c*, alpha*, beta*, gamma*] in Angstroms and degrees.
    Uses standard crystallographic relations (International Tables).
    """
    a, b, c, alpha, beta, gamma = cellpar
    V = _cell_volume(a, b, c, alpha, beta, gamma)
    ar, br, gr = np.radians([alpha, beta, gamma])
    cos_a, cos_b, cos_g = np.cos([ar, br, gr])
    sin_a, sin_b, sin_g = np.sin([ar, br, gr])

    astar = b * c * sin_a / V
    bstar = a * c * sin_b / V
    cstar = a * b * sin_g / V

    cos_alpha_s = (cos_b * cos_g - cos_a) / (sin_b * sin_g + 1e-30)
    cos_beta_s = (cos_a * cos_g - cos_b) / (sin_a * sin_g + 1e-30)
    cos_gamma_s = (cos_a * cos_b - cos_g) / (sin_a * sin_b + 1e-30)

    alpha_s = np.degrees(np.arccos(np.clip(cos_alpha_s, -1, 1)))
    beta_s = np.degrees(np.arccos(np.clip(cos_beta_s, -1, 1)))
    gamma_s = np.degrees(np.arccos(np.clip(cos_gamma_s, -1, 1)))

    return np.array([astar, bstar, cstar, alpha_s, beta_s, gamma_s], dtype=np.float64)


def reciprocal_star_to_cellpar(star: np.ndarray) -> np.ndarray:
    """
    Convert reciprocal cell parameters back to direct cell parameters.

    Inverse of cellpar_to_reciprocal_star; exact bijective mapping.
    """
    astar, bstar, cstar, alpha_s, beta_s, gamma_s = star
    asr, bsr, gsr = np.radians([alpha_s, beta_s, gamma_s])
    cos_as, cos_bs, cos_gs = np.cos([asr, bsr, gsr])
    sin_as, sin_bs, sin_gs = np.sin([asr, bsr, gsr])

    # Volume in reciprocal space
    Vstar = astar * bstar * cstar * np.sqrt(
        max(0.0, 1.0 - cos_as**2 - cos_bs**2 - cos_gs**2 + 2.0 * cos_as * cos_bs * cos_gs)
    )

    a = bstar * cstar * sin_as / Vstar
    b = astar * cstar * sin_bs / Vstar
    c = astar * bstar * sin_gs / Vstar

    cos_alpha = (cos_bs * cos_gs - cos_as) / (sin_bs * sin_gs + 1e-30)
    cos_beta = (cos_as * cos_gs - cos_bs) / (sin_as * sin_gs + 1e-30)
    cos_gamma = (cos_as * cos_bs - cos_gs) / (sin_as * sin_bs + 1e-30)

    alpha = np.degrees(np.arccos(np.clip(cos_alpha, -1, 1)))
    beta = np.degrees(np.arccos(np.clip(cos_beta, -1, 1)))
    gamma = np.degrees(np.arccos(np.clip(cos_gamma, -1, 1)))

    return np.array([a, b, c, alpha, beta, gamma], dtype=np.float64)


# --- Primary API used by the models ---

def cellpar_to_reciprocal_metric(cellpar: np.ndarray) -> np.ndarray:
    """Alias: returns reciprocal star parameters (6 components)."""
    return cellpar_to_reciprocal_star(cellpar)


def reciprocal_metric_to_cellpar(star: np.ndarray) -> np.ndarray:
    """Alias: reconstruct direct cellpar from reciprocal star parameters."""
    return reciprocal_star_to_cellpar(star)


def reciprocal_star_parameters(cellpar: np.ndarray) -> np.ndarray:
    """Return reciprocal cell parameters [a*, b*, c*, alpha*, beta*, gamma*]."""
    return cellpar_to_reciprocal_star(cellpar)


def log_transform_metric(star: np.ndarray) -> np.ndarray:
    """Log-transform lengths (a*,b*,c*); keep angles in degrees."""
    out = star.copy()
    out[0] = np.log(max(out[0], 1e-12))
    out[1] = np.log(max(out[1], 1e-12))
    out[2] = np.log(max(out[2], 1e-12))
    return out


def inv_log_transform_metric(star: np.ndarray) -> np.ndarray:
    """Inverse of log_transform_metric."""
    out = star.copy()
    out[0] = np.exp(out[0])
    out[1] = np.exp(out[1])
    out[2] = np.exp(out[2])
    return out
