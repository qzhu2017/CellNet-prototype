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


def metric_tensor(matrix: np.ndarray) -> np.ndarray:
    """Symmetric metric tensor G = M @ M.T for a lattice matrix M (rows = basis vectors)."""
    return matrix @ matrix.T
