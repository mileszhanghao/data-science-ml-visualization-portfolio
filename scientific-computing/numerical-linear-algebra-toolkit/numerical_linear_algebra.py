"""Small numerical linear algebra utilities for portfolio review.

The functions are intentionally compact and dependency-light. They demonstrate
the algorithmic structure behind common applied linear algebra tasks while
leaving private coursework prompts out of the public repository.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LinearSolveReport:
    """Diagnostic summary for a computed linear-system solution."""

    residual_norm: float
    relative_residual: float
    iterations: int | None = None
    converged: bool | None = None


def condition_number(matrix: np.ndarray) -> float:
    """Return the 2-norm condition number using singular values."""

    values = np.linalg.svd(np.asarray(matrix, dtype=float), compute_uv=False)
    smallest = values[-1]
    if np.isclose(smallest, 0.0):
        return float("inf")
    return float(values[0] / smallest)


def least_squares_qr(features: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Solve min ||Ax - b||_2 using a reduced QR factorization."""

    q_matrix, r_matrix = np.linalg.qr(np.asarray(features, dtype=float), mode="reduced")
    projected = q_matrix.T @ np.asarray(target, dtype=float)
    return np.linalg.solve(r_matrix, projected)


def power_iteration(
    matrix: np.ndarray,
    *,
    max_iter: int = 1_000,
    tolerance: float = 1e-10,
    seed: int = 0,
) -> tuple[float, np.ndarray, int]:
    """Estimate the dominant eigenvalue and eigenvector of a square matrix."""

    mat = np.asarray(matrix, dtype=float)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError("matrix must be square")

    rng = np.random.default_rng(seed)
    vector = rng.normal(size=mat.shape[0])
    vector = vector / np.linalg.norm(vector)
    previous_value = 0.0

    for iteration in range(1, max_iter + 1):
        next_vector = mat @ vector
        norm = np.linalg.norm(next_vector)
        if np.isclose(norm, 0.0):
            raise ValueError("power iteration encountered the zero vector")
        vector = next_vector / norm
        value = float(vector @ mat @ vector)
        if abs(value - previous_value) < tolerance:
            return value, vector, iteration
        previous_value = value

    return previous_value, vector, max_iter


def conjugate_gradient(
    matrix: np.ndarray,
    rhs: np.ndarray,
    *,
    tolerance: float = 1e-10,
    max_iter: int | None = None,
) -> tuple[np.ndarray, LinearSolveReport]:
    """Solve Ax = b for symmetric positive-definite A using CG."""

    mat = np.asarray(matrix, dtype=float)
    b_vec = np.asarray(rhs, dtype=float)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError("matrix must be square")
    if mat.shape[0] != b_vec.shape[0]:
        raise ValueError("rhs length must match matrix dimensions")

    max_iter = max_iter or mat.shape[0] * 10
    solution = np.zeros_like(b_vec, dtype=float)
    residual = b_vec - mat @ solution
    direction = residual.copy()
    residual_dot = float(residual @ residual)
    rhs_norm = np.linalg.norm(b_vec) or 1.0

    for iteration in range(1, max_iter + 1):
        mat_direction = mat @ direction
        step = residual_dot / float(direction @ mat_direction)
        solution = solution + step * direction
        residual = residual - step * mat_direction
        next_residual_dot = float(residual @ residual)
        relative_residual = np.sqrt(next_residual_dot) / rhs_norm
        if relative_residual < tolerance:
            return solution, LinearSolveReport(
                residual_norm=float(np.sqrt(next_residual_dot)),
                relative_residual=float(relative_residual),
                iterations=iteration,
                converged=True,
            )
        direction = residual + (next_residual_dot / residual_dot) * direction
        residual_dot = next_residual_dot

    residual_norm = float(np.linalg.norm(b_vec - mat @ solution))
    return solution, LinearSolveReport(
        residual_norm=residual_norm,
        relative_residual=float(residual_norm / rhs_norm),
        iterations=max_iter,
        converged=False,
    )


def solve_report(matrix: np.ndarray, solution: np.ndarray, rhs: np.ndarray) -> LinearSolveReport:
    """Compute direct residual diagnostics for a proposed solution."""

    residual = np.asarray(rhs, dtype=float) - np.asarray(matrix, dtype=float) @ np.asarray(solution, dtype=float)
    residual_norm = float(np.linalg.norm(residual))
    rhs_norm = float(np.linalg.norm(rhs)) or 1.0
    return LinearSolveReport(
        residual_norm=residual_norm,
        relative_residual=float(residual_norm / rhs_norm),
    )


if __name__ == "__main__":
    a_matrix = np.array([[4.0, 1.0], [1.0, 3.0]])
    b_vector = np.array([1.0, 2.0])
    solution, report = conjugate_gradient(a_matrix, b_vector)
    eigenvalue, _, iterations = power_iteration(a_matrix)
    print("condition_number", round(condition_number(a_matrix), 4))
    print("cg_solution", np.round(solution, 4).tolist())
    print("cg_report", report)
    print("dominant_eigenvalue", round(eigenvalue, 4), "iterations", iterations)
