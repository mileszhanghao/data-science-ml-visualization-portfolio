# Numerical Linear Algebra Toolkit

Portfolio-safe applied linear algebra and numerical analysis project based on AMATH-style coursework themes. The code focuses on reusable numerical methods rather than course prompts or submitted homework text.

## Scope

This module collects small, inspectable implementations for common numerical linear algebra tasks:

- QR-based least-squares solving.
- Matrix conditioning diagnostics.
- Power iteration for dominant eigenvalue estimation.
- Conjugate-gradient solves for symmetric positive-definite systems.
- Residual and relative-error checks for solver validation.

## Why This Matters

Numerical computing work depends on choosing algorithms that are stable enough for the matrix structure in front of you. The toolkit highlights the habits that matter in applied analysis: checking conditioning, validating residuals, avoiding unnecessary explicit inverses, and using iterative methods when direct factorization is not the right fit.

## Files

- `numerical_linear_algebra.py`: reusable NumPy implementations and a small demonstration.

## Skills Demonstrated

- NumPy implementation of matrix algorithms
- Least-squares modeling
- QR factorization
- Eigenvalue approximation
- Iterative solvers
- Conditioning and residual diagnostics
- Numerical stability reasoning

## Portfolio Boundary

The public version is a clean rewrite that demonstrates the methods and engineering patterns without publishing private assignment prompts, solutions, or course-specific scaffolding.
