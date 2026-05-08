#!/usr/bin/env python3
"""Regression checks for the rho-gradient stencil axis convention.

The production code uses AMReX's 2-D index convention: ``i`` advances in x and
``j`` advances in y.  This test deliberately uses an anisotropic mesh and a
non-uniform field whose x and y derivatives differ, so swapping the axes is not
masked by equal grid spacing or symmetric data.
"""
from __future__ import annotations

import math
import re
from pathlib import Path


def rho(i: int, j: int, dx0: float, dx1: float) -> float:
    x = (i + 0.5) * dx0
    y = (j + 0.5) * dx1
    return 1.0 + 2.0 * x + 5.0 * y + 0.25 * x * y


def centered_mag_grad_rho(i: int, j: int, dx0: float, dx1: float) -> float:
    grad_x = 0.5 * (rho(i + 1, j, dx0, dx1) - rho(i - 1, j, dx0, dx1)) / dx0
    grad_y = 0.5 * (rho(i, j + 1, dx0, dx1) - rho(i, j - 1, dx0, dx1)) / dx1
    return math.hypot(grad_x, grad_y)


def swapped_mag_grad_rho(i: int, j: int, dx0: float, dx1: float) -> float:
    grad_x = 0.5 * (rho(i, j + 1, dx0, dx1) - rho(i, j - 1, dx0, dx1)) / dx0
    grad_y = 0.5 * (rho(i + 1, j, dx0, dx1) - rho(i - 1, j, dx0, dx1)) / dx1
    return math.hypot(grad_x, grad_y)


def test_anisotropic_nonuniform_case_exposes_axis_swap() -> None:
    dx0 = 0.125
    dx1 = 0.5
    i = 3
    j = 2

    expected = centered_mag_grad_rho(i, j, dx0, dx1)
    swapped = swapped_mag_grad_rho(i, j, dx0, dx1)

    assert not math.isclose(expected, swapped, rel_tol=1.0e-12, abs_tol=1.0e-12)
    assert math.isclose(expected, math.hypot(2.0 + 0.25 * ((j + 0.5) * dx1),
                                            5.0 + 0.25 * ((i + 0.5) * dx0)),
                        rel_tol=1.0e-12, abs_tol=1.0e-12)


def test_advance_and_refinement_use_x_i_y_j_stencils() -> None:
    repo = Path(__file__).resolve().parents[1]

    advance = (repo / "Source" / "QGD_advance.cpp").read_text()
    advance_block = re.search(
        r"// solve magGradRho(?P<body>.*?)StateUpdate\[bi\]\(i,j,k,UMAGGRADRHO\)",
        advance,
        re.DOTALL,
    )
    assert advance_block is not None
    advance_body = advance_block.group("body")
    assert "GradRhoX = 0.5*(PrimOld[bi](i+1,j,k,QRHO) - PrimOld[bi](i-1,j,k,QRHO)) / dx[0]" in advance_body
    assert "GradRhoY = 0.5*(PrimOld[bi](i,j+1,k,QRHO) - PrimOld[bi](i,j-1,k,QRHO)) / dx[1]" in advance_body

    tagging = (repo / "Source" / "AmrQGD.cpp").read_text()
    tagging_block = re.search(
        r"else if \(refcond == 4\).*?\{(?P<body>.*?)if \(\(gradRho >= refdengrad\)",
        tagging,
        re.DOTALL,
    )
    assert tagging_block is not None
    tagging_body = tagging_block.group("body")
    assert "gradRhoX  = 0.5*(rho_at(i+1,j) - rho_at(i-1,j)) / dx[0]" in tagging_body
    assert "gradRhoY  = 0.5*(rho_at(i,j+1) - rho_at(i,j-1)) / dx[1]" in tagging_body


if __name__ == "__main__":
    test_anisotropic_nonuniform_case_exposes_axis_swap()
    test_advance_and_refinement_use_x_i_y_j_stencils()
