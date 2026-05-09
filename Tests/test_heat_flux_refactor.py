#!/usr/bin/env python3
"""Regression checks for face-centered QGD heat-flux decomposition hooks."""
from __future__ import annotations

import math
from pathlib import Path


def test_qgd_heat_flux_split_matches_original_formula() -> None:
    kap = 0.37
    tau = 0.11
    rho = 1.7
    un = -0.8
    ut = 0.35
    t_hi = 3.1
    t_lo = 2.4
    eps_hi = 5.0
    eps_lo = 4.2
    eps_t_hi = 4.8
    eps_t_lo = 4.4
    inv_rho_hi = 0.62
    inv_rho_lo = 0.58
    inv_rho_t_hi = 0.71
    inv_rho_t_lo = 0.68
    pressure = 2.3
    dn = 0.2
    dt = 0.5

    ns = -kap * (t_hi - t_lo) / dn
    qgd_p_over_rho = -tau * rho * un * (un * (eps_hi - eps_lo) / dn + ut * (eps_t_hi - eps_t_lo) / dt)
    qgd_inv_rho = -tau * rho * un * pressure * (
        un * (inv_rho_hi - inv_rho_lo) / dn + ut * (inv_rho_t_hi - inv_rho_t_lo) / dt
    )
    split_total = ns + qgd_p_over_rho + qgd_inv_rho

    original_formula = ns - tau * rho * un * (
        un * (eps_hi - eps_lo) / dn + ut * (eps_t_hi - eps_t_lo) / dt
        + pressure * (un * (inv_rho_hi - inv_rho_lo) / dn + ut * (inv_rho_t_hi - inv_rho_t_lo) / dt)
    )

    assert math.isclose(split_total, original_formula, rel_tol=1.0e-14, abs_tol=1.0e-14)


def test_advance_writes_named_face_heat_flux_components() -> None:
    repo = Path(__file__).resolve().parents[1]
    header = (repo / "Source" / "QGDHeatFlux.H").read_text()
    advance = (repo / "Source" / "QGD_advance.cpp").read_text()
    make_package = (repo / "Source" / "Make.package").read_text()

    assert "QGDHeatFlux.H" in make_package
    assert "QGDHeatFlux.cpp" in make_package
    assert "computeNSFourierHeatFlux" in header
    assert "computeQGDHeatFluxContributions" in header
    assert "HeatFluxNS" in header
    assert "HeatFluxQGDPOverRho" in header
    assert "HeatFluxQGDInvRho" in header
    assert "Array<MultiFab, AMREX_SPACEDIM> face_heat_fluxes" in advance
    assert "face_heat_fluxes[dir].define(face_ba, S_new.DistributionMap(), qgd::nHeatFluxComponents, 0)" in advance
    assert "xheat[bi](i,j,k,qgd::HeatFluxTotal)" in advance
    assert "yheat[bi](i,j,k,qgd::HeatFluxTotal)" in advance
    assert "const double qxA =" not in advance
    assert "const double qyC =" not in advance


if __name__ == "__main__":
    test_qgd_heat_flux_split_matches_original_formula()
    test_advance_writes_named_face_heat_flux_components()
