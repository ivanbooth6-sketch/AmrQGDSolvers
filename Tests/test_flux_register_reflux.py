#!/usr/bin/env python3
"""Regression checks for conservative AMR reflux bookkeeping."""
from __future__ import annotations

from pathlib import Path


def test_fine_flux_register_is_reset_before_coarse_flux_accumulation() -> None:
    repo = Path(__file__).resolve().parents[1]
    advance = (repo / "Source" / "QGD_advance.cpp").read_text()

    reset = "fine_level.flux_reg->setVal(0.0);"
    crse_init = "fine_level.flux_reg->CrseInit("

    assert reset in advance
    assert crse_init in advance
    assert advance.index(reset) < advance.index(crse_init)


def test_crse_init_accumulates_all_coarse_face_directions() -> None:
    repo = Path(__file__).resolve().parents[1]
    advance = (repo / "Source" / "QGD_advance.cpp").read_text()

    assert "for (int dir = 0; dir < AMREX_SPACEDIM; ++dir)" in advance
    assert (
        "fine_level.flux_reg->CrseInit(fluxes[dir], dir, 0, 0, nflux, "
        "-dt*area, FluxRegister::ADD);"
    ) in advance


def test_reflux_only_updates_conservative_components() -> None:
    repo = Path(__file__).resolve().parents[1]
    amrqgd = (repo / "Source" / "AmrQGD.cpp").read_text()

    assert "fine_level.flux_reg->Reflux(S_crse, 1.0, 0, URHO, n_cons, Geom());" in amrqgd


if __name__ == "__main__":
    test_fine_flux_register_is_reset_before_coarse_flux_accumulation()
    test_crse_init_accumulates_all_coarse_face_directions()
    test_reflux_only_updates_conservative_components()
