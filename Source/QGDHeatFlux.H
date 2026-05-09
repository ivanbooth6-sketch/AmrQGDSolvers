#ifndef QGD_HEAT_FLUX_H_
#define QGD_HEAT_FLUX_H_

#include <AMReX_GpuQualifiers.H>
#include <AMReX_REAL.H>

namespace qgd {

// Face-centered heat-flux regression components.  Kernels write all three
// physically meaningful contributions plus their sum so tests can compare the
// NS/Fourier term independently from the two tau-dependent QGD terms.
enum HeatFluxComponent : int {
    HeatFluxNS = 0,
    HeatFluxQGDPOverRho = 1,
    HeatFluxQGDInvRho = 2,
    HeatFluxTotal = 3,
    nHeatFluxComponents = 4
};

struct HeatFluxContributions
{
    amrex::Real ns = 0.0;
    amrex::Real qgd_p_over_rho = 0.0;
    amrex::Real qgd_inv_rho = 0.0;

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real total () const noexcept
    {
        return ns + qgd_p_over_rho + qgd_inv_rho;
    }
};

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real computeNSFourierHeatFlux (amrex::Real conductivity,
                                      amrex::Real temperature_hi,
                                      amrex::Real temperature_lo,
                                      amrex::Real dx) noexcept
{
    return -conductivity*(temperature_hi - temperature_lo) / dx;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
HeatFluxContributions computeQGDHeatFluxContributions (
    amrex::Real ns_heat_flux,
    amrex::Real tau,
    amrex::Real rho_face,
    amrex::Real normal_velocity,
    amrex::Real normal_velocity_for_normal_gradient,
    amrex::Real transverse_velocity_for_transverse_gradient,
    amrex::Real eps_hi,
    amrex::Real eps_lo,
    amrex::Real eps_transverse_hi,
    amrex::Real eps_transverse_lo,
    amrex::Real inv_rho_hi,
    amrex::Real inv_rho_lo,
    amrex::Real inv_rho_transverse_hi,
    amrex::Real inv_rho_transverse_lo,
    amrex::Real pressure_face,
    amrex::Real dnormal,
    amrex::Real dtransverse) noexcept
{
    const amrex::Real eps_gradient_projection =
        normal_velocity_for_normal_gradient*(eps_hi - eps_lo) / dnormal
        + transverse_velocity_for_transverse_gradient*(eps_transverse_hi - eps_transverse_lo) / dtransverse;

    const amrex::Real inv_rho_gradient_projection =
        normal_velocity_for_normal_gradient*(inv_rho_hi - inv_rho_lo) / dnormal
        + transverse_velocity_for_transverse_gradient*(inv_rho_transverse_hi - inv_rho_transverse_lo) / dtransverse;

    const amrex::Real qgd_scale = -tau*rho_face*normal_velocity;

    return {ns_heat_flux,
            qgd_scale*eps_gradient_projection,
            qgd_scale*pressure_face*inv_rho_gradient_projection};
}

} // namespace qgd

#endif
