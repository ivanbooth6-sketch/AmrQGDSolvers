#include "AmrQGD.H"
#include <cmath>

using namespace amrex;

void AmrQGD::initData ()
{
    const auto problo = Geom().ProbLoArray();
    const auto dx = Geom().CellSizeArray();
    MultiFab& S_new = get_new_data(State_Type);
    auto const& snew = S_new.arrays();

    amrex::ParallelFor(S_new,
    [=] AMREX_GPU_DEVICE (int bi, int i, int j, int k) noexcept
    {
        const Real rho = rhou;
        const Real ux = Uu;
        const Real uy = Vu;
        const Real p = pu;
        snew[bi](i,j,k,URHO) = rho;
        snew[bi](i,j,k,UMX) = rho*ux;
        snew[bi](i,j,k,UMY) = rho*uy;
        snew[bi](i,j,k,UENG) = p/(gamma - 1.) + 0.5*rho*(ux*ux + uy*uy);
        snew[bi](i,j,k,USC) = ScQgd;
        snew[bi](i,j,k,UCURL) = curl;
        snew[bi](i,j,k,UMAGGRADRHO) = magGradRho;
    });
    FillPatcherFill(S_new, 0, ncomp, nghost, 0, State_Type, 0); 
}

