#include "AmrQGD.H"
#include <cmath>

using namespace amrex;

void
AmrQGD::initData ()
{
    const auto problo = Geom().ProbLoArray();
    const auto dx = Geom().CellSizeArray();
    MultiFab& S_new = get_new_data(State_Type);
    auto const& snew = S_new.arrays();

    amrex::ParallelFor(S_new,
    [=] AMREX_GPU_DEVICE (int bi, int i, int j, int k) noexcept
    {
        Real x = problo[0] + (i+0.5)*dx[0];
        Real y = problo[1] + (j+0.5)*dx[1];
//        Real z = problo[2] + (k+0.5)*dx[2];

        double R = 0.25;
        double o_x = 1.0;
        double o_y = 1.0;
        double o_z = 1.0;
        double dx = x - o_x;
        double dy = y - o_y;
//        double dz = z - o_z;
        double rr = dx*dx + dy*dy;// + dz*dz;
        double r = sqrt(rr);

        Real rho = 1.0;
        Real ux = 0.0;
        Real uy = 0.0;
        Real p = 1.0;
        if (y <= 0.5)     //(r <= R)
        {
            uy = 1.0;
        }

        snew[bi](i,j,k,URHO) = rho;
        snew[bi](i,j,k,UMX) = rho*ux;
        snew[bi](i,j,k,UMY) = rho*uy;
        snew[bi](i,j,k,UENG) = p/(gamma - 1.) + 0.5*rho*(ux*ux + uy*uy);
        snew[bi](i,j,k,USC) = ScQgd;
        snew[bi](i,j,k,UCURL) = 0.0;
        snew[bi](i,j,k,UMAGGRADRHO) = 0.0;
    });
    FillPatcherFill(S_new, 0, ncomp, nghost, 0, State_Type, 0);
    amrex::Print() << "Amr QGD solver will start with next params: " << "AlphaQQD = " << alphaQgd << " and ScQGD = " << ScQgd << "\n" 
                   << " varScNumber is " << varScQgd << " grad value is " << gradVal << "\n\n" ;
}

