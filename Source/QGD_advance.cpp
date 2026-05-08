#include "AmrQGD.H"

#include <AMReX_Array.H>
#include <AMReX_MultiFabUtil.H>

#include <iostream>

using namespace amrex;

Real AmrQGD::advance (Real time, Real dt, int iteration, int ncycle)
{
    // At the beginning of step, we make the new data from previous step the
    // old data of this step.
    for (int k = 0; k < NUM_STATE_TYPE; ++k) {
        state[k].allocOldData();
        state[k].swapTimeLevels(dt);
    }

    double mu_T = mutGas;

    auto dx = Geom().CellSizeArray();

    MultiFab& S_new = state[0].newData();
    auto const& VectNew = S_new.arrays();
    MultiFab& S_old = state[0].oldData();
    FillPatcherFill(S_old, 0, ncomp, nghost, time, State_Type, 0);

    auto const& VectOld = S_old.arrays();

    if (varScQgd)
    {
        amrex::ParallelFor(S_old, [=] AMREX_GPU_DEVICE (int bi, int i, int j, int k)
        {
            if (amrex::Math::abs(VectOld[bi](i,j,k,0) - VectOld[bi](i-1,j,k,0)) / dx[0] >= gradVal or
                amrex::Math::abs(VectOld[bi](i,j,k,0) - VectOld[bi](i,j-1,k,0)) / dx[1] >= gradVal or
                amrex::Math::abs(VectOld[bi](i,j,k,0) - VectOld[bi](i+1,j,k,0)) / dx[0] >= gradVal or
                amrex::Math::abs(VectOld[bi](i,j,k,0) - VectOld[bi](i,j+1,k,0)) / dx[1] >= gradVal)
            {
                VectOld[bi](i,j,k,4) = 4.0;
            }
            else
            {
                VectOld[bi](i,j,k,4) = ScQgd;
            }
        });
    }

    static constexpr int nflux = 4; // rho, rho*u, rho*v, E
    Array<MultiFab, AMREX_SPACEDIM> fluxes;
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        BoxArray face_ba = amrex::convert(S_new.boxArray(), IntVect::TheDimensionVector(dir));
        fluxes[dir].define(face_ba, S_new.DistributionMap(), nflux, 0);
        fluxes[dir].setVal(0.0);
    }

    auto const& xflux = fluxes[0].arrays();
    amrex::ParallelFor(fluxes[0], [=] AMREX_GPU_DEVICE (int bi, int i, int j, int k)
    {
        const int il = i - 1;
        const double ScQGD = VectOld[bi](il,j,k,4);

        const double ROA = 0.5*(VectOld[bi](il,j,k,0) + VectOld[bi](il+1,j,k,0));
        const double UxA = 0.5*(VectOld[bi](il,j,k,1) + VectOld[bi](il+1,j,k,1));
        const double UyA = 0.5*(VectOld[bi](il,j,k,2) + VectOld[bi](il+1,j,k,2));
        const double PA  = 0.5*(VectOld[bi](il,j,k,3) + VectOld[bi](il+1,j,k,3));

        const double ROE = 0.25*(VectOld[bi](il,j,k,0) + VectOld[bi](il+1,j,k,0) + VectOld[bi](il,j-1,k,0) + VectOld[bi](il+1,j-1,k,0));
        const double ROF = 0.25*(VectOld[bi](il,j,k,0) + VectOld[bi](il+1,j,k,0) + VectOld[bi](il,j+1,k,0) + VectOld[bi](il+1,j+1,k,0));

        const double UxE = 0.25*(VectOld[bi](il,j,k,1) + VectOld[bi](il+1,j,k,1) + VectOld[bi](il,j-1,k,1) + VectOld[bi](il+1,j-1,k,1));
        const double UxF = 0.25*(VectOld[bi](il,j,k,1) + VectOld[bi](il+1,j,k,1) + VectOld[bi](il,j+1,k,1) + VectOld[bi](il+1,j+1,k,1));

        const double UyE = 0.25*(VectOld[bi](il,j,k,2) + VectOld[bi](il+1,j,k,2) + VectOld[bi](il,j-1,k,2) + VectOld[bi](il+1,j-1,k,2));
        const double UyF = 0.25*(VectOld[bi](il,j,k,2) + VectOld[bi](il+1,j,k,2) + VectOld[bi](il,j+1,k,2) + VectOld[bi](il+1,j+1,k,2));

        const double PE = 0.25*(VectOld[bi](il,j,k,3) + VectOld[bi](il+1,j,k,3) + VectOld[bi](il,j-1,k,3) + VectOld[bi](il+1,j-1,k,3));
        const double PF = 0.25*(VectOld[bi](il,j,k,3) + VectOld[bi](il+1,j,k,3) + VectOld[bi](il,j+1,k,3) + VectOld[bi](il+1,j+1,k,3));

        const double CsA = sqrt(gamma*PA / ROA);
        const double hh = sqrt(dx[0]*dx[0] + dx[1]*dx[1]);
        const double TauA = alphaQgd*hh/CsA + mu_T / PA;
        const double muA = mu_T + TauA*PA*ScQGD;
        const double kapA = (mu_T / PrGas + PA*TauA*ScQGD / PrQgd)*gamma*RGas / (gamma - 1.);

        const double WxA = (TauA / ROA)*((ROF*UyF*UxF - ROE*UyE*UxE) / dx[1]
                                + (VectOld[bi](il+1,j,k,0)*VectOld[bi](il+1,j,k,1)*VectOld[bi](il+1,j,k,1) - VectOld[bi](il,j,k,0)*VectOld[bi](il,j,k,1)*VectOld[bi](il,j,k,1)) / dx[0]
                                + (VectOld[bi](il+1,j,k,3) - VectOld[bi](il,j,k,3)) / dx[0]);
        const double JmxA = ROA*(UxA - WxA);

        const double divuA = (VectOld[bi](il+1,j,k,1) - VectOld[bi](il,j,k,1)) / dx[0] + (UyF - UyE) / dx[1];
        const double PxxNSA = 2.*muA*(VectOld[bi](il+1,j,k,1) - VectOld[bi](il,j,k,1)) / dx[0] - (2./3.)*muA*divuA;
        const double PxyNSA = muA*((VectOld[bi](il+1,j,k,2) - VectOld[bi](il,j,k,2)) / dx[0] + (UxF - UxE) / dx[1]);
        const double RGA = TauA*(UxA*(VectOld[bi](il+1,j,k,3) - VectOld[bi](il,j,k,3)) / dx[0] + UyA*(PF - PE) / dx[1] + gamma*PA*divuA);

        const double WWxA = TauA*(UxA*(VectOld[bi](il+1,j,k,1) - VectOld[bi](il,j,k,1)) / dx[0] + UyA*(UxF - UxE) / dx[1] + (1 / ROA)*(VectOld[bi](il+1,j,k,3) - VectOld[bi](il,j,k,3)) / dx[0]);
        const double WWyA = TauA*(UxA*(VectOld[bi](il+1,j,k,2) - VectOld[bi](il,j,k,2)) / dx[0] + UyA*(UyF - UyE) / dx[1] + (1 / ROA)*(PF - PE) / dx[1]);
        const double PxxA = PxxNSA + ROA*UxA*WWxA + RGA;
        const double PxyA = PxyNSA + ROA*UxA*WWyA;

        const double T0 = VectOld[bi](il,j,k,3)   / (VectOld[bi](il,j,k,0)*RGas);
        const double T1 = VectOld[bi](il+1,j,k,3) / (VectOld[bi](il+1,j,k,0)*RGas);
        const double eps0 = VectOld[bi](il,j,k,3)   / (VectOld[bi](il,j,k,0)*(gamma - 1.));
        const double eps1 = VectOld[bi](il+1,j,k,3) / (VectOld[bi](il+1,j,k,0)*(gamma - 1.));
        const double epsA = PA / (ROA*(gamma - 1.));
        const double epsE = PE / (ROE*(gamma - 1.));
        const double epsF = PF / (ROF*(gamma - 1.));
        const double HA = UxA*UxA/2. + UyA*UyA/2. + gamma*epsA;
        const double qxNSA = -kapA*(T1 - T0) / dx[0];
        const double qxA = qxNSA - TauA*ROA*UxA*(UxA*(eps1 - eps0) / dx[0] + UyA*(epsF - epsE) / dx[1]
            + PA*(UxA*(1./VectOld[bi](il+1,j,k,0) - 1./VectOld[bi](il,j,k,0)) / dx[0] + UyA*(1./ROF - 1./ROE) / dx[1]));

        xflux[bi](i,j,k,0) = JmxA;
        xflux[bi](i,j,k,1) = JmxA*UxA + PA - PxxA;
        xflux[bi](i,j,k,2) = JmxA*UyA - PxyA;
        xflux[bi](i,j,k,3) = JmxA*HA + qxA - PxxA*UxA - PxyA*UyA;
    });

    auto const& yflux = fluxes[1].arrays();
    amrex::ParallelFor(fluxes[1], [=] AMREX_GPU_DEVICE (int bi, int i, int j, int k)
    {
        const int jb = j - 1;
        const double ScQGD = VectOld[bi](i,jb,k,4);

        const double ROC = 0.5*(VectOld[bi](i,jb,k,0) + VectOld[bi](i,jb+1,k,0));
        const double UxC = 0.5*(VectOld[bi](i,jb,k,1) + VectOld[bi](i,jb+1,k,1));
        const double UyC = 0.5*(VectOld[bi](i,jb,k,2) + VectOld[bi](i,jb+1,k,2));
        const double PC  = 0.5*(VectOld[bi](i,jb,k,3) + VectOld[bi](i,jb+1,k,3));

        const double ROF = 0.25*(VectOld[bi](i,jb,k,0) + VectOld[bi](i+1,jb,k,0) + VectOld[bi](i,jb+1,k,0) + VectOld[bi](i+1,jb+1,k,0));
        const double ROG = 0.25*(VectOld[bi](i,jb,k,0) + VectOld[bi](i-1,jb,k,0) + VectOld[bi](i,jb+1,k,0) + VectOld[bi](i-1,jb+1,k,0));

        const double UxF = 0.25*(VectOld[bi](i,jb,k,1) + VectOld[bi](i+1,jb,k,1) + VectOld[bi](i,jb+1,k,1) + VectOld[bi](i+1,jb+1,k,1));
        const double UxG = 0.25*(VectOld[bi](i,jb,k,1) + VectOld[bi](i-1,jb,k,1) + VectOld[bi](i,jb+1,k,1) + VectOld[bi](i-1,jb+1,k,1));

        const double UyF = 0.25*(VectOld[bi](i,jb,k,2) + VectOld[bi](i+1,jb,k,2) + VectOld[bi](i,jb+1,k,2) + VectOld[bi](i+1,jb+1,k,2));
        const double UyG = 0.25*(VectOld[bi](i,jb,k,2) + VectOld[bi](i-1,jb,k,2) + VectOld[bi](i,jb+1,k,2) + VectOld[bi](i-1,jb+1,k,2));

        const double PF = 0.25*(VectOld[bi](i,jb,k,3) + VectOld[bi](i+1,jb,k,3) + VectOld[bi](i,jb+1,k,3) + VectOld[bi](i+1,jb+1,k,3));
        const double PG = 0.25*(VectOld[bi](i,jb,k,3) + VectOld[bi](i-1,jb,k,3) + VectOld[bi](i,jb+1,k,3) + VectOld[bi](i-1,jb+1,k,3));

        const double CsC = sqrt(gamma*PC / ROC);
        const double hh = sqrt(dx[0]*dx[0] + dx[1]*dx[1]);
        const double TauC = alphaQgd*hh/CsC + mu_T / PC;
        const double muC = mu_T + TauC*PC*ScQGD;
        const double kapC = (mu_T / PrGas + PC*TauC*ScQGD / PrQgd)*gamma*RGas / (gamma - 1.);

        const double WyC = (TauC / ROC)*((ROF*UyF*UxF - ROG*UyG*UxG) / dx[0]
                                + (VectOld[bi](i,jb+1,k,0)*VectOld[bi](i,jb+1,k,2)*VectOld[bi](i,jb+1,k,2) - VectOld[bi](i,jb,k,0)*VectOld[bi](i,jb,k,2)*VectOld[bi](i,jb,k,2)) / dx[1]
                                + (VectOld[bi](i,jb+1,k,3) - VectOld[bi](i,jb,k,3)) / dx[1]);
        const double JmyC = ROC*(UyC - WyC);

        const double divuC = (VectOld[bi](i,jb+1,k,2) - VectOld[bi](i,jb,k,2)) / dx[1] + (UxF - UxG) / dx[0];
        const double PyxNSC = muC*((VectOld[bi](i,jb+1,k,1) - VectOld[bi](i,jb,k,1)) / dx[1] + (UyF - UyG) / dx[0]);
        const double PyyNSC = 2.*muC*(VectOld[bi](i,jb+1,k,2) - VectOld[bi](i,jb,k,2)) / dx[1] - (2./3.)*muC*divuC;
        const double RGC = TauC*(UyC*(VectOld[bi](i,jb+1,k,3) - VectOld[bi](i,jb,k,3)) / dx[1] + UxC*(PF - PG) / dx[0] + gamma*PC*divuC);

        const double WWxC = TauC*(UyC*(VectOld[bi](i,jb+1,k,1) - VectOld[bi](i,jb,k,1)) / dx[1] + UxC*(UxF - UxG) / dx[0] + (1 / ROC)*(PF - PG) / dx[0]);
        const double WWyC = TauC*(UyC*(VectOld[bi](i,jb+1,k,2) - VectOld[bi](i,jb,k,2)) / dx[1] + UxC*(UyF - UyG) / dx[0] + (1 / ROC)*(VectOld[bi](i,jb+1,k,3) - VectOld[bi](i,jb,k,3)) / dx[1]);
        const double PyxC = PyxNSC + ROC*UyC*WWxC;
        const double PyyC = PyyNSC + ROC*UyC*WWyC + RGC;

        const double T0 = VectOld[bi](i,jb,k,3)   / (VectOld[bi](i,jb,k,0)*RGas);
        const double T3 = VectOld[bi](i,jb+1,k,3) / (VectOld[bi](i,jb+1,k,0)*RGas);
        const double eps0 = VectOld[bi](i,jb,k,3)   / (VectOld[bi](i,jb,k,0)*(gamma - 1.));
        const double eps3 = VectOld[bi](i,jb+1,k,3) / (VectOld[bi](i,jb+1,k,0)*(gamma - 1.));
        const double epsC = PC / (ROC*(gamma - 1.));
        const double epsF = PF / (ROF*(gamma - 1.));
        const double epsG = PG / (ROG*(gamma - 1.));
        const double HC = UxC*UxC/2. + UyC*UyC/2. + gamma*epsC;
        const double qyNSC = -kapC*(T3 - T0) / dx[1];
        const double qyC = qyNSC - TauC*ROC*UyC*(UyC*(eps3 - eps0) / dx[1] + UxC*(epsF - epsG) / dx[0]
            + PC*(UyC*(1./VectOld[bi](i,jb+1,k,0) - 1./VectOld[bi](i,jb,k,0)) / dx[1] + UxC*(1./ROF - 1./ROG) / dx[0]));

        yflux[bi](i,j,k,0) = JmyC;
        yflux[bi](i,j,k,1) = JmyC*UxC - PyxC;
        yflux[bi](i,j,k,2) = JmyC*UyC + PC - PyyC;
        yflux[bi](i,j,k,3) = JmyC*HC + qyC - PyyC*UyC - PyxC*UxC;
    });

#if (AMREX_SPACEDIM == 3)
    fluxes[2].setVal(0.0);
#endif

    auto const& fx = fluxes[0].const_arrays();
    auto const& fy = fluxes[1].const_arrays();
    amrex::ParallelFor(S_new, [=] AMREX_GPU_DEVICE (int bi, int i, int j, int k)
    {
        const double rho = VectOld[bi](i,j,k,0)
            - dt*(fx[bi](i+1,j,k,0) - fx[bi](i,j,k,0))/dx[0]
            - dt*(fy[bi](i,j+1,k,0) - fy[bi](i,j,k,0))/dx[1];

        const double momx = VectOld[bi](i,j,k,0)*VectOld[bi](i,j,k,1)
            - dt*(fx[bi](i+1,j,k,1) - fx[bi](i,j,k,1))/dx[0]
            - dt*(fy[bi](i,j+1,k,1) - fy[bi](i,j,k,1))/dx[1];

        const double momy = VectOld[bi](i,j,k,0)*VectOld[bi](i,j,k,2)
            - dt*(fx[bi](i+1,j,k,2) - fx[bi](i,j,k,2))/dx[0]
            - dt*(fy[bi](i,j+1,k,2) - fy[bi](i,j,k,2))/dx[1];

        const double E = VectOld[bi](i,j,k,0)*(VectOld[bi](i,j,k,1)*VectOld[bi](i,j,k,1) + VectOld[bi](i,j,k,2)*VectOld[bi](i,j,k,2))/2.
            + VectOld[bi](i,j,k,3)/(gamma - 1.);
        const double EN = E
            - dt*(fx[bi](i+1,j,k,3) - fx[bi](i,j,k,3))/dx[0]
            - dt*(fy[bi](i,j+1,k,3) - fy[bi](i,j,k,3))/dx[1];

        VectNew[bi](i,j,k,0) = rho;
        VectNew[bi](i,j,k,1) = momx / rho;
        VectNew[bi](i,j,k,2) = momy / rho;
        VectNew[bi](i,j,k,3) = (gamma - 1.)*(EN - 0.5*rho*(VectNew[bi](i,j,k,1)*VectNew[bi](i,j,k,1) + VectNew[bi](i,j,k,2)*VectNew[bi](i,j,k,2)));
        VectNew[bi](i,j,k,4) = VectOld[bi](i,j,k,4);

        //solve vorticity (curl)
        VectNew[bi](i,j,k,5) = 0.5*(VectOld[bi](i+1,j,k,2) - VectOld[bi](i-1,j,k,2)) / dx[0] - 0.5*(VectOld[bi](i,j+1,k,1) - VectOld[bi](i,j-1,k,1)) / dx[1];

        // solve magGradRho
        double GradRhoX = 0.5*(VectOld[bi](i,j+1,k,0) - VectOld[bi](i,j-1,k,0)) / dx[0];
        double GradRhoY = 0.5*(VectOld[bi](i+1,j,k,0) - VectOld[bi](i-1,j,k,0)) / dx[1];
        VectNew[bi](i,j,k,6) = sqrt(pow(GradRhoX,2) + pow(GradRhoY,2));

        //limiter
        if (VectNew[bi](i,j,k,3) <= 0)
        {
            if (pressureLimiter)
            {
                 amrex::Print() << "Warning! Pressure less then 0." << "\n";
                 VectNew[bi](i,j,k,3) = 0.01;
                 VectNew[bi](i,j,k,3) = (sqrt(pow(VectNew[bi](i+1,j,k,3),2.)  + pow(VectNew[bi](i-1,j,k,3),2.) )
                                       + sqrt(pow(VectNew[bi](i,j+1,k,3),2.)  + pow(VectNew[bi](i,j-1,k,3),2.) )
                                       + sqrt(pow(VectNew[bi](i-1,j+1,k,3),2.)+ pow(VectNew[bi](i+1,j-1,k,3),2.))
                                       + sqrt(pow(VectNew[bi](i+1,j+1,k,3),2.)+ pow(VectNew[bi](i-1,j-1,k,3),2.))) / 4;

                 VectNew[bi](i,j,k,0) = ((sqrt(pow(VectNew[bi](i+1,j,k,0),2.)  + pow(VectNew[bi](i-1,j,k,0),2.) )
                                       + sqrt(pow(VectNew[bi](i,j+1,k,0),2.)  + pow(VectNew[bi](i,j-1,k,0),2.) )
                                       + sqrt(pow(VectNew[bi](i-1,j+1,k,0),2.)+ pow(VectNew[bi](i+1,j-1,k,0),2.))
                                       + sqrt(pow(VectNew[bi](i+1,j+1,k,0),2.)+ pow(VectNew[bi](i-1,j-1,k,0),2.))) / 4
                                        + (sqrt(pow(VectNew[bi](i-2,j,k,0),2.) + pow(VectNew[bi](i+2,j,k,0),2.)))/2
                                        + (sqrt(pow(VectNew[bi](i,j-2,k,0),2.) + pow(VectNew[bi](i,j+2,k,0),2.)))/2) /3;
            }
        }
    });

    if (Level() < parent->finestLevel()) {
        auto& fine_level = getLevel(Level()+1);
        if (fine_level.flux_reg) {
            for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
                Real area = 1.0;
                for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                    if (d != dir) { area *= dx[d]; }
                }
                fine_level.flux_reg->CrseInit(fluxes[dir], dir, 0, 0, nflux, -dt*area);
            }
        }
    }

    if (flux_reg) {
        for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
            Real area = 1.0;
            for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                if (d != dir) { area *= dx[d]; }
            }
            flux_reg->FineAdd(fluxes[dir], dir, 0, 0, nflux, dt*area);
        }
    }

    Real maxval = S_new.max(0);
    Real minval = S_new.min(0);
    amrex::Print() << "min/max rho = " << minval << "/" << maxval;
    maxval = S_new.max(4);
    minval = S_new.min(4);
    amrex::Print() << "  min/max Sc number = " << minval << "/" << maxval << "\n";

    return dt;
}
