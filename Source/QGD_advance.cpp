#include "AmrQGD.H"

#include <AMReX_Array.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_Reduce.H>

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
    MultiFab& S_old = state[0].oldData();
    FillPatcherFill(S_old, 0, ncomp, nghost, time, State_Type, 0);

    auto const& ConsOld = S_old.arrays();

    if (varScQgd)
    {
        amrex::ParallelFor(S_old, [=] AMREX_GPU_DEVICE (int bi, int i, int j, int k)
        {
            if (amrex::Math::abs(ConsOld[bi](i,j,k,URHO) - ConsOld[bi](i-1,j,k,URHO)) / dx[0] >= gradVal or
                amrex::Math::abs(ConsOld[bi](i,j,k,URHO) - ConsOld[bi](i,j-1,k,URHO)) / dx[1] >= gradVal or
                amrex::Math::abs(ConsOld[bi](i,j,k,URHO) - ConsOld[bi](i+1,j,k,URHO)) / dx[0] >= gradVal or
                amrex::Math::abs(ConsOld[bi](i,j,k,URHO) - ConsOld[bi](i,j+1,k,URHO)) / dx[1] >= gradVal)
            {
                ConsOld[bi](i,j,k,USC) = 4.0;
            }
            else
            {
                ConsOld[bi](i,j,k,USC) = ScQgd;
            }
        });
    }

    MultiFab Prim_old(S_old.boxArray(), S_old.DistributionMap(), n_prim, nghost);
    for (MFIter mfi(Prim_old); mfi.isValid(); ++mfi) {
        const Box bx = amrex::grow(mfi.validbox(), nghost);
        auto const& u = S_old.const_array(mfi);
        auto const& q = Prim_old.array(mfi);
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            const Real rho = u(i,j,k,URHO);
            const Real ux = u(i,j,k,UMX) / rho;
            const Real uy = u(i,j,k,UMY) / rho;
            const Real kinetic = 0.5*rho*(ux*ux + uy*uy);
            const Real p = (gamma - 1.)*(u(i,j,k,UENG) - kinetic);
            q(i,j,k,QRHO) = rho;
            q(i,j,k,QUX) = ux;
            q(i,j,k,QUY) = uy;
            q(i,j,k,QP) = p;
            q(i,j,k,QT) = p / (rho*RGas);
            q(i,j,k,QCS) = sqrt(gamma*p / rho);
        });
    }
    auto const& PrimOld = Prim_old.const_arrays();

    static constexpr int nflux = n_cons; // rho, rho*u, rho*v, E
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
        const double ScQGD = ConsOld[bi](il,j,k,USC);

        const double ROA = 0.5*(PrimOld[bi](il,j,k,QRHO) + PrimOld[bi](il+1,j,k,QRHO));
        const double UxA = 0.5*(PrimOld[bi](il,j,k,QUX) + PrimOld[bi](il+1,j,k,QUX));
        const double UyA = 0.5*(PrimOld[bi](il,j,k,QUY) + PrimOld[bi](il+1,j,k,QUY));
        const double PA  = 0.5*(PrimOld[bi](il,j,k,QP) + PrimOld[bi](il+1,j,k,QP));

        const double ROE = 0.25*(PrimOld[bi](il,j,k,QRHO) + PrimOld[bi](il+1,j,k,QRHO) + PrimOld[bi](il,j-1,k,QRHO) + PrimOld[bi](il+1,j-1,k,QRHO));
        const double ROF = 0.25*(PrimOld[bi](il,j,k,QRHO) + PrimOld[bi](il+1,j,k,QRHO) + PrimOld[bi](il,j+1,k,QRHO) + PrimOld[bi](il+1,j+1,k,QRHO));

        const double UxE = 0.25*(PrimOld[bi](il,j,k,QUX) + PrimOld[bi](il+1,j,k,QUX) + PrimOld[bi](il,j-1,k,QUX) + PrimOld[bi](il+1,j-1,k,QUX));
        const double UxF = 0.25*(PrimOld[bi](il,j,k,QUX) + PrimOld[bi](il+1,j,k,QUX) + PrimOld[bi](il,j+1,k,QUX) + PrimOld[bi](il+1,j+1,k,QUX));

        const double UyE = 0.25*(PrimOld[bi](il,j,k,QUY) + PrimOld[bi](il+1,j,k,QUY) + PrimOld[bi](il,j-1,k,QUY) + PrimOld[bi](il+1,j-1,k,QUY));
        const double UyF = 0.25*(PrimOld[bi](il,j,k,QUY) + PrimOld[bi](il+1,j,k,QUY) + PrimOld[bi](il,j+1,k,QUY) + PrimOld[bi](il+1,j+1,k,QUY));

        const double PE = 0.25*(PrimOld[bi](il,j,k,QP) + PrimOld[bi](il+1,j,k,QP) + PrimOld[bi](il,j-1,k,QP) + PrimOld[bi](il+1,j-1,k,QP));
        const double PF = 0.25*(PrimOld[bi](il,j,k,QP) + PrimOld[bi](il+1,j,k,QP) + PrimOld[bi](il,j+1,k,QP) + PrimOld[bi](il+1,j+1,k,QP));

        const double CsA = 0.5*(PrimOld[bi](il,j,k,QCS) + PrimOld[bi](il+1,j,k,QCS));
        const double hh = sqrt(dx[0]*dx[0] + dx[1]*dx[1]);
        const double TauA = alphaQgd*hh/CsA + mu_T / PA;
        const double muA = mu_T + TauA*PA*ScQGD;
        const double kapA = (mu_T / PrGas + PA*TauA*ScQGD / PrQgd)*gamma*RGas / (gamma - 1.);

        const double WxA = (TauA / ROA)*((ROF*UyF*UxF - ROE*UyE*UxE) / dx[1]
                                + (PrimOld[bi](il+1,j,k,QRHO)*PrimOld[bi](il+1,j,k,QUX)*PrimOld[bi](il+1,j,k,QUX) - PrimOld[bi](il,j,k,QRHO)*PrimOld[bi](il,j,k,QUX)*PrimOld[bi](il,j,k,QUX)) / dx[0]
                                + (PrimOld[bi](il+1,j,k,QP) - PrimOld[bi](il,j,k,QP)) / dx[0]);
        const double JmxA = ROA*(UxA - WxA);

        const double divuA = (PrimOld[bi](il+1,j,k,QUX) - PrimOld[bi](il,j,k,QUX)) / dx[0] + (UyF - UyE) / dx[1];
        const double PxxNSA = 2.*muA*(PrimOld[bi](il+1,j,k,QUX) - PrimOld[bi](il,j,k,QUX)) / dx[0] - (2./3.)*muA*divuA;
        const double PxyNSA = muA*((PrimOld[bi](il+1,j,k,QUY) - PrimOld[bi](il,j,k,QUY)) / dx[0] + (UxF - UxE) / dx[1]);
        const double RGA = TauA*(UxA*(PrimOld[bi](il+1,j,k,QP) - PrimOld[bi](il,j,k,QP)) / dx[0] + UyA*(PF - PE) / dx[1] + gamma*PA*divuA);

        const double WWxA = TauA*(UxA*(PrimOld[bi](il+1,j,k,QUX) - PrimOld[bi](il,j,k,QUX)) / dx[0] + UyA*(UxF - UxE) / dx[1] + (1 / ROA)*(PrimOld[bi](il+1,j,k,QP) - PrimOld[bi](il,j,k,QP)) / dx[0]);
        const double WWyA = TauA*(UxA*(PrimOld[bi](il+1,j,k,QUY) - PrimOld[bi](il,j,k,QUY)) / dx[0] + UyA*(UyF - UyE) / dx[1] + (1 / ROA)*(PF - PE) / dx[1]);
        const double PxxA = PxxNSA + ROA*UxA*WWxA + RGA;
        const double PxyA = PxyNSA + ROA*UxA*WWyA;

        const double T0 = PrimOld[bi](il,j,k,QT);
        const double T1 = PrimOld[bi](il+1,j,k,QT);
        const double eps0 = PrimOld[bi](il,j,k,QP)   / (PrimOld[bi](il,j,k,QRHO)*(gamma - 1.));
        const double eps1 = PrimOld[bi](il+1,j,k,QP) / (PrimOld[bi](il+1,j,k,QRHO)*(gamma - 1.));
        const double epsA = PA / (ROA*(gamma - 1.));
        const double epsE = PE / (ROE*(gamma - 1.));
        const double epsF = PF / (ROF*(gamma - 1.));
        const double HA = UxA*UxA/2. + UyA*UyA/2. + gamma*epsA;
        const double qxNSA = -kapA*(T1 - T0) / dx[0];
        const double qxA = qxNSA - TauA*ROA*UxA*(UxA*(eps1 - eps0) / dx[0] + UyA*(epsF - epsE) / dx[1]
            + PA*(UxA*(1./PrimOld[bi](il+1,j,k,QRHO) - 1./PrimOld[bi](il,j,k,QRHO)) / dx[0] + UyA*(1./ROF - 1./ROE) / dx[1]));

        xflux[bi](i,j,k,URHO) = JmxA;
        xflux[bi](i,j,k,UMX) = JmxA*UxA + PA - PxxA;
        xflux[bi](i,j,k,UMY) = JmxA*UyA - PxyA;
        xflux[bi](i,j,k,UENG) = JmxA*HA + qxA - PxxA*UxA - PxyA*UyA;
    });

    auto const& yflux = fluxes[1].arrays();
    amrex::ParallelFor(fluxes[1], [=] AMREX_GPU_DEVICE (int bi, int i, int j, int k)
    {
        const int jb = j - 1;
        const double ScQGD = ConsOld[bi](i,jb,k,USC);

        const double ROC = 0.5*(PrimOld[bi](i,jb,k,QRHO) + PrimOld[bi](i,jb+1,k,QRHO));
        const double UxC = 0.5*(PrimOld[bi](i,jb,k,QUX) + PrimOld[bi](i,jb+1,k,QUX));
        const double UyC = 0.5*(PrimOld[bi](i,jb,k,QUY) + PrimOld[bi](i,jb+1,k,QUY));
        const double PC  = 0.5*(PrimOld[bi](i,jb,k,QP) + PrimOld[bi](i,jb+1,k,QP));

        const double ROF = 0.25*(PrimOld[bi](i,jb,k,QRHO) + PrimOld[bi](i+1,jb,k,QRHO) + PrimOld[bi](i,jb+1,k,QRHO) + PrimOld[bi](i+1,jb+1,k,QRHO));
        const double ROG = 0.25*(PrimOld[bi](i,jb,k,QRHO) + PrimOld[bi](i-1,jb,k,QRHO) + PrimOld[bi](i,jb+1,k,QRHO) + PrimOld[bi](i-1,jb+1,k,QRHO));

        const double UxF = 0.25*(PrimOld[bi](i,jb,k,QUX) + PrimOld[bi](i+1,jb,k,QUX) + PrimOld[bi](i,jb+1,k,QUX) + PrimOld[bi](i+1,jb+1,k,QUX));
        const double UxG = 0.25*(PrimOld[bi](i,jb,k,QUX) + PrimOld[bi](i-1,jb,k,QUX) + PrimOld[bi](i,jb+1,k,QUX) + PrimOld[bi](i-1,jb+1,k,QUX));

        const double UyF = 0.25*(PrimOld[bi](i,jb,k,QUY) + PrimOld[bi](i+1,jb,k,QUY) + PrimOld[bi](i,jb+1,k,QUY) + PrimOld[bi](i+1,jb+1,k,QUY));
        const double UyG = 0.25*(PrimOld[bi](i,jb,k,QUY) + PrimOld[bi](i-1,jb,k,QUY) + PrimOld[bi](i,jb+1,k,QUY) + PrimOld[bi](i-1,jb+1,k,QUY));

        const double PF = 0.25*(PrimOld[bi](i,jb,k,QP) + PrimOld[bi](i+1,jb,k,QP) + PrimOld[bi](i,jb+1,k,QP) + PrimOld[bi](i+1,jb+1,k,QP));
        const double PG = 0.25*(PrimOld[bi](i,jb,k,QP) + PrimOld[bi](i-1,jb,k,QP) + PrimOld[bi](i,jb+1,k,QP) + PrimOld[bi](i-1,jb+1,k,QP));

        const double CsC = 0.5*(PrimOld[bi](i,jb,k,QCS) + PrimOld[bi](i,jb+1,k,QCS));
        const double hh = sqrt(dx[0]*dx[0] + dx[1]*dx[1]);
        const double TauC = alphaQgd*hh/CsC + mu_T / PC;
        const double muC = mu_T + TauC*PC*ScQGD;
        const double kapC = (mu_T / PrGas + PC*TauC*ScQGD / PrQgd)*gamma*RGas / (gamma - 1.);

        const double WyC = (TauC / ROC)*((ROF*UyF*UxF - ROG*UyG*UxG) / dx[0]
                                + (PrimOld[bi](i,jb+1,k,QRHO)*PrimOld[bi](i,jb+1,k,QUY)*PrimOld[bi](i,jb+1,k,QUY) - PrimOld[bi](i,jb,k,QRHO)*PrimOld[bi](i,jb,k,QUY)*PrimOld[bi](i,jb,k,QUY)) / dx[1]
                                + (PrimOld[bi](i,jb+1,k,QP) - PrimOld[bi](i,jb,k,QP)) / dx[1]);
        const double JmyC = ROC*(UyC - WyC);

        const double divuC = (PrimOld[bi](i,jb+1,k,QUY) - PrimOld[bi](i,jb,k,QUY)) / dx[1] + (UxF - UxG) / dx[0];
        const double PyxNSC = muC*((PrimOld[bi](i,jb+1,k,QUX) - PrimOld[bi](i,jb,k,QUX)) / dx[1] + (UyF - UyG) / dx[0]);
        const double PyyNSC = 2.*muC*(PrimOld[bi](i,jb+1,k,QUY) - PrimOld[bi](i,jb,k,QUY)) / dx[1] - (2./3.)*muC*divuC;
        const double RGC = TauC*(UyC*(PrimOld[bi](i,jb+1,k,QP) - PrimOld[bi](i,jb,k,QP)) / dx[1] + UxC*(PF - PG) / dx[0] + gamma*PC*divuC);

        const double WWxC = TauC*(UyC*(PrimOld[bi](i,jb+1,k,QUX) - PrimOld[bi](i,jb,k,QUX)) / dx[1] + UxC*(UxF - UxG) / dx[0] + (1 / ROC)*(PF - PG) / dx[0]);
        const double WWyC = TauC*(UyC*(PrimOld[bi](i,jb+1,k,QUY) - PrimOld[bi](i,jb,k,QUY)) / dx[1] + UxC*(UyF - UyG) / dx[0] + (1 / ROC)*(PrimOld[bi](i,jb+1,k,QP) - PrimOld[bi](i,jb,k,QP)) / dx[1]);
        const double PyxC = PyxNSC + ROC*UyC*WWxC;
        const double PyyC = PyyNSC + ROC*UyC*WWyC + RGC;

        const double T0 = PrimOld[bi](i,jb,k,QT);
        const double T3 = PrimOld[bi](i,jb+1,k,QT);
        const double eps0 = PrimOld[bi](i,jb,k,QP)   / (PrimOld[bi](i,jb,k,QRHO)*(gamma - 1.));
        const double eps3 = PrimOld[bi](i,jb+1,k,QP) / (PrimOld[bi](i,jb+1,k,QRHO)*(gamma - 1.));
        const double epsC = PC / (ROC*(gamma - 1.));
        const double epsF = PF / (ROF*(gamma - 1.));
        const double epsG = PG / (ROG*(gamma - 1.));
        const double HC = UxC*UxC/2. + UyC*UyC/2. + gamma*epsC;
        const double qyNSC = -kapC*(T3 - T0) / dx[1];
        const double qyC = qyNSC - TauC*ROC*UyC*(UyC*(eps3 - eps0) / dx[1] + UxC*(epsF - epsG) / dx[0]
            + PC*(UyC*(1./PrimOld[bi](i,jb+1,k,QRHO) - 1./PrimOld[bi](i,jb,k,QRHO)) / dx[1] + UxC*(1./ROF - 1./ROG) / dx[0]));

        yflux[bi](i,j,k,URHO) = JmyC;
        yflux[bi](i,j,k,UMX) = JmyC*UxC - PyxC;
        yflux[bi](i,j,k,UMY) = JmyC*UyC + PC - PyyC;
        yflux[bi](i,j,k,UENG) = JmyC*HC + qyC - PyyC*UyC - PyxC*UxC;
    });

#if (AMREX_SPACEDIM == 3)
    fluxes[2].setVal(0.0);
#endif

    auto const& fx = fluxes[0].const_arrays();
    auto const& fy = fluxes[1].const_arrays();
    auto const& StateUpdate = S_new.arrays();
    amrex::ParallelFor(S_new, [=] AMREX_GPU_DEVICE (int bi, int i, int j, int k)
    {
        const double rho = ConsOld[bi](i,j,k,URHO)
            - dt*(fx[bi](i+1,j,k,URHO) - fx[bi](i,j,k,URHO))/dx[0]
            - dt*(fy[bi](i,j+1,k,URHO) - fy[bi](i,j,k,URHO))/dx[1];

        const double momx = ConsOld[bi](i,j,k,UMX)
            - dt*(fx[bi](i+1,j,k,UMX) - fx[bi](i,j,k,UMX))/dx[0]
            - dt*(fy[bi](i,j+1,k,UMX) - fy[bi](i,j,k,UMX))/dx[1];

        const double momy = ConsOld[bi](i,j,k,UMY)
            - dt*(fx[bi](i+1,j,k,UMY) - fx[bi](i,j,k,UMY))/dx[0]
            - dt*(fy[bi](i,j+1,k,UMY) - fy[bi](i,j,k,UMY))/dx[1];

        const double EN = ConsOld[bi](i,j,k,UENG)
            - dt*(fx[bi](i+1,j,k,UENG) - fx[bi](i,j,k,UENG))/dx[0]
            - dt*(fy[bi](i,j+1,k,UENG) - fy[bi](i,j,k,UENG))/dx[1];

        StateUpdate[bi](i,j,k,URHO) = rho;
        StateUpdate[bi](i,j,k,UMX) = momx;
        StateUpdate[bi](i,j,k,UMY) = momy;
        StateUpdate[bi](i,j,k,UENG) = EN;
        StateUpdate[bi](i,j,k,USC) = ConsOld[bi](i,j,k,USC);

        // solve vorticity (curl) from the primitive velocities
        StateUpdate[bi](i,j,k,UCURL) = 0.5*(PrimOld[bi](i+1,j,k,QUY) - PrimOld[bi](i-1,j,k,QUY)) / dx[0]
                                     - 0.5*(PrimOld[bi](i,j+1,k,QUX) - PrimOld[bi](i,j-1,k,QUX)) / dx[1];

        // solve magGradRho
        double GradRhoX = 0.5*(PrimOld[bi](i,j+1,k,QRHO) - PrimOld[bi](i,j-1,k,QRHO)) / dx[0];
        double GradRhoY = 0.5*(PrimOld[bi](i+1,j,k,QRHO) - PrimOld[bi](i-1,j,k,QRHO)) / dx[1];
        StateUpdate[bi](i,j,k,UMAGGRADRHO) = sqrt(pow(GradRhoX,2) + pow(GradRhoY,2));
    });

    if (pressureLimiter) {
        FillPatcherFill(S_new, 0, ncomp, nghost, time + dt, State_Type, 0);

        MultiFab stable_state(S_new.boxArray(), S_new.DistributionMap(), ncomp, nghost);
        MultiFab::Copy(stable_state, S_new, 0, 0, ncomp, nghost);

        auto const& StableState = stable_state.const_arrays();
        constexpr Real pressure_floor = 0.01;

        ReduceOps<ReduceOpSum> reduce_op;
        ReduceData<Long> reduce_data(reduce_op);
        using ReduceTuple = typename decltype(reduce_data)::Type;
        reduce_op.eval(stable_state, reduce_data, [=] AMREX_GPU_DEVICE (int bi, int i, int j, int k) -> ReduceTuple
        {
            const Real rho = StableState[bi](i,j,k,URHO);
            const Real ux = StableState[bi](i,j,k,UMX) / rho;
            const Real uy = StableState[bi](i,j,k,UMY) / rho;
            const Real kinetic = 0.5*rho*(ux*ux + uy*uy);
            const Real p = (gamma - 1.)*(StableState[bi](i,j,k,UENG) - kinetic);
            return {p <= 0.0 ? Long(1) : Long(0)};
        });
        const Long negative_pressure_count = amrex::get<0>(reduce_data.value());

        if (negative_pressure_count > 0) {
            amrex::Print() << "Warning! Pressure limiter applied to "
                           << negative_pressure_count << " cells." << "\n";

            auto const& LimitedState = S_new.arrays();
            amrex::ParallelFor(S_new, [=] AMREX_GPU_DEVICE (int bi, int i, int j, int k)
            {
                const Real rho = StableState[bi](i,j,k,URHO);
                const Real momx = StableState[bi](i,j,k,UMX);
                const Real momy = StableState[bi](i,j,k,UMY);
                const Real ux = momx / rho;
                const Real uy = momy / rho;
                const Real kinetic = 0.5*rho*(ux*ux + uy*uy);
                const Real p = (gamma - 1.)*(StableState[bi](i,j,k,UENG) - kinetic);

                if (p <= 0.0) {
                    LimitedState[bi](i,j,k,UENG) = pressure_floor/(gamma - 1.) + kinetic;
                }
            });
            Gpu::streamSynchronize();
        }
    }

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

    Real maxval = S_new.max(URHO);
    Real minval = S_new.min(URHO);
    amrex::Print() << "min/max rho = " << minval << "/" << maxval;
    maxval = S_new.max(USC);
    minval = S_new.min(USC);
    amrex::Print() << "  min/max Sc number = " << minval << "/" << maxval << "\n";

    return dt;
}
