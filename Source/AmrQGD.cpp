#include "AmrQGD.H"

#include <AMReX_ParmParse.H>
#include <AMReX_Reduce.H>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

using namespace amrex;

constexpr int AmrQGD::URHO;
constexpr int AmrQGD::UMX;
constexpr int AmrQGD::UMY;
constexpr int AmrQGD::UENG;
constexpr int AmrQGD::USC;
constexpr int AmrQGD::UCURL;
constexpr int AmrQGD::UMAGGRADRHO;
constexpr int AmrQGD::n_cons;
constexpr int AmrQGD::n_diag;
constexpr int AmrQGD::QRHO;
constexpr int AmrQGD::QUX;
constexpr int AmrQGD::QUY;
constexpr int AmrQGD::QP;
constexpr int AmrQGD::QT;
constexpr int AmrQGD::QCS;
constexpr int AmrQGD::n_prim;
constexpr int AmrQGD::ncomp;
constexpr int AmrQGD::nghost;
int  AmrQGD::verbose = 0;
Real AmrQGD::cfl = 0.2;
Real AmrQGD::deltaT0 = 0.1;
int AmrQGD::refcond = 0;
Real AmrQGD::refdengrad = 1.2;

Real AmrQGD::gamma = 1.4;
Real AmrQGD::RGas = 287;
Real AmrQGD::PrGas = 0.7;
Real AmrQGD::mutGas = 0.0;

Real AmrQGD::alphaQgd = 0.3;
Real AmrQGD::ScQgd = 1.0;
Real AmrQGD::PrQgd = 0.7;
bool AmrQGD::varScQgd = false;
bool AmrQGD::pressureLimiter = true;
Real AmrQGD::gradVal = 30;

AmrQGD::AmrQGD (Amr& amr, int lev, const Geometry& gm,
                            const BoxArray& ba, const DistributionMapping& dm,
                            Real time)
    : AmrLevel(amr,lev,gm,ba,dm,time)
{
    if (lev > 0) {
        flux_reg = std::make_unique<FluxRegister>(ba, dm, amr.refRatio(lev-1), lev, n_cons);
    }
}

AmrQGD::~AmrQGD () {}

void
AmrQGD::variableSetUp ()
{
    read_params();

    desc_lst.addDescriptor(State_Type, IndexType::TheCellType(),
                           StateDescriptor::Point, nghost, ncomp,
                           &cell_quartic_interp);


    int lo_bc[BL_SPACEDIM] = {AMREX_D_DECL(BCType::foextrap,
                                           BCType::foextrap,
                                           BCType::foextrap) };
    int hi_bc[BL_SPACEDIM] = {AMREX_D_DECL(BCType::foextrap,
                                           BCType::foextrap,
                                           BCType::foextrap) };

    Vector<BCRec> bcs(ncomp, BCRec(lo_bc, hi_bc));

    StateDescriptor::BndryFunc bndryfunc(bcfill);
    bndryfunc.setRunOnGPU(true);

    desc_lst.setComponent(State_Type, 0, {"rho", "rho_ux", "rho_uy", "E", "Sc", "curl", "magGradRho"}, bcs, bndryfunc);

}

void
AmrQGD::variableCleanUp ()
{
    desc_lst.clear();
}

void
AmrQGD::init (AmrLevel &old)
{
    Real dt_new    = parent->dtLevel(Level());
    Real cur_time  = old.get_state_data(State_Type).curTime();
    Real prev_time = old.get_state_data(State_Type).prevTime();
    Real dt_old    = cur_time - prev_time;
    setTimeLevel(cur_time,dt_old,dt_new);

    for (int k = 0; k < NUM_STATE_TYPE; ++k) {
        MultiFab& S_new = get_new_data(k);
        FillPatch(old, S_new, 0, cur_time, k, 0, ncomp);
    }
}

void
AmrQGD::init ()
{
    Real dt        = parent->dtLevel(Level());
    Real cur_time  = getLevel(Level()-1).state[State_Type].curTime();
    Real prev_time = getLevel(Level()-1).state[State_Type].prevTime();
    Real dt_old = (cur_time - prev_time)/(Real)parent->MaxRefRatio(Level()-1);
    setTimeLevel(cur_time,dt_old,dt);

    for (int k = 0; k < NUM_STATE_TYPE; ++k) {
        MultiFab& S_new = get_new_data(k);
        FillCoarsePatch(S_new, 0, cur_time, k, 0, ncomp);
    }
}

Real
AmrQGD::computeMaxWaveSpeed ()
{
    MultiFab const& S_new = get_new_data(State_Type);
    auto const& state_data = S_new.const_arrays();

    ReduceOps<ReduceOpMax> reduce_op;
    ReduceData<Real> reduce_data(reduce_op);
    using ReduceTuple = typename decltype(reduce_data)::Type;

    reduce_op.eval(S_new, IntVect(0), reduce_data, [=] AMREX_GPU_DEVICE (int bi, int i, int j, int k) -> ReduceTuple
    {
        const Real rho = state_data[bi](i,j,k,URHO);
        if (rho <= 0.0) {
            return {0.0};
        }

        const Real ux = state_data[bi](i,j,k,UMX) / rho;
        const Real uy = state_data[bi](i,j,k,UMY) / rho;
        const Real kinetic = 0.5*rho*(ux*ux + uy*uy);
        const Real p = (gamma - 1.0)*(state_data[bi](i,j,k,UENG) - kinetic);
        if (p <= 0.0) {
            return {0.0};
        }

        const Real sound_speed = sqrt(gamma*p / rho);
        const Real velocity_mag = sqrt(ux*ux + uy*uy);
        return {velocity_mag + sound_speed};
    });

    return amrex::get<0>(reduce_data.value());
}

Real
AmrQGD::computeMinTau ()
{
    MultiFab const& S_new = get_new_data(State_Type);
    auto const& state_data = S_new.const_arrays();
    const auto dx = Geom().CellSizeArray();
    const Real hh = sqrt(AMREX_D_TERM(dx[0]*dx[0], + dx[1]*dx[1], + dx[2]*dx[2]));
    const Real invalid_tau = std::numeric_limits<Real>::max();

    ReduceOps<ReduceOpMin> reduce_op;
    ReduceData<Real> reduce_data(reduce_op);
    using ReduceTuple = typename decltype(reduce_data)::Type;

    reduce_op.eval(S_new, IntVect(0), reduce_data, [=] AMREX_GPU_DEVICE (int bi, int i, int j, int k) -> ReduceTuple
    {
        const Real rho = state_data[bi](i,j,k,URHO);
        if (rho <= 0.0) {
            return {invalid_tau};
        }

        const Real ux = state_data[bi](i,j,k,UMX) / rho;
        const Real uy = state_data[bi](i,j,k,UMY) / rho;
        const Real kinetic = 0.5*rho*(ux*ux + uy*uy);
        const Real p = (gamma - 1.0)*(state_data[bi](i,j,k,UENG) - kinetic);
        if (p <= 0.0) {
            return {invalid_tau};
        }

        const Real sound_speed = sqrt(gamma*p / rho);
        if (sound_speed <= 0.0) {
            return {invalid_tau};
        }

        const Real tau = alphaQgd*hh/sound_speed + mutGas/p;
        return {tau > 0.0 ? tau : invalid_tau};
    });

    return amrex::get<0>(reduce_data.value());
}

Real
AmrQGD::computeStableDt (Real& max_u_plus_c, Real& min_tau)
{
    max_u_plus_c = computeMaxWaveSpeed();
    min_tau = computeMinTau();

    const auto dx = Geom().CellSizeArray();
    const Real dx_min = std::min({AMREX_D_DECL(dx[0], dx[1], dx[2])});

    if (min_tau <= 0.0 || min_tau == std::numeric_limits<Real>::max()) {
        amrex::Print() << "Warning! Non-positive or unavailable min_tau on level "
                       << Level() << "; falling back to advective CFL dt." << "\n";
        if (max_u_plus_c <= 0.0) {
            return std::numeric_limits<Real>::max();
        }
        return cfl * dx_min / max_u_plus_c;
    }

    const Real denominator = max_u_plus_c + dx_min/min_tau;
    if (denominator <= 0.0) {
        return std::numeric_limits<Real>::max();
    }

    return cfl * dx_min / denominator;
}

void
AmrQGD::computeInitialDt (int finest_level, int sub_cycle,
                                Vector<int>& n_cycle,
                                const Vector<IntVect>& ref_ratio,
                                Vector<Real>& dt_level, Real stop_time)
{
    if (Level() > 0) { return; } // Level 0 does this for every level.

    Vector<int> nsteps(finest_level + 1, 1); // Total number of steps in one level 0 step.
    for (int ilev = 1; ilev <= finest_level; ++ilev) {
        int cycle_ratio = 1;
        if (sub_cycle) {
            cycle_ratio = (ilev < static_cast<int>(n_cycle.size())) ? n_cycle[ilev] : 0;
            if (cycle_ratio <= 0 && (ilev-1) < static_cast<int>(ref_ratio.size())) {
                cycle_ratio = ref_ratio[ilev-1][0];
            }
        }
        cycle_ratio = std::max(1, cycle_ratio);
        nsteps[ilev] = nsteps[ilev-1] * cycle_ratio;
    }

    Real dt_0 = deltaT0 > 0.0 ? deltaT0 : std::numeric_limits<Real>::max();
    Vector<Real> stable_dt(finest_level + 1, std::numeric_limits<Real>::max());
    Vector<Real> max_u_plus_c(finest_level + 1, 0.0);
    Vector<Real> min_tau(finest_level + 1, std::numeric_limits<Real>::max());

    for (int ilev = 0; ilev <= finest_level; ++ilev) {
         AmrQGD& level_data = getLevel(ilev);
         stable_dt[ilev] = level_data.computeStableDt(max_u_plus_c[ilev], min_tau[ilev]);
         dt_0 = std::min(dt_0, nsteps[ilev] * stable_dt[ilev]);
    }
    // dt_0 will be the time step on level 0 (unless limited by stop_time).

    if (stop_time > 0)
    {
         // Limit dt's by the value of stop_time.
         const Real eps = 0.001 * dt_0;
         const Real cur_time = get_state_data(State_Type).curTime();
         if ((cur_time + dt_0) > (stop_time - eps)) {
             dt_0 = stop_time - cur_time;
         }
    }

    for (int ilev = 0; ilev <= finest_level; ++ilev) {
        dt_level[ilev] = dt_0 / nsteps[ilev];
        amrex::Print() << "dt diagnostics level " << ilev
                       << ": max_u_plus_c=" << max_u_plus_c[ilev]
                       << ", min_tau=" << min_tau[ilev]
                       << ", stable_dt=" << stable_dt[ilev]
                       << ", n_cycle_product=" << nsteps[ilev]
                       << ", dt_level=" << dt_level[ilev]
                       << "\n";
    }
}

void
AmrQGD::computeNewDt (int finest_level, int sub_cycle,
                            Vector<int>& n_cycle,
                            const Vector<IntVect>& ref_ratio,
                            Vector<Real>& dt_min, Vector<Real>& dt_level,
                            Real stop_time, int post_regrid_flag)
{
    // We are at the end of a coarse grid timecycle.
    // Compute the timesteps for the next iteration using the same
    // level reductions and AMR subcycling constraints as the initial dt.
    computeInitialDt(finest_level, sub_cycle, n_cycle, ref_ratio, dt_level, stop_time);

    for (int ilev = 0; ilev <= finest_level; ++ilev) {
        dt_min[ilev] = dt_level[ilev];
    }
}

void
AmrQGD::post_timestep (int iteration)
{
    //
    // Integration cycle on fine level grids is complete
    // do post_timestep stuff here.
    //
    int finest_level = parent->finestLevel();

    if (Level() < parent->finestLevel()) {
        auto& fine_level = getLevel(Level()+1);
        MultiFab& S_fine = fine_level.get_new_data(State_Type);
        MultiFab& S_crse =      this->get_new_data(State_Type);
        Real t = get_state_data(State_Type).curTime();

        IntVect ratio = parent->refRatio(Level());
        AMREX_ASSERT(ratio == 2 || ratio == 3);
        if (ratio == 2)
        {
            // Need to fill one ghost cell for the high-order interpolation below
            // maybe 2 chande to 1 aor ghost cells
            FillPatch(fine_level, S_fine, 2, t, State_Type, 0, ncomp);
        }

        if (fine_level.flux_reg) {
            // Reflux only the conservative components (rho, rho*ux, rho*uy, E).
            // Diagnostic/passive components are not conservative reflux quantities.
            fine_level.flux_reg->Reflux(S_crse, 1.0, 0, URHO, n_cons, Geom());
        }

        FourthOrderInterpFromFineToCoarse(S_crse, 0, ncomp, S_fine, ratio);
    }

    if (level < finest_level) {
        // fillpatcher on level+1 needs to be reset because data on this
        // level have changed.
        getLevel(level+1).resetFillPatcher();
    }

    AmrLevel::post_timestep(iteration);
}

/**
 * Do work after init().
 */
void
AmrQGD::post_init (Real /*stop_time*/)
{
    if (level > 0) {
        return;
    }

    // Restrict initialized state data from fine levels onto the cells they
    // cover on coarser levels.  There is no flux-register correction at
    // initialization time; refluxing is only applied after an advance in
    // post_timestep, before the same fine-to-coarse state restriction.
    const int finest_level = parent->finestLevel();
    for (int k = finest_level - 1; k >= 0; --k) {
        auto& crse_level = getLevel(k);
        auto& fine_level = getLevel(k+1);
        MultiFab& S_crse = crse_level.get_new_data(State_Type);
        MultiFab& S_fine = fine_level.get_new_data(State_Type);
        const IntVect ratio = parent->refRatio(k);

        FourthOrderInterpFromFineToCoarse(S_crse, 0, ncomp, S_fine, ratio);

        // This coarse level has changed, so any FillPatcher that uses it as
        // coarse data for the adjacent fine level must be rebuilt.
        fine_level.resetFillPatcher();
    }
}

void
AmrQGD::errorEst (TagBoxArray& tags, int clearval, int tagval,
                        Real /*time*/, int /*n_error_buf*/, int /*ngrow*/)
{
    const auto problo = Geom().ProbLoArray();
    const auto probhi = Geom().ProbHiArray();
    const auto dx = Geom().CellSizeArray();
    auto const& S_new = get_new_data(State_Type);
    //const char tagval = TagBox::SET;
    auto const& a = tags.arrays();
    auto const& s = S_new.const_arrays();
    amrex::ParallelFor(tags, [&] AMREX_GPU_DEVICE (int bi, int i, int j, int k)
    {
        const auto rho_at = [=] AMREX_GPU_DEVICE (int ii, int jj) noexcept {
            return s[bi](ii,jj,k,URHO);
        };
        const auto ux_at = [=] AMREX_GPU_DEVICE (int ii, int jj) noexcept {
            return s[bi](ii,jj,k,UMX) / s[bi](ii,jj,k,URHO);
        };
        const auto uy_at = [=] AMREX_GPU_DEVICE (int ii, int jj) noexcept {
            return s[bi](ii,jj,k,UMY) / s[bi](ii,jj,k,URHO);
        };

        if (refcond == 0) //grad(Uy)
        {
            if (amrex::Math::abs(uy_at(i,j)-uy_at(i-1,j))/dx[0] > refdengrad or
                amrex::Math::abs(uy_at(i,j)-uy_at(i,j-1))/dx[1] > refdengrad or
                amrex::Math::abs(uy_at(i,j)-uy_at(i+1,j))/dx[0] > refdengrad or
                amrex::Math::abs(uy_at(i,j)-uy_at(i,j+1))/dx[1] > refdengrad)
//             if (sqrt(uy_at(i,j)*uy_at(i,j) + ux_at(i,j)*ux_at(i,j)) > 2.3)
            {
                a[bi](i,j,k) = tagval;
            } else {
                a[bi](i,j,k) = clearval;
            }
        }
        else if (refcond == 1) //grad(rho)
        {
            if (amrex::Math::abs(rho_at(i,j)-rho_at(i-1,j))/dx[0] > refdengrad or
                amrex::Math::abs(rho_at(i,j)-rho_at(i,j-1))/dx[1] > refdengrad or
                amrex::Math::abs(rho_at(i,j)-rho_at(i+1,j))/dx[0] > refdengrad or
                amrex::Math::abs(rho_at(i,j)-rho_at(i,j+1))/dx[1] > refdengrad)
            {
                a[bi](i,j,k) = tagval;
            } else {
                a[bi](i,j,k) = clearval;
            }
        }
        else if (refcond == 2) //localRe
        {
            if (mutGas > 0)
            {
                if ((sqrt( uy_at(i,j)*uy_at(i,j)+ux_at(i,j)*ux_at(i,j))*dx[0]/mutGas) > refdengrad)
                {
                    a[bi](i,j,k) = tagval;
                } else {
                    a[bi](i,j,k) = clearval;
                }
             }
             else
             {
                if ((sqrt( uy_at(i,j)*uy_at(i,j)+ux_at(i,j)*ux_at(i,j))) > refdengrad)
                {
                    a[bi](i,j,k) = tagval;
                } else {
                    a[bi](i,j,k) = clearval;
                }
             }
        }
        else if (refcond == 3) //
        {
            amrex::Real ax = std::abs(rho_at(i+1,j) - rho_at(i,j));
            amrex::Real ay = std::abs(rho_at(i,j+1) - rho_at(i,j));
            ax = amrex::max(ax,std::abs(rho_at(i,j) - rho_at(i-1,j)));
            ay = amrex::max(ay,std::abs(rho_at(i,j) - rho_at(i,j-1)));

            if (amrex::max(ax,ay) >= refdengrad)
            {
                a[bi](i,j,k) = tagval;
            }
            else
            {
                a[bi](i,j,k) = clearval;
            }
        }
        else if (refcond == 4) //vorticity
        {
            amrex::Real gradRhoX  = 0.5*(rho_at(i+1,j) - rho_at(i-1,j)) / dx[0];
            amrex::Real gradRhoY  = 0.5*(rho_at(i,j+1) - rho_at(i,j-1)) / dx[1];
            amrex::Real gradRho   = sqrt(pow(gradRhoX,2) + pow(gradRhoY,2));
            amrex::Real vorticity = std::abs(0.5*(uy_at(i+1,j) - uy_at(i-1,j)) / dx[0] + 0.5*(ux_at(i,j+1) - ux_at(i,j-1)) / dx[1]);

            if ((gradRho >= refdengrad) || (vorticity >= 1000*refdengrad))
            {
                a[bi](i,j,k) = tagval;
            }
            else
            {
                a[bi](i,j,k) = clearval;
            }
        }
    });
}

void
AmrQGD::read_params ()
{
    ParmParse pp("qgdSolver");
    pp.query("v", verbose); // Could use this to control verbosity during the run
    pp.query("cfl", cfl);
    pp.query("deltaT0", deltaT0);
    pp.query("refine_dengrad", refdengrad);
    pp.query("refine_condition", refcond);
    if (refcond >= 5)
    {
         amrex::Print() << "Refinement condtion does not exist!" << "\n";
         refcond = 0;
    }


    ParmParse pp1("gasProperties");
    pp1.query("gamma", gamma);
    pp1.query("R", RGas);
    pp1.query("Pr", PrGas);
    pp1.query("mut", mutGas);

    ParmParse pp2("qgd");
    pp2.query("alphaQgd", alphaQgd);
    pp2.query("ScQgd", ScQgd);
    pp2.query("PrQgd", PrQgd);
    pp2.query("varScQgd", varScQgd);
    pp2.query("dengradVal", gradVal);
    pp2.query("pressure_limiter", pressureLimiter);

}
