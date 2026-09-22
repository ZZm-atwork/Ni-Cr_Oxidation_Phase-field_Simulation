/* Ni-Cr oxidation, 2D MPI (X-slab decomposition), optional evolving metal grains.
 * --grain-evolution on: coupled Allen-Cahn eta fields and eta-derived GB transport.
 * --grain-evolution off (default): original fixed-grain equations and Gaussian mask.
 * New grain parameters are independent, provisional model inputs; see MODEL.md.
 * Original oxide chemistry/transport and hard seeds are retained; eta is an opt-in extension.
 * --seed selects a reproducible Voronoi realization; grain/O switches default to on.
 * --nucleation center inserts ONE hard oxide at t=0 and disables later Ksp insertion.
 * --nucleation ksp (default) preserves the original Ksp-triggered insertion path.
 * Defaults: initialization only; legacy concentration BC and legacy Ck retained.
 * Experimental mu boundary and Ck-off are opt-in. See README before using results.
 * Build: mpic++ -O3 -std=c++17 -Wall -Wextra -pedantic NiCr2D_grains.cpp -o nicr2d_mpi
 */
#ifdef NICR_MPI_THREAD_TEST
#include "tests/mpi_threads.h" // Communication-logic tests only; not a real MPI runtime.
#elif defined(NICR_SERIAL_VERIFY)
#include "tests/mpi_serial.h" // Single-process verification only; not an MPI implementation.
#else
#include <mpi.h>
#endif
#include <memory>
#include <climits>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
constexpr double PI=3.14159265358979323846;
constexpr const char* VERSION="mpi-eta-v1";
struct Config {
    int nx=151, ny=101, steps=0, output_every=5000;
    int left_edge_sites=0; // 0: original random Voronoi; otherwise relocate existing sites.
    unsigned long long random_seed=20260909;
    double dx=1.e-8, dy=1.e-8, dt=2e-10;
    double grain_diameter_grid=50., gb_fwhm_grid=4.;
    // Literature-guided provisional numerical peak factors at 1000 C (1273 K).
    // They are mapped onto the diffuse Gaussian GB mask, so they are NOT raw physical D_GB/D_bulk ratios.
    // O: Park-type Q_GB ~= Q_bulk/2 plus a provisional 0.5 nm physical GB width -> ~28.2.
    // Cr: Chen et al. low-C Alloy B delta*D_GB / D_v mapped onto this mask -> ~563.
    double gb_factor_o=28.2, gb_factor_cr=563.;
    double radius_grid=5.5, exclusion_gap_grid=6., nucleation_period=1.e-7;
    double mu_res=364688., surface_c_ref=0.0023;
    std::string bc="concentration", ck="legacy", bounds="stop", out="nicr_grains_output";
    // center: one initial hard seed only; ksp: original automatic seeding.
    std::string nucleation="ksp";
    bool grains=true, oxygen_diffusion=true;
    bool self_test=false;
    // Dynamic grains are opt-in so unchanged commands retain the fixed baseline.
    bool grain_evolution=false;
    double eta_W=1.e4, eta_kappa=0., eta_L=0.2, eta_oxide_penalty=-1.;
    // eta_kappa=0: infer from the old Gaussian mask's integrated width.
    // eta_oxide_penalty=-1: use eta_W. Neither default is a measured Ni-Cr property.
    std::string eta_output="all"; // all: individual eta fields; summary: compact output.
    int max_eta_fields=512;
};
struct Physics {
    static constexpr double Vm=1., W=1e4, ko=1e-9, kc=7.5e-9, kp=1.7e-10, L=0.2;
    static constexpr double bmo=2.21e-16, bmc=3.5436e-20, boo=2.21e-18, boc=3.5436e-21;
    static constexpr double ckO=2e-10, ckCr=2e-10, ksp=1.28e-7;
    static constexpr double bulkO=0., bulkCr=0.1986, layerO=0.0023, layerCr=0.182;
    static constexpr double oxideO=0.56300203, oxideCr=0.35677915;
};
// ----- Evolving-grain free energy (new model, not copied from GB_new2.cpp) -----
// S=sum eta_i^2, P=sum_{i<j} eta_i^2 eta_j^2, H=h(phi), m=1-H.
// f_eta = W/4*(S-m)^2 + W*P + A/2*H*S + K/2*sum |grad eta_i|^2.
// Pure metal: one eta=1; pure oxide: all eta=0. Both have zero added bulk energy.
// Metal-only restriction is the symmetric gamma=1.5 multi-order-parameter model.
// BOTH eta and phi derivatives are included. O/Cr chemical free energies are unchanged.
static double h(double p);
static double hp(double p){return 30.*p*p*(p-1.)*(p-1.);}
static double reference_gb_width(const Config& c){
    return c.gb_fwhm_grid*c.dx*std::sqrt(PI)/(2.*std::sqrt(std::log(2.)));
}
static double eta_K(const Config& c){
    if(c.eta_kappa>0.)return c.eta_kappa;
    const double ell=3.*reference_gb_width(c)/8.;
    return 2.*c.eta_W*ell*ell;
}
static double eta_ell(const Config& c){return std::sqrt(eta_K(c)/(2.*c.eta_W));}
static double eta_A(const Config& c){return c.eta_oxide_penalty<0.?c.eta_W:c.eta_oxide_penalty;}
static double eta_transport_scale(const Config& c){
    // At an equilibrium planar bicrystal, 16*eta1^2*eta2^2 = sech^4(x/(2*ell)).
    // Its integral is 8*ell/3. Match the PREVIOUS excess integrated conductance,
    // not an unqualified peak multiplier when the eta interface width is changed.
    return reference_gb_width(c)/(8.*eta_ell(c)/3.);
}
static double grain_local_energy(double S,double P,double phi,const Config& c){
    const double H=h(phi), d=S-(1.-H);
    return 0.25*c.eta_W*d*d+c.eta_W*P+0.5*eta_A(c)*H*S;
}
static double grain_local_deta(double e,double S,double phi,const Config& c){
    return e*(c.eta_W*(S-(1.-h(phi))+2.*(S-e*e))+eta_A(c)*h(phi));
}
static double grain_local_dphi(double S,double phi,const Config& c){
    return 0.5*hp(phi)*(c.eta_W*(S-(1.-h(phi)))+eta_A(c)*S);
}

// One slab per rank; global indices are retained in the numerical formulas.
// MPI_COMM_SELF keeps the original serial self-tests available in this build.
static void collective_error(MPI_Comm comm,const std::string& message){
    int rank=0;MPI_Comm_rank(comm,&rank);
    int candidate=message.empty()?INT_MAX:rank,failed=INT_MAX;
    MPI_Allreduce(&candidate,&failed,1,MPI_INT,MPI_MIN,comm);
    if(failed==INT_MAX)return;
    int length=rank==failed?int(message.size()):0;
    MPI_Bcast(&length,1,MPI_INT,failed,comm);
    std::string shared=rank==failed?message:std::string(length,' ');
    MPI_Bcast(shared.data(),length,MPI_CHAR,failed,comm);
    throw std::runtime_error(shared);
}
template<class Fn> static void root_action(MPI_Comm comm,Fn fn){
    int rank=0;MPI_Comm_rank(comm,&rank);std::string error;
    if(rank==0)try{fn();}catch(const std::exception& e){error=e.what();}
    collective_error(comm,error);
}
struct Domain {
    MPI_Comm comm;
    int rank=0,size=1,first=1,last=0;
    std::vector<int> counts,displacements;
    Domain(int nx,int ny,MPI_Comm communicator):comm(communicator){
        MPI_Comm_rank(comm,&rank);MPI_Comm_size(comm,&size);
        if(size>nx)throw std::runtime_error("MPI process count must not exceed --nx (one or more X columns per rank).");
        if(static_cast<long long>(nx+2)*(static_cast<long long>(ny)+2)>INT_MAX)
            throw std::runtime_error("Grid exceeds the supported MPI gather count range.");
        int offset=0;
        for(int r=0;r<size;++r){
            const int width=nx/size+(r<nx%size?1:0);
            counts.push_back(width*(ny+2));displacements.push_back(offset*(ny+2));
            if(r==rank){first=offset+1;last=offset+width;}
            offset+=width;
        }
    }
};
struct Field {
    int nx,ny;
    Domain domain;
    std::vector<double> a;
    Field(int x,int y,const Domain& d,double v=0.):nx(x),ny(y),domain(d),
        a(std::size_t(d.last-d.first+3)*(y+2),v){}
    double& operator()(int i,int j){return a[std::size_t(i-domain.first+1)*(ny+2)+j];}
    const double& operator()(int i,int j)const{return a[std::size_t(i-domain.first+1)*(ny+2)+j];}
};
static double h(double p){return p*p*p*(10.-15.*p+6.*p*p);}
static double minimum_image(double dy,double period){return dy-period*std::round(dy/period);}
static void ghosts(Field& a) {
    const auto& d=a.domain;
    // Exchange concentrations/phi first, then chemical potentials in refresh().
    // Both exchanges are needed for the fourth-order concentration operator.
    if(d.size>1){
        const int left=d.rank==0?MPI_PROC_NULL:d.rank-1;
        const int right=d.rank==d.size-1?MPI_PROC_NULL:d.rank+1;
        MPI_Sendrecv(&a(d.first,1),a.ny,MPI_DOUBLE,left,0,
                     &a(d.last+1,1),a.ny,MPI_DOUBLE,right,0,d.comm,MPI_STATUS_IGNORE);
        MPI_Sendrecv(&a(d.last,1),a.ny,MPI_DOUBLE,right,1,
                     &a(d.first-1,1),a.ny,MPI_DOUBLE,left,1,d.comm,MPI_STATUS_IGNORE);
    }
    if(d.first==1)for(int j=1;j<=a.ny;++j)a(0,j)=a(1,j);
    if(d.last==a.nx)for(int j=1;j<=a.ny;++j)a(a.nx+1,j)=a(a.nx,j);
    for(int i=d.first-1;i<=d.last+1;++i){a(i,0)=a(i,a.ny);a(i,a.ny+1)=a(i,1);}
}
static double lap(const Field&a,int i,int j,const Config&c){
    return (a(i+1,j)-2.*a(i,j)+a(i-1,j))/(c.dx*c.dx)
          +(a(i,j+1)-2.*a(i,j)+a(i,j-1))/(c.dy*c.dy);
}
static double fmetal(double o,double cr){
    return (6000000.*o*o+583264.303945031*cr*cr+3200000.*o*cr
            -298432.*o-97045.64*cr-61732.2974223055)/Physics::Vm;
}
static double foxide(double o,double cr){
    return (12000000.*o*o+7000000.*cr*cr-13800000.*o*cr
            -8880000.*o+2680000.*cr+1860990.)/Physics::Vm;
}
static double mu_metal_o(double o,double cr){return (12000000.*o+3200000.*cr-298432.)/Physics::Vm;}
static double mobility(double conc,double p,double gb,bool oxygen,const Config&c){
    // Switch off O transport in BOTH matrix and oxide, including GB transport.
    // Chemical free energies, Cr mobility, hard seeding, and Ck are not changed.
    if(oxygen && !c.oxygen_diffusion)return 0.;
    double hh=h(p), bm=oxygen?Physics::bmo:Physics::bmc, bo=oxygen?Physics::boo:Physics::boc;
    const double gb_factor=oxygen?c.gb_factor_o:c.gb_factor_cr;
    const double active_gb=c.grains?gb:0.;
    return conc*((1.-hh)*bm*(1.+(gb_factor-1.)*active_gb)+hh*bo);
}
struct Seed { double x,y; };
struct Budget {long double nucO=0,nucCr=0,ckO=0,ckCr=0,clipO=0,clipCr=0,bcO=0,bcCr=0;long long nuclei=0,clipped=0;};
class Model {
public:
    Config c;
    Domain domain;
    std::unique_ptr<Model> snapshot; // Full arrays exist only on root, for ordered seeding/output.
    Field o,cr,p,on,crn,pn,muO,muCr,mo,mc,grain,gb;
    Field gb_initial,eta_s2,eta_pairs;
    std::vector<Field> eta,eta_new;
    bool has_eta()const{return c.grain_evolution && c.grains;}

    std::vector<Seed> seeds;
    std::vector<int> areas;
    Budget b;
    long double initialO=0,initialCr=0;
    int center_seed_cells=0;
    Model(Config cfg,MPI_Comm comm=MPI_COMM_SELF):c(std::move(cfg)),domain(c.nx,c.ny,comm),o(c.nx,c.ny,domain),cr(c.nx,c.ny,domain),p(c.nx,c.ny,domain),
      on(c.nx,c.ny,domain),crn(c.nx,c.ny,domain),pn(c.nx,c.ny,domain),muO(c.nx,c.ny,domain),muCr(c.nx,c.ny,domain),
      mo(c.nx,c.ny,domain),mc(c.nx,c.ny,domain),grain(c.nx,c.ny,domain),gb(c.nx,c.ny,domain),
      gb_initial(c.nx,c.ny,domain),eta_s2(c.nx,c.ny,domain),eta_pairs(c.nx,c.ny,domain){
        voronoi();
        gb_initial.a=gb.a;
        for(int i=domain.first;i<=domain.last;++i)for(int j=1;j<=c.ny;++j){
            o(i,j)=Physics::bulkO;cr(i,j)=Physics::bulkCr;p(i,j)=0.;
            // Keep the SAME initial concentrations for on/off comparisons.
            // The legacy concentration-BC layer is a one-time initial inventory
            // when O diffusion is off; it is not subsequently replenished.
            if(c.bc=="concentration" && i==1){o(i,j)=Physics::layerO;cr(i,j)=Physics::layerCr;}
        }
        // Retain the pre-insertion inventory: the initial seed is recorded in
        // O_seed/Cr_seed, so the diagnostics budget remains balanced at t=0.
        initialO=sum(o);initialCr=sum(cr);
        if(c.nucleation=="center")initialize_center_oxide();
        if(has_eta())initialize_eta();
        refresh();
    }
    static long double sum(const Field&a){
        long double v=0,total=0;for(int i=a.domain.first;i<=a.domain.last;++i)for(int j=1;j<=a.ny;++j)v+=a(i,j);
        MPI_Allreduce(&v,&total,1,MPI_LONG_DOUBLE,MPI_SUM,a.domain.comm);return total;
    }
    void voronoi(){
        seeds.clear();areas.clear();
        if(!c.grains){
            // A genuine single-grain representation: uniform ID, no GB mask.
            seeds.push_back({0.5*c.nx,0.5*c.ny});
            areas.push_back(c.nx*c.ny);
            std::fill(grain.a.begin(),grain.a.end(),1.);
            std::fill(gb.a.begin(),gb.a.end(),0.);
            return;
        }
        const double domain_area=double(c.nx)*c.ny;
        int ng=std::max(1,int(std::llround(4.*domain_area/(PI*c.grain_diameter_grid*c.grain_diameter_grid))));
        if(ng>c.nx*c.ny)throw std::runtime_error("Too many Voronoi sites for this grid.");
        std::mt19937_64 rng(c.random_seed);
        std::uniform_real_distribution<double> ux(0.,double(c.nx)),uy(0.,double(c.ny));
        for(int k=0;k<ng;++k)seeds.push_back({ux(rng),uy(rng)});
        if(c.left_edge_sites<0 || c.left_edge_sites>std::min(ng,c.ny))
            throw std::runtime_error("--left-edge-sites must be between 0 and min(area-based site count, ny).");
        if(c.left_edge_sites>0){
            // Keep ng unchanged. Anchor distinct inlet rows, with random phase and
            // subcell X jitter so this remains a nearest-site Voronoi construction.
            // Each anchor is <0.5 cells from its own inlet cell centre; all other
            // anchors are >=1 row away and unanchored sites are >=1 cell inward.
            // Thus every anchor owns at least one inlet cell (also at physical X=0).
            std::uniform_real_distribution<double> unit(0.,1.);
            const int phase=std::min(c.ny-1,int(unit(rng)*c.ny));
            for(int k=0;k<c.left_edge_sites;++k){
                const int row=(phase+int((static_cast<long long>(k)*c.ny)/c.left_edge_sites))%c.ny;
                seeds[k]={0.05+0.4*unit(rng),row+0.5};
            }
            for(int k=c.left_edge_sites;k<ng;++k)
                seeds[k].x=1.5+(c.nx-1.5)*(seeds[k].x/c.nx);
        }
        areas.assign(ng,0);
        for(int i=domain.first;i<=domain.last;++i)for(int j=1;j<=c.ny;++j){
            const double x=i-.5,y=j-.5;
            int owner=0;double rbest=std::numeric_limits<double>::infinity(),ownerY=0.;
            for(int k=0;k<ng;++k){
                double dy=minimum_image(seeds[k].y-y,c.ny);
                double r=(seeds[k].x-x)*(seeds[k].x-x)+dy*dy;
                if(r<rbest){rbest=r;owner=k;ownerY=y+dy;}
            }
            grain(i,j)=owner+1;areas[owner]++;
            double db=std::numeric_limits<double>::infinity();
            // Exact distances to Voronoi bisector planes; periodic Y images included.
            // Images of the SAME physical grain never count as a grain boundary.
            for(int k=0;k<ng;++k)if(k!=owner)for(int im=-1;im<=1;++im){
                double ym=seeds[k].y+im*c.ny;
                double rs=(seeds[k].x-x)*(seeds[k].x-x)+(ym-y)*(ym-y);
                double sep=std::hypot(seeds[k].x-seeds[owner].x,ym-ownerY);
                if(sep>1.e-14)db=std::min(db,std::max(0.,(rs-rbest)/(2.*sep)));
            }
            // Width is the FULL width at half maximum across a flat boundary.
            // This is a prescribed transport mask, not a grain-boundary free energy.
            gb(i,j)=ng==1?0.:std::exp(-4.*std::log(2.)*db*db/(c.gb_fwhm_grid*c.gb_fwhm_grid));
        }
        if(domain.size>1)MPI_Allreduce(MPI_IN_PLACE,areas.data(),int(areas.size()),MPI_INT,MPI_SUM,domain.comm);
        ghosts(grain);ghosts(gb);
    }
    // ----- Eta initialization, dynamic GB indicator and variational forces -----
    void initialize_eta(){
        if(seeds.size()>static_cast<std::size_t>(c.max_eta_fields))
            throw std::runtime_error("Too many eta fields. Increase --max-eta-fields only after checking memory, or use --grain-evolution off.");
        eta.reserve(seeds.size());eta_new.reserve(seeds.size());
        for(std::size_t g=0;g<seeds.size();++g){
            eta.emplace_back(c.nx,c.ny,domain);eta_new.emplace_back(c.nx,c.ny,domain);
        }
        const double ell_grid=eta_ell(c)/c.dx;
        // Soft nearest-seed weights, with distances to Voronoi bisectors. For an
        // isolated planar two-grain boundary this gives the exact logistic profile.
        // No grain-growth steps are hidden in initialization. Junctions relax later.
        std::vector<double> weights(seeds.size());
        for(int i=domain.first;i<=domain.last;++i)for(int j=1;j<=c.ny;++j){
            const int owner=int(grain(i,j))-1;
            const double x=i-.5,y=j-.5;
            const double oy=y+minimum_image(seeds[owner].y-y,c.ny);
            const double rbest=std::pow(seeds[owner].x-x,2)+std::pow(oy-y,2);
            double norm=0.;
            for(std::size_t g=0;g<seeds.size();++g){
                if(int(g)==owner){weights[g]=1.;norm+=1.;continue;}
                double distance=std::numeric_limits<double>::infinity();
                for(int im=-1;im<=1;++im){
                    const double gy=seeds[g].y+im*c.ny;
                    const double sep=std::hypot(seeds[g].x-seeds[owner].x,gy-oy);
                    const double r=std::pow(seeds[g].x-x,2)+std::pow(gy-y,2);
                    if(sep>1.e-14)distance=std::min(distance,std::max(0.,(r-rbest)/(2.*sep)));
                }
                weights[g]=std::exp(-distance/ell_grid);norm+=weights[g];
            }
            // Center hard seeds already have phi=1 here. Do not leave metal order
            // inside a particle which has been explicitly inserted as pure oxide.
            const double metal=1.-h(p(i,j));
            for(std::size_t g=0;g<eta.size();++g)eta[g](i,j)=metal*weights[g]/norm;
        }
    }
    void rebuild_eta_mask(){
        const double scale=eta_transport_scale(c);
        for(int i=domain.first;i<=domain.last;++i)for(int j=1;j<=c.ny;++j){
            double S=0.,pairs=0.;
            for(const auto& e:eta){const double q=e(i,j)*e(i,j);pairs+=S*q;S+=q;}
            eta_s2(i,j)=S;eta_pairs(i,j)=pairs;
            // An overlap of TWO METAL eta fields is required. A single metal/oxide
            // interface is NOT mistaken for a substrate grain boundary.
            // Clamping is only of this diagnostic/transport indicator, never of eta.
            gb(i,j)=scale*std::min(1.,16.*pairs);
        }
        ghosts(gb);
    }
    double eta_force(std::size_t g,int i,int j)const{
        return grain_local_deta(eta[g](i,j),eta_s2(i,j),p(i,j),c)-eta_K(c)*lap(eta[g],i,j,c);
    }
    double eta_phi_force(int i,int j)const{
        return has_eta()?grain_local_dphi(eta_s2(i,j),p(i,j),c):0.;
    }
    void eta_trial(){
        for(std::size_t g=0;g<eta.size();++g)
            for(int i=domain.first;i<=domain.last;++i)for(int j=1;j<=c.ny;++j){
                const double v=eta[g](i,j)-c.dt*c.eta_L*eta_force(g,i,j);
                if(!std::isfinite(v)||v < -1.e-12||v > 1.+1.e-12){
                    std::ostringstream msg;msg<<std::setprecision(17)
                        <<"Eta trial outside [0,1]: grain="<<g+1<<" cell=("<<i<<","<<j
                        <<") value="<<v<<". All PDE fields rejected together; eta is never clipped. Check eta coefficients/dt.";
                    throw std::runtime_error(msg.str());
                }
                eta_new[g](i,j)=v;
            }
    }
    int current_grain(int i,int j)const{
        if(p(i,j)>=.5)return 0;
        if(!has_eta())return int(grain(i,j));
        double best=1.e-12;int id=0;
        for(std::size_t g=0;g<eta.size();++g)if(eta[g](i,j)>best){best=eta[g](i,j);id=int(g)+1;}
        return id;
    }
    long double grain_energy()const{
        // Forward-face gradient energy, counted once per physical edge. Its exact
        // discrete derivative is -K*lap(eta) with the same Neumann X / periodic Y BCs.
        long double local=0.,total=0.;
        if(!has_eta())return 0.;
        for(int i=domain.first;i<=domain.last;++i)for(int j=1;j<=c.ny;++j){
            double density=grain_local_energy(eta_s2(i,j),eta_pairs(i,j),p(i,j),c);
            for(const auto& e:eta){
                if(i<c.nx)density+=.5*eta_K(c)*std::pow((e(i+1,j)-e(i,j))/c.dx,2);
                density+=.5*eta_K(c)*std::pow((e(i,j+1)-e(i,j))/c.dy,2);
            }
            local+=density*c.dx*c.dy;
        }
        MPI_Allreduce(&local,&total,1,MPI_LONG_DOUBLE,MPI_SUM,domain.comm);return total;
    }
    void grain_log(std::ostream& f,std::ostream& history,int n){
        if(!has_eta())return;
        if(domain.size>1){gather_state();root_action(domain.comm,[&]{snapshot->grain_log(f,history,n);});return;}
        refresh();
        std::vector<long double> amount(eta.size(),0.);
        std::vector<int> cells(eta.size(),0);
        double lo=1.,hi=0.,slo=std::numeric_limits<double>::infinity(),shi=0.,gbmax=0.;
        long double sumS=0.,mask=0.;int unassigned=0;
        for(int i=1;i<=c.nx;++i)for(int j=1;j<=c.ny;++j){
            const double S=eta_s2(i,j),metal=1.-h(p(i,j));
            slo=std::min(slo,S);shi=std::max(shi,S);sumS+=S;mask+=metal*gb(i,j);gbmax=std::max(gbmax,gb(i,j));
            const int id=current_grain(i,j);if(id)++cells[std::size_t(id-1)];else if(p(i,j)<.5)++unassigned;
            for(std::size_t g=0;g<eta.size();++g){
                const double v=eta[g](i,j);lo=std::min(lo,v);hi=std::max(hi,v);
                if(S>1.e-24)amount[g]+=metal*v*v/S;
            }
        }
        const auto alive=std::count_if(cells.begin(),cells.end(),[](int v){return v>0;});
        f<<std::setprecision(17)<<n<<","<<n*c.dt<<","<<lo<<","<<hi<<","<<slo<<","<<shi
         <<","<<sumS/(c.nx*c.ny)<<","<<alive<<","<<unassigned<<","<<mask*c.dx*c.dy<<","<<gbmax<<","<<grain_energy()<<"\n";
        for(std::size_t g=0;g<eta.size();++g)
            history<<std::setprecision(17)<<n<<","<<n*c.dt<<","<<g+1<<","<<cells[g]<<","<<amount[g]
                   <<","<<amount[g]*c.dx*c.dy<<"\n";
        if(!f||!history)throw std::runtime_error("Grain diagnostic write failed.");
    }

    void enforce_legacy_layer(){
        if(domain.first!=1 || c.bc!="concentration" || !c.oxygen_diffusion)return;
        for(int j=1;j<=c.ny;++j){b.bcO+=Physics::layerO-o(1,j);o(1,j)=Physics::layerO;}
    }
    void refresh(){
        ghosts(o);ghosts(cr);ghosts(p);
        if(has_eta()){
            for(auto& e:eta)ghosts(e);
            rebuild_eta_mask();
        }
        for(int i=domain.first;i<=domain.last;++i)for(int j=1;j<=c.ny;++j){
            double hh=h(p(i,j));
            muO(i,j)=(1.-hh)*mu_metal_o(o(i,j),cr(i,j))
                +hh*(24000000.*o(i,j)-13800000.*cr(i,j)-8880000.)/Physics::Vm
                -2.*Physics::ko*lap(o,i,j,c);
            muCr(i,j)=(1.-hh)*(1166528.60789006*cr(i,j)+3200000.*o(i,j)-97045.64)/Physics::Vm
                +hh*(14000000.*cr(i,j)-13800000.*o(i,j)+2680000.)/Physics::Vm
                -2.*Physics::kc*lap(cr,i,j,c);
        }
        ghosts(muO);ghosts(muCr);
        // Boundary is the FACE between ghost i=0 and the first cell i=1.
        if(domain.first==1 && c.bc=="mu" && c.oxygen_diffusion)for(int j=1;j<=c.ny;++j)muO(0,j)=2.*c.mu_res-muO(1,j);
        for(int i=domain.first-1;i<=domain.last+1;++i)for(int j=0;j<=c.ny+1;++j){
            mo(i,j)=mobility(o(i,j),p(i,j),gb(i,j),true,c);
            mc(i,j)=mobility(cr(i,j),p(i,j),gb(i,j),false,c);
        }
    }
    double left_flux(int j)const{
        if(domain.first!=1 || c.bc!="mu" || !c.oxygen_diffusion)return 0.;
        // Explicit, positive boundary-face kinetic closure for the zero-O start.
        // surface_c_ref sets a face mobility, NOT an enforced concentration.
        // Gating by the current oxide fraction prevents a GB short-circuit in oxide.
        const double mf=mobility(c.surface_c_ref,p(1,j),gb(1,j),true,c);
        return 2.*mf*(c.mu_res-muO(1,j))/c.dx; // + means into the material
    }
    double div_flux(const Field&m,const Field&mu,int i,int j,bool oxygen)const{
        const double right=-.5*(m(i,j)+m(i+1,j))*(mu(i+1,j)-mu(i,j))/c.dx;
        double left=-.5*(m(i-1,j)+m(i,j))*(mu(i,j)-mu(i-1,j))/c.dx;
        if(oxygen && i==1 && c.bc=="mu")left=left_flux(j);
        const double top=-.5*(m(i,j)+m(i,j+1))*(mu(i,j+1)-mu(i,j))/c.dy;
        const double bottom=-.5*(m(i,j-1)+m(i,j))*(mu(i,j)-mu(i,j-1))/c.dy;
        return (left-right)/c.dx+(bottom-top)/c.dy;
    }
    void gather_field(const Field& src,Field* dst){
        MPI_Gatherv(&src(src.domain.first,0),domain.counts[domain.rank],MPI_DOUBLE,
                    dst?&(*dst)(1,0):nullptr,domain.counts.data(),domain.displacements.data(),
                    MPI_DOUBLE,0,domain.comm);
    }
    void scatter_field(Field& dst,const Field* src){
        MPI_Scatterv(src?&(*src)(1,0):nullptr,domain.counts.data(),domain.displacements.data(),MPI_DOUBLE,
                     &dst(domain.first,0),domain.counts[domain.rank],MPI_DOUBLE,0,domain.comm);
    }
    void gather_state(){
        root_action(domain.comm,[&]{if(!snapshot)snapshot=std::make_unique<Model>(c);});
        gather_field(o,snapshot?&snapshot->o:nullptr);
        gather_field(cr,snapshot?&snapshot->cr:nullptr);
        gather_field(p,snapshot?&snapshot->p:nullptr);
        for(std::size_t g=0;g<eta.size();++g)gather_field(eta[g],snapshot?&snapshot->eta[g]:nullptr);
        long double local[]={b.nucO,b.nucCr,b.ckO,b.ckCr,b.clipO,b.clipCr,b.bcO,b.bcCr},total[8]={};
        long long local_count[]={b.nuclei,b.clipped},counts[2]={};
        MPI_Reduce(local,total,8,MPI_LONG_DOUBLE,MPI_SUM,0,domain.comm);
        MPI_Reduce(local_count,counts,2,MPI_LONG_LONG_INT,MPI_SUM,0,domain.comm);
        if(snapshot){
            snapshot->b={total[0],total[1],total[2],total[3],total[4],total[5],total[6],total[7],counts[0],counts[1]};
            snapshot->initialO=initialO;snapshot->initialCr=initialCr;
            snapshot->center_seed_cells=center_seed_cells;
        }
    }
    int ordered_seed(bool scan,int x=0,int y=0){
        gather_state();int changed=0;
        root_action(domain.comm,[&]{
            snapshot->b=Budget{};
            if(scan)snapshot->nucleate();else changed=snapshot->stamp(x,y);
        });
        scatter_field(o,snapshot?&snapshot->o:nullptr);
        scatter_field(cr,snapshot?&snapshot->cr:nullptr);
        scatter_field(p,snapshot?&snapshot->p:nullptr);
        for(std::size_t g=0;g<eta.size();++g)scatter_field(eta[g],snapshot?&snapshot->eta[g]:nullptr);
        if(snapshot){
            // Count each event/mass increment exactly once, even across slabs.
            b.nucO+=snapshot->b.nucO;b.nucCr+=snapshot->b.nucCr;b.nuclei+=snapshot->b.nuclei;
        }
        MPI_Bcast(&changed,1,MPI_INT,0,domain.comm);
        return changed;
    }
    bool valid_nucleus(int x,int y)const{
        const double r=c.radius_grid;
        // Original eligible x-region, but NO integer truncation of the radius.
        if(x-r<5. || x+r>double(c.nx-3))return false;
        int sr=int(std::ceil(r+c.exclusion_gap_grid));
        for(int i=std::max(1,x-sr);i<=std::min(c.nx,x+sr);++i)for(int j=1;j<=c.ny;++j){
            if(p(i,j)>.01 && std::hypot(double(i-x),minimum_image(j-y,c.ny))<r+c.exclusion_gap_grid)return false;
        }
        return true;
    }
    int stamp(int x,int y){
        if(domain.size>1)return ordered_seed(false,x,y);
        int changed=0,rr=int(std::ceil(c.radius_grid));
        for(int i=std::max(1,x-rr);i<=std::min(c.nx,x+rr);++i)for(int j=1;j<=c.ny;++j){
            double dy=minimum_image(j-y,c.ny);
            if((i-x)*(i-x)+dy*dy<=c.radius_grid*c.radius_grid && p(i,j)<.01){
                b.nucO+=Physics::oxideO-o(i,j);b.nucCr+=Physics::oxideCr-cr(i,j);
                o(i,j)=Physics::oxideO;cr(i,j)=Physics::oxideCr;p(i,j)=1.;++changed;
                for(auto& e:eta)e(i,j)=0.; // Same discrete insertion: consume metal order locally.
            }
        }
        if(changed){++b.nuclei;}
        return changed;
    }
    void initialize_center_oxide(){
        // Cell centres are at (i-0.5)*dx. Fractional index coordinates keep the
        // seed at the exact box centre for ODD as well as EVEN dimensions.
        const double x=0.5*(double(c.nx)+1.), y=0.5*(double(c.ny)+1.);
        const double r=c.radius_grid;
        // Keep every outermost row/column of cell centres outside the seed.
        // Reject oversized disks instead of silently cropping/wrapping them.
        if(r>=0.5*(double(std::min(c.nx,c.ny))-1.))
            throw std::runtime_error("Center seed is too large: --radius-grid must be less than (min(nx,ny)-1)/2.");
        if(b.nuclei!=0)
            throw std::runtime_error("The center oxide has already been initialized.");
        for(int i=domain.first;i<=domain.last;++i)for(int j=1;j<=c.ny;++j){
            const double dx=i-x, dy=j-y;
            if(dx*dx+dy*dy<=r*r){
                // Same hard-seed composition assignment as stamp(), no smoothing.
                b.nucO+=Physics::oxideO-o(i,j);b.nucCr+=Physics::oxideCr-cr(i,j);
                o(i,j)=Physics::oxideO;cr(i,j)=Physics::oxideCr;p(i,j)=1.;
                ++center_seed_cells;
                for(auto& e:eta)e(i,j)=0.;
            }
        }
        if(domain.size>1)MPI_Allreduce(MPI_IN_PLACE,&center_seed_cells,1,MPI_INT,MPI_SUM,domain.comm);
        if(center_seed_cells==0)
            throw std::runtime_error("Center seed covers no cell centres: increase --radius-grid.");
        if(domain.rank==0)++b.nuclei;
    }
    void nucleate(){
        // The initial centre seed is NOT replenished, even if it later dissolves.
        if(c.nucleation!="ksp")return;
        if(domain.size>1){ordered_seed(true);return;}
        for(int i=domain.first;i<=domain.last;++i)for(int j=1;j<=c.ny;++j){
            if(p(i,j)<=.01 && o(i,j)*cr(i,j)>Physics::ksp && valid_nucleus(i,j))stamp(i,j);
        }
    }
    void step(){
        enforce_legacy_layer();refresh();
        for(int j=1;j<=c.ny;++j)b.bcO+=c.dt*left_flux(j)/c.dx;
        std::string trial_error;
        try{
        for(int i=domain.first;i<=domain.last;++i)for(int j=1;j<=c.ny;++j){
            double oo=o(i,j),cc=cr(i,j),pp=p(i,j);
            // O diffusion off removes the transport increment, not other
            // mechanisms that modify O (hard nucleation, optional legacy Ck).
            on(i,j)=c.oxygen_diffusion?oo+c.dt*div_flux(mo,muO,i,j,true):oo;
            crn(i,j)=cc+c.dt*div_flux(mc,muCr,i,j,false);
            const double hp=30.*pp*pp*(pp-1.)*(pp-1.);
            double fp=Physics::W*(4.*pp*pp*pp-6.*pp*pp+2.*pp)
                      +hp*(foxide(oo,cc)-fmetal(oo,cc))-2.*Physics::kp*lap(p,i,j,c);
            if(has_eta())fp+=eta_phi_force(i,j);
            pn(i,j)=pp-c.dt*Physics::L*fp;
            if(c.ck=="legacy"){
                double dO=-Physics::ckO*((Physics::oxideO-on(i,j))*h(pn(i,j))-(Physics::oxideO-oo)*h(pp))/c.dt;
                double dCr=-Physics::ckCr*((Physics::oxideCr-crn(i,j))*h(pn(i,j))-(Physics::oxideCr-cc)*h(pp))/c.dt;
                on(i,j)+=dO;crn(i,j)+=dCr;b.ckO+=dO;b.ckCr+=dCr;
            }
            if(!std::isfinite(on(i,j)) || !std::isfinite(crn(i,j)) || !std::isfinite(pn(i,j)))
                throw std::runtime_error("Nonfinite trial field: reduce dt / inspect the model; no update accepted.");
            if(c.bounds=="legacy"){
                double oc=std::clamp(on(i,j),0.,1.),ccr=std::clamp(crn(i,j),0.,1.);
                b.clipO+=oc-on(i,j);b.clipCr+=ccr-crn(i,j);
                if(oc!=on(i,j)||ccr!=crn(i,j))++b.clipped;
                on(i,j)=oc;crn(i,j)=ccr;
                if(pn(i,j)>.99999)pn(i,j)=1.;
                if(pn(i,j)<1.e-5)pn(i,j)=0.;
            }else{
                if(on(i,j)<-1.e-12 || crn(i,j)<-1.e-12 || on(i,j)+crn(i,j)>1.+1.e-12 ||
                   pn(i,j)<-1.e-12 || pn(i,j)>1.+1.e-12){
                    std::ostringstream msg;
                    msg<<std::scientific<<std::setprecision(16)<<"Out-of-bounds trial at ("<<i<<","<<j<<"): O="<<on(i,j)
                       <<", Cr="<<crn(i,j)<<", phi="<<pn(i,j)
                       <<". No clipping/update accepted. Reduce dt; if failure persists, investigate model.";
                    throw std::runtime_error(msg.str());
                }
            }
        }
        if(has_eta())eta_trial();
        }catch(const std::exception& e){trial_error=e.what();}
        collective_error(domain.comm,trial_error);
        o.a.swap(on.a);cr.a.swap(crn.a);p.a.swap(pn.a);
        for(std::size_t g=0;g<eta.size();++g)eta[g].a.swap(eta_new[g].a);
    }
    void vtk(const std::string& path){
        if(domain.size>1){gather_state();root_action(domain.comm,[&]{snapshot->vtk(path);});return;}
        refresh();
        std::ofstream f(path);if(!f)throw std::runtime_error("Cannot write "+path);
        f<<std::setprecision(16)<<"# vtk DataFile Version 3.0\n"
         <<(has_eta()?"NiCr evolving metal grains; coordinates in metres\n":"NiCr fixed-grain prototype; coordinates in metres\n")
         <<"ASCII\nDATASET STRUCTURED_POINTS\n"
         <<"DIMENSIONS "<<c.nx<<" "<<c.ny<<" 1\nORIGIN "<<.5*c.dx<<" "<<.5*c.dy<<" 0\nSPACING "<<c.dx<<" "<<c.dy<<" 1\nPOINT_DATA "<<c.nx*c.ny<<"\n";
        auto write=[&](const char* name,auto fn){f<<"SCALARS "<<name<<" double 1\nLOOKUP_TABLE default\n";
            for(int j=1;j<=c.ny;++j){for(int i=domain.first;i<=domain.last;++i)f<<fn(i,j)<<" ";f<<"\n";}};
        write("phi",[&](int i,int j){return p(i,j);});
        write("conc_O",[&](int i,int j){return o(i,j);});
        write("conc_Cr",[&](int i,int j){return cr(i,j);});
        write("conc_Ni",[&](int i,int j){return 1.-o(i,j)-cr(i,j);});
        write("grain_id",[&](int i,int j){return has_eta()?double(current_grain(i,j)):grain(i,j);});
        write("GB_mask",[&](int i,int j){return gb(i,j);});
        write("GB_active",[&](int i,int j){return (1.-h(p(i,j)))*gb(i,j);});
        write("mu_O",[&](int i,int j){return muO(i,j);});
        write("mu_Cr",[&](int i,int j){return muCr(i,j);});
        write("M_O",[&](int i,int j){return mo(i,j);});
        write("M_Cr",[&](int i,int j){return mc(i,j);});
        if(has_eta()){
            write("grain_id_initial",[&](int i,int j){return grain(i,j);});
            write("GB_initial",[&](int i,int j){return gb_initial(i,j);});
            write("GB_eta_indicator",[&](int i,int j){return std::min(1.,16.*eta_pairs(i,j));});
            write("metal_fraction",[&](int i,int j){return 1.-h(p(i,j));});
            write("eta_sum_sq",[&](int i,int j){return eta_s2(i,j);});
            write("active_grain_id",[&](int i,int j){return double(current_grain(i,j));});
            if(c.eta_output=="all")for(std::size_t g=0;g<eta.size();++g){
                const std::string name="eta_"+std::to_string(g+1);
                write(name.c_str(),[&](int i,int j){return eta[g](i,j);});
            }
        }
        if(!f)throw std::runtime_error("Output write failed: "+path);
    }
    void metadata(){
        if(domain.rank!=0)return;
        std::ofstream f(c.out+"/parameters.txt");
        f<<std::setprecision(17)<<(has_eta()?"STATUS: COUPLED ETA/OXIDATION MODEL; GRAIN PARAMETERS REQUIRE CALIBRATION\n":"STATUS: UNCALIBRATED FIXED-GRAIN TRANSPORT PROTOTYPE\n")
         <<"nx="<<c.nx<<"\nny="<<c.ny<<"\ndx_m="<<c.dx<<"\ndy_m="<<c.dy<<"\ndt_s="<<c.dt
         <<"\nsteps="<<c.steps<<"\noxygen_bc="<<c.bc<<"\nCk="<<c.ck<<"\nbounds="<<c.bounds
         <<"\nmpi_processes="<<domain.size<<"\nmpi_decomposition=X slabs"
         <<"\nversion="<<VERSION<<"\nnucleation_mode="<<c.nucleation
         <<"\ninitial_center_seed_cells="<<center_seed_cells
         <<"\ngrain_structure="<<(c.grains?"on":"off")
         <<"\noxygen_diffusion="<<(c.oxygen_diffusion?"on":"off")
         <<"\noxygen_boundary_supply="<<(c.oxygen_diffusion?c.bc:"disabled")
         <<"\nW="<<Physics::W<<"\nkappa_phi="<<Physics::kp<<"\nL_phi="<<Physics::L
         <<"\nmu_res_model_units="<<c.mu_res<<"\nsurface_mobility_concentration_reference="<<c.surface_c_ref
         <<"\nleft_edge_sites_requested="<<c.left_edge_sites
         <<"\nleft_edge_sites_active="<<(c.grains?c.left_edge_sites:0)
         <<"\ngrain_diameter_target_grid="<<c.grain_diameter_grid
         <<"\ngrain_diameter_target_m="<<c.grain_diameter_grid*c.dx<<"\nvoronoi_sites="<<seeds.size()
         <<"\ngb_mask_FWHM_grid="<<c.gb_fwhm_grid<<"\ngb_mask_FWHM_m="<<c.gb_fwhm_grid*c.dx
         <<"\ngb_mask_integral_effective_width_grid="<<c.gb_fwhm_grid*std::sqrt(PI)/(2.*std::sqrt(std::log(2.)))
         <<"\ngb_mask_integral_effective_width_m="<<c.gb_fwhm_grid*c.dx*std::sqrt(PI)/(2.*std::sqrt(std::log(2.)))
         <<"\ngb_peak_mobility_factor_O="<<c.gb_factor_o
         <<"\ngb_peak_mobility_factor_Cr="<<c.gb_factor_cr<<"\nrandom_seed="<<c.random_seed
         <<"\nnucleus_geometric_radius_grid="<<c.radius_grid<<"\nnucleus_geometric_radius_m="<<c.radius_grid*c.dx
         <<"\nnucleation_period_s="<<c.nucleation_period
         <<"\nNOTE: Vm=1 is retained, not evidence that energy/mobility units have been physically audited.\n"
         <<"NOTE: no GB migration, GB energy, segregation, oxide GBs, or interphase enhancement.\n"
         <<"NOTE: cell-centred grid; periodic Y; no-flux right; natural concentration/phi derivatives in X.\n";
#ifdef NICR_SERIAL_VERIFY
        f<<"execution_backend=single_process_verification_shim\n";
#elif defined(NICR_MPI_THREAD_TEST)
        f<<"execution_backend=thread_rank_emulation\n";
#else
        f<<"execution_backend=native_MPI\n";
#endif
        f<<"grain_evolution_requested="<<(c.grain_evolution?"on":"off")
         <<"\ngrain_evolution_active="<<(has_eta()?"on":"off")<<"\n";
        if(has_eta()){
            f<<"eta_fields="<<eta.size()<<"\neta_W="<<c.eta_W<<"\neta_kappa="<<eta_K(c)
             <<"\neta_L="<<c.eta_L<<"\neta_oxide_penalty="<<eta_A(c)
             <<"\neta_width_10_90_grid="<<2.*std::log(9.)*eta_ell(c)/c.dx
             <<"\neta_mask_integrated_width_m="<<8.*eta_ell(c)/3.
             <<"\neta_transport_scale="<<eta_transport_scale(c)
             <<"\neta_planar_GB_energy_model_units="<<std::sqrt(2.*eta_K(c)*c.eta_W)/3.
             <<"\neta_planar_geometric_mobility_model_units="<<3.*c.eta_L*eta_ell(c)
             <<"\neta_output="<<c.eta_output<<"\n"
             <<"NOTE: New grain coefficients are provisional, not experimental Ni-Cr calibration.\n"
             <<"NOTE: Dynamic eta modifies the GB pathways AND adds the reciprocal grain-energy force to phi.\n"
             <<"NOTE: GB factors refer to the original integrated-width convention; GB_mask includes eta_transport_scale.\n";
        }else f<<"NOTE: fixed initial grain map and Gaussian GB mask.\n";
        if(c.nucleation=="center")f<<"initial_oxide_center_x_m="<<0.5*c.nx*c.dx
            <<"\ninitial_oxide_center_y_m="<<0.5*c.ny*c.dy
            <<"\ninitial_oxide_center_i="<<0.5*(double(c.nx)+1.)
            <<"\ninitial_oxide_center_j="<<0.5*(double(c.ny)+1.)
            <<"\nNOTE: one initial hard oxide only; no later Ksp insertion or replacement seed.\n";
        if(!c.grains)f<<"NOTE: single grain; grain_id=1 and GB_mask=0; configured GB factors are inactive.\n";
        if(!c.oxygen_diffusion)f<<"NOTE: O transport and continuing O boundary supply are off; initial inventory, hard seed assignments, optional Ck and bounds treatment remain.\n";
        std::ofstream g(c.out+"/grains.csv");g<<"grain_id,seed_x_grid,seed_y_grid,area_cells,equivalent_diameter_grid,equivalent_diameter_m\n";
        long double mean=0;
        for(std::size_t k=0;k<areas.size();++k){double d=2.*std::sqrt(areas[k]/PI);mean+=d;
            g<<std::setprecision(17)<<k+1<<","<<seeds[k].x<<","<<seeds[k].y<<","<<areas[k]<<","<<d<<","<<d*c.dx<<"\n";}
        mean/=areas.size();f<<"measured_arithmetic_mean_equivalent_diameter_grid="<<mean<<"\n";
        if(!f||!g)throw std::runtime_error("Metadata write failed.");
    }
    void log(std::ostream&f,int n){
        if(domain.size>1){gather_state();root_action(domain.comm,[&]{snapshot->log(f,n);});return;}
        long double so=sum(o),sc=sum(cr),po=sum(p);
        long double ro=so-initialO-b.bcO-b.nucO-b.ckO-b.clipO;
        long double rc=sc-initialCr-b.bcCr-b.nucCr-b.ckCr-b.clipCr;
        double omin=1.,omax=0.,crmin=1.,crmax=0.,nimin=1.,phimin=1.,phimax=0.;
        for(int i=domain.first;i<=domain.last;++i)for(int j=1;j<=c.ny;++j){
            omin=std::min(omin,o(i,j));omax=std::max(omax,o(i,j));crmin=std::min(crmin,cr(i,j));crmax=std::max(crmax,cr(i,j));
            nimin=std::min(nimin,1.-o(i,j)-cr(i,j));phimin=std::min(phimin,p(i,j));phimax=std::max(phimax,p(i,j));}
        f<<std::setprecision(17)<<n<<","<<n*c.dt<<","<<so<<","<<sc<<","<<po/(c.nx*c.ny)
         <<","<<b.bcO<<","<<b.nucO<<","<<b.nucCr<<","<<b.ckO<<","<<b.ckCr<<","<<b.clipO<<","<<b.clipCr
         <<","<<ro<<","<<rc<<","<<b.nuclei<<","<<b.clipped<<","<<omin<<","<<omax<<","<<crmin<<","<<crmax
         <<","<<nimin<<","<<phimin<<","<<phimax<<"\n";
    }
};
static bool parse_on_off(const std::string& value,const std::string& option){
    if(value=="on")return true;
    if(value=="off")return false;
    throw std::runtime_error(option+" expects on or off (received: "+value+")");
}
struct EarlyExit {};
static Config parse(int argc,char**argv){
    int rank=0;MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    Config c;
    for(int k=1;k<argc;++k){std::string a=argv[k];
        if(a=="--self-test"){c.self_test=true;continue;}
        if(a=="--version"){if(rank==0)std::cout<<VERSION<<"\n";throw EarlyExit{};}
        if(a=="--help"){
            if(rank==0)std::cout<<"Initialization only by default. Options:\n--steps N --dt s --nx N --ny N --dx m (sets dx=dy) --output-every N\n"
              <<"--grains on|off (default on; off: uniform single crystal, no GB enhancement)\n"
              <<"--grain-evolution on|off (default off; on: moving eta grains, dynamic GB transport and oxide coupling)\n"
              <<"--eta-W value --eta-kappa value (0: auto width) --eta-L value --eta-oxide-penalty value (-1: eta-W)\n"
              <<"--eta-output all|summary --max-eta-fields N (default 512; memory guard)\n"
              <<"--oxygen-diffusion on|off (default on; off: no O transport or ongoing O boundary supply)\n"
              <<"--left-edge-sites N (default 0; relocate N existing sites to partition inlet; ignored with grains off)\n"
              <<"--grain-diameter-grid N --gb-width-grid N (mask FWHM)\n"
              <<"--gb-factor-o N --gb-factor-cr N [--gb-factor N sets both; compatibility alias] --seed N\n"
              <<"--nucleation ksp|center|off (default ksp; center: ONE initial seed; off: transport/grain growth only)\n"
              <<"--radius-grid R (radius in grid spacings) --nucleation-period s (Ksp mode only)\n"
              <<"--bc concentration|mu|closed --mu-res value --surface-c-ref value\n--ck legacy|off --bounds stop|legacy --out directory --self-test --version\n"
              <<"Change --seed to change the Voronoi structure; repeat it with the same geometry/build to reproduce it.\n"
              <<"O diffusion off does NOT disable hard-seed concentration assignments or legacy Ck.\n"
              <<"With concentration BC, the initial left O/Cr layer is retained but O is not replenished when off.\n";
            throw EarlyExit{};
        }
        if(k+1>=argc)throw std::runtime_error("Missing value for "+a);
        std::string v=argv[++k];
        if(a=="--steps")c.steps=std::stoi(v);else if(a=="--dt")c.dt=std::stod(v);
        else if(a=="--nx")c.nx=std::stoi(v);else if(a=="--ny")c.ny=std::stoi(v);
        else if(a=="--dx")c.dx=c.dy=std::stod(v);else if(a=="--output-every")c.output_every=std::stoi(v);
        else if(a=="--grains")c.grains=parse_on_off(v,a);
        else if(a=="--grain-evolution")c.grain_evolution=parse_on_off(v,a);
        else if(a=="--eta-W")c.eta_W=std::stod(v);
        else if(a=="--eta-kappa")c.eta_kappa=std::stod(v);
        else if(a=="--eta-L")c.eta_L=std::stod(v);
        else if(a=="--eta-oxide-penalty")c.eta_oxide_penalty=std::stod(v);
        else if(a=="--eta-output")c.eta_output=v;
        else if(a=="--max-eta-fields")c.max_eta_fields=std::stoi(v);
        else if(a=="--oxygen-diffusion")c.oxygen_diffusion=parse_on_off(v,a);
        else if(a=="--left-edge-sites"){std::size_t used=0;c.left_edge_sites=std::stoi(v,&used);if(used!=v.size()||c.left_edge_sites<0)throw std::runtime_error("--left-edge-sites expects a nonnegative integer.");}
        else if(a=="--grain-diameter-grid")c.grain_diameter_grid=std::stod(v);
        else if(a=="--gb-width-grid")c.gb_fwhm_grid=std::stod(v);
        else if(a=="--gb-factor-o")c.gb_factor_o=std::stod(v);
        else if(a=="--gb-factor-cr")c.gb_factor_cr=std::stod(v);
        else if(a=="--gb-factor"){c.gb_factor_o=std::stod(v);c.gb_factor_cr=c.gb_factor_o;}
        else if(a=="--nucleation")c.nucleation=v;
        else if(a=="--radius-grid")c.radius_grid=std::stod(v);
        else if(a=="--nucleation-period")c.nucleation_period=std::stod(v);
        else if(a=="--seed")c.random_seed=std::stoull(v);else if(a=="--bc")c.bc=v;
        else if(a=="--ck")c.ck=v;else if(a=="--bounds")c.bounds=v;else if(a=="--out")c.out=v;
        else if(a=="--mu-res")c.mu_res=std::stod(v);else if(a=="--surface-c-ref")c.surface_c_ref=std::stod(v);
        else throw std::runtime_error("Unknown argument: "+a);
    }
    for(double x:{c.dx,c.dy,c.dt,c.grain_diameter_grid,c.gb_fwhm_grid,c.gb_factor_o,c.gb_factor_cr,c.radius_grid,c.nucleation_period,c.mu_res,c.surface_c_ref})
        if(!std::isfinite(x))throw std::runtime_error("All numerical parameters must be finite.");
    if(c.nx<3||c.ny<3||c.nx>INT_MAX-2||c.ny>INT_MAX-2||c.steps<0||c.output_every<1||c.dt<=0||c.dx<=0||c.dy<=0||c.grain_diameter_grid<=0||c.gb_fwhm_grid<=0||c.gb_factor_o<1||c.gb_factor_cr<1||c.radius_grid<=0||(c.nucleation=="ksp"&&c.nucleation_period<c.dt)||c.nucleation_period<=0.||c.surface_c_ref<=0)
        throw std::runtime_error("Invalid parameter bounds (Ksp nucleation period must be >= dt).");
    for(double v:{c.eta_W,c.eta_kappa,c.eta_L,c.eta_oxide_penalty})
        if(!std::isfinite(v))throw std::runtime_error("Eta parameters must be finite.");
    if(c.eta_W<=0.||c.eta_kappa<0.||c.eta_L<0.||
       (c.eta_oxide_penalty<0.&&c.eta_oxide_penalty!=-1.)||c.max_eta_fields<1)
        throw std::runtime_error("Invalid eta parameter bounds.");
    if(c.eta_output!="all"&&c.eta_output!="summary")throw std::runtime_error("--eta-output expects all or summary.");
    if(c.grain_evolution && c.grains && (!std::isfinite(eta_K(c))||!std::isfinite(eta_transport_scale(c))||eta_K(c)<=0.))
        throw std::runtime_error("Derived eta coefficients are invalid.");
    if(c.nucleation!="ksp"&&c.nucleation!="center"&&c.nucleation!="off")
        throw std::runtime_error("--nucleation expects ksp, center, or off.");
    if(c.nucleation=="center"&&c.radius_grid>=0.5*(double(std::min(c.nx,c.ny))-1.))
        throw std::runtime_error("Center seed is too large: --radius-grid must be less than (min(nx,ny)-1)/2.");
    if(c.bc!="mu"&&c.bc!="concentration"&&c.bc!="closed")throw std::runtime_error("Invalid bc.");
    if(c.ck!="legacy"&&c.ck!="off")throw std::runtime_error("Invalid Ck mode.");
    if(c.bounds!="stop"&&c.bounds!="legacy")throw std::runtime_error("Invalid bounds mode.");
    return c;
}
static void require(bool v,const char* msg){if(!v)throw std::runtime_error(std::string("Self-test failed: ")+msg);}
static void self_test(){
    for(int ny:{60,61})for(int seed=0;seed<20;++seed){
        Config edge;edge.nx=61;edge.ny=ny;edge.random_seed=seed;
        edge.left_edge_sites=2;Model anchored(edge),repeat(edge);
        require(anchored.seeds.size()==2,"area-based count unchanged by anchors");
        require(anchored.grain.a==repeat.grain.a,"anchor reproducibility");
        std::vector<bool> seen(2,false);
        for(int j=1;j<=ny;++j)seen[int(anchored.grain(1,j))-1]=true;
        require(seen[0]&&seen[1],"both grains partition inlet");
        edge.grains=false;Model off(edge);require(Model::sum(off.gb)==0.,"grains off overrides anchors");
    }
    Config many;many.nx=61;many.ny=61;many.grain_diameter_grid=15.;many.left_edge_sites=5;
    Model anchored_many(many);std::vector<bool> seen_many(5,false);
    for(int j=1;j<=many.ny;++j){int id=int(anchored_many.grain(1,j))-1;if(id<5)seen_many[id]=true;}
    for(bool seen:seen_many)require(seen,"anchors survive interior competitors");
    Config invalid;invalid.nx=61;invalid.ny=61;invalid.left_edge_sites=3;
    bool rejected=false;try{Model m(invalid);}catch(const std::runtime_error&){rejected=true;}
    require(rejected,"excess anchors rejected without changing grain count");
    invalid.grain_diameter_grid=1000.;invalid.left_edge_sites=1;Model one_anchor(invalid);
    require(one_anchor.seeds.size()==1&&Model::sum(one_anchor.gb)==0.,"one-site anchor has no GB");
    std::cout<<"PASS: left-edge Voronoi anchors; count preservation; inlet coverage; reproducibility; limits.\n";
    Config c;c.nx=31;c.ny=25;c.grain_diameter_grid=10.;c.bc="closed";c.ck="off";c.dt=1.e-12;
    Model m(c);
    require(Model::sum(m.p)==0.,"initial oxide must be zero");
    for(int i=1;i<=c.nx;++i)for(int j=1;j<=c.ny;++j){require(m.grain(i,j)>=1.,"complete Voronoi coverage");require(m.gb(i,j)>=0.&&m.gb(i,j)<=1.,"bounded GB mask");}
    require(std::abs(mobility(.1,0,1,true,c)/mobility(.1,0,0,true,c)-c.gb_factor_o)<1.e-10,"oxygen species-specific GB factor");
    require(std::abs(mobility(.1,0,1,false,c)/mobility(.1,0,0,false,c)-c.gb_factor_cr)<1.e-10,"Cr species-specific GB factor");
    require(mobility(.1,1,1,true,c)==mobility(.1,1,0,true,c),"no substrate GB enhancement in oxide");
    Config one=c;one.grain_diameter_grid=1000.;Model single(one);require(Model::sum(single.gb)==0.,"single grain has no GB");
    for(int i=1;i<=c.nx;++i)for(int j=1;j<=c.ny;++j){m.o(i,j)=.001*(1.+.01*std::sin(2.*PI*j/c.ny));m.cr(i,j)=.1986;}
    m.refresh();long double doSum=0,dcSum=0;
    for(int i=1;i<=c.nx;++i)for(int j=1;j<=c.ny;++j){doSum+=m.div_flux(m.mo,m.muO,i,j,true);dcSum+=m.div_flux(m.mc,m.muCr,i,j,false);}
    require(std::abs(doSum)<1.e-5 && std::abs(dcSum)<1.e-5,"closed face flux telescopes");
    long double so=Model::sum(m.o),sc=Model::sum(m.cr);m.step();
    require(std::abs(Model::sum(m.o)-so)<1.e-12&&std::abs(Model::sum(m.cr)-sc)<1.e-12,"conservative update");
    Config mu=c;mu.bc="mu";Model dry(mu);double flux=dry.left_flux(5);require(flux>0.,"mu BC starts from dry material");dry.step();require(Model::sum(dry.o)>0.,"oxygen enters without concentration reset");
    require(dry.o(1,5)!=Physics::layerO,"left cell not pinned to concentration");
    Model nucleus(c);int cells=nucleus.stamp(15,12);require(cells==97,"R=5.5 disk has 97 sites, not R=5 disk");
    require(std::abs((Model::sum(nucleus.o)-nucleus.initialO)-nucleus.b.nucO)<1.e-12,"seed oxygen budget");
    require(std::abs((Model::sum(nucleus.cr)-nucleus.initialCr)-nucleus.b.nucCr)<1.e-12,"seed chromium budget");
    require(std::abs(mu_metal_o(.0023,.1986)-364688.)<1.e-8,"mu reference arithmetic");
    // New switch tests (original tests above remain unchanged).
    Config defaults;
    require(defaults.grains&&defaults.oxygen_diffusion,"new switches default to on");
    Model repeat1(c),repeat2(c);
    require(repeat1.grain.a==repeat2.grain.a&&repeat1.gb.a==repeat2.gb.a,"same seed reproduces geometry");
    Config different=c;++different.random_seed;Model other(different);
    require(repeat1.grain.a!=other.grain.a&&repeat1.gb.a!=other.gb.a,"different seed changes geometry");
    Config ng=c;ng.grains=false;Model no_grains(ng);
    require(no_grains.seeds.size()==1&&no_grains.areas[0]==ng.nx*ng.ny,"single-crystal bookkeeping");
    for(double v:no_grains.grain.a)require(v==1.,"single crystal grain ID is uniform including ghosts");
    for(double v:no_grains.gb.a)require(v==0.,"single crystal GB mask is zero including ghosts");
    require(mobility(.1,0.,1.,true,ng)==mobility(.1,0.,0.,true,c),"single crystal ignores O GB multiplier");
    require(mobility(.1,0.,1.,false,ng)==mobility(.1,0.,0.,false,c),"single crystal ignores Cr GB multiplier");
    ++ng.random_seed;Model no_grains_other(ng);
    require(no_grains.grain.a==no_grains_other.grain.a&&no_grains.gb.a==no_grains_other.gb.a,"single crystal independent of RNG seed");
    Config od=c;od.oxygen_diffusion=false;
    for(double pp:{0.,0.5,1.}){
        require(mobility(.1,pp,1.,true,od)==0.,"O mobility off in metal, interface and oxide");
        require(mobility(.1,pp,1.,false,od)==mobility(.1,pp,1.,false,c),"Cr mobility unchanged by O switch");
    }
    for(const std::string boundary:{"mu","concentration","closed"}){
        Config q=od;q.bc=boundary;q.dt=1.e-10;Model test(q);
        // Nonuniform, positive fields: freezing zero everywhere is not enough
        // to test that the internal O transport was really disabled.
        for(int i=1;i<=q.nx;++i)for(int j=1;j<=q.ny;++j){
            test.o(i,j)=.001*(1.+.01*std::sin(2.*PI*j/q.ny));
            test.cr(i,j)=.1986+.0001*std::cos(2.*PI*j/q.ny);
            test.p(i,j)=.1;
        }
        test.refresh();const auto old_o=test.o.a,old_cr=test.cr.a,old_p=test.p.a;
        for(int n=0;n<10;++n)test.step();
        require(test.o.a==old_o,"O transport off keeps all O values unchanged without Ck/seeding");
        require(test.cr.a!=old_cr,"Cr still evolves when O transport is off");
        require(test.p.a!=old_p,"phi still evolves when O transport is off");
        require(test.b.bcO==0.&&test.b.clipO==0.,"no hidden O boundary reset/inflow or clipping");
        require(test.left_flux(5)==0.,"O face influx exactly zero when off");
    }
    Config lay=c;lay.bc="concentration";Model layer_on(lay);lay.oxygen_diffusion=false;Model layer_off(lay);
    require(layer_on.o.a==layer_off.o.a&&layer_on.cr.a==layer_off.cr.a,"O switch leaves initial legacy-layer inventory unchanged");
    Config nomu=mu;nomu.oxygen_diffusion=false;Model dry_off(nomu);
    for(int n=0;n<10;++n)dry_off.step();
    require(Model::sum(dry_off.o)==0.&&dry_off.b.bcO==0.,"no O supplied to dry mu-BC run when off");
    Model hard_off(od);require(hard_off.stamp(15,12)==97,"original hard seed still works with O diffusion off");
    require(hard_off.b.nucO>0.&&hard_off.b.nucCr>0.,"hard seed mass changes remain separately accounted");
    hard_off.refresh();const auto stamped_o=hard_off.o.a;hard_off.step();
    require(hard_off.o.a==stamped_o,"no O redistribution after hard stamp with Ck off");
    Config withck=od;withck.ck="legacy";Model ck_off(withck);
    for(int i=1;i<=withck.nx;++i)for(int j=1;j<=withck.ny;++j){ck_off.o(i,j)=.001;ck_off.p(i,j)=.1;}
    ck_off.refresh();ck_off.step();
    require(ck_off.b.ckO!=0.&&ck_off.b.bcO==0.,"O-off does not silently remove optional legacy Ck");
    require(parse_on_off("on","test")&&!parse_on_off("off","test"),"on/off parsing");
    bool bad=false;try{(void)parse_on_off("yes","test");}catch(const std::runtime_error&){bad=true;}
    require(bad,"invalid switch value rejected");
    // Center-seed geometry, budgets, independence from transport, and no reseeding.
    require(defaults.nucleation=="ksp","default Ksp insertion is preserved");
    for(int nx:{31,32})for(int ny:{25,26})for(double radius:{3.,5.,5.5}){
        Config t=c;t.nucleation="center";t.nx=nx;t.ny=ny;t.radius_grid=radius;
        t.oxygen_diffusion=false;
        Model ctr(t);
        const double cx=0.5*(nx+1.),cy=0.5*(ny+1.);
        int expected=0;long double sumx=0.,sumy=0.;
        for(int i=1;i<=nx;++i)for(int j=1;j<=ny;++j){
            double xx=i-cx,yy=j-cy;
            bool inside=xx*xx+yy*yy<=radius*radius;
            require(ctr.p(i,j)==(inside?1.:0.),"center seed disk and float radius");
            require(ctr.o(i,j)==(inside?Physics::oxideO:Physics::bulkO),"only center disk receives oxide O");
            require(ctr.cr(i,j)==(inside?Physics::oxideCr:Physics::bulkCr),"only center disk receives oxide Cr");
            require(ctr.p(i,j)==ctr.p(nx+1-i,j)&&ctr.p(i,j)==ctr.p(i,ny+1-j),"center seed reflection symmetry");
            if(inside){++expected;sumx+=i;sumy+=j;}
        }
        require(expected==ctr.center_seed_cells&&ctr.b.nuclei==1,"one initial oxide event");
        require(sumx/expected==cx&&sumy/expected==cy,"exact geometric centre, odd/even sizes");
        if(nx==31&&ny==25&&radius==5.5)require(expected==97,"center R=5.5 is not truncated to R=5");
        require(std::abs(Model::sum(ctr.o)-ctr.initialO-ctr.b.nucO)<1.e-12,"initial center O budget");
        require(std::abs(Model::sum(ctr.cr)-ctr.initialCr-ctr.b.nucCr)<1.e-12,"initial center Cr budget");
        const auto original_o=ctr.o.a;
        for(int k=0;k<5;++k)ctr.step();
        ctr.refresh();
        require(ctr.o.a==original_o,"O diffusion off leaves center O unchanged with Ck off");
        require(ctr.b.bcO==0.&&ctr.b.clipO==0.,"center seed has no hidden O supply/clipping");
        // Make every location supersaturated and erase the seed. Center mode
        // must still not call stamp(), including after its only seed disappears.
        for(int i=1;i<=nx;++i)for(int j=1;j<=ny;++j){ctr.p(i,j)=0.;ctr.o(i,j)=.001;}
        const auto before_scan=ctr.o.a;ctr.nucleate();ctr.nucleate();
        require(ctr.b.nuclei==1&&Model::sum(ctr.p)==0.&&ctr.o.a==before_scan,"center mode never reinserts after dissolution");
    }
    Config centered=c;centered.nucleation="center";
    Model cg(centered);centered.grains=false;Model cs(centered);
    require(cg.p.a==cs.p.a&&cg.o.a==cs.o.a&&cg.cr.a==cs.cr.a,"grain switch does not move the center seed");
    centered.radius_grid=0.1;centered.nx=32;centered.ny=26;
    bad=false;try{Model unresolved(centered);}catch(const std::runtime_error&){bad=true;}
    require(bad,"sub-grid unresolved center disk rejected");
    centered.radius_grid=13.;
    bad=false;try{Model oversized(centered);}catch(const std::runtime_error&){bad=true;}
    require(bad,"oversized center disk rejected");
    std::cout<<"PASS: center seed on odd/even boxes; float radius; oxide composition/locality; budgets; no reseeding.\n";
    std::cout<<"PASS: original self-tests; seed reproducibility; single-crystal geometry/mobility;\n"
             <<"O transport/inlet switches for all BCs; Cr/phi remain active; original hard-seed/Ck semantics; option parsing.\n";
}
static void mpi_compare(Model& distributed,const Model& serial,const char* label){
    bool same=true;
    for(int i=distributed.domain.first;i<=distributed.domain.last;++i)
        for(int j=1;j<=distributed.c.ny;++j){
            same=same&&distributed.o(i,j)==serial.o(i,j)&&distributed.cr(i,j)==serial.cr(i,j)
                &&distributed.p(i,j)==serial.p(i,j)&&distributed.grain(i,j)==serial.grain(i,j)
                &&distributed.gb(i,j)==serial.gb(i,j);
            same=same&&(distributed.eta.size()==serial.eta.size());
            for(std::size_t g=0;g<distributed.eta.size();++g)
                same=same&&distributed.eta[g](i,j)==serial.eta[g](i,j);
        }
    collective_error(distributed.domain.comm,same?"":std::string("MPI self-test field mismatch: ")+label);
    distributed.gather_state();
    root_action(distributed.domain.comm,[&]{
        const auto& got=distributed.snapshot->b;const auto& expected=serial.b;
        const long double a[]={got.nucO,got.nucCr,got.ckO,got.ckCr,got.clipO,got.clipCr,got.bcO,got.bcCr};
        const long double b[]={expected.nucO,expected.nucCr,expected.ckO,expected.ckCr,expected.clipO,expected.clipCr,expected.bcO,expected.bcCr};
        for(int k=0;k<8;++k)require(std::abs(a[k]-b[k])<=1.e-11L*(1.+std::abs(b[k])),"MPI budget reduction");
        require(got.nuclei==expected.nuclei&&got.clipped==expected.clipped,"MPI event counts");
    });
}
static void mpi_self_test(){
    int size=1,rank=0;MPI_Comm_size(MPI_COMM_WORLD,&size);MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    for(const std::string bc:{"mu","closed","concentration"})for(bool oxygen:{false,true}){
        Config c;c.nx=std::max(35,size+1);c.ny=19;c.dt=1.e-12;c.bc=bc;
        c.oxygen_diffusion=oxygen;c.bounds="legacy";c.grain_diameter_grid=10.;
        c.left_edge_sites=2;c.nucleation="center";c.radius_grid=3.5;
        Model distributed(c,MPI_COMM_WORLD),serial(c);
        mpi_compare(distributed,serial,"initial center seed and geometry");
        // Nonuniform fields exercise both X-slab faces and periodic Y.
        auto fill=[](Model& m){
            for(int i=m.domain.first;i<=m.domain.last;++i)for(int j=1;j<=m.c.ny;++j){
                m.o(i,j)=.001*(1.+.1*std::sin(.3*i+.5*j));
                m.cr(i,j)=.19+.001*std::cos(.2*i-.3*j);
                m.p(i,j)=.1+.01*std::cos(.1*i+.4*j);
            }
        };
        fill(distributed);fill(serial);
        for(int step=0;step<5;++step){distributed.step();serial.step();}
        mpi_compare(distributed,serial,"transport, Ck, clipping and physical boundaries");
    }
    Config c;c.nx=std::max(61,size+1);c.ny=31;c.radius_grid=3.;c.dt=1.e-12;
    c.bc="closed";c.ck="off";c.bounds="legacy";
    Model distributed(c,MPI_COMM_WORLD),serial(c);
    const int x=std::max(10,std::min(c.nx-10,c.nx/std::max(2,size)));
    require(distributed.stamp(x,1)==serial.stamp(x,1),"MPI hard stamp changed-cell count");
    mpi_compare(distributed,serial,"hard stamp across slab and periodic seam");
    for(int i=distributed.domain.first;i<=distributed.domain.last;++i)for(int j=1;j<=c.ny;++j)
        distributed.o(i,j)=.001;
    for(int i=1;i<=c.nx;++i)for(int j=1;j<=c.ny;++j)serial.o(i,j)=.001;
    distributed.nucleate();serial.nucleate();
    mpi_compare(distributed,serial,"ordered Ksp scan including exclusion across ranks");
    for(int k=0;k<3;++k){distributed.step();serial.step();}
    mpi_compare(distributed,serial,"post-nucleation update");
    Config thin=c;thin.nx=std::max(3,size);thin.ny=9;thin.grains=false;thin.radius_grid=1.;
    Model slab(thin,MPI_COMM_WORLD),reference(thin);
    for(int k=0;k<3;++k){slab.step();reference.step();}
    mpi_compare(slab,reference,"one-column slabs and grains off");
    slab.c.bounds="stop";
    if(rank==size-1)slab.cr(slab.domain.last,5)=-.1;
    bool rejected=false;
    try{slab.step();}catch(const std::runtime_error&){rejected=true;}
    collective_error(MPI_COMM_WORLD,rejected?"":"MPI trial failure was not propagated to every rank");
    if(rank==0)std::cout<<"PASS: MPI vs serial fields exactly equal; budgets; all BCs; O switch; center/hard/Ksp seeds;\n"
                       <<"uneven and one-column slabs; cross-rank/periodic exclusion; collective trial failure.\n";
}

// ----- Grain-subsystem verification (no experimental calibration implied) -----
static void eta_set_bicrystal(Model& m,bool circle){
    const double ell=eta_ell(m.c),cx=0.5*m.c.nx*m.c.dx,cy=0.5*m.c.ny*m.c.dy;
    for(int i=m.domain.first;i<=m.domain.last;++i)for(int j=1;j<=m.c.ny;++j){
        const double x=(i-.5)*m.c.dx,y=(j-.5)*m.c.dy;
        const double d=circle?std::hypot(x-cx,y-cy)-12.:x-cx;
        const double q=1./(1.+std::exp(d/ell));
        for(std::size_t g=0;g<m.eta.size();++g)m.eta[g](i,j)=g==0?q:(g==1?1.-q:0.);
        m.o(i,j)=0.;m.cr(i,j)=Physics::bulkCr;m.p(i,j)=0.;
    }
    m.refresh();
}
static void eta_self_test(){
    Config c;c.grain_evolution=true;c.nx=35;c.ny=29;c.grain_diameter_grid=16.;
    c.bc="closed";c.ck="off";c.nucleation="off";c.dt=1.e-10;
    Model m(c);require(m.eta.size()==m.seeds.size(),"one eta per initial Voronoi site");
    for(int i=1;i<=c.nx;++i)for(int j=1;j<=c.ny;++j){
        double sum=0.;for(const auto& e:m.eta)sum+=e(i,j);
        require(std::abs(sum-1.)<1.e-14,"smooth metal eta initialization partitions unity");
    }
    require(std::abs(eta_transport_scale(c)-1.)<1.e-14,"auto eta width matches old integrated GB mask");
    Config off=c;off.grains=false;Model single(off);
    require(single.eta.empty()&&Model::sum(single.gb)==0.,"grains off overrides eta evolution");
    Config one=c;one.grain_diameter_grid=1000.;Model one_eta(one);
    for(int i=1;i<=one.nx;++i)for(int j=1;j<=one.ny;++j){one_eta.eta[0](i,j)=.5;one_eta.p(i,j)=.4;}
    one_eta.refresh();require(Model::sum(one_eta.gb)==0.,"one metal/oxide interface is not a GB");

    // Check both variational derivatives, including physical X and periodic Y faces.
    Config fd=c;fd.dx=fd.dy=1.;fd.eta_W=1.;fd.eta_kappa=2.;fd.eta_oxide_penalty=.7;
    Model deriv(fd);
    for(int i=1;i<=fd.nx;++i)for(int j=1;j<=fd.ny;++j){
        deriv.p(i,j)=.25+.08*std::sin(.31*i+.27*j);
        for(std::size_t g=0;g<deriv.eta.size();++g)deriv.eta[g](i,j)=.1+.04*std::cos(.2*i+.3*j+.7*g);
    }
    deriv.refresh();double max_error=0.;
    for(int i:{1,fd.nx/2,fd.nx})for(int j:{1,fd.ny/2,fd.ny}){
        for(int variable=0;variable<2;++variable){
            double& v=variable?deriv.p(i,j):deriv.eta[0](i,j);
            const double original=v,epsilon=1.e-5;
            const double exact=variable?deriv.eta_phi_force(i,j):deriv.eta_force(0,i,j);
            v=original+epsilon;deriv.refresh();const long double plus=deriv.grain_energy();
            v=original-epsilon;deriv.refresh();const long double minus=deriv.grain_energy();
            v=original;deriv.refresh();
            const double approx=double((plus-minus)/(2.*epsilon*fd.dx*fd.dy));
            const double err=std::abs(exact-approx)/(1.+std::abs(exact));max_error=std::max(max_error,err);
            require(err<1.e-7,"grain energy vs phi/eta derivative");
        }
    }
    std::cout<<"ETA CHECK: largest scaled energy-derivative error="<<max_error<<"\n";

    // Exact planar logistic solution: second-order spatial residual convergence.
    double previous=0.;
    for(double dx:{1.,.5,.25}){
        Config t=c;t.dx=t.dy=dx;t.nx=int(64./dx);t.ny=5;
        t.grain_diameter_grid=std::sqrt(4.*t.nx*t.ny/(PI*2.));
        t.eta_W=1.;t.eta_kappa=8.;t.gb_fwhm_grid=4./dx;
        Model planar(t);require(planar.eta.size()==2,"planar test has exactly two eta fields");
        eta_set_bicrystal(planar,false);double residual=0.,integral=0.;
        for(int i=2;i<t.nx;++i)residual=std::max(residual,std::abs(planar.eta_force(0,i,3)));
        for(int i=1;i<=t.nx;++i)integral+=planar.gb(i,3)*dx;
        require(std::abs(integral/reference_gb_width(t)-1.)<1.e-7,"planar integrated GB conductance normalization");
        if(previous)require(previous/residual>3.4&&previous/residual<4.6,"planar residual converges second order");
        std::cout<<"ETA CHECK: planar dx="<<dx<<" max force="<<residual<<" mask integral="<<integral<<"\n";
        previous=residual;
    }
    // Curvature-driven motion in a chemically uniform metal: eta really evolves.
    Config grow=c;grow.dx=grow.dy=1.;grow.nx=grow.ny=65;
    grow.eta_W=1.;grow.eta_kappa=2.;grow.eta_L=1.;grow.dt=.005;
    grow.grain_diameter_grid=std::sqrt(4.*grow.nx*grow.ny/(PI*2.));
    Model curved(grow);eta_set_bicrystal(curved,true);
    const long double E0=curved.grain_energy(),area0=Model::sum(curved.eta[0]);
    const auto initial_mask=curved.gb.a;long double previous_energy=E0;
    for(int n=0;n<400;++n){
        curved.step();curved.refresh();
        const long double energy=curved.grain_energy();
        require(energy<=previous_energy+1.e-10L,"closed metal grain energy decreases");previous_energy=energy;
    }
    const long double area1=Model::sum(curved.eta[0]);
    require(area1<area0*.99L,"curved grain shrinks");require(curved.gb.a!=initial_mask,"GB mask moves with eta");
    require(Model::sum(curved.p)==0.,"grain-only benchmark does not create oxide");
    std::cout<<"ETA CHECK: circular grain area "<<double(area0)<<" -> "<<double(area1)
             <<"; energy "<<double(E0)<<" -> "<<double(previous_energy)<<"\n";

    // Eta must be suppressed by oxide even without a direct seeding reset.
    Config suppress=grow;suppress.eta_oxide_penalty=1.;Model oxide(suppress);
    for(int i=1;i<=suppress.nx;++i)for(int j=1;j<=suppress.ny;++j){
        for(auto& e:oxide.eta)e(i,j)=0.;
        oxide.eta[0](i,j)=.5;
        oxide.p(i,j)=1.;oxide.o(i,j)=Physics::oxideO;oxide.cr(i,j)=Physics::oxideCr;
    }
    oxide.refresh();const long double before=Model::sum(oxide.eta[0]);
    for(int n=0;n<100;++n)oxide.step();
    oxide.refresh();
    require(Model::sum(oxide.eta[0])<before*.8L,"oxide coupling suppresses residual metal order");
    require(Model::sum(oxide.gb)==0.,"oxide does not create false metal GBs");
    // Hard nucleation remains discrete and clears eta only inside the same disk.
    Model seed(c);const int count=seed.stamp(15,1);
    require(count>0,"hard seed inserted");int cleared=0;
    for(int i=1;i<=c.nx;++i)for(int j=1;j<=c.ny;++j)if(seed.p(i,j)==1.){
        for(const auto& e:seed.eta)require(e(i,j)==0.,"hard seed consumes every metal eta");
        ++cleared;
    }
    require(cleared==count,"eta reset follows hard-seed geometry");
    Config center=c;center.nucleation="center";Model initial(center);
    for(int i=1;i<=c.nx;++i)for(int j=1;j<=c.ny;++j)if(initial.p(i,j)==1.)
        for(const auto& e:initial.eta)require(e(i,j)==0.,"initial center seed has no metal eta");
    std::cout<<"PASS: dynamic eta initialization, variational coupling, planar convergence, conductance normalization, curvature motion, oxide suppression and hard seeds.\n";
}
static void mpi_eta_self_test(){
    int size=1,rank=0;MPI_Comm_size(MPI_COMM_WORLD,&size);MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    for(const std::string bc:{"mu","closed","concentration"})for(bool oxygen:{false,true}){
        Config c;c.grain_evolution=true;c.nx=std::max(35,size+1);c.ny=19;
        c.dt=1.e-12;c.bc=bc;c.oxygen_diffusion=oxygen;c.bounds="legacy";c.ck="off";
        c.grain_diameter_grid=10.;c.left_edge_sites=2;c.nucleation="center";c.radius_grid=3.5;
        Model distributed(c,MPI_COMM_WORLD),reference(c);
        mpi_compare(distributed,reference,"eta initialization and center seed");
        auto fill=[](Model& m){
            for(int i=m.domain.first;i<=m.domain.last;++i)for(int j=1;j<=m.c.ny;++j){
                m.o(i,j)=.001*(1.+.1*std::sin(.3*i+.5*j));
                m.cr(i,j)=.19+.001*std::cos(.2*i-.3*j);m.p(i,j)=.1+.01*std::cos(.1*i+.4*j);
                for(std::size_t g=0;g<m.eta.size();++g)m.eta[g](i,j)=.1+.02*std::cos(.2*i-.3*j+.7*g);
            }
        };
        fill(distributed);fill(reference);
        for(int n=0;n<8;++n){distributed.step();reference.step();}
        distributed.refresh();reference.refresh();
        mpi_compare(distributed,reference,"eta halos, coupled phi, dynamic transport and boundaries");
        // The full-domain output/Ksp snapshot must contain the ACTUAL evolved eta.
        distributed.gather_state();root_action(MPI_COMM_WORLD,[&]{
            for(std::size_t g=0;g<reference.eta.size();++g)
                for(int i=1;i<=c.nx;++i)for(int j=1;j<=c.ny;++j)
                    require(distributed.snapshot->eta[g](i,j)==reference.eta[g](i,j),"gathered eta matches serial reference");
        });
    }
    Config c;c.grain_evolution=true;c.nx=std::max(61,size+1);c.ny=31;
    c.grain_diameter_grid=20.;c.dt=1.e-12;c.bc="closed";c.ck="off";c.bounds="legacy";
    c.radius_grid=3.;Model m(c,MPI_COMM_WORLD),r(c);
    const int x=std::max(10,std::min(c.nx-10,c.nx/std::max(2,size)));
    require(m.stamp(x,1)==r.stamp(x,1),"eta MPI hard stamp count");
    mpi_compare(m,r,"eta reset across slab/periodic seam");
    for(int i=m.domain.first;i<=m.domain.last;++i)for(int j=1;j<=c.ny;++j)m.o(i,j)=.001;
    for(int i=1;i<=c.nx;++i)for(int j=1;j<=c.ny;++j)r.o(i,j)=.001;
    m.nucleate();r.nucleate();mpi_compare(m,r,"ordered Ksp and eta scatter");
    for(int n=0;n<4;++n){m.step();r.step();}
    m.refresh();r.refresh();mpi_compare(m,r,"coupled post-nucleation eta");
    // One-column slabs exercise eta halo exchange with the smallest local domain.
    Config thin=c;thin.nx=std::max(3,size);thin.ny=9;thin.nucleation="off";
    thin.grain_diameter_grid=2.;thin.radius_grid=1.;Model slab(thin,MPI_COMM_WORLD),one(thin);
    for(int n=0;n<4;++n){slab.step();one.step();}
    slab.refresh();one.refresh();mpi_compare(slab,one,"eta one-column slabs");
    // Collective eta trial failure must not accept the other PDE fields.
    if(rank==size-1)slab.eta[0](slab.domain.last,5)=2.;
    const auto old_o=slab.o.a;bool rejected=false;
    try{slab.step();}catch(const std::runtime_error&){rejected=true;}
    collective_error(MPI_COMM_WORLD,rejected?"":"Eta failure did not reach every rank");
    for(int i=slab.domain.first;i<=slab.domain.last;++i)for(int j=1;j<=thin.ny;++j){
        const std::size_t idx=std::size_t(i-slab.domain.first+1)*(thin.ny+2)+j;
        require(slab.o(i,j)==old_o[idx],"eta failure rejects concentration update too");
    }
    if(rank==0)std::cout<<"PASS: eta MPI/serial comparison; halo/gather/scatter; Ksp; one-column slabs; collective eta rejection.\n";
}

static int run(int argc,char**argv){
    int rank=0;MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    try{
        Config c=parse(argc,argv);if(c.self_test){root_action(MPI_COMM_WORLD,[]{
#ifdef NICR_SERIAL_VERIFY
            std::cout<<"SERIAL VERIFICATION ONLY: all communicators have one process; real multi-rank MPI is NOT tested by this binary.\n";
#endif
            self_test();eta_self_test();});mpi_self_test();mpi_eta_self_test();return 0;}
        root_action(MPI_COMM_WORLD,[&]{
        if(fs::exists(c.out) && !fs::is_empty(c.out))throw std::runtime_error("Output directory is not empty; choose a new --out.");
        fs::create_directories(c.out);
        });
        Model m(c,MPI_COMM_WORLD);root_action(MPI_COMM_WORLD,[&]{m.metadata();});
        std::ofstream log,grain_diag,grain_history;
        root_action(MPI_COMM_WORLD,[&]{
        log.open(c.out+"/diagnostics.csv");if(!log)throw std::runtime_error("Cannot open diagnostics.csv");
        log<<"step,time_s,sum_O,sum_Cr,mean_phi,O_boundary,O_seed,Cr_seed,O_Ck,Cr_Ck,O_clipping,Cr_clipping,O_budget_residual,Cr_budget_residual,nuclei,clipped_cells,min_O,max_O,min_Cr,max_Cr,min_Ni,min_phi,max_phi\n";
        if(m.has_eta()){
            grain_diag.open(c.out+"/grain_diagnostics.csv");grain_history.open(c.out+"/grain_history.csv");
            if(!grain_diag||!grain_history)throw std::runtime_error("Cannot open grain diagnostics.");
            grain_diag<<"step,time_s,min_eta,max_eta,min_sum_eta2,max_sum_eta2,mean_sum_eta2,grains_with_metal_cells,unassigned_metal_cells,integrated_active_GB_m2,max_GB_mask,grain_energy_per_depth\n";
            grain_history<<"step,time_s,grain_id,dominant_metal_cells,weighted_metal_area_cells,weighted_metal_area_m2\n";
        }
        });
        m.log(log,0);m.grain_log(grain_diag,grain_history,0);m.vtk(c.out+"/fields_0.vtk");
        if(rank==0){
#ifdef NICR_SERIAL_VERIFY
        std::cout<<"EXECUTION BACKEND: single-process verification shim, not native MPI.\n";
#elif defined(NICR_MPI_THREAD_TEST)
        std::cout<<"EXECUTION BACKEND: thread rank emulation, not native MPI.\n";
#endif
        std::cout<<"MPI processes: "<<m.domain.size<<" (X slabs)\n";
        std::cout<<"Version "<<VERSION<<"; grains="<<(c.grains?"on":"off")
                  <<"; oxygen_diffusion="<<(c.oxygen_diffusion?"on":"off")
                  <<"; nucleation="<<c.nucleation<<"; "<<m.seeds.size()<<" grain(s).\n"
                 <<"Box "<<c.nx<<" x "<<c.ny<<" cells; configured GB factors O="<<c.gb_factor_o<<", Cr="<<c.gb_factor_cr<<".\n";
        if(c.grains)std::cout<<"Voronoi random seed="<<c.random_seed<<"; target grain diameter="<<c.grain_diameter_grid
                            <<" grid = "<<c.grain_diameter_grid*c.dx*1.e6<<" um.\n";
        else std::cout<<"Single crystal: grain_id=1, GB_mask=0; GB enhancement inactive.\n";
        if(m.has_eta()){
            std::cout<<"Evolving eta: "<<m.eta.size()<<" fields; W_eta="<<c.eta_W<<", kappa_eta="<<eta_K(c)
                     <<", L_eta="<<c.eta_L<<", oxide penalty="<<eta_A(c)<<".\n"
                     <<"Eta 10-90 width="<<2.*std::log(9.)*eta_ell(c)/c.dx<<" grids; GB conductance scale="<<eta_transport_scale(c)<<".\n"
                     <<"NOTE: grain coefficients are provisional; dynamic coupling adds a grain-energy contribution to phi.\n";
            if(2.*std::log(9.)*eta_ell(c)/c.dx<4.)std::cout<<"WARNING: eta interface has fewer than 4 grid spacings across its 10-90 width.\n";
            if(m.eta.size()>64&&c.eta_output=="all")std::cout<<"NOTE: many eta output fields; --eta-output summary reduces file size.\n";
        }
        if(c.nucleation=="center")std::cout<<"One initial hard oxide at box centre ("
            <<0.5*c.nx*c.dx*1.e6<<", "<<0.5*c.ny*c.dy*1.e6<<") um; radius="
            <<c.radius_grid<<" grids; "<<m.center_seed_cells<<" cells. Additional Ksp insertion is OFF.\n";
        if(!c.oxygen_diffusion)std::cout<<"O diffusion and continuing O boundary supply are OFF; hard-seed assignments and optional Ck are unchanged.\n";
        if(c.ck=="legacy")std::cout<<"WARNING: legacy Ck is timestep dependent and nonconservative; retained only for comparison.\n";
        if(c.bc=="mu"&&c.oxygen_diffusion)std::cout<<"WARNING: experimental mu BC uses a finite positive surface mobility closure (see README).\n";
        if(c.grains&&((c.oxygen_diffusion&&c.gb_factor_o>1)||c.gb_factor_cr>1)&&c.steps>0)std::cout<<"WARNING: old timestep is NOT certified stable with enhanced GB mobility.\n";
        }
        double next_nucleation=c.nucleation_period;
        int last_completed=0;
        try{
            for(int n=1;n<=c.steps;++n){
                double time=n*c.dt;
                // Event time is fixed in physical time, not in the number of output steps.
                if(c.nucleation=="ksp"&&time+1.e-12*c.nucleation_period>=next_nucleation){m.nucleate();next_nucleation+=c.nucleation_period;}
                // Roll back budgets if a trial update is rejected. Fields are only swapped on acceptance.
                Budget before=m.b;
                try{m.step();}catch(...){m.b=before;throw;}
                last_completed=n;
                if(rank==0&&(n%1000==0 || n==c.steps)) std::cout<<"step "<<n<<"/"<<c.steps<<" ("<<100.0*n/c.steps<<"%), time="<<time<<" s"<<std::endl;
                if(n%c.output_every==0||n==c.steps){
                    m.log(log,n);m.grain_log(grain_diag,grain_history,n);
                    root_action(MPI_COMM_WORLD,[&]{
                        log.flush();if(!log)throw std::runtime_error("Diagnostics write failed.");
                        if(m.has_eta()){grain_diag.flush();grain_history.flush();if(!grain_diag||!grain_history)throw std::runtime_error("Grain diagnostics write failed.");}
                    });
                    m.vtk(c.out+"/fields_"+std::to_string(n)+".vtk");
                }
            }
        }catch(const std::exception&e){
            root_action(MPI_COMM_WORLD,[&]{std::ofstream err(c.out+"/STOPPED.txt");err<<"Last accepted time step: "<<last_completed<<"\n"<<e.what()<<"\n";});
            // A nucleation event may already have modified the current state before a rejected PDE step.
            m.vtk(c.out+"/state_at_failure.vtk");throw;
        }
        if(rank==0)std::cout<<"Finished "<<last_completed<<" steps. Output: "<<c.out<<"\n";
        return 0;
    }catch(const EarlyExit&){return 0;}
    catch(const std::exception&e){if(rank==0)std::cerr<<"ERROR: "<<e.what()<<"\n";return 1;}
}

int main(int argc,char**argv){
    MPI_Init(&argc,&argv);
    const int result=run(argc,argv);
    // A rank-local allocation/runtime failure must not leave peers blocked in MPI.
    if(result!=0){MPI_Abort(MPI_COMM_WORLD,result);return result;}
    MPI_Finalize();return result;
}
