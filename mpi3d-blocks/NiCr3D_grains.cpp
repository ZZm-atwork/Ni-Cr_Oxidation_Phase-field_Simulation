/* Ni-Cr oxidation in three dimensions, MPI Cartesian block decomposition.
 * Fields: oxygen, chromium, oxide phi, and optional metal-grain eta_i.
 * Boundary conditions: periodic Y/Z; oxygen inlet at X=0; no flux at right.
 * Fixed Voronoi or evolving eta-derived grain-boundary transport.
 * Build: mpic++ -O3 -std=c++17 NiCr3D_grains.cpp -o nicr3d_mpi
 * Run --help for inputs; --self-test for the built-in verification suite.
 * The numerical model, parameter definitions and source sections are in MODEL.md.
 */
// Use the portable MPI C interface from C++; no deprecated MPI C++ bindings.
#ifndef OMPI_SKIP_MPICXX
#define OMPI_SKIP_MPICXX 1
#endif
#ifndef MPICH_SKIP_MPICXX
#define MPICH_SKIP_MPICXX 1
#endif
#ifdef NICR_MPI_THREAD_TEST
#include "tests/mpi_threads.h" // Communication-logic tests only, NOT native MPI.
#elif defined(NICR_SERIAL_VERIFY)
#include "tests/mpi_serial.h" // Single-process numerical verification only.
#else
#include <mpi.h>
#endif
#include <array>
#include <cassert>
#include <exception>
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
constexpr const char* VERSION="mpi-eta-3d-cartesian-v2";
// [1] Numerical controls and material/model coefficients.
struct Config {
    int nx=151, ny=101, nz=101, steps=0, output_every=5000;
    int px=0, py=0, pz=0; // 0: automatic, positive: fix this process-grid axis.
    int left_edge_sites=0; // 0: original random Voronoi; otherwise relocate existing sites.
    unsigned long long random_seed=20260909;
    double dx=1.e-8, dy=1.e-8, dz=1.e-8, dt=2e-10;
    double grain_diameter_grid=50., gb_fwhm_grid=4.;
    // Literature-guided provisional numerical peak factors at 1000 C (1273 K).
    // They are mapped onto the diffuse Gaussian GB mask, so they are NOT raw physical D_GB/D_bulk ratios.
    // O: Park-type Q_GB ~= Q_bulk/2 plus a provisional 0.5 nm physical GB width -> ~28.2.
    // Cr: Chen et al. low-C Alloy B delta*D_GB / D_v mapped onto this mask -> ~563.
    double gb_factor_o=28.2, gb_factor_cr=563.;
    double radius_grid=5.5, exclusion_gap_grid=6., nucleation_period=1.e-7;
    double mu_res=364688., surface_c_ref=0.0023;
    std::string bc="concentration", ck="legacy", bounds="stop", out="nicr3d_grains_output";
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

// One Cartesian block per rank; global indices are retained in the numerical formulas.
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
// [2] Three-dimensional Cartesian blocks; one ghost face in each direction.
// Copies of Domain (including copies held by Field) share MPI resource ownership.
struct CartesianResources {
    MPI_Comm comm=MPI_COMM_NULL;
    bool distributed=false;
    std::array<MPI_Datatype,3> face{{MPI_DATATYPE_NULL,MPI_DATATYPE_NULL,MPI_DATATYPE_NULL}};
    CartesianResources()=default;
    CartesianResources(const CartesianResources&)=delete;
    CartesianResources& operator=(const CartesianResources&)=delete;
    ~CartesianResources(){
        // Fatal rank-local construction errors are handled by main's MPI_Abort.
        // Do not enter a collective communicator release while unwinding one
        // rank alone. Normal destruction frees all types and communicators.
        if(distributed&&std::uncaught_exceptions()>0)return;
        int finalized=0;MPI_Finalized(&finalized);
        if(finalized)return;
        for(auto& type:face)if(type!=MPI_DATATYPE_NULL)MPI_Type_free(&type);
        if(comm!=MPI_COMM_NULL)MPI_Comm_free(&comm);
    }
};
struct Block {
    std::array<int,3> coords{},lo{},hi{},n{};
};
static std::array<int,3> choose_process_grid(int size,const std::array<int,3>& mesh,
                                           const std::array<int,3>& requested){
    long long fixed=1;bool automatic=false;
    for(int axis=0;axis<3;++axis){
        const int p=requested[axis];
        if(p<0)throw std::runtime_error("--px/--py/--pz must be nonnegative (0 means automatic).");
        if(p>mesh[axis])throw std::runtime_error("A requested process-grid axis exceeds its mesh dimension; empty blocks are not supported.");
        if(p==0){automatic=true;continue;}
        if(p>size/fixed)throw std::runtime_error("Product of fixed --px/--py/--pz exceeds MPI process count.");
        fixed*=p;
    }
    if(size%fixed!=0 || (!automatic&&fixed!=size))
        throw std::runtime_error("Process-grid product must equal MPI process count; a partially fixed product must divide it.");
    auto dims=requested;
    MPI_Dims_create(size,3,dims.data());
    auto fits=[&](const std::array<int,3>& v){
        for(int a=0;a<3;++a)if(v[a]<1||v[a]>mesh[a]||(requested[a]>0&&v[a]!=requested[a]))return false;
        return true;
    };
    if(fits(dims))return dims;
    // MPI_Dims_create balances process counts, not physical mesh lengths. For a
    // thin mesh, find a legal factorization rather than silently creating empty
    // blocks. Fixed axes are never changed. Minimize ideal block surface/volume.
    std::array<int,3> best{};double score=std::numeric_limits<double>::infinity();
    std::vector<int> factors;
    for(int f=1;f<=size/f;++f)if(size%f==0){factors.push_back(f);if(f!=size/f)factors.push_back(size/f);}
    std::sort(factors.begin(),factors.end());
    for(int x:factors)for(int y:factors){
        if((size/x)%y!=0)continue;
        std::array<int,3> candidate{{x,y,(size/x)/y}};
        if(!fits(candidate))continue;
        const double q=double(x)/mesh[0]+double(y)/mesh[1]+double(candidate[2])/mesh[2];
        if(q<score){score=q;best=candidate;}
    }
    if(best[0]==0)throw std::runtime_error("No nonempty Cartesian decomposition fits this mesh and MPI process count.");
    return best;
}
struct Domain {
    std::shared_ptr<CartesianResources> resources;
    MPI_Comm comm=MPI_COMM_NULL;
    int rank=0,size=1,first=1,last=0,jfirst=1,jlast=0,kfirst=1,klast=0,plane=0;
    std::array<int,3> dims{},coords{},local{},lo{},hi{};
    std::array<int,3> minus{},plus{};
    std::vector<int> counts,displacements;
    std::vector<Block> blocks;
    Domain(int nx,int ny,int nz,MPI_Comm communicator,int px=0,int py=0,int pz=0){
        MPI_Comm_size(communicator,&size);
        if(nx<3||ny<3||nz<3 || nx>INT_MAX-2||ny>INT_MAX-2||nz>INT_MAX-2)
            throw std::runtime_error("3D grid dimensions must lie in [3, INT_MAX-2].");
        // Retain the existing int-count/root-snapshot size limit, checked before
        // integer products or allocation. Local field storage is block-sized.
        const long long yz=(static_cast<long long>(ny)+2)*(static_cast<long long>(nz)+2);
        if(yz>INT_MAX || (static_cast<long long>(nx)+2)>INT_MAX/yz)
            throw std::runtime_error("3D grid exceeds this solver's int-count MPI gather limit.");
        const std::array<int,3> mesh{{nx,ny,nz}};
        dims=choose_process_grid(size,mesh,{{px,py,pz}});
        resources=std::make_shared<CartesianResources>();
        resources->distributed=size>1;
        const int periods[3]={0,1,1};
        // No rank reordering: rank zero remains the I/O and ordered-seeding root.
        MPI_Cart_create(communicator,3,dims.data(),periods,0,&resources->comm);
        comm=resources->comm;
        if(comm==MPI_COMM_NULL)throw std::runtime_error("MPI_Cart_create excluded a rank unexpectedly.");
        MPI_Comm_rank(comm,&rank);MPI_Cart_coords(comm,rank,3,coords.data());
        counts.resize(size);displacements.resize(size);blocks.resize(size);
        int offset=0;
        for(int r=0;r<size;++r){
            auto& b=blocks[r];MPI_Cart_coords(comm,r,3,b.coords.data());
            for(int a=0;a<3;++a){
                const int base=mesh[a]/dims[a],extra=mesh[a]%dims[a];
                b.n[a]=base+(b.coords[a]<extra?1:0);
                b.lo[a]=1+b.coords[a]*base+std::min(b.coords[a],extra);
                b.hi[a]=b.lo[a]+b.n[a]-1;
            }
            counts[r]=b.n[0]*b.n[1]*b.n[2];displacements[r]=offset;offset+=counts[r];
        }
        local=blocks[rank].n;lo=blocks[rank].lo;hi=blocks[rank].hi;
        first=lo[0];last=hi[0];jfirst=lo[1];jlast=hi[1];kfirst=lo[2];klast=hi[2];
        plane=(local[1]+2)*(local[2]+2);
        const int padded[3]={local[0]+2,local[1]+2,local[2]+2},starts[3]={0,0,0};
        for(int axis=0;axis<3;++axis){
            MPI_Cart_shift(comm,axis,1,&minus[axis],&plus[axis]);
            auto face=local;face[axis]=1;
            // Relative face datatype. Pass a pointer to the face's first cell;
            // its type selects only the owned tangential range, not edges/corners.
            MPI_Type_create_subarray(3,padded,face.data(),starts,MPI_ORDER_C,MPI_DOUBLE,&resources->face[axis]);
            MPI_Type_commit(&resources->face[axis]);
        }
    }
};
struct Field {
    int nx,ny,nz; // global mesh extents; stored arrays are local + a one-cell halo.
    Domain domain;
    std::vector<double> a;
    Field(int x,int y,int z,const Domain& d,double v=0.):nx(x),ny(y),nz(z),domain(d),
        a(std::size_t(d.local[0]+2)*d.plane,v){}
    std::size_t index(int i,int j,int k)const{
#ifdef NICR_INDEX_CHECK
        assert(i>=domain.first-1&&i<=domain.last+1);
        assert(j>=domain.jfirst-1&&j<=domain.jlast+1);
        assert(k>=domain.kfirst-1&&k<=domain.klast+1);
#endif
        return (std::size_t(i-domain.first+1)*(domain.local[1]+2)+(j-domain.jfirst+1))
                *(domain.local[2]+2)+(k-domain.kfirst+1);
    }
    double& operator()(int i,int j,int k){return a[index(i,j,k)];}
    const double& operator()(int i,int j,int k)const{return a[index(i,j,k)];}
};
static double h(double p){return p*p*p*(10.-15.*p+6.*p*p);}
static double minimum_image(double delta,double period){return delta-period*std::round(delta/period);}
static void ghosts(Field& a) {
    const auto& d=a.domain;
    for(int axis=0;axis<3;++axis){
        auto low=d.lo,high=d.lo,receive_low=d.lo,receive_high=d.lo;
        high[axis]=d.hi[axis];receive_low[axis]=d.lo[axis]-1;receive_high[axis]=d.hi[axis]+1;
        const MPI_Datatype type=d.resources->face[axis];
        // Different direction tags also handle periodic dimensions of size two,
        // where both neighbors have the same rank. Size one uses MPI self-send.
        MPI_Sendrecv(&a(low[0],low[1],low[2]),1,type,d.minus[axis],2*axis,
                     &a(receive_high[0],receive_high[1],receive_high[2]),1,type,d.plus[axis],2*axis,
                     d.comm,MPI_STATUS_IGNORE);
        MPI_Sendrecv(&a(high[0],high[1],high[2]),1,type,d.plus[axis],2*axis+1,
                     &a(receive_low[0],receive_low[1],receive_low[2]),1,type,d.minus[axis],2*axis+1,
                     d.comm,MPI_STATUS_IGNORE);
    }
    if(d.first==1)for(int j=d.jfirst;j<=d.jlast;++j)for(int k=d.kfirst;k<=d.klast;++k)a(0,j,k)=a(1,j,k);
    if(d.last==a.nx)for(int j=d.jfirst;j<=d.jlast;++j)for(int k=d.kfirst;k<=d.klast;++k)a(a.nx+1,j,k)=a(a.nx,j,k);
    // Edge and corner padding is deliberately not exchanged. No current stencil
    // reads it. Cahn-Hilliard uses a second face exchange after constructing mu.
}
// Owned cells and six faces only. Never evaluate derived fields on unfilled
// edge/corner padding. In particular, mobility needs only these face values.
template<class Fn> static void cells_and_faces(const Domain& d,Fn fn){
    for(int i=d.first;i<=d.last;++i)for(int j=d.jfirst;j<=d.jlast;++j)for(int k=d.kfirst;k<=d.klast;++k)fn(i,j,k);
    for(int j=d.jfirst;j<=d.jlast;++j)for(int k=d.kfirst;k<=d.klast;++k){fn(d.first-1,j,k);fn(d.last+1,j,k);}
    for(int i=d.first;i<=d.last;++i)for(int k=d.kfirst;k<=d.klast;++k){fn(i,d.jfirst-1,k);fn(i,d.jlast+1,k);}
    for(int i=d.first;i<=d.last;++i)for(int j=d.jfirst;j<=d.jlast;++j){fn(i,j,d.kfirst-1);fn(i,j,d.klast+1);}
}
static Config serial_config(Config c){c.px=c.py=c.pz=0;return c;}
static double lap(const Field&a,int i,int j,int k,const Config&c){
    return (a(i+1,j,k)-2.*a(i,j,k)+a(i-1,j,k))/(c.dx*c.dx)
          +(a(i,j+1,k)-2.*a(i,j,k)+a(i,j-1,k))/(c.dy*c.dy)
          +(a(i,j,k+1)-2.*a(i,j,k)+a(i,j,k-1))/(c.dz*c.dz);
}
// [3] Ni-Cr/Cr2O3 chemical free energies and concentration-weighted mobilities.
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
struct Seed { double x,y,z; };
struct Budget {long double nucO=0,nucCr=0,ckO=0,ckCr=0,clipO=0,clipCr=0,bcO=0,bcCr=0;long long nuclei=0,clipped=0;};
class Model {
public:
    Config c;
    Domain domain;
    std::vector<double> packed_field,root_packed_fields; // Interior-only collective buffers, reused.
    std::unique_ptr<Model> snapshot; // Full arrays exist only on root, for ordered seeding/output.
    Field o,cr,p,on,crn,pn,muO,muCr,mo,mc,grain,gb;
    Field gb_initial,eta_s2,eta_pairs;
    std::vector<Field> eta,eta_new;
    bool has_eta()const{return c.grain_evolution && c.grains;}

    std::vector<Seed> seeds;
    std::vector<int> volumes;
    Budget b;
    long double initialO=0,initialCr=0;
    int center_seed_cells=0;
    Model(Config cfg,MPI_Comm comm=MPI_COMM_SELF):c(std::move(cfg)),domain(c.nx,c.ny,c.nz,comm,c.px,c.py,c.pz),o(c.nx,c.ny,c.nz,domain),cr(c.nx,c.ny,c.nz,domain),p(c.nx,c.ny,c.nz,domain),
      on(c.nx,c.ny,c.nz,domain),crn(c.nx,c.ny,c.nz,domain),pn(c.nx,c.ny,c.nz,domain),muO(c.nx,c.ny,c.nz,domain),muCr(c.nx,c.ny,c.nz,domain),
      mo(c.nx,c.ny,c.nz,domain),mc(c.nx,c.ny,c.nz,domain),grain(c.nx,c.ny,c.nz,domain),gb(c.nx,c.ny,c.nz,domain),
      gb_initial(c.nx,c.ny,c.nz,domain),eta_s2(c.nx,c.ny,c.nz,domain),eta_pairs(c.nx,c.ny,c.nz,domain){
        voronoi();
        gb_initial.a=gb.a;
        for(int i=domain.first;i<=domain.last;++i)for(int j=domain.jfirst;j<=domain.jlast;++j)for(int k=domain.kfirst;k<=domain.klast;++k){
            o(i,j,k)=Physics::bulkO;cr(i,j,k)=Physics::bulkCr;p(i,j,k)=0.;
            // Keep the SAME initial concentrations for on/off comparisons.
            // The legacy concentration-BC layer is a one-time initial inventory
            // when O diffusion is off; it is not subsequently replenished.
            if(c.bc=="concentration" && i==1){o(i,j,k)=Physics::layerO;cr(i,j,k)=Physics::layerCr;}
        }
        // Retain the pre-insertion inventory: the initial seed is recorded in
        // O_seed/Cr_seed, so the diagnostics budget remains balanced at t=0.
        initialO=sum(o);initialCr=sum(cr);
        if(c.nucleation=="center")initialize_center_oxide();
        if(has_eta())initialize_eta();
        refresh();
    }
    static long double sum(const Field&a){
        long double v=0,total=0;for(int i=a.domain.first;i<=a.domain.last;++i)for(int j=a.domain.jfirst;j<=a.domain.jlast;++j)for(int k=a.domain.kfirst;k<=a.domain.klast;++k)v+=a(i,j,k);
        MPI_Allreduce(&v,&total,1,MPI_LONG_DOUBLE,MPI_SUM,a.domain.comm);return total;
    }
    // [4] True 3D Voronoi grains, periodic in Y and Z, nonperiodic in X.
    // Distances below use grid units; the CLI enforces cubic cells.
    double competitor_distance(std::size_t g,int owner,double x,double y,double z,
                               double ownerY,double ownerZ,double rbest)const{
        double db=std::numeric_limits<double>::infinity();
        for(int im=-1;im<=1;++im)for(int jm=-1;jm<=1;++jm){
            const double gy=seeds[g].y+im*c.ny,gz=seeds[g].z+jm*c.nz;
            const double rs=std::pow(seeds[g].x-x,2)+std::pow(gy-y,2)+std::pow(gz-z,2);
            const double sep=std::sqrt(std::pow(seeds[g].x-seeds[owner].x,2)
                                      +std::pow(gy-ownerY,2)+std::pow(gz-ownerZ,2));
            if(sep>1.e-14)db=std::min(db,std::max(0.,(rs-rbest)/(2.*sep)));
        }
        return db;
    }
    void voronoi(){
        seeds.clear();volumes.clear();
        if(!c.grains){
            seeds.push_back({0.5*c.nx,0.5*c.ny,0.5*c.nz});
            volumes.push_back(c.nx*c.ny*c.nz);
            std::fill(grain.a.begin(),grain.a.end(),1.);
            std::fill(gb.a.begin(),gb.a.end(),0.);
            return;
        }
        const double volume=double(c.nx)*c.ny*c.nz;
        // Target diameter is volume-equivalent: pi*d^3/6 per initial grain.
        const double ng_real=6.*volume/(PI*std::pow(c.grain_diameter_grid,3));
        if(!std::isfinite(ng_real) || ng_real>volume+.5)
            throw std::runtime_error("Too many Voronoi sites for this grid.");
        const int ng=std::max(1,int(std::llround(ng_real)));
        if(ng>volume)throw std::runtime_error("Too many Voronoi sites for this grid.");
        if(has_eta() && ng>c.max_eta_fields)
            throw std::runtime_error("3D grain count exceeds --max-eta-fields; increase grain diameter or check memory before increasing the limit.");
        std::mt19937_64 rng(c.random_seed);
        std::uniform_real_distribution<double> ux(0.,double(c.nx)),uy(0.,double(c.ny)),uz(0.,double(c.nz));
        for(int g=0;g<ng;++g)seeds.push_back({ux(rng),uy(rng),uz(rng)});
        if(c.left_edge_sites<0 || c.left_edge_sites>std::min(ng,c.ny*c.nz))
            throw std::runtime_error("--left-face-sites must be between 0 and min(volume-based site count, ny*nz).");
        if(c.left_edge_sites>0){
            // Select distinct inlet cells on a 2D lattice; anchors each own an
            // inlet cell. Reposition existing sites without changing ng.
            const int n=c.left_edge_sites;
            int rows=std::clamp(int(std::llround(std::sqrt(double(n)*c.ny/c.nz))),1,std::min(n,c.ny));
            rows=std::max(rows,(n+c.nz-1)/c.nz);
            std::uniform_real_distribution<double> unit(0.,1.);
            const int phaseY=std::min(c.ny-1,int(unit(rng)*c.ny));
            const int phaseZ=std::min(c.nz-1,int(unit(rng)*c.nz));
            int g=0;
            for(int r=0;r<rows;++r){
                const int cols=n/rows+(r<n%rows?1:0);
                const int yy=(phaseY+int(static_cast<long long>(r)*c.ny/rows))%c.ny;
                for(int q=0;q<cols;++q){
                    const int zz=(phaseZ+int(static_cast<long long>(q)*c.nz/cols))%c.nz;
                    seeds[g++]={0.05+0.4*unit(rng),yy+0.5,zz+0.5};
                }
            }
            for(int g=n;g<ng;++g)seeds[g].x=1.5+(c.nx-1.5)*(seeds[g].x/c.nx);
        }
        volumes.assign(ng,0);
        for(int i=domain.first;i<=domain.last;++i)for(int j=domain.jfirst;j<=domain.jlast;++j)for(int k=domain.kfirst;k<=domain.klast;++k){
            const double x=i-.5,y=j-.5,z=k-.5;
            int owner=0;double rbest=std::numeric_limits<double>::infinity(),ownerY=0.,ownerZ=0.;
            for(int g=0;g<ng;++g){
                const double yy=minimum_image(seeds[g].y-y,c.ny),zz=minimum_image(seeds[g].z-z,c.nz);
                const double rr=std::pow(seeds[g].x-x,2)+yy*yy+zz*zz;
                if(rr<rbest){rbest=rr;owner=g;ownerY=y+yy;ownerZ=z+zz;}
            }
            grain(i,j,k)=owner+1;++volumes[owner];
            double db=std::numeric_limits<double>::infinity();
            for(int g=0;g<ng;++g)if(g!=owner)
                db=std::min(db,competitor_distance(g,owner,x,y,z,ownerY,ownerZ,rbest));
            gb(i,j,k)=ng==1?0.:std::exp(-4.*std::log(2.)*db*db/(c.gb_fwhm_grid*c.gb_fwhm_grid));
        }
        if(domain.size>1)MPI_Allreduce(MPI_IN_PLACE,volumes.data(),ng,MPI_INT,MPI_SUM,domain.comm);
        ghosts(grain);ghosts(gb);
    }
    // [5] One diffuse eta field per initial 3D metal grain.
    void initialize_eta(){
        if(seeds.size()>static_cast<std::size_t>(c.max_eta_fields))
            throw std::runtime_error("Too many eta fields; check --max-eta-fields and memory.");
        eta.reserve(seeds.size());eta_new.reserve(seeds.size());
        for(std::size_t g=0;g<seeds.size();++g){
            eta.emplace_back(c.nx,c.ny,c.nz,domain);eta_new.emplace_back(c.nx,c.ny,c.nz,domain);
        }
        const double ell_grid=eta_ell(c)/c.dx;
        std::vector<double> weights(seeds.size());
        for(int i=domain.first;i<=domain.last;++i)for(int j=domain.jfirst;j<=domain.jlast;++j)for(int k=domain.kfirst;k<=domain.klast;++k){
            const int owner=int(grain(i,j,k))-1;
            const double x=i-.5,y=j-.5,z=k-.5;
            const double oy=y+minimum_image(seeds[owner].y-y,c.ny),oz=z+minimum_image(seeds[owner].z-z,c.nz);
            const double rbest=std::pow(seeds[owner].x-x,2)+std::pow(oy-y,2)+std::pow(oz-z,2);
            double norm=0.;
            for(std::size_t g=0;g<seeds.size();++g){
                weights[g]=int(g)==owner?1.:std::exp(-competitor_distance(g,owner,x,y,z,oy,oz,rbest)/ell_grid);
                norm+=weights[g];
            }
            const double metal=1.-h(p(i,j,k));
            for(std::size_t g=0;g<eta.size();++g)eta[g](i,j,k)=metal*weights[g]/norm;
        }
    }
    void rebuild_eta_mask(){
        const double scale=eta_transport_scale(c);
        for(int i=domain.first;i<=domain.last;++i)for(int j=domain.jfirst;j<=domain.jlast;++j)for(int k=domain.kfirst;k<=domain.klast;++k){
            double S=0.,pairs=0.;
            for(const auto& e:eta){const double q=e(i,j,k)*e(i,j,k);pairs+=S*q;S+=q;}
            eta_s2(i,j,k)=S;eta_pairs(i,j,k)=pairs;
            // An overlap of TWO METAL eta fields is required. A single metal/oxide
            // interface is NOT mistaken for a substrate grain boundary.
            // Clamping is only of this diagnostic/transport indicator, never of eta.
            gb(i,j,k)=scale*std::min(1.,16.*pairs);
        }
        ghosts(gb);
    }
    double eta_force(std::size_t g,int i,int j,int k)const{
        return grain_local_deta(eta[g](i,j,k),eta_s2(i,j,k),p(i,j,k),c)-eta_K(c)*lap(eta[g],i,j,k,c);
    }
    double eta_phi_force(int i,int j,int k)const{
        return has_eta()?grain_local_dphi(eta_s2(i,j,k),p(i,j,k),c):0.;
    }
    void eta_trial(){
        for(std::size_t g=0;g<eta.size();++g)
            for(int i=domain.first;i<=domain.last;++i)for(int j=domain.jfirst;j<=domain.jlast;++j)for(int k=domain.kfirst;k<=domain.klast;++k){
                const double v=eta[g](i,j,k)-c.dt*c.eta_L*eta_force(g,i,j,k);
                if(!std::isfinite(v)||v < -1.e-12||v > 1.+1.e-12){
                    std::ostringstream msg;msg<<std::setprecision(17)
                        <<"Eta trial outside [0,1]: grain="<<g+1<<" cell=("<<i<<","<<j<<","<<k
                        <<") value="<<v<<". All PDE fields rejected together; eta is never clipped. Check eta coefficients/dt.";
                    throw std::runtime_error(msg.str());
                }
                eta_new[g](i,j,k)=v;
            }
    }
    int current_grain(int i,int j,int k)const{
        if(p(i,j,k)>=.5)return 0;
        if(!has_eta())return int(grain(i,j,k));
        double best=1.e-12;int id=0;
        for(std::size_t g=0;g<eta.size();++g)if(eta[g](i,j,k)>best){best=eta[g](i,j,k);id=int(g)+1;}
        return id;
    }
    long double grain_energy()const{
        // Forward-face gradient energy, counted once per physical edge. Its exact
        // discrete derivative is -K*lap(eta) with the same Neumann X / periodic Y/Z BCs.
        long double local=0.,total=0.;
        if(!has_eta())return 0.;
        for(int i=domain.first;i<=domain.last;++i)for(int j=domain.jfirst;j<=domain.jlast;++j)for(int k=domain.kfirst;k<=domain.klast;++k){
            double density=grain_local_energy(eta_s2(i,j,k),eta_pairs(i,j,k),p(i,j,k),c);
            for(const auto& e:eta){
                if(i<c.nx)density+=.5*eta_K(c)*std::pow((e(i+1,j,k)-e(i,j,k))/c.dx,2);
                density+=.5*eta_K(c)*std::pow((e(i,j+1,k)-e(i,j,k))/c.dy,2);
                density+=.5*eta_K(c)*std::pow((e(i,j,k+1)-e(i,j,k))/c.dz,2);
            }
            local+=density*c.dx*c.dy*c.dz;
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
        for(int i=1;i<=c.nx;++i)for(int j=1;j<=c.ny;++j)for(int k=1;k<=c.nz;++k){
            const double S=eta_s2(i,j,k),metal=1.-h(p(i,j,k));
            slo=std::min(slo,S);shi=std::max(shi,S);sumS+=S;mask+=metal*gb(i,j,k);gbmax=std::max(gbmax,gb(i,j,k));
            const int id=current_grain(i,j,k);if(id)++cells[std::size_t(id-1)];else if(p(i,j,k)<.5)++unassigned;
            for(std::size_t g=0;g<eta.size();++g){
                const double v=eta[g](i,j,k);lo=std::min(lo,v);hi=std::max(hi,v);
                if(S>1.e-24)amount[g]+=metal*v*v/S;
            }
        }
        const auto alive=std::count_if(cells.begin(),cells.end(),[](int v){return v>0;});
        f<<std::setprecision(17)<<n<<","<<n*c.dt<<","<<lo<<","<<hi<<","<<slo<<","<<shi
         <<","<<sumS/(c.nx*c.ny*c.nz)<<","<<alive<<","<<unassigned<<","<<mask*c.dx*c.dy*c.dz<<","<<gbmax<<","<<grain_energy()<<"\n";
        for(std::size_t g=0;g<eta.size();++g)
            history<<std::setprecision(17)<<n<<","<<n*c.dt<<","<<g+1<<","<<cells[g]<<","<<amount[g]
                   <<","<<amount[g]*c.dx*c.dy*c.dz<<"\n";
        if(!f||!history)throw std::runtime_error("Grain diagnostic write failed.");
    }

    // [6] Boundary supply, chemical potentials and conservative six-face transport.
    void enforce_legacy_layer(){
        if(domain.first!=1 || c.bc!="concentration" || !c.oxygen_diffusion)return;
        for(int j=domain.jfirst;j<=domain.jlast;++j)for(int k=domain.kfirst;k<=domain.klast;++k){b.bcO+=Physics::layerO-o(1,j,k);o(1,j,k)=Physics::layerO;}
    }
    void refresh(){
        ghosts(o);ghosts(cr);ghosts(p);
        if(has_eta()){
            for(auto& e:eta)ghosts(e);
            rebuild_eta_mask();
        }
        for(int i=domain.first;i<=domain.last;++i)for(int j=domain.jfirst;j<=domain.jlast;++j)for(int k=domain.kfirst;k<=domain.klast;++k){
            double hh=h(p(i,j,k));
            muO(i,j,k)=(1.-hh)*mu_metal_o(o(i,j,k),cr(i,j,k))
                +hh*(24000000.*o(i,j,k)-13800000.*cr(i,j,k)-8880000.)/Physics::Vm
                -2.*Physics::ko*lap(o,i,j,k,c);
            muCr(i,j,k)=(1.-hh)*(1166528.60789006*cr(i,j,k)+3200000.*o(i,j,k)-97045.64)/Physics::Vm
                +hh*(14000000.*cr(i,j,k)-13800000.*o(i,j,k)+2680000.)/Physics::Vm
                -2.*Physics::kc*lap(cr,i,j,k,c);
        }
        ghosts(muO);ghosts(muCr);
        // Boundary is the FACE between ghost i=0 and the first cell i=1.
        if(domain.first==1 && c.bc=="mu" && c.oxygen_diffusion)for(int j=domain.jfirst;j<=domain.jlast;++j)for(int k=domain.kfirst;k<=domain.klast;++k)muO(0,j,k)=2.*c.mu_res-muO(1,j,k);
        cells_and_faces(domain,[&](int i,int j,int k){
            mo(i,j,k)=mobility(o(i,j,k),p(i,j,k),gb(i,j,k),true,c);
            mc(i,j,k)=mobility(cr(i,j,k),p(i,j,k),gb(i,j,k),false,c);
        });
    }
    double left_flux(int j,int k)const{
        if(domain.first!=1 || c.bc!="mu" || !c.oxygen_diffusion)return 0.;
        // Explicit, positive boundary-face kinetic closure for the zero-O start.
        // surface_c_ref sets a face mobility, NOT an enforced concentration.
        // Gating by the current oxide fraction prevents a GB short-circuit in oxide.
        const double mf=mobility(c.surface_c_ref,p(1,j,k),gb(1,j,k),true,c);
        return 2.*mf*(c.mu_res-muO(1,j,k))/c.dx; // + means into the material
    }
    double div_flux(const Field&m,const Field&mu,int i,int j,int k,bool oxygen)const{
        const double right=-.5*(m(i,j,k)+m(i+1,j,k))*(mu(i+1,j,k)-mu(i,j,k))/c.dx;
        double left=-.5*(m(i-1,j,k)+m(i,j,k))*(mu(i,j,k)-mu(i-1,j,k))/c.dx;
        if(oxygen && i==1 && c.bc=="mu")left=left_flux(j,k);
        const double top=-.5*(m(i,j,k)+m(i,j+1,k))*(mu(i,j+1,k)-mu(i,j,k))/c.dy;
        const double bottom=-.5*(m(i,j-1,k)+m(i,j,k))*(mu(i,j,k)-mu(i,j-1,k))/c.dy;
        const double front=-.5*(m(i,j,k)+m(i,j,k+1))*(mu(i,j,k+1)-mu(i,j,k))/c.dz;
        const double back=-.5*(m(i,j,k-1)+m(i,j,k))*(mu(i,j,k)-mu(i,j,k-1))/c.dz;
        return (left-right)/c.dx+(bottom-top)/c.dy+(back-front)/c.dz;
    }
    // [7] Ordered global insertion, gather/scatter and spherical nuclei.
    void prepare_collective_buffers(){
        // If allocation fails on any rank, fail together BEFORE the collective.
        std::string error;
        try{
            packed_field.resize(domain.counts[domain.rank]);
            if(domain.rank==0)root_packed_fields.resize(std::size_t(c.nx)*c.ny*c.nz);
        }catch(const std::exception& e){error=e.what();}
        collective_error(domain.comm,error);
    }
    void gather_field(const Field& src,Field* dst){
        prepare_collective_buffers();
        std::size_t q=0;
        for(int i=domain.first;i<=domain.last;++i)for(int j=domain.jfirst;j<=domain.jlast;++j)for(int k=domain.kfirst;k<=domain.klast;++k)
            packed_field[q++]=src(i,j,k);
        MPI_Gatherv(packed_field.data(),domain.counts[domain.rank],MPI_DOUBLE,
                    domain.rank==0?root_packed_fields.data():nullptr,domain.counts.data(),domain.displacements.data(),
                    MPI_DOUBLE,0,domain.comm);
        if(dst)for(int r=0;r<domain.size;++r){
            const auto& b=domain.blocks[r];q=domain.displacements[r];
            for(int i=b.lo[0];i<=b.hi[0];++i)for(int j=b.lo[1];j<=b.hi[1];++j)for(int k=b.lo[2];k<=b.hi[2];++k)
                (*dst)(i,j,k)=root_packed_fields[q++];
        }
    }
    void scatter_field(Field& dst,const Field* src){
        prepare_collective_buffers();
        if(src)for(int r=0;r<domain.size;++r){
            const auto& b=domain.blocks[r];std::size_t q=domain.displacements[r];
            for(int i=b.lo[0];i<=b.hi[0];++i)for(int j=b.lo[1];j<=b.hi[1];++j)for(int k=b.lo[2];k<=b.hi[2];++k)
                root_packed_fields[q++]=(*src)(i,j,k);
        }
        MPI_Scatterv(domain.rank==0?root_packed_fields.data():nullptr,domain.counts.data(),domain.displacements.data(),MPI_DOUBLE,
                     packed_field.data(),domain.counts[domain.rank],MPI_DOUBLE,0,domain.comm);
        std::size_t q=0;
        for(int i=domain.first;i<=domain.last;++i)for(int j=domain.jfirst;j<=domain.jlast;++j)for(int k=domain.kfirst;k<=domain.klast;++k)
            dst(i,j,k)=packed_field[q++];
        // Ghost faces are rebuilt by refresh() before the next PDE evaluation.
    }
    void gather_state(){
        root_action(domain.comm,[&]{if(!snapshot)snapshot=std::make_unique<Model>(serial_config(c));});
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
    int ordered_seed(bool scan,int x=0,int y=0,int z=0){
        gather_state();int changed=0;
        root_action(domain.comm,[&]{
            snapshot->b=Budget{};
            if(scan)snapshot->nucleate();else changed=snapshot->stamp(x,y,z);
        });
        scatter_field(o,snapshot?&snapshot->o:nullptr);
        scatter_field(cr,snapshot?&snapshot->cr:nullptr);
        scatter_field(p,snapshot?&snapshot->p:nullptr);
        for(std::size_t g=0;g<eta.size();++g)scatter_field(eta[g],snapshot?&snapshot->eta[g]:nullptr);
        if(snapshot){
            // Count each event/mass increment exactly once, even across Cartesian blocks.
            b.nucO+=snapshot->b.nucO;b.nucCr+=snapshot->b.nucCr;b.nuclei+=snapshot->b.nuclei;
        }
        MPI_Bcast(&changed,1,MPI_INT,0,domain.comm);
        return changed;
    }
    bool valid_nucleus(int x,int y,int z)const{
        const double r=c.radius_grid;
        // Original eligible x-region, but NO integer truncation of the radius.
        if(x-r<5. || x+r>double(c.nx-3))return false;
        int sr=int(std::ceil(r+c.exclusion_gap_grid));
        for(int i=std::max(1,x-sr);i<=std::min(c.nx,x+sr);++i)for(int j=1;j<=c.ny;++j)for(int k=1;k<=c.nz;++k){
            if(p(i,j,k)>.01 && std::sqrt(std::pow(double(i-x),2)+std::pow(minimum_image(j-y,c.ny),2)+std::pow(minimum_image(k-z,c.nz),2))<r+c.exclusion_gap_grid)return false;
        }
        return true;
    }
    int stamp(int x,int y,int z){
        if(domain.size>1)return ordered_seed(false,x,y,z);
        int changed=0,rr=int(std::ceil(c.radius_grid));
        for(int i=std::max(1,x-rr);i<=std::min(c.nx,x+rr);++i)for(int j=1;j<=c.ny;++j)for(int k=1;k<=c.nz;++k){
            double dy=minimum_image(j-y,c.ny),dz=minimum_image(k-z,c.nz);
            if((i-x)*(i-x)+dy*dy+dz*dz<=c.radius_grid*c.radius_grid && p(i,j,k)<.01){
                b.nucO+=Physics::oxideO-o(i,j,k);b.nucCr+=Physics::oxideCr-cr(i,j,k);
                o(i,j,k)=Physics::oxideO;cr(i,j,k)=Physics::oxideCr;p(i,j,k)=1.;++changed;
                for(auto& e:eta)e(i,j,k)=0.; // Same discrete insertion: consume metal order locally.
            }
        }
        if(changed){++b.nuclei;}
        return changed;
    }
    void initialize_center_oxide(){
        // Cell centres are at (i-0.5)*dx. Fractional index coordinates keep the
        // seed at the exact box centre for ODD as well as EVEN dimensions.
        const double x=0.5*(double(c.nx)+1.), y=0.5*(double(c.ny)+1.), z=0.5*(double(c.nz)+1.);
        const double r=c.radius_grid;
        // Keep every outermost row/column of cell centres outside the seed.
        // Reject oversized spheres instead of silently cropping/wrapping them.
        if(r>=0.5*(double(std::min({c.nx,c.ny,c.nz}))-1.))
            throw std::runtime_error("Center seed is too large: --radius-grid must be less than (min(nx,ny,nz)-1)/2.");
        if(b.nuclei!=0)
            throw std::runtime_error("The center oxide has already been initialized.");
        for(int i=domain.first;i<=domain.last;++i)for(int j=domain.jfirst;j<=domain.jlast;++j)for(int k=domain.kfirst;k<=domain.klast;++k){
            const double dx=i-x, dy=j-y, dz=k-z;
            if(dx*dx+dy*dy+dz*dz<=r*r){
                // Same hard-seed composition assignment as stamp(), no smoothing.
                b.nucO+=Physics::oxideO-o(i,j,k);b.nucCr+=Physics::oxideCr-cr(i,j,k);
                o(i,j,k)=Physics::oxideO;cr(i,j,k)=Physics::oxideCr;p(i,j,k)=1.;
                ++center_seed_cells;
                for(auto& e:eta)e(i,j,k)=0.;
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
        for(int i=domain.first;i<=domain.last;++i)for(int j=domain.jfirst;j<=domain.jlast;++j)for(int k=domain.kfirst;k<=domain.klast;++k){
            if(p(i,j,k)<=.01 && o(i,j,k)*cr(i,j,k)>Physics::ksp && valid_nucleus(i,j,k))stamp(i,j,k);
        }
    }
    // [8] Explicit coupled PDE step; a rejected trial restores the inlet and budget.
    void step(){
        const Budget budget_before=b;
        std::vector<double> inlet_before;
        const bool reset_inlet=domain.first==1 && c.bc=="concentration" && c.oxygen_diffusion;
        if(reset_inlet){
            inlet_before.reserve(std::size_t(domain.local[1])*domain.local[2]);
            for(int j=domain.jfirst;j<=domain.jlast;++j)for(int k=domain.kfirst;k<=domain.klast;++k)inlet_before.push_back(o(1,j,k));
        }
        enforce_legacy_layer();refresh();
        for(int j=domain.jfirst;j<=domain.jlast;++j)for(int k=domain.kfirst;k<=domain.klast;++k)b.bcO+=c.dt*left_flux(j,k)/c.dx;
        std::string trial_error;
        try{
        for(int i=domain.first;i<=domain.last;++i)for(int j=domain.jfirst;j<=domain.jlast;++j)for(int k=domain.kfirst;k<=domain.klast;++k){
            double oo=o(i,j,k),cc=cr(i,j,k),pp=p(i,j,k);
            // O diffusion off removes the transport increment, not other
            // mechanisms that modify O (hard nucleation, optional legacy Ck).
            on(i,j,k)=c.oxygen_diffusion?oo+c.dt*div_flux(mo,muO,i,j,k,true):oo;
            crn(i,j,k)=cc+c.dt*div_flux(mc,muCr,i,j,k,false);
            const double hp=30.*pp*pp*(pp-1.)*(pp-1.);
            double fp=Physics::W*(4.*pp*pp*pp-6.*pp*pp+2.*pp)
                      +hp*(foxide(oo,cc)-fmetal(oo,cc))-2.*Physics::kp*lap(p,i,j,k,c);
            if(has_eta())fp+=eta_phi_force(i,j,k);
            pn(i,j,k)=pp-c.dt*Physics::L*fp;
            if(c.ck=="legacy"){
                double dO=-Physics::ckO*((Physics::oxideO-on(i,j,k))*h(pn(i,j,k))-(Physics::oxideO-oo)*h(pp))/c.dt;
                double dCr=-Physics::ckCr*((Physics::oxideCr-crn(i,j,k))*h(pn(i,j,k))-(Physics::oxideCr-cc)*h(pp))/c.dt;
                on(i,j,k)+=dO;crn(i,j,k)+=dCr;b.ckO+=dO;b.ckCr+=dCr;
            }
            if(!std::isfinite(on(i,j,k)) || !std::isfinite(crn(i,j,k)) || !std::isfinite(pn(i,j,k)))
                throw std::runtime_error("Nonfinite trial field: reduce dt / inspect the model; no update accepted.");
            if(c.bounds=="legacy"){
                double oc=std::clamp(on(i,j,k),0.,1.),ccr=std::clamp(crn(i,j,k),0.,1.);
                b.clipO+=oc-on(i,j,k);b.clipCr+=ccr-crn(i,j,k);
                if(oc!=on(i,j,k)||ccr!=crn(i,j,k))++b.clipped;
                on(i,j,k)=oc;crn(i,j,k)=ccr;
                if(pn(i,j,k)>.99999)pn(i,j,k)=1.;
                if(pn(i,j,k)<1.e-5)pn(i,j,k)=0.;
            }else{
                if(on(i,j,k)<-1.e-12 || crn(i,j,k)<-1.e-12 || on(i,j,k)+crn(i,j,k)>1.+1.e-12 ||
                   pn(i,j,k)<-1.e-12 || pn(i,j,k)>1.+1.e-12){
                    std::ostringstream msg;
                    msg<<std::scientific<<std::setprecision(16)<<"Out-of-bounds trial at ("<<i<<","<<j<<","<<k<<"): O="<<on(i,j,k)
                       <<", Cr="<<crn(i,j,k)<<", phi="<<pn(i,j,k)
                       <<". No clipping/update accepted. Reduce dt; if failure persists, investigate model.";
                    throw std::runtime_error(msg.str());
                }
            }
        }
        if(has_eta() && c.eta_L>0.)eta_trial();
        }catch(const std::exception& e){trial_error=e.what();}
        try{collective_error(domain.comm,trial_error);}
        catch(...){
            b=budget_before;
            if(reset_inlet){
                std::size_t q=0;
                for(int j=domain.jfirst;j<=domain.jlast;++j)for(int k=domain.kfirst;k<=domain.klast;++k)o(1,j,k)=inlet_before[q++];
            }
            throw;
        }
        o.a.swap(on.a);cr.a.swap(crn.a);p.a.swap(pn.a);
        if(c.eta_L>0.)for(std::size_t g=0;g<eta.size();++g)eta[g].a.swap(eta_new[g].a);
    }
    // [9] ParaView output: voxel-centred structured points in metres, X fastest.
    void vtk(const std::string& path){
        if(domain.size>1){gather_state();root_action(domain.comm,[&]{snapshot->vtk(path);});return;}
        refresh();
        std::ofstream f(path);if(!f)throw std::runtime_error("Cannot write "+path);
        f<<std::setprecision(17)<<"# vtk DataFile Version 3.0\nNiCr 3D oxidation and metal grains; metres\n"
         <<"ASCII\nDATASET STRUCTURED_POINTS\nDIMENSIONS "<<c.nx<<" "<<c.ny<<" "<<c.nz
         <<"\nORIGIN "<<.5*c.dx<<" "<<.5*c.dy<<" "<<.5*c.dz
         <<"\nSPACING "<<c.dx<<" "<<c.dy<<" "<<c.dz<<"\nPOINT_DATA "<<c.nx*c.ny*c.nz<<"\n";
        auto write=[&](const char* name,auto fn){
            f<<"SCALARS "<<name<<" double 1\nLOOKUP_TABLE default\n";
            for(int k=1;k<=c.nz;++k)for(int j=1;j<=c.ny;++j){
                for(int i=1;i<=c.nx;++i)f<<fn(i,j,k)<<" ";
                f<<"\n";
            }
        };
        write("phi",[&](int i,int j,int k){return p(i,j,k);});
        write("conc_O",[&](int i,int j,int k){return o(i,j,k);});
        write("conc_Cr",[&](int i,int j,int k){return cr(i,j,k);});
        write("conc_Ni",[&](int i,int j,int k){return 1.-o(i,j,k)-cr(i,j,k);});
        write("grain_id",[&](int i,int j,int k){return has_eta()?double(current_grain(i,j,k)):grain(i,j,k);});
        write("GB_mask",[&](int i,int j,int k){return gb(i,j,k);});
        write("GB_active",[&](int i,int j,int k){return (1.-h(p(i,j,k)))*gb(i,j,k);});
        write("mu_O",[&](int i,int j,int k){return muO(i,j,k);});
        write("mu_Cr",[&](int i,int j,int k){return muCr(i,j,k);});
        write("M_O",[&](int i,int j,int k){return mo(i,j,k);});
        write("M_Cr",[&](int i,int j,int k){return mc(i,j,k);});
        if(has_eta()){
            write("grain_id_initial",[&](int i,int j,int k){return grain(i,j,k);});
            write("GB_initial",[&](int i,int j,int k){return gb_initial(i,j,k);});
            write("GB_eta_indicator",[&](int i,int j,int k){return std::min(1.,16.*eta_pairs(i,j,k));});
            write("metal_fraction",[&](int i,int j,int k){return 1.-h(p(i,j,k));});
            write("eta_sum_sq",[&](int i,int j,int k){return eta_s2(i,j,k);});
            write("active_grain_id",[&](int i,int j,int k){return double(current_grain(i,j,k));});
            if(c.eta_output=="all")for(std::size_t g=0;g<eta.size();++g){
                const std::string name="eta_"+std::to_string(g+1);
                write(name.c_str(),[&](int i,int j,int k){return eta[g](i,j,k);});
            }
        }
        if(!f)throw std::runtime_error("Output write failed: "+path);
    }
    // [10] Reproducible inputs, 3D grain statistics and model diagnostics.
    void metadata(){
        if(domain.rank!=0)return;
        std::ofstream f(c.out+"/parameters.txt");
        f<<std::setprecision(17)<<"model=NiCr_3D_oxidation_metal_grains\nversion="<<VERSION
         <<"\nnx="<<c.nx<<"\nny="<<c.ny<<"\nnz="<<c.nz
         <<"\ndx_m="<<c.dx<<"\ndy_m="<<c.dy<<"\ndz_m="<<c.dz<<"\ndt_s="<<c.dt
         <<"\nsteps="<<c.steps<<"\noutput_every="<<c.output_every
         <<"\noxygen_bc="<<c.bc<<"\nCk="<<c.ck<<"\nbounds="<<c.bounds
         <<"\nmpi_processes="<<domain.size<<"\nmpi_decomposition=3D_Cartesian_blocks"
         <<"\nmpi_px="<<domain.dims[0]<<"\nmpi_py="<<domain.dims[1]<<"\nmpi_pz="<<domain.dims[2]
         <<"\nmpi_px_requested="<<c.px<<"\nmpi_py_requested="<<c.py<<"\nmpi_pz_requested="<<c.pz
         <<"\nmpi_halo=one_cell_six_faces_no_edges_or_corners"
         <<"\nmpi_rank_reordering=off"
         <<"\ntransverse_bc=periodic_Y_and_Z\nright_bc=no_flux"
         <<"\nnucleation_mode="<<c.nucleation<<"\ninitial_center_seed_cells="<<center_seed_cells
         <<"\ngrain_structure="<<(c.grains?"on":"off")
         <<"\ngrain_evolution_requested="<<(c.grain_evolution?"on":"off")
         <<"\ngrain_evolution_active="<<(has_eta()?"on":"off")
         <<"\noxygen_diffusion="<<(c.oxygen_diffusion?"on":"off")
         <<"\noxygen_boundary_supply="<<(c.oxygen_diffusion?c.bc:"disabled")
         <<"\nVm="<<Physics::Vm<<"\nW="<<Physics::W<<"\nkappa_phi="<<Physics::kp<<"\nL_phi="<<Physics::L
         <<"\nkappa_O="<<Physics::ko<<"\nkappa_Cr="<<Physics::kc
         <<"\nmatrix_mobility_prefactor_O="<<Physics::bmo<<"\nmatrix_mobility_prefactor_Cr="<<Physics::bmc
         <<"\noxide_mobility_prefactor_O="<<Physics::boo<<"\noxide_mobility_prefactor_Cr="<<Physics::boc
         <<"\nbulk_O="<<Physics::bulkO<<"\nbulk_Cr="<<Physics::bulkCr
         <<"\nlayer_O="<<Physics::layerO<<"\nlayer_Cr="<<Physics::layerCr
         <<"\noxide_O="<<Physics::oxideO<<"\noxide_Cr="<<Physics::oxideCr
         <<"\nCk_O="<<Physics::ckO<<"\nCk_Cr="<<Physics::ckCr<<"\nKsp="<<Physics::ksp
         <<"\nmu_res_model_units="<<c.mu_res<<"\nsurface_mobility_concentration_reference="<<c.surface_c_ref
         <<"\nleft_face_sites_requested="<<c.left_edge_sites<<"\nleft_face_sites_active="<<(c.grains?c.left_edge_sites:0)
         <<"\ngrain_diameter_target_grid="<<c.grain_diameter_grid
         <<"\ngrain_diameter_target_m="<<c.grain_diameter_grid*c.dx
         <<"\ngrain_count_rule=round(6*nx*ny*nz/(pi*d_grid^3)),minimum_1"
         <<"\nvoronoi_sites="<<seeds.size()<<"\nrandom_seed="<<c.random_seed
         <<"\ngb_mask_FWHM_grid="<<c.gb_fwhm_grid<<"\ngb_mask_FWHM_m="<<c.gb_fwhm_grid*c.dx
         <<"\ngb_mask_integral_effective_width_m="<<reference_gb_width(c)
         <<"\ngb_peak_mobility_factor_O="<<c.gb_factor_o<<"\ngb_peak_mobility_factor_Cr="<<c.gb_factor_cr
         <<"\nnucleus_shape=sphere\nnucleus_geometric_radius_grid="<<c.radius_grid
         <<"\nnucleus_geometric_radius_m="<<c.radius_grid*c.dx
         <<"\nexclusion_gap_grid="<<c.exclusion_gap_grid<<"\nnucleation_period_s="<<c.nucleation_period
         <<"\neta_output="<<c.eta_output<<"\nmax_eta_fields="<<c.max_eta_fields<<"\n";
#ifdef NICR_MPI_THREAD_TEST
        f<<"execution_backend=thread_rank_emulation_NOT_native_MPI\n";
#elif defined(NICR_SERIAL_VERIFY)
        f<<"execution_backend=single_process_verification_NOT_native_MPI\n";
#else
        f<<"execution_backend=native_MPI\n";
#endif
        const double bytes=8.*(15.+2.*eta.size())*(domain.local[0]+2.)*(domain.local[1]+2.)*(domain.local[2]+2.);
        const double full=8.*(15.+2.*eta.size())*(c.nx+2.)*(c.ny+2.)*(c.nz+2.);
        f<<"rank0_field_storage_estimate_MiB="<<bytes/(1024.*1024.)
         <<"\nroot_additional_snapshot_field_storage_MiB="<<(domain.size>1?full/(1024.*1024.):0.)
         <<"\nroot_additional_gather_buffer_MiB="<<(domain.size>1?8.*c.nx*c.ny*c.nz/(1024.*1024.):0.)
         <<"\nrank0_additional_pack_buffer_MiB="<<(domain.size>1?8.*domain.counts[0]/(1024.*1024.):0.)<<"\n";
        std::ofstream layout(c.out+"/mpi_layout.csv");
        layout<<"rank,coord_x,coord_y,coord_z,i_first,i_last,j_first,j_last,k_first,k_last,local_nx,local_ny,local_nz,owned_cells\n";
        for(int r=0;r<domain.size;++r){
            const auto& b=domain.blocks[r];
            layout<<r<<","<<b.coords[0]<<","<<b.coords[1]<<","<<b.coords[2]
                  <<","<<b.lo[0]<<","<<b.hi[0]<<","<<b.lo[1]<<","<<b.hi[1]<<","<<b.lo[2]<<","<<b.hi[2]
                  <<","<<b.n[0]<<","<<b.n[1]<<","<<b.n[2]<<","<<domain.counts[r]<<"\n";
        }
        if(!layout)throw std::runtime_error("MPI layout write failed.");
        if(has_eta()){
            f<<"eta_fields="<<eta.size()<<"\neta_W="<<c.eta_W<<"\neta_kappa="<<eta_K(c)
             <<"\neta_L="<<c.eta_L<<"\neta_oxide_penalty="<<eta_A(c)
             <<"\neta_width_10_90_grid="<<2.*std::log(9.)*eta_ell(c)/c.dx
             <<"\neta_mask_integrated_width_m="<<8.*eta_ell(c)/3.
             <<"\neta_transport_scale="<<eta_transport_scale(c)
             <<"\neta_planar_GB_energy_model_units="<<std::sqrt(2.*eta_K(c)*c.eta_W)/3.
             <<"\neta_planar_geometric_mobility_model_units="<<3.*c.eta_L*eta_ell(c)<<"\n";
        }
        if(c.nucleation=="center")f<<"initial_oxide_center_x_m="<<0.5*c.nx*c.dx
            <<"\ninitial_oxide_center_y_m="<<0.5*c.ny*c.dy<<"\ninitial_oxide_center_z_m="<<0.5*c.nz*c.dz
            <<"\ninitial_oxide_center_i="<<0.5*(c.nx+1.)<<"\ninitial_oxide_center_j="<<0.5*(c.ny+1.)
            <<"\ninitial_oxide_center_k="<<0.5*(c.nz+1.)<<"\n";
        f<<"mobility_note=prefactors_retained_from_uploaded_18wtpct_setup;local_M_includes_concentration_phase_and_GB_factors\n"
         <<"calibration_note=eta_and_GB_transport_parameters_require_material_calibration\n";
        std::ofstream g(c.out+"/grains.csv");
        g<<"grain_id,seed_x_grid,seed_y_grid,seed_z_grid,volume_cells,volume_m3,equivalent_diameter_grid,equivalent_diameter_m\n";
        long double mean=0.;
        for(std::size_t q=0;q<volumes.size();++q){
            const double d=std::cbrt(6.*volumes[q]/PI);mean+=d;
            g<<std::setprecision(17)<<q+1<<","<<seeds[q].x<<","<<seeds[q].y<<","<<seeds[q].z
             <<","<<volumes[q]<<","<<volumes[q]*c.dx*c.dy*c.dz<<","<<d<<","<<d*c.dx<<"\n";
        }
        mean/=volumes.size();f<<"measured_arithmetic_mean_equivalent_diameter_grid="<<mean<<"\n";
        if(!f||!g)throw std::runtime_error("Metadata write failed.");
    }
    void log(std::ostream&f,int n){
        if(domain.size>1){gather_state();root_action(domain.comm,[&]{snapshot->log(f,n);});return;}
        long double so=sum(o),sc=sum(cr),po=sum(p),sumH=0.;
        long double ro=so-initialO-b.bcO-b.nucO-b.ckO-b.clipO;
        long double rc=sc-initialCr-b.bcCr-b.nucCr-b.ckCr-b.clipCr;
        double omin=1.,omax=0.,crmin=1.,crmax=0.,nimin=1.,phimin=1.,phimax=0.;
        for(int i=domain.first;i<=domain.last;++i)for(int j=domain.jfirst;j<=domain.jlast;++j)for(int k=domain.kfirst;k<=domain.klast;++k){
            sumH+=h(p(i,j,k));
            omin=std::min(omin,o(i,j,k));omax=std::max(omax,o(i,j,k));crmin=std::min(crmin,cr(i,j,k));crmax=std::max(crmax,cr(i,j,k));
            nimin=std::min(nimin,1.-o(i,j,k)-cr(i,j,k));phimin=std::min(phimin,p(i,j,k));phimax=std::max(phimax,p(i,j,k));}
        f<<std::setprecision(17)<<n<<","<<n*c.dt<<","<<so<<","<<sc<<","<<po/(c.nx*c.ny*c.nz)
         <<","<<b.bcO<<","<<b.nucO<<","<<b.nucCr<<","<<b.ckO<<","<<b.ckCr<<","<<b.clipO<<","<<b.clipCr
         <<","<<ro<<","<<rc<<","<<b.nuclei<<","<<b.clipped<<","<<omin<<","<<omax<<","<<crmin<<","<<crmax
         <<","<<nimin<<","<<phimin<<","<<phimax<<","<<sumH/(c.nx*c.ny*c.nz)<<","<<sumH*c.dx*c.dy*c.dz<<"\n";
    }
};
// [11] Command-line input and validation.
static bool parse_on_off(const std::string& value,const std::string& option){
    if(value=="on")return true;
    if(value=="off")return false;
    throw std::runtime_error(option+" expects on or off (received: "+value+")");
}
struct EarlyExit {};
static Config parse(int argc,char**argv){
    int rank=0;MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    Config c;
    for(int g=1;g<argc;++g){std::string a=argv[g];
        if(a=="--self-test"){c.self_test=true;continue;}
        if(a=="--version"){if(rank==0)std::cout<<VERSION<<"\n";throw EarlyExit{};}
        if(a=="--help"){
            if(rank==0)std::cout<<"Initialization only by default. Options:\n--steps N --dt s --nx N --ny N --nz N --dx m (sets dx=dy=dz) --output-every N\n"
              <<"--px N --py N --pz N (0/omitted: automatic; positive: fix an axis; product = MPI ranks)\n"
              <<"--grains on|off (default on; off: uniform single crystal, no GB enhancement)\n"
              <<"--grain-evolution on|off (default off; on: moving eta grains, dynamic GB transport and oxide coupling)\n"
              <<"--eta-W value --eta-kappa value (0: auto width) --eta-L value --eta-oxide-penalty value (-1: eta-W)\n"
              <<"--eta-output all|summary --max-eta-fields N (default 512; memory guard)\n"
              <<"--oxygen-diffusion on|off (default on; off: no O transport or ongoing O boundary supply)\n"
              <<"--left-face-sites N (alias --left-edge-sites; relocate N existing sites over the YZ inlet)\n"
              <<"--grain-diameter-grid N --gb-width-grid N (mask FWHM)\n"
              <<"--gb-factor-o N --gb-factor-cr N [--gb-factor N sets both; compatibility alias] --seed N\n"
              <<"--nucleation ksp|center|off (default ksp; center: ONE initial seed; off: transport/grain growth only)\n"
              <<"--radius-grid R (radius in grid spacings) --nucleation-period s (Ksp mode only)\n"
              <<"--bc concentration|mu|closed --mu-res value --surface-c-ref value\n--ck legacy|off --bounds stop|legacy --out directory --self-test --version\n"
              <<"Change --seed to change the Voronoi structure; repeat it with the same 3D geometry/build to reproduce it.\n"
              <<"O diffusion off does NOT disable hard-seed concentration assignments or legacy Ck.\n"
              <<"With concentration BC, the initial left O/Cr layer is retained but O is not replenished when off.\n";
            throw EarlyExit{};
        }
        if(g+1>=argc)throw std::runtime_error("Missing value for "+a);
        std::string v=argv[++g];
        if(a=="--steps")c.steps=std::stoi(v);else if(a=="--dt")c.dt=std::stod(v);
        else if(a=="--nx")c.nx=std::stoi(v);else if(a=="--ny")c.ny=std::stoi(v);
        else if(a=="--nz")c.nz=std::stoi(v);
        else if(a=="--px"||a=="--py"||a=="--pz"){
            std::size_t used=0;const int n=std::stoi(v,&used);
            if(used!=v.size()||n<0)throw std::runtime_error(a+" expects a nonnegative integer (0: automatic).");
            if(a=="--px")c.px=n;else if(a=="--py")c.py=n;else c.pz=n;
        }
        else if(a=="--dx")c.dx=c.dy=c.dz=std::stod(v);else if(a=="--output-every")c.output_every=std::stoi(v);
        else if(a=="--grains")c.grains=parse_on_off(v,a);
        else if(a=="--grain-evolution")c.grain_evolution=parse_on_off(v,a);
        else if(a=="--eta-W")c.eta_W=std::stod(v);
        else if(a=="--eta-kappa")c.eta_kappa=std::stod(v);
        else if(a=="--eta-L")c.eta_L=std::stod(v);
        else if(a=="--eta-oxide-penalty")c.eta_oxide_penalty=std::stod(v);
        else if(a=="--eta-output")c.eta_output=v;
        else if(a=="--max-eta-fields")c.max_eta_fields=std::stoi(v);
        else if(a=="--oxygen-diffusion")c.oxygen_diffusion=parse_on_off(v,a);
        else if((a=="--left-edge-sites"||a=="--left-face-sites")){std::size_t used=0;c.left_edge_sites=std::stoi(v,&used);if(used!=v.size()||c.left_edge_sites<0)throw std::runtime_error("--left-edge-sites expects a nonnegative integer.");}
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
    for(double x:{c.dx,c.dy,c.dz,c.dt,c.grain_diameter_grid,c.gb_fwhm_grid,c.gb_factor_o,c.gb_factor_cr,c.radius_grid,c.nucleation_period,c.mu_res,c.surface_c_ref})
        if(!std::isfinite(x))throw std::runtime_error("All numerical parameters must be finite.");
    if(c.nx<3||c.ny<3||c.nz<3||c.nx>INT_MAX-2||c.ny>INT_MAX-2||c.nz>INT_MAX-2||c.steps<0||c.output_every<1||c.dt<=0||c.dx<=0||c.dy<=0||c.dz<=0||c.grain_diameter_grid<=0||c.gb_fwhm_grid<=0||c.gb_factor_o<1||c.gb_factor_cr<1||c.radius_grid<=0||(c.nucleation=="ksp"&&c.nucleation_period<c.dt)||c.nucleation_period<=0.||c.surface_c_ref<=0)
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
    if(c.nucleation=="center"&&c.radius_grid>=0.5*(double(std::min({c.nx,c.ny,c.nz}))-1.))
        throw std::runtime_error("Center seed is too large: --radius-grid must be less than (min(nx,ny,nz)-1)/2.");
    if(c.nucleation=="ksp" && c.radius_grid>=0.5*std::min(c.ny,c.nz))
        throw std::runtime_error("Ksp sphere radius must be less than min(ny,nz)/2 to avoid periodic self-overlap.");
    if(c.bc!="mu"&&c.bc!="concentration"&&c.bc!="closed")throw std::runtime_error("Invalid bc.");
    if(c.ck!="legacy"&&c.ck!="off")throw std::runtime_error("Invalid Ck mode.");
    if(c.bounds!="stop"&&c.bounds!="legacy")throw std::runtime_error("Invalid bounds mode.");
    int size=1;MPI_Comm_size(MPI_COMM_WORLD,&size);
    choose_process_grid(size,{{c.nx,c.ny,c.nz}},{{c.px,c.py,c.pz}});
    return c;
}
// [12] Built-in verification. These are small numerical tests, not calibration.
static void require(bool v,const char* msg){if(!v)throw std::runtime_error(std::string("Self-test failed: ")+msg);}
static bool near(long double a,long double b,long double atol=1.e-11L,long double rtol=1.e-11L){
    return std::abs(a-b)<=atol+rtol*std::max(std::abs(a),std::abs(b));
}
static Config test_config(){
    Config c;c.nx=23;c.ny=13;c.nz=11;c.grain_diameter_grid=12.;c.bc="closed";
    c.ck="off";c.nucleation="off";c.dt=1.e-12;c.radius_grid=2.5;return c;
}
static void fill_smooth(Model& m){
    for(int i=m.domain.first;i<=m.domain.last;++i)for(int j=m.domain.jfirst;j<=m.domain.jlast;++j)for(int k=m.domain.kfirst;k<=m.domain.klast;++k){
        m.o(i,j,k)=.001*(1.+.08*std::sin(.3*i+.5*j-.4*k));
        m.cr(i,j,k)=.19+.001*std::cos(.2*i-.3*j+.6*k);
        m.p(i,j,k)=.1+.01*std::cos(.1*i+.4*j-.2*k);
        for(std::size_t g=0;g<m.eta.size();++g)m.eta[g](i,j,k)=.2+.02*std::cos(.2*i-.3*j+.7*g+.5*k);
    }
    m.refresh();
}
static void self_test(){
    Config c=test_config();
    require(choose_process_grid(16,{{61,61,61}},{{4,2,2}})==std::array<int,3>{{4,2,2}},"explicit process grid");
    const auto partial=choose_process_grid(16,{{61,61,61}},{{0,2,0}});
    require(partial[1]==2&&partial[0]*partial[1]*partial[2]==16,"partially fixed process grid");
    const auto skinny=choose_process_grid(16,{{3,17,15}},{{0,0,0}});
    require(skinny[0]<=3&&skinny[1]<=17&&skinny[2]<=15&&skinny[0]*skinny[1]*skinny[2]==16,"automatic thin-mesh fallback");
    for(const auto& bad:std::vector<std::array<int,3>>{{{2,2,1}},{{3,0,0}},{{-1,0,0}},{{62,0,0}}}){
        bool refused=false;try{choose_process_grid(16,{{61,61,61}},bad);}catch(const std::runtime_error&){refused=true;}
        require(refused,"invalid process-grid request rejected");
    }
    bool impossible=false;try{choose_process_grid(32,{{3,3,3}},{{0,0,0}});}catch(const std::runtime_error&){impossible=true;}
    require(impossible,"no nonempty Cartesian factorization rejected");
    // All three Laplacian directions and transverse periodic faces.
    Model s(c);Field a=s.o;
    for(int i=1;i<=c.nx;++i)for(int j=1;j<=c.ny;++j)for(int k=1;k<=c.nz;++k)
        a(i,j,k)=i*i*c.dx*c.dx+2.*j*j*c.dy*c.dy+3.*k*k*c.dz*c.dz;
    ghosts(a);require(near(lap(a,5,5,5,c),12.,1.e-10L),"7-point Laplacian polynomial in X,Y,Z");
    require(a(0,3,4)==a(1,3,4)&&a(c.nx+1,3,4)==a(c.nx,3,4),"Neumann X ghosts");
    require(a(3,0,4)==a(3,c.ny,4)&&a(3,4,0)==a(3,4,c.nz),"periodic Y and Z ghosts");
    for(int axis=0;axis<3;++axis){
        const int n=axis==0?c.nx:(axis==1?c.ny:c.nz);
        const double lambda=-4.*std::pow(std::sin(PI/n),2)/(c.dx*c.dx);
        for(int i=1;i<=c.nx;++i)for(int j=1;j<=c.ny;++j)for(int k=1;k<=c.nz;++k){
            const int index=axis==0?i:(axis==1?j:k);a(i,j,k)=std::sin(2.*PI*(index-.5)/n);
        }
        ghosts(a);
        for(int i=2;i<c.nx;++i)for(int j=1;j<=c.ny;++j)for(int k=1;k<=c.nz;++k)
            require(near(lap(a,i,j,k,c),lambda*a(i,j,k),50.,2.e-13L),"Fourier Laplacian eigenvalue including periodic Z seam");
    }
    std::cout<<"PASS: 3D Laplacian; X no-flux; Y/Z periodic faces (edges/corners are unused).\n";
    // Volume-equivalent grain count, true volumetric structure, anchors and limits.
    Model repeat(c);require(s.grain.a==repeat.grain.a&&s.gb.a==repeat.gb.a,"3D seed reproducibility");
    const int ng=std::max(1,int(std::llround(6.*c.nx*c.ny*c.nz/(PI*std::pow(c.grain_diameter_grid,3)))));
    require(int(s.seeds.size())==ng,"3D volume-based grain count");
    long long count=0;for(int n:s.volumes)count+=n;require(count==c.nx*c.ny*c.nz,"grain volumes cover box");
    bool differsZ=false;for(int i=1;i<=c.nx;++i)for(int j=1;j<=c.ny;++j)for(int k=2;k<=c.nz;++k){
        require(s.gb(i,j,k)>=0.&&s.gb(i,j,k)<=1.,"bounded fixed GB mask");
        differsZ=differsZ||s.grain(i,j,k)!=s.grain(i,j,1);
    }
    require(differsZ,"3D grains are not an extruded 2D map");
    Config changed=c;++changed.random_seed;Model other(changed);require(other.grain.a!=s.grain.a,"different seed changes 3D geometry");
    for(int seed=0;seed<4;++seed){
        Config t=c;t.random_seed=seed;t.left_edge_sites=ng;Model m(t);std::vector<bool> seen(ng,false);
        for(int j=1;j<=t.ny;++j)for(int k=1;k<=t.nz;++k)seen[int(m.grain(1,j,k))-1]=true;
        for(bool v:seen)require(v,"anchored sites each own inlet-face cells");
    }
    Config t=c;t.grains=false;Model single(t);
    require(single.volumes[0]==t.nx*t.ny*t.nz&&Model::sum(single.gb)==0.,"3D single-crystal volume and mask");
    t=c;t.grain_diameter_grid=10000.;Model periodic_one(t);require(Model::sum(periodic_one.gb)==0.,"own periodic images do not make a GB");
    t=c;t.left_edge_sites=ng+1;bool rejected=false;try{Model bad(t);}catch(const std::runtime_error&){rejected=true;}
    require(rejected,"excess anchors rejected");
    t=c;t.grain_evolution=true;t.max_eta_fields=1;rejected=false;try{Model bad(t);}catch(const std::runtime_error&){rejected=true;}
    require(rejected,"eta count guard enforced");
    std::cout<<"PASS: 3D Voronoi volume rule, reproducibility, non-extruded grains, inlet anchors and memory guard.\n";
    // Constant-mobility manufactured divergence -div(-M grad(mu)) = -? Here
    // mu=sum coordinate^2 gives div(M grad mu)=6*M in all interior cells.
    Field mm=s.mo,mu=s.muO;std::fill(mm.a.begin(),mm.a.end(),2.);
    for(int i=1;i<=c.nx;++i)for(int j=1;j<=c.ny;++j)for(int k=1;k<=c.nz;++k)
        mu(i,j,k)=i*i*c.dx*c.dx+j*j*c.dy*c.dy+k*k*c.dz*c.dz;
    ghosts(mu);require(near(s.div_flux(mm,mu,5,5,5,false),12.,1.e-10L),"3D six-face flux divergence");
    fill_smooth(s);
    long double netO=0.,netCr=0.,absO=0.,absCr=0.;
    for(int i=1;i<=c.nx;++i)for(int j=1;j<=c.ny;++j)for(int k=1;k<=c.nz;++k){
        const double fo=s.div_flux(s.mo,s.muO,i,j,k,true),fc=s.div_flux(s.mc,s.muCr,i,j,k,false);
        netO+=fo;netCr+=fc;absO+=std::abs(fo);absCr+=std::abs(fc);
    }
    require(std::abs(netO)<1.e-12L*(1.+absO)&&std::abs(netCr)<1.e-12L*(1.+absCr),"closed 3D fluxes telescope");
    long double so=Model::sum(s.o),sc=Model::sum(s.cr);s.step();
    require(near(Model::sum(s.o),so)&&near(Model::sum(s.cr),sc),"closed 3D transport conserves O and Cr");
    require(near(mobility(.1,0.,1.,true,c)/mobility(.1,0.,0.,true,c),c.gb_factor_o),"O GB mobility factor");
    require(near(mobility(.1,0.,1.,false,c)/mobility(.1,0.,0.,false,c),c.gb_factor_cr),"Cr GB mobility factor");
    require(mobility(.1,1.,1.,true,c)==mobility(.1,1.,0.,true,c),"oxide gates metal GB transport");
    for(const std::string boundary:{"mu","closed","concentration"}){
        t=c;t.bc=boundary;t.oxygen_diffusion=false;Model m(t);fill_smooth(m);const auto old=m.o.a;
        for(int n=0;n<3;++n)m.step();
        m.refresh();
        require(m.o.a==old&&m.b.bcO==0.,"O off freezes O transport/inlet on every BC");
    }
    t=c;t.bc="mu";Model dry(t);dry.step();
    require(Model::sum(dry.o)>0.&&near(Model::sum(dry.o)-dry.initialO,dry.b.bcO),"mu inlet face inventory balance");
    t=c;t.bc="concentration";Model layer(t);layer.step();
    require(near(Model::sum(layer.o)-layer.initialO,layer.b.bcO),"concentration inlet inventory balance");
    std::cout<<"PASS: six-face conservative flux, O/Cr budgets, mobility gates and oxygen switch.\n";
    // Centered spheres on odd/even boxes, exact radius, no reseeding.
    for(int nx:{17,18})for(int ny:{15,16})for(int nz:{13,14}){
        t=c;t.nx=nx;t.ny=ny;t.nz=nz;t.grains=false;t.nucleation="center";t.radius_grid=3.5;
        Model m(t);int expected=0;long double xsum=0,ysum=0,zsum=0;
        const double cx=.5*(nx+1.),cy=.5*(ny+1.),cz=.5*(nz+1.);
        for(int i=1;i<=nx;++i)for(int j=1;j<=ny;++j)for(int k=1;k<=nz;++k){
            const bool inside=std::pow(i-cx,2)+std::pow(j-cy,2)+std::pow(k-cz,2)<=t.radius_grid*t.radius_grid;
            require(m.p(i,j,k)==double(inside),"center sphere has exact voxel geometry");
            require(m.p(i,j,k)==m.p(nx+1-i,j,k)&&m.p(i,j,k)==m.p(i,ny+1-j,k)&&m.p(i,j,k)==m.p(i,j,nz+1-k),"3D sphere reflection symmetry");
            if(inside){++expected;xsum+=i;ysum+=j;zsum+=k;}
        }
        require(m.center_seed_cells==expected&&m.b.nuclei==1,"one center sphere event");
        require(xsum/expected==cx&&ysum/expected==cy&&zsum/expected==cz,"exact 3D box center");
        require(near(Model::sum(m.o)-m.initialO,m.b.nucO)&&near(Model::sum(m.cr)-m.initialCr,m.b.nucCr),"spherical insertion budgets");
        const auto before=m.p.a;m.nucleate();require(m.p.a==before&&m.b.nuclei==1,"center never triggers Ksp reseeding");
    }
    t=c;t.nx=31;t.ny=25;t.nz=23;t.grains=false;t.radius_grid=5.5;Model sphere(t);
    const int voxels=sphere.stamp(16,13,12);require(voxels==739,"radius 5.5 sphere contains 739 voxels");
    t=c;t.grains=false;t.radius_grid=2.5;Model wrap(t);wrap.stamp(12,1,1);
    require(wrap.p(12,t.ny,t.nz)==1.&&wrap.p(12,2,2)==1.,"sphere wraps both periodic seams");
    require(!wrap.valid_nucleus(12,t.ny,t.nz),"3D exclusion sees opposite Y/Z faces");
    t.nucleation="ksp";Model ksp(t);for(int i=1;i<=t.nx;++i)for(int j=1;j<=t.ny;++j)for(int k=1;k<=t.nz;++k)ksp.o(i,j,k)=.001;
    ksp.nucleate();require(ksp.b.nuclei>0,"Ksp hard-seed scan operates in 3D");
    std::cout<<"PASS: odd/even spherical seeds, 739-voxel R=5.5, Y/Z seam wrapping, exclusion and Ksp.\n";
    // Eta initialization, freezing and exact discrete free-energy variations.
    t=c;t.grain_evolution=true;Model eta(t);
    for(int i=1;i<=t.nx;++i)for(int j=1;j<=t.ny;++j)for(int k=1;k<=t.nz;++k){
        double total=0.;for(const auto& e:eta.eta)total+=e(i,j,k);
        require(near(total,1.,1.e-14L,1.e-14L),"3D eta initialization partitions unity");
    }
    require(near(eta_transport_scale(t),1.),"eta auto width preserves integrated GB convention");
    t.eta_L=0.;Model frozen(t);const auto oldeta=frozen.eta;
    for(int n=0;n<5;++n)frozen.step();
    frozen.refresh();
    for(std::size_t g=0;g<oldeta.size();++g)require(oldeta[g].a==frozen.eta[g].a,"eta-L=0 freezes eta PDE");
    frozen.stamp(12,1,1);for(const auto& e:frozen.eta)require(e(12,1,1)==0.,"hard seeds consume eta even with eta-L=0");
    t=c;t.nx=9;t.ny=7;t.nz=5;t.dx=t.dy=t.dz=1.;t.grain_evolution=true;t.grain_diameter_grid=6.;
    t.eta_W=2.;t.eta_kappa=.3;t.eta_L=.1;t.dt=1.e-3;Model energy(t);fill_smooth(energy);
    const double eps=1.e-6;
    for(const auto& cell:std::vector<std::vector<int>>{{4,3,2},{1,1,1},{9,7,5}}){
        const int i=cell[0],j=cell[1],k=cell[2];
        const double e0=energy.eta[0](i,j,k),force=energy.eta_force(0,i,j,k);
        energy.eta[0](i,j,k)=e0+eps;energy.refresh();const auto plus=energy.grain_energy();
        energy.eta[0](i,j,k)=e0-eps;energy.refresh();const auto minus=energy.grain_energy();
        energy.eta[0](i,j,k)=e0;energy.refresh();
        require(near((plus-minus)/(2.*eps),force,2.e-7L,2.e-7L),"3D discrete eta energy derivative including boundary faces");
        const double p0=energy.p(i,j,k),pf=energy.eta_phi_force(i,j,k);
        energy.p(i,j,k)=p0+eps;energy.refresh();const auto pp=energy.grain_energy();
        energy.p(i,j,k)=p0-eps;energy.refresh();const auto pm=energy.grain_energy();
        energy.p(i,j,k)=p0;energy.refresh();
        require(near((pp-pm)/(2.*eps),pf,2.e-7L,2.e-7L),"reciprocal phi/grain energy derivative");
    }
    for(int i=1;i<=t.nx;++i)for(int j=1;j<=t.ny;++j)for(int k=1;k<=t.nz;++k){energy.o(i,j,k)=0.;energy.cr(i,j,k)=Physics::bulkCr;energy.p(i,j,k)=0.;}
    energy.refresh();auto previous=energy.grain_energy();
    for(int n=0;n<30;++n){energy.step();energy.refresh();const auto now=energy.grain_energy();require(now<=previous+1.e-12L,"closed grain-only energy is nonincreasing");previous=now;}
    std::cout<<"PASS: 3D eta initialization, frozen eta, seed consumption, variational derivatives and energy descent.\n";
}
// Manufactured global-coordinate field: check every face against its exact
// global neighbor, independent of any serial solver or gather/scatter routine.
static void cartesian_halo_test(const Config& c){
    Domain d(c.nx,c.ny,c.nz,MPI_COMM_WORLD,c.px,c.py,c.pz);
    Field a(c.nx,c.ny,c.nz,d,std::numeric_limits<double>::quiet_NaN());
    auto value=[&](int i,int j,int k){
        i=std::clamp(i,1,c.nx);
        j=(j-1+c.ny)%c.ny+1;k=(k-1+c.nz)%c.nz+1;
        return 1000000.*i+1000.*j+k;
    };
    for(int i=d.first;i<=d.last;++i)for(int j=d.jfirst;j<=d.jlast;++j)for(int k=d.kfirst;k<=d.klast;++k)
        a(i,j,k)=value(i,j,k);
    ghosts(a);bool ok=true;
    cells_and_faces(d,[&](int i,int j,int k){ok=ok&&a(i,j,k)==value(i,j,k);});
    ok=ok&&std::isnan(a(d.first-1,d.jfirst-1,d.kfirst-1));
    collective_error(d.comm,ok?"":"Manufactured Cartesian halo mismatch, or unused corner was overwritten.");
    long long count=static_cast<long long>(d.local[0])*d.local[1]*d.local[2],total=0;
    MPI_Allreduce(&count,&total,1,MPI_LONG_LONG_INT,MPI_SUM,d.comm);
    require(total==static_cast<long long>(c.nx)*c.ny*c.nz,"Cartesian blocks cover exactly the global cell count");
}
static void mpi_compare(Model& distributed,Model& serial,const char* label){
    distributed.refresh();serial.refresh();bool same=distributed.eta.size()==serial.eta.size();
    const Field* a[]={&distributed.o,&distributed.cr,&distributed.p,&distributed.grain,&distributed.gb,&distributed.muO,&distributed.muCr,&distributed.mo,&distributed.mc};
    const Field* b[]={&serial.o,&serial.cr,&serial.p,&serial.grain,&serial.gb,&serial.muO,&serial.muCr,&serial.mo,&serial.mc};
    // Interior indices in the serial reference are global; transverse ghost
    // indices are wrapped. Mu's physical X ghosts include the inlet treatment.
    cells_and_faces(distributed.domain,[&](int i,int j,int k){
        const int jj=(j-1+serial.c.ny)%serial.c.ny+1,kk=(k-1+serial.c.nz)%serial.c.nz+1;
        for(int q=0;q<9;++q)same=same&&(*a[q])(i,j,k)==(*b[q])(i,jj,kk);
        for(std::size_t g=0;g<distributed.eta.size();++g)same=same&&distributed.eta[g](i,j,k)==serial.eta[g](i,jj,kk);
    });
    collective_error(distributed.domain.comm,same?"":std::string("3D MPI field/face mismatch: ")+label);
    distributed.gather_state();root_action(distributed.domain.comm,[&]{
        distributed.snapshot->refresh();
        const auto& got=distributed.snapshot->b;const auto& ref=serial.b;
        const long double ga[]={got.nucO,got.nucCr,got.ckO,got.ckCr,got.clipO,got.clipCr,got.bcO,got.bcCr};
        const long double ra[]={ref.nucO,ref.nucCr,ref.ckO,ref.ckCr,ref.clipO,ref.clipCr,ref.bcO,ref.bcCr};
        for(int q=0;q<8;++q)require(near(ga[q],ra[q]),"MPI source/boundary budget reductions");
        require(got.nuclei==ref.nuclei&&got.clipped==ref.clipped,"MPI event count reductions");
        const Field* gathered[]={&distributed.snapshot->o,&distributed.snapshot->cr,&distributed.snapshot->p,
            &distributed.snapshot->grain,&distributed.snapshot->gb,&distributed.snapshot->muO,&distributed.snapshot->muCr,
            &distributed.snapshot->mo,&distributed.snapshot->mc};
        cells_and_faces(serial.domain,[&](int i,int j,int k){
            for(int q=0;q<9;++q)require((*gathered[q])(i,j,k)==(*b[q])(i,j,k),"global block unpacking and snapshot fields");
            for(std::size_t g=0;g<serial.eta.size();++g)
                require(distributed.snapshot->eta[g](i,j,k)==serial.eta[g](i,j,k),"global eta block unpacking");
        });
    });
}
static void mpi_self_test(const Config& request){
    int size=1,rank=0;MPI_Comm_size(MPI_COMM_WORLD,&size);MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    const auto dims=choose_process_grid(size,{{request.nx,request.ny,request.nz}},{{request.px,request.py,request.pz}});
    auto apply=[&](Config c){
        c.px=dims[0];c.py=dims[1];c.pz=dims[2];
        c.nx=std::max(c.nx,2*dims[0]+1);c.ny=std::max(c.ny,2*dims[1]+1);c.nz=std::max(c.nz,2*dims[2]+1);
        return c;
    };
    cartesian_halo_test(apply(test_config()));
    Config thin_cfg=apply(test_config());
    thin_cfg.nx=std::max(3,dims[0]);thin_cfg.ny=std::max(3,dims[1]);thin_cfg.nz=std::max(3,dims[2]);
    thin_cfg.grain_diameter_grid=4.;thin_cfg.nucleation="off";thin_cfg.bounds="legacy";
    cartesian_halo_test(thin_cfg);
    for(bool dynamic:{false,true})for(const std::string bc:{"mu","closed","concentration"})for(bool oxygen:{false,true}){
        Config c=apply(test_config());c.grain_evolution=dynamic;c.left_edge_sites=2;
        c.bc=bc;c.oxygen_diffusion=oxygen;c.nucleation="center";c.bounds="legacy";
        c.ck=oxygen?"off":"legacy";
        Model m(c,MPI_COMM_WORLD),s(serial_config(c));mpi_compare(m,s,"3D geometry and spherical initialization");
        fill_smooth(m);fill_smooth(s);
        for(int n=0;n<5;++n){m.step();s.step();}
        mpi_compare(m,s,"3D coupled equations, all BCs and O switch");
    }
    for(bool dynamic:{false,true}){
        Config c=apply(test_config());c.nx=std::max(31,c.nx);c.grain_evolution=dynamic;c.nucleation="ksp";c.bounds="legacy";
        Model m(c,MPI_COMM_WORLD),s(serial_config(c));
        const int x=std::max(9,std::min(c.nx-8,dims[0]>1?m.domain.blocks[0].hi[0]:15));
        require(m.stamp(x,1,1)==s.stamp(x,1,1),"MPI sphere count across blocks and periodic seams");
        mpi_compare(m,s,"hard sphere/eta scatter across rank and two periodic seams");
        const int yy=dims[1]>1?m.domain.blocks[0].hi[1]:c.ny/2;
        const int zz=dims[2]>1?m.domain.blocks[0].hi[2]:c.nz/2;
        require(m.stamp(x,yy,zz)==s.stamp(x,yy,zz),"sphere crossing an internal Y/Z block junction");
        mpi_compare(m,s,"seed crossing multiple internal block faces");
        for(int i=m.domain.first;i<=m.domain.last;++i)for(int j=m.domain.jfirst;j<=m.domain.jlast;++j)for(int k=m.domain.kfirst;k<=m.domain.klast;++k)m.o(i,j,k)=.001;
        for(int i=1;i<=c.nx;++i)for(int j=1;j<=c.ny;++j)for(int k=1;k<=c.nz;++k)s.o(i,j,k)=.001;
        m.nucleate();s.nucleate();mpi_compare(m,s,"ordered 3D Ksp scan and exclusion");
        for(int n=0;n<3;++n){m.step();s.step();}mpi_compare(m,s,"post-nucleation 3D PDEs");
        // Also test frozen eta with scatter, not just PDE-only freezing.
        if(dynamic){
            m.c.eta_L=s.c.eta_L=0.;
            for(int n=0;n<3;++n){m.step();s.step();}
            mpi_compare(m,s,"frozen eta on Cartesian blocks");
        }
        Config t=thin_cfg;t.grain_evolution=dynamic;
        Model thin(t,MPI_COMM_WORLD),reference(serial_config(t));fill_smooth(thin);fill_smooth(reference);
        for(int n=0;n<3;++n){thin.step();reference.step();}mpi_compare(thin,reference,"thin blocks, including one-cell local dimensions");
        thin.c.bounds="stop";thin.c.bc="concentration";
        if(rank==size-1){
            if(dynamic)thin.eta[0](thin.domain.last,thin.domain.jlast,thin.domain.klast)=2.;
            else thin.cr(thin.domain.last,thin.domain.jlast,thin.domain.klast)=-.1;
        }
        const auto oldO=thin.o.a;const Budget oldB=thin.b;bool rejected=false;
        try{thin.step();}catch(const std::runtime_error&){rejected=true;}
        collective_error(MPI_COMM_WORLD,rejected?"":"Trial failure not propagated to every rank");
        require(thin.b.bcO==oldB.bcO&&thin.b.ckO==oldB.ckO&&thin.b.clipO==oldB.clipO,"rejected step restores mass budget");
        for(int i=thin.domain.first;i<=thin.domain.last;++i)for(int j=thin.domain.jfirst;j<=thin.domain.jlast;++j)for(int k=thin.domain.kfirst;k<=thin.domain.klast;++k)
            require(thin.o(i,j,k)==oldO[thin.o.index(i,j,k)],"rejected update leaves owned O unchanged");
    }
    if(rank==0)std::cout<<"PASS: exact Cartesian/serial fields and six faces; ordered block gather/scatter; fixed/evolving/frozen eta; all BCs; O switch; Ck; spherical/Ksp seeds; uneven/thin blocks; collective rejection. Grid "
        <<dims[0]<<" x "<<dims[1]<<" x "<<dims[2]<<".\n";
}
// [13] Execution, output scheduling and error reporting.
static int run(int argc,char**argv){
    int rank=0;MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    try{
        Config c=parse(argc,argv);if(c.self_test){root_action(MPI_COMM_WORLD,[]{
#ifdef NICR_SERIAL_VERIFY
            std::cout<<"SINGLE-PROCESS VERIFICATION ONLY; not a native MPI test.\n";
#endif
            self_test();});mpi_self_test(c);return 0;}
        root_action(MPI_COMM_WORLD,[&]{
        if(fs::exists(c.out) && !fs::is_empty(c.out))throw std::runtime_error("Output directory is not empty; choose a new --out.");
        fs::create_directories(c.out);
        });
        Model m(c,MPI_COMM_WORLD);root_action(MPI_COMM_WORLD,[&]{m.metadata();});
        std::ofstream log,grain_diag,grain_history;
        root_action(MPI_COMM_WORLD,[&]{
        log.open(c.out+"/diagnostics.csv");if(!log)throw std::runtime_error("Cannot open diagnostics.csv");
        log<<"step,time_s,sum_O,sum_Cr,mean_phi,O_boundary,O_seed,Cr_seed,O_Ck,Cr_Ck,O_clipping,Cr_clipping,O_budget_residual,Cr_budget_residual,nuclei,clipped_cells,min_O,max_O,min_Cr,max_Cr,min_Ni,min_phi,max_phi,mean_h_phi,oxide_volume_m3\n";
        if(m.has_eta()){
            grain_diag.open(c.out+"/grain_diagnostics.csv");grain_history.open(c.out+"/grain_history.csv");
            if(!grain_diag||!grain_history)throw std::runtime_error("Cannot open grain diagnostics.");
            grain_diag<<"step,time_s,min_eta,max_eta,min_sum_eta2,max_sum_eta2,mean_sum_eta2,grains_with_metal_cells,unassigned_metal_cells,integrated_active_GB_m3,max_GB_mask,grain_energy_model_units\n";
            grain_history<<"step,time_s,grain_id,dominant_metal_cells,weighted_metal_volume_cells,weighted_metal_volume_m3\n";
        }
        });
        m.log(log,0);m.grain_log(grain_diag,grain_history,0);m.vtk(c.out+"/fields_0.vtk");
        if(rank==0){
#if defined(NICR_SERIAL_VERIFY) || defined(NICR_MPI_THREAD_TEST)
        std::cout<<"EXECUTION BACKEND: verification shim, NOT native MPI.\n";
#endif
        std::cout<<"MPI processes: "<<m.domain.size<<"; Cartesian grid "
                 <<m.domain.dims[0]<<" x "<<m.domain.dims[1]<<" x "<<m.domain.dims[2]
                 <<" (X x Y x Z; six-face halos)\n";
        std::cout<<"Version "<<VERSION<<"; grains="<<(c.grains?"on":"off")
                  <<"; oxygen_diffusion="<<(c.oxygen_diffusion?"on":"off")
                  <<"; nucleation="<<c.nucleation<<"; "<<m.seeds.size()<<" grain(s).\n"
                 <<"Box "<<c.nx<<" x "<<c.ny<<" x "<<c.nz<<" cells; configured GB factors O="<<c.gb_factor_o<<", Cr="<<c.gb_factor_cr<<".\n";
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
            <<0.5*c.nx*c.dx*1.e6<<", "<<0.5*c.ny*c.dy*1.e6<<", "<<0.5*c.nz*c.dz*1.e6<<") um; radius="
            <<c.radius_grid<<" grids; "<<m.center_seed_cells<<" cells. Additional Ksp insertion is OFF.\n";
        if(!c.oxygen_diffusion)std::cout<<"O diffusion and continuing O boundary supply are OFF; hard-seed assignments and optional Ck are unchanged.\n";
        if(c.ck=="legacy")std::cout<<"WARNING: legacy Ck is timestep dependent and nonconservative; retained only for comparison.\n";
        if(c.bc=="mu"&&c.oxygen_diffusion)std::cout<<"WARNING: experimental mu BC uses a finite positive surface mobility closure (see README).\n";
        if(c.grains&&((c.oxygen_diffusion&&c.gb_factor_o>1)||c.gb_factor_cr>1)&&c.steps>0)std::cout<<"WARNING: 3D adds a third transport/curvature direction; check timestep convergence before production.\n";
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
    catch(const std::exception&e){
        // A failing non-root rank may reach MPI_Abort first. Print its diagnostic
        // before aborting, rather than relying on rank zero to be scheduled first.
        std::cerr<<"ERROR [rank "<<rank<<"]: "<<e.what()<<"\n"<<std::flush;
        return 1;
    }
}

int main(int argc,char**argv){
    MPI_Init(&argc,&argv);
    const int result=run(argc,argv);
    // A rank-local allocation/runtime failure must not leave peers blocked in MPI.
    if(result!=0){MPI_Abort(MPI_COMM_WORLD,result);return result;}
    MPI_Finalize();return result;
}
