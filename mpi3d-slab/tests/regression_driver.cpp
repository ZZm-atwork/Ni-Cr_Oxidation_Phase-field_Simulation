// Compile with NICR_REFERENCE to use the exact pre-optimization reference source.
#define main solver_main_for_regression
#ifdef NICR_REFERENCE
#include "reference/NiCr3D_grains.cpp"
#else
#include "../NiCr3D_grains.cpp"
#endif
#undef main

static void invalidate(Model& m){
#ifndef NICR_REFERENCE
    m.invalidate_eta_cache();
#else
    (void)m;
#endif
}
static void snapshot_state(Model& m,const std::string& path){
    if(m.domain.size>1)m.gather_state();
    root_action(m.domain.comm,[&]{
        Model& s=m.domain.size>1?*m.snapshot:m;s.refresh();
        std::ofstream f(path,std::ios::binary);
        const auto write=[&](const Field& a){
            if(a.a.empty())return;
            for(int i=1;i<=s.c.nx;++i)for(int j=1;j<=s.c.ny;++j)for(int k=1;k<=s.c.nz;++k){
                const double v=a(i,j,k);f.write(reinterpret_cast<const char*>(&v),sizeof(v));
            }
        };
        for(const Field* a:{&s.o,&s.cr,&s.p,&s.muO,&s.muCr,&s.mo,&s.mc,&s.gb,&s.eta_s2})write(*a);
        for(const auto& e:s.eta)write(e);
        if(!f)throw std::runtime_error("State dump failed");
    });
}
static int exercise(int argc,char** argv){
    MPI_Init(&argc,&argv);
    try{
        if(argc<3)throw std::runtime_error("Usage: regression_driver CASE OUT [uncached]");
        const int id=std::stoi(argv[1]);const std::string out=argv[2];
        const int mode=id/12,bc=(id/4)%3;const bool oxygen=(id/2)%2,ck=id%2;
        Config c=test_config();c.nx=25;c.ny=15;c.nz=13;c.grain_diameter_grid=12.;c.bounds="legacy";
        c.grains=mode!=0;c.grain_evolution=mode>=2;c.eta_L=mode==3?0.:.2;
        c.bc=bc==0?"closed":(bc==1?"mu":"concentration");c.oxygen_diffusion=oxygen;
        c.ck=ck?"legacy":"off";c.nucleation="ksp";c.out=out;
#ifndef NICR_REFERENCE
        c.frozen_eta_cache=argc<4;c.output_precision=id%2?"float64":"float32";
#endif
        root_action(MPI_COMM_WORLD,[&]{fs::create_directories(out);});
        Model m(c,MPI_COMM_WORLD);fill_smooth(m);
        for(int i=m.domain.first;i<=m.domain.last;++i)for(int j=1;j<=c.ny;++j)for(int k=1;k<=c.nz;++k)m.p(i,j,k)=0.;
        invalidate(m);m.refresh();snapshot_state(m,out+"/initial.bin");
        m.stamp(12,1,1); // across the X-slab boundary and both periodic seams
        for(int n=0;n<24;++n){
            if(n==3 || n==14)m.nucleate();
            if(n==10 && m.has_eta()){
                // A change on only one rank must still invalidate all ranks.
                if(m.domain.first<=12 && m.domain.last>=12)m.eta[0](12,7,6)*=.95;
                invalidate(m);
            }
            m.step();
#ifndef NICR_REFERENCE
            if(m.has_eta() && c.eta_L==0. && n%4==0){
                m.refresh();const auto gb=m.gb.a,s2=m.eta_s2.a;
                invalidate(m);m.refresh();
                require(gb==m.gb.a && s2==m.eta_s2.a,"frozen cache equals forced rebuild after mixed events");
            }
#endif
        }
        m.refresh();snapshot_state(m,out+"/final.bin");
        std::ofstream log,gd,gh;
        root_action(m.domain.comm,[&]{log.open(out+"/diagnostics.csv");gd.open(out+"/grain_diagnostics.csv");gh.open(out+"/grain_history.csv");});
        m.log(log,24);m.grain_log(gd,gh,24);m.vtk(out+"/fields.vtk");
#ifndef NICR_REFERENCE
        m.vti(out+"/fields.vti");
#endif
        MPI_Finalize();return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<"\n";MPI_Abort(MPI_COMM_WORLD,1);return 1;}
}
int main(int argc,char** argv){
#ifdef NICR_MPI_THREAD_TEST
    if(argc<2)return 1;const int np=std::stoi(argv[1]);
    return mpi_threads::run(np,[&]{return exercise(argc-1,argv+1);});
#else
    return exercise(argc,argv);
#endif
}
