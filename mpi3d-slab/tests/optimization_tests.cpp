#define main solver_main_for_tests
#include "../NiCr3D_grains.cpp"
#undef main
#include <chrono>
// Deliberately retain the original full-plane predicates as independent oracles.
static bool reference_valid(const Model& m,int x,int y,int z){
    const double r=m.c.radius_grid;
    if(x-r<5. || x+r>double(m.c.nx-3))return false;
    const int sr=int(std::ceil(r+m.c.exclusion_gap_grid));
    for(int i=std::max(1,x-sr);i<=std::min(m.c.nx,x+sr);++i)for(int j=1;j<=m.c.ny;++j)for(int k=1;k<=m.c.nz;++k)
        if(m.p(i,j,k)>.01 && std::sqrt(std::pow(double(i-x),2)+std::pow(minimum_image(j-y,m.c.ny),2)+std::pow(minimum_image(k-z,m.c.nz),2))<r+m.c.exclusion_gap_grid)return false;
    return true;
}
static int reference_stamp(Model& m,int x,int y,int z){
    int changed=0,rr=int(std::ceil(m.c.radius_grid));
    for(int i=std::max(1,x-rr);i<=std::min(m.c.nx,x+rr);++i)for(int j=1;j<=m.c.ny;++j)for(int k=1;k<=m.c.nz;++k){
        double dy=minimum_image(j-y,m.c.ny),dz=minimum_image(k-z,m.c.nz);
        if((i-x)*(i-x)+dy*dy+dz*dz<=m.c.radius_grid*m.c.radius_grid && m.p(i,j,k)<.01){
            m.b.nucO+=Physics::oxideO-m.o(i,j,k);m.b.nucCr+=Physics::oxideCr-m.cr(i,j,k);
            m.o(i,j,k)=Physics::oxideO;m.cr(i,j,k)=Physics::oxideCr;m.p(i,j,k)=1.;++changed;
            for(auto& e:m.eta)e(i,j,k)=0.;
        }
    }
    if(changed){++m.b.nuclei;m.invalidate_eta_cache();}return changed;
}
static void checks(){
    std::uint64_t intervals=0,predicates=0,stamps=0;
    for(int n=3;n<=35;++n)for(int center=1;center<=n;++center)
    for(double radius:{.1,1.,1.5,2.,5.5,10.,40.,1.e50}){
        const auto a=periodic_neighborhood(center,radius,n);std::vector<int> got,ref;
        for(int r=0;r<a.count;++r)for(int j=a.lo[r];j<=a.hi[r];++j)got.push_back(j);
        for(int j=1;j<=n;++j)if(std::abs(minimum_image(j-center,n))<=std::ceil(radius))ref.push_back(j);
        require(got==ref,"sorted periodic interval coverage without duplicates");++intervals;
    }
    std::mt19937_64 rng(7813);
    for(int ny:{3,10,13})for(int nz:{3,9,14}){
        Config c=test_config();c.nx=35;c.ny=ny;c.nz=nz;c.grains=false;
        Model m(c);
        for(int i=1;i<=c.nx;++i)for(int j=1;j<=ny;++j)for(int k=1;k<=nz;++k){
            const int v=int(rng()%150);m.p(i,j,k)=v==0?1.:(v==1?std::nextafter(.01,1.):(v<20?.01:0.));
        }
        for(double radius:{.5,1.,std::sqrt(2.),std::nextafter(std::sqrt(2.),0.),2.5,5.5,6.5})
        for(double gap:{0.,.5,6.,30.}){
            m.c.radius_grid=radius;m.c.exclusion_gap_grid=gap;
            for(int q=0;q<200;++q){
                const int x=1+int(rng()%35),y=1+int(rng()%ny),z=1+int(rng()%nz);
                require(m.valid_nucleus(x,y,z)==reference_valid(m,x,y,z),"bounded exclusion matches full-plane oracle");++predicates;
            }
        }
        for(bool eta:{false,true})for(double radius:{.5,1.,std::sqrt(2.),2.5,5.5,8.}){
            c.grains=eta;c.grain_evolution=eta;c.eta_L=0.;c.grain_diameter_grid=8.;c.radius_grid=radius;
            Model a(c),b(c);fill_smooth(a);fill_smooth(b);
            for(int i=1;i<=35;++i)for(int j=1;j<=ny;++j)for(int k=1;k<=nz;++k){
                const double v=((i+3*j+k)%17==0)?.01:0.;a.p(i,j,k)=b.p(i,j,k)=v;
            }
            for(int q=0;q<4;++q){
                const int x=12+q,y=q==0?1:(q==1?ny:1+int(rng()%ny)),z=q==0?1:(q==1?nz:1+int(rng()%nz));
                require(a.stamp(x,y,z)==reference_stamp(b,x,y,z),"stamp voxel count");
                require(a.o.a==b.o.a && a.cr.a==b.cr.a && a.p.a==b.p.a,"stamp state and exact order");
                require(a.b.nucO==b.b.nucO && a.b.nucCr==b.b.nucCr && a.b.nuclei==b.b.nuclei,"exact stamp budgets");
                for(std::size_t g=0;g<a.eta.size();++g)require(a.eta[g].a==b.eta[g].a,"eta consumption matches");
                a.refresh();b.refresh();require(a.gb.a==b.gb.a,"cached GB after stamp");++stamps;
            }
        }
    }
    std::cout<<"PASS: "<<intervals<<" interval cases, "<<predicates<<" exact exclusion decisions, "<<stamps<<" sphere insertions and budgets.\n";
}
static double median(std::vector<double> a){std::sort(a.begin(),a.end());return a[a.size()/2];}
template<class Fn> static double timed(Fn fn){auto t=std::chrono::steady_clock::now();fn();return std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count();}
static void benchmark_io(const std::string& dir){
    fs::create_directories(dir);Config c=test_config();c.nx=151;c.ny=101;c.nz=101;c.grain_diameter_grid=85.;
    c.grain_evolution=true;c.eta_L=0.;Model m(c);fill_smooth(m);
    // Smooth synthetic fields, not a saved production state. Each format writes
    // precisely the same seven arrays and includes the existing refresh cost.
    std::cout<<"format,bytes,median_seconds,trial1,trial2,trial3\n";
    for(const std::string fmt:{"vtk","float64","float32"}){
        m.c.output_precision=fmt=="float64"?"float64":"float32";
        const std::string path=dir+"/io_"+fmt+(fmt=="vtk"?".vtk":".vti");std::vector<double> times;
        for(int q=0;q<3;++q)times.push_back(timed([&]{if(fmt=="vtk")m.vtk(path);else m.vti(path);}));
        std::cout<<fmt<<","<<fs::file_size(path)<<","<<median(times);for(double t:times)std::cout<<","<<t;std::cout<<"\n";
    }
}
static void benchmark_search(){
    Config c=test_config();c.nx=151;c.ny=c.nz=101;c.grains=false;c.radius_grid=5.5;c.exclusion_gap_grid=6.;Model m(c);
    // Occupied oxide voxels, with many candidates far from oxide; no early
    // exit in those regions. This measures valid_nucleus, not the entire run.
    for(int i=20;i<145;i+=25)for(int j=7;j<101;j+=23)for(int k=9;k<101;k+=21)m.p(i,j,k)=1.;
    std::mt19937 rng(9981);std::vector<std::array<int,3>> xyz;
    for(int n=0;n<2000;++n)xyz.push_back({11+int(rng()%128),1+int(rng()%101),1+int(rng()%101)});
    std::vector<double> oldt,newt;std::size_t count1=0,count2=0;
    for(int repeat=0;repeat<3;++repeat){
        oldt.push_back(timed([&]{count1=0;for(auto p:xyz)count1+=reference_valid(m,p[0],p[1],p[2]);}));
        newt.push_back(timed([&]{count2=0;for(auto p:xyz)count2+=m.valid_nucleus(p[0],p[1],p[2]);}));
        require(count1==count2,"benchmark exclusion decisions agree");
    }
    std::cout<<"routine,candidates,accepted,median_seconds\nfull_planes,"<<xyz.size()<<","<<count1<<","<<median(oldt)
             <<"\nbounded,"<<xyz.size()<<","<<count2<<","<<median(newt)<<"\n";
}
static void benchmark_cache(){
    Config c=test_config();c.nx=61;c.ny=41;c.nz=31;c.grain_diameter_grid=22.;c.grain_evolution=true;c.eta_L=0.;c.bounds="legacy";
    Model cached(c);c.frozen_eta_cache=false;Model uncached(c);fill_smooth(cached);fill_smooth(uncached);
    std::vector<double> oldt,newt;
    for(int r=0;r<3;++r){
        oldt.push_back(timed([&]{for(int n=0;n<100;++n)uncached.step();}));
        newt.push_back(timed([&]{for(int n=0;n<100;++n)cached.step();}));
        require(cached.o.a==uncached.o.a && cached.cr.a==uncached.cr.a && cached.p.a==uncached.p.a,"cached vs uncached PDE state");
    }
    std::cout<<"mode,steps,grains,median_seconds\nuncached,100,"<<cached.eta.size()<<","<<median(oldt)
             <<"\ncached,100,"<<cached.eta.size()<<","<<median(newt)<<"\n";
}
int main(int argc,char** argv){
    MPI_Init(&argc,&argv);
    try{
        if(argc<2 || std::string(argv[1])=="check")checks();
        else if(std::string(argv[1])=="io" && argc>2)benchmark_io(argv[2]);
        else if(std::string(argv[1])=="search")benchmark_search();
        else if(std::string(argv[1])=="cache")benchmark_cache();
        else throw std::runtime_error("Usage: optimization_tests check|search|cache|io DIRECTORY");
        MPI_Finalize();return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<"\n";MPI_Abort(MPI_COMM_WORLD,1);return 1;}
}
