#define main solver_main_for_writer_tests
#include "../NiCr3D_grains.cpp"
#undef main
int main(int argc,char** argv){
    MPI_Init(&argc,&argv);
    try{
        if(argc!=2)throw std::runtime_error("Usage: writer_tests OUTPUT_DIRECTORY");
        const std::string dir=argv[1];fs::create_directories(dir);int count=0;
        for(const auto shape:std::array<std::array<int,3>,3>{{{3,4,5},{16,32,32},{17,32,33}}})
        for(const std::string precision:{"float32","float64"})for(int level:{0,1,6,9}){
            Config c;c.nx=shape[0];c.ny=shape[1];c.nz=shape[2];c.output_precision=precision;c.compression_level=level;
            const std::string stem=dir+"/writer_"+std::to_string(count++);
            CompressedVtiWriter out(stem+".vti",c);std::ofstream raw(stem+".raw",std::ios::binary);
            for(int field=0;field<7;++field){
                auto value=[&](int i,int j,int k){
                    const int q=i+c.nx*(j+c.ny*k)+field;
                    if(q%31==0)return 1.e-100;
                    if(q%29==0)return 1.e-42;
                    return (q%3?-1.:1.)*.03125*(i+17*j+157*k+19*field);
                };
                out.array(value);
                for(int k=1;k<=c.nz;++k)for(int j=1;j<=c.ny;++j)for(int i=1;i<=c.nx;++i){
                    if(precision=="float32"){float v=static_cast<float>(value(i,j,k));raw.write(reinterpret_cast<const char*>(&v),4);}
                    else{double v=value(i,j,k);raw.write(reinterpret_cast<const char*>(&v),8);}
                }
            }
            out.finish();
        }
        Config c;c.nx=c.ny=c.nz=3;const std::string p=dir+"/overflow.vti";bool failed=false;
        try{CompressedVtiWriter w(p,c);w.array([](int,int,int){return 1.e100;});}
        catch(const std::runtime_error&){failed=true;}
        require(failed && !fs::exists(p) && !fs::exists(p+".tmp"),"Float32 overflow fails without a partial final file");
        std::cout<<"PASS: "<<count<<" VTI fixtures; sub-block, exact-block and multi-block payloads; Float32/Float64; zlib levels 0/1/6/9; atomic failure cleanup.\n";
        MPI_Finalize();return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<"\n";MPI_Abort(MPI_COMM_WORLD,1);return 1;}
}
