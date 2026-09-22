#define NICR_MPI_THREAD_TEST
#define main nicr_main_under_test
#include "../NiCr3D_grains.cpp"
#undef main
int main(int argc,char** argv){
    if(argc<2){std::cerr<<"Usage: emulated_mpi_test RANKS [solver arguments]\n";return 1;}
    const int np=std::stoi(argv[1]);if(np<1||np>32)return 1;
    std::cout<<"THREAD EMULATION: "<<np<<" ranks in one address space; NOT a real MPI runtime test.\n";
    return mpi_threads::run(np,[&]{return nicr_main_under_test(argc-1,argv+1);});
}
