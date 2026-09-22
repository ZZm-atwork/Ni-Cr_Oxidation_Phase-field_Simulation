#ifndef NICR_SERIAL_MPI_VERIFY_H
#define NICR_SERIAL_MPI_VERIFY_H
// Single-process API shim for numerical verification on hosts without MPI.
// This deliberately does NOT emulate interprocess communication or validate MPI.
// Production builds must omit NICR_SERIAL_VERIFY and use a real MPI installation.
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <iostream>
using MPI_Comm=int;using MPI_Datatype=int;using MPI_Op=int;using MPI_Status=int;
constexpr int MPI_COMM_WORLD=0,MPI_COMM_SELF=1,MPI_PROC_NULL=-1;
constexpr int MPI_INT=1,MPI_DOUBLE=2,MPI_LONG_DOUBLE=3,MPI_LONG_LONG_INT=4,MPI_CHAR=5;
constexpr int MPI_MIN=1,MPI_MAX=2,MPI_SUM=3;
static int mpi_in_place_object;
#define MPI_IN_PLACE (&mpi_in_place_object)
#define MPI_STATUS_IGNORE nullptr
inline std::size_t mpi_verify_size(int d){
    switch(d){case MPI_INT:return sizeof(int);case MPI_DOUBLE:return sizeof(double);
    case MPI_LONG_DOUBLE:return sizeof(long double);case MPI_LONG_LONG_INT:return sizeof(long long);
    case MPI_CHAR:return sizeof(char);default:throw std::runtime_error("Unsupported serial verification MPI type");}
}
inline int MPI_Init(int*,char***){
    for(const char* key:{"OMPI_COMM_WORLD_SIZE","PMI_SIZE","PMIX_UNIV_SIZE"}){
        const char* v=std::getenv(key);if(v&&std::atoi(v)>1){std::cerr<<"Serial verification binary cannot run under multiple MPI ranks.\n";std::exit(2);}
    }
    return 0;
}
inline int MPI_Finalize(){return 0;}
inline int MPI_Comm_rank(MPI_Comm,int* p){*p=0;return 0;}
inline int MPI_Comm_size(MPI_Comm,int* p){*p=1;return 0;}
inline int MPI_Bcast(void*,int,MPI_Datatype,int,MPI_Comm){return 0;}
inline int MPI_Allreduce(const void* s,void* d,int n,MPI_Datatype t,MPI_Op,MPI_Comm){if(s!=MPI_IN_PLACE)std::memcpy(d,s,std::size_t(n)*mpi_verify_size(t));return 0;}
inline int MPI_Reduce(const void* s,void* d,int n,MPI_Datatype t,MPI_Op op,int,MPI_Comm c){return MPI_Allreduce(s,d,n,t,op,c);}
inline int MPI_Sendrecv(const void*,int,MPI_Datatype,int to,int,void*,int,MPI_Datatype,int from,int,MPI_Comm,MPI_Status*){
    if(to!=MPI_PROC_NULL||from!=MPI_PROC_NULL)throw std::runtime_error("Serial verification cannot communicate between ranks");
    return 0;
}
inline int MPI_Gatherv(const void* s,int n,MPI_Datatype t,void* d,const int*,const int* offsets,MPI_Datatype,int,MPI_Comm){
    std::memcpy(static_cast<char*>(d)+std::size_t(offsets[0])*mpi_verify_size(t),s,std::size_t(n)*mpi_verify_size(t));return 0;
}
inline int MPI_Scatterv(const void* s,const int*,const int* offsets,MPI_Datatype t,void* d,int n,MPI_Datatype,int,MPI_Comm){
    std::memcpy(d,static_cast<const char*>(s)+std::size_t(offsets[0])*mpi_verify_size(t),std::size_t(n)*mpi_verify_size(t));return 0;
}
[[noreturn]] inline int MPI_Abort(MPI_Comm,int code){std::exit(code);}
#endif
