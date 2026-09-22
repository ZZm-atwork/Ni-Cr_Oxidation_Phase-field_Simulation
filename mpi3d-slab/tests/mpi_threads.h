#ifndef NICR_MPI_THREADS_VERIFY_H
#define NICR_MPI_THREADS_VERIFY_H
// Deterministic, thread-based verification of the MPI subset used by this solver.
// NOT an MPI library. This checks collective ordering and halo/gather/scatter logic
// in one address space; it cannot validate a vendor MPI ABI, launcher or network.
#include <algorithm>
#include <condition_variable>
#include <cstring>
#include <cstdlib>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <vector>
using MPI_Comm=int;using MPI_Datatype=int;using MPI_Op=int;using MPI_Status=int;
constexpr int MPI_COMM_WORLD=0,MPI_COMM_SELF=1,MPI_PROC_NULL=-1;
constexpr int MPI_INT=1,MPI_DOUBLE=2,MPI_LONG_DOUBLE=3,MPI_LONG_LONG_INT=4,MPI_CHAR=5;
constexpr int MPI_MIN=1,MPI_MAX=2,MPI_SUM=3;
inline int mpi_place_obj;
#define MPI_IN_PLACE (&mpi_place_obj)
#define MPI_STATUS_IGNORE nullptr
namespace mpi_threads {
inline thread_local int rank=0;
inline int ranks=1;
using Bytes=std::vector<char>;
inline std::size_t bytesize(int t){
    switch(t){case MPI_INT:return sizeof(int);case MPI_DOUBLE:return sizeof(double);
    case MPI_LONG_DOUBLE:return sizeof(long double);case MPI_LONG_LONG_INT:return sizeof(long long);
    case MPI_CHAR:return sizeof(char);default:throw std::runtime_error("Unknown emulated datatype");}
}
inline Bytes copy(const void* p,std::size_t n){Bytes v(n);if(n)std::memcpy(v.data(),p,n);return v;}
struct Slot {Bytes bytes;std::vector<int> counts,offsets;};
struct Round {
    std::string signature;int arrived=0;bool ready=false;
    std::vector<Slot> in;std::vector<Bytes> out;
    explicit Round(std::string s):signature(std::move(s)),in(ranks),out(ranks){}
};
inline std::mutex mutex;
inline std::condition_variable cv;
inline std::shared_ptr<Round> round;
inline std::map<std::tuple<int,int,int>,std::deque<Bytes>> queues;
inline Bytes collective(const std::string& sig,Slot slot,const std::function<void(Round&)>& finish){
    std::unique_lock<std::mutex> lock(mutex);
    if(!round)round=std::make_shared<Round>(sig);
    auto current=round;
    if(current->signature!=sig)throw std::runtime_error("Emulated MPI collective mismatch: "+sig+" vs "+current->signature);
    current->in[rank]=std::move(slot);
    if(++current->arrived==ranks){finish(*current);current->ready=true;round.reset();cv.notify_all();}
    else cv.wait(lock,[&]{return current->ready;});
    return current->out[rank];
}
template<class T> inline Bytes reduce(const std::vector<Slot>& inputs,int n,int op){
    Bytes out=inputs[0].bytes;
    for(std::size_t r=1;r<inputs.size();++r)for(int i=0;i<n;++i){
        T a,b;std::memcpy(&a,out.data()+i*sizeof(T),sizeof(T));std::memcpy(&b,inputs[r].bytes.data()+i*sizeof(T),sizeof(T));
        T v=op==MPI_SUM?a+b:(op==MPI_MIN?std::min(a,b):std::max(a,b));
        std::memcpy(out.data()+i*sizeof(T),&v,sizeof(T));
    }
    return out;
}
inline Bytes reduced(Round& r,int n,int t,int op){
    switch(t){case MPI_INT:return reduce<int>(r.in,n,op);case MPI_DOUBLE:return reduce<double>(r.in,n,op);
    case MPI_LONG_DOUBLE:return reduce<long double>(r.in,n,op);case MPI_LONG_LONG_INT:return reduce<long long>(r.in,n,op);
    default:throw std::runtime_error("Unsupported emulated reduction");}
}
inline int run(int count,const std::function<int()>& fn){
    ranks=count;round.reset();queues.clear();std::vector<std::thread> threads;std::vector<int> results(count,1);
    for(int r=0;r<count;++r)threads.emplace_back([&,r]{rank=r;try{results[r]=fn();}catch(const std::exception& e){
        std::fprintf(stderr,"MPI emulation rank %d: %s\n",r,e.what());std::_Exit(3);
    }});
    for(auto& t:threads)t.join();
    return *std::max_element(results.begin(),results.end());
}
}
inline int MPI_Init(int*,char***){return 0;}
inline int MPI_Finalize(){return 0;}
inline int MPI_Comm_rank(MPI_Comm c,int* p){*p=c==MPI_COMM_SELF?0:mpi_threads::rank;return 0;}
inline int MPI_Comm_size(MPI_Comm c,int* p){*p=c==MPI_COMM_SELF?1:mpi_threads::ranks;return 0;}
inline int MPI_Bcast(void* p,int n,int t,int root,MPI_Comm c){
    if(c==MPI_COMM_SELF)return 0;
    mpi_threads::Slot s;if(mpi_threads::rank==root)s.bytes=mpi_threads::copy(p,std::size_t(n)*mpi_threads::bytesize(t));
    auto out=mpi_threads::collective("Bcast:"+std::to_string(n)+":"+std::to_string(t)+":"+std::to_string(root),std::move(s),[&](auto& r){for(auto& o:r.out)o=r.in[root].bytes;});
    if(!out.empty())std::memcpy(p,out.data(),out.size());
    return 0;
}
inline int MPI_Allreduce(const void* p,void* d,int n,int t,int op,MPI_Comm c){
    const auto nbytes=std::size_t(n)*mpi_threads::bytesize(t);
    if(c==MPI_COMM_SELF){if(p!=MPI_IN_PLACE)std::memcpy(d,p,nbytes);return 0;}
    mpi_threads::Slot s;s.bytes=mpi_threads::copy(p==MPI_IN_PLACE?d:p,nbytes);
    auto out=mpi_threads::collective("Allreduce:"+std::to_string(n)+":"+std::to_string(t)+":"+std::to_string(op),std::move(s),[&](auto& r){auto result=mpi_threads::reduced(r,n,t,op);for(auto& o:r.out)o=result;});
    std::memcpy(d,out.data(),out.size());return 0;
}
inline int MPI_Reduce(const void* p,void* d,int n,int t,int op,int root,MPI_Comm c){
    const auto nbytes=std::size_t(n)*mpi_threads::bytesize(t);
    if(c==MPI_COMM_SELF){if(p!=MPI_IN_PLACE)std::memcpy(d,p,nbytes);return 0;}
    mpi_threads::Slot s;s.bytes=mpi_threads::copy(p==MPI_IN_PLACE?d:p,nbytes);
    auto out=mpi_threads::collective("Reduce:"+std::to_string(n)+":"+std::to_string(t)+":"+std::to_string(op)+":"+std::to_string(root),std::move(s),[&](auto& r){r.out[root]=mpi_threads::reduced(r,n,t,op);});
    if(mpi_threads::rank==root)std::memcpy(d,out.data(),out.size());
    return 0;
}
inline int MPI_Sendrecv(const void* p,int ns,int ts,int to,int stag,void* d,int nr,int tr,int from,int rtag,MPI_Comm c,MPI_Status*){
    if(c==MPI_COMM_SELF){if(to==0&&from==0)std::memcpy(d,p,std::size_t(ns)*mpi_threads::bytesize(ts));return 0;}
    std::unique_lock<std::mutex> lock(mpi_threads::mutex);
    if(to!=MPI_PROC_NULL){mpi_threads::queues[{mpi_threads::rank,to,stag}].push_back(mpi_threads::copy(p,std::size_t(ns)*mpi_threads::bytesize(ts)));mpi_threads::cv.notify_all();}
    if(from!=MPI_PROC_NULL){
        const auto key=std::make_tuple(from,mpi_threads::rank,rtag);
        mpi_threads::cv.wait(lock,[&]{return !mpi_threads::queues[key].empty();});
        auto b=std::move(mpi_threads::queues[key].front());mpi_threads::queues[key].pop_front();
        if(b.size()!=std::size_t(nr)*mpi_threads::bytesize(tr))throw std::runtime_error("Emulated halo size mismatch");
        std::memcpy(d,b.data(),b.size());
    }
    return 0;
}
inline int MPI_Gatherv(const void* p,int n,int t,void* d,const int* counts,const int* offsets,int rt,int root,MPI_Comm c){
    if(c==MPI_COMM_SELF){std::memcpy(static_cast<char*>(d)+offsets[0]*mpi_threads::bytesize(rt),p,n*mpi_threads::bytesize(t));return 0;}
    mpi_threads::Slot s;s.bytes=mpi_threads::copy(p,std::size_t(n)*mpi_threads::bytesize(t));
    if(mpi_threads::rank==root){s.counts.assign(counts,counts+mpi_threads::ranks);s.offsets.assign(offsets,offsets+mpi_threads::ranks);}
    auto out=mpi_threads::collective("Gatherv:"+std::to_string(t)+":"+std::to_string(rt)+":"+std::to_string(root),std::move(s),[&](auto& r){
        const auto& q=r.in[root];int extent=0;for(int k=0;k<mpi_threads::ranks;++k)extent=std::max(extent,q.offsets[k]+q.counts[k]);
        auto& b=r.out[root];b.resize(std::size_t(extent)*mpi_threads::bytesize(rt));
        for(int k=0;k<mpi_threads::ranks;++k){if(r.in[k].bytes.size()!=q.counts[k]*mpi_threads::bytesize(rt))throw std::runtime_error("Emulated gather count mismatch");std::memcpy(b.data()+q.offsets[k]*mpi_threads::bytesize(rt),r.in[k].bytes.data(),r.in[k].bytes.size());}
    });
    if(mpi_threads::rank==root)std::memcpy(d,out.data(),out.size());
    return 0;
}
inline int MPI_Scatterv(const void* p,const int* counts,const int* offsets,int t,void* d,int n,int rt,int root,MPI_Comm c){
    if(c==MPI_COMM_SELF){std::memcpy(d,static_cast<const char*>(p)+offsets[0]*mpi_threads::bytesize(t),n*mpi_threads::bytesize(rt));return 0;}
    mpi_threads::Slot s;
    if(mpi_threads::rank==root){s.counts.assign(counts,counts+mpi_threads::ranks);s.offsets.assign(offsets,offsets+mpi_threads::ranks);int extent=0;for(int k=0;k<mpi_threads::ranks;++k)extent=std::max(extent,offsets[k]+counts[k]);s.bytes=mpi_threads::copy(p,std::size_t(extent)*mpi_threads::bytesize(t));}
    auto out=mpi_threads::collective("Scatterv:"+std::to_string(t)+":"+std::to_string(rt)+":"+std::to_string(root),std::move(s),[&](auto& r){
        const auto& q=r.in[root];for(int k=0;k<mpi_threads::ranks;++k)r.out[k]=mpi_threads::copy(q.bytes.data()+q.offsets[k]*mpi_threads::bytesize(t),std::size_t(q.counts[k])*mpi_threads::bytesize(t));
    });
    if(out.size()!=std::size_t(n)*mpi_threads::bytesize(rt))throw std::runtime_error("Emulated scatter count mismatch");
    std::memcpy(d,out.data(),out.size());return 0;
}
[[noreturn]] inline int MPI_Abort(MPI_Comm,int code){std::_Exit(code);}
#endif
