#ifndef NICR_MPI_THREADS_VERIFY_H
#define NICR_MPI_THREADS_VERIFY_H
// Test-only implementation of this solver's MPI subset. NOT a native MPI runtime.
// Independent communicator contexts, Cartesian neighbors, subarray datatypes,
// datatype-aware messages, and collective signatures are checked in one process.
// A vendor ABI, network, launcher, and distributed-memory performance are NOT tested.
#include <algorithm>
#include <array>
#include <condition_variable>
#include <cstdio>
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
constexpr int MPI_COMM_WORLD=0,MPI_COMM_SELF=1,MPI_PROC_NULL=-1,MPI_COMM_NULL=-2;
constexpr int MPI_DATATYPE_NULL=0,MPI_INT=1,MPI_DOUBLE=2,MPI_LONG_DOUBLE=3,MPI_LONG_LONG_INT=4,MPI_CHAR=5;
constexpr int MPI_MIN=1,MPI_MAX=2,MPI_SUM=3,MPI_ORDER_C=0;
inline int mpi_place_obj;
#define MPI_IN_PLACE (&mpi_place_obj)
#define MPI_STATUS_IGNORE nullptr
namespace mpi_threads {
inline thread_local int rank=0,next_self=-100,next_type=100;
inline thread_local bool finalized=false;
inline int ranks=1,next_comm=2;
using Bytes=std::vector<char>;
struct CommInfo {bool self=false;std::array<int,3> dims{{1,1,1}},periods{};};
struct Type {std::size_t extent=0;std::vector<std::size_t> offsets;std::size_t atom=0;bool committed=false;};
inline thread_local std::map<int,CommInfo> comms;
inline thread_local std::map<int,Type> types;
inline bool self(int c){return c==MPI_COMM_SELF||(c< -2&&comms.at(c).self);}
inline std::size_t bytesize(int t){
    switch(t){case MPI_INT:return sizeof(int);case MPI_DOUBLE:return sizeof(double);
    case MPI_LONG_DOUBLE:return sizeof(long double);case MPI_LONG_LONG_INT:return sizeof(long long);
    case MPI_CHAR:return sizeof(char);default:{const auto& a=types.at(t);return a.atom*a.offsets.size();}}
}
inline std::size_t extent(int t){return t<100?bytesize(t):types.at(t).extent;}
inline Bytes copy(const void* p,std::size_t n){Bytes v(n);if(n)std::memcpy(v.data(),p,n);return v;}
inline Bytes pack(const void* p,int count,int type){
    if(type<100)return copy(p,std::size_t(count)*bytesize(type));
    const auto& t=types.at(type);if(!t.committed)throw std::runtime_error("Use of uncommitted emulated datatype");
    Bytes result(std::size_t(count)*bytesize(type));std::size_t q=0;
    for(int n=0;n<count;++n)for(auto offset:t.offsets){
        std::memcpy(result.data()+q,static_cast<const char*>(p)+n*t.extent+offset,t.atom);q+=t.atom;
    }
    return result;
}
inline void unpack(const Bytes& b,void* p,int count,int type){
    if(b.size()!=std::size_t(count)*bytesize(type))throw std::runtime_error("Emulated message signature/size mismatch");
    if(type<100){if(!b.empty())std::memcpy(p,b.data(),b.size());return;}
    const auto& t=types.at(type);if(!t.committed)throw std::runtime_error("Receive with uncommitted datatype");
    std::size_t q=0;
    for(int n=0;n<count;++n)for(auto offset:t.offsets){
        std::memcpy(static_cast<char*>(p)+n*t.extent+offset,b.data()+q,t.atom);q+=t.atom;
    }
}
struct Slot {Bytes bytes;std::vector<int> counts,offsets;};
struct Round {
    std::string signature;int arrived=0;bool ready=false;
    std::vector<Slot> in;std::vector<Bytes> out;
    explicit Round(std::string s):signature(std::move(s)),in(ranks),out(ranks){}
};
inline std::mutex mutex;
inline std::condition_variable cv;
inline std::map<int,std::shared_ptr<Round>> rounds;
inline std::map<std::tuple<int,int,int,int>,std::deque<Bytes>> queues;
inline Bytes collective(int comm,const std::string& sig,Slot slot,const std::function<void(Round&)>& finish){
    std::unique_lock<std::mutex> lock(mutex);
    auto& entry=rounds[comm];if(!entry)entry=std::make_shared<Round>(sig);
    auto current=entry;
    if(current->signature!=sig)throw std::runtime_error("Emulated MPI collective mismatch: "+sig+" vs "+current->signature);
    current->in[rank]=std::move(slot);
    if(++current->arrived==ranks){finish(*current);current->ready=true;entry.reset();cv.notify_all();}
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
    ranks=count;rounds.clear();queues.clear();next_comm=2;
    std::vector<std::thread> threads;std::vector<int> results(count,1);
    for(int r=0;r<count;++r)threads.emplace_back([&,r]{rank=r;try{results[r]=fn();
        if(!types.empty()||!comms.empty())throw std::runtime_error("Leaked communicator/datatype in solver");
    }catch(const std::exception& e){std::fprintf(stderr,"MPI emulation rank %d: %s\n",r,e.what());std::_Exit(3);}});
    for(auto& t:threads)t.join();
    for(const auto& q:queues)if(!q.second.empty())throw std::runtime_error("Unmatched emulated MPI message");
    return *std::max_element(results.begin(),results.end());
}
}
inline int MPI_Init(int*,char***){
    mpi_threads::finalized=false;
#ifdef NICR_SERIAL_VERIFY
    for(const char* key:{"OMPI_COMM_WORLD_SIZE","PMI_SIZE","PMIX_UNIV_SIZE"}){
        const char* v=std::getenv(key);if(v&&std::atoi(v)>1){std::fprintf(stderr,"Serial verification binary cannot run under multiple MPI ranks.\n");std::exit(2);}
    }
#endif
    return 0;
}
inline int MPI_Finalize(){mpi_threads::finalized=true;return 0;}
inline int MPI_Finalized(int* f){*f=mpi_threads::finalized?1:0;return 0;}
inline int MPI_Comm_rank(MPI_Comm c,int* p){*p=mpi_threads::self(c)?0:mpi_threads::rank;return 0;}
inline int MPI_Comm_size(MPI_Comm c,int* p){*p=mpi_threads::self(c)?1:mpi_threads::ranks;return 0;}
inline int MPI_Dims_create(int n,int nd,int* dims){
    int fixed=1;std::vector<int> axes;
    for(int a=0;a<nd;++a){if(dims[a]<0)throw std::runtime_error("Invalid dims");if(dims[a])fixed*=dims[a];else axes.push_back(a);}
    if(n%fixed)throw std::runtime_error("Invalid fixed dims product");
    int left=n/fixed;if(axes.empty()){if(left!=1)throw std::runtime_error("Invalid dims product");return 0;}
    std::vector<int> primes,values(axes.size(),1);
    for(int p=2;p<=left/p;++p)while(left%p==0){primes.push_back(p);left/=p;}
    if(left>1)primes.push_back(left);
    std::sort(primes.rbegin(),primes.rend());
    for(int p:primes)*std::min_element(values.begin(),values.end())*=p;
    std::sort(values.rbegin(),values.rend());
    for(std::size_t q=0;q<axes.size();++q)dims[axes[q]]=values[q];
    return 0;
}
inline int MPI_Cart_create(MPI_Comm parent,int nd,const int* dims,const int* periods,int reorder,MPI_Comm* out){
    if(nd!=3||reorder)throw std::runtime_error("Emulator supports unreordered 3D Cartesian topology only");
    int size;MPI_Comm_size(parent,&size);if(dims[0]*dims[1]*dims[2]!=size)throw std::runtime_error("Cartesian size mismatch");
    mpi_threads::CommInfo info;info.self=mpi_threads::self(parent);
    std::copy(dims,dims+3,info.dims.begin());std::copy(periods,periods+3,info.periods.begin());
    if(info.self)*out=mpi_threads::next_self--;
    else{
        std::string sig="Cart_create";for(int a=0;a<3;++a)sig+=":"+std::to_string(dims[a])+":"+std::to_string(periods[a]);
        auto bytes=mpi_threads::collective(parent,sig,{},[](auto& r){int id=mpi_threads::next_comm++;for(auto& b:r.out)b=mpi_threads::copy(&id,sizeof(id));});
        std::memcpy(out,bytes.data(),sizeof(int));
    }
    mpi_threads::comms[*out]=info;return 0;
}
inline int MPI_Comm_free(MPI_Comm* c){
    if(!mpi_threads::self(*c))mpi_threads::collective(*c,"Comm_free",{},[](auto&){});
    mpi_threads::comms.erase(*c);*c=MPI_COMM_NULL;return 0;
}
inline int MPI_Cart_coords(MPI_Comm c,int rank,int n,int* coords){
    const auto& info=mpi_threads::comms.at(c);if(n<3)throw std::runtime_error("Short coords array");
    for(int a=2;a>=0;--a){coords[a]=rank%info.dims[a];rank/=info.dims[a];}return 0;
}
inline int MPI_Cart_shift(MPI_Comm c,int axis,int disp,int* source,int* dest){
    const auto& info=mpi_threads::comms.at(c);int rank;MPI_Comm_rank(c,&rank);int xyz[3];MPI_Cart_coords(c,rank,3,xyz);
    auto neighbor=[&](int sign){std::array<int,3> q{{xyz[0],xyz[1],xyz[2]}};q[axis]+=sign*disp;
        if(info.periods[axis])q[axis]=(q[axis]%info.dims[axis]+info.dims[axis])%info.dims[axis];
        if(q[axis]<0||q[axis]>=info.dims[axis])return MPI_PROC_NULL;
        return (q[0]*info.dims[1]+q[1])*info.dims[2]+q[2];};
    *source=neighbor(-1);*dest=neighbor(1);return 0;
}
inline int MPI_Type_create_subarray(int nd,const int* sizes,const int* subs,const int* starts,int order,MPI_Datatype old,MPI_Datatype* out){
    if(nd!=3||order!=MPI_ORDER_C||old>=100)throw std::runtime_error("Unsupported emulated subarray request");
    for(int a=0;a<3;++a)if(subs[a]<1||subs[a]>sizes[a]||starts[a]<0||starts[a]+subs[a]>sizes[a])throw std::runtime_error("Invalid subarray extents");
    mpi_threads::Type t;t.atom=mpi_threads::bytesize(old);t.extent=std::size_t(sizes[0])*sizes[1]*sizes[2]*t.atom;
    for(int i=starts[0];i<starts[0]+subs[0];++i)for(int j=starts[1];j<starts[1]+subs[1];++j)for(int k=starts[2];k<starts[2]+subs[2];++k)
        t.offsets.push_back((std::size_t(i)*sizes[1]*sizes[2]+j*sizes[2]+k)*t.atom);
    *out=mpi_threads::next_type++;mpi_threads::types[*out]=std::move(t);return 0;
}
inline int MPI_Type_commit(MPI_Datatype* t){mpi_threads::types.at(*t).committed=true;return 0;}
inline int MPI_Type_free(MPI_Datatype* t){mpi_threads::types.erase(*t);*t=MPI_DATATYPE_NULL;return 0;}
inline int MPI_Bcast(void* p,int n,int t,int root,MPI_Comm c){
    if(mpi_threads::self(c))return 0;
    mpi_threads::Slot s;if(mpi_threads::rank==root)s.bytes=mpi_threads::copy(p,std::size_t(n)*mpi_threads::bytesize(t));
    auto out=mpi_threads::collective(c,"Bcast:"+std::to_string(n)+":"+std::to_string(t)+":"+std::to_string(root),std::move(s),[&](auto& r){for(auto& o:r.out)o=r.in[root].bytes;});
    if(!out.empty())std::memcpy(p,out.data(),out.size());
    return 0;
}
inline int MPI_Allreduce(const void* p,void* d,int n,int t,int op,MPI_Comm c){
    const auto nbytes=std::size_t(n)*mpi_threads::bytesize(t);
    if(mpi_threads::self(c)){if(p!=MPI_IN_PLACE)std::memcpy(d,p,nbytes);return 0;}
    mpi_threads::Slot s;s.bytes=mpi_threads::copy(p==MPI_IN_PLACE?d:p,nbytes);
    auto out=mpi_threads::collective(c,"Allreduce:"+std::to_string(n)+":"+std::to_string(t)+":"+std::to_string(op),std::move(s),[&](auto& r){auto result=mpi_threads::reduced(r,n,t,op);for(auto& o:r.out)o=result;});
    std::memcpy(d,out.data(),out.size());return 0;
}
inline int MPI_Reduce(const void* p,void* d,int n,int t,int op,int root,MPI_Comm c){
    const auto nbytes=std::size_t(n)*mpi_threads::bytesize(t);
    if(mpi_threads::self(c)){if(p!=MPI_IN_PLACE)std::memcpy(d,p,nbytes);return 0;}
    mpi_threads::Slot s;s.bytes=mpi_threads::copy(p==MPI_IN_PLACE?d:p,nbytes);
    auto out=mpi_threads::collective(c,"Reduce:"+std::to_string(n)+":"+std::to_string(t)+":"+std::to_string(op)+":"+std::to_string(root),std::move(s),[&](auto& r){r.out[root]=mpi_threads::reduced(r,n,t,op);});
    if(mpi_threads::rank==root)std::memcpy(d,out.data(),out.size());
    return 0;
}
inline int MPI_Sendrecv(const void* p,int ns,int ts,int to,int stag,void* d,int nr,int tr,int from,int rtag,MPI_Comm c,MPI_Status*){
    if(mpi_threads::self(c)){
        if(to==0&&from==0)mpi_threads::unpack(mpi_threads::pack(p,ns,ts),d,nr,tr);
        else if(to!=MPI_PROC_NULL||from!=MPI_PROC_NULL)throw std::runtime_error("Invalid self communicator peer");
        return 0;
    }
    auto outgoing=to==MPI_PROC_NULL?mpi_threads::Bytes{}:mpi_threads::pack(p,ns,ts);
    std::unique_lock<std::mutex> lock(mpi_threads::mutex);
    if(to!=MPI_PROC_NULL){mpi_threads::queues[{c,mpi_threads::rank,to,stag}].push_back(std::move(outgoing));mpi_threads::cv.notify_all();}
    if(from!=MPI_PROC_NULL){
        const auto key=std::make_tuple(c,from,mpi_threads::rank,rtag);
        mpi_threads::cv.wait(lock,[&]{return !mpi_threads::queues[key].empty();});
        auto b=std::move(mpi_threads::queues[key].front());mpi_threads::queues[key].pop_front();
        mpi_threads::unpack(b,d,nr,tr);
    }
    return 0;
}
inline int MPI_Gatherv(const void* p,int n,int t,void* d,const int* counts,const int* offsets,int rt,int root,MPI_Comm c){
    if(mpi_threads::self(c)){std::memcpy(static_cast<char*>(d)+offsets[0]*mpi_threads::bytesize(rt),p,n*mpi_threads::bytesize(t));return 0;}
    mpi_threads::Slot s;s.bytes=mpi_threads::copy(p,std::size_t(n)*mpi_threads::bytesize(t));
    if(mpi_threads::rank==root){s.counts.assign(counts,counts+mpi_threads::ranks);s.offsets.assign(offsets,offsets+mpi_threads::ranks);}
    auto out=mpi_threads::collective(c,"Gatherv:"+std::to_string(t)+":"+std::to_string(rt)+":"+std::to_string(root),std::move(s),[&](auto& r){
        const auto& q=r.in[root];int extent=0;for(int k=0;k<mpi_threads::ranks;++k)extent=std::max(extent,q.offsets[k]+q.counts[k]);
        auto& b=r.out[root];b.resize(std::size_t(extent)*mpi_threads::bytesize(rt));
        for(int k=0;k<mpi_threads::ranks;++k){if(r.in[k].bytes.size()!=q.counts[k]*mpi_threads::bytesize(rt))throw std::runtime_error("Emulated gather count mismatch");std::memcpy(b.data()+q.offsets[k]*mpi_threads::bytesize(rt),r.in[k].bytes.data(),r.in[k].bytes.size());}
    });
    if(mpi_threads::rank==root)std::memcpy(d,out.data(),out.size());
    return 0;
}
inline int MPI_Scatterv(const void* p,const int* counts,const int* offsets,int t,void* d,int n,int rt,int root,MPI_Comm c){
    if(mpi_threads::self(c)){std::memcpy(d,static_cast<const char*>(p)+offsets[0]*mpi_threads::bytesize(t),n*mpi_threads::bytesize(rt));return 0;}
    mpi_threads::Slot s;
    if(mpi_threads::rank==root){s.counts.assign(counts,counts+mpi_threads::ranks);s.offsets.assign(offsets,offsets+mpi_threads::ranks);int extent=0;for(int k=0;k<mpi_threads::ranks;++k)extent=std::max(extent,offsets[k]+counts[k]);s.bytes=mpi_threads::copy(p,std::size_t(extent)*mpi_threads::bytesize(t));}
    auto out=mpi_threads::collective(c,"Scatterv:"+std::to_string(t)+":"+std::to_string(rt)+":"+std::to_string(root),std::move(s),[&](auto& r){
        const auto& q=r.in[root];for(int k=0;k<mpi_threads::ranks;++k)r.out[k]=mpi_threads::copy(q.bytes.data()+q.offsets[k]*mpi_threads::bytesize(t),std::size_t(q.counts[k])*mpi_threads::bytesize(t));
    });
    if(out.size()!=std::size_t(n)*mpi_threads::bytesize(rt))throw std::runtime_error("Emulated scatter count mismatch");
    std::memcpy(d,out.data(),out.size());return 0;
}
[[noreturn]] inline int MPI_Abort(MPI_Comm,int code){std::_Exit(code);}
#endif
