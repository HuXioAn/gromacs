#include <cmath>
#include <array>
#include <iostream>
#include <filesystem>
#include <memory>
#include <cstring>
#include <vector>
#include <tuple>
#include <unordered_set>
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <utility>   // std::pair

//#include "gromacs/domdec/domdec_struct.h"
//#include "gromacs/domdec/haloexchange.h"
#include "gromacs/domdec/domdec.h"
#include "gromacs/gmxlib/network.h"
#include "gromacs/mdtypes/commrec.h"
#include "gromacs/mdtypes/enerdata.h"
#include "gromacs/mdtypes/forceoutput.h"
#include "gromacs/pbcutil/pbc.h"
#include "gromacs/utility/exceptions.h"
#include "gromacs/utility/logger.h"

#include "deepmdmodel.h"

#define GMX_DEEPMD_INFERENCE_MULTI_MPI_COLLECTIVE_REBUILD_DD 1

#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)
#pragma message("GMX_DEEPMD_INFERENCE_MULTI_MPI = " STR(GMX_DEEPMD_INFERENCE_MULTI_MPI))
#pragma message("GMX_DEEPMD_INFERENCE_MULTI_MPI_COLLECTIVE = " STR(GMX_DEEPMD_INFERENCE_MULTI_MPI_COLLECTIVE))
#pragma message("GMX_DEEPMD_INFERENCE_MULTI_MPI_COLLECTIVE_REBUILD_DD = " STR(GMX_DEEPMD_INFERENCE_MULTI_MPI_COLLECTIVE_REBUILD_DD))


namespace gmx
{
/*
* DeePMD-kit units                      GROMACS units
* Distance	Å (Angstrom)               nm
* Energy	eV
* Force	eV/Å
* Charge	e (proton charge)
*/
namespace {
    static constexpr real c_dp2gmx = 0.1;
    static constexpr real e_dp2gmx = 96.48533132;
    static constexpr real f_dp2gmx = 964.8533132;

    static constexpr real lambda = 1.0; 
}


DeepmdModel::DeepmdModel(const std::string& fileName, const MDLogger* logger) : logger_(logger)
{
    modelFileName_ = fileName;
    if(!std::filesystem::exists(modelFileName_))
    {
        GMX_THROW(FileIOError("DeepMD model file does not exist: " + modelFileName_));
    }

}

DeepmdModel::~DeepmdModel() {}

void DeepmdModel::initModel()
{
    isInit_ = false; // this shall be delayed to the cr set for device selection
    outputReady_ = false;
}

void DeepmdModel::prepareAtomPositions(std::vector<RVec>& positions)
{
    const int N = positions.size(); // local + ghost
    inferInfo_.totalNNAtomNum = N;
    auto& atomPos = inferInfo_.atomPosition_;
    atomPos.clear();

    for (int i = 0; i < N; ++i)
    {
        atomPos.push_back(positions[i][0] / c_dp2gmx);
        atomPos.push_back(positions[i][1] / c_dp2gmx);
        atomPos.push_back(positions[i][2] / c_dp2gmx);
    }

}

void DeepmdModel::prepareAtomNumbers(std::vector<int>& atomTypes)
{
    const int N = atomTypes.size();
    auto& atomType = inferInfo_.atomType_;
    atomType.clear();

    for (int i = 0; i < N; ++i)
    {
        atomType.push_back(atomTypes[i] - 1);
    }

}

void DeepmdModel::prepareBox(matrix& box)
{
    // convert box to 1D vector
    auto& boxVec = inferInfo_.box_;
    boxVec.resize(DIM * DIM);
    for (int i = 0; i < DIM; ++i)
    {
        for (int j = 0; j < DIM; ++j)
        {
            boxVec[i * DIM + j] = box[i][j] / c_dp2gmx; // assume pbc = true
        }
    }
    
}

void DeepmdModel::preparePbcType([[maybe_unused]] PbcType& pbcType)
{

    if (pbcType == PbcType::Xyz){ // all periodic
        inferInfo_.pbcType_ = true;

    } else if (pbcType == PbcType::No) { // no periodic
        inferInfo_.pbcType_ = false;
    }else {
        GMX_THROW(InconsistentInputError("Not supuorted PBC type for DeepMD model: " + std::to_string(static_cast<int>(pbcType))));
    }
    
}

static inline real euclideanDistanceSqrd(const std::array<real,3> ri, const std::array<real,3> rj)
{
    return (ri[0] - rj[0])*(ri[0] - rj[0]) + (ri[1] - rj[1])*(ri[1] - rj[1]) + (ri[2] - rj[2])*(ri[2] - rj[2]);
}


// Returns true if j's minimum-image relative to i uses a nonzero shift AND is within rcut.
// If true, out_shifted = coordinates of the image of j that should be duplicated,
// and out_n = integer shift vector (nx, ny, nz).
static inline real min_image_duplicate_if_needed(
    const std::array<real,3>& ri,     // atom i position
    const std::array<real,3>& rj,     // atom j position (base cell)
    const std::vector<real>& box,
    const real rcutSqrd,
    bool& inCutOffDistance,
    bool& need_dup,
    std::array<real,3>& out_shifted, // shifted coords for duplicate (if any)
    std::array<size_t,3>& out_n)  
{
    // Raw displacement
    real dx = rj[0] - ri[0];
    real dy = rj[1] - ri[1];
    real dz = rj[2] - ri[2];

    // Compute integer shifts that bring j to the minimum image relative to i
    const int nx = (int)std::llround(dx / box[0]);
    const int ny = (int)std::llround(dy / box[DIM + 1]);
    const int nz = (int)std::llround(dz / box[2 * DIM + 2]);

    // Minimum-image displacement
    dx -= nx * box[0];
    dy -= ny * box[DIM + 1];
    dz -= nz * box[2 * DIM + 2];

    const real d2   = dx*dx + dy*dy + dz*dz;
    inCutOffDistance = (d2 <= rcutSqrd);

    need_dup = inCutOffDistance && (nx != 0 || ny != 0 || nz != 0);
    // Coordinates of the image that realizes the minimum image
    out_shifted = { rj[0] - nx*box[0], rj[1] - ny*box[DIM + 1], rj[2] - nz*box[2 * DIM + 2] };
    out_n   = { nx, ny, nz };
        
    return d2;
}


struct ImgKey {
    int j;   // base atom index in your packed/global array
    int nx;  // lattice shift in x
    int ny;  // lattice shift in y
    int nz;  // lattice shift in z
    // equality
    bool operator==(const ImgKey& o) const noexcept {
        return j==o.j && nx==o.nx && ny==o.ny && nz==o.nz;
    }
};

// robust hash (no truncation); boost-like hash_combine
struct ImgKeyHash {
    static inline std::size_t mix(std::size_t h, std::size_t v) {
        // 64-bit constant works fine on 32-bit too
        return h ^ (v + 0x9e3779b97f4a7c15ULL + (h<<6) + (h>>2));
    }
    std::size_t operator()(ImgKey const& k) const noexcept {
        std::size_t h = std::hash<int>{}(k.j);
        h = mix(h, std::hash<int>{}(k.nx));
        h = mix(h, std::hash<int>{}(k.ny));
        h = mix(h, std::hash<int>{}(k.nz));
        return h;
    }
};

//using ImgSet = std::unordered_set<ImgKey, ImgKeyHash>;
//static inline bool add_atom_if_new(ImgSet& imgSet, int j, int nx, int ny, int nz) {

class ImgIndex {
public:
    explicit ImgIndex(std::size_t expected = 0) {
        order_.reserve(expected);
        index_.reserve(expected);
    }

    // Add if new; always returns the index and whether it was inserted now.
    // Result: {index, inserted}
    std::pair<std::size_t, bool> add_or_get(int j, int nx, int ny, int nz) {
        ImgKey key{j, nx, ny, nz};
        auto it = index_.find(key);
        if (it != index_.end())
            return {it->second, false};  // already present → give index

        const std::size_t idx = order_.size();
        order_.push_back(key);           // keep insertion order
        index_.emplace(key, idx);
        return {idx, true};              // new element → appended at end
    }

    // Optional helpers
    std::size_t size() const noexcept { return order_.size(); }
    const ImgKey& at(std::size_t i) const { return order_[i]; }
    const std::vector<ImgKey>& order() const noexcept { return order_; }

    // Iterate in insertion order:
    auto begin() const { return order_.begin(); }
    auto end()   const { return order_.end(); }

private:
    std::vector<ImgKey> order_;                                   // insertion order
    std::unordered_map<ImgKey, std::size_t, ImgKeyHash> index_;   // key → index
};

static inline bool add_atom_if_new(ImgIndex& imgSet, int j, int nx, int ny, int nz, size_t* out_idx=nullptr) {
    auto [idx, inserted] = imgSet.add_or_get(j, nx, ny, nz);
    if (out_idx) *out_idx = int(idx);
    return inserted;               // true  -> wasn't there (now added)
                                   // false -> already present
}

static inline void build_DD(const int numRanks, int& nx, int& ny, int& nz) {
    if (numRanks <= 0) { nx = ny = nz = 0; return; }

    // Trivial decomposition: initialize BOTH "best" metrics AND outputs.
    int bestx = numRanks, besty = 1, bestz = 1;
    nx = bestx; ny = besty; nz = bestz;

    auto ratio = [](int a, int b, int c){
        int mn = std::min({a,b,c});
        int mx = std::max({a,b,c});
        return static_cast<double>(mx) / static_cast<double>(mn);
    };
    auto surface = [](int a, int b, int c){
        return 1LL*a*b + 1LL*a*c + 1LL*b*c; // 64-bit products
    };

    double best_ratio   = ratio(bestx, besty, bestz);
    long long best_surf = surface(bestx, besty, bestz);

    // Robust integer cube-root floor of numRanks
    int cbrtN = static_cast<int>(std::cbrt(static_cast<double>(numRanks)));
    while (1LL*(cbrtN+1)*(cbrtN+1)*(cbrtN+1) <= numRanks) ++cbrtN;
    while (1LL*cbrtN*cbrtN*cbrtN > numRanks) --cbrtN;

    for (int i = 1; i <= cbrtN; ++i) {
        if (numRanks % i) continue;
        const int n1 = numRanks / i;

        // Robust integer sqrt floor of n1
        int sqrtN1 = static_cast<int>(std::sqrt(static_cast<double>(n1)));
        while (1LL*(sqrtN1+1)*(sqrtN1+1) <= n1) ++sqrtN1;
        while (1LL*sqrtN1*sqrtN1 > n1) --sqrtN1;

        for (int j = 1; j <= sqrtN1; ++j) {
            if (n1 % j) continue;
            const int k = n1 / j;

            int a = i, b = j, c = k;
            // sort so a <= b <= c
            if (a > b) std::swap(a, b);
            if (b > c) std::swap(b, c);
            if (a > b) std::swap(a, b);

            const double r = static_cast<double>(c) / static_cast<double>(a);
            const long long s = surface(a, b, c);

            // If you want to stay with doubles:
            constexpr double eps = 1e-15;
            if (r < best_ratio - eps || (std::abs(r - best_ratio) <= eps && s < best_surf)) {
                best_ratio = r;
                best_surf  = s;
                // return in descending order (x >= y >= z)
                nx = c; ny = b; nz = a;
            }
        }
    }
}


static inline void build_DD_manual(const int numRanks, int& nx, int& ny, int& nz) {
    if (numRanks <= 0) { nx = ny = nz = 0; return; }

    if(numRanks == 2){
        nx = 1;
        ny = 1;
        nz = 2;
    }else if(numRanks == 4){
        nx = 1;
        ny = 1;
        nz = 4;
    }else if(numRanks == 8){
        nx = 1;
        ny = 1;
        nz = 8;
    }else if(numRanks == 16){
        nx = 1;
        ny = 1;
        nz = 16;
    }else if(numRanks == 32){
        nx = 1;
        ny = 1;
        nz = 32;
    }else{
        build_DD(numRanks,nx,ny,nz);
    }
}

// Linear rank → (ix,iy,iz). Use MPI_Cart_coords if you already have a Cart grid.
static inline void rank_to_ijk(const int rank, const int nDDx, const int nDDy, const int nDDz,
                               int& ix, int& iy, int& iz)
{
    ix =  rank % nDDx;
    iy = (rank / nDDx) % nDDy;
    iz =  rank / (nDDx * nDDy);
}


static inline bool select_atom_local(const real atom_pos, const real lo, const real hi, const real Lbox, const real halo,
                                     real& atom_pos_shifted, bool& local) {
    const real width = hi - lo;                   // subdomain width
    local = false;
    if (width <= real(0)) return false;
    
    // select local atoms
    if (atom_pos >= lo && atom_pos < hi ){
        atom_pos_shifted = atom_pos;
        local = true;
        return true;
    }
    return false;
}


static inline bool select_atom_pbc(const real atom_pos, const real lo, const real hi, const real Lbox, const real halo,
                                     real& atom_pos_shifted, bool& local) {
    const real width = hi - lo;                   // subdomain width
    const real haloLow = lo - halo;
    const real haloHigh = hi + halo;
    real atom_pos_test;
    if (width <= real(0)) { local = false; return false; }
    
    // check local
    local = (atom_pos >= lo && atom_pos < hi);
    atom_pos_shifted = atom_pos;   // default to unshifted

    // left ghost atoms
    if ( haloLow >= 0.0 ) { atom_pos_test = atom_pos; }
    else{ atom_pos_test  = atom_pos - Lbox; }
    if (atom_pos_test >= haloLow && atom_pos_test < lo ){
        atom_pos_shifted = atom_pos_test;
        return true;
    }

    // right ghost atoms
    if (haloHigh < Lbox){ atom_pos_test = atom_pos; }
    else{ atom_pos_test = atom_pos + Lbox; }
    if (atom_pos_test >= hi && atom_pos_test < haloHigh ){
        atom_pos_shifted = atom_pos_test;
        return true;
    }

    return local;
}


static inline void select_pbc(
    const std::array<real,3>& orig,
    const std::array<real,3>& lo,
    const std::array<real,3>& hi,
    const std::array<real,3>& Lxyz,
    const real rHalo,
    const std::size_t idx,
    const std::vector<int>& atomTypeGlobal,
    std::vector<real>& atomPositionCollective,
    std::vector<int>&  atomTypeCollective,
    std::vector<int>&  atomIdxLocal,
    std::vector<int>&  atomIdxGlobal,
    int&               countLocal)
{
    real xShifted, yShifted, zShifted; // local coords relative to (xlo,ylo,zlo)
    bool localx, localy, localz;
    const bool inx = select_atom_pbc(orig[0], lo[0], hi[0], Lxyz[0], rHalo, xShifted, localx);
    if (!inx) return;
    const bool iny = select_atom_pbc(orig[1], lo[1], hi[1], Lxyz[1], rHalo, yShifted, localy);
    if (!iny) return;
    const bool inz = select_atom_pbc(orig[2], lo[2], hi[2], Lxyz[2], rHalo, zShifted, localz);
    if (!inz) return;

    if(localx && localy && localz) return;

    atomPositionCollective.push_back(xShifted);
    atomPositionCollective.push_back(yShifted);
    atomPositionCollective.push_back(zShifted);
    atomTypeCollective.push_back(atomTypeGlobal[idx]);
    atomIdxGlobal.push_back(idx);
    atomIdxLocal.push_back(-1);
}
static inline void select_local(
    const std::array<real,3>& orig,
    const std::array<real,3>& lo,
    const std::array<real,3>& hi,
    const std::array<real,3>& Lxyz,
    const real rHalo,
    const std::size_t idx,
    const std::vector<int>& atomTypeGlobal,
    std::vector<real>& atomPositionCollective,
    std::vector<int>&  atomTypeCollective,
    std::vector<int>&  atomIdxLocal,
    std::vector<int>&  atomIdxGlobal,
    int&               countLocal)
{
    real xShifted, yShifted, zShifted; // local coords relative to (xlo,ylo,zlo)
    bool localx, localy, localz;
    const bool inx = select_atom_local(orig[0], lo[0], hi[0], Lxyz[0], rHalo, xShifted, localx);
    if (!inx) return;
    const bool iny = select_atom_local(orig[1], lo[1], hi[1], Lxyz[1], rHalo, yShifted, localy);
    if (!iny) return;
    const bool inz = select_atom_local(orig[2], lo[2], hi[2], Lxyz[2], rHalo, zShifted, localz);
    if (!inz) return;

    atomPositionCollective.push_back(xShifted);
    atomPositionCollective.push_back(yShifted);
    atomPositionCollective.push_back(zShifted);
    atomTypeCollective.push_back(atomTypeGlobal[idx]);
    atomIdxGlobal.push_back(idx);
    if (localx && localy && localz){
        atomIdxLocal.push_back(atomIdxGlobal.back());
        countLocal++;
    }
    else{
        atomIdxLocal.push_back(-1);
    }
}



// Extract local+halo atoms with PBC and *shifted local coordinates*.
// Output layout matches input: [x0,y0,z0, x1,y1,z1, ...], but now positions
// live in [-halo, width+halo) per axis (i.e., subdomain-local frame).
static inline void extract_atoms_local_with_halo_shifted(
    const std::vector<int>& atomTypeGlobal,
    const std::vector<real>& atomPosGlobal,   // [x0,y0,z0, x1,y1,z1, ...], global coords
    const size_t Ntot,
    std::vector<real>&       atomPositionCollective,    // output (local coords, same layout)
    std::vector<int>&       atomTypeCollective, 
    std::vector<int>&       atomIdxLocal,    
    std::vector<int>&       atomIdxGlobal,
    int& countLocal,   
    const real Lx, const real Ly, const real Lz,
    const real rHalo,
    const int nDDx, const int nDDy, const int nDDz,
    const int rank)
{
    countLocal=0;
    // Which subdomain am I?
    int ix, iy, iz;
    rank_to_ijk(rank, nDDx, nDDy, nDDz, ix, iy, iz);

    // Subdomain extents (half-open). Clamp last slabs to L* to avoid FP gaps.
    const real dx = Lx / real(nDDx);
    const real dy = Ly / real(nDDy);
    const real dz = Lz / real(nDDz);

    const std::array<real,3> lo = {
        dx * real(ix),
        dy * real(iy),
        dz * real(iz)
    };

    const std::array<real,3> hi = {
        (ix == nDDx - 1) ? Lx : dx * real(ix + 1),
        (iy == nDDy - 1) ? Ly : dy * real(iy + 1),
        (iz == nDDz - 1) ? Lz : dz * real(iz + 1)
    };
    const std::array<real,3> Lxyz = {Lx,Ly,Lz};
    for (std::size_t i = 0; i < Ntot; ++i) {
        // Wrap globals into [0,L) first
        const std::array<real,3> orig = {atomPosGlobal[DIM*i + 0],atomPosGlobal[DIM*i + 1],atomPosGlobal[DIM*i + 2]};
        select_local(orig,lo,hi, Lxyz, rHalo, i,
                    atomTypeGlobal,
                    atomPositionCollective,
                    atomTypeCollective,
                    atomIdxLocal,
                    atomIdxGlobal,
                    countLocal);
        select_pbc(orig,lo,hi, Lxyz, rHalo, i,
                    atomTypeGlobal,
                    atomPositionCollective,
                    atomTypeCollective,
                    atomIdxLocal,
                    atomIdxGlobal,
                    countLocal);
    }
}

void  DeepmdModel::preProcessData(const NNPotParameters& params, std::vector<int>& idxLookup)
{

    constexpr real ghostCutOff = 6.5; // in Angstrom
    constexpr real rCutoff = 6.2;
    constexpr real twiceRCutoffSqrd = (real)4.0 * ghostCutOff * ghostCutOff;    
 
#if GMX_DEEPMD_INFERENCE_MULTI_MPI_COLLECTIVE

#if GMX_DEEPMD_INFERENCE_MULTI_MPI_COLLECTIVE_REBUILD_DD
    
    auto time0 = std::chrono::high_resolution_clock::now();

    const int numSubDomains = this->cr_->sizeOfDefaultCommunicator;
    const int myRank = this->cr_->rankInDefaultCommunicator;
    const int avgLocalNNAtom = inferInfo_.totalNNAtomNum / numSubDomains ;

    const real boxLx = inferInfo_.box_[0];
    const real boxLy = inferInfo_.box_[DIM + 1];
    const real boxLz = inferInfo_.box_[2*DIM + 2];

    int nDDx, nDDy, nDDz;
    build_DD_manual(numSubDomains,nDDx,nDDy,nDDz);
    //build_DD(numSubDomains,nDDx,nDDy,nDDz);
    
    //reserve three times the memory required by local atoms
    inferInfo_.localIdxes_.clear();
    inferInfo_.globalIdxes_.clear();
    inferInfo_.atomTypeCollective_.clear();
    inferInfo_.atomPositionCollective_.clear();
    inferInfo_.localIdxes_.reserve(3 * avgLocalNNAtom);
    inferInfo_.globalIdxes_.reserve(3 * avgLocalNNAtom);
    inferInfo_.atomTypeCollective_.reserve(3 * avgLocalNNAtom);
    inferInfo_.atomPositionCollective_.reserve(9 * avgLocalNNAtom);

    int countLocal = 0;
    extract_atoms_local_with_halo_shifted(inferInfo_.atomType_, inferInfo_.atomPosition_, inferInfo_.totalNNAtomNum, 
                                        inferInfo_.atomPositionCollective_, inferInfo_.atomTypeCollective_,
                                        inferInfo_.localIdxes_, inferInfo_.globalIdxes_, 
                                        countLocal, boxLx, boxLy, boxLz, 2*ghostCutOff, 
                                        nDDx, nDDy, nDDz, myRank);
    
    inferInfo_.totalLocalGhostNNAtomNum = inferInfo_.atomTypeCollective_.size();
    inferInfo_.localNNAtomNum = countLocal;
    inferInfo_.ghostNNAtomNum = inferInfo_.atomTypeCollective_.size() - inferInfo_.localNNAtomNum;

    auto time1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration1 = time1 - time0;

    std::cout<< "GMX_DEEPMD_INFERENCE_MULTI_MPI_COLLECTIVE Create rebuild DD Local- Rank " << myRank << 
    "\n numSubDomains " << numSubDomains << " nDDx "<< nDDx << " nDDy "<< nDDy << " nDDz "<< nDDz <<    
    "\n inferInfo_.atomTypeCollective_size()=" << inferInfo_.atomTypeCollective_.size() << 
    " inferInfo_.localNNAtomNum="<< inferInfo_.localNNAtomNum <<
    " inferInfo_.ghostNNAtomNum="<< inferInfo_.ghostNNAtomNum <<
    " - time compute DD " << (duration1.count())*1000 << " millis" <<
    " idxLookup.size()=" << idxLookup.size() << std::endl;

#else
    // create ghost halos computing pair to pair distances
    inferInfo_.localNNAtomNum = params.inpAtoms_->numAtomsLocal();
    //reserve three times the memory required by local atoms
    inferInfo_.localIdxes_.clear();
    inferInfo_.atomTypeCollective_.clear();
    inferInfo_.atomPositionCollective_.clear();
    inferInfo_.localIdxes_.reserve(3 * inferInfo_.localNNAtomNum);
    inferInfo_.globalIdxes_.reserve(3 * inferInfo_.localNNAtomNum);
    inferInfo_.atomTypeCollective_.reserve(3 * inferInfo_.localNNAtomNum);
    inferInfo_.atomPositionCollective_.reserve(9 * inferInfo_.localNNAtomNum);

    for (size_t i = 0; i < inferInfo_.totalNNAtomNum; i++){
        if (idxLookup[i] != -1){
            inferInfo_.globalIdxes_.push_back(i);
            inferInfo_.localIdxes_.push_back(idxLookup[i]);
            inferInfo_.atomTypeCollective_.push_back(inferInfo_.atomType_[i]);
            inferInfo_.atomPositionCollective_.push_back(inferInfo_.atomPosition_[i * DIM]);
            inferInfo_.atomPositionCollective_.push_back(inferInfo_.atomPosition_[i * DIM + 1]);
            inferInfo_.atomPositionCollective_.push_back(inferInfo_.atomPosition_[i * DIM + 2]);
        }
    }
    // add ghost and duplicated atoms to implement periodic bcs loop over local + non local atoms
    //ImgSet imgSet;
    //imgSet.reserve(4096);   // optional: avoid rehash during the loop
    ImgIndex imgSet(inferInfo_.localNNAtomNum);
    std::array<real,DIM> out_shifted;
    std::array<size_t,DIM> out_n;
    for (size_t i = 0; i < inferInfo_.localNNAtomNum; i++){
        //lIdx = params.inpAtoms_->localIndex()[i];
        const int lIdx = inferInfo_.localIdxes_[i];
        const std::array<real,DIM> ri = {inferInfo_.atomPositionCollective_[i * DIM],
                                    inferInfo_.atomPositionCollective_[i * DIM + 1],
                                    inferInfo_.atomPositionCollective_[i * DIM + 2]};

        for (size_t j = 0; j < inferInfo_.totalNNAtomNum; ++j){
            if (idxLookup[j] != lIdx)
            {
                const std::array<real,DIM> rj = {inferInfo_.atomPosition_[j * DIM],
                                            inferInfo_.atomPosition_[j * DIM + 1],
                                            inferInfo_.atomPosition_[j * DIM + 2]};
                
                bool inTwiceCutOffDistance = false;
                bool duplicate = false;
                const real dist2 = min_image_duplicate_if_needed(ri, rj, inferInfo_.box_, twiceRCutoffSqrd, 
                                    inTwiceCutOffDistance, duplicate, out_shifted, out_n);
                
                if (inTwiceCutOffDistance){
                    if (idxLookup[j] == -1 || duplicate){
                        if ( add_atom_if_new(imgSet, (int)j, out_n[0], out_n[1], out_n[2]) ) {
                            inferInfo_.globalIdxes_.push_back(j);
                            inferInfo_.atomTypeCollective_.push_back(inferInfo_.atomType_[j]);
                            inferInfo_.atomPositionCollective_.push_back(out_shifted[0]);
                            inferInfo_.atomPositionCollective_.push_back(out_shifted[1]);
                            inferInfo_.atomPositionCollective_.push_back(out_shifted[2]);
                        }
                    }
                }
            }
        }
    }

    inferInfo_.totalLocalGhostNNAtomNum = inferInfo_.atomTypeCollective_.size();
    inferInfo_.ghostNNAtomNum = inferInfo_.atomTypeCollective_.size() - inferInfo_.localNNAtomNum;

    std::cout<< "GMX_DEEPMD_INFERENCE_MULTI_MPI_COLLECTIVE Create Local- Rank " << this->cr_->rankInDefaultCommunicator << 
    " inferInfo_.atomTypeCollective_size()=" << inferInfo_.atomTypeCollective_.size() << 
    " inferInfo_.localNNAtomNum="<< inferInfo_.localNNAtomNum <<
    " inferInfo_.ghostNNAtomNum="<< inferInfo_.ghostNNAtomNum <<
    " idxLookup.size()=" << idxLookup.size() << std::endl;


#endif

#endif


}

void DeepmdModel::evaluateModel()
{
    if (!isInit_)
    {
        GMX_THROW(InternalError("deepmd not initialized before evaluateModel() was called."));
    }

    // periodic boundary conditions
    if (!inferInfo_.pbcType_)
    {
        inferInfo_.box_.resize(0); // no box needed
    }

    GMX_ASSERT(inferInfo_.atomPosition_.size() / DIM == inferInfo_.atomType_.size(),
               "Number of atom positions and atom types must match.");

    const int N = inferInfo_.atomType_.size();

#if GMX_DEEPMD_INFERENCE_MULTI_MPI_COLLECTIVE 

    inferInfo_.box_.clear();
    inferInfo_.step++;
    inferInfo_.atomForce_.clear();
    inferInfo_.atomEnergy_.clear();
    inferInfo_.atomVirial_.clear();

    auto time0 = std::chrono::high_resolution_clock::now();
    if(inferInfo_.atomPositionCollective_.size() > 0){
        dp_->compute<real>(inferInfo_.energy_, inferInfo_.atomForce_, inferInfo_.virial_, inferInfo_.atomEnergy_, inferInfo_.atomVirial_,
            inferInfo_.atomPositionCollective_, inferInfo_.atomTypeCollective_ , inferInfo_.box_);
    }
    auto time1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration1 = time1 - time0;
    std::cout<< "GMX_DEEPMD_INFERENCE_MULTI_MPI_COLLECTIVE compute - Rank " << this->cr_->rankInDefaultCommunicator <<
    "  inferInfo_.step " <<  inferInfo_.step <<
    " - time compute w/o inputlist " << (duration1.count())*1000 << " millis" << std::endl;

#else

    if (MAIN(cr_)){
        dp_->compute<real>(inferInfo_.energy_, inferInfo_.atomForce_, inferInfo_.virial_, inferInfo_.atomEnergy_, inferInfo_.atomVirial_,
            inferInfo_.atomPosition_, inferInfo_.atomType_ , inferInfo_.box_);   
    }
    else{
        inferInfo_.energy_= 0.0;
        inferInfo_.atomForce_.assign(DIM*N,0.0);
        inferInfo_.atomEnergy_.assign(N,0.0);
    }

#endif
    outputReady_ = true;
}


void DeepmdModel::writeForces(const int idx,const NNPotParameters& params, std::vector<int>& indices, const ArrayRef<RVec>& forces)
{
    // Grab the rank number
    int rank = this->cr_->rankInDefaultCommunicator;

    // Build a filename like "rank0", "rank1", etc.
    std::string filename = "rank" + std::to_string(rank) + "_" + std::to_string(idx) + ".dat";
    std::ofstream ofs(filename);
    if (!ofs) {
        std::cerr << "Error: could not open " << filename << " for output\n";
        return;
    }

    const int Ntot = indices.size(); // local + ghost
    const int Nlocal = this->localAtomNum;
    real maxFLocal=0.0, maxFghost=0.0;
    int lIdx, gIdx;
#if GMX_DEEPMD_INFERENCE_MULTI_MPI_COLLECTIVE
    for (size_t i = 0; i < inferInfo_.localNNAtomNum; i++) // local atoms
    {
        gIdx = inferInfo_.globalIdxes_[i];
        lIdx = inferInfo_.localIdxes_[i];
        const real fx = forces[lIdx][0];
        const real fy = forces[lIdx][1];
        const real fz = forces[lIdx][2];
        // ofs << "GlobalIdx="<< gIdx << " local - Force=("<<
        // forces[indices[i]][0]<<","<< forces[indices[i]][1]<<","<< forces[indices[i]][2]<<")"<< "\n";
        // std::cout<< "Rank " << this->cr_->rankInDefaultCommunicator<< " Local atoms idxLookup_["<< i <<"]=" << idxLookup_[i] << " params.inpAtoms_->numAtomsLocal() " << params.inpAtoms_->numAtomsLocal() << " localNNAtomNum " << localNNAtomNum <<std::endl;
        if (std::isfinite(fx) && std::isfinite(fy) && std::isfinite(fz)) {
            ofs << "GlobalIdx=" << gIdx
                << " local - Force=(" << fx << "," << fy << "," << fz << ")"
                << "\n";
            maxFLocal = std::fabs(fx) > maxFLocal ? std::fabs(fx) : maxFLocal;
            maxFLocal = std::fabs(fy) > maxFLocal ? std::fabs(fy) : maxFLocal;
            maxFLocal = std::fabs(fz) > maxFLocal ? std::fabs(fz) : maxFLocal;
        } else {
            ofs << "GlobalIdx=" << gIdx << " has invalid force values!\n";
        }
    }
    real maxF = (maxFLocal>maxFghost) ? maxFLocal : maxFghost;
    ofs << "maxF=" << maxF <<" maxFLocal=" << maxFLocal << " maxFGhost=" << maxFghost << "\n";
    // Flush & close (happens automatically in destructor)
    ofs.close();
#else
    for (size_t i = 0; i < params.inpAtoms_->numAtomsLocal(); i++) // local atoms
    {
        gIdx = params.inpAtoms_->globalIndex()[params.inpAtoms_->collectiveIndex()[i]];
        real fx = forces[indices[i]][0];
        real fy = forces[indices[i]][1];
        real fz = forces[indices[i]][2];
        // ofs << "GlobalIdx="<< gIdx << " local - Force=("<<
        // forces[indices[i]][0]<<","<< forces[indices[i]][1]<<","<< forces[indices[i]][2]<<")"<< "\n";
        // std::cout<< "Rank " << this->cr_->rankInDefaultCommunicator<< " Local atoms idxLookup_["<< i <<"]=" << idxLookup_[i] << " params.inpAtoms_->numAtomsLocal() " << params.inpAtoms_->numAtomsLocal() << " localNNAtomNum " << localNNAtomNum <<std::endl;
        if (std::isfinite(fx) && std::isfinite(fy) && std::isfinite(fz)) {
            ofs << "GlobalIdx=" << gIdx
                << " local - Force=(" << fx << "," << fy << "," << fz << ")"
                << "\n";
            //maxFLocal = std::fabs(forces[indices[i]][0]) > maxFLocal ? std::fabs(forces[indices[i]][0]) : maxFLocal;
            //maxFLocal = std::fabs(forces[indices[i]][1]) > maxFLocal ? std::fabs(forces[indices[i]][1]) : maxFLocal;
            //maxFLocal = std::fabs(forces[indices[i]][2]) > maxFLocal ? std::fabs(forces[indices[i]][2]) : maxFLocal;
        } else {
            ofs << "GlobalIdx=" << gIdx << " has invalid force values!\n";
        }
    }

    int iGhost;
    for (size_t i = params.inpAtoms_->numAtomsLocal(); i < Ntot; i++) // ghost atoms
    {
        iGhost = i - params.inpAtoms_->numAtomsLocal();
        gIdx = params.inpGhostAtoms_->globalIndex()[params.inpGhostAtoms_->collectiveIndex()[iGhost]];
        //ofs << "GlobalIdx="<< gIdx << " ghost - Force=("<<
        //forces[indices[i]][0]<<","<< forces[indices[i]][1]<<","<< forces[indices[i]][2]<<")"<< "\n";
        real fx = forces[indices[i]][0];
        real fy = forces[indices[i]][1];
        real fz = forces[indices[i]][2];
        if (std::isfinite(fx) && std::isfinite(fy) && std::isfinite(fz)) {
            ofs << "GlobalIdx=" << gIdx
                << " ghost - Force=(" << fx << "," << fy << "," << fz << ")"
                << "\n";
            maxFghost = std::fabs(forces[indices[i]][0]) > maxFghost ? std::fabs(forces[indices[i]][0]) : maxFghost;
            maxFghost = std::fabs(forces[indices[i]][1]) > maxFghost ? std::fabs(forces[indices[i]][1]) : maxFghost;
            maxFghost = std::fabs(forces[indices[i]][2]) > maxFghost ? std::fabs(forces[indices[i]][2]) : maxFghost;
        } else {
            ofs << "GlobalIdx=" << gIdx << " has invalid force values!\n";
        }
    }
    real maxF = (maxFLocal>maxFghost) ? maxFLocal : maxFghost;
    ofs << "maxF=" << maxF <<" maxFLocal=" << maxFLocal << " maxFGhost=" << maxFghost << "\n";
    // Flush & close (happens automatically in destructor)
    ofs.close();
#endif
}

void DeepmdModel::getOutputs(const NNPotParameters& params, std::vector<int>& indices, gmx_enerdata_t& enerd, const ArrayRef<RVec>& forces)
{
    if (!isInit_)
    {
        GMX_THROW(InternalError("Model not initialized before prepareInputs() was called."));
    }
    if (!outputReady_)
    {
        GMX_THROW(InternalError("Model outputs not ready before getOutputs() was called."));
    }

    const int Ntot = indices.size(); // local + ghost
    const int Nlocal = this->localAtomNum;

#if GMX_DEEPMD_INFERENCE_MULTI_MPI_COLLECTIVE

#if GMX_DEEPMD_INFERENCE_MULTI_MPI_COLLECTIVE_REBUILD_DD

    std::cout<< "GMX_DEEPMD_INFERENCE_MULTI_MPI_COLLECTIVE getOutputs - Rank " << this->cr_->rankInDefaultCommunicator << 
    " inferInfo_.totalNNAtomNum=" << inferInfo_.totalNNAtomNum << " inferInfo_.localNNAtomNum="<< inferInfo_.localNNAtomNum <<
    " inferInfo_.atomForce_.size()=" << inferInfo_.atomForce_.size()  << std::endl;
    
    std::vector<real> atomForcesGlobal(DIM * inferInfo_.totalNNAtomNum, 0.0);
    real localEnergy = 0;
    for (int i = 0; i < inferInfo_.totalLocalGhostNNAtomNum; ++i)
    {
        if(inferInfo_.localIdxes_[i] != -1){
            localEnergy += inferInfo_.atomEnergy_[i];
            atomForcesGlobal[DIM * inferInfo_.globalIdxes_[i] + 0] = inferInfo_.atomForce_[DIM * i + 0];
            atomForcesGlobal[DIM * inferInfo_.globalIdxes_[i] + 1] = inferInfo_.atomForce_[DIM * i + 1];
            atomForcesGlobal[DIM * inferInfo_.globalIdxes_[i] + 2] = inferInfo_.atomForce_[DIM * i + 2];
        }
    }

    enerd.term[F_ENNPOT] = localEnergy * e_dp2gmx * lambda;
    if (havePPDomainDecomposition(cr_)){
        gmx_sum(1, static_cast<real*>(&enerd.term[F_ENNPOT]), cr_);
    }
    // distribute forces
    if (havePPDomainDecomposition(cr_))
    {
        gmx_sum(3 * inferInfo_.totalNNAtomNum, static_cast<real*>(atomForcesGlobal.data()), cr_);
    }

    // accumulate forces only on local atoms
    
    for (int i = 0; i < inferInfo_.totalNNAtomNum; ++i)
    {
        // if value in lookup table is -1, the atom is not local
        if (indices[i] == -1) continue;
        forces[indices[i]][0] += atomForcesGlobal[i * DIM + 0] * f_dp2gmx * lambda; // convert to gromacs unit
        forces[indices[i]][1] += atomForcesGlobal[i * DIM + 1] * f_dp2gmx * lambda; // convert to gromacs unit
        forces[indices[i]][2] += atomForcesGlobal[i * DIM + 2] * f_dp2gmx * lambda; // convert to gromacs unit
    }

#else
    
    std::cout<< "GMX_DEEPMD_INFERENCE_MULTI_MPI_COLLECTIVE getOutputs - Rank " << this->cr_->rankInDefaultCommunicator << 
    " inferInfo_.totalNNAtomNum=" << inferInfo_.totalNNAtomNum << " inferInfo_.localNNAtomNum="<< inferInfo_.localNNAtomNum <<
    " inferInfo_.atomForce_.size()=" << inferInfo_.atomForce_.size()  << std::endl;
    
    real localEnergy = 0;
    for (int i = 0; i < inferInfo_.localNNAtomNum; ++i)
    {
        localEnergy += inferInfo_.atomEnergy_[i];
    }
    enerd.term[F_ENNPOT] = localEnergy * e_dp2gmx * lambda;
    if (havePPDomainDecomposition(cr_)){
        gmx_sum(1, static_cast<real*>(&enerd.term[F_ENNPOT]), cr_);
    }
    //writeForces(0,params,indices,forces);
    // accumulate forces only on local atoms
    for (int i = 0; i < inferInfo_.localNNAtomNum; ++i){
        for (int m = 0; m < DIM; ++m){
            forces[inferInfo_.localIdxes_[i]][m] += inferInfo_.atomForce_[i * DIM + m] * f_dp2gmx * lambda; // convert to gromacs unit
        }
    }
    //writeForces(1,params,indices,forces);

#endif

#else
    std::cout<< "GMX_DEEPMD_INFERENCE_MULTI_MPI = OFF getOutputs - Rank " << this->cr_->rankInDefaultCommunicator << 
    " inferInfo_.totalNNAtomNum=" << inferInfo_.totalNNAtomNum << " inferInfo_.localNNAtomNum="<< inferInfo_.localNNAtomNum <<
    " inferInfo_.atomForce_.size()=" << inferInfo_.atomForce_.size()  << std::endl;
    const bool modelOutputsForces = outputsForces();
    if (MAIN(cr_))
    {
        // set energy
        enerd.term[F_ENNPOT] = inferInfo_.energy_ * e_dp2gmx * lambda; 

        if (!modelOutputsForces)
        {
            GMX_THROW(InternalError("Model does not output forces, but getOutputs() was called."));
        }
    }

    // distribute forces
    if (havePPDomainDecomposition(cr_))
    {
        gmx_sum(3 * Ntot, static_cast<real*>(inferInfo_.atomForce_.data()), cr_);
    }

    // accumulate forces only on local atoms
    for (int m = 0; m < DIM; ++m)
    {
        for (int i = 0; i < Ntot; ++i)
        {
            // if value in lookup table is -1, the atom is not local
            if (indices[i] == -1)
            {
                continue;
            }
            forces[indices[i]][m] += inferInfo_.atomForce_[i * DIM + m] * f_dp2gmx * lambda; // convert to gromacs unit
        }
    }

#endif
    outputReady_ = false;
}

void DeepmdModel::setCommRec(const t_commrec* cr)
{
    cr_ = cr;

    if (!isInit_)
    {
        try
        {
            dp_ = std::make_unique<deepmd::DeepPot>(modelFileName_, getDevice(cr_));
        }
        GMX_CATCH_ALL_AND_EXIT_WITH_FATAL_ERROR;
        isInit_ = true; // now the model is initialized
    }
}

bool DeepmdModel::outputsForces() const
{
    if (!outputReady_)
    {
        GMX_THROW(InternalError("Model outputs not ready before modelOutputsForces() was called."));
    }
    return true;
}

int DeepmdModel::getDevice(const t_commrec* cr)
{

    if (cr == nullptr)
    {
        GMX_THROW(InternalError("Communication record is not set for DeepMD model."));
    }

    return cr->rankInDefaultCommunicator;

}


} // namespace gmx
