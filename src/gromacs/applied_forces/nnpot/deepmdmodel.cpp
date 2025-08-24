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

#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)
#pragma message("GMX_DEEPMD_INFERENCE_MULTI_MPI = " STR(GMX_DEEPMD_INFERENCE_MULTI_MPI))
#pragma message("GMX_DEEPMD_INFERENCE_MULTI_MPI_GHOST = " STR(GMX_DEEPMD_INFERENCE_MULTI_MPI_GHOST))
#pragma message("GMX_DEEPMD_INFERENCE_MULTI_MPI_COLLECTIVE = " STR(GMX_DEEPMD_INFERENCE_MULTI_MPI_COLLECTIVE))

#define GMX_DEEPMD_INFERENCE_MULTI_MPI_COLLECTIVE_BUILD_LIST 0

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
static inline bool min_image_duplicate_if_needed(
    const std::array<real,3>& ri,     // atom i position
    const std::array<real,3>& rj,     // atom j position (base cell)
    const std::vector<real>& box,
    const real rcutSqrd,
    bool& inCutOffDistance,
    std::array<real,3>& out_shifted, // shifted coords for duplicate (if any)
    std::array<int,3>& out_n)  
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

    const bool need_dup = inCutOffDistance && (nx != 0 || ny != 0 || nz != 0);
    // Coordinates of the image that realizes the minimum image
    out_shifted = { rj[0] - nx*box[0], rj[1] - ny*box[DIM + 1], rj[2] - nz*box[2 * DIM + 2] };
    out_n   = { nx, ny, nz };
        
    return need_dup;
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

using ImgSet = std::unordered_set<ImgKey, ImgKeyHash>;

static inline bool add_atom_if_new(ImgSet& imgSet, int j, int nx, int ny, int nz) {
    ImgKey key{j, nx, ny, nz};
    auto [it, inserted] = imgSet.insert(key);
    return inserted;               // true  -> wasn't there (now added)
                                   // false -> already present
}

void  DeepmdModel::preProcessData(const NNPotParameters& params, std::vector<int>& idxLookup)
{
#if GMX_DEEPMD_INFERENCE_MULTI_MPI_GHOST
    inferInfo_.localNNAtomNum = params.inpAtoms_->numAtomsLocal();
    inferInfo_.ghostNNAtomNum = params.inpGhostAtoms_->numAtomsLocal();
    inferInfo_.resizeNeighborList(inferInfo_.localNNAtomNum,int(inferInfo_.localNNAtomNum+inferInfo_.ghostNNAtomNum));
    //inferInfo_.neighList.world     = reinterpret_cast<void*>(&cr_->mpiDefaultCommunicator);
    inferInfo_.comm = MPI_COMM_SELF;
    inferInfo_.neighList.world     = reinterpret_cast<void*>(&inferInfo_.comm);
    inferInfo_.neighList.nswap     = 0;
    inferInfo_.neighList.sendnum   = nullptr;
    inferInfo_.neighList.recvnum   = nullptr;
    inferInfo_.neighList.firstrecv = nullptr;
    inferInfo_.neighList.sendlist  = nullptr;
    inferInfo_.neighList.sendproc  = nullptr;
    inferInfo_.neighList.recvproc  = nullptr;
    inferInfo_.rcutoff = 8.0; // in Angstrom
    int err = deepmd::build_nlist_cpu(inferInfo_.neighList, &inferInfo_.maxlistSize_,
                            inferInfo_.atomPosition_.data(),
                            inferInfo_.localNNAtomNum,
                            inferInfo_.localNNAtomNum + inferInfo_.ghostNNAtomNum,
                            inferInfo_.maxNeighPerAtom_,
                            float(inferInfo_.rcutoff) );
    if (err == 1) {
        std::ostringstream msg;
        msg << "DeepMD neighbor‐list overflow: "
            << "required per‐atom capacity = " << inferInfo_.maxlistSize_
            << ", but allocated maxNeighPerAtom = " << inferInfo_.maxNeighPerAtom_
            << ". Please increase mem_size and retry.";
        // Option A: print and exit
        std::cerr << msg.str() << std::endl;
        std::exit(EXIT_FAILURE);
    }

#elif GMX_DEEPMD_INFERENCE_MULTI_MPI_COLLECTIVE

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

    std::cout<< "GMX_DEEPMD_INFERENCE_MULTI_MPI_COLLECTIVE Create Local- Rank " << this->cr_->rankInDefaultCommunicator << 
    " inferInfo_.atomTypeCollective_size()=" << inferInfo_.atomTypeCollective_.size() << " inferInfo_.localNNAtomNum="<< inferInfo_.localNNAtomNum <<
    " idxLookup.size()=" << idxLookup.size() << std::endl;
    
    const real ghostCutOff = 7.0; // in Angstrom
    const real twiceRCutoffSqrd = (real)4.0 * ghostCutOff * ghostCutOff;    
 
    // add ghost and duplicated atoms to implement periodic bcs loop over local + non local atoms
    ImgSet imgSet;
    imgSet.reserve(4096);   // optional: avoid rehash during the loop
    std::array<real,DIM> out_shifted;
    std::array<int,DIM> out_n;
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
                
                bool inCutOffDistance = false;
                const bool duplicate = min_image_duplicate_if_needed(ri, rj, inferInfo_.box_, twiceRCutoffSqrd, inCutOffDistance, out_shifted, out_n);
                
                if (inCutOffDistance){
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

    std::cout<< "GMX_DEEPMD_INFERENCE_MULTI_MPI_COLLECTIVE Create Ghost- Rank " << this->cr_->rankInDefaultCommunicator << 
    " inferInfo_.totalLocalGhostNNAtomNum=" << inferInfo_.totalLocalGhostNNAtomNum << " inferInfo_.ghostNNAtomNum="<< inferInfo_.ghostNNAtomNum << std::endl;

#if GMX_DEEPMD_INFERENCE_MULTI_MPI_COLLECTIVE_BUILD_LIST
    inferInfo_.resizeNeighborList(inferInfo_.localNNAtomNum,int(inferInfo_.localNNAtomNum+inferInfo_.ghostNNAtomNum));
    //inferInfo_.neighList.world     = reinterpret_cast<void*>(&cr_->mpiDefaultCommunicator);
    inferInfo_.comm = MPI_COMM_SELF;
    inferInfo_.neighList.world     = reinterpret_cast<void*>(&inferInfo_.comm);
    inferInfo_.neighList.nswap     = 0;
    inferInfo_.neighList.sendnum   = nullptr;
    inferInfo_.neighList.recvnum   = nullptr;
    inferInfo_.neighList.firstrecv = nullptr;
    inferInfo_.neighList.sendlist  = nullptr;
    inferInfo_.neighList.sendproc  = nullptr;
    inferInfo_.neighList.recvproc  = nullptr;
    inferInfo_.rcutoff = 6.5; // in Angstrom
    int err = deepmd::build_nlist_cpu(inferInfo_.neighList, &inferInfo_.maxlistSize_,
                            inferInfo_.atomPositionCollective_.data(),
                            inferInfo_.localNNAtomNum,
                            inferInfo_.localNNAtomNum + inferInfo_.ghostNNAtomNum,
                            inferInfo_.maxNeighPerAtom_,
                            real(inferInfo_.rcutoff) );
    if (err == 1) {
        std::ostringstream msg;
        msg << "DeepMD neighbor‐list overflow: "
            << "required per‐atom capacity = " << inferInfo_.maxlistSize_
            << ", but allocated maxNeighPerAtom = " << inferInfo_.maxNeighPerAtom_
            << ". Please increase mem_size and retry.";
        // Option A: print and exit
        std::cerr << msg.str() << std::endl;
        std::exit(EXIT_FAILURE);
    }
#endif


#else

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
    //std::cout<< "Rank " << this->cr_->rankInDefaultCommunicator << " inferInfo_.atomType_.size()=" << N  << std::endl;
#if GMX_DEEPMD_INFERENCE_MULTI_MPI_GHOST

    if (havePPDomainDecomposition(cr_)){    
        inferInfo_.box_.clear();
    }
    int ago = 0;
    dp_->compute<real>(inferInfo_.energy_, inferInfo_.atomForce_, inferInfo_.virial_, inferInfo_.atomEnergy_, inferInfo_.atomVirial_,
            inferInfo_.atomPosition_, inferInfo_.atomType_ , inferInfo_.box_, inferInfo_.ghostNNAtomNum, inferInfo_.neighList, ago);

#elif GMX_DEEPMD_INFERENCE_MULTI_MPI_COLLECTIVE

    inferInfo_.box_.clear();
    int ago = 0;
    inferInfo_.atomForce_.clear();
    inferInfo_.atomEnergy_.clear();
    inferInfo_.atomVirial_.clear();
#if GMX_DEEPMD_INFERENCE_MULTI_MPI_COLLECTIVE_BUILD_LIST
    dp_->compute<real>(inferInfo_.energy_, inferInfo_.atomForce_, inferInfo_.virial_, inferInfo_.atomEnergy_, inferInfo_.atomVirial_,
            inferInfo_.atomPositionCollective_, inferInfo_.atomTypeCollective_ , inferInfo_.box_, inferInfo_.ghostNNAtomNum, inferInfo_.neighList, ago);
#else
    dp_->compute<real>(inferInfo_.energy_, inferInfo_.atomForce_, inferInfo_.virial_, inferInfo_.atomEnergy_, inferInfo_.atomVirial_,
        inferInfo_.atomPositionCollective_, inferInfo_.atomTypeCollective_ , inferInfo_.box_);
#endif

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

    //std::cout<< "Rank " << this->cr_->rankInDefaultCommunicator << " Ntot=" << Ntot << " Nlocal="<< Nlocal << std::endl;

#if GMX_DEEPMD_INFERENCE_MULTI_MPI_GHOST
    real localEnergy = 0;
    for (int i = 0; i < localAtomNum; ++i)
    {
        localEnergy += inferInfo_.atomEnergy_[i];
    }
    enerd.term[F_ENNPOT] = localEnergy * e_dp2gmx * lambda;
    
    writeForces(0,params,indices,forces);
    const int NtotLocal = forces.size(); 
    std::vector<real> tmpForce_(DIM*NtotLocal,0.0);
    for(int i = 0; i<NtotLocal; ++i){
        tmpForce_[i * DIM] = forces[i][0];
        tmpForce_[i * DIM + 1] = forces[i][1];
        tmpForce_[i * DIM + 2] = forces[i][2];
        forces[i][0] = 0;
        forces[i][1] = 0;
        forces[i][2] = 0;
    }
    
    for (int i = 0; i < Ntot; ++i){
        forces[indices[i]][0] += inferInfo_.atomForce_[i * DIM] * f_dp2gmx * lambda;
        forces[indices[i]][1] += inferInfo_.atomForce_[i * DIM + 1] * f_dp2gmx * lambda;
        forces[indices[i]][2] += inferInfo_.atomForce_[i * DIM + 2] * f_dp2gmx * lambda;
    }
    writeForces(1,params,indices,forces);
    if (havePPDomainDecomposition(cr_))
    {
        gmx_sum(1, static_cast<real*>(&enerd.term[F_ENNPOT]), cr_);
        dd_move_f_specialForces(cr_->dd, forces);
        //gmx::ArrayRef<gmx::RVec> emptyShiftForces(nullptr, 0);
        //cr_->dd->haloExchange->moveF(forces, emptyShiftForces);
    }
    writeForces(2,params,indices,forces);

    for (int i = Nlocal; i < Ntot; ++i){
        forces[indices[i]][0] = 0;
        forces[indices[i]][1] = 0;
        forces[indices[i]][2] = 0;
    }
    writeForces(3,params,indices,forces);
    
    for(int i = 0; i<NtotLocal; ++i){
        forces[i][0] += tmpForce_[i * DIM];
        forces[i][1] += tmpForce_[i * DIM + 1];
        forces[i][2] += tmpForce_[i * DIM + 2];
    }
    
    std::cout<< " In getOutput deepmd: Rank " << this->cr_->rankInDefaultCommunicator << " forces.size() " << forces.size() << " havePPDomainDecomposition(cr_)="<< havePPDomainDecomposition(cr_)<< std::endl;
#elif GMX_DEEPMD_INFERENCE_MULTI_MPI_COLLECTIVE
    
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
    writeForces(0,params,indices,forces);
    // accumulate forces only on local atoms
    for (int i = 0; i < inferInfo_.localNNAtomNum; ++i){
        for (int m = 0; m < DIM; ++m){
            forces[inferInfo_.localIdxes_[i]][m] += inferInfo_.atomForce_[i * DIM + m] * f_dp2gmx * lambda; // convert to gromacs unit
        }
    }
    writeForces(1,params,indices,forces);

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
