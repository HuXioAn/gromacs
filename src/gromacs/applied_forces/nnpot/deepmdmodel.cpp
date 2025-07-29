


#include <iostream>
#include <filesystem>
#include <memory>
#include <cstring>
#include <vector>
#include <tuple>

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

void  DeepmdModel::createNeighbList(const NNPotParameters& params)
{
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
#if GMX_DEEPMD_INFERENCE_MULTI_MPI
    if (havePPDomainDecomposition(cr_)){    
        inferInfo_.box_.clear();
    }
    int ago = 0;
    dp_->compute<real>(inferInfo_.energy_, inferInfo_.atomForce_, inferInfo_.virial_, inferInfo_.atomEnergy_, inferInfo_.atomVirial_,
            inferInfo_.atomPosition_, inferInfo_.atomType_ , inferInfo_.box_, inferInfo_.ghostNNAtomNum, inferInfo_.neighList, ago);
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
    for (size_t i = 0; i < params.inpAtoms_->numAtomsLocal(); i++) // local atoms
    {
        gIdx = params.inpAtoms_->globalIndex()[params.inpAtoms_->collectiveIndex()[i]];
        ofs << "GlobalIdx="<< gIdx << " local - Force=("<<forces[indices[i]][0]<<","<< forces[indices[i]][1]<<","<< forces[indices[i]][2]<<")"<< "\n";
        //std::cout<< "Rank " << this->cr_->rankInDefaultCommunicator<< " Local atoms idxLookup_["<< i <<"]=" << idxLookup_[i] << " params.inpAtoms_->numAtomsLocal() " << params.inpAtoms_->numAtomsLocal() << " localNNAtomNum " << localNNAtomNum <<std::endl;
        maxFLocal = abs(forces[indices[i]][0]) > maxFLocal ? abs(forces[indices[i]][0]) : maxFLocal;
        maxFLocal = abs(forces[indices[i]][1]) > maxFLocal ? abs(forces[indices[i]][1]) : maxFLocal;
        maxFLocal = abs(forces[indices[i]][2]) > maxFLocal ? abs(forces[indices[i]][2]) : maxFLocal;
    }

    int iGhost;
    for (size_t i = params.inpAtoms_->numAtomsLocal(); i < Ntot; i++) // ghost atoms
    {
        iGhost = i - params.inpAtoms_->numAtomsLocal();
        gIdx = params.inpGhostAtoms_->globalIndex()[params.inpGhostAtoms_->collectiveIndex()[iGhost]];
        ofs << "GlobalIdx="<< gIdx << " ghost - Force=("<<forces[indices[i]][0]<<","<< forces[indices[i]][1]<<","<< forces[indices[i]][2]<<")"<< "\n";
        maxFghost = abs(forces[indices[i]][0]) > maxFghost ? abs(forces[indices[i]][0]) : maxFghost;
        maxFghost = abs(forces[indices[i]][1]) > maxFghost ? abs(forces[indices[i]][1]) : maxFghost;
        maxFghost = abs(forces[indices[i]][2]) > maxFghost ? abs(forces[indices[i]][2]) : maxFghost;
    }
    real maxF = (maxFLocal>maxFghost) ? maxFLocal : maxFghost;
    ofs << "maxF=" << maxF <<" maxFLocal=" << maxFLocal << " maxFGhost=" << maxFghost << "\n";
    // Flush & close (happens automatically in destructor)
    ofs.close();
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

    const int N = indices.size(); // local + ghost
    const int Nlocal = this->localAtomNum;

    std::cout<< "Rank " << this->cr_->rankInDefaultCommunicator << " N=" << N << " Nlocal="<< Nlocal << std::endl;

#if GMX_DEEPMD_INFERENCE_MULTI_MPI
    real localEnergy = 0;
    for (int i = 0; i < localAtomNum; ++i)
    {
        localEnergy += inferInfo_.atomEnergy_[i];
    }
    enerd.term[F_ENNPOT] = localEnergy * e_dp2gmx * lambda;
    
    writeForces(0,params,indices,forces);
    const int NtotLocal = forces.size(); 
    std::vector<real> tmpForce_(3*NtotLocal,0.0);
    for(int i = 0; i<NtotLocal; ++i){
        tmpForce_[i * DIM] = forces[i][0];
        tmpForce_[i * DIM + 1] = forces[i][1];
        tmpForce_[i * DIM + 2] = forces[i][2];
        forces[i][0] = 0;
        forces[i][1] = 0;
        forces[i][2] = 0;
    }
    
    
    for (int i = 0; i < N; ++i){
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

    for (int i = Nlocal; i < N; ++i){
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
#else
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
        gmx_sum(3 * N, static_cast<real*>(inferInfo_.atomForce_.data()), cr_);
    }

    // accumulate forces only on local atoms
    for (int m = 0; m < DIM; ++m)
    {
        for (int i = 0; i < N; ++i)
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
