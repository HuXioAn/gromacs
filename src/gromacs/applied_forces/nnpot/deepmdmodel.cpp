


#include <iostream>
#include <filesystem>
#include <memory>
#include <cstring>
#include <vector>
#include <tuple>

#include "gromacs/gmxlib/network.h"
#include "gromacs/mdtypes/commrec.h"
#include "gromacs/mdtypes/enerdata.h"
#include "gromacs/mdtypes/forceoutput.h"
#include "gromacs/pbcutil/pbc.h"
#include "gromacs/utility/exceptions.h"
#include "gromacs/utility/logger.h"

#include "deepmdmodel.h"

namespace gmx
{

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

    dp_->compute<real>(inferInfo_.energy_, inferInfo_.atomForce_, inferInfo_.virial_, inferInfo_.atomEnergy_, inferInfo_.atomVirial_,
            inferInfo_.atomPosition_, inferInfo_.atomType_ , inferInfo_.box_);


    outputReady_ = true;
}

void DeepmdModel::getOutputs(std::vector<int>& indices, gmx_enerdata_t& enerd, const ArrayRef<RVec>& forces)
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

    real localEnergy = 0;
    for (int i = 0; i < localAtomNum; ++i)
    {
        localEnergy += inferInfo_.atomEnergy_[i];
    }
    enerd.term[F_ENNPOT] = localEnergy * e_dp2gmx * lambda; 

    for (int i = 0; i < N; ++i){
        for (int m = 0; m < DIM; ++m)
        forces[indices[i]][m] = inferInfo_.atomForce_[i * DIM + m] * f_dp2gmx * lambda; 
    }
    
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
