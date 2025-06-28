


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
    try
    {
        dp_ = std::make_unique<deepmd::DeepPot>(modelFileName_, getDevice());
    }
    GMX_CATCH_ALL_AND_EXIT_WITH_FATAL_ERROR;
    isInit_ = true;
}

void DeepmdModel::prepareAtomPositions(std::vector<RVec>& positions)
{
    const int N = positions.size();
    auto& atomPos = inferInfo_.atomPosition_;
    atomPos.resize(3 * N);

    if constexpr (sizeof(RVec) == 3 * sizeof(real)) { // nopadding
        const real *rawPtr = reinterpret_cast<const real *>(positions.data());
        std::memcpy(atomPos.data(), rawPtr, 3 * N * sizeof(real));
    } else {
        for (int i=0; i < N; i++) {
            atomPos[i * 3] = positions[i][0];
            atomPos[i * 3 + 1] = positions[i][1];
            atomPos[i * 3 + 2] = positions[i][2];
        }
    }
    
}

void DeepmdModel::prepareAtomNumbers(std::vector<int>& atomTypes)
{
    const int N = atomTypes.size();
    auto& atomType = inferInfo_.atomType_;
    atomType.resize(N);
    // usually it won't change
    std::memcpy(atomType.data(), atomTypes.data(), N * sizeof(int));
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

    dp_->compute<real>(inferInfo_.energy_, inferInfo_.atomForce_, inferInfo_.virial_, 
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

    const bool modelOutputsForces = outputsForces();

    const int     N           = indices.size();
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

    outputReady_ = false;
}

void DeepmdModel::setCommRec(const t_commrec* cr)
{
    cr_ = cr;
}

bool DeepmdModel::outputsForces() const
{
    if (!outputReady_)
    {
        GMX_THROW(InternalError("Model outputs not ready before modelOutputsForces() was called."));
    }
    return true;
}

int DeepmdModel::getDevice()
{
    // check if environment variable GMX_NN_DEVICE is set (should be something like cuda:0 or cpu)
    if (const char* env = std::getenv("GMX_DEEPMD_DEVICE"))
    {   
        // judge if it is a number
        if (std::isdigit(env[0]))
        {
            int device = std::atoi(env);
            return device; // return the device number
        }
        else if (std::string(env) == "-1")
        {
            return -1; // cpu
        }
        else
        {
            GMX_THROW(InternalError("Invalid GMX_DEEPMD_DEVICE value: " + std::string(env)));
        }
        
    }

    return -1; // default to cpu if not set

}


} // namespace gmx
