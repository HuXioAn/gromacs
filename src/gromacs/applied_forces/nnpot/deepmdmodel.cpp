


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

#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)
#pragma message("GMX_DEEPMD_INFERENCE_MULTI_MPI = " STR(GMX_DEEPMD_INFERENCE_MULTI_MPI))


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
    outputReadyMain_ = false;
    outputReadyPara_ = false;
}

void DeepmdModel::prepareAtomPositions(std::vector<RVec>& positions)
{
    const int N = positions.size(); // local + ghost
    auto& atomPos = inferInfoMain_.atomPosition_;
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
    auto& atomType = inferInfoMain_.atomType_;
    atomType.clear();

    for (int i = 0; i < N; ++i)
    {
        atomType.push_back(atomTypes[i] - 1);
    }

}

void DeepmdModel::prepareAtomPositionsPara(std::vector<RVec>& positions)
{
    const int N = positions.size(); // local + ghost
    auto& atomPos = inferInfoPara_.atomPosition_;
    atomPos.clear();

    for (int i = 0; i < N; ++i)
    {
        atomPos.push_back(positions[i][0] / c_dp2gmx);
        atomPos.push_back(positions[i][1] / c_dp2gmx);
        atomPos.push_back(positions[i][2] / c_dp2gmx);
    }

}

void DeepmdModel::prepareAtomNumbersPara(std::vector<int>& atomTypes)
{
    const int N = atomTypes.size();
    auto& atomType = inferInfoPara_.atomType_;
    atomType.clear();

    for (int i = 0; i < N; ++i)
    {
        atomType.push_back(atomTypes[i] - 1);
    }

}

void DeepmdModel::prepareBox(matrix& box)
{
    // convert box to 1D vector
    auto& boxVec = inferInfoMain_.box_;
    boxVec.resize(DIM * DIM);
    for (int i = 0; i < DIM; ++i)
    {
        for (int j = 0; j < DIM; ++j)
        {
            boxVec[i * DIM + j] = box[i][j] / c_dp2gmx; // assume pbc = true
        }
    }

    // also prepare the box for parallel inference
    auto& boxVecPara = inferInfoPara_.box_;
    boxVecPara.resize(DIM * DIM);
    for (int i = 0; i < DIM; ++i)
    {
        for (int j = 0; j < DIM; ++j)
        {
            boxVecPara[i * DIM + j] = box[i][j] / c_dp2gmx; // assume pbc = true
        }
    }

}

void DeepmdModel::preparePbcType([[maybe_unused]] PbcType& pbcType)
{

    if (pbcType == PbcType::Xyz){ // all periodic
        inferInfoMain_.pbcType_ = true;
        inferInfoPara_.pbcType_ = true;

    } else if (pbcType == PbcType::No) { // no periodic
        inferInfoMain_.pbcType_ = false;
        inferInfoPara_.pbcType_ = false;
    }else {
        GMX_THROW(InconsistentInputError("Not supuorted PBC type for DeepMD model: " + std::to_string(static_cast<int>(pbcType))));
    }
    
}

static void build_inputnlist(deepmd::InputNlist& nlist,
                      const std::vector<real>& posi3,
                      const int localAtomNum,
                      float rcut) {

    assert(posi3.size() % 3 == 0);
    int natoms_total = posi3.size() / 3;

    std::vector<int> ilist; // local atom
    for (int i = 0; i < natoms_total; ++i) {
        if (i < localAtomNum) {
            ilist.push_back(i);
        }
    }
    int inum = ilist.size();

    // allocate internal buffers
    int* ilist_buf = new int[inum];
    int* numneigh_buf = new int[inum];
    int** firstneigh_buf = new int*[inum];

    const double rcut2 = rcut * rcut;

    for (int i = 0; i < inum; ++i) {
        ilist_buf[i] = ilist[i];
        int ii = ilist[i];

        std::vector<int> neighs;
        for (int j = 0; j < natoms_total; ++j) {
            if (j == ii) continue;

            double dx = posi3[3*j + 0] - posi3[3*ii + 0];
            double dy = posi3[3*j + 1] - posi3[3*ii + 1];
            double dz = posi3[3*j + 2] - posi3[3*ii + 2];
            double r2 = dx*dx + dy*dy + dz*dz;
            if (r2 < rcut2) {
                neighs.push_back(j);  
            }
        }

        numneigh_buf[i] = neighs.size();
        int* neigh_array = new int[neighs.size()];
        std::copy(neighs.begin(), neighs.end(), neigh_array);
        firstneigh_buf[i] = neigh_array;
    }

    nlist.inum = inum;
    nlist.ilist = ilist_buf;
    nlist.numneigh = numneigh_buf;
    nlist.firstneigh = firstneigh_buf;
}


static void free_inputnlist(deepmd::InputNlist& nlist) {
    for (int i = 0; i < nlist.inum; ++i) {
        delete[] nlist.firstneigh[i];
    }
    delete[] nlist.firstneigh;
    delete[] nlist.numneigh;
    delete[] nlist.ilist;
}

void DeepmdModel::evaluateModel()
{
    if (!isInit_)
    {
        GMX_THROW(InternalError("deepmd not initialized before evaluateModel() was called."));
    }

    // periodic boundary conditions
    if (!inferInfoMain_.pbcType_)
    {
        inferInfoMain_.box_.resize(0); // no box needed
    }

    GMX_ASSERT(inferInfoMain_.atomPosition_.size() / DIM == inferInfoMain_.atomType_.size(),
               "Number of atom positions and atom types must match.");

    const int N = inferInfoMain_.atomType_.size();


    if (MAIN(cr_)){
        dp_->compute<real>(inferInfoMain_.energy_, inferInfoMain_.atomForce_, inferInfoMain_.virial_, inferInfoMain_.atomEnergy_, inferInfoMain_.atomVirial_,
            inferInfoMain_.atomPosition_, inferInfoMain_.atomType_ , inferInfoMain_.box_);   
    }
    else{
        inferInfoMain_.energy_= 0.0;
        inferInfoMain_.atomForce_.assign(DIM*N,0.0);
        inferInfoMain_.atomEnergy_.assign(N,0.0);
    }

    outputReadyMain_ = true;
}

void DeepmdModel::getOutputs(std::vector<int>& indices, gmx_enerdata_t& enerd, const ArrayRef<RVec>& forces)
{
    if (!isInit_)
    {
        GMX_THROW(InternalError("Model not initialized before prepareInputs() was called."));
    }
    if (!outputReadyMain_)
    {
        GMX_THROW(InternalError("Model outputs not ready before getOutputs() was called."));
    }

    const int N = indices.size(); // local + ghost

    const bool modelOutputsForces = outputsForces();
    if (MAIN(cr_))
    {
        // set energy
        enerd.term[F_ENNPOT] = inferInfoMain_.energy_ * e_dp2gmx * lambda;

        if (!modelOutputsForces)
        {
            GMX_THROW(InternalError("Model does not output forces, but getOutputs() was called."));
        }
    }

    // distribute forces
    if (havePPDomainDecomposition(cr_))
    {
        gmx_sum(3 * N, static_cast<real*>(inferInfoMain_.atomForce_.data()), cr_);
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
            forces[indices[i]][m] += inferInfoMain_.atomForce_[i * DIM + m] * f_dp2gmx * lambda; // convert to gromacs unit
        }
    }


    outputReadyMain_ = false;
}




void DeepmdModel::evaluateModelPara()
{
    if (!isInit_)
    {
        GMX_THROW(InternalError("deepmd not initialized before evaluateModel() was called."));
    }


    // periodic boundary conditions
    if (!inferInfoPara_.pbcType_)
    {
        inferInfoPara_.box_.resize(0); // no box needed
    }

    GMX_ASSERT(inferInfoPara_.atomPosition_.size() / DIM == inferInfoPara_.atomType_.size(),
               "Number of atom positions and atom types must match.");

    const int N = inferInfoPara_.atomType_.size();

    // nlist
    // build_inputnlist(inferInfoPara_.inputNlist_, inferInfoPara_.atomPosition_, localAtomNum, 0.6 / c_dp2gmx);

    dp_->compute<real>(inferInfoPara_.energy_, inferInfoPara_.atomForce_, inferInfoPara_.virial_, inferInfoPara_.atomEnergy_, inferInfoPara_.atomVirial_,
            inferInfoPara_.atomPosition_, inferInfoPara_.atomType_ , inferInfoPara_.box_, N-localAtomNum, inferInfoPara_.inputNlist_, 0);

    //  dp_->compute<real>(inferInfoPara_.energy_, inferInfoPara_.atomForce_, inferInfoPara_.virial_, inferInfoPara_.atomEnergy_, inferInfoPara_.atomVirial_,
    //         inferInfoPara_.atomPosition_, inferInfoPara_.atomType_ , inferInfoPara_.box_);

    // free_inputnlist(inferInfoPara_.inputNlist_);


    outputReadyPara_ = true;
}

void DeepmdModel::getOutputsPara(std::vector<int>& indices, gmx_enerdata_t& enerd, const ArrayRef<RVec>& forces)
{
    if (!isInit_)
    {
        GMX_THROW(InternalError("Model not initialized before prepareInputs() was called."));
    }
    if (!outputReadyPara_)
    {
        GMX_THROW(InternalError("Model outputs not ready before getOutputs() was called."));
    }

    const int N = indices.size(); // local + ghost
    const int Nlocal = this->localAtomNum;

    real localEnergy = 0.0;
    for (int i = 0; i < Nlocal; ++i)
    {
        localEnergy += inferInfoPara_.atomEnergy_[i];
    }

    enerd.term[F_ENNPOT] = localEnergy * e_dp2gmx * lambda;

    inferInfoPara_.ghostForceAggregation_.assign(3 * this->wholeSystemAtomNum, 0.0);


    for (int i = 0; i < N; ++i)
    {
        const auto& idx = this->idxLookupGlobalParaPtr_->at(i);

        inferInfoPara_.ghostForceAggregation_[idx * DIM]     = inferInfoPara_.atomForce_[i * DIM];
        inferInfoPara_.ghostForceAggregation_[idx * DIM + 1] = inferInfoPara_.atomForce_[i * DIM + 1];
        inferInfoPara_.ghostForceAggregation_[idx * DIM + 2] = inferInfoPara_.atomForce_[i * DIM + 2];
    }

    if (havePPDomainDecomposition(cr_))
    {
        gmx_sum(inferInfoPara_.ghostForceAggregation_.size(), static_cast<real*>(inferInfoPara_.ghostForceAggregation_.data()), cr_);
    }


    for (int i = 0; i < Nlocal; ++i){
        const auto& idxLocal = indices[i];
        const auto& idxGlobal = this->idxLookupGlobalParaPtr_->at(i);

        forces[idxLocal][0] += ((inferInfoPara_.ghostForceAggregation_[idxGlobal * DIM]) * f_dp2gmx * lambda);
        forces[idxLocal][1] += ((inferInfoPara_.ghostForceAggregation_[idxGlobal * DIM + 1]) * f_dp2gmx * lambda);
        forces[idxLocal][2] += ((inferInfoPara_.ghostForceAggregation_[idxGlobal * DIM + 2]) * f_dp2gmx * lambda);
    }

    outputReadyPara_ = false;
}

void DeepmdModel::compareOutput()
{

    std::cerr << "Comparing outputs between main and parallel inference..." << std::endl;

    std::ofstream outFileMain("./mainOutputLog.log", std::ios::out | std::ios::app);

    int rank = getDevice(cr_);

    std::ofstream outFilePara("./paraOutputLog" + std::to_string(rank) + ".log", std::ios::out | std::ios::app);

    // main

    // if (MAIN(cr_))
    // {
    //     // energy
    //     outFileMain << "MAIN Energy: " << inferInfoMain_.energy_ << std::endl;

    //     // force

    //     outFileMain << "MAIN Forces: " << inferInfoMain_.atomForce_.size() / DIM << " atoms" << std::endl;

    //     for (int i = 0; i < inferInfoMain_.atomForce_.size() / DIM; ++i)
    //     {
    //         outFileMain << "Atom " << idxLookupGlobalMainPtr_->at(i) << ": "
    //                     << inferInfoMain_.atomForce_[i * DIM] << ", "
    //                   << inferInfoMain_.atomForce_[i * DIM + 1] << ", "
    //                   << inferInfoMain_.atomForce_[i * DIM + 2] << std::endl;
    //     }
    // }

    // // parallel


    if (MAIN(cr_))
    {

        // compiute MSE 
        double mse = 0.0;
        for (int i = 0; i < inferInfoMain_.atomForce_.size() / DIM; ++i)
        {
            mse += (inferInfoMain_.atomForce_[i * DIM] - inferInfoPara_.ghostForceAggregation_[idxLookupGlobalMainPtr_->at(i) * DIM]) * 
                    (inferInfoMain_.atomForce_[i * DIM] - inferInfoPara_.ghostForceAggregation_[idxLookupGlobalMainPtr_->at(i) * DIM]) +
                   (inferInfoMain_.atomForce_[i * DIM + 1] - inferInfoPara_.ghostForceAggregation_[idxLookupGlobalMainPtr_->at(i) * DIM + 1]) *
                    (inferInfoMain_.atomForce_[i * DIM + 1] - inferInfoPara_.ghostForceAggregation_[idxLookupGlobalMainPtr_->at(i) * DIM + 1]) +
                   (inferInfoMain_.atomForce_[i * DIM + 2] - inferInfoPara_.ghostForceAggregation_[idxLookupGlobalMainPtr_->at(i) * DIM + 2]) *
                    (inferInfoMain_.atomForce_[i * DIM + 2] - inferInfoPara_.ghostForceAggregation_[idxLookupGlobalMainPtr_->at(i) * DIM + 2]);
        }
        mse /= (inferInfoMain_.atomForce_.size() / DIM);

        outFileMain << "MAIN Forces: " << inferInfoMain_.atomForce_.size() / DIM << " atoms" << " MSE: " << mse << std::endl;


        for (int i = 0; i < inferInfoMain_.atomForce_.size() / DIM; ++i)
        {
            outFileMain << "Atom " << idxLookupGlobalMainPtr_->at(i) << " MAIN : "
                        << inferInfoMain_.atomForce_[i * DIM] << ", "
                      << inferInfoMain_.atomForce_[i * DIM + 1] << ", "
                      << inferInfoMain_.atomForce_[i * DIM + 2] << ", PARA AGGRE: "
                      << inferInfoPara_.ghostForceAggregation_[idxLookupGlobalMainPtr_->at(i) * DIM] << ", "
                      << inferInfoPara_.ghostForceAggregation_[idxLookupGlobalMainPtr_->at(i) * DIM + 1] << ", "
                      << inferInfoPara_.ghostForceAggregation_[idxLookupGlobalMainPtr_->at(i) * DIM + 2]

                      << std::endl;
        }
    }

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
    if (!outputReadyMain_)
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
