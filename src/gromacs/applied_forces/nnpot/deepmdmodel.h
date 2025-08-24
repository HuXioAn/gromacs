


#ifndef GMX_APPLIED_FORCES_DEEPMDMODULE_H
#define GMX_APPLIED_FORCES_DEEPMDMODULE_H
#pragma once
#include <string>
#include <vector>
#include <memory>
#include "gromacs/domdec/localatomset.h"
#include "gromacs/applied_forces/nnpot/nnpotmodel.h"
#include "gromacs/applied_forces/nnpot/nnpotoptions.h"
#include "deepmd/DeepPot.h"
#include "deepmd/neighbor_list.h"

namespace gmx
{

class MDLogger;

struct deepmdInferenceInfo {

    int localNNAtomNum;
    int ghostNNAtomNum;
    int totalLocalGhostNNAtomNum;
    int totalNNAtomNum;

    MPI_Comm comm = MPI_COMM_NULL;

    /* input */
    std::vector<real> atomPosition_;
    std::vector<real> atomPositionCollective_;

    // the atom types, in for the model 
    std::vector<int> atomType_;
    std::vector<int> atomTypeCollective_;

    // local and global indexes
    std::vector<int> localIdxes_;
    std::vector<int> globalIdxes_;

    bool pbcType_;

    std::vector<real> box_ = std::vector<real>(DIM*DIM, 0.0);

    // neighboring list
    deepmd::InputNlist neighList;
    real    rcutoff = -100;
    int*    ilist_        = nullptr;
    int*    numneigh_     = nullptr;
    int*    neigh_storage = nullptr;
    int**   firstneigh_   = nullptr;
    int     nloc_         = 0;
    int     maxNeighPerAtom_  = 0;
    int     maxlistSize_ = 0;

    /* output */
    double energy_ = 0.0;
    std::vector<real> atomForce_;
    std::vector<real> virial_;
    std::vector<real> atomEnergy_;
    std::vector<real> atomVirial_;

    void resizeNeighborList(const int nloc, const int maxNeighPerAtom) 
    {
        //if(nloc > nloc_ || maxNeighPerAtom > maxNeighPerAtom_){
            // Free old buffers if allocated
            if (neigh_storage) {
                delete[] neigh_storage;
                neigh_storage = nullptr;
            }
            if (firstneigh_) {
                delete[] firstneigh_;
                firstneigh_ = nullptr;
            }
            if (numneigh_) {
                delete[] numneigh_;
                numneigh_ = nullptr;
            }
            if (ilist_) {
                delete[] ilist_;
                ilist_ = nullptr;
            }

            // Allocate new buffers
            ilist_        = new int[nloc];
            numneigh_     = new int[nloc];
            firstneigh_   = new int*[nloc];
            neigh_storage = new int[nloc * maxNeighPerAtom];
        

        // Point each firstneigh_[i] to its segment in neigh_storage
        for (int i = 0; i < nloc; ++i) {
            firstneigh_[i] = neigh_storage + i * maxNeighPerAtom;
        }
        // update parameters
        nloc_ = nloc;
        maxNeighPerAtom_ = maxNeighPerAtom;
        maxlistSize_ = 0;
        // Update DeepMD InputNlist
        neighList.inum      = nloc;
        neighList.ilist     = ilist_;
        neighList.numneigh  = numneigh_;
        neighList.firstneigh = firstneigh_;
    }

    ~deepmdInferenceInfo() {
        if (neigh_storage) {
            delete[] neigh_storage;
            neigh_storage = nullptr;
        }
        if (firstneigh_) {
            delete[] firstneigh_;
            firstneigh_ = nullptr;
        }
        if (numneigh_) {
            delete[] numneigh_;
            numneigh_ = nullptr;
        }
        if (ilist_) {
            delete[] ilist_;
            ilist_ = nullptr;
        }
        // Clear the InputNlist pointers so they don't dangle
        neighList.ilist      = nullptr;
        neighList.numneigh   = nullptr;
        neighList.firstneigh = nullptr;
    }
};

/*! \brief
 * Class responsible for initiating and inference a deepmd model.
 * Inherits from NNPotModel.
 */
class DeepmdModel : public INNPotModel
{
public:
    /*! \brief Constructor for DeepmdModel.
     * \param[in] filename path to the Deepmd model file
     * \param[in] logger pointer to the MDLogger
     */
    DeepmdModel(const std::string& filename, const MDLogger* logger);

    ~DeepmdModel();

    //! Initialize the neural network model
    void initModel() override;

    // inout for the model, however, only save the reference to the inference info, for later compute 
    void prepareAtomPositions(std::vector<RVec>& positions) override;
    void prepareAtomNumbers(std::vector<int>& atomTypes) override;
    void prepareBox(matrix& box) override;
    void preparePbcType(PbcType& pbcType) override;

    void evaluateModel() override;
    
    void preProcessData(const NNPotParameters& params, std::vector<int>&) override;

    void getOutputs(std::vector<int>& indices, gmx_enerdata_t& enerd, const ArrayRef<RVec>& forces) override {}

    void getOutputs(const NNPotParameters& params, std::vector<int>& indices, gmx_enerdata_t& enerd, const ArrayRef<RVec>& forces) override;

    void writeForces(const int idx, const NNPotParameters& params, std::vector<int>& indices, const ArrayRef<RVec>& forces);

    //! Set communication record for possible communication of input/output data between ranks
    void setCommRec(const t_commrec* cr) override;

    //! helper function to check if model outputs forces
    bool outputsForces() const override;

private:
    //! determine which GPU to use depending on GMX_DEEPMD_DEVICE environment variable, -1 for cpu
    int getDevice(const t_commrec* cr);


private:
    std::string modelFileName_;
    std::unique_ptr<deepmd::DeepPot> dp_;

    //! flag to check if deepmd is initialized
    bool isInit_ = false;
    bool outputReady_ = false;


    //! pointer to the communication record
    const t_commrec* cr_ = nullptr;
    //! pointer to the MDLogger
    const MDLogger* logger_ = nullptr;

    deepmdInferenceInfo inferInfo_;

};

} // namespace gmx

#endif // GMX_APPLIED_FORCES_DEEPMDMODULE_H