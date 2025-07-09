


#ifndef GMX_APPLIED_FORCES_DEEPMDMODULE_H
#define GMX_APPLIED_FORCES_DEEPMDMODULE_H

#include <string>
#include <vector>
#include <memory>
#include "gromacs/applied_forces/nnpot/nnpotmodel.h"
#include "deepmd/DeepPot.h"

namespace gmx
{

class MDLogger;

struct deepmdInferenceInfo {

    /* input */
    std::vector<real> atomPosition_;

    // the atom types, in for the model 
    std::vector<int> atomType_;

    bool pbcType_;

    std::vector<real> box_ = std::vector<real>(DIM*DIM, 0.0);

    /* output */
    double energy_ = 0.0;
    std::vector<real> atomForce_;
    std::vector<real> virial_;
    std::vector<real> atomEnergy_;
    std::vector<real> atomVirial_;

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
    void getOutputs(std::vector<int>& indices, gmx_enerdata_t& enerd, const ArrayRef<RVec>& forces) override;

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