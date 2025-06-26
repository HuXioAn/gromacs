/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright 2024- The GROMACS Authors
 * and the project initiators Erik Lindahl, Berk Hess and David van der Spoel.
 * Consult the AUTHORS/COPYING files and https://www.gromacs.org for details.
 *
 * GROMACS is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License
 * as published by the Free Software Foundation; either version 2.1
 * of the License, or (at your option) any later version.
 *
 * GROMACS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with GROMACS; if not, see
 * https://www.gnu.org/licenses, or write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA.
 *
 * If you want to redistribute modifications to GROMACS, please
 * consider that scientific software is very special. Version
 * control is crucial - bugs must be traceable. We will be happy to
 * consider code for inclusion in the official distribution, but
 * derived work must not be called official GROMACS. Details are found
 * in the README & COPYING files - if they are missing, get the
 * official version at https://www.gromacs.org.
 *
 * To help us fund GROMACS development, we humbly ask that you cite
 * the research papers on the package. Check out https://www.gromacs.org.
 */
/*! \internal \file
 * \brief
 * Declares deepmd force provider class
 *
 * \author 
 * \ingroup module_applied_forces
 */
#ifndef GMX_APPLIED_FORCES_DEEPMDFORCEPROVIDER_H
#define GMX_APPLIED_FORCES_DEEPMDFORCEPROVIDER_H

#include <memory>
#include "gromacs/mdtypes/iforceprovider.h"
#include "gromacs/math/vectypes.h"



#include "deepmd/DeepPot.h"

enum class PbcType;


namespace gmx
{
struct deepmdOptions;
struct deepmdInferenceInfo;

class deepmdForceProvider final : public IForceProvider
{
public:

    /*! \brief
     * Constructor for deepmd force provider, which initializes the deepmd::DeepPot object
     * provide the information needed to calculate forces.
     *
     * \param[in] options  Options for the deepmd force provider
     * \param[in] info     Additional info from Gromacs for model inference
     */
    deepmdForceProvider(const deepmdOptions& options, deepmdInferenceInfo& info, const MDLogger*& logger);
    ~deepmdForceProvider();
    
    // fixed interface for IForceProvider
    void calculateForces(const ForceProviderInput& forceProviderInput,
                         ForceProviderOutput*      forceProviderOutput) override;

private:
    std::unique_ptr<deepmd::DeepPot> dp_;

    const deepmdOptions& options;

    // get it from the deepmdModule, updated by notification subscription
    deepmdInferenceInfo& inferenceInfo; 

    // from the module
    const MDLogger*& logger_;

};

/*! \brief
 * Struct to hold part of the necessary data for inference
 */
struct deepmdInferenceInfo {

    /* input */
    std::vector<real> atomPosition_;

    // the atom types, in for the model 
    std::vector<int> atomType_;
    // mapping between gromacs and models
    std::vector<int> idxLookup_;

    std::unique_ptr<PbcType> pbcType_;

    std::vector<real> box_ = std::vector<real>(DIM*DIM, 0.0);

    /* output */
    double energy_ = 0.0;
    std::vector<real> atomForce_;
    std::vector<real> virial_;


};



} // namespace gmx

#endif // GMX_APPLIED_FORCES_DEEPMDFORCEPROVIDER_H
