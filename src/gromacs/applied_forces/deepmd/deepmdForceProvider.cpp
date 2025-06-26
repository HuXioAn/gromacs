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
 * Implementation of the deepmd force provider class
 *
 * \author 
 * \ingroup module_applied_forces
 */


#include "gromacs/domdec/domdec.h"
#include "gromacs/domdec/domdec_struct.h"
#include "gromacs/math/units.h"
#include "gromacs/mdrunutility/handlerestart.h"
#include "gromacs/mdtypes/commrec.h"
#include "gromacs/mdtypes/enerdata.h"
#include "gromacs/mdtypes/forceoutput.h"
#include "gromacs/utility/exceptions.h"

#include "deepmdForceProvider.h"
#include "deepmdOptions.h"

#include <filesystem>
#include <string>


namespace gmx
{

deepmdForceProvider::~deepmdForceProvider() = default;
deepmdForceProvider::deepmdForceProvider(const deepmdOptions& options, deepmdInferenceInfo& info, const MDLogger*& logger) 
                                        : options(options), inferenceInfo(info), logger_(logger)
{   
    auto& modelPath = options.modelFile;

    if (std::filesystem::exists(modelPath)) {
        // initialize the deepot
        dp_ = std::make_unique<deepmd::DeepPot>(modelPath); 
        GMX_LOG(logger_->info).appendText("deepmd: deePot initialized");

    } else {
        GMX_THROW(FileIOError("deepmd: the provided model file path doesn't exist: " + modelPath));
    }
    
    
}



void deepmdForceProvider::calculateForces(const ForceProviderInput& forceProviderInput,
                                          ForceProviderOutput*      forceProviderOutput)
{

    // deepmd inference
    GMX_LOG(logger_->info).appendText("deepmd: calculateForce for " 
        + std::to_string(forceProviderInput.homenr_) + "atom, in step " 
        + std::to_string(forceProviderInput.step_) + "at time: " 
        + std::to_string(forceProviderInput.t_));

    // input
    auto& atomPos = inferenceInfo.atomPosition_;
    auto& inputAtomPos = forceProviderInput.x_;
    atomPos.reserve(3 * forceProviderInput.homenr_);

    if ((int)inputAtomPos.size() == forceProviderInput.homenr_ && sizeof(RVec) == 3*sizeof(real)) {
        // direct copy
        std::copy((real*)inputAtomPos.begin().data(), (real*)inputAtomPos.end().data(), atomPos.begin());
    } else {
        for (int i=0; i < forceProviderInput.homenr_; i++) {
            atomPos[i * 3] = inputAtomPos[i][0];
            atomPos[i * 3 + 1] = inputAtomPos[i][1];
            atomPos[i * 3 + 2] = inputAtomPos[i][2];
        }
    }

    auto& box = inferenceInfo.box_;
    std::copy(&forceProviderInput.box_[0][0], &forceProviderInput.box_[DIM][DIM], box.begin());

    // inference
    // energy, force, virial | atom_pos, atom_type, box
    dp_->compute<real>(inferenceInfo.energy_, 
                    inferenceInfo.atomForce_, 
                    inferenceInfo.virial_, 
                    atomPos, inferenceInfo.atomType_ , box);


    // output
    forceProviderOutput->enerd_.term[F_DEEPMD] = inferenceInfo.energy_;

    auto& forceResult = inferenceInfo.atomForce_;
    auto outputForce = forceProviderOutput->forceWithVirial_.force();
    for (int i=0; i < forceProviderInput.homenr_; i++) {
        outputForce[i][0] = forceResult[i * 3];
        outputForce[i][1] = forceResult[i * 3 + 1];
        outputForce[i][2] = forceResult[i * 3 + 2];
    }
    
    matrix outputVirial;
    for (int dim1 = 0; dim1 < DIM; dim1++)
    for (int dim2 = 0; dim2 < DIM; dim2++) 
    outputVirial[dim1][dim2] += inferenceInfo.virial_[dim1*DIM+dim2];

    forceProviderOutput->forceWithVirial_.addVirialContribution(outputVirial);

    GMX_LOG(logger_->debug).appendText("deepmd: inference result "
        + std::to_string(forceProviderOutput->enerd_.term[F_DEEPMD]) + " energy, "
        + std::to_string(forceProviderOutput->forceWithVirial_.force().size()) + " forces, "
        + std::to_string(inferenceInfo.virial_.size()) + " virial elements");

}


} // namespace gmx
