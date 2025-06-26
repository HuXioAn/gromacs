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
 * Implements deepmdModule class
 *
 * \author 
 * \ingroup module_applied_forces
 */
#include "gmxpre.h"

#include "deepmdModule.h"

#include <memory>
#include <string>

#include "gromacs/domdec/localatomsetmanager.h"
#include "gromacs/fileio/checkpoint.h"
#include "gromacs/mdrunutility/mdmodulesnotifiers.h"
#include "gromacs/mdtypes/commrec.h"
#include "gromacs/mdtypes/imdmodule.h"
#include "gromacs/topology/mtop_util.h"
#include "gromacs/topology/atoms.h"

#include "gromacs/utility/keyvaluetreebuilder.h"

#include "deepmdOptions.h"
#include "deepmdForceProvider.h"

namespace gmx
{

namespace
{

class deepmdMDModule final : public IMDModule
{
public:
    explicit deepmdMDModule() = default;

    void subscribeToPreProcessingNotifications(MDModulesNotifiers* notifier) override 
    {
        if (!options_.isActive())
        {
            return;
        }

        const auto setLoggerFunction = [this](const MDLogger& logger)
        { loggerPtr = &logger; };
        notifier->preProcessingNotifier_.subscribe(setLoggerFunction);

    }

    void subscribeToSimulationSetupNotifications(MDModulesNotifiers* notifier) override
    {
        if (!options_.isActive())
        {
            return;
        }

        const auto setTopologyFunction = [this](const gmx_mtop_t& top)
        {   // setup the atom types (number) 
            auto atoms = gmx_mtop_global_atoms(top);
            auto numAtom = atoms.nr;
            auto& atomTypeVec = inferenceInfo_.atomType_;

            atomTypeVec.assign(numAtom, 0);

            for (int i = 0; i < numAtom; ++i)
            {
                // TODO: no mapping to the model yet
                atomTypeVec[i] = atoms.atom[i].atomnumber;
            }
        };
        notifier->simulationSetupNotifier_.subscribe(setTopologyFunction);

        // Add deepmd output to energy file
        const auto requestEnergyOutput = [](MDModulesEnergyOutputToDEEPMDRequestChecker* energyOutputRequest)
        { energyOutputRequest->energyOutputToDEEPMD_ = true; };
        notifier->simulationSetupNotifier_.subscribe(requestEnergyOutput);

    }

    IMdpOptionProvider* mdpOptionProvider() override { return &options_; }

    IMDOutputProvider* outputProvider() override { return nullptr; }

    void initForceProviders(ForceProviders* forceProviders) override
    {
        if (options_.isActive())
        {
            deepmdForceProvider_ = std::make_unique<deepmdForceProvider>(options_.options(), inferenceInfo_, loggerPtr);
            forceProviders->addForceProvider(deepmdForceProvider_.get(), "DeepMD");
        }
    }


private:
    DeepmdOptionsProvider options_{};
    std::unique_ptr<deepmdForceProvider> deepmdForceProvider_{};

    // initialzed in notifier handlers
    deepmdInferenceInfo inferenceInfo_{};

    // pointer to the logger from notifier
    const MDLogger* loggerPtr;

    
};

} // namespace

std::unique_ptr<IMDModule> deepmdModuleInfo::create()
{
    return std::make_unique<deepmdMDModule>();
}
} // namespace gmx
