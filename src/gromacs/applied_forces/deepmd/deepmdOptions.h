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
 * Declares options for deepmd. This class handles parameters set during
 * pre-processing time.
 *
 * \author 
 * \ingroup module_applied_forces
 */
#ifndef GMX_APPLIED_FORCES_DEEPMDOPTIONPROVIDER_H
#define GMX_APPLIED_FORCES_DEEPMDOPTIONPROVIDER_H

#include <optional>
#include <string>

#include "gromacs/utility/real.h"
#include "gromacs/mdtypes/imdpoptionprovider.h"

struct gmx_mtop_t;
struct t_commrec;
enum class PbcType;

namespace gmx
{
enum class StartingBehavior;
struct EnsembleTemperature;

struct deepmdOptions
{
    bool active = false;
    std::string modelFile = "model.pth";

    bool provideVerletList = false;

};

class DeepmdOptionsProvider : public IMdpOptionProvider
{
public:
    //! Implementation of IMdpOptionProvider methods
    void initMdpTransform(IKeyValueTreeTransformRules* rules) override;
    void initMdpOptions(IOptionsContainerWithSections* options) override;
    void buildMdpOutput(KeyValueTreeObjectBuilder* builder) const override;

    bool isActive() const;
    std::string getModelPath() const;
    bool getProvideVerletList() const;

    const deepmdOptions& options() const
    {
        return options_;
    }

private:
    deepmdOptions options_;
};
} // namespace gmx
#endif // GMX_APPLIED_FORCES_DEEPMDOPTIONPROVIDER_H
