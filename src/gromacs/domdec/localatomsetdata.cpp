/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright 2018- The GROMACS Authors
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
/*! \file
 * \internal \brief
 * Declares gmx::internal::LocalAtomSetDataData.
 *
 * \author Christian Blau <cblau@gwdg.de>
 * \ingroup module_domdec
 */
#include "gmxpre.h"

#include "localatomsetdata.h"

#include <algorithm>
#include <numeric>
#include <iostream>

#include "gromacs/domdec/ga2la.h"

namespace gmx
{

namespace internal
{

/********************************************************************
 * LocalAtomSetData
 */

LocalAtomSetData::LocalAtomSetData(ArrayRef<const Index> globalIndex) :
    globalIndex_(globalIndex.begin(), globalIndex.end()),
    localIndex_(globalIndex.begin(), globalIndex.end())
{
    collectiveIndex_.resize(localIndex_.size());
    std::iota(collectiveIndex_.begin(), collectiveIndex_.end(), 0);
}

void LocalAtomSetData::setLocalAndCollectiveIndices(const gmx_ga2la_t& ga2la)
{
    /* Loop over all the atom indices of the set to check which ones are local.
     * cf. dd_make_local_group_indices in groupcoord.cpp
     */
    const int numAtomsGlobal = globalIndex_.size();

    /* Clear vector without changing capacity,
     * because we expect the size of the vectors to vary little. */
    localIndex_.resize(0);
    collectiveIndex_.resize(0);

    for (int iCollective = 0; iCollective < numAtomsGlobal; iCollective++)
    {
        //std::cout<< "LocalAtomSetData::setLocalAndCollectiveIndices globalIndex_["<<iCollective<<"]="<< globalIndex_[iCollective] << " globalIndex_.size()="<<globalIndex_.size() <<std::endl;
        if (const int* iLocal = ga2la.findHome(globalIndex_[iCollective]))
        {
            /* Save the atoms index in the local atom numbers array */
            /* The atom with this index is a home atom. */
            localIndex_.push_back(*iLocal);

            /* Keep track of where this local atom belongs in the collective index array.
             * This is needed when reducing the local arrays to a collective/global array
             * in communicate_group_positions */
            collectiveIndex_.push_back(iCollective);
        }
    }
    //std::cout<< "In LocalAtomSetData::setLocalAndCollectiveIndices globalIndex_.size()="<<globalIndex_.size() << " localIndex_.size()=" <<localIndex_.size() <<std::endl;
}

GhostAtomSetData::GhostAtomSetData(ArrayRef<const Index> globalIndex) : LocalAtomSetData(globalIndex)
{
    /*
    * need to clear localIndex_ and collectiveIndex_ to have correct ghost list
    * in case of single MPI rank setLocalAndCollectiveIndices() is never triggered
    */
    localIndex_.clear();
    collectiveIndex_.clear();
}

void GhostAtomSetData::setLocalAndCollectiveIndices(const gmx_ga2la_t& ga2la)
{
    /* Loop over all the atom indices of the set to check which ones are local ghost.
     */
    const int numAtomsGlobal = globalIndex_.size();
    //std::cout<< "In GhostAtomSetData::setLocalAndCollectiveIndices globalIndex_.size()="<<globalIndex_.size() <<std::endl;

    localIndex_.resize(0);
    collectiveIndex_.resize(0);

    for (int iCollective = 0; iCollective < numAtomsGlobal; iCollective++)
    {   
        //std::cout<< "GhostAtomSetData::setLocalAndCollectiveIndices globalIndex_["<<iCollective<<"]="<< globalIndex_[iCollective] <<std::endl;
        if (const auto iLocalEntry = ga2la.find(globalIndex_[iCollective]))
        {
            //std::cout<< "GhostAtomSetData::setLocalAndCollectiveIndices globalIndex_["<<iCollective<<"]="<< globalIndex_[iCollective] << " iLocalEntry->cell="<<iLocalEntry->cell<<std::endl; 
            if (iLocalEntry->cell != 0){ // ghost
                localIndex_.push_back(iLocalEntry->la);
                collectiveIndex_.push_back(iCollective);
            }
        }
    }
    //std::cout<< "In GhostAtomSetData::setLocalAndCollectiveIndices globalIndex_.size()="<<globalIndex_.size()<< " localIndex_.size()=" <<localIndex_.size() <<std::endl;
}

} // namespace internal

} // namespace gmx
