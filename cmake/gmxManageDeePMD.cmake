#
# This file is part of the GROMACS molecular simulation package.
#
# Copyright 2025- The GROMACS Authors
# and the project initiators Erik Lindahl, Berk Hess and David van der Spoel.
# Consult the AUTHORS/COPYING files and https://www.gromacs.org for details.
#
# GROMACS is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public License
# as published by the Free Software Foundation; either version 2.1
# of the License, or (at your option) any later version.
#
# GROMACS is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with GROMACS; if not, see
# https://www.gnu.org/licenses, or write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA.
#
# If you want to redistribute modifications to GROMACS, please
# consider that scientific software is very special. Version
# control is crucial - bugs must be traceable. We will be happy to
# consider code for inclusion in the official distribution, but
# derived work must not be called official GROMACS. Details are found
# in the README & COPYING files - if they are missing, get the
# official version at https://www.gromacs.org.
#
# To help us fund GROMACS development, we humbly ask that you cite
# the research papers on the package. Check out https://www.gromacs.org.

# link DeePMD library to Gromacs

#
# Called from the top-level CMakeLists.txt via include(gmxManageDeePMD)
#

# This is the user-visible switch:
gmx_option_multichoice(GMX_USE_DEEPMD
    "Enable DeePMD potential interface (requires DeePMD C++ library)"
    OFF
    ON OFF)

#
# Build an INTERFACE library named 'deepmdgmx'
# that we can hook onto 'applied_forces' later.
#
function(gmx_manage_deepmd)
    add_library(deepmdgmx INTERFACE)
    set(GMX_DEEPMD_ACTIVE OFF PARENT_SCOPE)

    if(GMX_USE_DEEPMD STREQUAL "ON")
        find_package(DeePMD REQUIRED)
        # print out where we found it:
        message(STATUS "Found DeePMD in: ${DeePMD_DIR}")
        if(TARGET DeePMD::deepmd_cc)
            get_target_property(_libloc DeePMD::deepmd_cc LOCATION)
            if(_libloc)
                message(STATUS "  DeePMD::deepmd_cc library at: ${_libloc}")
            else()
                # Fallback: maybe the package defined DeePMD_LIBRARIES
                message(STATUS "  DeePMD_LIBRARIES = ${DeePMD_LIBRARIES}")
            endif()
        else()
            message(WARNING "  DeePMD::deepmd_cc target not found; did the package config create a different target?")
        endif()

        # hook libraries and includes into our INTERFACE target
        target_link_libraries(deepmdgmx
            INTERFACE
                DeePMD::deepmd_cc
        )
        target_include_directories(deepmdgmx
            SYSTEM INTERFACE
            $<BUILD_INTERFACE:${DeePMD_DIR}/../include>  # adjust if DeePMD exports DeePMD_INCLUDE_DIRS
        )

        set(GMX_DEEPMD_ACTIVE ON PARENT_SCOPE)
    endif()
endfunction()

gmx_manage_deepmd()
