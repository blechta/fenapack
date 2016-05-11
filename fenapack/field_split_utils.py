# Copyright (C) 2015-2016 Jan Blechta
#
# This file is part of FENaPack.
#
# FENaPack is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# FENaPack is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License

from dolfin import compile_extension_module
import os

__all__ = ['dofmap_dofs_is', 'SubfieldBC']


dofmap_dofs_is_doc = """
Converts DofMap::dofs() to IS
"""

dofmap_dofs_is_cpp_code = """
#ifdef SWIG
%include "petsc4py/petsc4py.i"
#endif

#include <vector>
#include <petscis.h>
#include <dolfin/fem/GenericDofMap.h>
#include <dolfin/la/PETScObject.h>
#include <dolfin/log/log.h>

namespace dolfin {

  IS dofmap_dofs_is(const GenericDofMap& dofmap)
  {
    PetscErrorCode ierr;
    const std::vector<dolfin::la_index> dofs = dofmap.dofs();
    IS is;
    dolfin_assert(dofmap.index_map());
    ierr = ISCreateGeneral(dofmap.index_map()->mpi_comm(), dofs.size(),
                           dofs.data(), PETSC_COPY_VALUES, &is);
    if (ierr != 0)
      PETScObject::petsc_error(ierr, "field_split.py", "ISCreateGeneral");
    return is;
  }

}
"""

# Load compiled function dofmap_dofs_is
module = compile_extension_module(dofmap_dofs_is_cpp_code, cppargs='-g -O2')
dofmap_dofs_is = module.dofmap_dofs_is
dofmap_dofs_is.__doc__ += dofmap_dofs_is_doc
del dofmap_dofs_is_cpp_code, dofmap_dofs_is_doc, module


# Load compiled class SubfieldBC
path = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
code = open(os.path.join(path, "SubfieldBC.h")).read()
module = compile_extension_module(code, cppargs='-g -O2')
SubfieldBC = module.SubfieldBC
del path, code, module
