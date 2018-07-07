// Copyright (C) 2016 Jan Blechta, Martin Rehor
//
// This file is part of FENaPack.
//
// dolfin-tape is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// FENaPack is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with FENaPack. If not, see <http://www.gnu.org/licenses/>.

#ifndef __SUBFIELDBC_H_
#define __SUBFIELDBC_H_

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

#include <vector>
#include <algorithm>
#include <petscis.h>
#include <petscvec.h>
#include <petscmat.h>
#include <dolfin/fem/DirichletBC.h>
#include <dolfin/fem/GenericDofMap.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/la/PETScObject.h>
#include <dolfin/common/MPI.h>
#include <dolfin/common/Timer.h>
#include <dolfin/log/log.h>
#include "petsc_casters.h"

using namespace dolfin;


class SubfieldBC : public PETScObject
{
public:

  /// Constructor. Accepts DirichletBC on subspace and IS which
  /// should be acquired using
  ///   IS is = dofmap_dofs_is(subspace->dofmap());
  SubfieldBC(const DirichletBC& bc, const IS subfield_is)
  {
    dolfin_assert(bc.function_space());
    compute_subfield_bc(_subfield_bc_indices, _subfield_bc_values, bc,
                        *bc.function_space()->dofmap(), subfield_is);
  }

  /// Return whether BC is homogeneous
  bool is_homogeneous()
  {
    return std::all_of(_subfield_bc_values.cbegin(),
                       _subfield_bc_values.cend(),
                       [](double v){ return v == 0.0; });
  }

  /// Return computed bc indices and values (for debugging purposes)
  void get_boundary_values(DirichletBC::Map& boundary_values) const
  {
    dolfin_assert(_subfield_bc_indices.size() == _subfield_bc_values.size());
    boundary_values.clear();
    boundary_values.reserve(_subfield_bc_indices.size());
    for (std::size_t i = 0; i <  _subfield_bc_indices.size(); ++i)
      boundary_values[_subfield_bc_indices[i]] = _subfield_bc_values[i];
  }

  /// Apply bc to PETSc Vec comming from fieldsplit PC
  void apply(Vec x)
  {
    apply_subfield_bc(x, _subfield_bc_indices, _subfield_bc_values);
  }

  /// Apply homogeneous bc to PETSc Mat (comming from fieldsplit PC)
  /// in finite-difference (ghost node) fashion
  void apply_fdm(Mat A)
  {
    apply_subfield_bc_fdm(A, _subfield_bc_indices);
  }

private:

  std::vector<la_index> _subfield_bc_indices;

  std::vector<double> _subfield_bc_values;

  static void compute_subfield_bc(
    std::vector<la_index>& subfield_bc_indices,
    std::vector<double>& subfield_bc_values,
    const DirichletBC& bc,
    const GenericDofMap& dofmap,
    const IS subfield_is)
  {
    Timer timer("FENaPack: Subfield BC compute bc");

    PetscErrorCode ierr;

    dolfin_assert(dofmap.index_map());
    const IndexMap& index_map = *dofmap.index_map();

    // Compute boundary values using supplied bc
    DirichletBC::Map bv_local;
    bc.get_boundary_values(bv_local);
    // FIXME: Is gather needed?!
    if (MPI::size(index_map.mpi_comm()) > 1 && bc.method() != "pointwise")
      bc.gather(bv_local);

    // Pass to global indices in boundary values
    DirichletBC::Map bv_global;
    bv_global.reserve(bv_local.size());
    for (const auto& v : bv_local)
      // FIXME: Consider inlining IndexMap::local_to_global()?!
      bv_global[index_map.local_to_global(v.first)] = v.second;
    bv_local.clear();

    // Get subfield size
    la_index subfield_size;
    ierr = ISGetLocalSize(subfield_is, &subfield_size);
    if (ierr != 0) petsc_error(ierr, "SubfieldBC.h", "ISGetSize");

    // Get subfield indices
    const la_index* subfield_indices;
    ierr = ISGetIndices(subfield_is, &subfield_indices);
    if (ierr != 0) petsc_error(ierr, "SubfieldBC.h", "ISGetIndices");

    // Prepare data for storing results
    subfield_bc_indices.clear();
    subfield_bc_values.clear();
    subfield_bc_indices.reserve(bv_global.size());
    subfield_bc_values.reserve(bv_global.size());

    // Compute subfield offset to global indexing
    const std::size_t subfield_offset
      = MPI::global_offset(index_map.mpi_comm(),
                           std::size_t(subfield_size), true);

    // Pass to fieldsplit indexing
    DirichletBC::Map::const_iterator it;
    const auto end = bv_global.cend();
    for (std::size_t i=0; i < subfield_size; ++i)
    {
      // NOTE: subfield_indices contains only owned part!
      // FIXME: Isn't lookup between unowned dofs also needed?
      it = bv_global.find(subfield_indices[i]);
      if (it != end)
      {
        subfield_bc_indices.push_back(i+subfield_offset);
        subfield_bc_values.push_back(it->second);
      }
    }

    // Destroy subfield indices
    ierr = ISRestoreIndices(subfield_is, &subfield_indices);
    if (ierr != 0) petsc_error(ierr, "SubfieldBC.h", "ISRestoreIndices");
  }

  static void apply_subfield_bc(Vec x,
                                const std::vector<la_index>& indices,
                                const std::vector<double>& values)
  {
    Timer timer("FENaPack: Subfield BC apply bc vec");

    PetscErrorCode ierr;
    dolfin_assert(indices.size() == values.size());

    // Set bc values of Vec
    ierr = VecSetValues(x, indices.size(), indices.data(), values.data(),
                        INSERT_VALUES);
    if (ierr != 0) petsc_error(ierr, "SubfieldBC.h", "VecSetValues");

    // Assemble vector after completing all calls to VecSetValues()
    // FIXME: Maybe wait with assembly?!
    ierr = VecAssemblyBegin(x);
    if (ierr != 0) petsc_error(ierr, "SubfieldBC.h", "VecAssemblyBegin");
    ierr = VecAssemblyEnd(x);
    if (ierr != 0) petsc_error(ierr, "SubfieldBC.h", "VecAssemblyEnd");
  }

  static void apply_subfield_bc_fdm(Mat A,
                                    const std::vector<la_index>& indices)
  {
    Timer timer("FENaPack: Subfield BC apply bc mat fdm");

    PetscErrorCode ierr;
    Vec diag, scale;

    // A shall be square, let's abuse notation
    ierr = MatCreateVecs(A, &scale, &diag);
    if (ierr != 0) petsc_error(ierr, "SubfieldBC.h", "MatCreateVecs");

    // Create vector with twos on boundary and ones in bulk
    ierr = VecSet(scale, 1.0);
    if (ierr != 0) petsc_error(ierr, "SubfieldBC.h", "VecSet");
    std::vector<double> values(indices.size(), 2.0);
    ierr = VecSetValues(scale, indices.size(), indices.data(),
                        values.data(), INSERT_VALUES);
    if (ierr != 0) petsc_error(ierr, "SubfieldBC.h", "VecSetValues");
    ierr = VecAssemblyBegin(scale);
    if (ierr != 0) petsc_error(ierr, "SubfieldBC.h", "VecAssemblyBegin");
    ierr = VecAssemblyEnd(scale);
    if (ierr != 0) petsc_error(ierr, "SubfieldBC.h", "VecAssemblyEnd");

    // Scale the diagonal
    ierr = MatGetDiagonal(A, diag);
    if (ierr != 0) petsc_error(ierr, "SubfieldBC.h", "MatGetDiagonal");
    ierr = VecPointwiseMult(diag, scale, diag);
    if (ierr != 0) petsc_error(ierr, "SubfieldBC.h", "VecPointwiseMult");
    ierr = MatDiagonalSet(A, diag, INSERT_VALUES);
    if (ierr != 0) petsc_error(ierr, "SubfieldBC.h", "MatDiagonalSet");

    // Clean-up
    ierr = VecDestroy(&diag);
    if (ierr != 0) petsc_error(ierr, "SubfieldBC.h", "VecDestroy");
    ierr = VecDestroy(&scale);
    if (ierr != 0) petsc_error(ierr, "SubfieldBC.h", "VecDestroy");
  }

};


PYBIND11_MODULE(SIGNATURE, m)
{
  pybind11::class_<SubfieldBC, std::shared_ptr<SubfieldBC>>(m, "SubfieldBC")
    .def(pybind11::init<const DirichletBC&, const IS>())
    .def("is_homogeneous", &SubfieldBC::is_homogeneous)
    .def("get_boundary_values", &SubfieldBC::get_boundary_values)
    .def("apply", &SubfieldBC::apply)
    .def("apply_fdm", &SubfieldBC::apply_fdm);
}

namespace pybind11
{
  namespace detail
  {
    PETSC_CASTER_MACRO(IS, is);
    PETSC_CASTER_MACRO(Vec, vec);
    PETSC_CASTER_MACRO(Mat, mat);
  }
}

#endif
