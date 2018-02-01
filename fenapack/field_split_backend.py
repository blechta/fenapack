# Copyright (C) 2017 Jan Blechta
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

"""Tools for extraction and management of fieldsplit
submatrices, subvectors, subbcs, subksps intended to
be hidden from user interface"""

from dolfin import PETScMatrix, Timer, DirichletBC
from petsc4py import PETSc

from fenapack._field_split_utils import SubfieldBC
from fenapack.assembling import PCDAssembler


class PCDInterface(object):
    """Wrapper of PCDAssembler for interfacing with PCD PC
    fieldsplit implementation. Convection fieldsplit submatrices
    are extracted as shallow or deep submatrices according to
    ``deep_submats`` parameter."""

    def __init__(self, pcd_assembler, is_u, is_p, deep_submats=False):
        """Create PCDInterface instance given PCDAssembler instance,
        and velocity and pressure index sets"""

        # Check input
        assert isinstance(pcd_assembler, PCDAssembler)
        assert isinstance(is_u, PETSc.IS)
        assert isinstance(is_p, PETSc.IS)

        # Store what needed
        self.assembler = pcd_assembler
        self.is_u = is_u
        self.is_p = is_p

        # Choose submatrix implementation
        assert isinstance(deep_submats, bool)
        if deep_submats:
            self.assemble_operator = self._assemble_operator_deep
        else:
            self.assemble_operator = self._assemble_operator_shallow

        # Dictionary for storing work mats
        self.scratch = {}


    def setup_ksp_Ap(self, ksp):
        """Setup pressure Laplacian ksp and assemble matrix"""
        self.setup_ksp_once(ksp, self.assembler.ap, self.is_p, spd=True)


    def setup_ksp_Mp(self, ksp):
        """Setup pressure mass matrix ksp and assemble matrix"""
        self.setup_ksp_once(ksp, self.assembler.mp, self.is_p, spd=True)


    def setup_ksp_Mu(self, ksp):
        """Setup velocity mass matrix ksp and assemble matrix"""
        self.setup_ksp_once(ksp, self.assembler.mu, self.is_u, spd=True)


    def setup_mat_Kp(self, mat=None):
        """Setup and assemble pressure convection
        matrix and return it"""
        return self.assemble_operator(self.assembler.kp, self.is_p, submat=mat)


    def setup_mat_Fp(self, mat=None):
        """Setup and assemble pressure convection-diffusion
        matrix and return it"""
        return self.assemble_operator(self.assembler.fp, self.is_p, submat=mat)


    def apply_pcd_bcs(self, vec):
        """Apply bcs to intermediate pressure vector of PCD pc"""
        self.apply_bcs(vec, self.assembler.pcd_bcs, self.is_p)


    def get_work_dolfin_mat(self, key, comm,
                            can_be_destroyed=None, can_be_shared=None):
        """Get working DOLFIN matrix by key. ``can_be_destroyed=True`` tells
        that it is probably favourable to not store the matrix unless it is
        shared as it will not be used ever again, ``None`` means that it can
        be destroyed but it is not probably favourable and ``False`` forbids
        the destruction. ``can_be_shared`` tells if a work matrix can be the
        same with work matrices for other keys."""
        # TODO: Add mechanism for sharing DOLFIN mats
        # NOTE: Maybe we don't really need sharing. If only persistent matrix
        #       is convection then there is nothing to be shared.

        # Check if requested matrix is in scratch
        dolfin_mat = self.scratch.get(key, None)

        # Allocate new matrix otherwise
        if dolfin_mat is None:
            dolfin_mat = PETScMatrix(comm)

        # Store or pop the matrix as requested
        if can_be_destroyed in [False, None]:
            self.scratch[key] = dolfin_mat
        else:
            assert can_be_destroyed is True
            self.scratch.pop(key, None)

        return dolfin_mat


    def setup_ksp_once(self, ksp, assemble_func, iset, spd=False):
        """Assemble into operator of given ksp if not yet assembled"""
        mat = ksp.getOperators()[0]
        # FIXME: This logic that it is created once should be visible
        #        in higher level, not in these internals
        if mat.type is None or not mat.isAssembled():
            # Assemble matrix
            dolfin_mat = self.get_work_dolfin_mat(assemble_func, mat.comm,
                                                  can_be_destroyed=True,
                                                  can_be_shared=True)
            assemble_func(dolfin_mat)
            mat = self._get_deep_submat(dolfin_mat, iset, submat=None)

            # Use eventual spd flag
            mat.setOption(PETSc.Mat.Option.SPD, spd)

            # Set correct options prefix
            prefix = ksp.getOptionsPrefix()
            mat.setOptionsPrefix(prefix)

            # Use also as preconditioner matrix
            ksp.setOperators(mat, mat)
            assert ksp.getOperators()[0].isAssembled()

            # Setup ksp
            with Timer("FENaPack: {} setup".format(prefix)):
                ksp.setUp()


    def _assemble_operator_shallow(self, assemble_func, iset, submat=None):
        """Assemble operator of given name using shallow submat"""
        # Assemble into persistent DOLFIN matrix everytime
        # TODO: Does not shallow submat take care of parents lifetime? How?
        dolfin_mat = self.get_work_dolfin_mat(assemble_func, iset.comm,
                                              can_be_destroyed=False,
                                              can_be_shared=False)
        assemble_func(dolfin_mat)

        # FIXME: This logic that it is created once should be visible
        #        in higher level, not in these internals
        # Create shallow submatrix (view into dolfin mat) once
        if submat is None or submat.type is None or not submat.isAssembled():
            submat = self._get_shallow_submat(dolfin_mat, iset, submat=submat)
            assert submat.isAssembled()

        return submat


    def _assemble_operator_deep(self, assemble_func, iset, submat=None):
        """Assemble operator of given name using deep submat"""
        dolfin_mat = self.get_work_dolfin_mat(assemble_func, iset.comm,
                                              can_be_destroyed=None,
                                              can_be_shared=True)
        assemble_func(dolfin_mat)
        return self._get_deep_submat(dolfin_mat, iset, submat=submat)


    def apply_bcs(self, vec, bcs_getter, iset):
        """Transform dolfin bcs obtained using ``bcs_getter`` function
        into fieldsplit subBCs and apply them to fieldsplit vector.
        SubBCs are cached."""
        # Fetch subbcs from cache or construct it
        subbcs = getattr(self, "_subbcs", None)
        if subbcs is None:
            bcs = bcs_getter()
            bcs = [bcs] if isinstance(bcs, DirichletBC) else bcs
            subbcs = [SubfieldBC(bc, iset) for bc in bcs]
            self._subbcs = subbcs

        # Apply bcs
        for bc in subbcs:
            bc.apply(vec)


    if PETSc.Sys.getVersion()[0:2] <= (3, 7) and PETSc.Sys.getVersionInfo()['release']:

        @staticmethod
        def _get_deep_submat(dolfin_mat, iset, submat=None):
            return dolfin_mat.mat().getSubMatrix(iset, iset, submat=submat)

        @staticmethod
        def _get_shallow_submat(dolfin_mat, iset, submat=None):
            if submat is None:
                submat = PETSc.Mat().create(iset.comm)
            return submat.createSubMatrix(dolfin_mat.mat(), iset, iset)


    else:

        @staticmethod
        def _get_deep_submat(dolfin_mat, iset, submat=None):
            return dolfin_mat.mat().createSubMatrix(iset, iset, submat=submat)

        @staticmethod
        def _get_shallow_submat(dolfin_mat, iset, submat=None):
            if submat is None:
                submat = PETSc.Mat().create(iset.comm)
            return submat.createSubMatrixVirtual(dolfin_mat.mat(), iset, iset)
