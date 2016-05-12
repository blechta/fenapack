# -*- coding: utf-8 -*-

# Copyright (C) 2014-2016 Jan Blechta and Martin Řehoř
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
# along with FENaPack.  If not, see <http://www.gnu.org/licenses/>.

"""This is FENaPack, FEniCS Navier-Stokes preconditioning package."""

__author__ = "Jan Blechta, Martin Řehoř"
__version__ = "1.7.0dev"
__license__ = "GNU LGPL v3"

from fenapack.field_split import *
from fenapack.preconditioners import *
from fenapack.nonlinear_solvers import *
from fenapack.stabilization import *
