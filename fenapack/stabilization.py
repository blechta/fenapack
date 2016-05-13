# Copyright (C) 2015-2016 Martin Rehor
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

from dolfin import Expression, FiniteElement

__all__ = ['StabilizationParameterSD']


_streamline_diffusion_cpp = """
class StabilizationParameterSD : public Expression
{
public:
  std::shared_ptr<GenericFunction> viscosity;
  std::shared_ptr<GenericFunction> wind;

  StabilizationParameterSD() : Expression() { }

  void eval(Array<double>& values, const Array<double>& x,
            const ufc::cell& c) const
  {
    // Get dolfin cell and its diameter
    // FIXME: Avoid dynamical allocation
    const std::shared_ptr<const Mesh> mesh = wind->function_space()->mesh();
    const Cell cell(*mesh, c.index);
    double h = cell.h();

    // Evaluate viscosity at given coordinates
    // FIXME: Avoid dynamical allocation
    Array<double> nu(viscosity->value_size());
    viscosity->eval(nu, x, c);

    // Compute l2 norm of wind
    double wind_norm = 0.0;
    // FIXME: Avoid dynamical allocation
    Array<double> w(wind->value_size());
    wind->eval(w, x, c);
    for (uint i = 0; i < w.size(); ++i)
      wind_norm += w[i]*w[i];
    wind_norm = sqrt(wind_norm);

    // Compute Peclet number and evaluate stabilization parameter
    double PE = 0.5*wind_norm*h/nu[0];
    values[0] = (PE > 1.0) ? 0.5*h*(1.0 - 1.0/PE)/wind_norm : 0.0;
  }
};
"""


def StabilizationParameterSD(wind, viscosity):
    """Returns a subclass of dolfin.Expression representing streamline
    diffusion stabilization parameter.

    This kind of stabilization is convenient when a multigrid method is used
    for the convection term in the Navier-Stokes equation. The idea of the
    stabilization involves adding an additional term of the form

      delta_sd*inner(dot(grad(u), w), dot(grad(v), w))*dx

    into the Navier-Stokes equation. Here u is a trial function, v is a test
    function and w defines so-called "wind" which is a known vector function.
    Regularization parameter delta_sd is determined by the local mesh Peclet
    number (PE), see the implementation below.

    *Arguments*
        wind (:py:class:`GenericFunction`)
            A vector field determining convective velocity.
        viscosity (:py:class:`GenericFunction`)
            A scalar field determining kinematic viscosity.
    """
    mesh = wind.function_space().mesh()
    element = FiniteElement("DG", mesh.ufl_cell(), 0)
    delta_sd = Expression(_streamline_diffusion_cpp,
                          element=element, domain=mesh)
    delta_sd.wind = wind
    delta_sd.viscosity = viscosity
    return delta_sd
