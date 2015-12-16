# Copyright (C) 2015 Martin Rehor
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

__all__ = ['streamline_diffusion_cpp']

# Stabilization based on streamline diffusion method
# -----------------------------------------------------------------------------
#   This kind of stabilization is convenient when a multigrid method is used
#   for the convection term in the Navier-Stokes equation. The idea of the
#   stabilization involves adding an additional term of the form
#
#     delta_sd*inner(dot(grad(u), w), dot(grad(v), w))*dx
#
#   into the Navier-Stokes equation. Here u is a trial function, v is a test
#   function and w defines so-called "wind" which is a known vector.
#   Regularization parameter delta_sd is determined by the local mesh Peclet
#   number (PE), see the implementation below.
#
#   USAGE:
#     1. Define regularization parameter
#         delta_sd = Expression(streamline_diffusion_cpp)
#     2. Update variables used in the definition of delta_sd
#         delta_sd.nu ..... kinematic viscosity
#         delta_sd.mesh ... mesh
#         delta_sd.wind ... known vector
#
streamline_diffusion_cpp = """
class Stabilization : public Expression
{
public:
  double nu;
  std::shared_ptr<GenericFunction> wind;
  std::shared_ptr<Mesh> mesh;

  Stabilization() : Expression() { }

  void eval(Array<double>& values, const Array<double>& x, const ufc::cell& c) const
  {
    Cell cell(*mesh, c.index);
    double h = cell.diameter();
    double wind_norm = 0.0;
    Array<double> w(x.size());
    wind->eval(w, x, c);
    for (uint i = 0; i < x.size(); ++i)
      wind_norm += w[i]*w[i];
    wind_norm = sqrt(wind_norm);
    values[0] = 0.0;
    double PE = 0.5*wind_norm*h/nu;
    if (PE > 1.0)
      values[0] = 0.5*h*(1.0 - 1.0/PE)/wind_norm;
  }
};
"""
