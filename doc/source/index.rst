.. title:: FENaPack

.. include:: ../../README.rst


.. _math_background:

Mathematical background
=======================

Navier-Stokes equations

.. math::

    \left[\begin{array}{cc}
        -\nu\Delta + \mathbf{v}\cdot\nabla & \nabla \\
        -\operatorname{div}                & 0
    \end{array}\right]
    \left[\begin{array}{c}
        \mathbf{u} \\
        p
    \end{array}\right]
    =
    \left[\begin{array}{c}
        \mathbf{f} \\
        0
    \end{array}\right]

solved by GMRES preconditioned from right by

.. math::

    \mathbb{P} :=
    \left[\begin{array}{cc}
        -\nu\Delta + \mathbf{v}\cdot\nabla & \nabla \\
                                           & -\mathbb{S}
    \end{array}\right]

with Schur complement :math:`\mathbb{S} =
-\operatorname{div}\left(-\nu\Delta+\mathbf{v}\cdot\nabla\right)^{-1}\nabla`
would converge in two iterations. Unfortunately :math:`\mathbb{S}` is
dense. Possible trick is to approximate :math:`\mathbb{S}` by swapping the
order of the operators

.. math::

    \mathbb{S} \approx
    \mathbb{X}_\mathrm{BRM1}
    := (-\Delta) \left(-\nu\Delta+\mathbf{v}\cdot\nabla\right)^{-1}

or

.. math::

    \mathbb{S} \approx
    \mathbb{X}_\mathrm{BRM2}
    := \left(-\nu\Delta+\mathbf{v}\cdot\nabla\right)^{-1}(-\Delta).

This gives rise to the action of 11-block of preconditioner
:math:`\mathbb{P}^{-1}` given by

.. math::

    \mathbb{X}_\mathrm{BRM1}^{-1}
    := \left(-\nu\Delta+\mathbf{v}\cdot\nabla\right)(-\Delta)^{-1}.

or

.. math::

    \mathbb{X}_\mathrm{BRM2}^{-1}
    := (-\Delta)^{-1}\left(-\nu\Delta+\mathbf{v}\cdot\nabla\right).

Obviously additional artificial boundary condition for Laplacian solve
:math:`-\Delta^{-1}` is needed in the action of preconditioner. Modifying
the approach from [2]_ we implement :math:`\mathbb{X}_\mathrm{BRM1}^{-1}` as

.. math::

    \mathbb{X}_\mathrm{BRM1}^{-1}
    := \mathbb{M}_p^{-1} (\mathbb{I} + \mathbb{K}_p\mathbb{A}_p^{-1})

where :math:`\mathbb{M}_p` is :math:`\nu^{-1}`-multiple of mass matrix on
pressure, :math:`\mathbb{K}_p \approx \nu^{-1}\mathbf{v}\cdot\nabla` is
a pressure convection matrix, and :math:`\mathbb{A}_p^{-1} \approx
(-\Delta)^{-1}` is a pressure Laplacian solve with *zero boundary condition
on inlet*. This is implemented by :py:class:`fenapack.preconditioners.PCDPC_BRM1`
and :doc:`demos/navier-stokes-pcd`.

Analogically we prefer to express BRM2 approach as

.. math::

    \mathbb{X}_\mathrm{BRM2}^{-1}
    := (\mathbb{I} + \mathbb{A}_p^{-1}\mathbb{K}_p) \mathbb{M}_p^{-1}

now with *zero boundary condition on outlet for Laplacian solve* and
additional Robin term in convection matrix :math:`\mathbb{K}_p` roughly
as stated in [1]_, section 9.2.2. See also :doc:`demos/navier-stokes-pcd`
and :py:class:`fenapack.preconditioners.PCDPC_BRM2`.

Extension to time-dependent problems (PCDR preconditioners)
-----------------------------------------------------------

Time disretization applied in unsteady problems typically leads to the
need to incorporate a reaction term into the preconditioner. Typically, we end
up with

.. math::

    \mathbb{X}_\mathrm{BRM1}^{-1}
    := \left(\frac{1}{\tau}-\nu\Delta+\mathbf{v}\cdot\nabla\right)(-\Delta)^{-1},

or

.. math::

    \mathbb{X}_\mathrm{BRM2}^{-1}
    := (-\Delta)^{-1}\left(\frac{1}{\tau}-\nu\Delta+\mathbf{v}\cdot\nabla\right),

where :math:`\tau` denotes a fixed time step and the original PCD
preconditioner thus becomes PCDR (pressure-convection-diffusion-reaction)
preconditioner. A straightforward way of how to implement the above
actions is to update the pressure convection matrix :math:`\mathbb{K}_p` by
a contribution corresponding to the scaled pressure mass matrix, namely

.. math::

    \mathbb{X}_\mathrm{BRM1}^{-1}
    := \mathbb{M}_p^{-1} \left(\mathbb{I}
      + \left(\mathbb{K}_p + \tau^{-1} \mathbb{M}_p\right)\mathbb{A}_p^{-1}\right),

or

.. math::

    \mathbb{X}_\mathrm{BRM2}^{-1}
    := \left(\mathbb{I} + \mathbb{A}_p^{-1}\left(\mathbb{K}_p
      + \tau^{-1} \mathbb{M}_p\right)\right)\mathbb{M}_p^{-1}.

However, for unsteady problems we prefer to use the following elaborated
implementation of PCDR preconditioners, namely

.. math::

    \mathbb{X}_\mathrm{BRM1}^{-1}
    := \mathbb{R}_p^{-1} + \mathbb{M}_p^{-1} (\mathbb{I} + \mathbb{K}_p\mathbb{A}_p^{-1}),

or

.. math::

    \mathbb{X}_\mathrm{BRM2}^{-1}
    := \mathbb{R}_p^{-1} + (\mathbb{I} + \mathbb{A}_p^{-1}\mathbb{K}_p) \mathbb{M}_p^{-1},

where :math:`\mathbb{R}_p^{-1} \approx \frac{1}{\tau} (-\Delta)^{-1}`,
while :math:`\mathbb{R}_p` itself is approximated and implemented as

.. math::

    \mathbb{R}_p
    := \mathbb{B} \left(\tau^{-1} \mathbb{D}_\mathrm{M}\right)^{-1} \mathbb{B}^T,

Here, :math:`\mathbb{D}_\mathrm{M}` is the diagonal of the velocity mass
matrix, :math:`\mathbb{D}_\mathrm{M} = \operatorname{diag}(\mathbb{M}_{\mathbf{u}})`,
and :math:`\mathbb{B}^T` corresponds to the discrete pressure gradient which
is obtained as the 01-block of the original system matrix. Let us emphasize
that this submatrix is extracted from **the system matrix with velocity
Dirichlet boundary conditions being applied on it**.

The choice of :math:`\mathbb{R}_p` as above can be justified especially in the
case of :math:`\tau \rightarrow 0_+`, for which

.. math::

    \mathbb{S}^{-1}
    := \left(-\operatorname{div}\left(\frac{1}{\tau}
      - \nu\Delta+\mathbf{v}\cdot\nabla\right)^{-1}\nabla\right)^{-1}
    \approx
      \frac{1}{\tau}\left(\mathbb{B} \mathbb{M}_{\mathbf{u}}^{-1} \mathbb{B}^T\right)^{-1},


and simultaneously
:math:`\mathbb{X}^{-1} \approx \mathbb{R}_p^{-1}
= \frac{1}{\tau} \left(\mathbb{B} \mathbb{D}_\mathrm{M}^{-1} \mathbb{B}^T\right)^{-1}`.
The same approximation of the minus Laplacian operator was previously used also
in [1]_, see Remark 9.6 therein.


Documented demos
================

.. toctree::
   :titlesonly:

   demos/navier-stokes-pcd


Developer's resources
=====================

.. toctree::
   :titlesonly:

   circle


Manual and API Reference
========================

.. toctree::
   :titlesonly:

   API Reference <api-doc/fenapack>

* :ref:`modindex`
* :ref:`genindex`
* :ref:`search`
