.. title:: FEnaPack

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
dense. Possible trick is to approximate :math:`\mathbb{S}` by swaping the
order of the operators

.. math::

    \mathbb{S} \approx
    \mathbb{X}_\mathrm{BRM}
    -\Delta\left(-\nu\Delta+\mathbf{v}\cdot\nabla\right)^{-1}

or

.. math::

    \mathbb{S} \approx
    \mathbb{X}_\mathrm{SEW}
    := \left(-\nu\Delta+\mathbf{v}\cdot\nabla\right)^{-1}(-\Delta).

This gives rise to the action of 11-block of preconditioner
:math:`\mathbb{P}^{-1}` given by

.. math::

    \mathbb{X}_\mathrm{BRM}^{-1}
    := \left(-\nu\Delta+\mathbf{v}\cdot\nabla\right)(-\Delta)^{-1}.

or

.. math::

    \mathbb{X}_\mathrm{SEW}^{-1}
    := (-\Delta)^{-1}\left(-\nu\Delta+\mathbf{v}\cdot\nabla\right).

Obviously additional artificial boundary condition for Laplacian solve
:math:`-\Delta^{-1}` is needed in the action of preconditioner. Modifying
the approach from [2]_ we implement :math:`\mathbb{X}_\mathrm{BRM}^{-1}` as

.. math::

    \mathbb{X}_\mathrm{BRM}^{-1}
    := \mathbb{M}_p^{-1} (\mathbb{I} + \mathbb{K}_p\mathbb{A}_p^{-1})

where :math:`\mathbb{M}_p` is :math:`\nu^{-1}`-multiple of mass matrix on
pressure, :math:`\mathbb{K}_p \approx \nu^{-1}\mathbf{v}\cdot\nabla` is
a pressure convection matrix, and :math:`\mathbb{A}_p^{-1} \approx
(-\Delta)^{-1}` is a pressure Laplacian solve with *zero boundary condition
on inlet*. This is implemented by :py:class:`fenapack.preconditioners.PCDPC_BRM`
and :doc:`demos/navier-stokes-pcd`.

On the other hand SEW approach

.. math::

    \mathbb{X}_\mathrm{SEW}^{-1}
    := \mathbb{A}^{-1} \mathbb{F}_p \mathbb{M}_p^{-1}

with pressure convection-diffusion (PCD) matrix :math:`\mathbb{F}_p \approx
-\Delta + \mathbf{v}\cdot\nabla` which also needs at least boundary condition
for Laplacian solve is advocated in [1]_ with use of "boundary conditions"
for zero Dirichlet on *outlet* and natural conditions for both
:math:`\mathbb{A}_p` and :math:`\mathbb{F}_p`. Moreover approach, if written
in this form, requires a nasty ghost layer trick around Dirichlet boundary to
compensate for poor approximation of

.. math::

    \mathbb{I} \approx \mathbb{A}_p^{-1} \mathbb{A}_p

around Dirichlet boundary as can be found in [2]_ and [3]_. It shows crucial
to approximate well leading (Stokes) part of the operator. This is the reason
why there is :math:`\mathbb{I}` instead of :math:`\mathbb{A}_p
\mathbb{A}_p^{-1} \approx \mathbb{I}` in :math:`\mathbb{X}_\mathrm{BRM}^{-1}`.
SEW variant (with approximate finite difference-like ghost layer trick around
Dirichlet boundary) is implemented in classes
:py:class:`fenapack.preconditioners.PCDPC_SEW` and
:py:class:`fenapack.preconditioners.UnsteadyPCDPC_SEW`. Its usage is
demonstrated in some scripts in ``apps`` directory.

.. [3] Silvester D., Elman H., Ramage A.,
       *IFISS. Incompressible Flow & Iterative Solver Software.*
       http://www.maths.manchester.ac.uk/~djs/ifiss/

Documented demos
================

.. toctree::
   :titlesonly:

   demos/navier-stokes-pcd


Manual and API Reference
========================

.. toctree::
   :titlesonly:

   API Reference <api-doc/fenapack>

* :ref:`modindex`
* :ref:`genindex`
* :ref:`search`
