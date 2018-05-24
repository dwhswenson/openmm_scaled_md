import openmmtools
from simtk import unit

"""
Integrators for scaled MD simulations.

These are based on (a subset of) the integrators implemented in
``openmmtools``. More documentation (and most of the implementation) is
there. Much of the documentation of classes here is taken from the
openmmtools documentation.
"""


class LangevinIntegrator(openmmtools.integrators.LangevinIntegrator):
    """
    Langevin integrators for scaled MD. Based on openmmtools integrators.

    In addition to the normal Langevin behavior (as including in
    openmmtools, this includes a ``force_scaling`` parameter for the scaled
    MD>

    Remaining documentation based heavily on
    :class:`openmmtools.integrators.LangevinIntegrator`.

    One way to divide the Langevin system is into three parts which can each
    be solved "exactly:"
        - R: Linear "drift" / Constrained "drift"
            Deterministic update of *positions*, using current velocities
            x <- x + v dt
        - V: Linear "kick" / Constrained "kick"
            Deterministic update of *velocities*, using current forces
            v <- v + (f/m) dt
                where f = force, m = mass
        - O: Ornstein-Uhlenbeck
            Stochastic update of velocities, simulating interaction with a
            heat bath
            v <- av + b sqrt(kT/m) R
                where
                a = e^(-gamma dt)
                b = sqrt(1 - e^(-2gamma dt))
                R is i.i.d. standard normal
    We can then construct integrators by solving each part for a certain
    timestep in sequence.  (We can further split up the V step by force
    group, evaluating cheap but fast-fluctuating forces more frequently than
    expensive but slow-fluctuating forces. Since forces are only evaluated
    in the V step, we represent this by including in our "alphabet" V0, V1,
    ...) When the system contains holonomic constraints, these steps are
    confined to the constraint manifold.

    Examples
    --------
        - VVVR
            splitting="O V R V O"
        - BAOAB:
            splitting="V R O R V"
        - g-BAOAB, with K_r=3:
            splitting="V R R R O R R R V"
        - g-BAOAB with solvent-solute splitting, K_r=K_p=2:
            splitting="V0 V1 R R O R R V1 R R O R R V1 V0"

    Attributes
    ----------
    _kinetic_energy : str
        This is 0.5*m*v*v by default, and is the expression used for the
        kinetic energy
    shadow_work : unit.Quantity with units of energy
       Shadow work (if integrator was constructed with
       measure_shadow_work=True)
    heat : unit.Quantity with units of energy
       Heat (if integrator was constructed with measure_heat=True)

    Parameters
    ----------
    temperature : np.unit.Quantity compatible with kelvin, default: 298.0*unit.kelvin
        Fictitious "bath" temperature
    collision_rate : np.unit.Quantity compatible with 1/picoseconds, default: 1.0/unit.picoseconds
        Collision rate
    timestep : np.unit.Quantity compatible with femtoseconds, default: 1.0*unit.femtoseconds
        Integration timestep
    force_scaling : float
        scaled MD scaling constant :math:`\lambda` for the forces
    splitting : string, default: "V R O R V"
        Sequence of "R", "V", "O" (and optionally "{", "}", "V0", "V1", ...)
        substeps to be executed each timestep.  Forces are only used in
        V-step. Handle multiple force groups by appending the force group
        index to V-steps, e.g. "V0" will only use forces from force group 0.
        "V" will perform a step using all forces.  "{" will cause
        metropolization, and must be followed later by a "}".
    constraint_tolerance : float, default: 1.0e-8
        Tolerance for constraint solver
    measure_shadow_work : boolean, default: False
        Accumulate the shadow work performed by the symplectic substeps, in
        the global `shadow_work`
    measure_heat : boolean, default: False
        Accumulate the heat exchanged with the bath in each step, in the
        global `heat`


    References
    ----------
    [Leimkuhler and Matthews, 2015] Molecular dynamics: with deterministic
    and stochastic numerical methods, Chapter 7
    """
    def __init__(self,
                 temperature=298.0 * unit.kelvin,
                 collision_rate=1.0 / unit.picosecond,
                 timestep=1.0 * unit.femtosecond,
                 force_scaling=1.0,
                 splitting="V R O R V",
                 constraint_tolerance=1e-8,
                 measure_shadow_work=False,
                 measure_heat=False):
        # see openmm
        self.force_scaling = force_scaling
        super(LangevinIntegrator, self).__init__(
            temperature, collision_rate, timestep, splitting,
            constraint_tolerance, measure_shadow_work, measure_heat
        )

    def _add_V_step(self, force_group="0"):
        # note that this is very much based on openmmtools implementation
        """
        Deterministic velocity update, using only forces from ``force_group`.

        Parameters
        ----------
        force_group : str, optional, default="0"
           Force group to use for this step
        """
        if self._measure_shadow_work:
            self.addComputeSum("old_ke", self._kinetic_energy)

        # update velocities
        if self._mts:
            self.addComputePerDof(
                "v", "v + ((dt / {}) * f{} * {} / m)".format(
                    self._force_group_nV[force_group],
                    force_group,
                    str(self.force_scaling)
            ))
        else:
            self.addComputePerDof(
                "v", "v + (dt / {}) * f * {} / m".format(
                    self._force_group_nV["0"],
                    str(self.force_scaling)
            ))

        self.addConstrainVelocities()

        if self._measure_shadow_work:
            self.addComputeSum("new_ke", self._kinetic_energy)
            self.addComputeGlobal("shadow_work",
                                  "shadow_work + (new_ke - old_ke)")

class VVVRIntegrator(LangevinIntegrator):
    """
    Create a velocity verlet with velocity randomization (VVVR) integrator.

    This integrator is equivalent to a Langevin integrator in the velocity
    Verlet discretization with a timestep correction to ensure that the
    field-free diffusion constant is timestep invariant.  The global 'heat'
    keeps track of the heat accumulated during integration, and can be
    used to correct the sampled statistics or in a Metropolization scheme.

    References
    ----------
    David A. Sivak, John D. Chodera, and Gavin E. Crooks.
    Time step rescaling recovers continuous-time dynamical properties for
    discrete-time Langevin integration of nonequilibrium systems
    http://arxiv.org/abs/1301.3800

    Examples
    --------
    Create a VVVR integrator for scaled MD.
    >>> temperature = 298.0 * unit.kelvin
    >>> collision_rate = 1.0 / unit.picoseconds
    >>> timestep = 1.0 * unit.femtoseconds
    >>> force_scaling = 0.7
    >>> integrator = VVVRIntegrator(temperature, collision_rate, timestep,
                                    force_scaling)
    """
    def __init__(self, *args, **kwargs):
        kwargs['splitting'] = "O V R V O"
        super(VVVRIntegrator, self).__init__(*args, **kwargs)


class BAOABIntegrator(LangevinIntegrator):
    """Create a BAOAB integrator."""
    def __init__(self, *args, **kwargs):
        """Create an integrator of Langevin dynamics using the BAOAB operator splitting.
        Parameters
        ----------
        temperature : np.unit.Quantity compatible with kelvin, default: 298.0*unit.kelvin
           Fictitious "bath" temperature
        collision_rate : np.unit.Quantity compatible with 1/picoseconds, default: 91.0/unit.picoseconds
           Collision rate
        timestep : np.unit.Quantity compatible with femtoseconds, default: 1.0*unit.femtoseconds
           Integration timestep
        constraint_tolerance : float, default: 1.0e-8
            Tolerance for constraint solver
        measure_shadow_work : boolean, default: False
            Accumulate the shadow work performed by the symplectic substeps, in the global `shadow_work`
        measure_heat : boolean, default: False
            Accumulate the heat exchanged with the bath in each step, in the global `heat`
        References
        ----------
        [Leimkuhler and Matthews, 2013] Rational construction of stochastic numerical methods for molecular sampling
        https://academic.oup.com/amrx/article-abstract/2013/1/34/166771/Rational-Construction-of-Stochastic-Numerical
        Examples
        --------
        Create a BAOAB integrator.
        >>> temperature = 298.0 * unit.kelvin
        >>> collision_rate = 1.0 / unit.picoseconds
        >>> timestep = 1.0 * unit.femtoseconds
        >>> integrator = BAOABIntegrator(temperature, collision_rate, timestep)
        """
        kwargs['splitting'] = "V R O R V"
        super(BAOABIntegrator, self).__init__(*args, **kwargs)

