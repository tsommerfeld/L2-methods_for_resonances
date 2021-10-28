# *L*<sup>2</sup>-methods for resonances

A resonance is *quasi-bound* or *temporary* state of a quantum state. It can be thought of as a discrete state embedded in and interacting with a continuum so that the discrete state can decay into the continuum and thus aquires a lifetimes.

In principle, resonances are features of a scattering continuum, however, imposing scattering boundary conditions is far more challenging than imposing bound-state (or *L*<sup>2</sup>) boundary conditions, and therefore many *L*<sup>2</sup>-methods for computing the energy and the lifetime of resonance states have been developed.

Even though these methods are based on completely different ideas, all *L*<sup>2</sup>-methods follow a similar computational protocol:
1. Parametrize the physical Hamiltonian **H** &rarr; **H**(&lambda;).
2. Diagonalize **H**(&lambda;) repeatedly.
3. Identify the resonance feature, normaly one- possibly two- trajectories *E*<sub>*n*</sub>(&lambda;<sub>*j*</sub>).
4. Analyze the identified trajectories to find the resonance energy *E*<sub>*r*</sub> and the lifetime &tau;.

## Implementation

All methods are implememted as jupyter notebooks, but a substantial part of the auxilary functions are hidden in Python libraries.

* The **notebooks** directory contains subdirectories for the four major methods (see below), and two directories collect analytical (sympy) notebooks for Gaussian integrals and for gradients of the RAC extrapolation formulas.
* The **Python_libs** directories collects helper functions imported into various notebooks. See documentation of each function.


## Methods

The goal is to compare different *L*<sup>2</sup>-methods and, in particular, their  variants on equal footing. To do so, a model potential is used:

*V*(*r*) = (*ar*<sup> 2</sup> - *b*) exp(-*cr*<sup> 2</sup>) + (l(l+1))/(2*r*<sup> 2</sup>)

where *l* = 1, and the parameters *a*, *b*, and *c* are chosen so that *V*(*r*) supports a bound state at *E* = -7.2 eV and a resonance at *E*<sub>*r*</sub> = 3.2 eV.


### *L*<sup>2</sup>-methods implemented

The four major methods implemented are:
1. Complex scaling (CS).
2. Complex absorbing potential (CAP).
3. Howard-Taylor stabilization (HTS).
4. Regularized analytic continuation (RAC).


### Basis sets

Each method needs to be combined with a represation of the Hamiltonian, that is, with a 'basis set'. Two basis set types are compared: A quasi-exact discrete variable representation (DVR)- a dense grid basis- and three small Gaussian basis sets modeling basis sets typically used in electronic structure theory (20 GTOs or less).

### Variants

The major method- CS, CAP, HTS, or RAC- and the basis essentially define steps 1 to 3. But each major methods has several analyis variants for step 4. In addition, RAC has also step 1 variants. It is these variants that are often brushed under the rug, but that turn out to yield vastly different resonance energies even though the input data for all variants of one major method are identical.

Two notebooks in each of the four main subdirectories deal with step 2: One DVR and one Gaussian basis notebook. The other notebooks deal with different analysis variants used in step 4.

### Licence

MIT licence.

