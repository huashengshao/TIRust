= Reference conventions
These conventions were read from the supplied original sources. Their presence
in this report is not a claim that every adapter has passed its numerical tests.

== Kinematics and scale
Use Minkowski signature $(+, -, -, -)$ with component order $(p^0, p^1, p^2,
p^3)$. Propagators are $D_j=(q+p_j)^2-m_j^2+i 0$, raised to positive integer
powers. Generalized integrals use $d=4+"idim"-2epsilon$ and the native raw
measure; dimension shifts introduce powers of $pi$. Native tensor reduction
accepts one shift per propagator and removes
the first common shift internally, then translates tensor coefficients back.
The MadLoop interface instead receives the nonzero shifts and inserts a zero
first shift. A reference comparison must preserve the chosen numerator routing
when it changes the propagator direction.

The original #raw("MU") / #raw("MUR") argument is an energy scale, not its
square. Its scalar wrapper passes $mu^2$ to QCDLoop and $mu$ to OneLOop.
Complex mass-squared inputs must retain their imaginary part; the reference
switches to its OneLOop backend for complex masses.
OneLoopMaster accepts real external squared invariants, complex squared masses
with $Im(m^2) <= 0$, and a positive real squared scale. Its continued logarithms
and polylogarithms own the Feynman branch prescription; no independent branch
implementation belongs in the reduction adapter. Preserve inputs by default;
legacy on-shell snapping must be requested explicitly. Dual derivatives follow
the selected analytic branch away from nondifferentiable boundaries.

== Native poles and MadLoop conversion
The native result order is
$ (U, I, J, F) = (epsilon_("UV")^(-1), epsilon_("IR")^(-1), epsilon_("IR")^(-2), "finite"). $
These symbols label separate coefficients. In particular, a scaleless integral can have
canceling UV and IR contributions even when its combined Laurent series is
zero. Three combined coefficients do not determine their separate origin.
OneLoopMaster uses $("finite", epsilon^(-1), epsilon^(-2))$ ordering.

The original #raw("IMLOOP") adapter combines the native poles and maps a
tensor of homogeneous rank $r$ to MadLoop order (finite, simple pole, double
pole). Defining $a = gamma_E + ln pi$, $b = a^2 / 2 + pi^2 / 12$ and
$c_r = (-1)^r / (i pi^2)$, its explicit mapping is
$ (F_"ML", S_"ML", D_"ML") = c_r (F + a (U + I) + b J, U + I + a J, J). $
For a OneLOop-normalized scalar master $(L_0,L_1,L_2)$, the scalar wrappers use
$ J = i pi^2 L_2, quad U+I = i pi^2 (L_1-a L_2), $
$ F = i pi^2 (L_0-a L_1+(a^2/2-pi^2/12)L_2). $
The UV/IR split requires independent integral classification. In particular,
the scaleless bubble is $(U,I,J,F)=(i pi^2,-i pi^2,0,0)$ in the original wrapper.
Applying the MadLoop conversion to an already normalized master result would
count normalization twice.

== Polynomial and tensor interfaces
A rank bound $R$ has $binom(R+4,4)$ Cartesian coefficients in the original
four-component basis. #raw("ti_reduce::calc_pos") traverses total degree in
ascending order, then exponents of $q^0$, $q^1$, $q^2$ in descending order;
the exponent of $q^3$ is the remaining degree. Thus the start is
$(1,q^0,q^1,q^2,q^3,(q^0)^2,q^0 q^1,q^0 q^2,q^0 q^3,(q^1)^2, dots)$.
Numerator preparation and tensor output must share this order. The original
native and MadLoop tensor conventions differ in
multinomial factors: the native path multiplies symmetric components by the
factor returned by #raw("all_integers"), whereas
#raw("ML5_CONVENTION=.TRUE.") uses unweighted components. The independent
oracle sets this flag *without* applying #raw("IMLOOP"), preserving the native
four-pole normalization. Mixed-monomial tests detect multiplicity mismatches.

Reduction modes are 0: IBP, 1: PaVe, 2: PaVe with IBP stability improvement.
Direction mode 2 reverses the propagator order and shifts the loop origin.
The source labels direction modes 4/5 as unimplemented quadruple precision;
their numbers are not evidence of high-precision support.
