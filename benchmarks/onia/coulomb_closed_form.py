#!/usr/bin/env python3
"""Closed form of the Coulomb-exchange triangle at every tensor rank.

The triangle (c, g, c~) with D_c = (l+q_c)^2 - m^2, D_g = (l+q_g)^2,
D_c~ = (l+q_c~)^2 - m^2, q_g = (q_c + q_c~)/2 and p = (q_c - q_c~)/2 on
shell (p^2 = m^2) has linearly dependent propagators, 2 D_g = D_c + D_c~,
so

    1/(D_c D_g D_c~) = 1/2 [1/(D_g^2 D_c) + 1/(D_g^2 D_c~)].

With k = l + q_g each term is int (k - q_g)^alpha / ((k^2)^2 ((k +- p)^2 - m^2)).
Feynman parameters (weight 2 (1 - x), K = k +- x p, Delta = x^2 m^2) and
int d^DK K^(2j)/(K^2 - Delta)^3 give, per Cartesian term with j pairs of
K and x^s from the shift,

    (-1)^(j+1) Gamma(1-j+eps) / 2^j * pairings * (m^2)^(j-1-eps) / ((t+1)(t+2)),
    t = s + 2j - 2 - 2 eps,

in OneLOop's normalization (times (mu^2)^eps / r_Gamma). UV poles come from
Gamma(1 - j + eps) (j >= 1), IR poles from the x endpoint (s + 2j <= 1); no
term has both, and there is no double pole. This module evaluates that sum
(exact rational coefficients, binary64 logarithms) and converts it to IREGI's
native normalization, the convention of every array in benchmarks/onia.

Independent of TIRust and of IREGI.
"""
import math
from fractions import Fraction

EULER = 0.5772156649015329


def basis(rank):
    out = []
    for r in range(rank + 1):
        for t in range(r, -1, -1):
            for x in range(r - t, -1, -1):
                for y in range(r - t - x, -1, -1):
                    out.append((t, x, y, r - t - x - y))
    return out


def double_factorial_odd(n):
    """(n-1)!! for even n >= 0 (pairings of n equal indices)."""
    out = 1
    for k in range(n - 1, 0, -2):
        out *= k
    return out


def harmonic(n):
    return sum(Fraction(1, k) for k in range(1, n + 1))


def term_laurent(j, s, log_ratio):
    """(-1)^(j+1) Gamma(1-j+eps)/(2^j r_Gamma) (mu^2/m^2)^eps / ((t+1)(t+2))
    as (uv, ir, finite) floats (without (m^2)^(j-1))."""
    sign = -1 if (j + 1) % 2 else 1
    # Gamma(1-j+eps)/r_Gamma: 1 + O(eps^2) for j = 0, else a UV pole.
    if j == 0:
        g_pole, g_fin = 0.0, 1.0
    else:
        n = j
        c = (-1) ** (n - 1) / math.factorial(n - 1)
        g_pole, g_fin = c, c * float(harmonic(n - 1))
    a = s + 2 * j - 1                       # t + 1 = a - 2 eps, t + 2 = a + 1 - 2 eps
    if a == 0:
        r_pole, r_fin = -0.5, -1.0          # 1/((-2 eps)(1 - 2 eps))
    elif a == -1:
        r_pole, r_fin = 0.5, -1.0           # 1/((-1 - 2 eps)(-2 eps))
    else:
        r0 = 1.0 / (a * (a + 1))
        r_pole, r_fin, r_eps = 0.0, r0, r0 * 2.0 * (1.0 / a + 1.0 / (a + 1))
    # Product G(eps) R(eps) (1 + eps L); at most one of G, R has a pole.
    if g_pole and r_pole:
        raise AssertionError("no term has both poles")
    scale = 2.0 ** -j * sign
    if g_pole:
        # (g_pole/eps + g_fin)(r_fin + r_eps eps)(1 + eps L)
        uv = g_pole * r_fin
        fin = g_fin * r_fin + g_pole * r_eps + g_pole * r_fin * log_ratio
        return scale * uv, 0.0, scale * fin
    if r_pole:
        # g_fin (r_pole/eps + r_fin)(1 + eps L), g_fin = 1
        ir = g_fin * r_pole
        fin = g_fin * r_fin + g_fin * r_pole * log_ratio
        return 0.0, scale * ir, scale * fin
    return 0.0, 0.0, scale * g_fin * r_fin


def coulomb_triangle(shifts, masses_squared, rank, mu):
    """IREGI-native [uv, ir1, ir2, finite] moments (complex pairs) of the
    Coulomb triangle; shifts and masses as binary64 values in any order."""
    q = [[Fraction(x) for x in s] for s in shifts]
    m2s = [Fraction(m) for m in masses_squared]
    gluon = [i for i, m in enumerate(m2s) if m == 0]
    assert len(gluon) == 1, "one massless propagator"
    g = gluon[0]
    c, cbar = [i for i in range(3) if i != g]
    assert m2s[c] == m2s[cbar] > 0
    m2 = m2s[c]
    qg = q[g]
    p = [(a - b) / 2 for a, b in zip(q[c], q[cbar])]
    assert all(2 * x == a + b for x, a, b in zip(qg, q[c], q[cbar])), "q_g = (q_c + q_c~)/2"
    metric = [1, -1, -1, -1]
    assert sum(metric[m] * p[m] * p[m] for m in range(4)) == m2, "p^2 = m^2"
    log_ratio = math.log(mu * mu / float(m2))
    out = []
    for alpha in basis(rank):
        uv = ir = fin = 0.0
        for sign_p in (1, -1):
            pp = [sign_p * x for x in p]
            # prod_mu (K + c_mu)^alpha_mu, c = -(x pp + q_g); collect K^beta x^s.
            terms = {(): Fraction(1)}
            # Per component: list of (beta_mu, s_mu, coefficient).
            per = []
            for mu_, a_mu in enumerate(alpha):
                entries = []
                for beta in range(a_mu + 1):
                    rest = a_mu - beta
                    for s_mu in range(rest + 1):
                        coef = (Fraction(math.comb(a_mu, beta)) * math.comb(rest, s_mu)
                                * (-pp[mu_]) ** s_mu * (-qg[mu_]) ** (rest - s_mu))
                        if coef:
                            entries.append((beta, s_mu, coef))
                per.append(entries)
            acc = {}
            def walk(mu_, beta_total, s_total, pair_weight, coef):
                if mu_ == 4:
                    key = (beta_total // 2, s_total)
                    acc[key] = acc.get(key, Fraction(0)) + coef * pair_weight
                    return
                for beta, s_mu, c_mu in per[mu_]:
                    if beta % 2:
                        continue
                    weight = double_factorial_odd(beta) * metric[mu_] ** (beta // 2)
                    walk(mu_ + 1, beta_total + beta, s_total + s_mu, pair_weight * weight, coef * c_mu)
            walk(0, 0, 0, 1, Fraction(1))
            for (j, s), coef in acc.items():
                if not coef:
                    continue
                u, i, f = term_laurent(j, s, log_ratio)
                factor = 0.5 * float(coef) * float(m2) ** (j - 1)
                uv += factor * u
                ir += factor * i
                fin += factor * f
        # OneLOop -> IREGI native: times i pi^2, finite shifted by -(gamma_E + ln pi)
        # times the single poles (no double pole here).
        a = EULER + math.log(math.pi)
        pi2 = math.pi ** 2
        native = [uv, ir, 0.0, fin - a * (uv + ir)]
        out.append([[0.0, pi2 * v] for v in native])
    return out
