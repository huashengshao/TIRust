"""Fixture for P-wave dotted propagators in lightlike bubbles.

Writes crates/iregi-rust/tests/fixtures/onia-dotted-lightlike.json: the PV
coefficients C(lambda, k) (of g^lambda p^k, 2 lambda + k <= 4) of two-point
integrals at p^2 = 0 with one propagator doubled, from their d-dimensional
Feynman-parameter representation, in OneLOop's normalization
(d^d l / (i pi^(d/2)), (mu^2)^eps / Gamma(1 + eps), single poles only):

  C = (-1)^(k+nu+lambda) / (2^lambda Gamma(nu0) Gamma(nu1))
      Gamma(nu - lambda - 2 + eps) / Gamma(1 + eps)
      int_0^1 x^(k+nu1-1) (1-x)^(nu0-1) Delta^(2+lambda-nu) (Delta/mu^2)^-eps dx,
  Delta = (1-x) m0^2 + x m1^2,   propagators l^2 - m0^2, (l + p)^2 - m1^2.

With a massless line Delta is proportional to x or 1 - x and the integral is
a Beta function (mpmath continues it analytically); otherwise it converges
near eps = 0 and is integrated numerically. The Laurent coefficients come
from eps = +-1e-100 at 250 digits (the Beta functions near their poles
need them); residues below 1e-150 are that stencil's eps^2 term. A pole
is UV when it comes from the Gamma function (lambda at least the dotted
power minus one), IR when from the integral. A scaleless integral (both
lines massless) is zero, and its logarithmically divergent coefficient
keeps cancelling poles c (1/eps_UV - 1/eps_IR), c the UV residue of the
same coefficient with a massive line.

This is independent of TIRust's closed forms (direct/exec.rs lightlike_jets),
which expand the same integrals by hand.
"""
import json
import pathlib

import mpmath as mp

mp.mp.dps = 250
EPS = mp.mpf("1e-100")
MU2 = mp.mpf(9)
P = [3, 0, 0, 3]
MAX_RANK = 4


def coefficient(lam, k, nu0, nu1, m0, m1, eps):
    nu = nu0 + nu1
    pref = (-1) ** (k + nu + lam) / (2**lam * mp.gamma(nu0) * mp.gamma(nu1))
    pref *= mp.gamma(nu - lam - 2 + eps) / mp.gamma(1 + eps)
    expo = 2 + lam - nu - eps
    if m0 == 0 and m1 == 0:
        return mp.mpf(0)
    if m1 == 0:
        integral = m0**expo * mp.beta(k + nu1, nu0 + expo)
    elif m0 == 0:
        integral = m1**expo * mp.beta(k + nu1 + expo, nu0)
    else:
        f = lambda x: x ** (k + nu1 - 1) * (1 - x) ** (nu0 - 1) * ((1 - x) * m0 + x * m1) ** expo
        integral = mp.quad(f, [0, 1])
    return pref * integral * MU2**eps


def laurent(lam, k, nu0, nu1, m0, m1):
    a = coefficient(lam, k, nu0, nu1, m0, m1, EPS)
    b = coefficient(lam, k, nu0, nu1, m0, m1, -EPS)
    return (a - b) / 2 * EPS, (a + b) / 2


def slots(lam, k, nu0, nu1, m0, m1):
    """[uv, ir1, ir2, finite]."""
    dotted_power = max(nu0, nu1) - 1
    if m0 == 0 and m1 == 0:
        if lam != dotted_power:
            return [0, 0, 0, 0]
        massive = (4, 0) if nu0 > 1 else (0, 4)
        residue, _ = laurent(lam, k, nu0, nu1, *massive)
        return [residue, -residue, 0, 0]
    pole, finite = laurent(lam, k, nu0, nu1, m0, m1)
    if abs(pole) < mp.mpf("1e-150"):
        pole = mp.mpf(0)
    if lam >= dotted_power:
        return [pole, 0, 0, finite]
    return [0, pole, 0, finite]


CASES = [
    ("massless line 1 doubled, massive partner", (4, 0), (1, 2)),
    ("massless line 0 doubled, massive partner", (0, 4), (2, 1)),
    ("massive line 0 doubled, massless partner", (4, 0), (2, 1)),
    ("massive line 1 doubled, massless partner", (0, 4), (1, 2)),
    ("equal masses, line 0 doubled", (4, 4), (2, 1)),
    ("unequal masses, line 1 doubled", (4, 9), (1, 2)),
    ("unequal masses, line 0 doubled (|b/a| <= 1/2)", (4, 5), (2, 1)),
    ("scaleless, line 0 doubled", (0, 0), (2, 1)),
    ("scaleless, line 1 doubled", (0, 0), (1, 2)),
]


def main():
    cases = []
    for label, (m0, m1), (nu0, nu1) in CASES:
        coefficients = {}
        for lam in range(0, MAX_RANK // 2 + 1):
            for k in range(0, MAX_RANK - 2 * lam + 1):
                values = slots(lam, k, nu0, nu1, mp.mpf(m0), mp.mpf(m1))
                coefficients[f"{lam},{k}"] = [mp.nstr(mp.mpf(v), 34, min_fixed=-5, max_fixed=5) for v in values]
        cases.append(
            {
                "label": label,
                "shifts": [[0, 0, 0, 0], P],
                "masses_squared": [m0, m1],
                "powers": [nu0, nu1],
                "mu_squared": int(MU2),
                "coefficients": coefficients,
            }
        )
    out = pathlib.Path(__file__).resolve().parents[2] / "crates/iregi-rust/tests/fixtures/onia-dotted-lightlike.json"
    out.write_text(json.dumps({"max_rank": MAX_RANK, "cases": cases}, indent=1) + "\n")
    print(f"wrote {out} ({len(cases)} cases)")


if __name__ == "__main__":
    main()
