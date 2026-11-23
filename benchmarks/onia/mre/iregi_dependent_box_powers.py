"""Quad IREGI's scalar API with powers on a box whose propagators are
linearly dependent (benchmarks/onia/mre/IREGI_ONIA_MRE.md, item 5).

The box is a pinch of a g g > chic0 chic0 pentagon (two P-wave states at
sqrt(s) = 20 GeV): lines Q(p_2/2), g(p_2), g(p_3), Q(p_3/2) with p_2^2 =
p_3^2, so D_1 - D_2/2 + D_3/2 - D_4 = 0 exactly (every input is binary64 and
the halvings are exact). Raising line 1, 1 = (D_2/2 - D_3/2 + D_4) / D_1, so

    B(n1, n2, n3, n4) = B(n1+1, n2-1, n3, n4)/2 - B(n1+1, n2, n3-1, n4)/2
                        + B(n1+1, n2, n3, n4-1)

down to triangles. This script evaluates the box and those triangles with
quad IREGI (ONIA_QUAD) and prints both sides. IREGI satisfies the identity
at powers (2,1,1,1), to 1e-16, but not at (3,1,1,1) or (2,1,1,2): there its
IR pole and finite part are off by 5% to 1000%, and its UV slot, which must
vanish, is 1e-25 instead of 1e-38.
"""
import json
import os
import pathlib
import subprocess
from decimal import Decimal

ROOT = pathlib.Path(__file__).resolve().parents[3]
QUAD = pathlib.Path(
    os.environ.get("ONIA_QUAD", ROOT / "target/quad-iregi-source-audit/build-v6-001/iregi-reference-quad")
)
M2 = "0x1.33851eb851eb9p+1"  # m_c^2 = 2.4025
LINES = [
    (["-0x1.3ffffffffffffp+2", "-0x1.e2fb56247673bp+1", "0x1.2541aee163b5ap+1", "0x1.c384084d604c8p+0"], M2),
    (["-0x1.3ffffffffffffp+3", "-0x1.e2fb56247673bp+2", "0x1.2541aee163b5ap+2", "0x1.c384084d604c8p+1"], "0x0.0p+0"),
    (["0x1.3ffffffffffffp+3", "-0x1.e2fb56247673bp+2", "0x1.2541aee163b5ap+2", "0x1.c384084d604c8p+1"], "0x0.0p+0"),
    (["0x1.3ffffffffffffp+2", "-0x1.e2fb56247673bp+1", "0x1.2541aee163b5ap+1", "0x1.c384084d604c8p+0"], M2),
]
MU = "20.0"


def exact(h):
    return format(Decimal(float.fromhex(h)), ".40e")


def main():
    requests = []
    for powers in [(2, 1, 1, 1), (3, 1, 1, 1), (2, 1, 1, 2)]:
        requests.append(powers)
        for k in (1, 2, 3):
            if powers[k] == 1:
                requests.append(tuple(powers[0] + 1 if i == 0 else (0 if i == k else p) for i, p in enumerate(powers)))
        if powers[3] == 2:
            requests.append((powers[0] + 1, 1, 1, 1))
    requests = list(dict.fromkeys(requests))
    text = ""
    for k, powers in enumerate(requests):
        lines = [(line, p) for line, p in zip(LINES, powers) if p > 0]
        text += f"EVAL r{k} scalar 0 {len(lines)} 0 0 {MU} 1 1e-12\n"
        for (shift, mass), p in lines:
            text += " ".join(exact(x) for x in shift) + f" {exact(mass)} 0.0 {p}\n"
    run = subprocess.run([str(QUAD)], input=text + "QUIT\n", capture_output=True, text=True, timeout=600, check=True)
    value = {}
    for line in run.stdout.splitlines():
        if line.startswith("RESULT "):
            r = json.loads(line[7:])
            value[requests[int(r["id"][1:])]] = [complex(float(a), float(b)) for a, b in r["moments"][0]]

    def box(powers):
        """The identity's right-hand side, recursively down to triangles."""
        n1, n2, n3, n4 = powers
        terms = [(0.5, (n1 + 1, n2 - 1, n3, n4)), (-0.5, (n1 + 1, n2, n3 - 1, n4)), (1.0, (n1 + 1, n2, n3, n4 - 1))]
        total = [0j] * 4
        for weight, term in terms:
            part = value[term] if 0 in term else box(term)
            total = [t + weight * v for t, v in zip(total, part)]
        return total

    for powers in [(2, 1, 1, 1), (3, 1, 1, 1), (2, 1, 1, 2)]:
        print(f"box {powers}  (slots UV, IR1, IR2, finite)")
        for slot, (direct, identity) in enumerate(zip(value[powers], box(powers))):
            print(f"  {slot}: IREGI {direct.real:+.15e} {direct.imag:+.15e}i   identity {identity.real:+.15e} {identity.imag:+.15e}i")


if __name__ == "__main__":
    main()
