#!/usr/bin/env python3
"""Smoke: run original IREGI binary64 timing driver on opening inputs, compute E/A vs saved reference.

Usage (from iregi-rust root): python3 smoke_iregi.py BINARY MODE ID [ID...]
MODE: 0=IBP 1=PV 2=Hybrid.  Read-only w.r.t. the repo.
"""
import json, math, subprocess, sys, time
from fractions import Fraction as F
from pathlib import Path

ROOT = Path('/shared/external/iregi/iregi-rust')
SUITE = ROOT / 'validation/benchmark/opening-rank6-20261026/inputs.json'


def wire(case, mode, warmup, repeats, iterations, tag='BENCH'):
    req = case['driver_request']
    s = req['samples'][0]
    mu = float(F(s['mu']))
    lines = [f"{tag} {case['id']} {mode} {req['n']} {req['rank']} {mu!r} 1 {warmup} {repeats} {iterations}"]
    for shift, mass in zip(s['shifts'], s['masses_squared']):
        lines.append(' '.join(repr(float(F(x))) for x in shift) + f" ({float(F(mass[0]))!r},{float(F(mass[1]))!r})")
    return '\n'.join(lines) + '\n'


def reference(case):
    path = ROOT / 'validation/benchmark/opening-rank6-20261026/cases' / case['id'] / 'result.json'
    v = json.loads(path.read_text())['first_output']
    vals = v['lanes'][0]['values']
    return v['basis'], [tuple(F(c['exact_rational']) for c in pair) for pair in vals]


def ea(case, moments):
    req = case['driver_request']
    basis, ref = reference(case)
    mu = F(req['samples'][0]['mu'])
    E2 = A2 = F(0)
    for m, mono in enumerate(basis):
        expo = sum(mono) + 4 + 2 * req['dimension_shift'] - 2 * sum(req['powers'])
        w2 = mu ** (2 * expo)
        for slot in range(4):
            a = tuple(F(x) for x in moments[m][slot])
            b = ref[4 * m + slot]
            E2 = max(E2, ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) / w2)
            A2 = max(A2, (b[0] ** 2 + b[1] ** 2) / w2)
    return math.sqrt(float(E2 / A2)) if A2 else (0.0 if E2 == 0 else None)


def main():
    binary, mode, ids = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
    cases = {c['id']: c for c in json.loads(SUITE.read_text())['cases']}
    for ident in ids:
        case = cases[ident]
        text = wire(case, mode, 1, 3, 1) + 'QUIT\n'
        t0 = time.monotonic()
        try:
            out = subprocess.run([binary], input=text, capture_output=True, text=True, timeout=300)
        except subprocess.TimeoutExpired:
            print(json.dumps({'id': ident, 'status': 'timeout'}))
            continue
        res = [json.loads(l[7:]) for l in out.stdout.splitlines() if l.startswith('RESULT ')]
        row = {'id': ident, 'mode': mode, 'rc': out.returncode, 'wall_s': round(time.monotonic() - t0, 3)}
        if not res:
            row['status'] = 'no RESULT'
            row['tail'] = out.stdout[-400:]
        else:
            r = res[0]
            lane = r['lanes'][0]
            row.update(stable=r['stable'], total_us=[t / 1000 for t in r['total_ns']], lane_status=lane['status'])
            if lane['moments']:
                row['E_over_A'] = ea(case, lane['moments'])
        print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
