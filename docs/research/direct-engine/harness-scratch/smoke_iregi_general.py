#!/usr/bin/env python3
"""Smoke for scratch generalized IREGI driver. Usage: BINARY API MODE ID..."""
import json, subprocess, sys, time
from fractions import Fraction as F
sys.path.insert(0, sys.argv[0].rsplit('/', 1)[0])
from smoke_iregi import SUITE, ea

def wire(case, api, mode, warmup, repeats, iterations):
    req = case['driver_request']; s = req['samples'][0]
    lines = [f"BENCH {case['id']} {api} {mode} {req['n']} {req['rank']} {2*req['dimension_shift']} {float(F(s['mu']))!r} 1 {warmup} {repeats} {iterations}"]
    for shift, mass, p in zip(s['shifts'], s['masses_squared'], req['powers']):
        lines.append(' '.join(repr(float(F(x))) for x in shift) + f" ({float(F(mass[0]))!r},{float(F(mass[1]))!r}) {p}")
    return '\n'.join(lines) + '\n'

binary, api, mode, ids = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4:]
cases = {c['id']: c for c in json.loads(SUITE.read_text())['cases']}
for ident in ids:
    case = cases[ident]
    t0 = time.monotonic()
    try:
        out = subprocess.run([binary], input=wire(case, api, mode, 1, 3, 1) + 'QUIT\n', capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        print(json.dumps({'id': ident, 'status': 'timeout'})); continue
    res = [json.loads(l[7:]) for l in out.stdout.splitlines() if l.startswith('RESULT ')]
    row = {'id': ident, 'api': api, 'mode': mode, 'rc': out.returncode, 'wall_s': round(time.monotonic()-t0, 3)}
    if not res:
        row.update(status='no RESULT', tail=out.stdout[-300:])
    else:
        r = res[0]; lane = r['lanes'][0]
        row.update(stable=r['stable'], total_us=[t/1000 for t in r['total_ns']], lane_status=lane['status'])
        if lane['moments']: row['E_over_A'] = ea(case, lane['moments'])
    print(json.dumps(row), flush=True)
