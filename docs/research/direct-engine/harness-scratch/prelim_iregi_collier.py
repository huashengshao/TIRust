#!/usr/bin/env python3
"""Preliminary (non-archival) same-CPU timings: original IREGI f64 (IBP/PV/Hybrid) and COLLIER (selected
opening-double option) on the 18 opening inputs. Run pinned via taskset. Output JSON lines to stdout.

usage: prelim_iregi_collier.py IREGI_TENSOR_BIN IREGI_GENERAL_BIN COLLIER_BIN
"""
import json, math, statistics, subprocess, sys, time
from fractions import Fraction as F
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from smoke_iregi import SUITE, ea

ROOT = Path('/shared/external/iregi/iregi-rust')
COLLIER_SUMMARY = ROOT / 'validation/benchmark/opening-double-20261026/collier/summary.json'
TARGET_NS = 20_000_000
REPEATS = 7


def f(x):
    return repr(float(F(x)))


def run(binary, text, timeout=600):
    out = subprocess.run([binary], input=text + 'QUIT\n', capture_output=True, text=True, timeout=timeout)
    res = [json.loads(l[7:]) for l in out.stdout.splitlines() if l.startswith('RESULT ')]
    return res, out


def iregi_wire(case, api, mode, warmup, repeats, iterations, general):
    req = case['driver_request']; s = req['samples'][0]
    if general:
        head = f"BENCH {case['id']} {api} {mode} {req['n']} {req['rank']} {2*req['dimension_shift']} {f(s['mu'])} 1 {warmup} {repeats} {iterations}"
        rows = [' '.join(f(x) for x in sh) + f" ({f(m[0])},{f(m[1])}) {p}" for sh, m, p in zip(s['shifts'], s['masses_squared'], req['powers'])]
    else:
        head = f"BENCH {case['id']} {mode} {req['n']} {req['rank']} {f(s['mu'])} 1 {warmup} {repeats} {iterations}"
        rows = [' '.join(f(x) for x in sh) + f" ({f(m[0])},{f(m[1])})" for sh, m in zip(s['shifts'], s['masses_squared'])]
    return '\n'.join([head] + rows) + '\n'


def collier_wire(case, opt, warmup, repeats, iterations):
    req = case['driver_request']; s = req['samples'][0]
    head = (f"BENCH {case['id']} {opt['mode']} {req['n']} {req['rank']} {f(s['mu'])} 1 {warmup} {repeats} {iterations} "
            f"{opt['requested_accuracy']} {opt['critical_accuracy']} {opt['direct_tensor_threshold']} {opt['argument_permutation']}")
    rows = [' '.join(f(x) for x in sh) + f" ({f(m[0])},{f(m[1])})" for sh, m in zip(s['shifts'], s['masses_squared'])]
    return '\n'.join([head] + rows) + '\n'


def timed(make, binary, case):
    pilot, out = run(binary, make(3, 1, 1))
    if not pilot:
        return {'status': 'no RESULT (STOP/crash)', 'tail': out.stdout.strip().splitlines()[-1:]}
    it = max(1, min(100000, math.ceil(TARGET_NS / pilot[0]['total_ns'][0])))
    res, out = run(binary, make(3, REPEATS, it))
    if not res:
        return {'status': 'no RESULT (STOP/crash)'}
    r = res[0]; lane = r['lanes'][0]
    us = [t / 1000 for t in r['total_ns']]
    row = {'status': lane['status'], 'stable': r['stable'], 'iterations': it, 'median_us': statistics.median(us),
           'mean_us': statistics.mean(us), 'min_us': min(us), 'se_us': statistics.stdev(us) / math.sqrt(len(us))}
    moments = lane.get('moments') or lane.get('diagnostic_moments')
    if moments and all(all(isinstance(z, list) for z in m) for m in moments):
        row['E_over_A'] = ea(case, moments)
    return row


def main():
    tensor_bin, general_bin, collier_bin = sys.argv[1:4]
    cases = json.loads(SUITE.read_text())['cases']
    collier_sel = {r['id']: r for r in json.loads(COLLIER_SUMMARY.read_text())['rows']}
    for case in cases:
        ordinary = case['table'] == 'ordinary'
        modes = (0, 1, 2) if ordinary else (0,)
        for mode in modes:
            general = not ordinary
            binary = general_bin if general else tensor_bin
            api = 'general' if general else 'tensor'
            t0 = time.monotonic()
            row = timed(lambda w, r, i: iregi_wire(case, api, mode, w, r, i, general), binary, case)
            row.update(id=case['id'], engine='IREGI', mode={0: 'IBP', 1: 'PV', 2: 'Hybrid'}[mode], api=api,
                       wall_s=round(time.monotonic() - t0, 2))
            print(json.dumps(row), flush=True)
        if ordinary:
            sel = collier_sel[case['id']]
            opt = next(a['configuration'] for a in sel['all_candidate_attempts'] if a['id'] == sel['selected_candidate'])
            row = timed(lambda w, r, i: collier_wire(case, opt, w, r, i), collier_bin, case)
            row.update(id=case['id'], engine='COLLIER', mode=opt['id'], archived_mean_us=sel['timing_us']['mean'])
            print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
