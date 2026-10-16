"""Recheck frozen R1 configurations; do not rerun or replace the design search.

Proposition 3: exact rational arithmetic for the finite-window certificate.
R1 result snapshots are inputs; only directly affected fields are updated.
"""
import copy
import hashlib
import itertools
import json
import math
import os
import sys
from fractions import Fraction as F
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import design_v2 as D

RES = HERE.parent / 'results'
REF = RES / 'r1_reference'


def load(name):
    return json.loads((REF / name).read_text())


def sf(n, u, r):
    if r <= 0:
        return F(1)
    if r > n:
        return F(0)
    return sum((math.comb(n, i) * u**i * (1-u)**(n-i)
                for i in range(r, n+1)), F(0))


def calculate(row, L_obs=None, N=None, k=None):
    h, T, W = row['L_h'], row['T_f'], row['W']
    N = row['N_scr'] if N is None else N
    k = row['k_scr'] if k is None else k
    L = row['L_obs'] if L_obs is None else L_obs
    # p0 is an exact dyadic fair-bit tail for the reported balanced headers.
    p0 = F(sum(math.comb(h, i) for i in range(row['tau'], h+1)), 2**h)
    assert float(p0) == row['p0_bound']
    p = min(F(1), 2*p0)  # original f_fa=2, both capture and subsequent slots
    u, r = min(F(1), W*p), k-1
    B = sf(N-1, u, r)
    exact = D.p_fa_window_exact(p, W, N, k, L, h, T)
    R = (W-1)//2
    M = max(0, L-h-(N-1)*T+1)
    Mk = max(0, L-h-r*(T-R)+1)
    return dict(old_nominal=float(M*p*B), corrected=D._upward_float(exact),
                exact_numerator=str(exact.numerator),
                exact_denominator=str(exact.denominator),
                M_nominal=M, M_k=Mk,
                M_eff=float(exact/(p*B)) if p*B else 0.0,
                coarse_bound=float(Mk*p*B))


def update_rows(data):
    rows = data['rows'].values() if isinstance(data, dict) else data
    changes = []
    for row in rows:
        if not row.get('feasible'):
            continue
        result = calculate(row)
        changes.append(dict(lam=row['lam'], asym=row.get('asym', 0),
                            P_FA_cert_R1=row['P_FA_cert'], **result))
        row['P_FA_cert'] = result['corrected']
        row['M_eff'] = result['M_eff']
        row['FA_formula'] = 'Proposition 3 finite-window weighted binomial tail'
        assert row['P_FA_cert'] <= 1e-9
    return changes


def recheck_stage1(main_rows):
    original = load('probeJ_two_stage.json')
    updated, changes = copy.deepcopy(original), []
    for before, row in zip(original, updated):
        design = main_rows['%g' % row['lam']]
        result = calculate(design, row['L1'], row['N1'], row['k1'])
        old_alpha = row['alpha1']
        # Preserve the actual R1 screen; adjust only its budget label if necessary.
        row['alpha1'] = next(a for a in [.01, .03, .1, .3]
                             if a >= result['corrected'])
        row['P_FA1'] = result['corrected']
        row['E_reject_ipds_upper'] = (row['L1'] + row['P_FA1']
                                     * (row['L_obs']-row['L1']))
        row.pop('E_reject_packets')
        row['length_unit'] = 'received IPDs; packet count is IPD count + 1'
        changes.append(dict(lam=row['lam'], alpha1_R1=old_alpha,
                            alpha1=row['alpha1'], P_FA1_R1=before['P_FA1'],
                            P_FA1=row['P_FA1'],
                            E_reject_R1=before['E_reject_packets'],
                            E_reject_ipds_upper=row['E_reject_ipds_upper']))
        assert row['P_FA1'] <= row['alpha1']
    return updated, changes

def finite_checks():
    checks = 0
    # Independent expression: sum cumulative reference tails over each fixed start.
    for n, r, L, p in itertools.product([2, 4, 6], [1, 2], [5, 10, 20],
                                       [F(0), F(1, 32), F(1, 3)]):
        h, T, W, R = 2, 4, 3, 1
        u = min(F(1), W*p)
        Mk = max(0, L-h-r*(T-R)+1)
        direct = p*sum((sf(min(n, (L-h-s+r*R)//T), u, r)
                        for s in range(Mk)), F(0))
        exact = D.p_fa_window_exact(p, W, n+1, r+1, L, h, T)
        assert exact == min(F(1), direct)
        checks += 1

    # Exhaustive finite-window blind detector: h=2,g=10,tau=2,R=1,T=4,N=k=3.
    # Only complete candidate heads count; ties go left; all initial heads are tested.
    by_weight = [0]*11
    for bits in itertools.product([0, 1], repeat=10):
        confirmed = False
        for s in range(9):
            if bits[s:s+2] != (1, 0):
                continue
            anchor, hits = s, 1
            for _ in range(2):
                pred = anchor+4
                candidates = [a for a in range(pred-1, pred+2) if 0 <= a <= 8]
                if not candidates:
                    break
                scores = [sum(x == y for x, y in zip(bits[a:a+2], (1, 0)))
                          for a in candidates]
                best = max(scores)
                if best == 2:
                    hits += 1
                    anchor = candidates[scores.index(best)]
                else:
                    anchor = pred
            if hits >= 3:
                confirmed = True
                break
        by_weight[sum(bits)] += confirmed
    enumeration = []
    for pi in [F(1, 4), F(1, 2), F(3, 4)]:
        measured = sum((c*pi**j*(1-pi)**(10-j)
                        for j, c in enumerate(by_weight)), F(0))
        bound = D.p_fa_window_exact(F(1, 4), 3, 3, 3, 10, 2, 4)
        assert measured <= bound
        enumeration.append(dict(pi=str(pi), actual=str(measured), bound=str(bound)))
        checks += 1
    assert enumeration[1]['actual'] == '37/256'
    return dict(exact_identity_checks=checks-3, exhaustive_cases=enumeration,
                purpose='Finite checks support implementation; they do not replace the proof.')


def main():
    table = load('design_v2_table.json')
    asym = load('design_v2_asym.json')
    changes = dict(main=update_rows(table), asym=update_rows(asym))
    stage1, changes['stage1'] = recheck_stage1(table['rows'])
    checks = finite_checks()
    nominal = []
    for row in table['rows'].values():
        if not row.get('feasible'):
            continue
        pd = D.p_d_scan(.85*row['q_cap'], .85*row['q'], row['N_scr'],
                       row['k_scr'], row['J'], row['N_max'], n_sim=80000)
        assert pd == row['P_D_scan'], (row['lam'], pd, row['P_D_scan'])
        nominal.append(dict(lam=row['lam'], P_D_scan=pd, unchanged=True))
    table['variant'] = ('R1 configurations retained; Proposition 3 finite-window '
                        'certificate rechecked; independent-bit analytical model')
    table['revision_scope'] = 'Recheck fixed R1 rows; no new optimality search.'
    for name, data in [('design_v2_table.json', table), ('design_v2_asym.json', asym),
                       ('probeJ_two_stage.json', stage1)]:
        (RES/name).write_text(json.dumps(data, indent=1)+'\n')
    report = dict(changes=changes, checks=checks, nominal_detection_reproduced=nominal,
                  inputs={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted(REF.glob('*.json'))})
    (RES/'recheck_local.json').write_text(json.dumps(report, indent=1)+'\n')
    for group, rows in changes.items():
        print(group)
        for row in rows:
            print(' ', row['lam'], row.get('asym', ''),
                  row.get('corrected', row.get('P_FA1')))
    print('Exact checks:', checks['exact_identity_checks'], '; blind enumerations: 3')
    print('Reproduced all seven original nominal P_D estimates without changing parameters.')


if __name__ == '__main__':
    main()
