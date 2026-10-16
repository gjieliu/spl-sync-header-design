"""
design_v2 -- resubmission design core.  Three changes vs the submitted letter.

(A) FALSE ALARM IS NOW A CERTIFIED UPPER BOUND, not an independence approximation.
    (A1) balanced sync header (Hamming weight L_h/2) makes E[#matches]=L_h/2 for ANY
         background marginal pi, and sup_pi P(match>=tau) is attained at pi=1/2, i.e.
         the "fair bit" value P(Bin(L_h,1/2)>=tau) is an UPPER BOUND, not an assumption.
    (A2) p_W <= W*p_0  is the union bound over the W scanned offsets (any dependence).
    (A3) Under 2R <= L_c, scans after misses reveal fresh independent bits.
         Hit-run tails are bounded by u**ell, u=min(1,W*p0). Conditional geometric
         coupling gives binomial domination of cumulative hits, not slot independence.
    (A4) Sum fixed-start binomial tails over feasible finite-window hit times:
         P_FA <= p0 * sum_t [L_obs-L_h-t*T_f+(k-1)*R+1]_+
                    * comb(t-1,k-2)*u**(k-1)*(1-u)**(t-k+1).
         This equals M_eff*p0*P[Bin(N-1,u)>=k-1] (Proposition 3).

(B) DETECTION: the submitted product form 1-(1-p_track)^J is PROVABLY OPTIMISTIC.
    In the nominal model, tracks share independent coordinates H_i~Bern(q) and
    cap_i~Bern(q_cap), all 2*N_max coordinates mutually independent. Confirmation
    events are increasing, so Harris/FKG gives P_D <= 1-(1-p_track)^J.
    The product form is therefore a valid SEARCH RELAXATION (never discards a feasible
    point) but not a guarantee. The nominal scan probability is estimated by
    simulation of the unchanged two-sequence per-slot Bernoulli model.

(C) OBSERVATION WINDOW semantics are made explicit.  The submitted code counted the
    M starts and J true phases over L_max=2000 while the text said L_rej=500; the two
    give P_D 0.99 vs 0.07-0.44.  Here L_obs is an explicit input and the reported
    rejection latency IS L_obs.
"""
import numpy as np
from math import comb, log1p, expm1, nextafter, inf
from fractions import Fraction

# ---------- binomial helpers (no scipy on the target box) ----------

def binom_pmf(n, p, i):
    return comb(n, i) * p ** i * (1 - p) ** (n - i)


def binom_sf_ge(n, p, k):
    """P(Bin(n,p) >= k)."""
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    return float(sum(binom_pmf(n, p, i) for i in range(k, n + 1)))


# ---------- (A1) balanced-header false-hit bound ----------

def p0_balanced_sup(L_h, tau, n_grid=201):
    """Return (bound, maximizing pi, fair tail).

    For even balanced headers and integer tau>L_h/2 the continuous maximum is
    the fair tail: paired matches give symmetric unimodal three-point convolutions,
    whose upper tails increase in pi*(1-pi); the argument is Proposition 2 in the manuscript.
    Odd lengths retain the original grid diagnostic; it is NOT a certified supremum.
    """
    if L_h % 2 == 0 and L_h // 2 < tau <= L_h:
        fair = sum(comb(L_h, i) for i in range(tau, L_h + 1)) / 2 ** L_h
        return fair, 0.5, fair
    w = L_h // 2
    z = L_h - w
    best, best_pi = 0.0, 0.5
    for pi in np.linspace(0.0, 1.0, n_grid):
        pmf = np.zeros(L_h + 1)
        for a in range(w + 1):
            pa = binom_pmf(w, pi, a)
            if pa == 0.0:
                continue
            for b in range(z + 1):
                pmf[a + b] += pa * binom_pmf(z, 1 - pi, b)
        v = float(pmf[tau:].sum())
        if v > best:
            best, best_pi = v, float(pi)
    return best, best_pi, binom_sf_ge(L_h, 0.5, tau)


# ---------- (A) certified false alarm ----------

def _upward_float(value):
    """Round a nonnegative exact rational upward to binary64."""
    out = float(value)
    if Fraction.from_float(out) < value:
        out = nextafter(out, inf)
    return out


def p_fa_window_exact(p0, W, N, k, L_obs, L_h, T_f):
    """Exact rational version of Proposition 3's finite-window sum.

    Assumes iid background, valid p0 and 2R<=L_c, R=(W-1)/2, L_c=T_f-L_h.
    The factors are reference rth-success masses, not actual first-hit masses.
    """
    if W < 1 or W % 2 != 1 or not 1 <= k <= N:
        raise ValueError('Require odd W>=1 and 1<=k<=N')
    R = (W - 1) // 2
    if 2 * R > T_f - L_h:
        raise ValueError('Certification requires 2R<=L_c; no odd/even fallback')
    p = Fraction(p0)
    if not 0 <= p <= 1:
        raise ValueError('p0 must lie in [0,1]')
    if k == 1:
        return min(Fraction(1), max(0, L_obs - L_h + 1) * p)
    n, r, u = N - 1, k - 1, min(Fraction(1), W * p)
    total = Fraction(0)
    for t in range(r, n + 1):
        weight = max(0, L_obs - L_h - t * T_f + r * R + 1)
        if not weight:
            break
        mass = comb(t - 1, r - 1) * u ** r * (1 - u) ** (t - r)
        total += weight * mass
    return min(Fraction(1), p * total)


def p_fa_certified(p0, W, N, k, L_obs, L_h, T_f):
    """Finite-window iid certificate, rounded upward. No slot-independence claim."""
    return _upward_float(p_fa_window_exact(p0, W, N, k, L_obs, L_h, T_f))


def p_fa_product(p0, pW, N, k, M):
    """The submitted (non-certified) form, kept for comparison."""
    inner = 1.0 if k <= 1 else binom_sf_ge(N - 1, pW, k - 1)
    ps = min(1.0, p0 * inner)
    if ps <= 0:
        return 0.0
    return float(-expm1(M * log1p(-min(ps, 1 - 1e-16))))


# ---------- (B) detection ----------

def p_d_product(q_cap, q, N, k, J):
    """FKG-valid UPPER bound on P_D (the submitted formula)."""
    inner = 1.0 if k <= 1 else binom_sf_ge(N - 1, q, k - 1)
    ps = min(1.0, max(0.0, q_cap * inner))
    if J <= 0 or ps <= 0:
        return 0.0
    return float(-expm1(J * log1p(-min(ps, 1 - 1e-16))))


_SCAN_CACHE = {}


def p_d_scan(q_cap, q, N, k, J, N_max, n_sim=60000, seed=20260830):
    """Nominal P_D estimate: tracks share H~Bern(q) and cap~Bern(q_cap),
    with all 2*N_max coordinates mutually independent. Confirm iff some
    f<min(J,N_max-N+1) has cap[f]=1 and >=k-1 subsequent H hits.
    This working model is not a pathwise representation of the edit detector.
    """
    key = (round(q_cap, 6), round(q, 6), N, k, J, N_max, n_sim)
    if key in _SCAN_CACHE:
        return _SCAN_CACHE[key]
    fmax = min(J, N_max - N + 1)
    if fmax <= 0:
        _SCAN_CACHE[key] = 0.0
        return 0.0
    rng = np.random.default_rng(seed)
    H = rng.random((n_sim, N_max)) < q
    cap = rng.random((n_sim, N_max)) < q_cap
    cs = np.concatenate([np.zeros((n_sim, 1), dtype=np.int32),
                         np.cumsum(H, axis=1, dtype=np.int32)], axis=1)
    conf = np.zeros(n_sim, dtype=bool)
    for f in range(fmax):
        conf |= (cap[:, f] & ((cs[:, f + N] - cs[:, f + 1]) >= k - 1))
    v = float(conf.mean())
    _SCAN_CACHE[key] = v
    return v


# ---------- drift / lock-window sizing (unchanged from the submitted version) ----------

def drift_pmf(T_f, lam_d, lam_i):
    step = np.array([lam_d, 1.0 - lam_d - lam_i, lam_i])       # P(-1), P(0), P(+1)
    pmf = np.array([1.0])
    for _ in range(T_f):
        pmf = np.convolve(pmf, step)
    return np.arange(-T_f, T_f + 1), pmf


def size_R_cumulative(T_f, lam, N_fr, eta_lock, R_cap=40, asym=0.0):
    """Smallest R holding lock over N_fr frames: 1-(1-P_out)^N_fr <= eta_lock.
    asym = (lam_i - lam_d)/lam  in [-1,1];  asym=0 is the symmetric baseline."""
    lam_i = lam * (1 + asym) / 2.0
    lam_d = lam * (1 - asym) / 2.0
    per_frame = 1.0 - (1.0 - eta_lock) ** (1.0 / max(int(N_fr), 1))
    support, pmf = drift_pmf(T_f, lam_d, lam_i)
    a = np.abs(support)
    for R in range(0, R_cap + 1):
        P_out = float(pmf[a > R].sum())
        if P_out <= per_frame:
            return R, 1.0 - P_out
    return R_cap, 1.0 - float(pmf[a > R_cap].sum())


# ---------- the constrained search ----------

def search(lam, alpha=1e-9, eps=0.05, L_c=10, delta=0.01, eta_lock=1e-3,
           m_q=0.15, f_fa=2.0, L_max=2000, L_obs=2000, track_cap=None,
           L_h_lo=6, L_h_hi=26, require_even=True, require_disjoint=True,
           n_sim=60000, asym=0.0, sep_min=10.0, verbose=False):
    """Max conservative aligned-frame yield subject to
       (i)  certified P_FA <= alpha,
       (ii) scan-statistic P_D >= 1-delta   (product form used only as a relaxation),
       (iii) 2R <= L_c  (fresh scans after misses -> binomial domination),
       (iv) optional cap on the confirmation-track span N*T_f,
       (v)  a MARGINED feasibility bound  q >= sep_min * p_W.

    (v) is Prop. 1's condition beta*L_h > ln(2R+1) with an explicit separation factor.
    Without it the search accepts sync words so short that p_W ~ 0.1, and a chance full
    match inside the codeword region then steals the anchor: the H1-side yield becomes
    strongly sync-word dependent (it varies by 17% across balanced 6-bit words, reaching
    the design mu_Y itself for words of large autocorrelation sidelobe), so the
    conservative yield stops being conservative.  sep_min = 10 restores the regime
    q >> p_W in which the per-slot model is accurate.

    L_obs is the screening observation window and IS the rejection latency."""
    if not require_even or not require_disjoint:
        raise ValueError('Certified search requires even balance and 2R<=L_c')
    best = None
    for L_h in range(L_h_lo, L_h_hi + 1):
        if require_even and L_h % 2:
            continue
        T_f = L_h + L_c
        N_max = int(L_max // T_f)
        if N_max < 2:
            continue
        R, P_in = size_R_cumulative(T_f, lam, N_max, eta_lock, asym=asym)
        W = 2 * R + 1
        disjoint = (2 * R <= L_c)
        if require_disjoint and not disjoint:
            continue
        for tau in range(max(L_h // 2 + 1, int(np.floor(0.55 * L_h))), L_h + 1):
            p0 = p0_balanced_sup(L_h, tau)[0]
            P_tau = sum(binom_pmf(L_h, eps, i) for i in range(0, L_h - tau + 1))
            q_cap = (1 - lam) ** L_h * P_tau
            q = q_cap * P_in
            p0_chk = min(1.0, f_fa * p0)
            q_chk, q_cap_chk = (1 - m_q) * q, (1 - m_q) * q_cap
            pW_chk = min(1.0, W * p0_chk)
            # separation is stated on the RAW rates so that sep_min is interpretable
            # as the reported q/p_W; the safety margins act on the budget checks only.
            if q <= sep_min * min(1.0, W * p0):
                continue
            if q_chk <= pW_chk:
                continue
            hit = None
            for N in range(2, N_max + 1):
                span = (N - 1) * T_f + L_h
                if span > L_obs:
                    break
                if track_cap is not None and N * T_f > track_cap:
                    break
                M = L_obs - span + 1
                J = (L_obs - span + 1) // T_f
                kmin = None
                for k in range(2, N + 1):
                    if p_fa_certified(p0_chk, W, N, k, L_obs, L_h, T_f) <= alpha:
                        kmin = k
                        break
                if kmin is None:
                    continue
                if p_d_product(q_cap_chk, q_chk, N, kmin, J) < 1 - delta:
                    continue                       # FKG: product is an upper bound
                pd = p_d_scan(q_cap_chk, q_chk, N, kmin, J, N_max, n_sim=n_sim)
                if pd < 1 - delta:
                    continue
                hit = (N, kmin, pd, M, J)
                break
            if hit is None:
                continue
            N, k, pd, M, J = hit
            muY = N_max * q_chk
            cand = dict(lam=lam, L_h=L_h, tau=tau, R=R, W=W, T_f=T_f, N_scr=N, k_scr=k,
                        track_span=N * T_f, L_obs=L_obs, N_max=N_max,
                        muY=float(muY), q=float(q), q_cap=float(q_cap),
                        p0_bound=float(p0), pW_bound=float(min(1.0, W * p0)),
                        P_in=float(P_in), M=int(M), J=int(J),
                        P_FA_cert=float(p_fa_certified(p0_chk, W, N, k, L_obs, L_h, T_f)),
                        P_FA_product=float(p_fa_product(p0_chk, min(1, W * p0_chk),
                                                        N, k, M)),
                        P_D_scan=float(pd), sep=float(q / min(1.0, W * p0)),
                        P_D_product=float(p_d_product(q_cap_chk, q_chk, N, k, J)),
                        disjoint_scan=bool(disjoint), L_c=int(L_c))
            if best is None or cand["muY"] > best["muY"]:
                best = cand
                if verbose:
                    print("   cand", cand["L_h"], cand["tau"], "muY=%.1f" % cand["muY"])
    return best
