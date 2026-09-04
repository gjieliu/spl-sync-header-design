"""
design_v2 -- resubmission design core.  Three changes vs the submitted letter.

(A) FALSE ALARM IS NOW A CERTIFIED UPPER BOUND, not an independence approximation.
    (A1) balanced sync header (Hamming weight L_h/2) makes E[#matches]=L_h/2 for ANY
         background marginal pi, and sup_pi P(match>=tau) is attained at pi=1/2, i.e.
         the "fair bit" value P(Bin(L_h,1/2)>=tau) is an UPPER BOUND, not an assumption.
    (A2) p_W <= W*p_0  is the union bound over the W scanned offsets (any dependence).
    (A3) if  2R <= L_c  the +-R scan windows of consecutive frame slots are DISJOINT
         (span W+L_h-1 <= T_f), so under independent background bits the per-slot
         indicators along one track are exactly independent -> the binomial is exact.
    (A4) union bound over the M candidate starts.
    =>  P_FA <= M * p_0 * P[Bin(N-1, W p_0) >= k-1]      (certified)

(B) DETECTION: the submitted product form 1-(1-p_track)^J is PROVABLY OPTIMISTIC.
    The J candidate tracks of a marked flow read the SAME periodic hit sequence
    H_1..H_Nmax ~ Bern(q); each confirmation event is increasing in that vector, so by
    Harris/FKG  P(no confirm) >= prod P(no confirm_f), i.e. P_D <= 1-(1-p_track)^J.
    The product form is therefore a valid SEARCH RELAXATION (never discards a feasible
    point) but not a guarantee.  True P_D is the SCAN STATISTIC, evaluated exactly in
    the per-slot Bernoulli model by simulation of that model.

(C) OBSERVATION WINDOW semantics are made explicit.  The submitted code counted the
    M starts and J true phases over L_max=2000 while the text said L_rej=500; the two
    give P_D 0.99 vs 0.07-0.44.  Here L_obs is an explicit input and the reported
    rejection latency IS L_obs.
"""
import numpy as np
from math import comb, log1p, expm1

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
    """sup over background marginal pi of P(#matches >= tau) for a header of weight
    w = floor(L_h/2)  (exact balance when L_h is even).  Matches = Bin(w,pi)+Bin(L_h-w,1-pi).
    Returns (sup, pi*, fair_bit_value)."""
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

def p_fa_certified(p0, W, N, k, M, disjoint=True):
    """Union-bound certified P_FA.

    disjoint=True  : requires 2R <= L_c, so the +-R scan windows of consecutive frame
                     slots do not overlap and the per-slot indicators along one track
                     are exactly independent -> the binomial is exact.
    disjoint=False : windows of ADJACENT slots overlap (2R > L_c).  Slots two apart are
                     still disjoint, so split the N-1 slots into the odd- and even-index
                     subsets, each internally independent, and use
                        P[sum >= k-1] <= 2 * P[Bin(ceil((N-1)/2), pW) >= ceil((k-1)/2)] ,
                     since at least one subset must carry half the required hits.
                     Rigorous for ANY dependence between the two subsets."""
    pW = min(1.0, W * p0)
    if k <= 1:
        inner = 1.0
    elif disjoint:
        inner = binom_sf_ge(N - 1, pW, k - 1)
    else:
        n_half = -(-(N - 1) // 2)
        k_half = -(-(k - 1) // 2)
        inner = min(1.0, 2.0 * binom_sf_ge(n_half, pW, k_half))
    return min(1.0, M * p0 * inner)


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
    """True P_D in the per-slot Bernoulli model: the J tracks share ONE hit sequence.
    confirm iff exists f<=J-1 with capture at f and >= k-1 hits in slots f+1..f+N-1."""
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
       (iii) 2R <= L_c  (disjoint scan windows -> the binomial in (i) is exact),
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
                    if p_fa_certified(p0_chk, W, N, k, M, disjoint) <= alpha:
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
                        P_FA_cert=float(p_fa_certified(p0_chk, W, N, k, M, disjoint)),
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
