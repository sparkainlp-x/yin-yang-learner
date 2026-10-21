"""Yin-Yang Learner: two conjugate holographic membranes that
crystallize by learning from EACH OTHER across an asymmetric tic.

Each membrane integrates a noisy stream Hebbian-style:
    W_t = (1 - alpha) * W_{t-1} + alpha * |x_t><x_t|
and extracts its top-K eigenvectors as its learned patterns.

Two conditions, 5 generations each, same seeds:

  SYMMETRIC baseline: generation g+1 -- A crystallizes from noisy
  observations of B_g's extracted top-K, and B crystallizes from
  noisy observations of A_g's extracted top-K. Both read the
  other's PRESENT.

  TIC (predictive): A_{g+1} crystallizes from B_g's extraction;
  THEN B_{g+1} crystallizes from A_{g+1}'s FRESH extraction --
  yang reads yin's predicted next state, one generation ahead.
  The asymmetric time coupling from the yin-yang brief, now in
  the learning dynamics.

Plus a CONTROL: a single membrane's self-recreation chain (no
partner), as in the crystallization build.

THE QUESTION: does the predictive tic extend the self-recreation
horizon vs the symmetric baseline? Whatever the numbers say is
the finding.

Core math (stdlib only): complex amplitudes, Hebbian update, and
a cyclic Jacobi eigensolver for Hermitian matrices via the real
symmetric embedding [[Re, -Im], [Im, Re]].
"""

import math
import random


# ---------------------------------------------------------------- utils
def normalize(v):
    n = math.sqrt(sum(abs(c) ** 2 for c in v))
    return [c / n for c in v]


def fidelity(a, b):
    return abs(sum(x.conjugate() * y for x, y in zip(a, b))) ** 2


def noisy(state, sigma, rng):
    return normalize([c + complex(rng.gauss(0, sigma), rng.gauss(0, sigma))
                      for c in state])


def zeros(n):
    return [[0j] * n for _ in range(n)]


def hebbian_update(W, x, alpha):
    """W <- (1-alpha) W + alpha |x><x|  (in place)."""
    n = len(W)
    for i in range(n):
        xi = x[i]
        Wi = W[i]
        for j in range(n):
            Wi[j] = (1 - alpha) * Wi[j] + alpha * xi * x[j].conjugate()
    return W


# ------------------------------------------------- Jacobi eigensolver
def _jacobi_real(A, tol=1e-13, max_sweeps=200):
    """Eigenvalues/vectors of a real symmetric matrix (cyclic Jacobi)."""
    n = len(A)
    A = [row[:] for row in A]
    V = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for _ in range(max_sweeps):
        changed = False
        for p in range(n - 1):
            for q in range(p + 1, n):
                apq = A[p][q]
                if abs(apq) > tol:
                    changed = True
                    app, aqq = A[p][p], A[q][q]
                    tau = (aqq - app) / (2.0 * apq)
                    t = (1.0 if tau >= 0 else -1.0) / (abs(tau) + math.sqrt(1 + tau * tau))
                    c = 1.0 / math.sqrt(1 + t * t)
                    s = t * c
                    for k in range(n):
                        akp, akq = A[k][p], A[k][q]
                        A[k][p] = c * akp - s * akq
                        A[k][q] = s * akp + c * akq
                    for k in range(n):
                        apk, aqk = A[p][k], A[q][k]
                        A[p][k] = c * apk - s * aqk
                        A[q][k] = s * apk + c * aqk
                    for k in range(n):
                        vkp, vkq = V[k][p], V[k][q]
                        V[k][p] = c * vkp - s * vkq
                        V[k][q] = s * vkp + c * vkq
        if not changed:
            break
    return [A[i][i] for i in range(n)], V


def eigh_hermitian(H):
    """Eigenvalues (desc) and eigenvectors of complex Hermitian H."""
    n = len(H)
    M = [[0.0] * (2 * n) for _ in range(2 * n)]
    for i in range(n):
        for j in range(n):
            c = H[i][j]
            M[i][j] = c.real
            M[i][j + n] = -c.imag
            M[i + n][j] = c.imag
            M[i + n][j + n] = c.real
    vals, vecs = _jacobi_real(M)
    # each eigenvalue appears twice; take every other after sorting desc
    order = sorted(range(2 * n), key=lambda i: vals[i], reverse=True)
    eigvals, eigvecs = [], []
    used = [False] * (2 * n)
    for i in order:
        if used[i]:
            continue
        used[i] = True
        # mark its pair twin (nearest unused with ~equal eigenvalue)
        for j in order:
            if not used[j] and abs(vals[j] - vals[i]) < 1e-9:
                used[j] = True
                break
        eigvals.append(vals[i])
        eigvecs.append([complex(vecs[k][i], vecs[k + n][i])
                        for k in range(n)])
    eigvecs = [normalize(v) for v in eigvecs]
    return eigvals, eigvecs


def spectrum_entropy(eigvals):
    """Von Neumann entropy of the normalized spectrum (nats)."""
    tot = sum(eigvals)
    s = 0.0
    for l in eigvals:
        p = l / tot
        if p > 1e-15:
            s -= p * math.log(p)
    return s


def subspace_fidelity(extracted, true):
    """Mean squared projection of each true pattern onto the span of
    the extracted top-K."""
    tot = 0.0
    for t in true:
        tot += sum(abs(sum(v[i].conjugate() * t[i]
                           for i in range(len(t)))) ** 2
                   for v in extracted)
    return tot / len(true)


def match_fidelity(extracted, true):
    """Greedy best-matching mean fidelity (order-free)."""
    remaining = list(range(len(extracted)))
    tot = 0.0
    for t in true:
        j = max(remaining, key=lambda j: fidelity(extracted[j], t))
        remaining.remove(j)
        tot += fidelity(extracted[j], t)
    return tot / len(true)


# ------------------------------------------------- crystallization
def crystallize(patterns, k, steps, sigma, alpha, seed, freqs=None):
    """Stream noisy observations of `patterns`; return the extracted
    top-K eigenvectors."""
    n = len(patterns[0])
    rng = random.Random(seed)
    if freqs is None:
        freqs = [1.0 / len(patterns)] * len(patterns)
    cum = []
    s = 0.0
    for f in freqs:
        s += f
        cum.append(s)
    W = zeros(n)
    for _ in range(steps):
        r = rng.random()
        p = patterns[next(i for i, c in enumerate(cum) if r < c)]
        hebbian_update(W, noisy(p, sigma, rng), alpha)
    _, vecs = eigh_hermitian(W)
    return vecs[:k]


# ------------------------------------------------- coupled chains
def _init(true):
    return [normalize(list(p)) for p in true]


def symmetric_chain(true, k, generations, steps, sigma, alpha,
                    freqs, seed):
    """Both membranes read the other's PRESENT extraction.

    Returns [(fid_A, fid_B), ...] per generation, fidelity to the
    true patterns."""
    cur_a, cur_b = _init(true), _init(true)
    rows = []
    for g in range(generations):
        A = crystallize(cur_b, k, steps, sigma, alpha,
                        seed + 2000 * g + 1, freqs)
        B = crystallize(cur_a, k, steps, sigma, alpha,
                        seed + 2000 * g + 2, freqs)
        rows.append((match_fidelity(A, true), match_fidelity(B, true)))
        cur_a, cur_b = A, B
    return rows


def tic_chain(true, k, generations, steps, sigma, alpha, freqs, seed):
    """Predictive tic: A_{g+1} from B_g's extraction, then B_{g+1}
    from A_{g+1}'s FRESH extraction -- yang reads yin's predicted
    next state, one generation ahead.

    Returns [(fid_A, fid_B), ...] per generation."""
    cur_a, cur_b = _init(true), _init(true)
    rows = []
    for g in range(generations):
        A = crystallize(cur_b, k, steps, sigma, alpha,
                        seed + 2000 * g + 1, freqs)
        # the tic: B reads A's *updated* extraction, not the stale one
        B = crystallize(A, k, steps, sigma, alpha,
                        seed + 2000 * g + 2, freqs)
        rows.append((match_fidelity(A, true), match_fidelity(B, true)))
        cur_a, cur_b = A, B
    return rows


def control_chain(true, k, generations, steps, sigma, alpha,
                  freqs, seed):
    """Single membrane self-recreation (no partner): each generation
    crystallizes from the previous generation's extracted patterns."""
    cur = _init(true)
    fids = []
    for g in range(generations):
        ext = crystallize(cur, k, steps, sigma, alpha,
                          seed + 1000 * g, freqs)
        fids.append(match_fidelity(ext, true))
        cur = ext
    return fids
