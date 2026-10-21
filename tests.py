"""Tests for yin-yang-learner. Run: python tests.py"""

import math
import random

from ylearner import (normalize, fidelity, noisy, zeros, hebbian_update,
                      _jacobi_real, eigh_hermitian, spectrum_entropy,
                      crystallize, match_fidelity, subspace_fidelity,
                      symmetric_chain, tic_chain, control_chain)

S2 = 2 ** -0.5
BELLS = [[S2, 0j, 0j, S2], [S2, 0j, 0j, -S2], [0j, S2, S2, 0j]]
TRUE = [b + [0j] * 4 for b in BELLS]
FREQS = [0.6, 0.3, 0.1]
P = dict(k=3, steps=1500, sigma=0.25, alpha=0.02, freqs=FREQS, seed=7)


def test_jacobi_known():
    vals, _ = _jacobi_real([[2.0, 1.0], [1.0, 2.0]])
    assert abs(sorted(vals)[0] - 1.0) < 1e-9
    assert abs(sorted(vals)[1] - 3.0) < 1e-9


def test_hermitian_reconstruction():
    rng = random.Random(3)
    n = 4
    H = zeros(n)
    for _ in range(50):
        x = normalize([complex(rng.gauss(0, 1), rng.gauss(0, 1))
                       for _ in range(n)])
        hebbian_update(H, x, 0.1)
    vals, vecs = eigh_hermitian(H)
    err = 0.0
    for a in range(n):
        for b in range(n):
            rec = sum(vals[i] * vecs[i][a] * vecs[i][b].conjugate()
                      for i in range(n))
            err = max(err, abs(rec - H[a][b]))
    assert err < 1e-9, err


def test_chains_deterministic():
    r1 = symmetric_chain(TRUE, generations=2, **P)
    r2 = symmetric_chain(TRUE, generations=2, **P)
    assert r1 == r2
    t1 = tic_chain(TRUE, generations=2, **P)
    t2 = tic_chain(TRUE, generations=2, **P)
    assert t1 == t2


def test_tic_reads_fresh_extraction():
    # At generation 0 both conditions start from identical inputs,
    # so A's extraction is identical in both. B differs iff the tic
    # reads A's *fresh* extraction instead of the stale one.
    sym = symmetric_chain(TRUE, generations=1, **P)
    tic = tic_chain(TRUE, generations=1, **P)
    assert sym[0][0] == tic[0][0], "A must be identical at gen 0"
    assert sym[0][1] != tic[0][1], \
        "B must differ: tic reads fresh A, symmetric reads stale A"


def test_tic_b_matches_direct_fresh_read():
    # tic's B at gen 0 must equal crystallizing directly from
    # symmetric's gen-0 A (the fresh extraction) with B's seed.
    from ylearner import crystallize
    sym = symmetric_chain(TRUE, generations=1, **P)
    tic = tic_chain(TRUE, generations=1, **P)
    A_fresh = crystallize([normalize(list(p)) for p in TRUE],
                          P["k"], P["steps"], P["sigma"], P["alpha"],
                          P["seed"] + 1, P["freqs"])
    B_direct = crystallize(A_fresh, P["k"], P["steps"], P["sigma"],
                           P["alpha"], P["seed"] + 2, P["freqs"])
    assert abs(match_fidelity(B_direct, TRUE) - tic[0][1]) < 1e-12


def test_fidelities_bounded():
    for chain in (symmetric_chain, tic_chain):
        for fa, fb in chain(TRUE, generations=2, **P):
            assert 0.0 <= fa <= 1.0 and 0.0 <= fb <= 1.0
    for f in control_chain(TRUE, generations=2, **P):
        assert 0.0 <= f <= 1.0


def test_first_generation_recovers_patterns():
    # generation 0 starts from the true patterns: extraction must
    # still find them (sanity on the whole pipeline).
    sym = symmetric_chain(TRUE, generations=1, **P)
    tic = tic_chain(TRUE, generations=1, **P)
    ctl = control_chain(TRUE, generations=1, **P)
    assert sym[0][0] > 0.80 and sym[0][1] > 0.80
    assert tic[0][0] > 0.80 and tic[0][1] > 0.80
    assert ctl[0] > 0.80


def test_entropy_extremes():
    assert abs(spectrum_entropy([1.0, 1.0]) - math.log(2)) < 1e-12
    assert abs(spectrum_entropy([1.0, 0.0])) < 1e-12


if __name__ == "__main__":
    for name, fn in sorted(
            [(k, v) for k, v in globals().items() if k.startswith("test_")]):
        fn()
        print(f"PASS {name}")
    print("all tests passed")
