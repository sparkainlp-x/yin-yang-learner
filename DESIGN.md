# DESIGN — Deep think: the yin-yang learner

## The idea

The yin-yang membranes had the loop with the little tic; the
crystallization build had the learning. This fuses them: two
conjugate membranes that crystallize by learning from EACH OTHER
across an asymmetric time coupling.

- **SYMMETRIC baseline:** at generation g+1, A crystallizes from
  noisy observations of B_g's extracted top-K, and B from A_g's.
  Both read the other's PRESENT.
- **TIC (predictive):** A_{g+1} crystallizes from B_g's extraction;
  THEN B_{g+1} crystallizes from A_{g+1}'s FRESH extraction --
  yang reads yin's predicted next state, one generation ahead.
- **CONTROL:** a single membrane's self-recreation chain, no
  partner (the crystallization build's setup).

Parameters are the ones that worked: N=8, K=3 embedded Bell-state amplitude vectors (used as
classical complex vectors; no quantum effect),
freqs 0.6/0.3/0.1, sigma=0.25, alpha=0.01, 6000 steps/generation,
5 generations. Same seeds across conditions -- the comparison is
apples to apples.

## The verdict: the tic does not help. It hurts.

Mean fidelity to true patterns, generations 1-4 (gen 0 excluded)
-- SYNTHETIC, reproduce with `python demo.py 7 21 42`:

| seed | sym-A  | sym-B  | tic-A  | tic-B  | control |
|------|--------|--------|--------|--------|---------|
| 7    | 0.8334 | 0.8722 | 0.7780 | 0.7304 | 0.8619  |
| 21   | 0.9046 | 0.9151 | 0.8426 | 0.8165 | 0.8813  |
| 42   | 0.8556 | 0.8628 | 0.7955 | 0.7533 | 0.8382  |

The predictive tic shortens the self-recreation horizon on every
seed, for both membranes. The symmetric pair is roughly on par
with the single-membrane control (sometimes a hair above,
sometimes below) -- a partner reading your present neither
reliably helps nor hurts.

## Why the tic hurts (the mechanism)

The "predicted next state" is a *re-estimated* state. In the tic,
B learns from A_{g+1}, which was itself just extracted from B_g:
B's teacher is a twice-processed signal, and each extraction
adds estimation error. Worse, it feeds back -- B_g drifts faster,
so the next generation's A reads a worse B_g, and the whole
system compounds error faster than the symmetric pair. The
asymmetry creates an error-compounding loop, not a foresight
advantage. Reading the future is only useful if the prediction
is *better* than the present; here it is strictly noisier.

## Honest limits

- **One parameter set**, the crystallization build's working
  point. A different noise/plasticity regime could in principle
  flip the verdict -- untested, claimed nowhere.
- **The "prediction" is trivial**: B reads A's already-computed
  extraction. A genuine predictive model (extrapolating A's
  trajectory, denoising rather than re-estimating) was not
  tested and might behave differently.
- **Three seeds**, five generations, toy scale (8 oscillators,
  3 patterns). The verdict is consistent across the seeds
  measured, not proven in general.
- **Match fidelity, not subspace**: per-pattern identity is the
  reported metric (the interesting one for self-recreation);
  subspace fidelities are uniformly higher.
- The symmetric-vs-control comparison is a wash -- do not read
  "partners help" into noise-level differences.
