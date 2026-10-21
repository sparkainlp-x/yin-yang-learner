# Yin-Yang Learner

**Two conjugate holographic membranes that crystallize by learning
from each other across an asymmetric tic -- and the measured
finding that the tic shortens the self-recreation horizon.**

A (yin) and B (yang) each integrate noisy Hebbian streams and
extract their top-K eigenvectors. Symmetric baseline: both read
the other's present extraction. Predictive tic: B reads A's fresh
next-generation extraction. Plus a single-membrane control.
Stdlib-only Python (includes a cyclic Jacobi Hermitian
eigensolver). AGPL-3.0.

## Verified (SYNTHETIC — seeded simulation, output of `python demo.py 7 21 42`)

- Predictive tic underperforms symmetric and control on all
  3 seeds (e.g. seed 7, mean gens 1-4: tic 0.73-0.78 vs
  symmetric 0.83-0.87 vs control 0.86)
- Mechanism: the "predicted" state is a re-estimated state --
  each extraction adds error, and the asymmetry compounds it
- 8/8 tests pass, including proofs that B reads A's fresh (not
  stale) extraction in the tic condition

All numbers are SYNTHETIC (fixed seeds, toy scale). "Yin" and
"yang" are naming metaphors only, for the two coupled membranes;
no philosophical, spiritual, energetic or health claim is made.
The "Bell" patterns are ordinary orthonormal complex vectors in a
classical Python simulation: no quantum hardware or effect is
claimed. Nothing here is a medical or clinical tool.

## Run it

```bash
python tests.py   # 8 tests
python demo.py            # per-generation fidelities, seed 7
python demo.py 7 21 42   # all three seeds in DESIGN.md
```

## Honest limits

One parameter set; the "prediction" is a re-read, not a true
predictive model; 3 seeds, 5 generations, toy scale. Full
account in `DESIGN.md`.

## Acknowledgements

Thanks to Mun Seok Lee for discussions and contributions.
