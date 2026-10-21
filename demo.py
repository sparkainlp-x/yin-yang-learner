"""Demo: does the predictive tic extend the self-recreation
horizon vs the symmetric baseline?

Two membranes A (yin) and B (yang), N=8, K=3 embedded Bell
patterns, observation freqs 0.6/0.3/0.1, sigma=0.25, alpha=0.01,
6000 steps per generation, 5 generations, seeded.

Prints per-generation fidelity to the true patterns:
  - SYMMETRIC: both read the other's present extraction
  - TIC: B reads A's fresh (next-generation) extraction
  - CONTROL: single membrane self-recreation (no partner)

Run: python demo.py            (seed 7)
     python demo.py 7 21 42   (the seeds in DESIGN.md)
"""

from ylearner import symmetric_chain, tic_chain, control_chain

S2 = 2 ** -0.5
BELLS = [[S2, 0j, 0j, S2], [S2, 0j, 0j, -S2], [0j, S2, S2, 0j]]
TRUE = [b + [0j] * 4 for b in BELLS]
FREQS = [0.6, 0.3, 0.1]
K, SIGMA, ALPHA, STEPS, GENS, SEED = 3, 0.25, 0.01, 6000, 5, 7


def main(SEED=SEED):
    print(f"seed {SEED}")
    sym = symmetric_chain(TRUE, K, GENS, STEPS, SIGMA, ALPHA,
                          FREQS, SEED)
    tic = tic_chain(TRUE, K, GENS, STEPS, SIGMA, ALPHA, FREQS, SEED)
    ctl = control_chain(TRUE, K, GENS, STEPS, SIGMA, ALPHA,
                        FREQS, SEED)
    print(f"{'gen':>4} {'sym-A':>7} {'sym-B':>7} "
          f"{'tic-A':>7} {'tic-B':>7} {'control':>7}")
    for g in range(GENS):
        print(f"{g:>4} {sym[g][0]:>7.4f} {sym[g][1]:>7.4f} "
              f"{tic[g][0]:>7.4f} {tic[g][1]:>7.4f} {ctl[g]:>7.4f}")
    print()
    print("mean fidelity over generations 1-4 (horizon, excluding "
          "the seeded gen 0):")
    for name, rows in (("symmetric-A", [r[0] for r in sym]),
                       ("symmetric-B", [r[1] for r in sym]),
                       ("tic-A", [r[0] for r in tic]),
                       ("tic-B", [r[1] for r in tic]),
                       ("control", ctl)):
        m = sum(rows[1:]) / (GENS - 1)
        print(f"  {name:>11}: {m:.4f}")
    print()
    print("RESULT: the table is the verdict -- see DESIGN.md.")


if __name__ == "__main__":
    import sys
    for s in (sys.argv[1:] or [SEED]):
        main(int(s))
