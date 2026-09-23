# PSQP

PSLP extended to convex QPs, objective (1/2) x'Px + c'x. P lives in
`Objective.P` (full symmetric CSR, no extra space), row sizes in
`State.P_row_sizes`.

- A variable in P (`col_in_P`) is never fixed, substituted, dual-reduced or
  merged; `fix_col` asserts it. P is only renumbered, never modified.
- Every path that eliminates a variable calls `mark_P_row_inactive`.
- LP behaviour must stay bit-identical: run the netlib A/B in PSLP-benchmark.
