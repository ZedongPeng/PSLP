# PSQP

PSLP extended to convex QPs, objective (1/2) x'Px + c'x. P lives in
`Objective.P` (full symmetric CSR, no extra space), row sizes in
`State.P_row_sizes`.

- No reduction creates fill-in in P. A variable in P (`col_in_P`) is removed
  only by fixing (`fix_var_in_obj` folds its row of P into c) or, when its row
  of P is a positive diagonal alone (`diagonal_only_P_entry`), by dual fixing,
  the empty-column reduction or the doubleton eliminator
  (`sub_diag_var_in_obj_dton`). Singleton and parallel columns skip it.
- Every path that eliminates a variable calls `mark_P_row_inactive`.
- A P variable's dual: the eliminating record leaves the P part of the
  gradient for `DELETED_ROW_ZK` to move into the row multiplier.
- LP behaviour must stay bit-identical: run the netlib A/B in PSLP-benchmark.
