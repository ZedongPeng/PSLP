#ifndef TEST_QP_H
#define TEST_QP_H

#include "Constraints.h"
#include "CoreTransformations.h"
#include "Debugger.h"
#include "Matrix.h"
#include "Numerics.h"
#include "PSLP_API.h"
#include "PSLP_sol.h"
#include "PSLP_stats.h"
#include "Postsolver.h"
#include "Problem.h"
#include "SimpleReductions.h"
#include "State.h"
#include "Workspace.h"
#include "dVec.h"
#include "iVec.h"
#include "kkt.h"
#include "minunit.h"
#include "test_macros.h"
#include "u16Vec.h"
#include <stdio.h>

/* Tests for the quadratic objective term (1/2) x'Px. A variable that
   appears in P can be fixed (its row and column of P are folded away); it is
   never substituted or dual-reduced.
   Test 1: structural validation of P in new_presolver
   Test 2: fix_col folds the coupling terms of P into c and removes the
           row/column of P
   Test 3: singleton equality row fixing a variable that appears in P
           (postsolve recovers y from the gradient)
   Test 4: empty columns: fixed by the sign of c when absent from P, kept
           when present in P
   Test 5: simple dual fix skips a variable that appears in P
   Test 7: parallel columns are not merged when one of them appears in P
   Test 8: dual infeasibility ray after fixing a variable that appears in P
   Test 9: primal propagation fixing coupled variables (bound-change replay)
   Test 15: two coupled variables fixed by two singleton equality rows, in
            both orders (the first multiplier needs the later-fixed neighbour)
   Test 16: primal infeasibility ray after fixing a variable that appears in P
   (doubleton equality rows are not supported for QPs yet; new_presolver
   returns NULL if dton_eq is set)
*/

#define QP_TOL 1e-8
static int counter_qp = 0;

/*  Input validation. The constraint part is x0 + x1 <= 1, -1 <= x <= 1. */
static char *test_1_qp()
{
    double Ax[] = {1, 1};
    int Ai[] = {0, 1};
    int Ap[] = {0, 2};
    double lhs[] = {-INF};
    double rhs[] = {1};
    double lbs[] = {-1, -1};
    double ubs[] = {1, 1};
    double c[] = {1, 1};
    Settings *stgs = default_settings();
    stgs->verbose = false;
    stgs->dton_eq = false;
    Presolver *presolver;

    // asymmetric values
    {
        double Px[] = {1, 0.5, 0.7, 1};
        int Pi[] = {0, 1, 0, 1};
        int Pp[] = {0, 2, 4};
        presolver = new_presolver(Ax, Ai, Ap, 1, 2, 2, lhs, rhs, lbs, ubs, c, Px, Pi,
                                  Pp, 4, stgs);
        mu_assert("asymmetric P should be rejected", presolver == NULL);
    }

    // only one triangle given
    {
        double Px[] = {1, 0.5, 1};
        int Pi[] = {0, 1, 1};
        int Pp[] = {0, 2, 3};
        presolver = new_presolver(Ax, Ai, Ap, 1, 2, 2, lhs, rhs, lbs, ubs, c, Px, Pi,
                                  Pp, 3, stgs);
        mu_assert("upper triangular P should be rejected", presolver == NULL);
    }

    // unsorted row
    {
        double Px[] = {1, 2, 1, 2};
        int Pi[] = {1, 0, 0, 1};
        int Pp[] = {0, 2, 4};
        presolver = new_presolver(Ax, Ai, Ap, 1, 2, 2, lhs, rhs, lbs, ubs, c, Px, Pi,
                                  Pp, 4, stgs);
        mu_assert("unsorted P row should be rejected", presolver == NULL);
    }

    // duplicate column index in a row
    {
        double Px[] = {1, 1, 1};
        int Pi[] = {0, 0, 1};
        int Pp[] = {0, 2, 3};
        presolver = new_presolver(Ax, Ai, Ap, 1, 2, 2, lhs, rhs, lbs, ubs, c, Px, Pi,
                                  Pp, 3, stgs);
        mu_assert("duplicate P entry should be rejected", presolver == NULL);
    }

    // column index out of range
    {
        double Px[] = {1, 1};
        int Pi[] = {0, 2};
        int Pp[] = {0, 1, 2};
        presolver = new_presolver(Ax, Ai, Ap, 1, 2, 2, lhs, rhs, lbs, ubs, c, Px, Pi,
                                  Pp, 2, stgs);
        mu_assert("out of range P column should be rejected", presolver == NULL);
    }

    // row pointers: Pp[0] != 0, non-decreasing, Pp[n] == Pnnz
    {
        double Px[] = {1, 1, 1};
        int Pi[] = {0, 1, 1};
        int Pp_first[] = {1, 2, 3};
        int Pp_decreasing[] = {0, 2, 1};
        int Pp_last[] = {0, 1, 2};
        presolver = new_presolver(Ax, Ai, Ap, 1, 2, 2, lhs, rhs, lbs, ubs, c, Px, Pi,
                                  Pp_first, 3, stgs);
        mu_assert("Pp[0] != 0 should be rejected", presolver == NULL);
        presolver = new_presolver(Ax, Ai, Ap, 1, 2, 2, lhs, rhs, lbs, ubs, c, Px, Pi,
                                  Pp_decreasing, 1, stgs);
        mu_assert("decreasing Pp should be rejected", presolver == NULL);
        presolver = new_presolver(Ax, Ai, Ap, 1, 2, 2, lhs, rhs, lbs, ubs, c, Px, Pi,
                                  Pp_last, 3, stgs);
        mu_assert("Pp[n] != Pnnz should be rejected", presolver == NULL);
    }

    // a non-empty P needs all three arrays
    {
        double Px[] = {1, 1};
        int Pi[] = {0, 1};
        int Pp[] = {0, 1, 2};
        presolver = new_presolver(Ax, Ai, Ap, 1, 2, 2, lhs, rhs, lbs, ubs, c, Px,
                                  NULL, Pp, 2, stgs);
        mu_assert("missing Pi should be rejected", presolver == NULL);
        presolver = new_presolver(Ax, Ai, Ap, 1, 2, 2, lhs, rhs, lbs, ubs, c, Px, Pi,
                                  NULL, 2, stgs);
        mu_assert("missing Pp should be rejected", presolver == NULL);
    }

    // valid P, and explicit zeros are dropped
    {
        double Px[] = {2, 0, 0, 2};
        int Pi[] = {0, 1, 0, 1};
        int Pp[] = {0, 2, 4};
        presolver = new_presolver(Ax, Ai, Ap, 1, 2, 2, lhs, rhs, lbs, ubs, c, Px, Pi,
                                  Pp, 4, stgs);
        mu_assert("valid P should be accepted", presolver != NULL);
        mu_assert("explicit zeros in P should be dropped",
                  presolver->prob->obj->P->nnz == 2);
        mu_assert("P row sizes",
                  presolver->prob->constraints->state->P_row_sizes[0] == 1 &&
                      presolver->prob->constraints->state->P_row_sizes[1] == 1);
        mu_assert("Pnnz_original counts the entries after dropping zeros",
                  presolver->stats->Pnnz_original == 2);
        free_presolver(presolver);
    }

    // dton_eq with a non-empty P is rejected
    {
        double Px[] = {2, 2};
        int Pi[] = {0, 1};
        int Pp[] = {0, 1, 2};
        stgs->dton_eq = true;
        presolver = new_presolver(Ax, Ai, Ap, 1, 2, 2, lhs, rhs, lbs, ubs, c, Px, Pi,
                                  Pp, 2, stgs);
        mu_assert("dton_eq with P should be rejected", presolver == NULL);
        stgs->dton_eq = false;
    }

    // no P at all is an LP
    {
        presolver = new_presolver(Ax, Ai, Ap, 1, 2, 2, lhs, rhs, lbs, ubs, c, NULL,
                                  NULL, NULL, 0, stgs);
        mu_assert("empty P should be accepted", presolver != NULL);
        mu_assert("empty P should have no entries",
                  presolver->prob->obj->P->nnz == 0);
        mu_assert("empty P stats", presolver->stats->Pnnz_original == 0);
        free_presolver(presolver);
    }

    PS_FREE(stgs);
    return 0;
}

/*  Direct fix_col with a coupling term.
    min. (1/2) x'Px + [1 -1 0.5] x,  P = [2 1 0; 1 2 0; 0 0 4]
    s.t. x0 + x1 + x2 <= 10, 0 <= x <= 5
    Fixing x0 = 2 gives offset = 1*2 + (1/2)*2*4 = 6, c1 = -1 + 1*2 = 1,
    and P = [2 0; 0 4] on (x1, x2). */
static char *test_2_qp()
{
    double Ax[] = {1, 1, 1};
    int Ai[] = {0, 1, 2};
    int Ap[] = {0, 3};
    double lhs[] = {-INF};
    double rhs[] = {10};
    double lbs[] = {0, 0, 0};
    double ubs[] = {5, 5, 5};
    double c[] = {1, -1, 0.5};
    double Px[] = {2, 1, 1, 2, 4};
    int Pi[] = {0, 1, 0, 1, 2};
    int Pp[] = {0, 2, 4, 5};

    Settings *stgs = default_settings();
    stgs->verbose = false;
    stgs->dton_eq = false;
    Presolver *presolver = new_presolver(Ax, Ai, Ap, 1, 3, 3, lhs, rhs, lbs, ubs, c,
                                         Px, Pi, Pp, 5, stgs);
    mu_assert("init", presolver != NULL);
    Problem *prob = presolver->prob;
    Constraints *constraints = prob->constraints;
    State *state = constraints->state;
    Matrix *P = prob->obj->P;
    DEBUG(run_debugger_problem(prob, false));

    mu_assert("status", fix_col(prob, 0, 2.0) == REDUCED);

    // objective is updated eagerly
    mu_assert("offset", prob->obj->offset == 6.0);
    mu_assert("c after fold", prob->obj->c[1] == 1.0 && prob->obj->c[2] == 0.5);

    // row and column 0 of P are gone
    mu_assert("P nnz", P->nnz == 2);
    mu_assert("P row sizes", state->P_row_sizes[0] == SIZE_INACTIVE_COL &&
                                 state->P_row_sizes[1] == 1 &&
                                 state->P_row_sizes[2] == 1);
    mu_assert("P row 0 empty", P->p[0].start == P->p[0].end);
    mu_assert("P row 1", P->p[1].end - P->p[1].start == 1 &&
                             P->i[P->p[1].start] == 1 && P->x[P->p[1].start] == 2.0);

    // the postsolve record carries the folded c0 and the P row (2 entries)
    PostsolveInfo *info = state->postsolve_info;
    mu_assert("one record",
              info->type->len == 1 && info->type->data[0] == FIXED_COL);
    FixedColRecord rec = decode_fixed_col(info->indices->data, info->vals->data,
                                          info->starts->data[1]);
    mu_assert("record col/val/ck", rec.col == 0 && rec.val == 2.0 && rec.ck == 1.0);
    mu_assert("record a", rec.len == 1 && rec.rows[0] == 0 && rec.vals[0] == 1);
    mu_assert("record P", rec.P_len == 2 && rec.P_cols[0] == 0 &&
                              rec.P_vals[0] == 2 && rec.P_cols[1] == 1 &&
                              rec.P_vals[1] == 1);

    delete_fixed_cols_from_problem(prob);
    delete_inactive_cols_from_A_and_AT(constraints);
    mu_assert("rhs", constraints->rhs[0] == 8.0);
    DEBUG(run_debugger_problem(prob, false));

    problem_clean(prob);
    DEBUG(run_debugger_problem(prob, true));
    mu_assert("P dims after clean", P->m == 2 && P->n == 2 && P->nnz == 2);
    double Px_correct[] = {2, 4};
    int Pi_correct[] = {0, 1};
    int Pp_correct[] = {0, 1, 2};
    mu_assert("Px after clean", ARRAYS_EQUAL_DOUBLE(Px_correct, P->x, 2));
    mu_assert("Pi after clean", ARRAYS_EQUAL_INT(Pi_correct, P->i, 2));
    mu_assert("Pp after clean", check_row_starts(P, Pp_correct));
    mu_assert("c after clean", prob->obj->c[0] == 1.0 && prob->obj->c[1] == 0.5);
    mu_assert("P row sizes after clean",
              state->P_row_sizes[0] == 1 && state->P_row_sizes[1] == 1);

    PS_FREE(stgs);
    free_presolver(presolver);
    return 0;
}

/*  Singleton equality row fixing a variable that appears in P.
    min. (1/2) x'Px + [1 -3 1] x,  P = [2 1 0; 1 2 0; 0 0 0]
    s.t. x0 = 2,  x1 + x2 <= 3,  0 <= x <= 10
    Reduced (x1, x2): min x1^2 - x1 + x2 s.t. x1 + x2 <= 3, offset 6.
    Reduced solution x = (0.5, 0), y = 0, z = (0, 1). */
static char *test_3_qp()
{
    double Ax[] = {1, 1, 1};
    int Ai[] = {0, 1, 2};
    int Ap[] = {0, 1, 3};
    int n_rows = 2, n_cols = 3;
    double lhs[] = {2, -INF};
    double rhs[] = {2, 3};
    double lbs[] = {0, 0, 0};
    double ubs[] = {10, 10, 10};
    double c[] = {1, -3, 1};
    double Px[] = {2, 1, 1, 2};
    int Pi[] = {0, 1, 0, 1};
    int Pp[] = {0, 2, 4, 4};

    Settings *stgs = default_settings();
    set_settings_false(stgs);
    Presolver *presolver = new_presolver(Ax, Ai, Ap, n_rows, n_cols, 3, lhs, rhs,
                                         lbs, ubs, c, Px, Pi, Pp, 4, stgs);
    mu_assert("init", presolver != NULL);
    mu_assert("status", run_presolver(presolver) == REDUCED);

    PresolvedProblem *red = presolver->reduced_prob;
    mu_assert("reduced dims", red->m == 1 && red->n == 2 && red->nnz == 2);
    mu_assert("reduced offset", red->obj_offset == 6.0);
    mu_assert("reduced c", red->c[0] == -1.0 && red->c[1] == 1.0);
    mu_assert("reduced P nnz", red->Pnnz == 1);
    mu_assert("reduced P", red->Pp[0] == 0 && red->Pp[1] == 1 && red->Pp[2] == 1 &&
                               red->Pi[0] == 0 && red->Px[0] == 2.0);
    mu_assert("stats", presolver->stats->Pnnz_original == 4 &&
                           presolver->stats->Pnnz_reduced == 1);

    double x[] = {0.5, 0};
    double y[] = {0};
    double z[] = {0, 1};
    postsolve(presolver, x, y, z);

    double correct_x[] = {2, 0.5, 0};
    double correct_y[] = {5.5, 0};
    double correct_z[] = {0, 0, 1};
    mu_assert("postsolve values",
              is_solution_correct(presolver->sol->x, correct_x, presolver->sol->y,
                                  correct_y, presolver->sol->z, correct_z, n_rows,
                                  n_cols, QP_TOL));
    PresolvedProblem orig =
        problem_from_csr_qp(Ax, Ai, Ap, n_rows, n_cols, Ap[n_rows], lhs, rhs, lbs,
                            ubs, c, Px, Pi, Pp, Pp[n_cols]);
    mu_assert("KKT", is_kkt_point(&orig, presolver->sol->x, presolver->sol->y,
                                  presolver->sol->z, QP_TOL));

    PS_FREE(stgs);
    free_presolver(presolver);
    return 0;
}

/*  Empty columns with and without a quadratic term.
    min. (1/2) x'Px + [-4 1 0 0 0] x,
         P = 2 I on (x0, x2, x3), plus P_22 = P_44 = 2, P_24 = P_42 = 1
    s.t. 2 x2 + x3 = 4, -1 <= x0 <= 1, 0 <= x1 <= 5, -10 <= x2, x3, x4 <= 10
    x0, x1 and x4 are empty in A. x1 is not in P and goes to its lower bound
    (LP logic). x0 (diagonal only) and x4 (coupled to x2) appear in P and
    are kept. Reduced (x0, x2, x3, x4):
    min x0^2 - 4 x0 + x2^2 + x2 x4 + x4^2 + x3^2 s.t. 2 x2 + x3 = 4,
    solution (1, 32/19, 12/19, -16/19) with y = 24 / 19 and z0 = -2. */
static char *test_4_qp()
{
    double Ax[] = {2, 1};
    int Ai[] = {2, 3};
    int Ap[] = {0, 2};
    int n_rows = 1, n_cols = 5;
    double lhs[] = {4};
    double rhs[] = {4};
    double lbs[] = {-1, 0, -10, -10, -10};
    double ubs[] = {1, 5, 10, 10, 10};
    double c[] = {-4, 1, 0, 0, 0};
    double Px[] = {2, 2, 1, 2, 1, 2};
    int Pi[] = {0, 2, 4, 3, 2, 4};
    int Pp[] = {0, 1, 1, 3, 4, 6};

    Settings *stgs = default_settings();
    set_settings_false(stgs);
    Presolver *presolver = new_presolver(Ax, Ai, Ap, n_rows, n_cols, 2, lhs, rhs,
                                         lbs, ubs, c, Px, Pi, Pp, 6, stgs);
    mu_assert("init", presolver != NULL);
    mu_assert("status", run_presolver(presolver) == REDUCED);

    PresolvedProblem *red = presolver->reduced_prob;
    mu_assert("reduced dims", red->m == 1 && red->n == 4 && red->nnz == 2);
    mu_assert("reduced offset", red->obj_offset == 0.0);
    double c_correct[] = {-4, 0, 0, 0};
    mu_assert("reduced c", ARRAYS_EQUAL_DOUBLE(c_correct, red->c, 4));
    mu_assert("reduced P nnz", red->Pnnz == 6);
    double Px_correct[] = {2, 2, 1, 2, 1, 2};
    int Pi_correct[] = {0, 1, 3, 2, 1, 3};
    int Pp_correct[] = {0, 1, 3, 4, 6};
    mu_assert("reduced Px", ARRAYS_EQUAL_DOUBLE(Px_correct, red->Px, 6));
    mu_assert("reduced Pi", ARRAYS_EQUAL_INT(Pi_correct, red->Pi, 6));
    mu_assert("reduced Pp", ARRAYS_EQUAL_INT(Pp_correct, red->Pp, 5));
    mu_assert("P unchanged",
              presolver->stats->Pnnz_original == presolver->stats->Pnnz_reduced);

    // the kept columns (x0 and x4, now columns 0 and 3) are still listed as
    // empty
    State *state = presolver->prob->constraints->state;
    mu_assert("kept empty cols", state->empty_cols->len == 2 &&
                                     state->empty_cols->data[0] == 0 &&
                                     state->empty_cols->data[1] == 3 &&
                                     !has_removable_empty_cols(state));

    double x[] = {1, 32.0 / 19, 12.0 / 19, -16.0 / 19};
    double y[] = {24.0 / 19};
    double z[] = {-2, 0, 0, 0};
    postsolve(presolver, x, y, z);

    double correct_x[] = {1, 0, 32.0 / 19, 12.0 / 19, -16.0 / 19};
    double correct_y[] = {24.0 / 19};
    double correct_z[] = {-2, 1, 0, 0, 0};
    mu_assert("postsolve values",
              is_solution_correct(presolver->sol->x, correct_x, presolver->sol->y,
                                  correct_y, presolver->sol->z, correct_z, n_rows,
                                  n_cols, QP_TOL));
    PresolvedProblem orig =
        problem_from_csr_qp(Ax, Ai, Ap, n_rows, n_cols, Ap[n_rows], lhs, rhs, lbs,
                            ubs, c, Px, Pi, Pp, Pp[n_cols]);
    mu_assert("KKT", is_kkt_point(&orig, presolver->sol->x, presolver->sol->y,
                                  presolver->sol->z, QP_TOL));

    PS_FREE(stgs);
    free_presolver(presolver);
    return 0;
}

/*  Simple dual fix must skip a variable that appears in P.
    min. x0^2 + x0 x2 + x2^2 + x0 + x1
    s.t. x0 + x1 <= 5, x1 + x2 <= 5, -10 <= x <= 10
    For an LP all three variables would be dual-fixed. Here only x1 (not in
    P) is fixed to its lower bound; the rows become redundant singletons and
    x0, x2 empty columns that appear in P, so they stay. Reduced:
    min x0^2 + x0 x2 + x2^2 + x0 - 10, solution (-2/3, 1/3). */
static char *test_5_qp()
{
    double Ax[] = {1, 1, 1, 1};
    int Ai[] = {0, 1, 1, 2};
    int Ap[] = {0, 2, 4};
    int n_rows = 2, n_cols = 3;
    double lhs[] = {-INF, -INF};
    double rhs[] = {5, 5};
    double lbs[] = {-10, -10, -10};
    double ubs[] = {10, 10, 10};
    double c[] = {1, 1, 0};
    double Px[] = {2, 1, 1, 2};
    int Pi[] = {0, 2, 0, 2};
    int Pp[] = {0, 2, 2, 4};

    Settings *stgs = default_settings();
    stgs->verbose = false;
    stgs->dton_eq = false;
    Presolver *presolver = new_presolver(Ax, Ai, Ap, n_rows, n_cols, 4, lhs, rhs,
                                         lbs, ubs, c, Px, Pi, Pp, 4, stgs);
    mu_assert("init", presolver != NULL);
    mu_assert("status", run_presolver(presolver) == REDUCED);

    PresolvedProblem *red = presolver->reduced_prob;
    mu_assert("reduced dims", red->m == 0 && red->n == 2 && red->nnz == 0);
    mu_assert("reduced offset", red->obj_offset == -10.0);
    mu_assert("reduced c", red->c[0] == 1.0 && red->c[1] == 0.0);
    double Px_correct[] = {2, 1, 1, 2};
    int Pi_correct[] = {0, 1, 0, 1};
    mu_assert("reduced P", red->Pnnz == 4 &&
                               ARRAYS_EQUAL_DOUBLE(Px_correct, red->Px, 4) &&
                               ARRAYS_EQUAL_INT(Pi_correct, red->Pi, 4));
    mu_assert("P unchanged",
              presolver->stats->Pnnz_original == presolver->stats->Pnnz_reduced);

    double x[] = {-2.0 / 3, 1.0 / 3};
    double z[] = {0, 0};
    postsolve(presolver, x, NULL, z);

    double correct_x[] = {-2.0 / 3, -10, 1.0 / 3};
    double correct_y[] = {0, 0};
    double correct_z[] = {0, 1, 0};
    mu_assert("postsolve values",
              is_solution_correct(presolver->sol->x, correct_x, presolver->sol->y,
                                  correct_y, presolver->sol->z, correct_z, n_rows,
                                  n_cols, QP_TOL));
    PresolvedProblem orig =
        problem_from_csr_qp(Ax, Ai, Ap, n_rows, n_cols, Ap[n_rows], lhs, rhs, lbs,
                            ubs, c, Px, Pi, Pp, Pp[n_cols]);
    mu_assert("KKT", is_kkt_point(&orig, presolver->sol->x, presolver->sol->y,
                                  presolver->sol->z, QP_TOL));

    PS_FREE(stgs);
    free_presolver(presolver);
    return 0;
}

/*  Parallel columns are not merged when one of them appears in P.
    min. x0^2 - x0 - 2 x1 s.t. x0 + 2 x1 + x2 <= 1, x0 + 2 x1 - x2 <= 1,
    -1 <= x <= 1. Columns 0 and 1 are parallel with c1 = 2 c0, so an LP
    would merge them. Nothing can be reduced here; the solution is
    x = (0, 0.5, 0), y = (-0.5, -0.5). */
static char *test_7_qp()
{
    double Ax[] = {1, 2, 1, 1, 2, -1};
    int Ai[] = {0, 1, 2, 0, 1, 2};
    int Ap[] = {0, 3, 6};
    int n_rows = 2, n_cols = 3;
    double lhs[] = {-INF, -INF};
    double rhs[] = {1, 1};
    double lbs[] = {-1, -1, -1};
    double ubs[] = {1, 1, 1};
    double c[] = {-1, -2, 0};
    double Px[] = {2};
    int Pi[] = {0};
    int Pp[] = {0, 1, 1, 1};

    Settings *stgs = default_settings();
    stgs->verbose = false;
    stgs->dton_eq = false;
    Presolver *presolver = new_presolver(Ax, Ai, Ap, n_rows, n_cols, 6, lhs, rhs,
                                         lbs, ubs, c, Px, Pi, Pp, 1, stgs);
    mu_assert("init", presolver != NULL);
    mu_assert("status", run_presolver(presolver) == UNCHANGED);
    mu_assert("reduced dims",
              presolver->reduced_prob->n == 3 && presolver->reduced_prob->m == 2);
    mu_assert("P unchanged",
              presolver->reduced_prob->Pnnz == 1 &&
                  presolver->stats->Pnnz_original == presolver->stats->Pnnz_reduced);

    double x[] = {0, 0.5, 0};
    double y[] = {-0.5, -0.5};
    double z[] = {0, 0, 0};
    postsolve(presolver, x, y, z);
    mu_assert("postsolve values",
              is_solution_correct(presolver->sol->x, x, presolver->sol->y, y,
                                  presolver->sol->z, z, n_rows, n_cols, QP_TOL));
    PresolvedProblem orig =
        problem_from_csr_qp(Ax, Ai, Ap, n_rows, n_cols, Ap[n_rows], lhs, rhs, lbs,
                            ubs, c, Px, Pi, Pp, Pp[n_cols]);
    mu_assert("KKT", is_kkt_point(&orig, presolver->sol->x, presolver->sol->y,
                                  presolver->sol->z, QP_TOL));

    PS_FREE(stgs);
    free_presolver(presolver);
    return 0;
}

/*  Dual infeasibility ray after a variable in P is fixed by a singleton
    equality row.
    min. (1/2) x'Px + [1 0 -1] x,  P = [2 1 0; 1 2 0; 0 0 0]
    s.t. x0 = 1, x1 - x2 <= 5, 0 <= x0, x1 <= 10, 0 <= x2 < INF
    Reduced (x1, x2): min x1^2 + x1 - x2 s.t. x1 - x2 <= 5, unbounded along
    d = (0, 1). The recovered ray must be (0, 0, 1). */
static char *test_8_qp()
{
    double Ax[] = {1, 1, -1};
    int Ai[] = {0, 1, 2};
    int Ap[] = {0, 1, 3};
    int n_rows = 2, n_cols = 3;
    double lhs[] = {1, -INF};
    double rhs[] = {1, 5};
    double lbs[] = {0, 0, 0};
    double ubs[] = {10, 10, INF};
    double c[] = {1, 0, -1};
    double Px[] = {2, 1, 1, 2};
    int Pi[] = {0, 1, 0, 1};
    int Pp[] = {0, 2, 4, 4};

    Settings *stgs = default_settings();
    set_settings_false(stgs);
    Presolver *presolver = new_presolver(Ax, Ai, Ap, n_rows, n_cols, 3, lhs, rhs,
                                         lbs, ubs, c, Px, Pi, Pp, 4, stgs);
    mu_assert("init", presolver != NULL);
    mu_assert("status", run_presolver(presolver) == REDUCED);

    PresolvedProblem *red = presolver->reduced_prob;
    mu_assert("reduced dims", red->m == 1 && red->n == 2);
    mu_assert("reduced c", red->c[0] == 1.0 && red->c[1] == -1.0);
    mu_assert("reduced P", red->Pnnz == 1 && red->Px[0] == 2.0 && red->Pi[0] == 0);

    // the reduced ray is a valid certificate for the reduced problem
    double d_red[] = {0, 1};
    mu_assert("reduced ray", is_dual_ray_certificate(red, d_red, QP_TOL));

    double d[3];
    postsolve_dual_infeas_ray(presolver, d_red, d);
    double correct_d[] = {0, 0, 1};
    mu_assert("ray values", ABS(d[0] - correct_d[0]) <= QP_TOL &&
                                ABS(d[1] - correct_d[1]) <= QP_TOL &&
                                ABS(d[2] - correct_d[2]) <= QP_TOL);
    PresolvedProblem orig =
        problem_from_csr_qp(Ax, Ai, Ap, n_rows, n_cols, Ap[n_rows], lhs, rhs, lbs,
                            ubs, c, Px, Pi, Pp, Pp[n_cols]);
    mu_assert("ray certificate", is_dual_ray_certificate(&orig, d, QP_TOL));

    PS_FREE(stgs);
    free_presolver(presolver);
    return 0;
}

/*  Primal propagation fixing two coupled variables (the FIXED_COL records
    are replayed inside the bound-change walk of the postsolver).
    min. (1/2) x'Px + [1 1] x,  P = [2 1; 1 2]
    s.t. x0 + x1 <= 0, 0 <= x <= 10
    Both variables are forced to 0, so the reduced problem is empty and the
    solution is x = 0, y = 0, z = (1, 1). */
static char *test_9_qp()
{
    double Ax[] = {1, 1};
    int Ai[] = {0, 1};
    int Ap[] = {0, 2};
    int n_rows = 1, n_cols = 2;
    double lhs[] = {-INF};
    double rhs[] = {0};
    double lbs[] = {0, 0};
    double ubs[] = {10, 10};
    double c[] = {1, 1};
    double Px[] = {2, 1, 1, 2};
    int Pi[] = {0, 1, 0, 1};
    int Pp[] = {0, 2, 4};

    Settings *stgs = default_settings();
    set_settings_false(stgs);
    stgs->primal_propagation = true;
    stgs->finite_bound_tightening = true;
    Presolver *presolver = new_presolver(Ax, Ai, Ap, n_rows, n_cols, 2, lhs, rhs,
                                         lbs, ubs, c, Px, Pi, Pp, 4, stgs);
    mu_assert("init", presolver != NULL);
    mu_assert("status", run_presolver(presolver) == REDUCED);

    PresolvedProblem *red = presolver->reduced_prob;
    mu_assert("reduced dims", red->m == 0 && red->n == 0 && red->Pnnz == 0);
    mu_assert("reduced offset", red->obj_offset == 0.0);

    postsolve(presolver, NULL, NULL, NULL);

    double correct_x[] = {0, 0};
    double correct_y[] = {0};
    double correct_z[] = {1, 1};
    mu_assert("postsolve values",
              is_solution_correct(presolver->sol->x, correct_x, presolver->sol->y,
                                  correct_y, presolver->sol->z, correct_z, n_rows,
                                  n_cols, QP_TOL));
    PresolvedProblem orig =
        problem_from_csr_qp(Ax, Ai, Ap, n_rows, n_cols, Ap[n_rows], lhs, rhs, lbs,
                            ubs, c, Px, Pi, Pp, Pp[n_cols]);
    mu_assert("KKT", is_kkt_point(&orig, presolver->sol->x, presolver->sol->y,
                                  presolver->sol->z, QP_TOL));

    PS_FREE(stgs);
    free_presolver(presolver);
    return 0;
}

/*  Two coupled variables fixed by singleton equality rows, in both row
    orders. The multiplier of the row fixed first needs the neighbour that is
    recovered by the later FIXED_COL record.
    min. (1/2) x'Px + [1 -1 -4] x,  P = [2 1 0; 1 2 1; 0 1 2]
    s.t. x1 = 3, x0 = 2, x1 + x2 <= 10, 0 <= x <= 10
    Reduced (x2): min x2^2 - x2 s.t. x2 <= 7 (the last row becomes a bound),
    offset 18, solution x2 = 0.5.
    Original: x = (2, 3, 0.5), z = 0, y = (7.5, 8, 0) for the row order
    above and (8, 7.5, 0) with the two equality rows swapped. */
static char *test_15_qp()
{
    for (int swap = 0; swap < 2; ++swap)
    {
        double Ax[] = {1, 1, 1, 1};
        int Ai[] = {1, 0, 1, 2};
        int Ap[] = {0, 1, 2, 4};
        int n_rows = 3, n_cols = 3;
        double lhs[] = {3, 2, -INF};
        double rhs[] = {3, 2, 10};
        double correct_y[] = {7.5, 8, 0};
        if (swap)
        {
            Ai[0] = 0;
            Ai[1] = 1;
            lhs[0] = rhs[0] = 2;
            lhs[1] = rhs[1] = 3;
            correct_y[0] = 8;
            correct_y[1] = 7.5;
        }
        double lbs[] = {0, 0, 0};
        double ubs[] = {10, 10, 10};
        double c[] = {1, -1, -4};
        double Px[] = {2, 1, 1, 2, 1, 1, 2};
        int Pi[] = {0, 1, 0, 1, 2, 1, 2};
        int Pp[] = {0, 2, 5, 7};

        Settings *stgs = default_settings();
        set_settings_false(stgs);
        Presolver *presolver = new_presolver(Ax, Ai, Ap, n_rows, n_cols, 4, lhs, rhs,
                                             lbs, ubs, c, Px, Pi, Pp, 7, stgs);
        mu_assert("init", presolver != NULL);
        mu_assert("status", run_presolver(presolver) == REDUCED);

        PresolvedProblem *red = presolver->reduced_prob;
        mu_assert("reduced dims", red->m == 0 && red->n == 1 && red->Pnnz == 1);
        mu_assert("reduced c", red->c[0] == -1.0 && red->obj_offset == 18.0 &&
                                   red->ubs[0] == 7.0);

        double x[] = {0.5};
        double y[] = {0};
        double z[] = {0};
        postsolve(presolver, x, y, z);

        double correct_x[] = {2, 3, 0.5};
        double correct_z[] = {0, 0, 0};
        mu_assert("postsolve values",
                  is_solution_correct(
                      presolver->sol->x, correct_x, presolver->sol->y, correct_y,
                      presolver->sol->z, correct_z, n_rows, n_cols, QP_TOL));
        PresolvedProblem orig =
            problem_from_csr_qp(Ax, Ai, Ap, n_rows, n_cols, Ap[n_rows], lhs, rhs,
                                lbs, ubs, c, Px, Pi, Pp, Pp[n_cols]);
        mu_assert("KKT", is_kkt_point(&orig, presolver->sol->x, presolver->sol->y,
                                      presolver->sol->z, QP_TOL));

        PS_FREE(stgs);
        free_presolver(presolver);
    }
    return 0;
}

/*  Primal infeasibility ray after a variable in P is fixed by a singleton
    equality row.
    min. (1/2) x'Px + [1 -1 -4] x,  P = [2 1 0; 1 2 0; 0 0 0]
    s.t. x0 = 2, x1 + x2 >= 5, x1 + x2 <= 4, 0 <= x <= 10
    The reduced rows contradict each other; the reduced certificate (-1, 1)
    must come back as (0, -1, 1). */
static char *test_16_qp()
{
    double Ax[] = {1, 1, 1, 1, 1};
    int Ai[] = {0, 1, 2, 1, 2};
    int Ap[] = {0, 1, 3, 5};
    int n_rows = 3, n_cols = 3;
    double lhs[] = {2, 5, -INF};
    double rhs[] = {2, INF, 4};
    double lbs[] = {0, 0, 0};
    double ubs[] = {10, 10, 10};
    double c[] = {1, -1, -4};
    double Px[] = {2, 1, 1, 2};
    int Pi[] = {0, 1, 0, 1};
    int Pp[] = {0, 2, 4, 4};

    Settings *stgs = default_settings();
    set_settings_false(stgs);
    Presolver *presolver = new_presolver(Ax, Ai, Ap, n_rows, n_cols, 5, lhs, rhs,
                                         lbs, ubs, c, Px, Pi, Pp, 4, stgs);
    mu_assert("init", presolver != NULL);
    mu_assert("status", run_presolver(presolver) == REDUCED);

    PresolvedProblem *red = presolver->reduced_prob;
    mu_assert("reduced dims", red->m == 2 && red->n == 2);

    double y_red[] = {-1, 1};
    mu_assert("reduced certificate", is_primal_ray_certificate(red, y_red, QP_TOL));

    double y[3];
    postsolve_primal_infeas_ray(presolver, y_red, y);
    double correct_y[] = {0, -1, 1};
    mu_assert("ray values", ABS(y[0] - correct_y[0]) <= QP_TOL &&
                                ABS(y[1] - correct_y[1]) <= QP_TOL &&
                                ABS(y[2] - correct_y[2]) <= QP_TOL);
    PresolvedProblem orig =
        problem_from_csr_qp(Ax, Ai, Ap, n_rows, n_cols, Ap[n_rows], lhs, rhs, lbs,
                            ubs, c, Px, Pi, Pp, Pp[n_cols]);
    mu_assert("ray certificate", is_primal_ray_certificate(&orig, y, QP_TOL));

    PS_FREE(stgs);
    free_presolver(presolver);
    return 0;
}

static const char *all_tests_qp()
{
    mu_run_test(test_1_qp, counter_qp);
    mu_run_test(test_2_qp, counter_qp);
    mu_run_test(test_3_qp, counter_qp);
    mu_run_test(test_4_qp, counter_qp);
    mu_run_test(test_5_qp, counter_qp);
    mu_run_test(test_7_qp, counter_qp);
    mu_run_test(test_8_qp, counter_qp);
    mu_run_test(test_9_qp, counter_qp);
    mu_run_test(test_15_qp, counter_qp);
    mu_run_test(test_16_qp, counter_qp);
    return 0;
}

int test_qp()
{
    const char *result = all_tests_qp();
    if (result != 0)
    {
        printf("%s\n", result);
        printf("QP: TEST FAILED!\n");
    }
    else
    {
        printf("QP: ALL TESTS PASSED\n");
    }
    printf("QP: Tests run: %d\n", counter_qp);
    return result == 0;
}

#endif // TEST_QP_H
