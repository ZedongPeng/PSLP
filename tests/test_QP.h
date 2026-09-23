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

/* Tests for the quadratic objective term (1/2) x'Px. The presolver never
   modifies P: a variable that appears in P is never fixed, substituted or
   dual-reduced, so it stays in the reduced problem.
   Test 1: structural validation of P in new_presolver
   Test 4: empty columns: fixed by the sign of c when absent from P, kept
           when present in P
   Test 5: simple dual fix skips a variable that appears in P
   Test 7: parallel columns are not merged when one of them appears in P
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

static const char *all_tests_qp()
{
    mu_run_test(test_1_qp, counter_qp);
    mu_run_test(test_4_qp, counter_qp);
    mu_run_test(test_5_qp, counter_qp);
    mu_run_test(test_7_qp, counter_qp);
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
