#ifndef TEST_DOMAINPROPAGATION_H
#define TEST_DOMAINPROPAGATION_H

#include "CoreTransformations.h"
#include "Numerics.h"
#include "Primal_propagation.h"
#include "SimpleReductions.h"
#include "Tags.h"
#include "minunit.h"
#include <stdio.h>

static int counter_domain = 0;

// clang-format off
/*
The following tests are/should be implemented for the DomainPropagation class
(a "✓" indicates that the test has been implemented).
---- Tests based on forcing rows ----
Test 1: Lower constraint that is forcing wrt the initial bounds (✓) 
        (integer arithmetic). 
Test 2: Lower constraint that is forcing wrt the initial bounds (✓)
        (floating point number arithmetic).
Test 3: Upper constraint that is forcing wrt the initial bounds (✓) 
        (integer arithmetic). 
Test 4: Upper constraint that is forcing wrt the initial bounds (✓)
        (floating point number arithmetic).
Test 5: Constraint 1 tightens an infinite upper bound so constraint 2 becomes (✓)
         lower forcing (integer arithmetic and NOT all bounds are tightened).
Test 6: Constraint 1 tightens an infinite lower bound so constraint 2 becomes (✓)
         upper forcing (integer arithmetic and NOT all bounds are tightened).
Test 7: Test 5 two rows swapped (✓)
Test 8: Test 6 two rows swapped (✓)
*/
// clang-format on

/*  Lower constraint that is forcing wrt the initial bounds
        (integer arithmetic).
    min. [-1 -1 -1 3 2]x
    s.t. [ 1  0   1   1   3]   >= 4
         [-1 -2   2   0   0] x >= 2
         [ 1  4   0   3  -1]   <= 5
         [ 2  0   0  -2   2]   >= 6
         [ 0  0   0   2  -3]   >= 0
         x1, x2 >= 0
         x3 <= 1
         x4, x5 >= 0
*/
static char *test_1_domain()
{
    double Ax[] = {1, 1, 1, 3, -1, -2, 2, 1, 4, 3, -1, 2, -2, 2, 2, -3};
    int Ai[] = {0, 2, 3, 4, 0, 1, 2, 0, 1, 3, 4, 0, 3, 4, 3, 4};
    int Ap[] = {0, 4, 7, 11, 14, 16};
    int nnz = 16;
    int n_rows = 5;
    int n_cols = 5;

    double lhs[] = {4, 2, -INF, 6, 0};
    double rhs[] = {INF, INF, 5, INF, INF};
    double lbs[5] = {0, 0, -INF, 0, 0};
    double ubs[] = {INF, INF, 1, INF, INF};
    double c[] = {-1, -1, -1, 3, 2};

    Settings *stgs = default_settings();
    Presolver *presolver =
        new_presolver(Ax, Ai, Ap, n_rows, n_cols, nnz, lhs, rhs, lbs, ubs, c, stgs);

    Problem *prob = presolver->prob;
    Constraints *constraints = prob->constraints;
    Matrix *A = constraints->A;
    PresolveStatus status = propagate_primal(prob, true);
    mu_assert("error status", status != INFEASIBLE);
    problem_clean(prob, true);

    // check that new A is correct
    double Ax_correct[] = {1, 3, 3, -1, -2, 2, 2, -3};
    int Ai_correct[] = {0, 1, 0, 1, 0, 1, 0, 1};
    int Ap_correct[] = {0, 2, 4, 6, 8};
    mu_assert("error Ax", ARRAYS_EQUAL_DOUBLE(Ax_correct, A->x, 8));
    mu_assert("error Ai", ARRAYS_EQUAL_INT(Ai_correct, A->i, 8));
    mu_assert("rows", check_row_starts(A, Ap_correct));

    // check that new variable bounds are correct
    double lbs_correct[] = {0, 0};
    double ubs_correct[] = {INF, INF};
    mu_assert("error bounds",
              check_bounds(constraints->bounds, lbs_correct, ubs_correct, 2));

    // check that lhs and rhs are correct
    double lhs_correct[] = {3, -INF, 6, 0};
    double rhs_correct[] = {INF, 5, INF, INF};
    mu_assert("error lhs", ARRAYS_EQUAL_DOUBLE(lhs_correct, constraints->lhs, 4));
    mu_assert("error rhs", ARRAYS_EQUAL_DOUBLE(rhs_correct, constraints->rhs, 4));

    PS_FREE(stgs);
    DEBUG(run_debugger(constraints, false));
    free_presolver(presolver);
    return 0;
}

/*  Lower constraint that is forcing wrt the initial bounds
        (floating point number arithmetic). The reduction is very
        sensitive to the size of the perturbation of the second constraint.
    min. [-1 -1 -1 3 2]x
    s.t. [ 1      0      1     1   3]   >= 4
         [-2/13  -1/17   7/3   0   0] x >= -2/(13*3) - 1/(17*7) + 7/(3*9) -
   1e-10 [ 1      4      0     3  -1]   <= 5 [ 2      0      0    -2   2]   >= 6
         x1 >= 1/3
         x2 >= 1/7
         x3 <= 1/9
         x4, x5 >= 0
*/
static char *test_2_domain()
{
    double Ax[] = {1, 1, 1, 3, -2.0 / 13, -1.0 / 17, 7.0 / 3, 1, 4, 3, -1, 2, -2, 2};
    int Ai[] = {0, 2, 3, 4, 0, 1, 2, 0, 1, 3, 4, 0, 3, 4};
    int Ap[] = {0, 4, 7, 11, 14};
    int nnz = 14;
    int n_rows = 4;
    int n_cols = 5;

    double lhs[] = {4,
                    -2.0 / (13 * 3) - 1.0 / (17 * 7) + 7.0 / (3 * 9) -
                        1e-10,
                    -INF, 6};
    double rhs[] = {INF, INF, 5, INF};
    double lbs[5] = {1.0 / 3, 1.0 / 7, -INF, 0, 0};
    double ubs[] = {INF, INF, 1.0 / 9, INF, INF};
    double c[] = {-1, -1, -1, 3, 2};

    Settings *stgs = default_settings();
    Presolver *presolver =
        new_presolver(Ax, Ai, Ap, n_rows, n_cols, nnz, lhs, rhs, lbs, ubs, c, stgs);

    Problem *prob = presolver->prob;
    Constraints *constraints = prob->constraints;
    PresolveStatus status = propagate_primal(prob, true);
    mu_assert("error status", status != INFEASIBLE);

    // Nonzero slack must not fix variables.
    mu_assert("x1 was approximately fixed",
              !HAS_TAG(constraints->col_tags[0], C_TAG_INACTIVE) &&
                  constraints->bounds[0].lb == 1.0 / 3 &&
                  constraints->bounds[0].ub > 1.0 / 3);
    mu_assert("x2 was approximately fixed",
              !HAS_TAG(constraints->col_tags[1], C_TAG_INACTIVE) &&
                  constraints->bounds[1].lb == 1.0 / 7 &&
                  constraints->bounds[1].ub > 1.0 / 7);
    mu_assert("x3 was approximately fixed",
              !HAS_TAG(constraints->col_tags[2], C_TAG_INACTIVE) &&
                  constraints->bounds[2].lb < 1.0 / 9 &&
                  constraints->bounds[2].ub == 1.0 / 9);
    mu_assert("error offset", prob->obj->offset == 0);

    PS_FREE(stgs);
    DEBUG(run_debugger(constraints, false));
    free_presolver(presolver);
    return 0;
}

/*  Upper constraint that is forcing wrt the initial bounds
        (integer arithmetic).
    min. [1 1 -1 3 2]x
    s.t. [ 1  0   1   1   3]   >= 4
         [-1 -2   2   0   0] x <= 2
         [ 1  4   0   3  -1]   <= 5
         [ 2  0   0  -2   2]   >= 6
         x1, x2 <= 0
         x3 >= 1
         x4, x5 >= 0
*/
static char *test_3_domain()
{
    double Ax[] = {1, 1, 1, 3, -1, -2, 2, 1, 4, 3, -1, 2, -2, 2};
    int Ai[] = {0, 2, 3, 4, 0, 1, 2, 0, 1, 3, 4, 0, 3, 4};
    int Ap[] = {0, 4, 7, 11, 14};
    int nnz = 14;
    int n_rows = 4;
    int n_cols = 5;

    double lhs[] = {4, -INF, -INF, 6};
    double rhs[] = {INF, 2, 5, INF};
    double lbs[5] = {-INF, -INF, 1, 0, 0};
    double ubs[] = {0, 0, INF, INF, INF};
    double c[] = {1, -1, -1, 3, 2};

    Settings *stgs = default_settings();
    Presolver *presolver =
        new_presolver(Ax, Ai, Ap, n_rows, n_cols, nnz, lhs, rhs, lbs, ubs, c, stgs);

    Problem *prob = presolver->prob;
    Constraints *constraints = prob->constraints;
    Matrix *A = constraints->A;
    PresolveStatus status = propagate_primal(prob, true);
    mu_assert("error status", status != INFEASIBLE);
    problem_clean(prob, true);

    // check that new A is correct
    double Ax_correct[] = {1, 3, 3, -1, -2, 2};
    int Ai_correct[] = {0, 1, 0, 1, 0, 1};
    int Ap_correct[] = {0, 2, 4, 6};
    mu_assert("error Ax", ARRAYS_EQUAL_DOUBLE(Ax_correct, A->x, 6));
    mu_assert("error Ai", ARRAYS_EQUAL_INT(Ai_correct, A->i, 6));
    mu_assert("rows", check_row_starts(A, Ap_correct));

    // check that new variable bounds are correct
    double lbs_correct[] = {0, 0};
    double ubs_correct[] = {INF, INF};
    mu_assert("error bounds",
              check_bounds(constraints->bounds, lbs_correct, ubs_correct, 2));

    // check that objective offset is correct
    mu_assert("error offset", prob->obj->offset == -1);

    // check that lhs and rhs are correct
    double lhs_correct[] = {3, -INF, 6};
    double rhs_correct[] = {INF, 5, INF};
    mu_assert("error lhs", ARRAYS_EQUAL_DOUBLE(lhs_correct, constraints->lhs, 3));
    mu_assert("error rhs", ARRAYS_EQUAL_DOUBLE(rhs_correct, constraints->rhs, 3));

    PS_FREE(stgs);
    DEBUG(run_debugger(constraints, false));
    free_presolver(presolver);
    return 0;
}

/*  Upper constraint that is forcing wrt the initial bounds
        (floating point number arithmetic). The reduction is very
        sensitive to the size of the perturbation of the second constraint.
    min. [1 1 -1 3 2]x
    s.t. [ 1      0      1     1   3]   >= 4
         [-2/13  -1/17   7/3   0   0] x <= -2/(13*3) - 1/(17*7) + 7/(3*9) +
   1e-10 [ 1      4      0     3  -1]   <= 5 [ 2      0      0    -2   2]   >= 6
         x1 <= 1/3
         x2 <= 1/7
         x3 >= 1/9
         x4, x5 >= 0
*/
static char *test_4_domain()
{
    double Ax[] = {1, 1, 1, 3, -2.0 / 13, -1.0 / 17, 7.0 / 3, 1, 4, 3, -1, 2, -2, 2};
    int Ai[] = {0, 2, 3, 4, 0, 1, 2, 0, 1, 3, 4, 0, 3, 4};
    int Ap[] = {0, 4, 7, 11, 14};
    int nnz = 14;
    int n_rows = 4;
    int n_cols = 5;

    double lhs[] = {4, -INF, -INF, 6};
    double rhs[] = {INF,
                    -2.0 / (13 * 3) - 1.0 / (17 * 7) + 7.0 / (3 * 9) +
                        1e-10,
                    5, INF};
    double lbs[5] = {-INF, -INF, 1.0 / 9, 0, 0};
    double ubs[] = {1.0 / 3, 1.0 / 7, INF, INF, INF};
    double c[] = {1, -1, -1, 3, 2};

    Settings *stgs = default_settings();
    Presolver *presolver =
        new_presolver(Ax, Ai, Ap, n_rows, n_cols, nnz, lhs, rhs, lbs, ubs, c, stgs);

    Problem *prob = presolver->prob;
    Constraints *constraints = prob->constraints;
    PresolveStatus status = propagate_primal(prob, true);
    mu_assert("error status", status != INFEASIBLE);

    // Symmetric nonzero-slack case.
    mu_assert("x1 was approximately fixed",
              !HAS_TAG(constraints->col_tags[0], C_TAG_INACTIVE) &&
                  constraints->bounds[0].lb < 1.0 / 3 &&
                  constraints->bounds[0].ub == 1.0 / 3);
    mu_assert("x2 was approximately fixed",
              !HAS_TAG(constraints->col_tags[1], C_TAG_INACTIVE) &&
                  constraints->bounds[1].lb < 1.0 / 7 &&
                  constraints->bounds[1].ub == 1.0 / 7);
    mu_assert("x3 was approximately fixed",
              !HAS_TAG(constraints->col_tags[2], C_TAG_INACTIVE) &&
                  constraints->bounds[2].lb == 1.0 / 9 &&
                  constraints->bounds[2].ub > 1.0 / 9);
    mu_assert("error offset", prob->obj->offset == 0);

    PS_FREE(stgs);
    DEBUG(run_debugger(constraints, false));
    free_presolver(presolver);
    return 0;
}

/*  Constraint 1 tightens an infinite upper bound so constraint 2 becomes
        lower forcing (integer arithmetic and NOT all bounds are tightened).

    min. [1 1 -1 -3 2]x
    s.t. [ 1  0    1   0   0]   <= 4
         [ 2  1    0   0   0] x >= 10
         [ 1  1   -1   3  -1]   <= 7
         [ 2  0    0  -2   2]   >= 6
         x1, x2, x3, x4, x5 >= 0
         x2 <= 2
*/
static char *test_5_domain()
{
    double Ax[] = {1, 1, 2, 1, 1, 1, -1, 3, -1, 2, -2, 2};
    int Ai[] = {0, 2, 0, 1, 0, 1, 2, 3, 4, 0, 3, 4};
    int Ap[] = {0, 2, 4, 9, 12};
    int nnz = 12;
    int n_rows = 4;
    int n_cols = 5;

    double lhs[] = {-INF, 10, -INF, 6};
    double rhs[] = {4, INF, 7, INF};
    double lbs[5] = {0, 0, 0, 0, 0};
    double ubs[] = {INF, 2, INF, INF, INF};
    double c[] = {1, 1, -1, -3, 2};

    Settings *stgs = default_settings();
    Presolver *presolver =
        new_presolver(Ax, Ai, Ap, n_rows, n_cols, nnz, lhs, rhs, lbs, ubs, c, stgs);

    Problem *prob = presolver->prob;
    Constraints *constraints = prob->constraints;
    PresolveStatus status = propagate_primal(prob, true);
    mu_assert("error status", status != INFEASIBLE);

    // Integer propagation must keep a nonempty interval.
    mu_assert("integer cascade fixed x1",
              !HAS_TAG(constraints->col_tags[0], C_TAG_INACTIVE) &&
                  constraints->bounds[0].lb < 4 &&
                  constraints->bounds[0].ub > 4);
    mu_assert("integer cascade fixed x2",
              !HAS_TAG(constraints->col_tags[1], C_TAG_INACTIVE) &&
                  constraints->bounds[1].lb < 2 &&
                  constraints->bounds[1].ub == 2);
    mu_assert("error offset", prob->obj->offset == 0);

    PS_FREE(stgs);
    DEBUG(run_debugger(constraints, false));
    free_presolver(presolver);
    return 0;
}

/*  Constraint 1 tightens an infinite lower bound so constraint 2 becomes
        upper forcing (integer arithmetic and NOT all bounds are tightened).

    min. [1 1 -1 -3 2]x
    s.t. [ 1  0   -1   0   0]   >= 4
         [ 2  1    0   0   0] x <= 10
         [ 1  1   -1   3  -1]   <= 7
         [ 2  0    0  -2   2]   >= 6
         x2, x4, x5 >= 0
         x3 >= 1
         x1 <= 100
*/
static char *test_6_domain()
{
    double Ax[] = {1, -1, 2, 1, 1, 1, -1, 3, -1, 2, -2, 2};
    int Ai[] = {0, 2, 0, 1, 0, 1, 2, 3, 4, 0, 3, 4};
    int Ap[] = {0, 2, 4, 9, 12};
    int nnz = 12;
    int n_rows = 4;
    int n_cols = 5;

    double lhs[] = {4, -INF, -INF, 6};
    double rhs[] = {INF, 10, 7, INF};
    double lbs[5] = {-INF, 0, 1, 0, 0};
    double ubs[] = {100, INF, INF, INF, INF};
    double c[] = {1, 1, -1, -3, 2};

    Settings *stgs = default_settings();
    Presolver *presolver =
        new_presolver(Ax, Ai, Ap, n_rows, n_cols, nnz, lhs, rhs, lbs, ubs, c, stgs);

    Problem *prob = presolver->prob;
    Constraints *constraints = prob->constraints;
    PresolveStatus status = propagate_primal(prob, true);
    mu_assert("error status", status != INFEASIBLE);

    // Symmetric integer-propagation case.
    mu_assert("integer cascade fixed x1",
              !HAS_TAG(constraints->col_tags[0], C_TAG_INACTIVE) &&
                  constraints->bounds[0].lb < 5 &&
                  constraints->bounds[0].ub > 5);
    mu_assert("integer cascade fixed x2",
              !HAS_TAG(constraints->col_tags[1], C_TAG_INACTIVE) &&
                  constraints->bounds[1].lb == 0 &&
                  constraints->bounds[1].ub > 0);
    mu_assert("error offset", prob->obj->offset == 0);

    PS_FREE(stgs);
    DEBUG(run_debugger(constraints, false));
    free_presolver(presolver);
    return 0;
}

/*  Test 5 with two rows swapped.

    min. [1 1 -1 -3 2]x
    s.t. [ 2  1    0   0   0] x >= 10
         [ 1  0    1   0   0]   <= 4
         [ 1  1   -1   3  -1]   <= 7
         [ 2  0    0  -2   2]   >= 6
         x1, x2, x3, x4, x5 >= 0
         x2 <= 2
*/
static char *test_7_domain()
{
    double Ax[] = {2, 1, 1, 1, 1, 1, -1, 3, -1, 2, -2, 2};
    int Ai[] = {0, 1, 0, 2, 0, 1, 2, 3, 4, 0, 3, 4};
    int Ap[] = {0, 2, 4, 9, 12};
    int nnz = 12;
    int n_rows = 4;
    int n_cols = 5;

    double lhs[] = {10, -INF, -INF, 6};
    double rhs[] = {INF, 4, 7, INF};
    double lbs[5] = {0, 0, 0, 0, 0};
    double ubs[] = {INF, 2, INF, INF, INF};
    double c[] = {1, 1, -1, -3, 2};

    Settings *stgs = default_settings();
    Presolver *presolver =
        new_presolver(Ax, Ai, Ap, n_rows, n_cols, nnz, lhs, rhs, lbs, ubs, c, stgs);

    Problem *prob = presolver->prob;
    Constraints *constraints = prob->constraints;
    PresolveStatus status = propagate_primal(prob, true);
    mu_assert("error status", status != INFEASIBLE);

    // Check row-order independence.
    mu_assert("row order fixed x1",
              !HAS_TAG(constraints->col_tags[0], C_TAG_INACTIVE) &&
                  constraints->bounds[0].lb < 4 &&
                  constraints->bounds[0].ub > 4);
    mu_assert("row order fixed x2",
              !HAS_TAG(constraints->col_tags[1], C_TAG_INACTIVE) &&
                  constraints->bounds[1].lb < 2 &&
                  constraints->bounds[1].ub == 2);
    mu_assert("error offset", prob->obj->offset == 0);

    PS_FREE(stgs);
    DEBUG(run_debugger(constraints, false));
    free_presolver(presolver);
    return 0;
}

/*  Test 6 two rows swapped

    min. [1 1 -1 -3 2]x
    s.t. [ 2  1    0   0   0] x <= 10
         [ 1  0   -1   0   0]   >= 4
         [ 1  1   -1   3  -1]   <= 7
         [ 2  0    0  -2   2]   >= 6
         x2, x4, x5 >= 0
         x3 >= 1
         x1 <= 100
*/
static char *test_8_domain()
{
    double Ax[] = {2, 1, 1, -1, 1, 1, -1, 3, -1, 2, -2, 2};
    int Ai[] = {0, 1, 0, 2, 0, 1, 2, 3, 4, 0, 3, 4};
    int Ap[] = {0, 2, 4, 9, 12};
    int nnz = 12;
    int n_rows = 4;
    int n_cols = 5;

    double lhs[] = {-INF, 4, -INF, 6};
    double rhs[] = {10, INF, 7, INF};
    double lbs[5] = {-INF, 0, 1, 0, 0};
    double ubs[] = {100, INF, INF, INF, INF};
    double c[] = {1, 1, -1, -3, 2};

    Settings *stgs = default_settings();
    Presolver *presolver =
        new_presolver(Ax, Ai, Ap, n_rows, n_cols, nnz, lhs, rhs, lbs, ubs, c, stgs);

    Problem *prob = presolver->prob;
    Constraints *constraints = prob->constraints;
    PresolveStatus status = propagate_primal(prob, true);
    mu_assert("error status", status != INFEASIBLE);

    mu_assert("row order fixed x1",
              !HAS_TAG(constraints->col_tags[0], C_TAG_INACTIVE) &&
                  constraints->bounds[0].lb < 5 &&
                  constraints->bounds[0].ub > 5);
    mu_assert("row order fixed x2",
              !HAS_TAG(constraints->col_tags[1], C_TAG_INACTIVE) &&
                  constraints->bounds[1].lb == 0 &&
                  constraints->bounds[1].ub > 0);
    mu_assert("error offset", prob->obj->offset == 0);

    PS_FREE(stgs);
    DEBUG(run_debugger(constraints, false));
    free_presolver(presolver);
    return 0;
}

/* Check the safety margin on integer bounds. */
static char *test_9_domain_integer()
{
    double Ax[] = {2, 3, 1, 8};
    int Ai[] = {0, 1, 0, 1};
    int Ap[] = {0, 2, 4};
    int nnz = 4;
    int n_rows = 2;
    int n_cols = 2;

    double lhs[] = {-INF, -INF};
    double rhs[] = {8, 8};
    double lbs[5] = {0, 0};
    double ubs[] = {INF, INF};
    double c[] = {1, 1};

    Settings *stgs = default_settings();
    Presolver *presolver =
        new_presolver(Ax, Ai, Ap, n_rows, n_cols, nnz, lhs, rhs, lbs, ubs, c, stgs);

    Problem *prob = presolver->prob;
    Constraints *constraints = prob->constraints;
    PresolveStatus status = propagate_primal(prob, true);
    mu_assert("error status", status != INFEASIBLE);
    problem_clean(prob, true);

    double ub0 = 4 + BOUND_MARGINAL * 4;
    double ub1 = 1 + BOUND_MARGINAL;
    mu_assert("integer upper bound was not relaxed",
              ABS(constraints->bounds[0].ub - ub0) <= 1e-12);
    mu_assert("integer upper bound was not relaxed",
              ABS(constraints->bounds[1].ub - ub1) <= 1e-12);
    mu_assert("error bound", !HAS_TAG(constraints->col_tags[0], C_TAG_UB_INF));
    mu_assert("error bound", !HAS_TAG(constraints->col_tags[1], C_TAG_UB_INF));

    remove_redundant_bounds(constraints);

    mu_assert("error bound", HAS_TAG(constraints->col_tags[0], C_TAG_UB_INF));
    mu_assert("error bound", HAS_TAG(constraints->col_tags[1], C_TAG_UB_INF));

    PS_FREE(stgs);
    DEBUG(run_debugger(constraints, false));
    free_presolver(presolver);
    return 0;
}

/* Check the safety margin on decimal bounds. */
static char *test_9_domain_decimal()
{
    double Ax[] = {3, 3, 1, 7};
    int Ai[] = {0, 1, 0, 1};
    int Ap[] = {0, 2, 4};
    int nnz = 4;
    int n_rows = 2;
    int n_cols = 2;

    double lhs[] = {-INF, -INF};
    double rhs[] = {8, 8};
    double lbs[5] = {0, 0};
    double ubs[] = {INF, INF};
    double c[] = {1, 1};

    Settings *stgs = default_settings();
    Presolver *presolver =
        new_presolver(Ax, Ai, Ap, n_rows, n_cols, nnz, lhs, rhs, lbs, ubs, c, stgs);

    Problem *prob = presolver->prob;
    Constraints *constraints = prob->constraints;
    PresolveStatus status = propagate_primal(prob, true);
    mu_assert("error status", status != INFEASIBLE);
    problem_clean(prob, true);

    double raw_ub0 = 8.0 / 3;
    double raw_ub1 = 8.0 / 7;
    mu_assert("decimal upper bound was not relaxed",
              ABS(constraints->bounds[0].ub -
                  (raw_ub0 + BOUND_MARGINAL * raw_ub0)) <= 1e-12);
    mu_assert("decimal upper bound was not relaxed",
              ABS(constraints->bounds[1].ub -
                  (raw_ub1 + BOUND_MARGINAL * raw_ub1)) <= 1e-12);
    mu_assert("error bound", !HAS_TAG(constraints->col_tags[0], C_TAG_UB_INF));
    mu_assert("error bound", !HAS_TAG(constraints->col_tags[1], C_TAG_UB_INF));

    remove_redundant_bounds(constraints);

    mu_assert("error bound", HAS_TAG(constraints->col_tags[0], C_TAG_UB_INF));
    mu_assert("error bound", HAS_TAG(constraints->col_tags[1], C_TAG_UB_INF));

    PS_FREE(stgs);
    DEBUG(run_debugger(constraints, false));
    free_presolver(presolver);
    return 0;
}

static const char *all_tests_domain()
{
    mu_run_test(test_1_domain, counter_domain);
    mu_run_test(test_2_domain, counter_domain);
    mu_run_test(test_3_domain, counter_domain);
    mu_run_test(test_4_domain, counter_domain);
    mu_run_test(test_5_domain, counter_domain);
    mu_run_test(test_6_domain, counter_domain);
    mu_run_test(test_7_domain, counter_domain);
    mu_run_test(test_8_domain, counter_domain);
    mu_run_test(test_9_domain_integer, counter_domain);
    mu_run_test(test_9_domain_decimal, counter_domain);
    return 0;
}

int test_domain()
{
    const char *result = all_tests_domain();
    if (result != 0)
    {
        printf("%s\n", result);
        printf("domain: TEST FAILED!\n");
    }
    else
    {
        printf("domain: ALL TESTS PASSED\n");
    }
    printf("domain: Tests run: %d\n", counter_domain);
    return result == 0;
}

#endif
