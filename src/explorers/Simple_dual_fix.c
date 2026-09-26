/*
 * Copyright 2025-2026 Daniel Cederberg
 *
 * This file is part of the PSLP project (LP Presolver).
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "Simple_dual_fix.h"
#include "Bounds.h"
#include "Constraints.h"
#include "CoreTransformations.h"
#include "Debugger.h"
#include "Locks.h"
#include "Matrix.h"
#include "Numerics.h"
#include "Problem.h"
#include "SimpleReductions.h"
#include "State.h"
#include "Workspace.h"
#include "iVec.h"

// a column that becomes fixed because of dual fix should have zk = ck - ak^T y
static inline PresolveStatus _simple_dual_fix(Problem *prob, double ck, double lb,
                                              double ub, Lock lock, ColTag col_tag,
                                              int col, iVec *cols_to_inf)
{
    assert(!HAS_TAG(col_tag, C_TAG_INACTIVE));
    assert(!col_in_P(prob->constraints->state, col));

    // --------------------------------------------------
    // see if we can fix the variable to its lower bound
    // --------------------------------------------------
    if (ck > 0 && lock.down == 0)
    {
        if (HAS_TAG(col_tag, C_TAG_LB_INF))
        {
            return UNBNDORINFEAS;
        }

        assert(!IS_ABS_INF(lb));
        fix_col(prob, col, lb);
        return UNCHANGED;
    }

    // ---------------------------------------------------
    // see if we can fix the variable to its upper bound
    // ---------------------------------------------------
    if (ck < 0 && lock.up == 0)
    {
        if (ck != 0 && HAS_TAG(col_tag, C_TAG_UB_INF))
        {
            return UNBNDORINFEAS;
        }

        assert(!IS_ABS_INF(ub));
        fix_col(prob, col, ub);
        return UNCHANGED;
    }

    // ------------------------------------------------------------------------
    // If the objective coefficient is 0 and there are no downlocks,
    // then the variable may be set to -infinity if it is unbounded from below,
    // making the constraints it appears in redundant. If the variable is
    // bounded from below we can set the variable to any value. Our design
    // choice is to set the variable to its lower bound.
    // A similar comment applies to the case when there are no uplocks.
    // ------------------------------------------------------------------------
    if (ck == 0)
    {
        if (lock.down == 0)
        {
            if (HAS_TAG(col_tag, C_TAG_LB_INF))
            {
                iVec_append(cols_to_inf, -col);
            }
            else
            {
                assert(!IS_ABS_INF(lb));
                fix_col(prob, col, lb);
            }
            return UNCHANGED;
        }

        if (lock.up == 0)
        {
            if (HAS_TAG(col_tag, C_TAG_UB_INF))
            {
                iVec_append(cols_to_inf, col);
            }
            else
            {
                assert(!IS_ABS_INF(ub));
                fix_col(prob, col, ub);
            }
            return UNCHANGED;
        }
    }

    return UNCHANGED;
}

/* Tightens the upper (is_upper_bound) or lower bound of a variable to
   new_bound and updates the activities. No postsolve record: the caller has
   shown that no optimal solution has the variable beyond new_bound, so the
   new bound is never binding. */
static void tighten_bound(Problem *prob, int col, double new_bound,
                          bool is_upper_bound)
{
    Constraints *constraints = prob->constraints;
    State *data = constraints->state;
    Bound *bounds = constraints->bounds;
    ColTag *col_tag = constraints->col_tags + col;
    ColTag inf_tag = is_upper_bound ? C_TAG_UB_INF : C_TAG_LB_INF;
    bool is_inf = HAS_TAG(*col_tag, inf_tag);
    double old_bound = is_upper_bound ? bounds[col].ub : bounds[col].lb;
    assert(!HAS_TAG(*col_tag, C_TAG_INACTIVE));
    assert(!IS_ABS_INF(new_bound) && !IS_HUGE(new_bound));
    assert(is_inf ||
           (is_upper_bound ? new_bound < old_bound : new_bound > old_bound));
    assert(is_upper_bound
               ? (HAS_TAG(*col_tag, C_TAG_LB_INF) || new_bound >= bounds[col].lb)
               : (HAS_TAG(*col_tag, C_TAG_UB_INF) || new_bound <= bounds[col].ub));

    const Matrix *AT = constraints->AT;
    const double *vals = AT->x + AT->p[col].start;
    const int *rows = AT->i + AT->p[col].start;
    int len = AT->p[col].end - AT->p[col].start;

    if (is_upper_bound)
    {
        bounds[col].ub = new_bound;
    }
    else
    {
        bounds[col].lb = new_bound;
    }

    update_activities_bound_change(data->activities, constraints->A, bounds, vals,
                                   rows, len, old_bound, new_bound, !is_inf,
                                   !is_upper_bound,
                                   data->updated_activities HUGE_BOUND_IS_NOT_OK);
    REMOVE_TAG(*col_tag, inf_tag);
}

/* Dual fix for a variable whose row of P is the single diagonal Pkk > 0. The
   objective in xk is (1/2) Pkk xk^2 + ck xk with minimizer x* = -ck / Pkk.
   Without down-locks some optimal solution has xk <= max(lb, x*), without
   up-locks xk >= min(ub, x*): fix xk when the interval is a point, otherwise
   tighten the bound. */
static inline void diagonal_dual_fix(Problem *prob, int col, double Pkk)
{
    const Constraints *constraints = prob->constraints;
    const Bound *bounds = constraints->bounds;
    ColTag col_tag = constraints->col_tags[col];
    Lock lock = constraints->state->col_locks[col];
    bool lb_inf = HAS_TAG(col_tag, C_TAG_LB_INF);
    bool ub_inf = HAS_TAG(col_tag, C_TAG_UB_INF);
    double lb = bounds[col].lb;
    double ub = bounds[col].ub;
    double xstar = -prob->obj->c[col] / Pkk;
    assert(!HAS_TAG(col_tag, C_TAG_INACTIVE) && Pkk > 0);
    assert(constraints->state->col_sizes[col] > 0);

    if (IS_HUGE(xstar))
    {
        return;
    }

    // no locks at all: xk = clip(x*, lb, ub)
    if (lock.down == 0 && lock.up == 0)
    {
        double val = xstar;
        if (!lb_inf && val < lb)
        {
            val = lb;
        }
        else if (!ub_inf && val > ub)
        {
            val = ub;
        }

        fix_col(prob, col, val);
        return;
    }

    // no down-locks: xk <= max(lb, x*)
    if (lock.down == 0)
    {
        if (!lb_inf && xstar <= lb)
        {
            fix_col(prob, col, lb);
        }
        else if (ub_inf || xstar <= ub - FEAS_TOL)
        {
            tighten_bound(prob, col, xstar, true);
        }
        return;
    }

    // no up-locks: xk >= min(ub, x*)
    assert(lock.up == 0);
    if (!ub_inf && xstar >= ub)
    {
        fix_col(prob, col, ub);
    }
    else if (lb_inf || xstar >= lb + FEAS_TOL)
    {
        tighten_bound(prob, col, xstar, false);
    }
}

PresolveStatus simple_dual_fix(Problem *prob)
{
    assert(!has_removable_empty_cols(prob->constraints->state));
    DEBUG(verify_problem_up_to_date(prob->constraints));

    Constraints *constraints = prob->constraints;
    const double *c = prob->obj->c;
    const Bound *bounds = constraints->bounds;
    const ColTag *col_tags = constraints->col_tags;
    const Lock *locks = constraints->state->col_locks;
    const int *P_row_sizes = constraints->state->P_row_sizes;
    const int *col_sizes = constraints->state->col_sizes;
    const Matrix *P = prob->obj->P;
    size_t n_cols = constraints->n;
    iVec *cols_to_inf = constraints->state->work->int_vec;
    iVec_clear_no_resize(cols_to_inf);
    PresolveStatus status = UNCHANGED;

    for (size_t k = 0; k < n_cols; k++)
    {
        // if it is too slow to loop through all columns, we can keep a list
        // of candidate columns to process
        if (HAS_TAG(col_tags[k], C_TAG_INACTIVE) ||
            (locks[k].up > 0 && locks[k].down > 0))
        {
            continue;
        }

        // the sign of the objective gradient depends on x unless row k of P is
        // empty or a single diagonal; empty columns are left to remove_empty_cols
        if (col_in_P(constraints->state, (int) k))
        {
            double Pkk;
            if (diagonal_only_P_entry(P, P_row_sizes, (int) k, &Pkk) &&
                col_sizes[k] > 0)
            {
                diagonal_dual_fix(prob, (int) k, Pkk);
            }
            continue;
        }

        // if simple_dual_fix returns UNBNDORINFEAS it will be propagated
        // to run_fast_presolvers where it is detected
        status |= _simple_dual_fix(prob, c[k], bounds[k].lb, bounds[k].ub, locks[k],
                                   col_tags[k], (int) k, cols_to_inf);
    }

    // now fix the columns that can be fixed to inf
    // (processing the inf columns after processing the other columns
    // simplifies the dual postsolve a lot)
    for (size_t i = 0; i < cols_to_inf->len; ++i)
    {
        int col = cols_to_inf->data[i];

        if (col < 0)
        {
            fix_col_to_negative_inf(prob, -col);
        }
        else if (col > 0)
        {
            fix_col_to_positive_inf(prob, col);
        }
        else
        {
            assert(col == 0);
            assert(locks[col].up == 0 || locks[col].down == 0);
            if (locks[col].down == 0)
            {
                fix_col_to_negative_inf(prob, col);
            }
            else if (locks[col].up == 0)
            {
                fix_col_to_positive_inf(prob, col);
            }
        }
    }

    // after the simple dual fix we might have inactive rows, and fixed
    // columns
    delete_inactive_rows(prob->constraints);
    delete_fixed_cols_from_problem(prob);
    delete_inactive_cols_from_A_and_AT(prob->constraints);

    return status;
}
