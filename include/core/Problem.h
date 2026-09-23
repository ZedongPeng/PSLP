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

#ifndef CORE_PROBLEM_H
#define CORE_PROBLEM_H

#include <stdbool.h>
#include <stddef.h>
struct Constraints;
struct Matrix;

/* Objective (1/2) x'Px + c'x + offset. P is the full symmetric matrix in
   CSR without extra row space and is never modified during presolve (see
   col_in_P); problem_clean drops the rows of removed variables. For an LP,
   P has no entries. */
typedef struct
{
    double *c;
    double offset;
    struct Matrix *P;
} Objective;

typedef struct Problem
{
    struct Constraints *constraints;
    Objective *obj;
} Problem;

/* Constructor and destructor */
Problem *new_problem(struct Constraints *constraints, Objective *obj);
void free_problem(Problem *problem);

/* Cleans the problem in the sense that all data associated with
   inactive rows and inactive columns is removed. Rows and columns
   are also re-indexed to take the removal of rows/columns into
   account. */
void problem_clean(Problem *problem);

/* Constructor and destructor. The objective takes ownership of c and P. */
Objective *objective_new(double *c, struct Matrix *P);
void free_objective(Objective *obj);

/* Adds c[col] * value to the offset when variable 'col' is fixed to 'value'
   and marks row 'col' of P (which must be empty) as inactive. */
void fix_var_in_obj(Problem *prob, int col, double value);

/* Substitutes variable 'k' from the objective using row 'i'
   (assumed to be an equality). Row 'i' is specified by
   (vals, cols, len, rhs). Row k of P must be empty and is marked inactive. */
void sub_var_in_obj(Problem *prob, const double *vals, const int *cols, int len,
                    int k, double aik, double rhs);

/* Marks row k of P (which must be empty) as removed. */
void mark_P_row_inactive(Problem *prob, int k);

/* Substitutes variable 'subst' from the objective using the doubleton
   equality row aij * x_stay + aik * x_subst = rhs. Row subst of P must be
   empty and is marked inactive. c[subst] is left unchanged; the caller
   stores it for the postsolve. */
void sub_var_in_obj_dton(Problem *prob, int stay, int subst, double aik, double aij,
                         double rhs);

#endif // CORE_PROBLEM_H
