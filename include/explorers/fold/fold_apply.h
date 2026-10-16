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

#ifndef FOLD_APPLY_H
#define FOLD_APPLY_H

#include "Bounds.h"
#include "Matrix.h"
#include "Tags.h"
#include "fold.h"

/* Builds the folded problem of 'maps' into a new matrix: the rows are the
   representatives with their coefficients summed over each column class, lhs,
   rhs and row tags those of the representatives, bounds and column tags those
   of the keepers, and c the keeper's cost times the class size, all compacted
   in place. The caller rebuilds the transpose. Returns FOLD_FOLDABLE with
   *folded set, or FOLD_FAILED when an allocation fails, in which case *folded
   is NULL and nothing is touched. */
FoldReason fold_apply(const Matrix *A, double *lhs, double *rhs, RowTag *row_tags,
                      Bound *bounds, ColTag *col_tags, double *c,
                      const FoldMaps *maps, Matrix **folded);

/* As fold_apply, with the quadratic term P (n_cols x n_cols, symmetric) folded
   into *P_folded: the entry of the classes C and D is the class size |C|
   times the entries of C's keeper over D, which the partition makes equal to
   the entry of D and C; the lower triangle is copied from the upper so that
   the result is symmetric bit for bit, and exact-zero sums are dropped. The
   rows of *P_folded have no extra space. *P_folded is NULL when P is NULL. On
   failure both outputs are NULL and nothing is touched. */
FoldReason fold_apply_qp(const Matrix *A, const Matrix *P, double *lhs, double *rhs,
                         RowTag *row_tags, Bound *bounds, ColTag *col_tags,
                         double *c, const FoldMaps *maps, Matrix **folded,
                         Matrix **P_folded);

#endif // FOLD_APPLY_H
