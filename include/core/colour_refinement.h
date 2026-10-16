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

#ifndef COLOUR_REFINEMENT_H
#define COLOUR_REFINEMENT_H

/* Colour refinement (1-dimensional Weisfeiler-Leman) of the bipartite graph of
   a sparse matrix: a node per line (a row or a column) and an edge per entry,
   labelled with the bit pattern of its value (-0.0 read as 0.0). Starting from
   colours given by the caller, the rows and the columns are split into the
   coarsest equitable partition: two rows of a class have, for every column
   class, the same multiset of labels on their edges into it, and likewise for
   columns.

   A square matrix P on the columns (colour_refinement_new_qp) adds the pairs
   (class of the other column, value bits) of its entries to the signatures of
   the columns, kept apart from the pairs of A: two columns of a class then
   also have, for every column class, the same multiset of labels in P.

   The multisets are compared exactly, so the partition is equitable whenever
   the run converges. Every round recomputes and sorts the signature of every
   line, which costs about nnz log nnz.

   The numbering of the classes is unspecified; only the lowest member ids
   identify a class across implementations. */

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

struct Matrix;

typedef struct ColourRefinement ColourRefinement;

typedef enum ColourRefinementResult
{
    COLOUR_REFINEMENT_CONVERGED,    /* Stable partition. */
    COLOUR_REFINEMENT_NOT_CONVERGED /* Work budget or round limit reached. */
} ColourRefinementResult;

typedef struct ColourRefinementOptions
{
    /* Entries touched before giving up, in multiples of the entries of the
       problem (A, plus P when given); a round touches 2 nnz(A) + nnz(P). */
    double work_budget;
    int max_rounds; /* A round refines the rows, then the columns. */
} ColourRefinementOptions;

typedef struct ColourRefinementStats
{
    int rounds;
    double work; /* Entries touched, in multiples of the entries of the problem. */
    int n_row_classes;
    int n_col_classes;
    /* Sum over the classes of both sides of (size - 1) times the entries of a
       member line: the entries that duplicate those of another line of the
       class. */
    int duplicate_entries;
} ColourRefinementStats;

/* State and scratch for A and its transpose. Keeps pointers to both. NULL on
   failure; nothing after this allocates. */
ColourRefinement *colour_refinement_new(const struct Matrix *A,
                                        const struct Matrix *AT);

/* As colour_refinement_new, with the square matrix P (n_cols x n_cols) on the
   columns. P == NULL or an empty P gives the refinement of A alone. */
ColourRefinement *colour_refinement_new_qp(const struct Matrix *A,
                                           const struct Matrix *AT,
                                           const struct Matrix *P);
void colour_refinement_free(ColourRefinement *refine);

/* Refines from the start colours, one per row and one per column, to a
   fixpoint or a limit. Lines with equal colours and equally many entries start
   in one class. Writes the class id of every row and column, in
   [0, n_row_classes) and [0, n_col_classes), and the counts into 'stats'. The
   classes are valid whether or not the refinement converged. */
ColourRefinementResult colour_refinement_run(ColourRefinement *refine,
                                             const uint64_t *row_colours,
                                             const uint64_t *col_colours,
                                             const ColourRefinementOptions *options,
                                             int *row_classes, int *col_classes,
                                             ColourRefinementStats *stats);

/* True when the partition (row_classes, col_classes) is equitable: refining it
   once more splits no class. The ids may be any ints. */
bool colour_refinement_verify(ColourRefinement *refine, const int *row_classes,
                              const int *col_classes);

#endif // COLOUR_REFINEMENT_H
