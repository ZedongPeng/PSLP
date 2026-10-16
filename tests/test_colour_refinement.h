#ifndef TEST_COLOUR_REFINEMENT_H
#define TEST_COLOUR_REFINEMENT_H

#include "Matrix.h"
#include "colour_refinement.h"
#include "minunit.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int counter_colour_refinement = 0;

/* A graph for the refinement: the matrix and its transpose, with the classes
   of the last run. */
typedef struct RefineGraph
{
    Matrix *A;
    Matrix *AT;
    Matrix *P; /* NULL without P */
    int *work;
    ColourRefinement *refine;
    int *row_classes;
    int *col_classes;
} RefineGraph;

static ColourRefinementOptions refine_test_options(void)
{
    ColourRefinementOptions options = {1e9, 16384};
    return options;
}

/* Builds the graph of (Ap, Ai, Ax) and runs the refinement from the given
   colours (NULL for all equal) with the given options (NULL for the
   defaults). */
static ColourRefinementResult
graph_refine(RefineGraph *graph, int m, int n, int nnz, const int *Ap, const int *Ai,
             const double *Ax, const uint64_t *row_colours,
             const uint64_t *col_colours, ColourRefinementStats *stats,
             const ColourRefinementOptions *options)
{
    memset(graph, 0, sizeof(*graph));
    graph->A = matrix_new(Ax, Ai, Ap, (size_t) m, (size_t) n, (size_t) nnz);
    graph->work = (int *) calloc((size_t) n + 1, sizeof(int));
    graph->AT = transpose(graph->A, graph->work, 0);
    graph->refine = colour_refinement_new(graph->A, graph->AT);
    graph->row_classes = (int *) calloc((size_t) m + 1, sizeof(int));
    graph->col_classes = (int *) calloc((size_t) n + 1, sizeof(int));
    uint64_t *zero_rows = (uint64_t *) calloc((size_t) m + 1, sizeof(uint64_t));
    uint64_t *zero_cols = (uint64_t *) calloc((size_t) n + 1, sizeof(uint64_t));
    ColourRefinementOptions defaults = refine_test_options();
    ColourRefinementResult result = colour_refinement_run(
        graph->refine, row_colours ? row_colours : zero_rows,
        col_colours ? col_colours : zero_cols, options ? options : &defaults,
        graph->row_classes, graph->col_classes, stats);
    free(zero_rows);
    free(zero_cols);
    return result;
}

/* Like graph_refine, with the square matrix (Pp, Pi, Px) on the columns. */
static ColourRefinementResult
graph_refine_qp(RefineGraph *graph, int m, int n, int nnz, const int *Ap,
                const int *Ai, const double *Ax, int Pnnz, const int *Pp,
                const int *Pi, const double *Px, ColourRefinementStats *stats)
{
    memset(graph, 0, sizeof(*graph));
    graph->A = matrix_new(Ax, Ai, Ap, (size_t) m, (size_t) n, (size_t) nnz);
    graph->P = matrix_new(Px, Pi, Pp, (size_t) n, (size_t) n, (size_t) Pnnz);
    graph->work = (int *) calloc((size_t) n + 1, sizeof(int));
    graph->AT = transpose(graph->A, graph->work, 0);
    graph->refine = colour_refinement_new_qp(graph->A, graph->AT, graph->P);
    graph->row_classes = (int *) calloc((size_t) m + 1, sizeof(int));
    graph->col_classes = (int *) calloc((size_t) n + 1, sizeof(int));
    uint64_t *zero_rows = (uint64_t *) calloc((size_t) m + 1, sizeof(uint64_t));
    uint64_t *zero_cols = (uint64_t *) calloc((size_t) n + 1, sizeof(uint64_t));
    ColourRefinementOptions defaults = refine_test_options();
    ColourRefinementResult result =
        colour_refinement_run(graph->refine, zero_rows, zero_cols, &defaults,
                              graph->row_classes, graph->col_classes, stats);
    free(zero_rows);
    free(zero_cols);
    return result;
}

static void graph_free(RefineGraph *graph)
{
    free_matrix(graph->P);
    colour_refinement_free(graph->refine);
    free(graph->row_classes);
    free(graph->col_classes);
    free(graph->work);
    free_matrix(graph->AT);
    free_matrix(graph->A);
}

/* Two copies of the block [[1, 2], [3, 4]] on the diagonal. */
static const int blocks_Ap[5] = {0, 2, 4, 6, 8};
static const int blocks_Ai[8] = {0, 1, 0, 1, 2, 3, 2, 3};
static const double blocks_Ax[8] = {1, 2, 3, 4, 1, 2, 3, 4};

static const char *test_refine_blocks(void)
{
    RefineGraph graph;
    ColourRefinementStats stats;
    mu_assert("converged",
              graph_refine(&graph, 4, 4, 8, blocks_Ap, blocks_Ai, blocks_Ax, NULL,
                           NULL, &stats, NULL) == COLOUR_REFINEMENT_CONVERGED);
    int *rows = graph.row_classes;
    const int *cols = graph.col_classes;
    mu_assert("two row classes", stats.n_row_classes == 2);
    mu_assert("two column classes", stats.n_col_classes == 2);
    mu_assert("rows 0 and 2 together",
              rows[0] == rows[2] && rows[1] == rows[3] && rows[0] != rows[1]);
    mu_assert("columns 0 and 2 together",
              cols[0] == cols[2] && cols[1] == cols[3] && cols[0] != cols[1]);
    /* Each of the four classes has two lines of two entries. */
    mu_assert("duplicate entries", stats.duplicate_entries == 8);

    mu_assert("verified", colour_refinement_verify(graph.refine, graph.row_classes,
                                                   graph.col_classes));

    /* Row 1 moved into the class of row 0 has different entries. */
    rows[1] = rows[0];
    mu_assert("tampered partition rejected",
              !colour_refinement_verify(graph.refine, graph.row_classes,
                                        graph.col_classes));
    graph_free(&graph);
    return 0;
}

/* One differing edge label splits everything. */
static const char *test_refine_edge_label(void)
{
    double Ax[8];
    memcpy(Ax, blocks_Ax, sizeof(Ax));
    Ax[4] = 5;
    RefineGraph graph;
    ColourRefinementStats stats;
    mu_assert("converged",
              graph_refine(&graph, 4, 4, 8, blocks_Ap, blocks_Ai, Ax, NULL, NULL,
                           &stats, NULL) == COLOUR_REFINEMENT_CONVERGED);
    mu_assert("four row classes", stats.n_row_classes == 4);
    mu_assert("four column classes", stats.n_col_classes == 4);
    mu_assert("no duplicate entries", stats.duplicate_entries == 0);
    graph_free(&graph);
    return 0;
}

/* Start colours split the blocks; -0.0 and 0.0 are one label. */
static const char *test_refine_start_colours(void)
{
    uint64_t row_colours[4] = {7, 7, 9, 9};
    RefineGraph graph;
    ColourRefinementStats stats;
    mu_assert("converged", graph_refine(&graph, 4, 4, 8, blocks_Ap, blocks_Ai,
                                        blocks_Ax, row_colours, NULL, &stats,
                                        NULL) == COLOUR_REFINEMENT_CONVERGED);
    mu_assert("four row classes", stats.n_row_classes == 4);
    graph_free(&graph);

    /* Every bit of a colour counts. */
    uint64_t bit_colours[4] = {0, 0, 1, 1};
    mu_assert("bit 0 splits", graph_refine(&graph, 4, 4, 8, blocks_Ap, blocks_Ai,
                                           blocks_Ax, bit_colours, NULL, &stats,
                                           NULL) == COLOUR_REFINEMENT_CONVERGED);
    mu_assert("four row classes", stats.n_row_classes == 4);
    graph_free(&graph);
    bit_colours[2] = bit_colours[3] = 1ULL << 63;
    mu_assert("bit 63 splits", graph_refine(&graph, 4, 4, 8, blocks_Ap, blocks_Ai,
                                            blocks_Ax, bit_colours, NULL, &stats,
                                            NULL) == COLOUR_REFINEMENT_CONVERGED);
    mu_assert("four row classes", stats.n_row_classes == 4);
    graph_free(&graph);

    /* A colour shared by no other line gives a singleton class. */
    uint64_t mixed_colours[4] = {0, 0, 2, 1};
    mu_assert("singleton colours",
              graph_refine(&graph, 4, 4, 8, blocks_Ap, blocks_Ai, blocks_Ax,
                           mixed_colours, NULL, &stats,
                           NULL) == COLOUR_REFINEMENT_CONVERGED);
    mu_assert("four row classes", stats.n_row_classes == 4);
    graph_free(&graph);

    double Ax[8];
    memcpy(Ax, blocks_Ax, sizeof(Ax));
    Ax[0] = 0.0;
    Ax[4] = -0.0;
    mu_assert("signed zeros agree",
              graph_refine(&graph, 4, 4, 8, blocks_Ap, blocks_Ai, Ax, NULL, NULL,
                           &stats, NULL) == COLOUR_REFINEMENT_CONVERGED);
    mu_assert("two row classes", stats.n_row_classes == 2);

    graph_free(&graph);
    return 0;
}

/* A path r0 - c0 - r1 - c1 - ... - r7 - c7 refines to singletons from the ends
   inwards, which takes several rounds. */
static const char *test_refine_path(void)
{
    int Ap[9], Ai[15];
    double Ax[15];
    int nnz = 0;
    for (int ii = 0; ii < 8; ii++)
    {
        Ap[ii] = nnz;
        if (ii > 0)
        {
            Ai[nnz] = ii - 1;
            Ax[nnz++] = 1;
        }
        Ai[nnz] = ii;
        Ax[nnz++] = 1;
    }
    Ap[8] = nnz;
    RefineGraph graph;
    ColourRefinementStats stats;
    mu_assert("converged",
              graph_refine(&graph, 8, 8, nnz, Ap, Ai, Ax, NULL, NULL, &stats,
                           NULL) == COLOUR_REFINEMENT_CONVERGED);
    mu_assert("a class per row", stats.n_row_classes == 8);
    mu_assert("several rounds", stats.rounds >= 2);
    graph_free(&graph);

    /* One round is not enough; the classes so far are still valid. */
    ColourRefinementOptions round_limit = {1e9, 1};
    mu_assert("round limit",
              graph_refine(&graph, 8, 8, nnz, Ap, Ai, Ax, NULL, NULL, &stats,
                           &round_limit) == COLOUR_REFINEMENT_NOT_CONVERGED);
    mu_assert("one round", stats.rounds == 1);
    int n_row_classes = stats.n_row_classes;
    mu_assert("partly refined", n_row_classes >= 2 && n_row_classes < 8);
    mu_assert("not equitable yet",
              !colour_refinement_verify(graph.refine, graph.row_classes,
                                        graph.col_classes));
    graph_free(&graph);

    /* A round touches the entries of both sides, 2 nnz. */
    ColourRefinementOptions work_limit = {1.0, 16384};
    mu_assert("work budget",
              graph_refine(&graph, 8, 8, nnz, Ap, Ai, Ax, NULL, NULL, &stats,
                           &work_limit) == COLOUR_REFINEMENT_NOT_CONVERGED);
    mu_assert("one round of work", stats.rounds == 1 && stats.work == 2.0);
    graph_free(&graph);
    return 0;
}

/* Rows without entries form a class of their own. */
static const char *test_refine_empty(void)
{
    const int Ap[4] = {0, 2, 2, 4};
    const int Ai[4] = {0, 1, 0, 1};
    const double Ax[4] = {1, 2, 1, 2};
    RefineGraph graph;
    ColourRefinementStats stats;
    mu_assert("converged",
              graph_refine(&graph, 3, 2, 4, Ap, Ai, Ax, NULL, NULL, &stats, NULL) ==
                  COLOUR_REFINEMENT_CONVERGED);
    const int *rows = graph.row_classes;
    mu_assert("two row classes", stats.n_row_classes == 2);
    mu_assert("rows 0 and 2 together", rows[0] == rows[2] && rows[0] != rows[1]);
    mu_assert("duplicate entries", stats.duplicate_entries == 2);
    graph_free(&graph);

    /* No rows at all. */
    const int none_Ap[1] = {0};
    mu_assert("converged",
              graph_refine(&graph, 0, 2, 0, none_Ap, Ai, Ax, NULL, NULL, &stats,
                           NULL) == COLOUR_REFINEMENT_CONVERGED);
    mu_assert("no row classes", stats.n_row_classes == 0);
    mu_assert("one column class", stats.n_col_classes == 1);
    graph_free(&graph);
    return 0;
}

/* The two blocks with P on the columns. A diagonal entry on column 0 alone
   splits columns 0 and 2, and through them the rows; a P that is the same on
   both blocks keeps the classes of A, and its entries count in the signatures
   and the duplicate entries. */
static const char *test_refine_P(void)
{
    RefineGraph graph;
    ColourRefinementStats stats;

    /* P = 5 e0 e0' */
    const int one_Pp[5] = {0, 1, 1, 1, 1}, one_Pi[1] = {0};
    const double one_Px[1] = {5};
    mu_assert("converged", graph_refine_qp(&graph, 4, 4, 8, blocks_Ap, blocks_Ai,
                                           blocks_Ax, 1, one_Pp, one_Pi, one_Px,
                                           &stats) == COLOUR_REFINEMENT_CONVERGED);
    mu_assert("four column classes", stats.n_col_classes == 4);
    mu_assert("four row classes", stats.n_row_classes == 4);
    mu_assert("no duplicate entries", stats.duplicate_entries == 0);
    mu_assert("verified", colour_refinement_verify(graph.refine, graph.row_classes,
                                                   graph.col_classes));
    graph_free(&graph);

    /* P = [[5, 7], [7, 6]] on both blocks */
    const int two_Pp[5] = {0, 2, 4, 6, 8};
    const int two_Pi[8] = {0, 1, 0, 1, 2, 3, 2, 3};
    const double two_Px[8] = {5, 7, 7, 6, 5, 7, 7, 6};
    mu_assert("converged", graph_refine_qp(&graph, 4, 4, 8, blocks_Ap, blocks_Ai,
                                           blocks_Ax, 8, two_Pp, two_Pi, two_Px,
                                           &stats) == COLOUR_REFINEMENT_CONVERGED);
    const int *cols = graph.col_classes;
    mu_assert("two column classes", stats.n_col_classes == 2);
    mu_assert("two row classes", stats.n_row_classes == 2);
    mu_assert("columns 0 and 2 together",
              cols[0] == cols[2] && cols[1] == cols[3] && cols[0] != cols[1]);
    /* rows: 2 classes x 1 duplicate x 2 entries; columns: 2 x 1 x (2 + 2) */
    mu_assert("duplicate entries count P", stats.duplicate_entries == 12);
    mu_assert("verified", colour_refinement_verify(graph.refine, graph.row_classes,
                                                   graph.col_classes));
    /* Column 1 moved into the class of column 0 has other P entries. */
    graph.col_classes[1] = graph.col_classes[0];
    mu_assert("tampered partition rejected",
              !colour_refinement_verify(graph.refine, graph.row_classes,
                                        graph.col_classes));
    graph_free(&graph);

    /* The coupling 7 between columns 0 and 3 instead of 2 and 3 is not an
       automorphism of P, though columns 0 and 2 still have the same P
       entries: it is their classes that differ. */
    const int cross_Pp[5] = {0, 2, 3, 4, 6};
    const int cross_Pi[6] = {0, 3, 1, 2, 0, 3};
    const double cross_Px[6] = {5, 7, 6, 5, 7, 6};
    mu_assert("converged",
              graph_refine_qp(&graph, 4, 4, 8, blocks_Ap, blocks_Ai, blocks_Ax, 6,
                              cross_Pp, cross_Pi, cross_Px,
                              &stats) == COLOUR_REFINEMENT_CONVERGED);
    mu_assert("four column classes", stats.n_col_classes == 4);
    graph_free(&graph);
    return 0;
}

/* Columns without entries in A are refined by P alone: [[1, 2], [2, 1]] keeps
   one class, a diagonal with two values gives two. The entries of P and of A
   never mix: a P entry equal to an A entry towards an equal class id. */
static const char *test_refine_P_only(void)
{
    const int Ap[1] = {0};
    const int Ai[1] = {0};
    const double Ax[1] = {0};
    RefineGraph graph;
    ColourRefinementStats stats;

    const int Pp[3] = {0, 2, 4}, Pi[4] = {0, 1, 0, 1};
    const double Px[4] = {1, 2, 2, 1};
    mu_assert("converged",
              graph_refine_qp(&graph, 0, 2, 0, Ap, Ai, Ax, 4, Pp, Pi, Px, &stats) ==
                  COLOUR_REFINEMENT_CONVERGED);
    mu_assert("one column class", stats.n_col_classes == 1);
    mu_assert("duplicate entries", stats.duplicate_entries == 2);
    graph_free(&graph);

    const int diag_Pp[3] = {0, 1, 2}, diag_Pi[2] = {0, 1};
    const double diag_Px[2] = {1, 2};
    mu_assert("converged",
              graph_refine_qp(&graph, 0, 2, 0, Ap, Ai, Ax, 2, diag_Pp, diag_Pi,
                              diag_Px, &stats) == COLOUR_REFINEMENT_CONVERGED);
    mu_assert("two column classes", stats.n_col_classes == 2);
    graph_free(&graph);

    /* Column 0 has the entry 1 in P towards itself (column class 0), column 1
       the entry 1 in A towards row 0 (row class 0): one entry each, same
       value, same class id, different sides. */
    const int mix_Ap[2] = {0, 1}, mix_Ai[1] = {1};
    const double mix_Ax[1] = {1};
    const int mix_Pp[3] = {0, 1, 1}, mix_Pi[1] = {0};
    const double mix_Px[1] = {1};
    mu_assert("converged", graph_refine_qp(&graph, 1, 2, 1, mix_Ap, mix_Ai, mix_Ax,
                                           1, mix_Pp, mix_Pi, mix_Px,
                                           &stats) == COLOUR_REFINEMENT_CONVERGED);
    mu_assert("two column classes", stats.n_col_classes == 2);
    graph_free(&graph);
    return 0;
}

/* 'copies' copies of the block on the diagonal. */
static void replicate_blocks(int copies, int **Ap_out, int **Ai_out, double **Ax_out)
{
    int m = 2 * copies, nnz = 4 * copies;
    int *Ap = (int *) malloc(((size_t) m + 1) * sizeof(int));
    int *Ai = (int *) malloc((size_t) nnz * sizeof(int));
    double *Ax = (double *) malloc((size_t) nnz * sizeof(double));
    for (int copy = 0; copy < copies; copy++)
    {
        for (int ii = 0; ii < 2; ii++)
        {
            Ap[2 * copy + ii] = 4 * copy + 2 * ii;
            for (int jj = 0; jj < 2; jj++)
            {
                Ai[4 * copy + 2 * ii + jj] = 2 * copy + jj;
                Ax[4 * copy + 2 * ii + jj] = blocks_Ax[2 * ii + jj];
            }
        }
    }
    Ap[m] = nnz;
    *Ap_out = Ap;
    *Ai_out = Ai;
    *Ax_out = Ax;
}

/* Many copies of the block, with few classes and with many. */
static const char *test_refine_large(void)
{
    int copies = 70000;
    int m = 2 * copies, nnz = 4 * copies;
    int *Ap, *Ai;
    double *Ax;
    replicate_blocks(copies, &Ap, &Ai, &Ax);
    /* Odd copies differ in a column colour, so the classes are not trivial. */
    uint64_t *row_colours = (uint64_t *) malloc((size_t) m * sizeof(uint64_t));
    uint64_t *col_colours = (uint64_t *) malloc((size_t) m * sizeof(uint64_t));
    for (int jj = 0; jj < m; jj++)
    {
        col_colours[jj] = (uint64_t) ((jj / 2) % 2);
    }
    for (int many_groups = 0; many_groups <= 1; many_groups++)
    {
        /* Rows 5000 apart agree in colour, block row and column colours. */
        for (int ii = 0; ii < m; ii++)
        {
            row_colours[ii] = many_groups ? (uint64_t) (ii % 5000) : 0;
        }
        RefineGraph graph;
        ColourRefinementStats stats;
        mu_assert("converged", graph_refine(&graph, m, m, nnz, Ap, Ai, Ax,
                                            row_colours, col_colours, &stats,
                                            NULL) == COLOUR_REFINEMENT_CONVERGED);
        mu_assert("row classes", stats.n_row_classes == (many_groups ? 5000 : 4));
        graph_free(&graph);
    }
    free(col_colours);
    free(row_colours);
    free(Ax);
    free(Ai);
    free(Ap);
    return 0;
}

static const char *all_tests_colour_refinement(void)
{
    mu_run_test(test_refine_blocks, counter_colour_refinement);
    mu_run_test(test_refine_edge_label, counter_colour_refinement);
    mu_run_test(test_refine_start_colours, counter_colour_refinement);
    mu_run_test(test_refine_path, counter_colour_refinement);
    mu_run_test(test_refine_empty, counter_colour_refinement);
    mu_run_test(test_refine_large, counter_colour_refinement);
    mu_run_test(test_refine_P, counter_colour_refinement);
    mu_run_test(test_refine_P_only, counter_colour_refinement);
    return 0;
}

int test_colour_refinement(void)
{
    const char *result = all_tests_colour_refinement();
    if (result != 0)
    {
        printf("%s\n", result);
        printf("colour refinement: TEST FAILED!\n");
    }
    else
    {
        printf("colour refinement: ALL TESTS PASSED\n");
    }
    printf("colour refinement: Tests run: %d\n", counter_colour_refinement);
    return result == 0;
}

#endif // TEST_COLOUR_REFINEMENT_H
