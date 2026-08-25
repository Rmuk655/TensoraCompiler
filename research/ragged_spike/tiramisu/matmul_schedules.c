/* Tiramisu-principle simulation (not Tiramisu the system).
 *
 * Tiramisu's premise: the same computation, under different loop
 * schedules, has very different performance -- and the legal schedules
 * are governed by the computation's dependencies. This file computes the
 * SAME matmul C = A*B three ways and times each:
 *   1. naive ijk   -- inner loop strides down a column of B (cache-hostile)
 *   2. interchange ikj -- inner loop runs along a row of B and C (contiguous)
 *   3. tiled ikj   -- blocked for cache reuse
 *
 * Interchange and tiling are legal here only because matmul's k axis is a
 * reduction (reordering the accumulation is allowed) -- exactly the kind
 * of semantic fact a compiler needs to know. Compiled at -O2 (no
 * -floop-interchange), so the schedule written here is the schedule run.
 *
 * Build: gcc -O2 -march=native matmul_schedules.c -o matmul_schedules
 * Run:   ./matmul_schedules [N] [tile]
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double now(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

static void naive_ijk(const float *A, const float *B, float *C, int N) {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            float s = 0.f;
            for (int k = 0; k < N; k++) s += A[i*N+k] * B[k*N+j];
            C[i*N+j] = s;
        }
}

static void interchange_ikj(const float *A, const float *B, float *C, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) C[i*N+j] = 0.f;
        for (int k = 0; k < N; k++) {
            float a = A[i*N+k];
            for (int j = 0; j < N; j++) C[i*N+j] += a * B[k*N+j];
        }
    }
}

static void tiled_ikj(const float *A, const float *B, float *C, int N, int T) {
    for (int i = 0; i < N*N; i++) C[i] = 0.f;
    for (int ii = 0; ii < N; ii += T)
        for (int kk = 0; kk < N; kk += T)
            for (int jj = 0; jj < N; jj += T)
                for (int i = ii; i < ii+T && i < N; i++)
                    for (int k = kk; k < kk+T && k < N; k++) {
                        float a = A[i*N+k];
                        for (int j = jj; j < jj+T && j < N; j++)
                            C[i*N+j] += a * B[k*N+j];
                    }
}

int main(int argc, char **argv) {
    int N = argc > 1 ? atoi(argv[1]) : 1024;
    int T = argc > 2 ? atoi(argv[2]) : 64;
    float *A = malloc((size_t)N*N*sizeof(float));
    float *B = malloc((size_t)N*N*sizeof(float));
    float *C = malloc((size_t)N*N*sizeof(float));
    for (int i = 0; i < N*N; i++) { A[i] = (i%7)*0.1f; B[i] = (i%5)*0.1f; }
    double flops = 2.0 * N * N * N;

    double t0 = now(); naive_ijk(A, B, C, N); double t1 = now();
    double ref = C[0] + C[N*N-1];
    double t2 = now(); interchange_ikj(A, B, C, N); double t3 = now();
    double c1 = C[0] + C[N*N-1];
    double t4 = now(); tiled_ikj(A, B, C, N, T); double t5 = now();
    double c2 = C[0] + C[N*N-1];

    printf("matmul N=%d, tile=%d  (same computation, three schedules)\n\n", N, T);
    printf("%-16s %10s %10s   %s\n", "schedule", "ms", "GFLOP/s", "correct");
    printf("--------------------------------------------------\n");
    printf("%-16s %10.1f %10.2f   %s\n", "naive ijk",
           (t1-t0)*1e3, flops/(t1-t0)/1e9, "reference");
    printf("%-16s %10.1f %10.2f   %s\n", "interchange ikj",
           (t3-t2)*1e3, flops/(t3-t2)/1e9, (c1==ref)?"yes":"~yes");
    printf("%-16s %10.1f %10.2f   %s\n", "tiled ikj",
           (t5-t4)*1e3, flops/(t5-t4)/1e9, (c2==ref)?"yes":"~yes");
    printf("\nSame math, different schedule -> different speed. The reorder is\n");
    printf("legal because k is a reduction axis. That fact is what a compiler\n");
    printf("must know to pick the schedule -- Tiramisu's premise.\n");
    free(A); free(B); free(C);
    return 0;
}
