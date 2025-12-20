#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdint.h>
#include <string.h>

struct XLA_CPU_KernelCallFrame {
    void* num_workgroups;
    void* workgroup_id;
    uint64_t thread_id;
    void* args;
};

struct XLA_CPU_NumWorkGroups {
    uint64_t x, y, z;
};

struct XLA_CPU_WorkGroupId {
    uint64_t x, y, z;
};

struct XLA_CPU_KernelArg {
    void* data;
    uint64_t size;
};

// External symbol required by the kernel
extern "C" uint64_t size_global_ptr = 0;

// External kernel function from compiled LLVM IR
extern "C" void* out_kernel(XLA_CPU_KernelCallFrame* frame);

static uint64_t get_time_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

int main(int argc, char** argv) {
    const char* kernel_file = (argc > 1) ? argv[1] : "kernels/microkernel_0.ll";
    int num_runs = (argc > 2) ? atoi(argv[2]) : 100;
    int tile_m = (argc > 3) ? atoi(argv[3]) : 16;
    int tile_n = (argc > 4) ? atoi(argv[4]) : 16;
    int tile_k = (argc > 5) ? atoi(argv[5]) : 8;

    printf("Benchmarking: %s\n", kernel_file);
    printf("Tile sizes: [%d, %d, %d]\n", tile_m, tile_n, tile_k);
    printf("Number of runs: %d\n\n", num_runs);

    // Arrays: A[batch, tile_m, K], B[batch, K, tile_n], C[batch, tile_m, tile_n]
    const int batch = 8;
    const int K = 16 * tile_k;  // K scales with tile_k to match generated kernels

    size_global_ptr = K;

    float* A = (float*)malloc(batch * tile_m * K * sizeof(float));
    float* B = (float*)malloc(batch * K * tile_n * sizeof(float));
    float* C = (float*)malloc(batch * tile_m * tile_n * sizeof(float));

    // Initialize with random values
    for (int i = 0; i < batch * tile_m * K; i++) A[i] = (float)rand() / RAND_MAX;
    for (int i = 0; i < batch * K * tile_n; i++) B[i] = (float)rand() / RAND_MAX;
    memset(C, 0, batch * tile_m * tile_n * sizeof(float));

    // Set up XLA runtime structures
    XLA_CPU_NumWorkGroups num_wg = {1, 1, 1};
    XLA_CPU_WorkGroupId wg_id = {0, 0, 0};
    XLA_CPU_KernelArg args[3] = {
        {A, batch * tile_m * K * sizeof(float)},
        {B, batch * K * tile_n * sizeof(float)},
        {C, batch * tile_m * tile_n * sizeof(float)}
    };
    XLA_CPU_KernelCallFrame frame = {&num_wg, &wg_id, 0, args};

    // Warmup runs
    for (int i = 0; i < 10; i++) {
        out_kernel(&frame);
    }

    // Timed runs
    uint64_t total_time = 0;
    uint64_t min_time = UINT64_MAX;
    uint64_t max_time = 0;

    for (int i = 0; i < num_runs; i++) {
        uint64_t start = get_time_ns();
        out_kernel(&frame);
        uint64_t end = get_time_ns();
        uint64_t elapsed = end - start;

        total_time += elapsed;
        if (elapsed < min_time) min_time = elapsed;
        if (elapsed > max_time) max_time = elapsed;
    }

    double avg_time_us = (double)total_time / num_runs / 1000.0;
    double min_time_us = (double)min_time / 1000.0;
    double max_time_us = (double)max_time / 1000.0;
    double avg_time_ms = avg_time_us / 1000.0;

    // Calculate FLOPs
    uint64_t total_flops = 2ULL * batch * tile_m * tile_n * K;  // 2 ops per multiply-add
    double gflops = (total_flops / 1e9) / (avg_time_us / 1e6);

    printf("Results:\n");
    printf("--------\n");
    printf("Average time: %.3f us (%.6f ms)\n", avg_time_us, avg_time_ms);
    printf("Min time:     %.3f us\n", min_time_us);
    printf("Max time:     %.3f us\n", max_time_us);
    printf("Total FLOPs:  %lu\n", total_flops);
    printf("Performance:  %.2f GFLOPS\n", gflops);

    int nonzero = 0;
    for (int i = 0; i < 10 && i < batch * tile_m * tile_n; i++) {
        if (C[i] != 0.0f) nonzero++;
    }
    printf("%d/10 output values are non-zero\n", nonzero);

    free(A);
    free(B);
    free(C);

    return 0;
}
