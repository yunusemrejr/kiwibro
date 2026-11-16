#define _POSIX_C_SOURCE 200809L

#include "ggml-cpu.h"

#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

enum { EXPERTS = 80, ROUTES = 6 };

static uint32_t advance(uint32_t *state) {
    uint32_t n = *state;
    n ^= n << 13;
    n ^= n >> 17;
    n ^= n << 5;
    return *state = n;
}

static double time_ms(void) {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return now.tv_sec * 1000.0 + now.tv_nsec / 1e6;
}

static double run(struct ggml_cgraph *graph, struct ggml_cplan *plan, int optimized) {
    if (setenv("GGML_KIWI_MOE_ROUTE_WORKERS", optimized ? "1" : "0", 1) != 0) {
        perror("setenv");
        exit(2);
    }
    const double start = time_ms();
    if (ggml_graph_compute(graph, plan) != GGML_STATUS_SUCCESS) {
        fputs("failed to compute expert graph\n", stderr);
        exit(1);
    }
    return time_ms() - start;
}

static void compare(const void *weights, enum ggml_type type, int tokens, int threads, int pairs, int reverse) {
    const int gate = type == GGML_TYPE_Q4_K;
    const int width = gate ? 2816 : 704;
    const int rows = gate ? 1408 : 2816;
    const int inputs = gate ? 1 : ROUTES;
    struct ggml_context *ctx = ggml_init((struct ggml_init_params) {
        .mem_size = 1 << 20, .no_alloc = true,
    });
    struct ggml_tensor *bank = ggml_new_tensor_3d(ctx, type, width, rows, EXPERTS);
    struct ggml_tensor *input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, width, inputs, tokens);
    struct ggml_tensor *ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, ROUTES, tokens);
    struct ggml_tensor *output = ggml_mul_mat_id(ctx, bank, input, ids);
    float *activation = malloc(ggml_nbytes(input));
    int32_t *routes = malloc(ggml_nbytes(ids));
    float *result = malloc(ggml_nbytes(output));
    float *reference = malloc(ggml_nbytes(output));
    if (!activation || !routes || !result || !reference) {
        fputs("could not allocate expert benchmark tensors\n", stderr);
        exit(2);
    }
    bank->data = (void *)weights;
    input->data = activation;
    ids->data = routes;
    output->data = result;
    struct ggml_cgraph *graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);
    struct ggml_threadpool_params config = ggml_threadpool_params_default(threads);
    struct ggml_threadpool *pool = ggml_threadpool_new(&config);
    struct ggml_cplan plan = ggml_graph_plan(graph, threads, pool);
    plan.work_data = plan.work_size ? malloc(plan.work_size) : NULL;
    if (!pool || (plan.work_size && !plan.work_data)) {
        fputs("could not allocate expert benchmark compute plan\n", stderr);
        exit(2);
    }

    uint32_t seed = 7530;
    const int coverage = 20;
    for (int pass = -coverage; pass < pairs; ++pass) {
        for (int i = 0; i < width * inputs * tokens; ++i) {
            activation[i] = ((int32_t)(advance(&seed) % 65536) - 32768) / 32768.0f;
        }
        for (int token = 0; token < tokens; ++token) {
            const int base = advance(&seed) % EXPERTS;
            for (int i = 0; i < ROUTES; ++i) {
                routes[token * ROUTES + i] = (base + 13 * i) % EXPERTS;
            }
        }
        if (pass == -coverage) {
            routes[0] = -1;
            routes[1] = -1;
        }
        if (pass == -coverage + 1) {
            routes[0] = routes[ROUTES - 1];
        }
        double times[2];
        int first = (pass & 1) ^ reverse;
        times[first] = run(graph, &plan, first);
        memcpy(reference, result, ggml_nbytes(output));
        times[first ^ 1] = run(graph, &plan, first ^ 1);
        if (memcmp(reference, result, ggml_nbytes(output)) != 0) {
            fprintf(stderr, "parity failed for type %s at tokens=%d threads=%d pass=%d\n",
                    ggml_type_name(type), tokens, threads, pass);
            exit(1);
        }
        if (pass >= 0) {
            printf("%.6f %.6f\n", times[0], times[1]);
        }
    }
    fprintf(stderr, "bit-exact %s for %d activations x %d outputs (tokens=%d workers=%d)\n",
            ggml_type_name(type), coverage + pairs, tokens * ROUTES * rows, tokens, threads);
    free(plan.work_data);
    ggml_threadpool_free(pool);
    free(reference);
    free(result);
    free(routes);
    free(activation);
    ggml_free(ctx);
}

int main(int argc, char **argv) {
    if (argc < 5 || argc > 6 || (argc == 6 && strcmp(argv[5], "--reverse") != 0) ||
            (strcmp(argv[3], "q4_k") != 0 && strcmp(argv[3], "q5_0") != 0 && strcmp(argv[3], "q8_0") != 0)) {
        fprintf(stderr, "Usage: %s MODEL.gguf TENSOR_OFFSET {q4_k|q5_0|q8_0} PAIRS [--reverse]\n", argv[0]);
        return 2;
    }
    const int gate = strcmp(argv[3], "q4_k") == 0;
    const enum ggml_type type = gate ? GGML_TYPE_Q4_K :
        strcmp(argv[3], "q5_0") == 0 ? GGML_TYPE_Q5_0 : GGML_TYPE_Q8_0;
    char *offset_end;
    char *pairs_end;
    errno = 0;
    const off_t offset = strtoll(argv[2], &offset_end, 10);
    const long requested_pairs = strtol(argv[4], &pairs_end, 10);
    if (errno || offset_end == argv[2] || *offset_end || pairs_end == argv[4] || *pairs_end ||
            requested_pairs <= 0 || requested_pairs > 100000) {
        fputs("invalid expert tensor offset or pair count\n", stderr);
        return 2;
    }
    const int pairs = (int)requested_pairs;
    const size_t size = ggml_row_size(type, gate ? 2816 : 704) * (gate ? 1408 : 2816) * EXPERTS;
    const long page = sysconf(_SC_PAGESIZE);
    const off_t beginning = page > 0 && offset >= 0 ? offset - (offset % page) : 0;
    const int fd = open(argv[1], O_RDONLY);
    struct stat info;
    if (fd < 0 || offset < 0 || page <= 0 || fstat(fd, &info) != 0 ||
            offset > info.st_size || size > (uintmax_t)(info.st_size - offset)) {
        fputs("invalid expert tensor range or benchmark parameters\n", stderr);
        return 2;
    }
    const size_t delta = offset - beginning;
    const size_t length = delta + size;
    unsigned char *mapping = mmap(NULL, length, PROT_READ, MAP_PRIVATE, fd, beginning);
    close(fd);
    if (mapping == MAP_FAILED) {
        perror("mmap");
        return 2;
    }
    // Warm the file mapping equally for the two arms. Router choices replace
    // the active rows between timed pairs and therefore exercise varied banks.
    const volatile unsigned char *pages = mapping;
    unsigned char touched = 0;
    for (size_t i = 0; i < length; i += page) touched ^= pages[i];
    fprintf(stderr, "page checksum %u, %.1f MiB mapped\n", touched, length / 1048576.0);

    ggml_cpu_init();
    const int reverse = argc == 6;
    compare(mapping + delta, type, 1, 6, pairs, reverse);
    compare(mapping + delta, type, 2, 6, 0, reverse);
    compare(mapping + delta, type, 1, 4, 0, reverse);
    munmap(mapping, length);
    return 0;
}
