"""
Matrix Multiplication Algorithms in CUDA

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - matmul_cpu
void matmul_cpu(const float* A, const float* B, float* C, int M, int N, int K) {
    // TODO: C[m*N + n] = sum over k of A[m*K + k] * B[k*N + n]
    // (void)A; (void)B; (void)C; (void)M; (void)N; (void)K;

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            // make sure c is empty for this inner loop
            // this might be a test case issue
            C[i*N + j] = 0.0f;

            for (int k = 0; k < K; k++) {
                // Two dim would looks like:
                // C[M][N] += (A[i][k] * B[k][j])
                C[i*N + j] += (A[i*K + k] * B[k*N + j]);
            }
        }
    }
}

# Step 2 - max_abs_diff
#include <cmath>

float max_abs_diff(const float* a, const float* b, int n) {
    // TODO: largest |a[i] - b[i]| over n elements, 0 for n == 0
    (void)a; (void)b; (void)n;

    float result = 0.0f;

    for (int i = 0; i < n; i++) {
        result = std::max(result, std::abs(a[i] - b[i]));
    }

    return result;
}

# Step 3 - matmul_naive_kernel
#include <cuda_runtime.h>

__global__ void matmul_naive_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    // TODO: row from threadIdx.x, col from threadIdx.y, guard, accumulate over k, store
    (void)A; (void)B; (void)C; (void)M; (void)N; (void)K;

    // each thread compute one output element, which means do K add-mul
    // this row is for real 2d matrix
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    // do not let kernel over bound
    if (row >= M || col >= N) return;

    float accum = 0.0f;
    for (int k = 0; k < K; k++) {
        accum += (A[row * K + k] * B[k * N + col]);
    }
    C[row * N + col] = accum;
}

void launch_matmul_naive(const float* A, const float* B, float* C, int M, int N, int K) {
    // TODO: one block has 16x16 threads, grid ((M+15)/16, (N+15)/16)
    (void)A; (void)B; (void)C; (void)M; (void)N; (void)K;

    // a wrap is 32 threads
    // devide the M and N into multiple block, based on the block dim

    dim3 block(16, 16);
    dim3 grid(
        (M + 15) / 16,
        (N + 15) / 16
    );
    matmul_naive_kernel<<< grid, block >>> (A, B, C, M, N, K);
    cudaDeviceSynchronize();
}

# Step 4 - matmul_coalesced_kernel
#include <cuda_runtime.h>

__global__ void matmul_coalesced_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    // 简单来说，wrap 内是行优先
    // 尽可能让wrap内的thread不跨行读

    // 这种coalesced，A广播（wrap内相同），B连续，C连续
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= M || col >= N) return;

    float sum = 0.0f;
    for (int k = 0; k < K; k++) {
        sum += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
}

void launch_matmul_coalesced(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16,
              (M + 15) / 16);

    matmul_coalesced_kernel<<<grid, block>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}

# Step 5 - time_launch_ms
#include <cuda_runtime.h>

typedef void (*matmul_launch_fn)(const float*, const float*, float*, int, int, int);

float time_launch_ms(matmul_launch_fn launch, const float* dA, const float* dB, float* dC, int M, int N, int K, int iters) {
    // TODO: warm-up launch + sync; record event, iters launches, record event, sync; return elapsed / iters
    if (iters <= 0) return 0.0f;

    // warm up
    launch(dA, dB, dC, M, N, K);
    cudaDeviceSynchronize();

    // create event
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // start the timer
    cudaEventRecord(start);
    for (int i = 0; i < iters; i++) {
        launch(dA, dB, dC, M, N, K);
    }
    cudaEventRecord(stop);
    
    cudaEventSynchronize(stop);

    float total_ms = 0.0f;
    cudaEventElapsedTime(&total_ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return total_ms / iters;
}

double matmul_gflops(int M, int N, int K, float ms) {
    // TODO: 2*M*N*K flops over ms milliseconds, in GFLOP/s
    if (ms <= 0.0f) return 0.0;

    double flops = 2.0 * (double)M * (double)N * (double)K;
    return flops / ((double)ms * 1e-3) / 1e9;
}

# Step 6 - matmul_tiled_kernel
#include <cuda_runtime.h>

constexpr int TILE_SMEM = 16;

__global__ void matmul_tiled_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    // in the interview, please draw a picture of
    // A -> TILE_SMEM * K
    // B -> K * TILE_SMEM
    // sliding window size TILE_SMEM * TILE_SMEM

    // inside of each thread in charge of a whole row/col
    // we let them in charge of the things inside a tile

    // this means the x and y inside a tile (local)
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // this means the x and y in A and B (global)
    // note: coalesced
    int col = blockIdx.x * blockDim.x + tx;
    int row = blockIdx.y * blockDim.y + ty;

    __shared__ float sA [TILE_SMEM][TILE_SMEM]; 
    __shared__ float sB [TILE_SMEM][TILE_SMEM];

    // for each block, each time sliding TILE_SMEM
    float sum = 0.0f; 
    for (int t = 0; t < K; t += TILE_SMEM) {
        // move this thread element to share mem
        // take care of the current sliding status
        int t_col = t + tx;
        sA[ty][tx] = (row < M && t_col < K) ? A[row * K + t_col] : 0.0f;

        int t_row = t + ty;
        sB[ty][tx] = (t_row < K && col < N) ? B[t_row * N + col] : 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_SMEM; k++) {
            sum += sA[ty][k] * sB[k][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

void launch_matmul_tiled(const float* A, const float* B, float* C, int M, int N, int K) {
    // TODO: 16x16 blocks, grid ((N+15)/16, (M+15)/16)
    dim3 block(TILE_SMEM, TILE_SMEM);
    dim3 grid(
        (N + TILE_SMEM - 1) / TILE_SMEM,
        (M + TILE_SMEM - 1) / TILE_SMEM
    );
    matmul_tiled_kernel<<<grid, block>>>(A, B, C, M, N, K);
}

# Step 7 - matmul_tiled_1d_kernel
#include <cuda_runtime.h>

constexpr int R1_BM = 64, R1_BN = 64, R1_BK = 8, R1_TM = 8;

__global__ void matmul_tiled_1d_kernel(const float* A, const float* B, float* C,
                                       int M, int N, int K) {
    // 每个线程没读一个B，和8个A做乘加
    __shared__ float As[R1_BM][R1_BK];
    __shared__ float Bs[R1_BK][R1_BN];

    const int tid = threadIdx.x;
    const int blockRow = blockIdx.y * R1_BM;
    const int blockCol = blockIdx.x * R1_BN;

    // 这个线程负责的输出：第 threadCol 列，从 threadRowBase 开始的 8 行
    const int threadCol     = tid % R1_BN;
    const int threadRowBase = (tid / R1_BN) * R1_TM;

    // 加载映射
    const int aRow = tid / R1_BK, aCol = tid % R1_BK;   // 64 x 8
    const int bRow = tid / R1_BN, bCol = tid % R1_BN;   // 8 x 64

    float acc[R1_TM] = {0.0f};

    for (int t = 0; t < K; t += R1_BK) {
        int gAr = blockRow + aRow, gAc = t + aCol;
        As[aRow][aCol] = (gAr < M && gAc < K) ? A[gAr * K + gAc] : 0.0f;

        int gBr = t + bRow, gBc = blockCol + bCol;
        Bs[bRow][bCol] = (gBr < K && gBc < N) ? B[gBr * N + gBc] : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < R1_BK; ++k) {
            float b = Bs[k][threadCol];          // 读一次，复用 8 次
            #pragma unroll
            for (int i = 0; i < R1_TM; ++i) {
                acc[i] += As[threadRowBase + i][k] * b;
            }
        }

        __syncthreads();
    }

    const int gCol = blockCol + threadCol;
    #pragma unroll
    for (int i = 0; i < R1_TM; ++i) {
        int gRow = blockRow + threadRowBase + i;
        if (gRow < M && gCol < N) {
            C[gRow * N + gCol] = acc[i];
        }
    }
}

void launch_matmul_tiled_1d(const float* A, const float* B, float* C,
                            int M, int N, int K) {
    dim3 block(512);
    dim3 grid((N + R1_BN - 1) / R1_BN, (M + R1_BM - 1) / R1_BM);
    matmul_tiled_1d_kernel<<<grid, block>>>(A, B, C, M, N, K);
}

# Step 8 - matmul_tiled_2d_kernel (not yet solved)
# TODO: implement

# Step 9 - matmul_vectorized_kernel (not yet solved)
# TODO: implement

# Step 10 - matmul_double_buffered_kernel (not yet solved)
# TODO: implement

# Step 11 - matmul_nt_kernel (not yet solved)
# TODO: implement

# Step 12 - matmul_batched_kernel (not yet solved)
# TODO: implement

# Step 13 - matmul_splitk_kernel (not yet solved)
# TODO: implement

# Step 14 - gemv_kernel (not yet solved)
# TODO: implement

# Step 15 - matmul_bias_relu_kernel (not yet solved)
# TODO: implement

# Step 16 - matrix_addsub_kernel (not yet solved)
# TODO: implement

# Step 17 - strassen_one_level (not yet solved)
# TODO: implement

# Step 18 - csr_spmm_kernel (not yet solved)
# TODO: implement

# Step 19 - matmul_lower_triangular_kernel (not yet solved)
# TODO: implement

# Step 20 - matmul_dispatch (not yet solved)
# TODO: implement

