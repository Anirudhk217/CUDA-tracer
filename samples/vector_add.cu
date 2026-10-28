// A small program that exercises the main things cudatrace records:
// allocations, copies in both directions, a memset, and kernels on two streams.
#include <cstdio>
#include <vector>

#define CHECK(call)                                                               \
  do {                                                                            \
    cudaError_t err = (call);                                                     \
    if (err != cudaSuccess) {                                                     \
      std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
      return 1;                                                                   \
    }                                                                             \
  } while (0)

__global__ void vectorAdd(const float *a, const float *b, float *c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}

__global__ void scale(float *x, float factor, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] *= factor;
}

int main() {
  const int n = 1 << 22;
  const size_t bytes = n * sizeof(float);
  std::vector<float> a(n, 1.0f), b(n, 2.0f), c(n);

  float *dA, *dB, *dC;
  CHECK(cudaMalloc(&dA, bytes));
  CHECK(cudaMalloc(&dB, bytes));
  CHECK(cudaMalloc(&dC, bytes));
  CHECK(cudaMemset(dC, 0, bytes));

  CHECK(cudaMemcpy(dA, a.data(), bytes, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(dB, b.data(), bytes, cudaMemcpyHostToDevice));

  const int block = 256, grid = (n + block - 1) / block;
  for (int iter = 0; iter < 5; ++iter) vectorAdd<<<grid, block>>>(dA, dB, dC, n);
  CHECK(cudaGetLastError());

  cudaStream_t s1, s2;
  CHECK(cudaStreamCreate(&s1));
  CHECK(cudaStreamCreate(&s2));
  scale<<<grid, block, 0, s1>>>(dA, 2.0f, n);
  scale<<<grid, block, 0, s2>>>(dB, 3.0f, n);
  CHECK(cudaStreamSynchronize(s1));
  CHECK(cudaStreamSynchronize(s2));

  CHECK(cudaMemcpy(c.data(), dC, bytes, cudaMemcpyDeviceToHost));
  std::printf("c[0] = %.1f (expected 3.0)\n", c[0]);

  CHECK(cudaStreamDestroy(s1));
  CHECK(cudaStreamDestroy(s2));
  CHECK(cudaFree(dA));
  CHECK(cudaFree(dB));
  CHECK(cudaFree(dC));
  return 0;
}
