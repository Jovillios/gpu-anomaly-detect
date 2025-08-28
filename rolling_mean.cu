#include <torch/extension.h>
#include <vector>

// CUDA kernel: simple rolling mean with window=2 (toy example)
__global__ void rolling_mean_kernel(const float* __restrict__ x, float* __restrict__ y, int N, int W) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return ;

    int start = max(0, idx - W + 1);
    int count = idx - start + 1;
    float sum = 0.0;

    for (int j=start; j <= idx; j++)
    {
        sum += x[j];
    }
    y[idx] = sum / count;
}

// C++ wrapper
torch::Tensor rolling_mean(torch::Tensor x, int W) {
    auto y = torch::zeros_like(x);
    int N = x.size(0);
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    rolling_mean_kernel<<<blocks, threads>>>(x.data_ptr<float>(), y.data_ptr<float>(), N, W);
    return y;
}

// Bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rolling_mean", &rolling_mean, "Rolling mean (CUDA)");
}
