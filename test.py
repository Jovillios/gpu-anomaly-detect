import torch
import rolling_mean_cuda

x = torch.arange(10, dtype=torch.float32, device="cuda")
y = rolling_mean_cuda.rolling_mean(x)
print("Input:", x)
print("Rolling mean:", y)
