# Rolling Mean CUDA

This project implements a CUDA-accelerated rolling mean function using PyTorch.

## Features

- CUDA-accelerated rolling mean calculation
- PyTorch integration

## Installation

1. Clone the repository : `git clone https://github.com/jovillios/gpu-anomaly-detect`
2. Install the required dependencies : `pip install -r requirements.txt`
3. Build the CUDA extension : `python setup.py install`

## Usage

```python
import torch
import rolling_mean_cuda

x = torch.arange(10, dtype=torch.float32, device="cuda")
y = rolling_mean_cuda.rolling_mean(x, 3)
print("Input:", x)
print("Rolling mean:", y)
```

## License

MIT License

Copyright (c) [2025] [Jovillios]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
