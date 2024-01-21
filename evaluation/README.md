# TorchSparse++ Benchmark Code Usage

You will be able to directly run commands look like

```bash

OMP_NUM_THREADS=1 python evaluate.py --fast

```

to perform fast evaluation (100 samples per benchmark) or full evaluation:

```bash

OMP_NUM_THREADS=1 python evaluate.py

```

fp32 and tf32 precisions are also supported:

```bash

OMP_NUM_THREADS=1 python evaluate.py --precision fp32

```
