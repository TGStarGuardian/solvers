# Initial CPU baseline

This report describes the original implementation. See [OPTIMIZATION.md](OPTIMIZATION.md)
for the subsequent storage and solver changes and current CPU measurements.

Measured on AMD Ryzen 7 5825U with Radeon Graphics, Python 3.12.3,
PyTorch 2.14.0+cpu, using one CPU thread and float32.
The workload uses dense Gaussian coefficients scaled by sqrt(n), seed 1729,
20 sweeps/accepted flips, one warm-up and three timed repetitions.
Annealing temperatures are 2.0 to 0.05. Histories are disabled.

| Solver | Variables | Restarts | Resident median (ms) | End-to-end median (ms) |
| --- | ---: | ---: | ---: | ---: |
| GreedyLocalSearch | 32 | 1 | 1.80 | 2.09 |
| SimulatedAnnealing | 32 | 1 | 150.09 | 149.36 |
| GreedyLocalSearch | 32 | 32 | 5.30 | 5.43 |
| SimulatedAnnealing | 32 | 32 | 114.34 | 93.72 |
| GreedyLocalSearch | 128 | 1 | 2.93 | 3.12 |
| SimulatedAnnealing | 128 | 1 | 270.98 | 283.87 |
| GreedyLocalSearch | 128 | 32 | 4.58 | 4.51 |
| SimulatedAnnealing | 128 | 32 | 305.34 | 305.38 |

These are small synthetic characterization workloads. They do not establish
production performance targets. Raw timings, quality metrics, environment,
and memory measurements are in [baseline_cpu.json](baseline_cpu.json).

Reproduce with:

```sh
python benchmarks/benchmark.py --output benchmarks/baseline_cpu.json
```

## Profiling and CUDA status

A focused cProfile run of ten solves per algorithm at 32 variables, 32
restarts and budget 20 spent about 1.70 seconds in annealing and 0.07 seconds
in greedy search. The shared flip-update routine accounted for about 0.96
seconds cumulatively. These profiled times identify work distribution; use
the unprofiled benchmark above for latency comparisons.

The annealing hot path consists of many small tensor operations inside a
sequential variable loop. Independent restarts provide parallel work, but
kernel launch overhead may limit CUDA benefit at these sizes. Greedy also
checks whether any restart can improve each round, which would synchronize
CUDA execution. These are profiling candidates, not measured GPU bottlenecks.

Following the optimize-for-gpu workflow, the CPU implementation and correctness
baseline are established. The portable PyTorch implementation retains explicit
CUDA placement and device-resident run state. No separate GPU-specific code
has been added without performance evidence.

This environment has a CPU-only PyTorch build, reports CUDA unavailable, and
has no nvidia-smi command. CUDA correctness tests and synchronized GPU benchmarks
are **pending**. The test suite and benchmark runner include CUDA cases for
execution on a suitable host. No GPU speedup is claimed.
