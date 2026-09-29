# CPU storage and solver optimization

Measured on AMD Ryzen 7 5825U with Radeon Graphics, Linux, Python 3.12.3,
PyTorch 2.14.0+cpu, one CPU thread, float32. Each case runs
in a fresh process: one cold call followed by three timed calls. Timings below
are medians. Synthetic dense coefficients use seed 1729 and scale 1/sqrt(n).
Greedy uses 20 steps; annealing uses 3 sweeps with temperatures 2.0 to 0.05.
History is disabled. A conservative 256 MiB tensor-estimate cap bounds cases;
this cap excludes process overhead and profiler metadata.

## Latency

| Operation | Variables / restarts | Before (ms) | After (ms) | Speedup |
| --- | ---: | ---: | ---: | ---: |
| qubo construct | 256 / 32 | 0.55 | 0.50 | 1.09× |
| qubo greedy | 256 / 32 | 12.39 | 2.88 | 4.31× |
| qubo annealing | 256 / 32 | 95.54 | 60.85 | 1.57× |
| ising construct | 256 / 32 | 0.34 | 0.26 | 1.28× |
| ising greedy | 256 / 32 | 6.60 | 2.58 | 2.56× |
| ising annealing | 256 / 32 | 95.57 | 48.87 | 1.96× |
| ising convert | 256 / 32 | 0.35 | 0.18 | 1.91× |
| qubo construct | 1024 / 128 | 11.18 | 6.10 | 1.83× |
| qubo greedy | 1024 / 128 | 140.24 | 30.26 | 4.63× |
| qubo annealing | 1024 / 128 | 923.40 | 574.41 | 1.61× |
| ising construct | 1024 / 128 | 12.95 | 6.20 | 2.09× |
| ising greedy | 1024 / 128 | 159.50 | 26.65 | 5.98× |
| ising annealing | 1024 / 128 | 945.91 | 558.06 | 1.69× |
| ising convert | 1024 / 128 | 14.51 | 8.53 | 1.70× |

These are characterization measurements from one host, not production targets
or statistical guarantees. Workload-specific speedups may differ. Random
initialization now fills a floating-point tensor directly rather than allocating
an int64 intermediate, and Ising searches use native spin arithmetic. The same
seed therefore need not produce the same trajectories across versions. Raw
records include best energies; neither matching energies nor global optimality
is assumed for these heuristic benchmarks.

## Memory evidence

Construction creates one owned normalized matrix rather than cloning and then
creating several full-size arithmetic intermediates. Conversion updates the
output diagonal directly. Ising solves share the original coefficient matrix;
they no longer create and retain a QUBO matrix. Annealing uses a coefficient-row
view and fused in-place field update; best-assignment updates reuse storage.
Greedy refreshes periodically and directly verifies apparent local optima.

At 256 variables and 32 restarts, Ising annealing's positive operator allocation
traffic fell from approximately 81.5 MB to 2.1 MB for three sweeps. This metric
sums positive self-allocation totals per operator; it is not peak/live memory
and may net allocations and releases within an operator category.

At 1,024 variables and 128 restarts, the cold process RSS high-water growth for
Ising greedy fell from 16.5 MiB to approximately 6.6 MiB, and annealing from
17.3 MiB to 6.4 MiB. Growth is measured above the prior process high-water mark
following input construction. Different construction allocation patterns affect
that baseline. Small QUBO cases showed higher growth despite reduced allocation
traffic; these RSS figures are not exact live tensor peaks. Raw records also
include absolute process peak RSS, which includes Python, PyTorch and inputs.

Assignments now occupy one byte per variable in results instead of four/eight.
Explicit batching bounds the working tensors by batch_size. Full outputs still
scale with all restarts; best-only outputs retain one winner with no views into
completed batches. Initial caller storage and the coefficient matrix remain
resident. Histories are preallocated for the budget and are unavailable in
best-only mode. No automatic memory retries or precision changes occur.

## Reproduction and limitations

[Before records](optimization_before.json) and [after records](optimization_after.json)
contain all samples, allocation measurements, memory readings, and quality data.
The before implementation was preserved at /tmp/qubo-before during measurement.
To compare another version, point --source at its src directory:

```sh
python benchmarks/optimization.py --source src --include-batching --output /tmp/after.json
python benchmarks/optimization.py --source /path/to/previous/src --output /tmp/before.json
```

The recorded 256-variable cases include allocation profiles, including the
batched cases. The runner now limits future profiles to 32-restart cases because
profiler metadata can dominate memory. Profiling occurs after latency and RSS
measurements. Batching examples in the after records demonstrate the explicit
memory mode; their changed random streams prevent direct solution comparisons
with other batch sizes.

No native extension was needed for these improvements. The PyTorch implementation
remains the default and requires no compiler. Native implementations remain an
option after profiling establishes a worthwhile target and benchmarks validate it.

CUDA hardware/runtime are unavailable here. CUDA correctness, device memory,
and synchronized GPU performance measurements remain pending. The portable
implementation and conditional tests are ready for that validation; CPU results
do not establish GPU performance.
