# qubo-solvers

A small PyTorch library for dense QUBO and Ising minimization, with simulated
annealing and greedy local search. Algorithms are developed and tested on CPU
first. The same tensor implementation accepts CUDA problems; CUDA correctness
and performance are pending validation on suitable hardware.

## Install

Requires Python 3.10 or later and PyTorch 2.2 or later. Install the PyTorch build
appropriate for your hardware, then install this repository:

```sh
python -m pip install -e '.[dev]'
pytest
```

PyTorch is the only direct runtime dependency. NumPy is not required. Some
PyTorch builds emit an optional NumPy initialization warning when it is absent.

## Solve

```python
import torch
from qubo_solvers import QUBO, SimulatedAnnealing, GreedyLocalSearch

problem = QUBO(torch.tensor([[-1., 1.], [1., -2.]]), offset=0.)
solver = SimulatedAnnealing(
    sweeps=100, start_temperature=2., end_temperature=.01,
)
result = solver.solve(problem, restarts=16, seed=42, history_interval=10)
print(result.best_assignment, result.best_energy)

# Refine the best solutions from annealing.
result = GreedyLocalSearch(max_steps=100).solve(
    problem, restarts=16, initial_assignments=result.best_assignments,
)

# Exact conversion of the objective, including its constant offset.
ising = problem.to_ising()
spins = 2 * result.best_assignment - 1
assert torch.allclose(problem.energy(result.best_assignment), ising.energy(spins))
```

Run `python examples/basic.py` for the example. To execute on CUDA, explicitly
move the problem with `problem.to(device="cuda")`. Inputs and results stay on
the selected device; use `result.to(device="cpu")` for transfer. There is no
silent fallback. Supplied initial assignments must be on the problem's device.

## Mathematical and tensor contract

- QUBO: `x.T @ Q @ x + offset`, with binary `x`.
- Ising: `s.T @ J @ s + h @ s + offset`, with spins `s = 2*x - 1`.
- Constructors copy/detach coefficients, symmetrize matrices, and move the Ising
  diagonal into the offset. Upper-triangular QUBO matrices are accepted.
  Symmetric off-diagonal entries count twice.
- Coefficients must be finite, dense, nonempty `float32` or `float64` tensors.
  Ising `h` must match `J` in device and dtype. Stored tensors are read-only by
  convention: frozen objects prevent attribute replacement, not tensor mutation.
- `energy(assignments)` accepts shape `(..., n)` and returns shape `(...)`.
  Solving accepts initial assignments of shape `(restarts, n)`; otherwise it
  initializes each restart uniformly. Assignment outputs use int8; energies retain coefficient dtype.
- `.to(device=..., dtype=...)` creates a new problem. Solving does not build an
  autograd graph or modify caller tensors or global random-generator state.

`QUBO`, `Ising`, and `OptimizationResult` own their respective data contracts.
Solvers are immutable configuration objects; run state is local to `solve`.
The structural `Solver` protocol lets downstream solvers participate without
inheritance. No generic problem hierarchy is required.

## Algorithms and results

Annealing makes single-variable Metropolis proposals, visiting each variable
once per sweep in a shuffled order shared across restarts within each batch. Acceptance draws are
independent across restarts. Temperatures follow a geometric schedule including
both endpoints; one sweep uses the start temperature. Zero sweeps return the
initial solutions. Start/end temperatures must be finite and positive, with
start at least end.

Greedy search takes the most improving single-variable flip independently per
restart, resolving ties by the lowest variable index. It stops at a local optimum
or the accepted-flip budget. A zero budget still checks local optimality. If the
last permitted flip reaches a local optimum, that is the reported reason.

Both use incremental flip calculations in the problem's native representation;
Ising solving does not materialize a QUBO coefficient matrix. Annealing refreshes
fields after each sweep. Greedy refreshes periodically (`refresh_interval=32`),
and directly verifies fields before declaring a local optimum. Set
`refresh_interval=1` for a more frequent refresh baseline. Returned energies
are evaluated directly under the original objective. Floating-point rounding
may affect near-ties; use float64 when sensitivity warrants it.

A seed provides repeatability for the same batch size, implementation, and
supported environment. Different batch sizes, versions, or CPU/CUDA execution
may produce different trajectories. Neither heuristic guarantees a global optimum.

Full-result mode (the default):

| Result field | Shape / meaning |
| --- | --- |
| `final_assignments`, `best_assignments` | `(restarts, n)`, original problem domain |
| `final_energies`, `best_energies` | `(restarts,)`; best tracks intermediate solutions too |
| `best_assignment`, `best_energy` | `(n,)` and scalar; overall best, first restart on ties |
| `best_restart_index` | Scalar index of the winning restart |
| `iterations` | `(restarts,)`, sweeps for annealing, accepted flips for greedy |
| `termination_reasons` | `(restarts,)`, integer `TerminationReason` values |
| `energy_history` | `(samples, restarts)` or `None`; current energy, not best-so-far |
| `history_iterations` | `(samples,)` or `None`; global sweep/greedy round indices |

Histories include initialization, interval samples, and the final state without
duplicating the final index. Greedy restarts that stop early hold their energy
while other restarts continue. Trajectory storage is opt-in and preallocated
for the configured budget; an early-stop history view may retain that capacity.

## Memory controls

Speed takes priority while the workload fits. Dense matrices remain fully
stored for fast row access and matrix multiplication. Construction, transfers,
and conversions allocate owned coefficients without redundant normalization
copies. Internal assignments remain floating-point for efficient arithmetic;
returned assignments use one byte each.

```python
from qubo_solvers import estimate_memory

estimate = estimate_memory(problem, restarts=4096, batch_size=64, best_only=True)
print(estimate.total_bytes)
result = solver.solve(
    problem, restarts=4096, batch_size=64, best_only=True, seed=42,
    memory_limit_bytes=512 * 1024**2,
)
print(result.best_assignment, result.best_energy, result.best_restart_index)
```

`batch_size` bounds the number of concurrent restarts and may reduce throughput.
Without it, all restarts run together. Full mode stores outputs for every restart;
`best_only=True` retains only the winning restart across completed batches.
Its best assignment/energy and diagnostic arrays have one row, `restart_indices`
contains the original winning index, and final assignment/energy fields are
`None`. Histories are rejected in this mode. Winner diagnostics describe the
winning restart's entire run, not the moment its best solution was found.

`estimate_memory` reports problem storage, output storage, and a conservative
workspace allowance in bytes. For full histories, pass `history_samples` equal
to `budget // interval + 1 + bool(budget % interval)`. Estimates exclude caller
initial assignments, Python overhead, allocator caches, library workspaces,
and CUDA context. They are planning estimates, not a guarantee of fit.
`memory_limit_bytes` checks that estimate before allocating search state.
Allocation failures do not trigger automatic retries or batch-size changes.
Batching cannot make a coefficient matrix larger than available memory fit.

The public interface stays Python/PyTorch. Optional compiled acceleration is
permitted when benchmarks justify it, but no compiled backend is currently
included. Installation and solving require no compiler.

## Verification and benchmarks

```sh
pytest -q
python benchmarks/benchmark.py --sizes 32 128 --restarts 1 32
```

Tests exhaustively check small objectives and conversions, greedy decisions
against direct neighbor enumeration, best-solution retention, termination,
reproducibility, and input ownership. The same tests run on CUDA when available.

Benchmarks report warmed, synchronized device-resident solve latency and
end-to-end latency including problem transfer and result transfer to CPU.
JSON output includes repeated timings, solution quality, configuration, hardware,
versions, and memory measurements. CPU RSS is a cumulative process high-water
mark, not an isolated allocation measurement. Initialization/context startup is
excluded after warm-up. Change problem sizes, restarts, dtype, budgets, and thread
count to match your workloads; these are characterization results, not universal
speedup claims.

GPU optimization follows the `optimize-for-gpu` workflow after CPU correctness.
Use profiling and measured improvements to justify specialized implementations;
portable tensor code remains shared where practical. See
[the baseline report](benchmarks/BASELINE.md) for the current validation status.

The [optimization report](benchmarks/OPTIMIZATION.md) records before/after CPU
latency, process memory and allocation traffic, including their limitations.
Reproduce the current side with:

```sh
python benchmarks/optimization.py --source src --include-batching --output /tmp/optimization.json
```

## Additional classical solvers

Every solver accepts unconstrained `QUBO` or `Ising` and lives in its own module.
The problem's device selects CPU or CUDA execution:

```python
from qubo_solvers import QUBO, TransverseRoute, TabuSearch

problem = QUBO(Q).to(device="cuda", dtype=torch.float32)
solver = TransverseRoute(max_steps=500)
result = solver.solve(problem, restarts=128, batch_size=32, seed=7)
print(result.best_energy)

# A separate solver, with its own output; no implicit refinement stage.
tabu_result = TabuSearch(max_steps=500, tenure=7).solve(problem, restarts=128, seed=7)
```

Also available: `HeatBathAnnealing`, `SpinVectorLangevin`, `AngularAnnealing`,
`EasyAxisAnnealing`, `SpinCoherentAnnealing`, `VectorAmplitudeAnnealing`,
`MeanFieldAnnealing`, `TAPAnnealing`, `SphericalAnnealing`, `ContactAnnealing`,
`ReplicaAnnealing`, `ExchangeCascade`, and `RandomSearch`.

New solvers expose `solver.estimate_memory(problem, restarts=..., batch_size=...)`
for their auxiliary working state. All retain the common result, explicit device
transfer, best-only, and energy-history contracts. Continuous solver iteration
counts are integration steps. No quantum simulation or constrained-problem API is
included. See [the catalogue](docs/SOLVER_CATALOGUE.md) for equations, selected
integrators, source provenance, and the report-based patent selection.

Reproduce random/Biq Mac characterization with:

```bash
PYTHONPATH=src python benchmarks/catalogue.py --device cpu --output cpu.json
PYTHONPATH=src python benchmarks/catalogue.py --device cuda --output cuda.json
```

Each solver is scored directly; there is no postprocessing solver. These are
heuristic comparisons, not assertions of global optimality or GPU speedup.
