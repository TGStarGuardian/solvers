# Project scope

Build a lightweight Python library for QUBO/Ising optimization, reusable
in applications and research. Use PyTorch as the sole required runtime
dependency of the base package. Support CPU and CUDA, dense coefficients, and multiple
independent restarts for one problem. Performance targets remain unspecified.

Initial solvers are simulated annealing and greedy local search.
Differentiating through solver execution is outside the initial scope.
Keep examples, benchmarks, and development dependencies separate from
the core package.

# Responsibilities and SOLID

Problem, solver, and result are conceptual roles, not a class-count limit.

- QUBO and Ising own coefficients, validation, energy evaluation, and
  conversion. They contain no search behavior.
- Solvers own search configuration and algorithms. Keep per-run state
  local to solve().
- Results hold outcomes and diagnostics, with explicit device transfer.

Expose a small structural Solver protocol centered on solve().
Concrete solvers accept QUBO | Ising and return a common result type.
Downstream solvers can satisfy the protocol without inheritance.

Keep shared interfaces limited to behavior every implementation supports.
Preserve the same input, output, and mutation contracts across solvers.
Introduce abstractions for demonstrated variation; prefer composition
and focused helpers over speculative hierarchies.

# Mathematical contract

Minimize:
- QUBO: E(x) = xᵀQx + c, with x in {0, 1}ⁿ.
- Ising: E(s) = sᵀJs + hᵀs + c, with s in {-1, +1}ⁿ.

Use s = 2x - 1. Preserve every corresponding assignment's energy during
conversion, including the constant offset.

Symmetrize square coefficient matrices while preserving their quadratic
forms. Store Ising J with zero diagonal and absorb its diagonal into c.
Under these definitions, symmetric off-diagonal entries contribute twice.

# Tensor contract

Copy and normalize caller coefficients at construction. Treat stored
coefficients as read-only. Solvers never mutate problem coefficients or
caller-provided initial assignments.

Support float32 and float64 coefficients. The problem determines execution
device and precision. Problem .to(device=..., dtype=...) returns a new
problem. Device changes are explicit; never silently fall back to CPU.

Validate coefficient shapes, supported types, and finiteness at
construction. Validate initial assignment shapes, domain values, and
device compatibility once per solve. Keep validation outside search loops.

# Solver contract

Both solvers accept either problem type and may convert internally.
Return assignments in the original problem's domain and energies under
its original objective, including its offset.

Solver objects hold configuration. Each solve call owns its run state,
optional initial assignments, and optional random seed.

Initialize restarts uniformly at random when assignments are omitted.
Use a local random generator without altering PyTorch's global random
state. Seeded repeatability applies within the same supported environment;
CPU and CUDA trajectories need not match.

Vectorize across independent restarts.

## Simulated annealing

Use single-variable Metropolis proposals within each restart.
One sweep visits every variable in randomized order.

Use an explicit sweep budget and a geometric temperature schedule.
Require positive start and end temperatures, with start >= end.
Automatic temperature calibration is outside the initial scope.

## Greedy local search

Choose the most improving single-variable flip per restart.
Resolve ties deterministically.

Stop each restart when no strictly improving flip remains or its
accepted-flip budget is exhausted.

# Result contract

In full-result mode, retain for every restart:
- Final assignment and energy.
- Best encountered assignment and energy.
- Iteration count and termination reason.

Expose the overall best encountered solution.
Keep tensor outputs on the execution device and provide explicit .to(...)
transfer. Document tensor shapes and iteration-count units.

Support opt-in energy histories sampled at a configurable interval.
Assignment histories, callbacks, time limits, and target-energy stopping
are outside the initial scope.

# CPU-first development and GPU optimization

Implement each solver in CPU PyTorch first and establish correctness
before GPU optimization. CPU-first describes development order, not a
requirement for separate CPU and GPU source code.

After CPU tests pass, use the optimize-for-gpu skill to profile, port,
validate, and benchmark CUDA execution. If suitable hardware is
unavailable, explicitly report GPU validation and benchmarking as pending.

A user-provided GPU implementation bypasses CPU-first development order.
Preserve and validate that implementation against the common solver
contract, and establish CPU support for portability and comparison.

Keep one public solver interface. The problem's device selects execution.
Share code where practical; introduce private device-specific
implementations only when profiling justifies them. Unsupported devices
fail explicitly.

Preserve algorithm semantics and supported precision. Optional acceleration
dependencies follow the compiled-acceleration policy below.

Retain portable PyTorch code that correctly supports both devices even
when CUDA is slower on measured workloads. Retain additional GPU-specific
code only when correctness checks and representative benchmarks
demonstrate a useful improvement. Document measured limitations.

Keep reproducible benchmarks outside the core package. Cover multiple
problem sizes and restart counts. Measure both device-resident solve time
and end-to-end time including transfers, synchronizing CUDA measurements.

Record hardware, package versions, precision, solver budgets, warm-up
policy, memory use, and solution quality alongside timings. Treat results
as characterization until target workloads are specified.

# Performance and memory workflow

Prioritize solve speed over memory reduction while the workload fits in
available memory. When memory would prevent solving, use a bounded-memory
execution strategy and measure its speed tradeoff.

Whenever adding a feature, use the memory-optimization skill to inspect
storage and allocation behavior, measure relevant memory costs, and validate
changes against correctness and runtime baselines. Distinguish peak/live
memory from cumulative allocation traffic.

Whenever adding a GPU implementation, use memory-optimization together with
optimize-for-gpu. Measure device memory and synchronized runtime on suitable
hardware; explicitly report measurements that remain pending.

# Verification

Use tiny exhaustive problems to verify objective evaluation and
energy-preserving conversions in both directions.

Check incremental energy updates against direct evaluation.
Check greedy termination and seeded repeatability.
Verify assignment validity and reported energies for both solvers.

Run equivalent correctness checks on CUDA when available.
Heuristic tests must not assume global optimality on arbitrary problems.

# Storage and bounded-memory solving

Retain full dense coefficient matrices initially. Eliminate redundant
copies and conversion buffers before adopting compressed storage.
Evaluate alternative representations against measured solve speed.

Support optional restart batching through batch_size. Batching bounds
working state, but full results still scale with total restart count.
Provide memory estimates and fail clearly when a configuration cannot
fit. Do not silently change batch size or retry allocation failures.

Return assignments as int8; preserve float32/float64 coefficients and
energies. Choose internal assignment representation by measurement.

Offer an explicit best-only mode retaining the winning restart's best
assignment, energy, index, iteration count, and termination reason.
Best-only mode does not retain per-restart histories.

Seeded repeatability requires the same batch size, implementation, and
supported environment. Different batch sizes or implementations may
produce different trajectories.

# Optional compiled acceleration

Keep the public interface and problem/result tensors in Python/PyTorch.
Permit compiled implementations of performance-critical operations behind
private seams when benchmarks demonstrate speed or memory benefits.

Keep PyTorch as the sole required runtime dependency of the base package.
Optional acceleration may introduce explicitly documented build tools
and optional dependencies.

Keep the tested portable PyTorch implementation as the default and
correctness reference. Compiled acceleration requires explicit setup and
selection; ordinary installation and solving require no compiler.

Missing requested acceleration fails clearly. Implementation selection
preserves the problem's device. Report compilation/setup costs separately
from warmed execution time.

# Numerical safeguards

Evaluate periodic field refresh against correctness and speed baselines.
Before declaring a local optimum, recompute the relevant fields directly.
Score returned solutions directly under the original objective.

Preserve supported precision and algorithm semantics. Test sensitivity
to accumulated floating-point error; changes in operation order need not
preserve identical trajectories.
