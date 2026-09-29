"""Isolated CPU before/after measurements with a bounded synthetic sweep."""
import argparse
import json
import os
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import sys
import time


def worker(case):
    import torch
    from qubo_solvers import QUBO, Ising, GreedyLocalSearch, SimulatedAnnealing
    torch.set_num_threads(1)
    n, r = case['n'], case['restarts']
    # Conservative cap for synthetic input, coefficient copies and working state.
    if 16 * n * n * 4 + 32 * n * r * 4 > 256 * 1024**2:
        raise ValueError('case exceeds the 256 MiB estimated tensor cap')
    g = torch.Generator().manual_seed(1729)
    matrix = torch.randn(n, n, generator=g) / n**.5
    field = torch.randn(n, generator=g)
    problem = Ising(matrix, field) if case['kind'] == 'ising' else QUBO(matrix)
    kwargs = {'restarts': r, 'seed': 1729}
    if case.get('batch_size'):
        kwargs['batch_size'] = case['batch_size']
    if case.get('best_only'):
        kwargs['best_only'] = True
    if case['op'] == 'construct':
        def operation():
            return Ising(matrix, field) if case['kind'] == 'ising' else QUBO(matrix)
    elif case['op'] == 'convert':
        def operation():
            return problem.to_qubo()
    else:
        solver = (GreedyLocalSearch(20) if case['op'] == 'greedy'
                  else SimulatedAnnealing(3, 2., .05))
        def operation():
            return solver.solve(problem, **kwargs)
    before_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    result = operation()
    after_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    energy = float(result.best_energy) if hasattr(result, 'best_energy') else None
    del result
    samples = []
    for _ in range(3):
        start = time.perf_counter()
        result = operation()
        samples.append(time.perf_counter() - start)
        del result
    allocation = None
    if n == 256 and r <= 32:
        with torch.profiler.profile(profile_memory=True) as profile:
            result = operation()
        allocation = sum(max(0, event.self_cpu_memory_usage) for event in profile.key_averages())
        del result
    return {'case': case, 'median_seconds': statistics.median(samples),
            'samples_seconds': samples, 'cold_process_peak_rss_bytes': after_peak,
            'cold_high_water_growth_bytes': max(0, after_peak - before_peak),
            'positive_operator_allocation_bytes': allocation, 'best_energy': energy,
            'torch': torch.__version__, 'python': platform.python_version()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker')
    parser.add_argument('--source', help='src directory of the version to measure')
    parser.add_argument('--output', default='benchmark-results-optimization.json')
    parser.add_argument('--include-batching', action='store_true')
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(json.loads(args.worker))))
        return
    cases = []
    for n, r in ((256, 32), (1024, 128)):
        for kind in ('qubo', 'ising'):
            for op in ('construct', 'greedy', 'annealing'):
                cases.append(dict(n=n, restarts=r, kind=kind, op=op))
        cases.append(dict(n=n, restarts=r, kind='ising', op='convert'))
    if args.include_batching:
        for op in ('greedy', 'annealing'):
            for best_only in (False, True):
                cases.append(dict(n=256, restarts=512, kind='ising', op=op,
                                  batch_size=32, best_only=best_only))
    env = os.environ.copy()
    if args.source:
        env['PYTHONPATH'] = str(Path(args.source).resolve())
    records = []
    for case in cases:
        completed = subprocess.run([sys.executable, __file__, '--worker', json.dumps(case)],
                                   env=env, capture_output=True, text=True, check=True)
        records.append(json.loads(completed.stdout))
        print(case, flush=True)
    cpu = platform.processor()
    if Path('/proc/cpuinfo').exists():
        cpu = next((line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')), cpu)
    report = {'processor': cpu, 'platform': platform.platform(), 'threads': 1, 'dtype': 'float32',
              'seed': 1729, 'warmup': 'one cold call, then three timed calls',
              'estimated_tensor_cap_bytes': 256 * 1024**2,
              'memory_notes': 'Fresh process per case. RSS includes Python/PyTorch/input; growth is above prior RSS high-water mark, not exact live tensor peak. Operator allocation is cumulative positive self allocation by operator, not peak memory. Profiling occurs after timing/RSS measurement.',
              'records': records}
    Path(args.output).write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
