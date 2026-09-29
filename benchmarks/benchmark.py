"""Reproducible latency/quality characterization; no extra dependencies."""

import argparse
from dataclasses import asdict
import json
import platform
from pathlib import Path
import resource
from statistics import median
import time

import torch

from qubo_solvers import GreedyLocalSearch, QUBO, SimulatedAnnealing


def cpu_name():
    info = Path("/proc/cpuinfo")
    if info.exists():
        for line in info.read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine()


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure(function, device, warmup, repeats):
    for _ in range(warmup):
        function()
    synchronize(device)
    samples = []
    result = None
    for _ in range(repeats):
        synchronize(device)
        started = time.perf_counter()
        result = function()
        synchronize(device)
        samples.append(time.perf_counter() - started)
    return {"median_seconds": median(samples), "samples_seconds": samples}, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[32, 128])
    parser.add_argument("--restarts", nargs="+", type=int, default=[1, 32])
    parser.add_argument("--devices", nargs="+", choices=["cpu", "cuda"], default=["cpu", "cuda"])
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--budget", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--best-only", action="store_true")
    parser.add_argument("--output", default="benchmark-results.json")
    args = parser.parse_args()
    if min(args.sizes + args.restarts + [args.repeats, args.threads]) < 1 or min(args.budget, args.warmup) < 0:
        parser.error("sizes, restarts, repeats, threads must be positive; budget and warmup nonnegative")
    torch.set_num_threads(args.threads)
    dtype = getattr(torch, args.dtype)
    report = {
        "environment": {"platform": platform.platform(), "processor": cpu_name(),
                        "python": platform.python_version(), "torch": torch.__version__,
                        "cuda_runtime": torch.version.cuda, "threads": args.threads},
        "configuration": vars(args), "seed": 1729, "records": [], "skipped": [],
        "memory_notes": "CPU RSS is cumulative process high-water mark, not per-solve allocation. CUDA peak includes resident problem and results.",
        "workload": "Dense symmetric Gaussian QUBO, coefficients scaled by sqrt(n), offset zero; no history.",
    }
    for name in args.devices:
        if name == "cuda" and not torch.cuda.is_available():
            report["skipped"].append({"device": name, "reason": "CUDA unavailable; validation and benchmarking pending"})
            continue
        device = torch.device(name)
        if name == "cuda":
            report["environment"]["gpu"] = torch.cuda.get_device_name(device)
        for n in args.sizes:
            generator = torch.Generator().manual_seed(1729)
            raw = torch.randn(n, n, dtype=dtype, generator=generator) / n ** .5
            cpu_problem = QUBO(raw)
            problem = cpu_problem.to(device=device)
            for restarts in args.restarts:
                for solver in (GreedyLocalSearch(args.budget), SimulatedAnnealing(args.budget, 2., .05)):
                    def resident():
                        return solver.solve(problem, restarts=restarts, seed=1729,
                                            batch_size=args.batch_size, best_only=args.best_only)

                    def end_to_end():
                        transferred = cpu_problem.to(device=device)
                        return solver.solve(transferred, restarts=restarts, seed=1729,
                                            batch_size=args.batch_size, best_only=args.best_only).to(device="cpu")

                    if name == "cuda":
                        torch.cuda.reset_peak_memory_stats(device)
                    resident_timing, result = measure(resident, device, args.warmup, args.repeats)
                    full_timing, _ = measure(end_to_end, device, args.warmup, args.repeats)
                    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    rss_bytes = rss if platform.system() == "Darwin" else rss * 1024
                    report["records"].append({
                        "device": name, "n_variables": n, "restarts": restarts,
                        "dtype": args.dtype, "solver": type(solver).__name__,
                        "solver_configuration": asdict(solver),
                        "resident": resident_timing, "end_to_end": full_timing,
                        "best_energy": result.best_energy.item(),
                        "mean_best_energy": result.best_energies.mean().item(),
                        "mean_final_energy": result.final_energies.mean().item() if result.final_energies is not None else None,
                        "process_peak_rss_bytes": rss_bytes,
                        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if name == "cuda" else None,
                    })
    with open(args.output, "w") as output:
        json.dump(report, output, indent=2)
        output.write("\n")
    print(f"Wrote {len(report['records'])} measurements to {args.output}")
    for skipped in report["skipped"]:
        print(skipped["reason"])


if __name__ == "__main__":
    main()
