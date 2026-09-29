import torch

from qubo_solvers import GreedyLocalSearch, QUBO, SimulatedAnnealing

problem = QUBO(torch.tensor([[-1., 1.], [1., -2.]]))
annealed = SimulatedAnnealing(
    sweeps=100, start_temperature=2., end_temperature=.01,
).solve(problem, restarts=16, seed=42)

refined = GreedyLocalSearch(max_steps=100).solve(
    problem, restarts=16, initial_assignments=annealed.best_assignments,
)
print("Best assignment:", refined.best_assignment)
print("Best energy:", refined.best_energy)
