import random
import time
import math
import numpy as np
from typing import List, Dict, Tuple
from search.local_search import inter_route_local_search
from search.perturbation import (
    double_bridge_perturbation,
    lns_perturbation,
    perturbation,
)


def iterated_local_search(
    routes: Dict[int, List[int]],
    distances: np.ndarray,
    demands: np.ndarray,
    capacity: int,
    max_no_improve: int = 20,
    time_limit: float = 60.0,
    use_sa: bool = False,
) -> Tuple[Dict[int, List[int]], int]:
    """
    Iterated Local Search with perturbation and optional Simulated Annealing acceptance.
    """
    print("\n=== Iterated Local Search ===")

    start_time = time.time()

    best_routes, best_cost = inter_route_local_search(
        routes, distances, demands, capacity, verbose=False
    )
    current_routes, current_cost = best_routes, best_cost
    print(f"  Initial local optimum: {best_cost}")

    no_improve = 0
    iteration = 0

    if use_sa:
        temp = best_cost * 0.05  # Initial temperature
        cooling_rate = 0.95

    while no_improve < max_no_improve:
        if time.time() - start_time > time_limit:
            print(f"  Time limit reached")
            break

        iteration += 1

        perturb_choice = iteration % 8
        destroy_pct = min(0.5, 0.2 + 0.05 * (no_improve // 3))

        if perturb_choice == 0:
            perturbed = double_bridge_perturbation(
                current_routes, distances, demands, capacity
            )
        elif perturb_choice == 1:
            perturbed = lns_perturbation(
                current_routes,
                distances,
                demands,
                capacity,
                destroy_pct,
                destroy_method="random",
            )
        elif perturb_choice == 2:
            perturbed = lns_perturbation(
                current_routes,
                distances,
                demands,
                capacity,
                destroy_pct,
                destroy_method="worst",
            )
        elif perturb_choice == 3:
            perturbed = lns_perturbation(
                current_routes,
                distances,
                demands,
                capacity,
                destroy_pct,
                destroy_method="shaw",
            )
        else:
            strength = 3 + (no_improve // 5)
            perturbed = perturbation(
                current_routes, distances, demands, capacity, strength=strength
            )

        # Apply local search
        new_routes, new_cost = inter_route_local_search(
            perturbed, distances, demands, capacity, verbose=False
        )

        # Accept decision
        accept = False
        if new_cost < current_cost:
            accept = True
        elif use_sa and temp > 0.1:
            # Simulated Annealing acceptance
            delta = new_cost - current_cost
            prob = math.exp(-delta / temp)
            if random.random() < prob:
                accept = True

        if accept:
            current_routes = new_routes
            current_cost = new_cost

            if new_cost < best_cost:
                print(
                    f"  Iteration {iteration}: {best_cost} -> {new_cost} (IMPROVED -{best_cost - new_cost})"
                )
                best_routes = new_routes
                best_cost = new_cost
                no_improve = 0
            else:
                no_improve += 1
        else:
            no_improve += 1

        # Cool down
        if use_sa:
            temp *= cooling_rate

    print(f"  Final cost: {best_cost} (after {iteration} iterations)")
    return best_routes, best_cost


def multi_start_ils(
    routes: Dict[int, List[int]],
    distances: np.ndarray,
    demands: np.ndarray,
    capacity: int,
    n_restarts: int = 5,
    time_limit: float = 60.0,
) -> Tuple[Dict[int, List[int]], int]:
    """
    Multi-start ILS: Run ILS multiple times with different random seeds.
    """
    print("\n=== Multi-Start ILS ===")

    start_time = time.time()

    best_routes = None
    best_cost = float("inf")

    time_per_restart = time_limit / n_restarts

    for restart in range(n_restarts):
        if time.time() - start_time > time_limit:
            break

        random.seed(restart * 42 + 7)

        if restart > 0:
            perturbed = perturbation(routes, distances, demands, capacity, strength=10)
        else:
            perturbed = {k: list(v) for k, v in routes.items()}

        ils_routes, ils_cost = iterated_local_search(
            perturbed,
            distances,
            demands,
            capacity,
            max_no_improve=50,
            time_limit=time_per_restart,  # Increased from 30 to 50
            use_sa=False,
        )

        if ils_cost < best_cost:
            print(f"  Restart {restart + 1}: NEW BEST {ils_cost}")
            best_routes = ils_routes
            best_cost = ils_cost
        else:
            print(f"  Restart {restart + 1}: {ils_cost}")

    print(f"  Best across restarts: {best_cost}")
    return best_routes, best_cost
