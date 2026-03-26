# CVRP Post-Improvement Using Max-SAT

A solver for the **Capacitated Vehicle Routing Problem (CVRP)** that combines classical heuristics with exact optimization. The pipeline constructs an initial solution with an enhanced Clarke-Wright savings algorithm, improves it with 2-Opt local search, then refines it globally using a Weighted Partial Max-SAT / PBO solver (`clasp`).

## Pipeline

```
┌─────────────────┐    ┌────────┐    ┌──────────────┐    ┌───────────────┐
│  Clarke-Wright   │───▶│  2-Opt │───▶│  K-Nearest   │───▶│  Max-SAT/PBO  │
│  + Reduction     │    │        │    │  Neighbors   │    │  Solver       │
└─────────────────┘    └────────┘    └──────────────┘    └───────────────┘
   Initial K routes     Intra-route    Search space         Global
                        improvement    expansion            optimization
```

### Phase 1 — Enhanced Clarke-Wright

Constructs K feasible routes using the savings heuristic, followed by a **four-strategy route reduction** mechanism:

1. **Direct Merge** — merge two routes whose combined demand ≤ Q
2. **Relocate + Merge** — move customers between routes to enable a merge
3. **Dissolve** — distribute the smallest route's customers into other routes
4. **BFD + Recursive Swap Repair** — bin packing with bounded-depth swap search for extremely tight instances (capacity tightness up to 100%)

If all strategies fail, a **sweep construction** fallback generates an alternative starting solution.

### Phase 2 — 2-Opt

Applies intra-route 2-opt until no improving segment reversal exists.

### Phase 3 — K-Nearest Neighbors

Builds restricted distance matrices by adding K nearest neighbors (via MST) for each customer. This defines the candidate edge set for the exact solver.

### Phase 4 — Max-SAT Solver

Encodes the CVRP as a Pseudo-Boolean Optimization (PBO) problem with:
- Edge variables `w_ijv`, assignment variables `t_iv`, binary position variables `u_ibv`
- MTZ subtour elimination with binary encoding
- Capacity constraints
- Search space restricted to K-neighbors edges

Solved by the `clasp` PBO solver with configurable time limit.

## Project Structure

```
.
├── run.py                      # Main entry point (single instance)
├── run_all.py                  # Batch runner for all instances
├── classes/
│   ├── __init__.py
│   ├── instance.py             # VRP file parser (EUC_2D, ATT, EXPLICIT)
│   ├── clarke_wright.py        # Enhanced CW with 4-strategy reduction + sweep
│   ├── route.py                # Route data structure with lazy caching
│   ├── two_opt.py              # Intra-route 2-opt
│   ├── k_neighbors.py          # MST-based K-nearest neighbor matrices
│   ├── solver.py               # PBO encoding + clasp interface
│   └── utils.py                # Timer decorator
├── route_optimizer.py          # Extended optimizer (ILS, GLS, LNS, MaxSAT)
├── advanced_optimizer.py       # Pairwise route optimization via MaxSAT
├── incremental_cvrp_solver.py  # Incremental MaxSAT solver (RTSS adaptation)
├── clasp                       # PBO solver binary
├── instances/                  # TSPLIB-format benchmark instances
│   ├── A-n33-k6.vrp / .sol
│   ├── B/                      # Christofides-Mingozzi-Toth B-series (23 instances)
│   ├── E-n31-k7.vrp / .sol
│   ├── F-n45-k4.vrp / .sol
│   ├── P-n*.vrp / .sol
│   └── X-n101-k25.vrp         # Extreme tightness instance (τ ≈ 100%)
├── docs/                       # LaTeX documentation + PDFs
├── requirements.txt
└── .gitignore
```

## Requirements

- **Python ≥ 3.10** (uses `match` statements)
- `numpy` — distance matrices and demand arrays
- `networkx` — minimum spanning tree
- `python-sat` — PySAT library (for `route_optimizer.py` and `incremental_cvrp_solver.py`)
- `clasp` — PBO solver binary (included in repo)

## Setup

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Single Instance

```bash
python run.py <instance_file> <num_vehicles> <num_neighbors>
```

**Example:**
```bash
python run.py instances/A-n33-k6.vrp 6 5
python run.py instances/B/B-n45-k6.vrp 6 5
python run.py instances/X-n101-k25.vrp 25 5
```

**Output:**
```
CW + 2-Opt cost: 846 (0.001s)
CW + 2-Opt routes: [Route[...], ...]
Solver cost: 780 (45.2s)
Solver routes: [w_0_5_0, w_5_12_0, ...]
Improvement: 7.80%
```

### All Instances

```bash
python run_all.py
```

Runs all `.vrp` files in `instances/` with neighbor counts 3, 4, 5.

## Instance Format

Standard TSPLIB `.vrp` format:

```
NAME : A-n33-k6
TYPE : CVRP
DIMENSION : 33
EDGE_WEIGHT_TYPE : EUC_2D
CAPACITY : 100
NODE_COORD_SECTION
 1  82  76
 2  96  44
 ...
DEMAND_SECTION
 1  0
 2  19
 ...
DEPOT_SECTION
 1
 -1
EOF
```

Supported edge weight types: `EUC_2D`, `ATT`, `EXPLICIT` (`LOWER_ROW`, `LOWER_COL`).

## Documentation

Detailed technical documentation is available in `docs/`:

| Document | Description |
|----------|-------------|
| `CVRP_PostImprovement_Documentation.pdf` | **Main documentation** — full pipeline with enhanced CW, 4-strategy reduction, PBO encoding |
| `CVRP_MaxSAT_Documentation.pdf` | Max-SAT encoding details and constraint formulation |
| `Incremental_MaxSAT_CVRP_Adaptation.pdf` | Adaptation of RTSS incremental MaxSAT approach to CVRP |

## References

- Ferreira & Arruda (2024). *Post-improving the Capacitated Vehicle Routing Problem Using a Max-SAT Solver.*
- Zha et al. *Real-Time Taxi-Sharing Service using Incremental MaxSAT.*
- Clarke & Wright (1964). *Scheduling of Vehicles from a Central Depot to a Number of Delivery Points.*
- Augerat et al. (1995). *Computational Results with a Branch and Cut Code for the CVRP.* (B-series instances)

