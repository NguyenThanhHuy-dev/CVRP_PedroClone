# CVRP Solver

This repository provides an implementation of a solver for the **Capacitated Vehicle Routing Problem** (CVRP), a well-known combinatorial optimization problem. The solver constructs initial routes using the **Clarke and Wright savings algorithm** and iteratively improves the solution using **2-opt local search**. The **k-nearest neighbors** of each customer are considered to expand the solution space, and finally, a **Max-SAT solver** is used to refine the solution to near-optimality.

This project includes two main scripts:

- `run.py`: A script that solves the CVRP for a specific instance, given the number of vehicles and neighbors to consider during the search process.
- `run_all.py`: A script that iterates over all instance files in the `instances` directory and solves them using the solver.

## Setup

To ensure a clean environment and avoid dependency conflicts, it is recommended to use a Python virtual environment.

### Creating and Activating the Virtual Environment

1. **Create the virtual environment** (if not already created):

   ```bash
   python -m venv venv
   ```

2. **Activate the virtual environment**:
   - On **Windows**:
     ```bash
     venv\Scripts\activate
     ```
   - On **Mac/Linux**:
     ```bash
     source venv/bin/activate
     ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running a Single Instance

To run the solver on a specific instance, execute the following command:

```bash
python run.py <instance> <vehicle_number> <neighbor_number>
```

### Running All Instances

To run the solver on all instances in the `instances` directory, execute the following command:

```bash
python run_all.py
```

### Deactivating the Virtual Environment

Once finished, deactivate the virtual environment by running:

```bash
deactivate
```

