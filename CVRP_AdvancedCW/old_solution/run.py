from sys import argv

from classes import Instance, ClarkeWright, TwoOpt, KNeighbors, Solver

if __name__ == '__main__':
    if len(argv) < 4:
        print('Usage: python run.py <instance> <vehicle_number> <neighbor_number>')
        print('Example: python run.py instances/A-n33-k6.vrp 6 5')
        exit(1)
        
    cvrp = Instance(argv[1]).load()
    cw_time, routes = ClarkeWright.run(cvrp, int(argv[2]))
    to_time, routes = TwoOpt.run(routes)
    
    cw_time = cw_time + to_time
    cw_cost = sum(route.cost for route in routes.values())
    cw_routes = list(routes.values())
    
    print(f'CW + 2-Opt cost: {cw_cost} ({cw_time:.3f}s)')
    print('CW + 2-Opt routes:', cw_routes)
    
    _, matrices = KNeighbors.run(cvrp, int(argv[3]), routes)
        
    solver_time, solver_cost, solver_routes = Solver.run(cvrp, matrices)
    
    print(f'Solver cost: {solver_cost} ({solver_time:.3f}s)')
    print('Solver routes:', solver_routes)
    
    print(f'Improvement: {(cw_cost - solver_cost) / cw_cost * 100:.2f}%')