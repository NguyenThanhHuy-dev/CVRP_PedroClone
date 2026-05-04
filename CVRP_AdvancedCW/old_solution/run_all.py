from os import listdir

from classes import Instance, ClarkeWright, TwoOpt, KNeighbors, Solver

if __name__ == '__main__':
    for file in sorted(listdir('instances')):
        if not file.endswith('.vrp'):
            continue
        
        print(f' {file} '.center(80, '-'))
        
        try:
            vehicle_number = int(file.split('.')[0].split('-')[-1][1:])
        
            cvrp = Instance(f'instances/{file}').load()
            cw_time, routes = ClarkeWright.run(cvrp, vehicle_number)
            to_time, routes = TwoOpt.run(routes)
            
            cw_2opt_time = cw_time + to_time
            cw_2opt_cost = sum(route.cost for route in routes.values())
            cw_2opt_routes = list(routes.values())
            
            print(f'CW + 2-Opt cost: {cw_2opt_cost} ({cw_2opt_time}s)')
            
            for neighbor_number in [3, 4, 5]:
                print(f' {neighbor_number} neighbors '.center(80, '-'))
                
                _, matrices = KNeighbors.run(cvrp, neighbor_number, routes)
                
                for _ in range(5):
                    solver_time, solver_cost, solver_routes = Solver.run(cvrp, matrices)
                    
                    print(f'Solver cost: {solver_cost} ({solver_time:.3f}s)')
                
            print(' % '.center(80, '-'))
            print(f'Improvement: {(cw_2opt_cost - solver_cost) / cw_2opt_cost * 100:.2f}%')           
        
        except Exception as e:
            print(f'{file}: {e}')
        