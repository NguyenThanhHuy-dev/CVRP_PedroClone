from classes.route import Route
from classes.utils import Utils

class TwoOpt:
    ''' Class for the 2-opt heuristic '''
    
    def __init__(self, routes: dict[int, Route]):
        self.routes = routes # Routes list

    def improve_routes(self):
        ''' Improve the routes '''
        
        for idx in self.routes:
            best_route = self.routes[idx]
            
            route = best_route
            
            while True:
                improved = False
                
                for i in range(len(route) - 1):
                    for j in range(i + 1, len(route)):
                        new_route = route.reversed(i, j + 1)
                        if new_route.cost < best_route.cost:
                            best_route = new_route
                            improved = True        
                
                route = best_route
                
                if not improved:
                    break
            
            self.routes[idx] = route

    @Utils.timer
    @staticmethod
    def run(routes: dict[int, Route]) -> tuple[float, dict[int, Route]]:
        ''' Run the 2-opt heuristic '''
        
        to = TwoOpt(routes)
        
        to.improve_routes()
    
        return to.routes