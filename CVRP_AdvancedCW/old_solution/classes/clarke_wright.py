from classes.instance import Instance
from classes.route import Route
from classes.utils import Utils

class ClarkeWright:
    ''' Class for the Clarke-Wright savings heuristic '''
    
    def __init__(self, cvrp: Instance, vehicle_number: int):
        ''' Initialize class with the CVRP instance and the number of vehicles '''
        
        self.cvrp = cvrp # CVRP instance
        self.vehicle_number = vehicle_number # Number of vehicles
        
        self.savings: list[tuple[int, int, int]] = [] # Savings list
        
        self.routes: dict[int, Route] = {} # Routes dictionary
    
    def load_savings(self):
        ''' Load the savings for the CVRP instance '''
        
        for i in range(1, self.cvrp.dimension):
            for j in range(i + 1, self.cvrp.dimension):
                saving = self.cvrp.distances[0, i] + self.cvrp.distances[0, j] - self.cvrp.distances[i, j]
                
                self.savings.append((saving, i, j))
                
        self.savings.sort(key=lambda x: x[0], reverse=True)
    
    def load_routes(self):
        ''' Load initial routes '''
        
        for customer in range(1, self.cvrp.dimension):
            self.routes[customer] = Route(self.cvrp, [customer])

    def combine_routes(self):
        ''' Combine the routes '''
        
        for saving, i, j in self.savings:
            if i not in self.routes or j not in self.routes:
                continue
            
            if self.routes[i] == self.routes[j]:
                continue
            
            if self.routes[i][0] == i:
                self.routes[i] = self.routes[i].reversed()
            
            if self.routes[j][-1] == j:
                self.routes[j] = self.routes[j].reversed()
                
            if self.routes[i][-1] != i or self.routes[j][0] != j:
                continue
            
            if self.routes[i].demand + self.routes[j].demand > self.cvrp.capacity:
                continue
            
            self.routes[i] += self.routes[j]
            del self.routes[j]
        
    def reduce_routes(self):
        ''' Reduce the number of routes '''
        
        while len(self.routes) > self.vehicle_number:
            route_removed = False
            
            # Try to remove the route with the least number of customers
            for remotion in sorted(self.routes, key=lambda i: len(self.routes[i])):
                remotion_route = self.routes[remotion]
                
                del self.routes[remotion]
            
                remaining_customer = False
                
                # Try to add the customers of the removed route to the other routes
                for customer in remotion_route:
                    customer_added = False
                    
                    # Try to add the customer the least loaded route
                    for addition in sorted(self.routes, key=lambda i: self.routes[i].demand):
                        if self.routes[addition].demand + self.cvrp.demands[customer] > self.cvrp.capacity:
                            continue
                        
                        self.routes[addition].append(customer)
                        
                        customer_added = True
                        break
                    
                    if not customer_added:
                        remaining_customer = True
                        break
                
                # If could not add a customer to any route, restore the removed route
                if remaining_customer:
                    for customer in remotion_route:
                        for route in self.routes:
                            if customer in self.routes[route]:
                                self.routes[route].remove(customer)
                    
                    self.routes[remotion] = remotion_route
                else:
                    route_removed = True
                    break
            
            # If could not remove any route, stop the process
            if not route_removed:
                raise Exception('Cannot reduce the number of routes')
                
    @Utils.timer
    @staticmethod
    def run(cvrp: Instance, vehicle_number: int) -> tuple[float, dict[int, Route]]:
        ''' Run the Clarke-Wright savings heuristic '''
        
        cw = ClarkeWright(cvrp, vehicle_number)
        
        cw.load_savings()
        cw.load_routes()
        
        cw.combine_routes()
        cw.reduce_routes()
        
        return cw.routes