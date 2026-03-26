from typing import Union   

from classes.instance import Instance

class Route:
    ''' Class for the route '''
    
    def __init__(self, cvrp: Instance, value: list[int], demand: int = -1):
        self.cvrp = cvrp # CVRP instance
        self.value = value # Route list
        
        self._demand: int = demand # Route demand
        self._cost: float = -1 # Route cost
        
    def __repr__(self):
        ''' Return the string representation of the route '''
        
        return f'Route{self.value}'
        
    def __iter__(self):
        ''' Iterate over the route '''
        
        return iter(self.value)
        
    def __eq__(self, other: Union[list[int], 'Route']):
        ''' Check if two routes are equal '''
        
        if isinstance(other, Route):
            other = other.value
        
        return self.value == other
        
    def __contains__(self, customer: int):
        ''' Check if a customer is in the route '''
        
        return customer in self.value
    
    def __getitem__(self, idx: int | slice):
        ''' Get the customer at the index '''
        
        return self.value[idx]
    
    def __len__(self):
        ''' Get the length of the route '''
        
        return len(self.value)
    
    def append(self, customer: int):
        ''' Append a customer to the route '''
        
        self.value.append(customer)
        
        if self._demand >= 0:
            self._demand += self.cvrp.demands[customer]
            
        self._cost = -1 # Reset the cost
    
    def remove(self, customer: int):
        ''' Remove a customer from the route '''
        
        self.value.remove(customer)
        
        if self._demand >= 0:
            self._demand -= self.cvrp.demands[customer]
            
        self._cost = -1 # Reset the cost
    
    def __add__(self, other: Union[list[int], 'Route']):
        ''' Add a customer to the route '''
        
        demand = -1
        
        if isinstance(other, Route):
            demand = other.demand   
            other = other.value
            
        if demand >= 0 and self.demand >= 0:
            demand += self.demand
        
        return Route(self.cvrp, self.value + other, demand)
        
    def __radd__(self, other: Union[list[int], 'Route']):
        ''' Add a customer to the route '''
        
        demand = -1
        
        if isinstance(other, Route):
            demand = other.demand
            other = other.value
        
        if demand >= 0 and self.demand >= 0:
            demand += self.demand
        
        return Route(self.cvrp, other + self.value, demand)
    
    def reversed(self, i = None, j = None):
        ''' Returns a reversed route on the given indexes '''
        
        route = self.value[:]
        
        route[i:j] = route[i:j][::-1]
        
        return Route(self.cvrp, route, self.demand)
    
    @property
    def demand(self):
        ''' Get the route demand '''
        
        if self._demand < 0:
            self._demand = self.calculate_demand()
        
        return self._demand
    
    @property
    def cost(self):
        ''' Get the route cost '''
        
        if self._cost < 0:
            self._cost = self.calculate_cost()
        
        return self._cost
        
    def calculate_cost(self):
        ''' Calculate the cost for the route '''
        
        cost = self.cvrp.distances[0, self.value[0]]
        for i in range(len(self.value) - 1):
            cost += self.cvrp.distances[self.value[i], self.value[i + 1]]
        cost += self.cvrp.distances[0, self.value[-1]]
        
        return cost
    
    def calculate_demand(self):
        ''' Calculate the demand for the route '''
        
        return sum(self.cvrp.demands[c] for c in self.value)