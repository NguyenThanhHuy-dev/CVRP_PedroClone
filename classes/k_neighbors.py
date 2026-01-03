import numpy as np

from networkx import Graph, minimum_spanning_tree

from classes.instance import Instance
from classes.route import Route
from classes.utils import Utils

class KNeighbors:
    ''' Class for the k-nearest neighbors heuristic '''
    
    def __init__(self, cvrp: Instance, neighbor_number: int, routes: dict[int, Route]):  
        self.cvrp = cvrp # CVRP instance
        self.routes = routes # Routes list
        self.neighbor_number = neighbor_number # Number of neighbors
        
        self.mst: Graph = None # Minimum spanning tree
        self.matrices: dict[int, np.ndarray] = {} # Distance matrices
        
    def load_mst(self):
        ''' Load the minimum spanning tree '''  
        
        graph = Graph()
        for i in range(self.cvrp.dimension):
            for j in range(self.cvrp.dimension):
                graph.add_edge(i, j, weight=self.cvrp.distances[i, j])
    
        self.mst = minimum_spanning_tree(graph)
        
    def nearest_neighbors_mst(self, customer: int) -> list[int]:
        ''' Get the nearest neighbors from the minimum spanning tree '''
    
        neighbors = list(self.mst.neighbors(customer))
        weights = [self.mst.get_edge_data(customer, neighbor)['weight'] for neighbor in neighbors]
        
        sorted_neighbors = [neighbor for _, neighbor in sorted(zip(weights, neighbors))]
        
        return sorted_neighbors[:self.neighbor_number]
        
    def nearest_neighbors_mat(self, customer: int):
        ''' Get the nearest neighbors from the distance matrix '''
        
        neighbors = list(range(self.cvrp.dimension))
        weights = list(self.cvrp.distances[customer])
        
        sorted_neighbors = [neighbor for _, neighbor in sorted(zip(weights, neighbors)) if neighbor != customer]
        
        return sorted_neighbors[:self.neighbor_number]
        
    def nearest_neighbors(self, customer: int):
        ''' Get the nearest neighbors '''
        
        neighbors = self.nearest_neighbors_mst(customer)
        
        if len(neighbors) < self.neighbor_number:
            for neighbor in self.nearest_neighbors_mat(customer):
                if neighbor not in neighbors:
                    neighbors.append(neighbor)
                    
                if len(neighbors) == self.neighbor_number:
                    break
        
        if len(neighbors) < self.neighbor_number:
            raise Exception('Cannot find all neighbors')
        
        return neighbors
    
    def load_matrices(self):
        ''' Load distance matrices based on nearest neighbors '''
        
        for idx in self.routes:
            route = self.routes[idx]
            
            matrix = np.full((self.cvrp.dimension, self.cvrp.dimension), -1, dtype=int)
            
            for i in range(self.cvrp.dimension):
                matrix[i, i] = 0
            
            matrix[0, route[0]] = matrix[route[0], 0] = self.cvrp.distances[0, route[0]]  
            for i in range(len(route) - 1):
                distance = self.cvrp.distances[route[i], route[i + 1]]
                matrix[route[i], route[i + 1]] = matrix[route[i + 1], route[i]] = distance
            matrix[0, route[-1]] = matrix[route[-1], 0] = self.cvrp.distances[0, route[-1]]
                
            for customer in route:
                for neighbor in self.nearest_neighbors(customer):
                    distance = self.cvrp.distances[customer, neighbor]
                    matrix[customer, neighbor] = matrix[neighbor, customer] = distance
            
            self.matrices[idx] = matrix
    
    @Utils.timer
    @staticmethod
    def run(cvrp: Instance, neighbor_number: int, routes: dict[int, Route]) -> tuple[float, list[np.ndarray]]:
        ''' Run the k-nearest neighbors heuristic '''
        
        kn = KNeighbors(cvrp, neighbor_number, routes)
        
        kn.load_mst()
        kn.load_matrices()
        
        return list(kn.matrices.values())