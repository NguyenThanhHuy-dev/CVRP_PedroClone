import numpy as np

class Instance:
    ''' Class for the Capacitated Vehicle Routing Problem ''' 
    
    def __init__(self, instance_file: str):
        self._instance_file = instance_file # Instance file name
        
        self.dimension = 0 # Number of customers
        self.edge_weight_type = '' # Edge weight type
        self.edge_weight_format = '' # Edge weight format
        self.capacity = 0 # Each vehicle capacity
        
        self.node_coords: list[tuple[float, float]] = [] # Nodes coordinates
        self.demands: list[int] = [] # Demands list
        
        self.distances: np.ndarray = None # Distance matrix
        
        # Auxiliary variables
        
        self._section = '' # Section name
        self._i, self._j = 0, 0 # Matrix indexes
    
    def load_field(self, line: str):
        ''' Load a field from the line '''
        
        field, value = map(lambda x: x.strip(), line.split(':', 1))
        
        match field:
            case 'TYPE':
                if value != 'CVRP':
                    raise Exception('Only CVRP instances are supported')
                
            case 'DIMENSION':
                self.dimension = int(value)
                
                self.distances = np.zeros((self.dimension, self.dimension), dtype=int)
                
            case 'EDGE_WEIGHT_TYPE':
                if value not in ('EUC_2D', 'ATT', 'EXPLICIT'):
                    raise Exception('Only (EUC_2D, ATT, EXPLICIT) edge weight types are supported')
                
                self.edge_weight_type = value
                
            case 'EDGE_WEIGHT_FORMAT':
                if value not in ('LOWER_COL', 'LOWER_ROW'):
                    raise Exception('Only (LOWER_COL, LOWER_ROW) edge weight formats are supported')

                self.edge_weight_format = value
                
                match self.edge_weight_format:
                    case 'LOWER_ROW':
                        self._i, self._j = 0, 1
                    case 'LOWER_COL':
                        self._i, self._j = 1, 0
                
            case 'CAPACITY':
                self.capacity = int(value)
    
    def load_section(self, line: str):
        ''' Load a section from the line '''
            
        values = line.split()
                
        match self._section:            
            case 'EDGE_WEIGHT_SECTION':
                for value in values:
                    self.distances[self._i, self._j] = self.distances[self._j, self._i] = round(float(value))
                    
                    self._i += 1
                    
                    match self.edge_weight_format:
                        case 'LOWER_ROW':      
                           if self._i == self._j:
                                self._i = 0
                                self._j += 1
                            
                        case 'LOWER_COL':
                            if self._i == self.dimension:
                                self._j += 1 
                                self._i = self._j + 1
                
            case 'NODE_COORD_SECTION':
                self.node_coords.append((float(values[1]), float(values[2])))
                
            case 'DEMAND_SECTION':
                self.demands.append(int(values[1]))

            case 'DEPOT_SECTION':
                depot = int(values[0])
                
                if depot > 0 and depot != 1:
                    raise Exception('Depot must be the first node')
        
    def load_distances(self):
        ''' Load the distances for the CVRP instance '''
          
        for i in range(self.dimension):
            for j in range(i + 1, self.dimension):
                sdx = (self.node_coords[i][0] - self.node_coords[j][0]) ** 2
                sdy = (self.node_coords[i][1] - self.node_coords[j][1]) ** 2
                
                distance = np.sqrt(sdx + sdy)
                
                if self.edge_weight_type == 'ATT':
                    distance /= 10
                    
                self.distances[i, j] = self.distances[j, i] = round(distance)
    
    def load(self):
        ''' Load an instance from the file '''
        
        with open(self._instance_file, 'r') as file:
            for l in file.readlines():
                line = l.strip()
                
                if not line:
                    continue
                
                if ':' in line:
                    self.load_field(line)
                    
                if line.isupper():
                    self._section = line
                    
                else:    
                    self.load_section(line)
            
            if len(self.node_coords):
                self.load_distances()
            
        return self