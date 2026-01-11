import numpy as np

def calculate_route_cost(route, distances):
    if not route: return 0
    c= distances[0, route[0]]
    for i in range(len(route)-1):
        c+= distances[route[i], route[i+1]]
    c+= distances[route[-1], 0]
    return int(c)