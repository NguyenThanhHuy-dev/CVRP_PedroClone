from classes.instance import Instance
from classes.route import Route
from classes.utils import Utils

import math

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
        ''' Reduce the number of routes using multiple strategies '''
        
        while len(self.routes) > self.vehicle_number:
            reduced = False
            
            # Strategy 1: Find two routes whose combined demand fits capacity
            reduced = self._try_direct_merge()
            if reduced:
                continue
            
            # Strategy 2: Relocate customers between routes to enable merging
            reduced = self._try_relocate_and_merge()
            if reduced:
                continue
            
            # Strategy 3: Dissolve smallest route by distributing its customers
            reduced = self._try_dissolve_route()
            if reduced:
                continue
            
            # Strategy 4: Repack all customers into K bins using Best Fit Decreasing
            if self._try_repack():
                return  # _try_repack produces exactly K routes
            
            raise Exception('Cannot reduce the number of routes')
    
    def _try_direct_merge(self) -> bool:
        ''' Try to merge two routes whose combined demand <= capacity '''
        
        route_keys = list(self.routes.keys())
        best_merge = None
        best_merge_cost = float('inf')
        
        for i in range(len(route_keys)):
            for j in range(i + 1, len(route_keys)):
                r1, r2 = route_keys[i], route_keys[j]
                if self.routes[r1].demand + self.routes[r2].demand > self.cvrp.capacity:
                    continue
                
                # Pick the pair with smallest linking distance
                cost = min(
                    self.cvrp.distances[self.routes[r1].value[-1], self.routes[r2].value[0]],
                    self.cvrp.distances[self.routes[r2].value[-1], self.routes[r1].value[0]],
                )
                if cost < best_merge_cost:
                    best_merge_cost = cost
                    best_merge = (r1, r2)
        
        if best_merge:
            r1, r2 = best_merge
            # Choose better concatenation direction
            cost_r1_r2 = self.cvrp.distances[self.routes[r1][-1], self.routes[r2][0]]
            cost_r2_r1 = self.cvrp.distances[self.routes[r2][-1], self.routes[r1][0]]
            
            if cost_r1_r2 <= cost_r2_r1:
                self.routes[r1] = self.routes[r1] + self.routes[r2]
            else:
                self.routes[r1] = self.routes[r2] + self.routes[r1]
            del self.routes[r2]
            return True
        
        return False
    
    def _try_relocate_and_merge(self) -> bool:
        route_keys = list(self.routes.keys())
        pairs_by_excess = []
        for i in range(len(route_keys)):
            for j in range(i + 1, len(route_keys)):
                r1, r2 = route_keys[i], route_keys[j]
                excess = self.routes[r1].demand + self.routes[r2].demand - self.cvrp.capacity
                if excess > 0:
                    pairs_by_excess.append((excess, r1, r2))
        
        pairs_by_excess.sort()
        
        for excess, r1, r2 in pairs_by_excess:
            candidates = []
            # FIX: Thêm .value để duyệt qua list khách hàng
            for c in self.routes[r1].value:
                candidates.append((self.cvrp.demands[c], c, r1))
            for c in self.routes[r2].value:
                candidates.append((self.cvrp.demands[c], c, r2))
            candidates.sort(reverse=True)
            
            moved = []
            remaining_excess = excess
            
            for demand_c, customer, source in candidates:
                if remaining_excess <= 0:
                    break
                
                best_target = None
                best_insert_cost = float('inf')
                
                for t in route_keys:
                    if t == r1 or t == r2:
                        continue
                    if self.routes[t].demand + demand_c > self.cvrp.capacity:
                        continue
                    
                    route_val = self.routes[t].value
                    for pos in range(len(route_val) + 1):
                        prev_node = route_val[pos - 1] if pos > 0 else 0
                        next_node = route_val[pos] if pos < len(route_val) else 0
                        cost = (self.cvrp.distances[prev_node, customer] +
                                self.cvrp.distances[customer, next_node] -
                                self.cvrp.distances[prev_node, next_node])
                        if cost < best_insert_cost:
                            best_insert_cost = cost
                            best_target = (t, pos)
                
                if best_target:
                    t, pos = best_target
                    
                    # FIX: Xóa khỏi .value và tạo Route object mới để an toàn tính demand/cost
                    src_val = list(self.routes[source].value)
                    src_val.remove(customer)
                    self.routes[source] = Route(self.cvrp, src_val)
                    
                    tgt_val = list(self.routes[t].value)
                    tgt_val.insert(pos, customer)
                    self.routes[t] = Route(self.cvrp, tgt_val)
                    
                    moved.append((customer, source, t, pos))
                    remaining_excess -= demand_c
            
            if remaining_excess <= 0:
                cost_r1_r2 = self.cvrp.distances[self.routes[r1].value[-1], self.routes[r2].value[0]]
                cost_r2_r1 = self.cvrp.distances[self.routes[r2].value[-1], self.routes[r1].value[0]]
                
                if cost_r1_r2 <= cost_r2_r1:
                    self.routes[r1] = Route(self.cvrp, self.routes[r1].value + self.routes[r2].value)
                else:
                    self.routes[r1] = Route(self.cvrp, self.routes[r2].value + self.routes[r1].value)
                del self.routes[r2]
                return True
            else:
                # Undo all moves (Phục hồi lại nếu không đủ chỗ)
                for customer, source, target, pos in reversed(moved):
                    tgt_val = list(self.routes[target].value)
                    tgt_val.remove(customer)
                    self.routes[target] = Route(self.cvrp, tgt_val)
                    
                    src_val = list(self.routes[source].value)
                    src_val.append(customer)
                    self.routes[source] = Route(self.cvrp, src_val)
        
        return False
    
    def _try_dissolve_route(self) -> bool:
        # FIX: Dùng len(self.routes[i].value) thay vì len(self.routes[i])
        for remotion in sorted(self.routes, key=lambda i: len(self.routes[i].value)):
            remotion_route = self.routes[remotion]
            del self.routes[remotion]
            
            customers_to_place = sorted(remotion_route.value, 
                                        key=lambda c: self.cvrp.demands[c], reverse=True)
            placed = []
            success = True
            
            for customer in customers_to_place:
                best_target = None
                best_insert_cost = float('inf')
                
                for route_id in self.routes:
                    if self.routes[route_id].demand + self.cvrp.demands[customer] > self.cvrp.capacity:
                        continue
                    
                    route_val = self.routes[route_id].value
                    for pos in range(len(route_val) + 1):
                        prev_node = route_val[pos - 1] if pos > 0 else 0
                        next_node = route_val[pos] if pos < len(route_val) else 0
                        cost = (self.cvrp.distances[prev_node, customer] +
                                self.cvrp.distances[customer, next_node] -
                                self.cvrp.distances[prev_node, next_node])
                        if cost < best_insert_cost:
                            best_insert_cost = cost
                            best_target = (route_id, pos)
                
                if best_target:
                    route_id, pos = best_target
                    # FIX: Tạo Route object mới
                    val = list(self.routes[route_id].value)
                    val.insert(pos, customer)
                    self.routes[route_id] = Route(self.cvrp, val)
                    placed.append((customer, route_id))
                else:
                    success = False
                    break
            
            if success:
                return True
            
            # Undo
            for customer, route_id in reversed(placed):
                val = list(self.routes[route_id].value)
                val.remove(customer)
                self.routes[route_id] = Route(self.cvrp, val)
                
            self.routes[remotion] = remotion_route
        
        return False
    
    def _try_repack(self) -> bool:
        ''' Repack all customers into K routes using bin packing heuristics. '''
        
        all_customers = []
        for route in self.routes.values():
            all_customers.extend(route.value)
        
        # Sort by demand descending
        all_customers.sort(key=lambda c: self.cvrp.demands[c], reverse=True)
        
        # Try greedy strategies first (fast)
        for strategy in ('balanced', 'ffd', 'bfd'):
            result = self._bin_pack(all_customers, strategy)
            if result is not None:
                self._apply_packing(result)
                return True
        
        # Greedy failed → backtracking search with pruning
        result = self._backtrack_bin_pack(all_customers)
        if result is not None:
            self._apply_packing(result)
            return True
        
        return False
    
    def _apply_packing(self, bins: list[list[int]]):
        ''' Apply a packing result to self.routes '''
        self.routes.clear()
        for i, customers in enumerate(bins):
            if customers:
                self.routes[customers[0]] = Route(self.cvrp, customers)
    
    def _bin_pack(self, customers: list[int], strategy: str) -> list[list[int]] | None:
        ''' Pack customers into K bins using the given strategy.
            Returns list of bins or None if infeasible. '''
        
        K = self.vehicle_number
        bins: list[list[int]] = [[] for _ in range(K)]
        bin_demands = [0] * K
        target = sum(self.cvrp.demands[c] for c in customers) / K
        
        for customer in customers:
            d = self.cvrp.demands[customer]
            chosen = -1
            
            if strategy == 'balanced':
                # Place in bin that will be closest to (but not exceed) target load
                best_diff = float('inf')
                for b in range(K):
                    if bin_demands[b] + d > self.cvrp.capacity:
                        continue
                    diff = abs(bin_demands[b] + d - target)
                    if diff < best_diff:
                        best_diff = diff
                        chosen = b
            elif strategy == 'ffd':
                # First Fit: place in first bin with room
                for b in range(K):
                    if bin_demands[b] + d <= self.cvrp.capacity:
                        chosen = b
                        break
            else:  # bfd
                # Best Fit: place in fullest bin with room
                best_remaining = float('inf')
                for b in range(K):
                    remaining = self.cvrp.capacity - bin_demands[b]
                    if remaining >= d and remaining < best_remaining:
                        best_remaining = remaining
                        chosen = b
            
            if chosen < 0:
                return None
            
            bins[chosen].append(customer)
            bin_demands[chosen] += d
        
        return bins
    
    def _backtrack_bin_pack(self, customers: list[int]) -> list[list[int]] | None:
        ''' BFD with recursive swap repair for tight instances. '''
        
        K = self.vehicle_number
        cap = self.cvrp.capacity
        demands = self.cvrp.demands
        
        bins: list[list[int]] = [[] for _ in range(K)]
        bin_loads = [0] * K
        unplaced = []
        
        # Phase 1: BFD placement
        for c in customers:
            d = demands[c]
            best_b = -1
            best_rem = float('inf')
            for b in range(K):
                rem = cap - bin_loads[b]
                if rem >= d and rem < best_rem:
                    best_rem = rem
                    best_b = b
            if best_b >= 0:
                bins[best_b].append(c)
                bin_loads[best_b] += d
            else:
                unplaced.append(c)
        
        if not unplaced:
            return bins
        
        # Phase 2: Recursive swap repair
        def try_place(customer: int, depth: int, forbidden: set) -> bool:
            ''' Try to place customer, potentially swapping out others (up to depth). '''
            d_c = demands[customer]
            
            # Direct placement
            for b in range(K):
                if bin_loads[b] + d_c <= cap:
                    bins[b].append(customer)
                    bin_loads[b] += d_c
                    return True
            
            if depth <= 0:
                return False
            
            # Swap: remove an item x from a bin, place customer there, then place x
            for b in range(K):
                for i, x in enumerate(bins[b]):
                    if x in forbidden:
                        continue
                    d_x = demands[x]
                    # Would removing x and adding customer fit?
                    if bin_loads[b] - d_x + d_c <= cap:
                        # Try the swap
                        bins[b].remove(x)
                        bin_loads[b] -= d_x
                        bins[b].append(customer)
                        bin_loads[b] += d_c
                        
                        if try_place(x, depth - 1, forbidden | {customer}):
                            return True
                        
                        # Undo
                        bins[b].remove(customer)
                        bin_loads[b] -= d_c
                        bins[b].insert(i, x)
                        bin_loads[b] += d_x
            
            return False
        
        for c in list(unplaced):
            if try_place(c, 3, {c}):
                unplaced.remove(c)
            else:
                return None
        
        return bins if not unplaced else None
                
    @Utils.timer
    @staticmethod
    def run(cvrp: Instance, vehicle_number: int) -> tuple[float, dict[int, Route]]:
        ''' Run the Clarke-Wright savings heuristic '''
        
        cw = ClarkeWright(cvrp, vehicle_number)
        
        cw.load_savings()
        cw.load_routes()
        
        cw.combine_routes()
        
        try:
            cw.reduce_routes()
        except Exception:
            # Fallback: sweep construction, then reduce the sweep result
            cw._sweep_construction()
            cw.reduce_routes()
        
        return cw.routes
    
    def _sweep_construction(self):
        ''' Construct routes using sweep algorithm (fallback for tight instances).
            Sorts customers by polar angle from depot, sweeps and fills routes.
            May produce more than K routes - reduce_routes should be called after. '''
        
        try:
            depot_x, depot_y = self.cvrp.coordinates[0]
            get_coord = lambda c: self.cvrp.coordinates[c]
        except AttributeError:
            depot_x, depot_y = self.cvrp.node_coords[0]
            get_coord = lambda c: self.cvrp.node_coords[c]
        
        # Compute polar angle for each customer and sort
        angles = []
        for c in range(1, self.cvrp.dimension):
            cx, cy = get_coord(c)
            angle = math.atan2(cy - depot_y, cx - depot_x)
            angles.append((angle, c))
        angles.sort(key=lambda x: x[0])
        
        # Try each starting position, pick the one with fewest routes
        best_routes = None
        best_count = float('inf')
        n = len(angles)
        
        for start_idx in range(n):
            routes: dict[int, Route] = {}
            current_route = []
            current_demand = 0
            
            for offset in range(n):
                _, customer = angles[(start_idx + offset) % n]
                d = self.cvrp.demands[customer]
                
                if current_demand + d > self.cvrp.capacity and current_route:
                    routes[current_route[0]] = Route(self.cvrp, current_route)
                    current_route = []
                    current_demand = 0
                
                current_route.append(customer)
                current_demand += d
            
            if current_route:
                routes[current_route[0]] = Route(self.cvrp, current_route)
            
            if len(routes) < best_count:
                best_count = len(routes)
                best_routes = routes
        
        self.routes = best_routes