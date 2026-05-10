import sys
import time
import random

def read_input():
    input_data = sys.stdin.read().split()
    if not input_data:
        return None
    
    N = int(input_data[0])
    K = int(input_data[1])
    
    d = [0] * (N + 1)
    idx = 2
    for i in range(1, N + 1):
        d[i] = int(input_data[idx])
        idx += 1
        
    t = []
    for i in range(N + 1):
        row = []
        for j in range(N + 1):
            row.append(int(input_data[idx]))
            idx += 1
        t.append(row)
        
    return N, K, d, t

def calc_route_time(route, t, d):
    s = 0
    for i in range(len(route) - 1):
        s += t[route[i]][route[i+1]] + d[route[i]]
    return s

def get_fitness_2(costs, r1, nc1, r2, nc2):
    """Tính fitness O(1) khi update 2 tuyến đường."""
    mx = -1
    sm = 0
    for k in range(len(costs)):
        if k == r1: c = nc1
        elif k == r2: c = nc2
        else: c = costs[k]
        
        if c > mx: mx = c
        sm += c
    return mx * 1000000000 + sm

def local_search(routes, costs, t, d, start_time, max_time):
    N_routes = len(routes)
    N_total = sum(len(r)-2 for r in routes)
    
    improved = True
    while improved and time.time() - start_time < max_time:
        improved = False
        best_fitness = max(costs) * 1000000000 + sum(costs)
        best_move = None
        
        max_cost = max(costs)
        bottleneck_routes = [k for k in range(N_routes) if costs[k] == max_cost]
        
        # Nếu số lượng khách hàng lớn, chỉ tìm kiếm xung quanh các tuyến bận rộn nhất để tiết kiệm thời gian
        routes_to_explore = range(N_routes) if N_total <= 150 else bottleneck_routes

        for r1 in routes_to_explore:
            route1 = routes[r1]
            if len(route1) <= 2: continue
            
            for i in range(1, len(route1) - 1):
                u = route1[i]
                p1 = route1[i-1]
                n1 = route1[i+1]
                
                rem1 = t[p1][u] + d[u] + t[u][n1]
                delta1 = t[p1][n1] - rem1
                
                # 1. Đảo vị trí trong cùng tuyến 
                if r1 in bottleneck_routes:
                    for j in range(1, len(route1)):
                        if j == i or j == i + 1: continue
                        temp = route1[:]
                        node = temp.pop(i)
                        insert_pos = j if j < i else j - 1
                        temp.insert(insert_pos, node)
                        nc1 = calc_route_time(temp, t, d)
                        if nc1 < costs[r1]:
                            mx = -1
                            sm = 0
                            for k in range(N_routes):
                                c = nc1 if k == r1 else costs[k]
                                if c > mx: mx = c
                                sm += c
                            fit = mx * 1000000000 + sm
                            if fit < best_fitness:
                                best_fitness = fit
                                best_move = ('intra', r1, i, r1, insert_pos, nc1, nc1)

                # Xét chuyển đổi với các tuyến khác
                for r2 in range(N_routes):
                    if r1 == r2: continue
                    route2 = routes[r2]
                    
                    # 2. Rút khách hàng chuyển sang tuyến khác 
                    for j in range(1, len(route2)):
                        p2 = route2[j-1]
                        n2 = route2[j]
                        delta2 = t[p2][u] + d[u] + t[u][n2] - t[p2][n2]
                        
                        nc1 = costs[r1] + delta1
                        nc2 = costs[r2] + delta2
                        
                        fit = get_fitness_2(costs, r1, nc1, r2, nc2)
                        if fit < best_fitness:
                            best_fitness = fit
                            best_move = ('relocate', r1, i, r2, j, nc1, nc2)
                            
                    # 3. Hoán đổi 2 khách hàng của 2 tuyến (Inter-swap)
                    for j in range(1, len(route2) - 1):
                        v = route2[j]
                        p2 = route2[j-1]
                        n2 = route2[j+1]
                        
                        rem2_swap = t[p2][v] + d[v] + t[v][n2]
                        add1_swap = t[p1][v] + d[v] + t[v][n1]
                        add2_swap = t[p2][u] + d[u] + t[u][n2]
                        
                        nc1 = costs[r1] - rem1 + add1_swap
                        nc2 = costs[r2] - rem2_swap + add2_swap
                        
                        fit = get_fitness_2(costs, r1, nc1, r2, nc2)
                        if fit < best_fitness:
                            best_fitness = fit
                            best_move = ('swap', r1, i, r2, j, nc1, nc2)
                            
        # Thực thi bước di chuyển tốt nhất
        if best_move:
            m_type, r1, i, r2, j, cost_r1, cost_r2 = best_move
            if m_type == 'intra':
                node = routes[r1].pop(i)
                routes[r1].insert(j, node)
                costs[r1] = cost_r1
            elif m_type == 'relocate':
                node = routes[r1].pop(i)
                routes[r2].insert(j, node)
                costs[r1] = cost_r1
                costs[r2] = cost_r2
            elif m_type == 'swap':
                routes[r1][i], routes[r2][j] = routes[r2][j], routes[r1][i]
                costs[r1] = cost_r1
                costs[r2] = cost_r2
            improved = True
            
    return routes, costs

def solve():
    data = read_input()
    if not data: return
    N, K, d, t = data

    start_time = time.time()
    max_time = 1.8 # Đảm bảo không quá 2s Time Limit Của Hệ Thống

    global_best_routes = None
    global_best_costs = None
    global_best_fitness = float('inf')
    
    first_run = True
    
    # Meta-heuristic: Random Restart GRASP
    while first_run or time.time() - start_time < max_time:
        first_run = False
        unassigned = list(range(1, N + 1))
        random.shuffle(unassigned) # Trộn ngẫu nhiên khách hàng
        
        routes = [[0, 0] for _ in range(K)]
        costs = [0] * K
        
        # Tạo lời giải tham lam
        for u in unassigned:
            best_fit = float('inf')
            best_pos = -1
            best_k = -1
            best_nc = -1
            
            for k in range(K):
                for j in range(1, len(routes[k])):
                    p = routes[k][j-1]
                    n = routes[k][j]
                    delta = t[p][u] + d[u] + t[u][n] - t[p][n]
                    nc = costs[k] + delta
                    
                    # Tính Tie-breaker Fitness
                    mx = -1
                    sm = 0
                    for k_idx in range(K):
                        c = nc if k_idx == k else costs[k_idx]
                        if c > mx: mx = c
                        sm += c
                    fit = mx * 1000000000 + sm
                    
                    if fit < best_fit:
                        best_fit = fit
                        best_pos = j
                        best_k = k
                        best_nc = nc
                        
            routes[best_k].insert(best_pos, u)
            costs[best_k] = best_nc
            
        routes, costs = local_search(routes, costs, t, d, start_time, max_time)
        
        # Ghi nhận kết quả tốt nhất
        fit = max(costs) * 1000000000 + sum(costs)
        if fit < global_best_fitness:
            global_best_fitness = fit
            global_best_routes = [r[:] for r in routes]
            global_best_costs = costs[:]

    # In kết quả chuẩn Output
    print(K)
    for r in global_best_routes:
        print(len(r))
        print(" ".join(map(str, r)))

if __name__ == '__main__':
    solve()