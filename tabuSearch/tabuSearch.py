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

def solve():
    data = read_input()
    if not data: return
    N, K, d, t = data

    start_time = time.time()
    max_time_limit = 1.8 # Giới hạn thời gian chạy

    # 1. KHỞI TẠO LỜI GIẢI BAN ĐẦU (Tham lam đơn giản)
    routes = [[0, 0] for _ in range(K)]
    costs = [0] * K
    
    # Sắp xếp khách hàng ngẫu nhiên một chút để init
    clients = list(range(1, N + 1))
    
    for u in clients:
        best_k, best_pos, best_nc = -1, -1, float('inf')
        for k in range(K):
            for j in range(1, len(routes[k])):
                p = routes[k][j-1]
                n = routes[k][j]
                nc = costs[k] + t[p][u] + d[u] + t[u][n] - t[p][n]
                if nc < best_nc:
                    best_nc, best_pos, best_k = nc, j, k
        routes[best_k].insert(best_pos, u)
        costs[best_k] = best_nc

    # 2. CẤU TRÚC TABU SEARCH
    global_best_max = max(costs)
    global_best_sum = sum(costs)
    global_best_routes = [r[:] for r in routes]
    
    # Ma trận Tabu: tabu_matrix[u][r] = vòng lặp mà u được phép quay lại tuyến r
    tabu_matrix = [[0] * K for _ in range(N + 1)]
    iteration = 0
    
    # Độ dài danh sách cấm
    tenure_min, tenure_max = 10, 20

    # VÒNG LẶP TABU SEARCH CHÍNH
    while time.time() - start_time < max_time_limit:
        iteration += 1
        
        best_move = None
        best_move_eval = float('inf')
        
        # Để tăng tốc cho N lớn, ưu tiên xét các tuyến đang có chi phí cao nhất
        sorted_routes = sorted(range(K), key=lambda x: costs[x], reverse=True)
        # Chỉ xét rút khách hàng từ 1/3 số tuyến bận rộn nhất (hoặc ít nhất 2 tuyến)
        routes_to_explore = sorted_routes[:max(2, K // 3)] if N > 100 else range(K)

        for r1 in routes_to_explore:
            if len(routes[r1]) <= 2: continue
            
            for i in range(1, len(routes[r1]) - 1):
                u = routes[r1][i]
                p1 = routes[r1][i-1]
                n1 = routes[r1][i+1]
                
                # Chi phí tuyến r1 sau khi rút u
                nc1 = costs[r1] - (t[p1][u] + d[u] + t[u][n1] - t[p1][n1])
                
                for r2 in range(K):
                    if r1 == r2: continue
                    
                    # Giới hạn số lượng vị trí chèn nếu số khách hàng trên tuyến r2 quá lớn
                    positions = range(1, len(routes[r2]))
                    if len(positions) > 50:
                        positions = random.sample(positions, 50)
                        
                    for j in positions:
                        p2 = routes[r2][j-1]
                        n2 = routes[r2][j]
                        
                        # Chi phí tuyến r2 sau khi chèn u
                        nc2 = costs[r2] + t[p2][u] + d[u] + t[u][n2] - t[p2][n2]
                        
                        # ĐÁNH GIÁ HÀM MỤC TIÊU KÉP (Tie-breaker)
                        mx = -1
                        sm = 0
                        for k in range(K):
                            c = nc1 if k == r1 else (nc2 if k == r2 else costs[k])
                            if c > mx: mx = c
                            sm += c
                            
                        move_eval = mx * 1000000000 + sm
                        
                        # KIỂM TRA TABU VÀ ASPIRATION CRITERION
                        is_tabu = iteration < tabu_matrix[u][r1] # Cấm u quay lại r1
                        is_aspiration = move_eval < (global_best_max * 1000000000 + global_best_sum)
                        
                        if not is_tabu or is_aspiration:
                            # Luôn lưu lại nước đi tốt nhất trong lân cận (kể cả khi tệ hơn hiện tại)
                            if move_eval < best_move_eval:
                                best_move_eval = move_eval
                                best_move = (r1, i, r2, j, u, nc1, nc2, mx, sm)
                                
        # Nếu kẹt hoàn toàn (mọi nước đi đều Tabu và không vi phạm Aspiration)
        if not best_move:
            break
            
        # 3. THỰC THI NƯỚC ĐI TỐT NHẤT LÂN CẬN
        r1, i, r2, j, u, nc1, nc2, move_max, move_sum = best_move
        
        # Cập nhật cấu trúc dữ liệu
        routes[r1].pop(i)
        routes[r2].insert(j, u)
        costs[r1] = nc1
        costs[r2] = nc2
        
        # CẬP NHẬT TABU LIST: Khách hàng u vừa chuyển từ r1 sang r2. 
        # Cấm u quay ngược lại r1 trong khoảng (10 -> 20) bước lặp tiếp theo.
        tabu_matrix[u][r1] = iteration + random.randint(tenure_min, tenure_max)
        
        # CẬP NHẬT KỶ LỤC TOÀN CỤC (GLOBAL BEST)
        if move_max < global_best_max or (move_max == global_best_max and move_sum < global_best_sum):
            global_best_max = move_max
            global_best_sum = move_sum
            global_best_routes = [r[:] for r in routes]

    # In ra output kết quả kỷ lục
    print(K)
    for r in global_best_routes:
        print(len(r))
        print(" ".join(map(str, r)))

if __name__ == '__main__':
    solve()