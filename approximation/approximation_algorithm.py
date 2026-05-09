from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2

"""
    N: số khách hàng
    K: số nhân viên
    d: thời gian bảo trì, d[0] = 0
    t: ma trận thời gian di chuyển
"""

def solver_network_maintenance(N, K, d, t):
    manager = pywrapcp.RoutingIndexManager(
        N+1, #N khach + depot 0
        K, #so route = so nhan vien
        0 #starts: depot
    )

    routing = pywrapcp.RoutingModel(manager)

    #cost callback
    def transit_callback(from_index, to_index):
        from_note = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)

        travel = t[from_note][to_node]

        #chỉ note tới (to_note hay arrived node) là khách hàng thì mới có thời gian bảo trì (node di chuyển không có)
        service = 0 if to_node == 0 else d[to_node]

        return travel + service

    transit_idx = routing.RegisterTransitCallback(transit_callback)

    #cost cho toan route (tat ca cac)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_idx)

    #dimension để minimize max route (https://developers.google.com/optimization/routing/dimensions)
    routing.AddDimension(
        evaluator_index=transit_idx, #evaluator_index: id cua ham cost
        slack_max=0, #slack: không có thời gian đợi tại mỗi điểm (slack time i: độ trễ): đặt về 0
        capacity=10**9, #capacity: bài toán không liên quan đến carry (CVRP): đặt về vô cùng để không bị ràng buộc
        fix_start_cumul_to_zero=True, #fix_start_cumulative_to_zero: Boolean value. Nếu đúng, giá trị tích lũy của đại lượng sẽ bắt đầu từ 0 (trong bài này là thời gian sẽ tính từ 0s và tích lũy qua các node)
        name="Time" # dimension_name: string name cho dimension
    )

    time_dim = routing.GetDimensionOrDie("Time")

    #minimize max workload
    time_dim.SetGlobalSpanCostCoefficient(100)
    # span = max(route_end) - min(route_start)  : vì start = 0 => chính là minimize max workload

    #params
    search_params = pywrapcp.DefaultRoutingSearchParameters()

    #nghiệm ban đầu heuristic (greedy)
    search_params.first_solution_strategy = (routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    #cải thiện nghiệm metaheuristic (local search)
    search_params.local_search_metaheuristic = (routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)

    search_params.time_limit.seconds = 10

    #solve
    solution = routing.SolveWithParameters(search_params)
    if not solution:
        print("No solution found")
        return

    #output đúng định dạng
    print(K)

    for vehicle_id in range(K):
        index = routing.Start(vehicle_id)
        route = []

        while not routing.IsEnd(index): #khi route cua nguoi vehicle_id chua ket thuc (quay ve diem 0 depot)
            route.append(manager.IndexToNode(index))
            index = solution.Value(routing.NextVar(index))

        route.append(0)

        print(len(route))
        print(*route)

def read_input():
    # dòng 1: N K
    N, K = map(int, input().split())

    # dòng 2: d(1)...d(N)
    service_times = list(map(int, input().split()))

    # thêm d[0] = 0 cho depot
    d = [0] + service_times

    # đọc ma trận thời gian (N+1 dòng)
    t = []
    for _ in range(N + 1):
        row = list(map(int, input().split()))
        t.append(row)

    return N, K, d, t


if __name__ == "__main__":
    N, K, d, t = read_input()
    solver_network_maintenance(N, K, d, t)
