# This SA solution scores 782 in hustack.
import sys
import math
import random
sys.setrecursionlimit(10**7)


def input_():
    """
    Read input data.

    Returns:
        N (int): Number of customers.
        K (int): Number of routes/vehicles.
        d (list[int]): Service time of each customer.
        t (list[list[int]]): Travel time matrix.
    """
    N, K = map(int, input().split())
    d = [0] + list(map(int, input().split()))
    t = [list(map(int, input().split())) for _ in range(N + 1)]

    return N, K, d, t


def build_route(lst, t):
    """
    Construct a route using nearest neighbor heuristic.

    Args:
        lst (list[int]): Customers assigned to a route.
        t (list[list[int]]): Travel time matrix.

    Returns:
        list[int]: Route starting and ending at depot 0.
    """
    if not lst:
        return [0, 0]

    unused = set(lst)

    cur = 0
    route = [0]

    while unused:
        nxt = min(unused, key=lambda x: t[cur][x])
        route.append(nxt)
        unused.remove(nxt)
        cur = nxt

    route.append(0)

    return route


def route_cost(route, t, d):
    """
    Compute total cost of a single route.

    Args:
        route (list[int]): Route path.
        t (list[list[int]]): Travel time matrix.
        d (list[int]): Service times.

    Returns:
        int: Route cost.
    """
    total = 0

    for i in range(len(route) - 1):
        u = route[i]
        v = route[i + 1]

        total += t[u][v]

        if v != 0:
            total += d[v]

    return total


def max_cost(solution, t, d):
    """
    Compute objective value of a solution.

    Args:
        solution (list[list[int]]): List of routes.
        t (list[list[int]]): Travel time matrix.
        d (list[int]): Service times.

    Returns:
        int: Maximum route cost.
    """
    return max(route_cost(r, t, d) for r in solution)


def neighbor(solution, K):
    """
    Generate a neighboring solution by moving one customer between two routes.

    Args:
        solution (list[list[int]]): Current solution.
        K (int): Number of routes.

    Returns:
        list[list[int]]: Neighbor solution.
    """
    new_solution = [r[:] for r in solution]

    # Choose random non-empty route
    non_empty = [
        i for i in range(K)
        if len(new_solution[i]) > 2
    ]

    if not non_empty:
        return new_solution

    a = random.choice(non_empty)
    b = random.randint(0, K - 1)

    while b == a and K > 1:
        b = random.randint(0, K - 1)

    route_a = new_solution[a]
    route_b = new_solution[b]

    # Choose customer from route_a
    pos_a = random.randint(1, len(route_a) - 2)

    customer = route_a[pos_a]

    # Remove customer
    route_a.pop(pos_a)

    # Insert into route_b
    pos_b = random.randint(1, len(route_b) - 1)

    route_b.insert(pos_b, customer)

    return new_solution


def solve(N, K, d, t):
    """
    Solve the routing problem using Simulated Annealing.

    Args:
        N (int): Number of customers.
        K (int): Number of routes.
        d (list[int]): Service times.
        t (list[list[int]]): Travel time matrix.

    Returns:
        list[list[int]]: Best solution found.
    """

    # Sort customers by difficulty
    customers = list(range(1, N + 1))
    customers.sort(
        key=lambda x: d[x] + t[0][x] + t[x][0],
        reverse=True
    )

    # Initial solution: greedy load balancing
    routes = [[] for _ in range(K)]
    loads = [0] * K

    for c in customers:
        k = min(range(K), key=lambda x: loads[x])

        routes[k].append(c)

        loads[k] += d[c] + t[0][c] + t[c][0]

    # Route construction: nearest neighbor
    routes = [build_route(r, t) for r in routes]

    # Initial state
    current = routes
    current_cost = max_cost(current, t, d)

    best = [r[:] for r in current]
    best_cost = current_cost

    # SA parameters
    T = 10000.0
    T_min = 1e-3
    alpha = 0.995
    ITER = 200000

    # Simulated Annealing
    for _ in range(ITER):

        nxt = neighbor(current, K)

        nxt_cost = max_cost(nxt, t, d)

        delta = nxt_cost - current_cost

        # Minimization
        if delta < 0:
            current = nxt
            current_cost = nxt_cost

        else:
            prob = math.exp(-delta / T)

            if random.random() < prob:
                current = nxt
                current_cost = nxt_cost

        # Update best solution
        if current_cost < best_cost:
            best = [r[:] for r in current]
            best_cost = current_cost

        # Cooling schedule
        T *= alpha

        if T < T_min:
            break

    return best


def output_(K, solution):
    """
    Print solution in required format.

    Args:
        K (int): Number of routes.
        solution (list[list[int]]): Final routes.
    """
    print(K)
    for route in solution:
        print(len(route))
        print(*route)


def main():
    N, K, d, t = input_()
    solution = solve(N, K, d, t)
    output_(K, solution)


if __name__ == "__main__":
    main()
