# This SA solution scores 780 in hustack.
import sys
import math
import random
from collections import defaultdict
sys.setrecursionlimit(10**7)


# input
N, K = map(int, input().split())
d = [0] + list(map(int, input().split()))
t = [list(map(int, input().split())) for _ in range(N + 1)]


# sort customers by difficulty
customers = list(range(1, N + 1))
customers.sort(
    key=lambda x: d[x] + t[0][x] + t[x][0],
    reverse=True
)


# initial solution: greedy load balancing
routes = [[] for _ in range(K)]
loads = [0] * K

for c in customers:
    k = min(range(K), key=lambda x: loads[x])
    routes[k].append(c)
    loads[k] += d[c] + t[0][c] + t[c][0]


# route construction: nearest neighbor
def build_route(lst):
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


routes = [build_route(r) for r in routes]


# cost
def route_cost(route):
    total = 0

    for i in range(len(route) - 1):
        u = route[i]
        v = route[i + 1]

        total += t[u][v]

        if v != 0:
            total += d[v]

    return total


def max_cost(solution):
    return max(route_cost(r) for r in solution)


# neighbor
def neighbor(solution):
    new_solution = [r[:] for r in solution]

    # choose random non-empty route
    non_empty = [i for i in range(K) if len(new_solution[i]) > 2]

    if not non_empty:
        return new_solution

    a = random.choice(non_empty)

    b = random.randint(0, K - 1)

    while b == a and K > 1:
        b = random.randint(0, K - 1)

    route_a = new_solution[a]
    route_b = new_solution[b]

    # choose customer from route_a
    pos_a = random.randint(1, len(route_a) - 2)

    customer = route_a[pos_a]

    # remove
    route_a.pop(pos_a)

    # insert into route_b
    pos_b = random.randint(1, len(route_b) - 1)

    route_b.insert(pos_b, customer)

    return new_solution


# SIMULATED ANNEALING
current = routes
current_cost = max_cost(current)

best = [r[:] for r in current]
best_cost = current_cost

T = 10000.0
T_min = 1e-3
alpha = 0.995
ITER = 200000

for it in range(ITER):
    nxt = neighbor(current)
    nxt_cost = max_cost(nxt)
    delta = nxt_cost - current_cost

    # minimization
    if delta < 0:
        current = nxt
        current_cost = nxt_cost
    else:
        prob = math.exp(-delta / T)

        if random.random() < prob:
            current = nxt
            current_cost = nxt_cost

    if current_cost < best_cost:
        best = [r[:] for r in current]
        best_cost = current_cost

    T *= alpha

    if T < T_min:
        break


# output
print(K)
for route in best:
    print(len(route))
    print(*route)
